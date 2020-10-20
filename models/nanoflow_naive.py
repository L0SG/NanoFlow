from torch import nn
from modules import Wavenet2D, Conv2D, ZeroConv2d
from torch.distributions.normal import Normal
from functions import *
#################################################################################
# experimental mdoel 1: shared parameters for each entire flow

class WaveFlowCoupling2D(nn.Module):
    def __init__(self, in_channel, cin_channel, filter_size=256, num_layer=6, num_height=None,
                 layers_per_dilation_h_cycle=3):
        super().__init__()
        assert num_height is not None
        self.in_channel = in_channel
        self.num_height = num_height
        self.layers_per_dilation_h_cycle = layers_per_dilation_h_cycle
        # dilation for width & height generation loop
        self.dilation_h = []
        self.dilation_w = []
        self.kernel_size = 3
        for i in range(num_layer):
            self.dilation_h.append(2 ** (i % self.layers_per_dilation_h_cycle))
            self.dilation_w.append(2 ** i)

        self.num_layer = num_layer
        self.filter_size = filter_size
        self.net = Wavenet2D(in_channels=in_channel, out_channels=filter_size,
                             num_layers=num_layer, residual_channels=filter_size,
                             gate_channels=filter_size, skip_channels=filter_size,
                             kernel_size=3, cin_channels=cin_channel, dilation_h=self.dilation_h,
                             dilation_w=self.dilation_w)
        # projector for log_s and t
        self.proj_log_s_t = ZeroConv2d(filter_size, 2*in_channel)

    def forward(self, x, c=None, i=None, debug=False):
        x_shift = shift_1d(x)

        feat = self.net(x_shift, c)

        log_s_t = self.proj_log_s_t(feat)
        log_s = log_s_t[:, :self.in_channel]
        t = log_s_t[:, self.in_channel:]

        out = x * torch.exp(log_s) + t
        logdet = torch.sum(log_s)

        if debug:
            return out, logdet, log_s, t
        else:
            return out, logdet, None, None

    def reverse(self, z, c=None):
        x = torch.zeros_like(z[:, :, 0:1, :])
        # pre-compute conditioning tensors and cache them
        c_cache = []
        for i, resblock in enumerate(self.net.res_blocks):
            filter_gate_conv_c = resblock.filter_gate_conv_c(c)
            c_cache.append(filter_gate_conv_c)
        c_cache = torch.stack(c_cache)  # [num_layers, batch_size, res_channels, width, height]

        for i_h in range(self.num_height):
            feat = self.net.reverse(x, c_cache[:, :, :, :i_h + 1, :])[:, :, -1, :].unsqueeze(2)
            log_s_t = self.proj_log_s_t(feat)
            log_s = log_s_t[:, :self.in_channel]
            t = log_s_t[:, self.in_channel:]

            x_new = (z[:, :, i_h, :].unsqueeze(2) - t) * torch.exp(-log_s)
            x = torch.cat((x, x_new), 2)

        return x[:, :, 1:, :], c

    def reverse_fast(self, z, c_cache=None):
        x = torch.zeros_like(z[:, :, 0:1, :])
        self.net.conv_queue_init(x)

        # already cached
        # c_cache =self.net.fused_filter_gate_conv_c(c)
        # c_cache = c_cache.reshape(c_cache.shape[0], self.num_layer, self.filter_size*2, c_cache.shape[2], c_cache.shape[3])
        # c_cache = c_cache.permute(1, 0, 2, 3, 4) # [num_layers, batch_size, res_channels, height, width]

        for i_h in range(self.num_height):
            feat = self.net.reverse_fast(x if i_h == 0 else x_new, c_cache[:, :, :, i_h:i_h+1, :])[:, :, -1, :].unsqueeze(2)

            log_s_t = self.proj_log_s_t(feat)
            log_s = log_s_t[:, :self.in_channel]
            t = log_s_t[:, self.in_channel:]

            x_new = (z[:, :, i_h, :].unsqueeze(2) - t) * torch.exp(-log_s)
            x = torch.cat((x, x_new), 2)

        return x[:, :, 1:, :], c_cache


class Flow(nn.Module):
    def __init__(self, in_channel, cin_channel, filter_size, num_layer, num_height, layers_per_dilation_h_cycle):
        super().__init__()

        self.coupling = WaveFlowCoupling2D(in_channel, cin_channel, filter_size=filter_size, num_layer=num_layer,
                                           num_height=num_height,
                                           layers_per_dilation_h_cycle=layers_per_dilation_h_cycle)

    def forward(self, x, c=None, i=None, debug=False):
        logdet = 0

        # vanilla reverse_order ops
        x = reverse_order(x)
        c = reverse_order(c)

        out, logdet_af, log_s, t = self.coupling(x, c, debug)
        logdet = logdet + logdet_af

        if debug:
            return out, c, logdet, log_s, t
        else:
            return out, c, logdet, None, None

    def reverse(self, z, c=None):
        z, c = self.coupling.reverse(z, c)

        z = reverse_order(z)
        c = reverse_order(c)

        return z, c

    def reverse_fast(self, z, c_cache=None):
        z = reverse_order(z)
        # c = reverse_order(c) # safe

        # for naive version, uses c_cache
        z, c_cache = self.coupling.reverse_fast(z, c_cache)

        return z, c_cache


class NanoFlowNaive(nn.Module):
    def __init__(self, in_channel, cin_channel, res_channel, n_height, n_flow, n_layer, layers_per_dilation_h_cycle):
        super().__init__()
        self.in_channel = in_channel
        self.cin_channel = cin_channel
        self.res_channel = res_channel
        self.n_height = n_height
        self.n_flow = n_flow
        self.n_layer = n_layer

        self.layers_per_dilation_h_cycle = layers_per_dilation_h_cycle

        # major change: shared flow weights
        self.flow = Flow(self.in_channel, self.cin_channel, filter_size=self.res_channel,
                         num_layer=self.n_layer, num_height=self.n_height,
                         layers_per_dilation_h_cycle=self.layers_per_dilation_h_cycle)

        self.upsample_conv = nn.ModuleList()
        for s in [16, 16]:
            convt = nn.ConvTranspose2d(1, 1, (3, 2 * s), padding=(1, s // 2), stride=(1, s))
            convt = nn.utils.weight_norm(convt)
            nn.init.kaiming_normal_(convt.weight)
            self.upsample_conv.append(convt)
            self.upsample_conv.append(nn.LeakyReLU(0.4))

        self.upsample_conv_kernel_size = (2 * s) ** 2
        self.upsample_conv_stride = s ** 2

    def forward(self, x, c, debug=False):
        x = x.unsqueeze(1)
        B, _, T = x.size()
        #  Upsample spectrogram to size of audio
        c = self.upsample(c)
        assert(c.size(2) >= x.size(2))
        if c.size(2) > x.size(2):
            c = c[:, :, :x.size(2)]
        x, c = squeeze_to_2d(x, c, h=self.n_height)
        out = x

        logdet = 0
        for i in range(self.n_flow):
            out, c, logdet_new, _, _ = self.flow(out, c, i)
            logdet = logdet + logdet_new

        return out, logdet

    def reverse(self, c, temp=1.0, debug_z=None):
        c = self.upsample(c)
        # trim conv artifacts. maybe pad spec to kernel multiple
        time_cutoff = self.upsample_conv_kernel_size - self.upsample_conv_stride
        c = c[:, :, :-time_cutoff]

        B, _, T_c = c.size()

        _, c = squeeze_to_2d(None, c, h=self.n_height)

        if debug_z is None:
            # sample gaussian noise that matches c
            q_0 = Normal(c.new_zeros((B, 1, c.size()[2], c.size()[3])), c.new_ones((B, 1, c.size()[2], c.size()[3])))
            z = q_0.sample() * temp
        else:
            z = debug_z

        for i in range(self.n_flow):
            z, c = self.flow.reverse(z, c)

        x = unsqueeze_to_1d(z, self.n_height)

        return x

    def reverse_fast(self, c, temp=1.0, debug_z=None):
        c = self.upsample(c)
        # trim conv artifacts. maybe pad spec to kernel multiple
        time_cutoff = self.upsample_conv_kernel_size - self.upsample_conv_stride
        c = c[:, :, :-time_cutoff]

        B, _, T_c = c.size()

        _, c = squeeze_to_2d(None, c, h=self.n_height)

        if debug_z is None:
            # sample gaussian noise that matches c
            q_0 = Normal(c.new_zeros((B, 1, c.size()[2], c.size()[3])), c.new_ones((B, 1, c.size()[2], c.size()[3])))
            z = q_0.sample() * temp
        else:
            z = debug_z

        c_cache =self.flow.coupling.net.fused_filter_gate_conv_c(c)
        c_cache = c_cache.reshape(c_cache.shape[0], self.n_layer, self.res_channel*2, c_cache.shape[2], c_cache.shape[3])
        c_cache = c_cache.permute(1, 0, 2, 3, 4) # [num_layers, batch_size, res_channels, height, width]
        c_cache_reversed = reverse_order(c_cache, dim=3)

        for i in range(self.n_flow):
            c_cache_i = c_cache if i % 2 == 0 else c_cache_reversed
            z, _ = self.flow.reverse_fast(z, c_cache_i)

        x = unsqueeze_to_1d(z, self.n_height)

        return x

    def upsample(self, c):
        c = c.unsqueeze(1)
        for f in self.upsample_conv:
            c = f(c)
        c = c.squeeze(1)
        return c

    def remove_weight_norm(self):
        # remove weight norm from all weights
        for layer in self.upsample_conv.children():
            try:
                torch.nn.utils.remove_weight_norm(layer)
            except ValueError:
                pass

        net = self.flow.coupling.net
        torch.nn.utils.remove_weight_norm(net.front_conv[0].conv)
        for resblock in net.res_blocks.children():
            torch.nn.utils.remove_weight_norm(resblock.filter_gate_conv.conv)
            torch.nn.utils.remove_weight_norm(resblock.filter_gate_conv_c)
            torch.nn.utils.remove_weight_norm(resblock.res_skip_conv)

        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("weight_norm removed: {} params".format(total_params))

    def fuse_conditioning_layers(self):
        # fuse mel-spec conditioning layers into one big conv weight
        net = self.flow.coupling.net
        cin_channels = net.res_blocks[0].cin_channels
        out_channels = net.res_blocks[0].out_channels
        fused_filter_gate_conv_c = nn.Conv2d(cin_channels, 2 * out_channels * self.n_layer, kernel_size=1)
        fused_filter_gate_conv_c_weight = []
        fused_filter_gate_conv_c_bias = []
        for resblock in net.res_blocks.children():
            fused_filter_gate_conv_c_weight.append(resblock.filter_gate_conv_c.weight)
            fused_filter_gate_conv_c_bias.append(resblock.filter_gate_conv_c.bias)
            del resblock.filter_gate_conv_c

        fused_filter_gate_conv_c.weight = torch.nn.Parameter(torch.cat(fused_filter_gate_conv_c_weight).clone())
        fused_filter_gate_conv_c.bias = torch.nn.Parameter(torch.cat(fused_filter_gate_conv_c_bias).clone())
        net.fused_filter_gate_conv_c = fused_filter_gate_conv_c

        print("INFO: conditioning layers fused for performance: only reverse_fast function can be used for inference!")
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("model after optimization: {} params".format(total_params))


# end of experimental model 1
#################################################################################

if __name__ == "__main__":
    x = torch.randn((2, 15872)).cuda()
    c = torch.randn((2, 80, 62)).cuda()
    net = NanoFlowNaive(1, 80, 64, 8, 4, 8, 1).cuda()
    out = net(x, c)

    with torch.no_grad():
        out = net.reverse(c)
        # remove all weight_norm from the model
        net.remove_weight_norm()
        # fuse mel-spec conditioning layer weights to maximize speed
        net.fuse_conditioning_layers()
        out = net.reverse_fast(c)