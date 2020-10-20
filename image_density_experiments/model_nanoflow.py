import math

import torch
import torch.nn as nn

from modules import (Conv2d, Conv2dZeros, ActNorm2d, InvertibleConv1x1,
                     Permute2d, LinearZeros, SqueezeLayer,
                     Split2d, gaussian_likelihood, gaussian_sample)
from utils import split_feature, uniform_binning_correction


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, flow_embed_dim, level):
        super().__init__()
        self.conv1 = Conv2d(in_channels+(flow_embed_dim*(2**(level))), hidden_channels)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = Conv2d(hidden_channels, hidden_channels, kernel_size=(3,3))
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = Conv2d(hidden_channels, hidden_channels, kernel_size=(3,3))
        self.relu3 = nn.ReLU(inplace=False)
        self.conv4 = Conv2d(hidden_channels, hidden_channels, kernel_size=(3,3))
        self.relu4 = nn.ReLU(inplace=False)
        self.conv5 = Conv2d(hidden_channels, hidden_channels, kernel_size=(3,3))
        self.relu5 = nn.ReLU(inplace=False)
        self.conv6 = Conv2d(hidden_channels, hidden_channels, kernel_size=(3,3))
        self.relu6 = nn.ReLU(inplace=False)

    def forward(self, flow_embed, multgate, input):
        out = torch.cat((input, flow_embed), dim=1)
        multgate = torch.exp(multgate)

        out = self.conv1(out)
        out = out * multgate[0]
        out = self.relu1(out)
        out = self.conv2(out)
        out = out * multgate[1]
        out = self.relu2(out)
        out = self.conv3(out)
        out = out * multgate[2]
        out = self.relu3(out)
        out = self.conv4(out)
        out = out * multgate[3]
        out = self.relu4(out)
        out = self.conv5(out)
        out = out * multgate[4]
        out = self.relu5(out)
        out = self.conv6(out)
        out = out * multgate[5]
        out = self.relu6(out)

        return out


class FlowStep(nn.Module):
    def __init__(self, in_channels, hidden_channels, actnorm_scale,
                 flow_permutation, flow_coupling, LU_decomposed, flow_embed_dim, level):
        super().__init__()
        self.flow_coupling = flow_coupling

        self.actnorm = ActNorm2d(in_channels, actnorm_scale)
        self.actnorm_embed = ActNorm2d(flow_embed_dim * (2**level), actnorm_scale)

        if flow_coupling == "additive":
            self.conv_proj = Conv2dZeros(hidden_channels, in_channels // 2, kernel_size=(1,1))
        elif flow_coupling == "affine":
            self.conv_proj = Conv2dZeros(hidden_channels, in_channels, kernel_size=(1,1))

        # 2. permute
        if flow_permutation == "invconv":
            self.invconv = InvertibleConv1x1(in_channels,
                                             LU_decomposed=LU_decomposed)
            self.flow_permutation = \
                lambda z, logdet, rev: self.invconv(z, logdet, rev)
        elif flow_permutation == "shuffle":
            self.shuffle = Permute2d(in_channels, shuffle=True)
            self.flow_permutation = \
                lambda z, logdet, rev: (self.shuffle(z, rev), logdet)
        else:
            self.reverse = Permute2d(in_channels, shuffle=False)
            self.flow_permutation = \
                lambda z, logdet, rev: (self.reverse(z, rev), logdet)

        self.multgate = nn.Parameter(torch.zeros((6, hidden_channels, 1, 1)))

    def forward(self, estimator, flow_embed,  i_flow, input, logdet=None, reverse=False):
        if not reverse:
            return self.normal_flow(estimator, flow_embed, self.multgate, i_flow, input, logdet)
        else:
            return self.reverse_flow(estimator, flow_embed, self.multgate, i_flow, input, logdet)

    def normal_flow(self, estimator, flow_embed, multgate, i_flow, input, logdet):
        assert input.size(1) % 2 == 0

        # 1. actnorm
        z, logdet = self.actnorm(input, logdet=logdet, reverse=False)
        # 1.1 also actnorm embed
        flow_embed = flow_embed.expand((z.shape[0], flow_embed.shape[1], flow_embed.shape[2], flow_embed.shape[3]))
        flow_embed, _= self.actnorm_embed(flow_embed)

        # 2. permute
        z, logdet = self.flow_permutation(z, logdet, False)

        # 3. coupling
        z1, z2 = split_feature(z, "split")
        if self.flow_coupling == "additive":
            z1_out = estimator(flow_embed, multgate, z1)
            z1_out = self.conv_proj(z1_out)
            z2 = z2 + z1_out
        elif self.flow_coupling == "affine":
            h = estimator(flow_embed, multgate, z1)
            h = self.conv_proj(h)
            shift, scale = split_feature(h, "cross")
            scale = torch.sigmoid(scale + 2.)
            z2 = z2 + shift
            z2 = z2 * scale
            logdet = torch.sum(torch.log(scale), dim=[1, 2, 3]) + logdet
        z = torch.cat((z1, z2), dim=1)

        return z, logdet

    def reverse_flow(self, estimator, flow_embed, multgate, i_flow, input, logdet):
        assert input.size(1) % 2 == 0

        flow_embed = flow_embed.expand((input.shape[0], flow_embed.shape[1], flow_embed.shape[2], flow_embed.shape[3]))
        flow_embed, _= self.actnorm_embed(flow_embed, reverse=False)  # NOTE: reverse=False is the correct usage for this

        # 1.coupling
        z1, z2 = split_feature(input, "split")
        if self.flow_coupling == "additive":
            out = estimator(flow_embed, multgate, z1)
            out = self.conv_proj(out)
            z2 = z2 - out
        elif self.flow_coupling == "affine":
            # h = self.block(z1)
            h = estimator(flow_embed, multgate, z1)
            h = self.conv_proj(h)
            shift, scale = split_feature(h, "cross")
            scale = torch.sigmoid(scale + 2.)
            z2 = z2 / scale
            z2 = z2 - shift
            logdet = -torch.sum(torch.log(scale), dim=[1, 2, 3]) + logdet
        z = torch.cat((z1, z2), dim=1)

        # 2. permute
        z, logdet = self.flow_permutation(z, logdet, True)

        # 3. actnorm
        z, logdet = self.actnorm(z, logdet=logdet, reverse=True)

        return z, logdet


class FlowNet(nn.Module):
    def __init__(self, image_shape, hidden_channels, K, L,
                 actnorm_scale, flow_permutation, flow_coupling,
                 LU_decomposed, flow_embed_dim):
        super().__init__()

        self.layers = nn.ModuleList()
        self.output_shapes = []

        self.K = K
        self.L = L

        H, W, C = image_shape

        self.estimator = nn.ModuleList()
        self.estimator_idx = []
        self.flow_idx = []
        self.embedding = nn.ParameterList()
        self.flow_embed_dim = flow_embed_dim
        flow_counter = 0
        for i in range(L):
            # 1. Squeeze
            C, H, W = C * 4, H // 2, W // 2
            self.layers.append(SqueezeLayer(factor=2))
            self.output_shapes.append([-1, C, H, W])
            self.estimator_idx.append(i)
            self.embedding.append(None)
            self.flow_idx.append(flow_counter)
            flow_counter += 1
            # 3. coupling
            if flow_coupling == "additive":
                block = Block(C // 2,
                                  C // 2,
                                  hidden_channels, flow_embed_dim, i)
            elif flow_coupling == "affine":
                block = Block(C // 2,
                                  C,
                                  hidden_channels, flow_embed_dim, i)

            self.estimator.append(block)
            # 2. K FlowStep
            for _ in range(K):
                self.layers.append(
                    FlowStep(in_channels=C,
                             hidden_channels=hidden_channels,
                             actnorm_scale=actnorm_scale,
                             flow_permutation=flow_permutation,
                             flow_coupling=flow_coupling,
                             LU_decomposed=LU_decomposed,
                             flow_embed_dim=flow_embed_dim,
                             level=i))
                self.output_shapes.append([-1, C, H, W])
                self.estimator_idx.append(i)
                self.embedding.append(nn.Parameter(torch.randn([1, flow_embed_dim * (2**i), H, W]), requires_grad=True))
                self.flow_idx.append(flow_counter)
                flow_counter += 1

            # 3. Split2d
            if i < L - 1:
                self.layers.append(Split2d(num_channels=C))
                self.output_shapes.append([-1, C // 2, H, W])
                self.estimator_idx.append(i)
                self.embedding.append(None)
                self.flow_idx.append(flow_counter)
                flow_counter += 1
                C = C // 2

    def forward(self, input, logdet=0., reverse=False, temperature=None):
        if reverse:
            return self.decode(input, temperature)
        else:
            return self.encode(input, logdet)

    def encode(self, z, logdet=0.0):

        for layer, shape, i_est, i_flow in zip(self.layers, self.output_shapes, self.estimator_idx, self.flow_idx):
            if isinstance(layer, SqueezeLayer) or isinstance(layer, Split2d):
                z, logdet = layer(z, logdet, reverse=False)
            elif isinstance(layer, FlowStep):
                flow_embed_i = self.embedding[i_flow]
                z, logdet = layer(self.estimator[i_est], flow_embed_i, i_flow, z, logdet, reverse=False)

        return z, logdet

    def decode(self, z, temperature=None):
        for layer, i_est, i_flow in zip(reversed(self.layers), reversed(self.estimator_idx), reversed(self.flow_idx)):
            if isinstance(layer, Split2d):
                z, logdet = layer(z, logdet=0, reverse=True,
                                  temperature=temperature)
            elif isinstance(layer, SqueezeLayer):
                z, logdet = layer(z, logdet=0, reverse=True)
            else:
                flow_embed_i = self.embedding[i_flow]
                z, logdet = layer(self.estimator[i_est], flow_embed_i, i_flow, z, logdet=0, reverse=True)
        return z


class NanoFlow(nn.Module):
    def __init__(self, image_shape, hidden_channels, K, L, actnorm_scale,
                 flow_permutation, flow_coupling, LU_decomposed, flow_embed_dim, y_classes,
                 learn_top, y_condition):
        super().__init__()
        self.flow = FlowNet(image_shape=image_shape,
                            hidden_channels=hidden_channels,
                            K=K,
                            L=L,
                            actnorm_scale=actnorm_scale,
                            flow_permutation=flow_permutation,
                            flow_coupling=flow_coupling,
                            LU_decomposed=LU_decomposed,
                            flow_embed_dim=flow_embed_dim)
        self.y_classes = y_classes
        self.y_condition = y_condition

        self.learn_top = learn_top

        # learned prior
        if learn_top:
            C = self.flow.output_shapes[-1][1]
            self.learn_top_fn = Conv2dZeros(C * 2, C * 2)

        if y_condition:
            C = self.flow.output_shapes[-1][1]
            self.project_ycond = LinearZeros(y_classes, 2 * C)
            self.project_class = LinearZeros(C, y_classes)

        self.register_buffer("prior_h",
                             torch.zeros([1,
                                          self.flow.output_shapes[-1][1] * 2,
                                          self.flow.output_shapes[-1][2],
                                          self.flow.output_shapes[-1][3]]))

        self.num_param = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("num_param: {}".format(self.num_param))

    def prior(self, data, y_onehot=None):
        if data is not None:
            h = self.prior_h.repeat(data.shape[0], 1, 1, 1)
        else:
            # Hardcoded a batch size of 32 here
            h = self.prior_h.repeat(32, 1, 1, 1)

        channels = h.size(1)

        if self.learn_top:
            h = self.learn_top_fn(h)

        if self.y_condition:
            assert y_onehot is not None
            yp = self.project_ycond(y_onehot)
            h += yp.view(data.shape[0], channels, 1, 1)

        return split_feature(h, "split")

    def forward(self, x=None, y_onehot=None, z=None, temperature=None,
                reverse=False):
        if reverse:
            return self.reverse_flow(z, y_onehot, temperature)
        else:
            return self.normal_flow(x, y_onehot)

    def normal_flow(self, x, y_onehot):
        b, c, h, w = x.shape

        x, logdet = uniform_binning_correction(x)

        z, objective = self.flow(x, logdet=logdet, reverse=False)

        mean, logs = self.prior(x, y_onehot)
        objective += gaussian_likelihood(mean, logs, z)

        if self.y_condition:
            y_logits = self.project_class(z.mean(2).mean(2))
        else:
            y_logits = None

        # Full objective - converted to bits per dimension
        bpd = (-objective) / (math.log(2.) * c * h * w)

        return z, bpd, y_logits

    def reverse_flow(self, z, y_onehot, temperature):
        with torch.no_grad():
            if z is None:
                mean, logs = self.prior(z, y_onehot)
                z = gaussian_sample(mean, logs, temperature)
            x = self.flow(z, temperature=temperature, reverse=True)
        return x

    def set_actnorm_init(self):
        for name, m in self.named_modules():
            if isinstance(m, ActNorm2d):
                m.inited = True
