import torch
import torchtestcase

from nde.transforms import splines


class CubicSplineTest(torchtestcase.TorchTestCase):
    def test_forward_inverse_are_consistent(self):
        num_bins = 10
        shape = [2,3,4]

        unnormalized_widths = torch.randn(*shape, num_bins)
        unnormalized_heights = torch.randn(*shape, num_bins)
        unnorm_derivatives_left = torch.randn(*shape, 1)
        unnorm_derivatives_right = torch.randn(*shape, 1)

        def call_spline_fn(inputs, inverse=False):
            return splines.cubic_spline(
                inputs=inputs,
                unnormalized_widths=unnormalized_widths,
                unnormalized_heights=unnormalized_heights,
                unnorm_derivatives_left=unnorm_derivatives_left,
                unnorm_derivatives_right=unnorm_derivatives_right,
                inverse=inverse
            )

        inputs = torch.rand(*shape)
        outputs, logabsdet = call_spline_fn(inputs, inverse=False)
        inputs_inv, logabsdet_inv = call_spline_fn(outputs, inverse=True)

        self.eps = 1e-4
        self.assertEqual(inputs, inputs_inv)
        self.assertEqual(logabsdet + logabsdet_inv, torch.zeros_like(logabsdet))

class UnconstrainedCubicSplineTest(torchtestcase.TorchTestCase):
    def test_forward_inverse_are_consistent(self):
        num_bins = 10
        shape = [2,3,4]

        unnormalized_widths = torch.randn(*shape, num_bins)
        unnormalized_heights = torch.randn(*shape, num_bins)
        unnorm_derivatives_left = torch.randn(*shape, 1)
        unnorm_derivatives_right = torch.randn(*shape, 1)

        def call_spline_fn(inputs, inverse=False):
            return splines.unconstrained_cubic_spline(
                inputs=inputs,
                unnormalized_widths=unnormalized_widths,
                unnormalized_heights=unnormalized_heights,
                unnorm_derivatives_left=unnorm_derivatives_left,
                unnorm_derivatives_right=unnorm_derivatives_right,
                inverse=inverse
            )

        inputs = 3 * torch.randn(*shape) # Note inputs are outside [0,1].
        outputs, logabsdet = call_spline_fn(inputs, inverse=False)
        inputs_inv, logabsdet_inv = call_spline_fn(outputs, inverse=True)

        self.eps = 1e-4
        self.assertEqual(inputs, inputs_inv)
        self.assertEqual(logabsdet + logabsdet_inv, torch.zeros_like(logabsdet))
