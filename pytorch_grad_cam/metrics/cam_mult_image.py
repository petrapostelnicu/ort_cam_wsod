# Code taken from: https://github.com/jacobgil/pytorch-grad-cam
# MIT License
#
# Copyright (c) 2021 Jacob Gildenblat
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE


import torch
import numpy as np
from typing import List, Callable
from pytorch_grad_cam.metrics.perturbation_confidence import PerturbationConfidenceMetric


def multiply_tensor_with_cam(input_tensor: torch.Tensor,
                             cam: torch.Tensor):
    """ Multiply an input tensor (after normalization)
        with a pixel attribution map
    """
    return input_tensor * cam


class CamMultImageConfidenceChange(PerturbationConfidenceMetric):
    def __init__(self):
        super(CamMultImageConfidenceChange,
              self).__init__(multiply_tensor_with_cam)


class DropInConfidence(CamMultImageConfidenceChange):
    def __init__(self):
        super(DropInConfidence, self).__init__()

    def __call__(self, *args, **kwargs):
        scores = super(DropInConfidence, self).__call__(*args, **kwargs)
        scores = -scores
        return np.maximum(scores, 0)


class IncreaseInConfidence(CamMultImageConfidenceChange):
    def __init__(self):
        super(IncreaseInConfidence, self).__init__()

    def __call__(self, *args, **kwargs):
        scores = super(IncreaseInConfidence, self).__call__(*args, **kwargs)
        return np.float32(scores > 0)
