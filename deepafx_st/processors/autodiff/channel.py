import torch

from deepafx_st.processors.autodiff.compressor import Compressor
from deepafx_st.processors.autodiff.peq import ParametricEQ
from deepafx_st.processors.autodiff.fir import FIRFilter
from deepafx_st.processors.autodiff.reverb import SchroederReverberator, NoiseShapedReverb


class AutodiffChannel(torch.nn.Module):
    def __init__(self, sample_rate):
        super().__init__()

        self.peq = ParametricEQ(sample_rate)
        self.comp = Compressor(sample_rate)
        self.ports = [self.peq.ports, self.comp.ports]
        self.num_control_params = (
            self.peq.num_control_params + self.comp.num_control_params
        )

    def forward(self, x, p, sample_rate=24000, **kwargs):

        # split params between EQ and Comp.
        p_peq = p[:, : self.peq.num_control_params]
        p_comp = p[:, self.peq.num_control_params :]

        y = self.peq(x, p_peq, sample_rate)
        y = self.comp(y, p_comp, sample_rate)

        return y


class AutodiffChannelReverb(torch.nn.Module):
    def __init__(self, sample_rate):
        super().__init__()

        self.reverb = NoiseShapedReverb(sample_rate)
        self.peq = ParametricEQ(sample_rate)
        self.comp = Compressor(sample_rate)
        self.ports = [self.reverb.ports, self.peq.ports, self.comp.ports]
        self.num_control_params = (
            self.reverb.num_control_params + self.peq.num_control_params + self.comp.num_control_params
        )

    def forward(self, x, p, sample_rate=24000, **kwargs):

        # split params between reverb EQ and Comp.
        p_reverb = p[:, : self.reverb.num_control_params]
        p_peq = p[:, self.reverb.num_control_params : self.reverb.num_control_params + self.peq.num_control_params]
        p_comp = p[:, self.reverb.num_control_params + self.peq.num_control_params :]

        y = self.reverb(x, p_reverb, sample_rate)
        y = self.peq(y, p_peq, sample_rate)
        y = self.comp(y, p_comp, sample_rate)

        return y