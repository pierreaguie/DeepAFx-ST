import torch
from deepafx_st.processors.processor import Processor
import dasp_pytorch
from functools import partial


def schroeder(x : torch.Tensor,
              fs : float,
              m_comb : torch.Tensor,
              m_ap : torch.Tensor,
              g_comb_1 : float,
              g_comb_2 : float,
              g_comb_3 : float,
              g_comb_4 : float,
              g_ap_1 : float,
              g_ap_2 : float):
    
    g_comb = torch.tensor([g_comb_1, g_comb_2, g_comb_3, g_comb_4], device=x.device)
    g_ap = torch.tensor([g_ap_1, g_ap_2], device=x.device)
    m_comb = m_comb.to(x.device)
    m_ap = m_ap.to(x.device)
    
    n_fft = 2 ** torch.ceil(torch.log2(torch.tensor(x.shape[-1] + x.shape[-1] - 1)))
    n_fft = n_fft.int()


    z = torch.exp(1j*2*torch.pi*torch.arange(0, n_fft//2+1, device=x.device)/n_fft)

    z_comb = torch.pow(z.expand(m_comb.size(1), -1).permute(1, 0), -m_comb)
    H_comb = z_comb / (1 - g_comb * z_comb)
    H_comb = torch.sum(H_comb, dim=1)

    z_ap = torch.pow(z.expand(m_ap.size(1), -1).permute(1, 0), -m_ap)
    H_ap = (-g_ap + z_ap) / (1 - g_ap * z_ap)
    H_ap = torch.prod(H_ap, dim=1)

    H = H_comb * H_ap

    X = torch.fft.rfft(x, n_fft)
    Y = H * X
    y = torch.fft.irfft(Y.squeeze(), n_fft)
    y = y[: x.shape[-1]]

    return y


class SchroederReverberator(Processor):
    def __init__(
        self,
        sample_rate,
        min_g_comb=0.0,
        default_g_comb=0.1,
        max_g_comb=.9,
        min_g_ap=0.0,
        default_g_ap=0.1,
        max_g_ap=0.9,
        eps=1e-8,
    ):
        """ """
        super().__init__()
        self.sample_rate = sample_rate
        self.eps = eps
        self.ports = [
            {
                "name": "Comb filter #1 gain",
                "min": min_g_comb,
                "max": max_g_comb,
                "default": default_g_comb,
                "units": "",
            },
            {
                "name": "Comb filter #2 gain",
                "min": min_g_comb,
                "max": max_g_comb,
                "default": default_g_comb,
                "units": "",
            },
            {
                "name": "Comb filter #3 gain",
                "min": min_g_comb,
                "max": max_g_comb,
                "default": default_g_comb,
                "units": "",
            },
            {
                "name": "Comb filter #4 gain",
                "min": min_g_comb,
                "max": max_g_comb,
                "default": default_g_comb,
                "units": "",
            },

            {
                "name": "Allpass filter #1 gain",
                "min": min_g_ap,
                "max": max_g_ap,
                "default": default_g_ap,
                "units": "s",
            },
            {
                "name": "Allpass filter #2 gain",
                "min": min_g_ap,
                "max": max_g_ap,
                "default": default_g_ap,
                "units": "s",
            },
        ]

        self.num_control_params = len(self.ports)
        self.m_comb = (torch.tensor([[29.7, 37.1, 41.4, 43.7]]) * 1e-3 * sample_rate).to(int)
        self.m_ap = (torch.tensor([[96.83, 32.92]]) * 1e-3 * sample_rate).to(int)

    def forward(self, x, p, sample_rate=24000, **kwargs):

        bs, chs, s = x.size()

        self.m_comb.to(x.device)
        self.m_ap.to(x.device)

        inputs = torch.split(x, 1, 0)
        params = torch.split(p, 1, 0)

        y = []  # loop over batch dimension
        for input, param in zip(inputs, params):
            denorm_param = self.denormalize_params(param.view(-1))
            y.append(schroeder(input.view(-1), sample_rate, self.m_comb, self.m_ap, *denorm_param))

        return torch.stack(y, dim=0).view(bs, 1, -1)



def noise_shaped_reverberation(
    x: torch.Tensor,
    sample_rate: float,
    band0_gain: torch.Tensor,
    band1_gain: torch.Tensor,
    band2_gain: torch.Tensor,
    band3_gain: torch.Tensor,
    band4_gain: torch.Tensor,
    band5_gain: torch.Tensor,
    band6_gain: torch.Tensor,
    band7_gain: torch.Tensor,
    band8_gain: torch.Tensor,
    band9_gain: torch.Tensor,
    band10_gain: torch.Tensor,
    band11_gain: torch.Tensor,
    band0_decay: torch.Tensor,
    band1_decay: torch.Tensor,
    band2_decay: torch.Tensor,
    band3_decay: torch.Tensor,
    band4_decay: torch.Tensor,
    band5_decay: torch.Tensor,
    band6_decay: torch.Tensor,
    band7_decay: torch.Tensor,
    band8_decay: torch.Tensor,
    band9_decay: torch.Tensor,
    band10_decay: torch.Tensor,
    band11_decay: torch.Tensor,
    mix: torch.Tensor,
    num_samples: int = 65536,
    num_bandpass_taps: int = 1023,
):
    """Artificial reverberation using frequency-band noise shaping.

    This differentiable artificial reverberation model is based on the idea of
    filtered noise shaping, similar to that proposed in [1]. This approach leverages
    the well known idea that a room impulse response (RIR) can be modeled as the direct sound,
    a set of early reflections, and a decaying noise-like tail [2].

    [1] Steinmetz, Christian J., Vamsi Krishna Ithapu, and Paul Calamia.
        "Filtered noise shaping for time domain room impulse response estimation from reverberant speech."
        2021 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA). IEEE, 2021.

    [2] Moorer, James A.
        "About this reverberation business."
        Computer Music Journal (1979): 13-28.

    Args:
        x (torch.Tensor): Input audio signal. Shape (bs, chs, seq_len).
        sample_rate (float): Audio sample rate.
        band0_gain (torch.Tensor): Gain for first octave band on (0,1). Shape (bs, 1).
        band1_gain (torch.Tensor): Gain for second octave band on (0,1). Shape (bs, 1).
        band2_gain (torch.Tensor): Gain for third octave band on (0,1). Shape (bs, 1).
        band3_gain (torch.Tensor): Gain for fourth octave band on (0,1). Shape (bs, 1).
        band4_gain (torch.Tensor): Gain for fifth octave band on (0,1). Shape (bs, 1).
        band5_gain (torch.Tensor): Gain for sixth octave band on (0,1). Shape (bs, 1).
        band6_gain (torch.Tensor): Gain for seventh octave band on (0,1). Shape (bs, 1).
        band7_gain (torch.Tensor): Gain for eighth octave band on (0,1). Shape (bs, 1).
        band8_gain (torch.Tensor): Gain for ninth octave band on (0,1). Shape (bs, 1).
        band9_gain (torch.Tensor): Gain for tenth octave band on (0,1). Shape (bs, 1).
        band10_gain (torch.Tensor): Gain for eleventh octave band on (0,1). Shape (bs, 1).
        band11_gain (torch.Tensor): Gain for twelfth octave band on (0,1). Shape (bs, 1).
        band0_decays (torch.Tensor): Decay parameter for first octave band (0,1). Shape (bs, 1).
        band1_decays (torch.Tensor): Decay parameter for second octave band (0,1). Shape (bs, 1).
        band2_decays (torch.Tensor): Decay parameter for third octave band (0,1). Shape (bs, 1).
        band3_decays (torch.Tensor): Decay parameter for fourth octave band (0,1). Shape (bs, 1).
        band4_decays (torch.Tensor): Decay parameter for fifth octave band (0,1). Shape (bs, 1).
        band5_decays (torch.Tensor): Decay parameter for sixth octave band (0,1). Shape (bs, 1).
        band6_decays (torch.Tensor): Decay parameter for seventh octave band (0,1). Shape (bs, 1).
        band7_decays (torch.Tensor): Decay parameter for eighth octave band (0,1). Shape (bs, 1).
        band8_decays (torch.Tensor): Decay parameter for ninth octave band (0,1). Shape (bs, 1).
        band9_decays (torch.Tensor): Decay parameter for tenth octave band (0,1). Shape (bs, 1).
        band10_decays (torch.Tensor): Decay parameter for eleventh octave band (0,1). Shape (bs, 1).
        band11_decays (torch.Tensor): Decay parameter for twelfth octave band (0,1). Shape (bs, 1).
        mix (torch.Tensor): Mix between dry and wet signal. Shape (bs, 1).
        num_samples (int, optional): Number of samples to use for IR generation. Defaults to 88200.
        num_bandpass_taps (int, optional): Number of filter taps for the octave band filterbank filters. Must be odd. Defaults to 1023.

    Returns:
        y (torch.Tensor): Reverberated signal. Shape (bs, chs, seq_len).

    """
    assert num_bandpass_taps % 2 == 1, "num_bandpass_taps must be odd"

    bs, chs, seq_len = x.size()
    assert chs <= 2, "only mono/stereo signals are supported"

    # if mono copy to stereo
    if chs == 1:
        x = x.repeat(1, 2, 1)
        chs = 2

    # stack gains and decays into a single tensor
    band_gains = torch.stack(
        [
            band0_gain,
            band1_gain,
            band2_gain,
            band3_gain,
            band4_gain,
            band5_gain,
            band6_gain,
            band7_gain,
            band8_gain,
            band9_gain,
            band10_gain,
            band11_gain,
        ],
        dim=1
    ).to(x.device)
    band_gains = band_gains.unsqueeze(-1)

    band_decays = torch.stack(
        [
            band0_decay,
            band1_decay,
            band2_decay,
            band3_decay,
            band4_decay,
            band5_decay,
            band6_decay,
            band7_decay,
            band8_decay,
            band9_decay,
            band10_decay,
            band11_decay,
        ],
        dim=1,
    ).to(x.device)
    band_decays = band_decays.unsqueeze(-1)

    # create the octave band filterbank filters
    filters = dasp_pytorch.signal.octave_band_filterbank(1023, sample_rate)
    filters = filters.type_as(x)
    num_bands = filters.shape[0]

    # reshape gain, decay, and mix parameters
    band_gains = band_gains.view(bs, 1, num_bands, 1)
    band_decays = band_decays.view(bs, 1, num_bands, 1)
    mix = mix.view(bs, 1, 1)

    # generate white noise for IR generation
    pad_size = num_bandpass_taps - 1
    wn = torch.randn(bs * 2, num_bands, num_samples + pad_size).type_as(x)

    # filter white noise signals with each bandpass filter
    wn_filt = torch.nn.functional.conv1d(
        wn,
        filters,
        groups=num_bands,
        # padding=self.num_taps -1,
    )
    # shape: (bs * 2, num_bands, num_samples)
    wn_filt = wn_filt.view(bs, 2, num_bands, num_samples)

    # apply bandwise decay parameters (envelope)
    t = torch.linspace(0, 1, steps=num_samples).type_as(x)  # timesteps
    band_decays = (band_decays * 10.0) + 1.0
    env = torch.exp(-band_decays * t.view(1, 1, 1, -1))
    wn_filt *= env * band_gains

    # sum signals to create impulse shape: bs, 2, 1, num_samp
    w_filt_sum = wn_filt.mean(2, keepdim=True)

    # apply impulse response for each batch item (vectorized)

    x_pad = torch.nn.functional.pad(x, (num_samples - 1, 0))
    vconv1d = torch.vmap(partial(torch.nn.functional.conv1d, groups=2), in_dims=0)
    y = vconv1d(x_pad, torch.flip(w_filt_sum, dims=[-1]))

    # create a wet/dry mix
    y = (1 - mix) * x + mix * y
    return y[:, :1, :]



class NoiseShapedReverb(Processor):

    def __init__(self,
                 sample_rate,
                min_band_gain: float = 0.0,
                max_band_gain: float = 1.0,
                min_band_decay: float = 0.0,
                max_band_decay: float = 1.0,
                min_mix: float = 0.0,
                max_mix: float = 1.0,
                ):
        super().__init__()
        self.sample_rate = sample_rate
        self.ports = [{
                "name": f"Band {i} gain",
                "min": min_band_gain,
                "max": max_band_gain,
                "default": 0.0,
                "units": "",
            } for i in range(12)] + [{
                "name": f"Band {i} decay",
                "min": min_band_decay,
                "max": max_band_decay,
                "default": 0.0,
                "units": "",
            } for i in range(12)] + [{
                "name": "Mix",
                "min": min_mix,
                "max": max_mix,
                "default": 0.5,
                "units": "",
            }]
        
        self.num_control_params = len(self.ports)


    def forward(self, x, p, sample_rate=24000, **kwargs):
        bs, chs, s = x.size()
        inputs = torch.split(x, 1, 0)
        params = torch.split(p, 1, 0)

        y = []  # loop over batch dimension
        for input, param in zip(inputs, params):
            denorm_param = self.denormalize_params(param.view(-1))
            band0_gain = torch.tensor([[denorm_param[0]]], device=x.device)
            band1_gain = torch.tensor([[denorm_param[1]]], device=x.device)
            band2_gain = torch.tensor([[denorm_param[2]]], device=x.device)
            band3_gain = torch.tensor([[denorm_param[3]]], device=x.device)
            band4_gain = torch.tensor([[denorm_param[4]]], device=x.device)
            band5_gain = torch.tensor([[denorm_param[5]]], device=x.device)
            band6_gain = torch.tensor([[denorm_param[6]]], device=x.device)
            band7_gain = torch.tensor([[denorm_param[7]]], device=x.device)
            band8_gain = torch.tensor([[denorm_param[8]]], device=x.device)
            band9_gain = torch.tensor([[denorm_param[9]]], device=x.device)
            band10_gain = torch.tensor([[denorm_param[10]]], device=x.device)
            band11_gain = torch.tensor([[denorm_param[11]]], device=x.device)
            band0_decay = torch.tensor([[denorm_param[12]]], device=x.device)
            band1_decay = torch.tensor([[denorm_param[13]]], device=x.device)
            band2_decay = torch.tensor([[denorm_param[14]]], device=x.device)
            band3_decay = torch.tensor([[denorm_param[15]]], device=x.device)
            band4_decay = torch.tensor([[denorm_param[16]]], device=x.device)
            band5_decay = torch.tensor([[denorm_param[17]]], device=x.device)
            band6_decay = torch.tensor([[denorm_param[18]]], device=x.device)
            band7_decay = torch.tensor([[denorm_param[19]]], device=x.device)
            band8_decay = torch.tensor([[denorm_param[20]]], device=x.device)
            band9_decay = torch.tensor([[denorm_param[21]]], device=x.device)
            band10_decay = torch.tensor([[denorm_param[22]]], device=x.device)
            band11_decay = torch.tensor([[denorm_param[23]]], device=x.device)
            mix = torch.tensor([[denorm_param[24]]], device=x.device)
            y.append(noise_shaped_reverberation(input.view(1,1,-1), sample_rate, 
                                                band0_gain, band1_gain, band2_gain, band3_gain, band4_gain, band5_gain, 
                                                band6_gain, band7_gain, band8_gain, band9_gain, band10_gain, band11_gain, 
                                                band0_decay, band1_decay, band2_decay, band3_decay, band4_decay, band5_decay, 
                                                band6_decay, band7_decay, band8_decay, band9_decay, band10_decay, band11_decay,
                                                mix))

        return torch.stack(y, dim=0).view(bs, 1, -1)
