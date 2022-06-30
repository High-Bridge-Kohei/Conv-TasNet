import torch
import torch.nn as nn
from typing import Tuple, Optional
import torch.nn.functional as F
from asteroid.masknn import TDConvNet


class ConvTasNet(nn.Module):
    def __init__(
        self,
        in_chan=512,
        n_src=2,
        out_chan=None,
        n_blocks=8,
        n_repeats=3,
        bn_chan=128,
        hid_chan=512,
        skip_chan=128,
        conv_kernel_size=3,
        norm_type="gLN",
        mask_act="relu",
        causal=False,
    ):
        super().__init__()
        self.n_src = n_src
        self.channels = in_chan
        self.encoder = nn.Sequential(
            nn.Conv1d(1, in_chan, 16, 8, 4),
            nn.ReLU()
        )
        self.decoder = nn.ConvTranspose1d(in_chan, 1, 16, 8, 4)
        # generare masker
        self.masker = TDConvNet(
            in_chan,
            n_src,
            out_chan=out_chan,
            n_blocks=n_blocks,
            n_repeats=n_repeats,
            bn_chan=bn_chan,
            hid_chan=hid_chan,
            skip_chan=skip_chan,
            conv_kernel_size=conv_kernel_size,
            norm_type=norm_type,
            mask_act=mask_act,
            causal=causal,
        )


    def forward(self, mix: torch.Tensor) -> torch.Tensor:
        batch, _, seq_len = mix.size()
        mapped_mix = self.encoder(mix)
        est_mask = self.masker(mapped_mix)
        mapped_est_src = est_mask * mapped_mix.unsqueeze(1)
        est_src = self.decoder(
            mapped_est_src.view(batch * self.n_src, self.channels, -1)
        ).view(batch, self.n_src, -1)
        est_seq_len = est_src.size(-1)
        est_src = F.pad(est_src, [0, seq_len-est_seq_len])
        return est_src

def count_params(model):
    n_params = 0
    for p in model.parameters():
        if p.requires_grad:
            n_params += p.numel()
    print("Total parameters: %.2e" % (n_params))

if __name__ == "__main__":
    # from ptflops import get_model_complexity_info
    model = ConvTasNet()
    count_params(model)
    mix = torch.randn(1, 1, 16000)
    model.eval()
    with torch.no_grad():
        x = model(mix)
        print(x.size())

    # flops, params = get_model_complexity_info(
    #     model, (1, 16000), as_strings=True, print_per_layer_stat=False, verbose=True
    # )
    # print('{:<30}  {:<8}'.format('Computational complexity: ', flops))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))




