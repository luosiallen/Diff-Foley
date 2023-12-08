import torch
import torch.nn as nn
import torch.nn.functional as F

import sys


class STFTDiscriminator_wrapper(nn.Module):
    def __init__(self, disc_num, disc_last_act):
        super().__init__()
        """ Multi Scale STFT Discriminator """
        self.discriminators = torch.nn.ModuleList()
        for i in range(disc_num):
            self.discriminators += [STFTDiscriminator(disc_last_act)]
    
    def forward(self, x_list):
        outs = []
        # print('len xlist', len(x_list))
        for i in range(len(self.discriminators)):
            # print(i)
            x = x_list[i]                    # different scale input
            disc = self.discriminators[i]    # diff for disc
            outs += [disc(x)]
        return outs


class STFTDiscriminator(nn.Module):
    def __init__(self, disc_last_act):
        super().__init__()
        # Input: STFT Feat: B x 2 x F x T
        if disc_last_act:
            self.layers = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(in_channels=2, out_channels=32, kernel_size=(3, 8)),
                    nn.ELU()
                ),
                nn.Sequential(
                    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, dilation=(1, 1), stride=(2, 2)),
                    nn.ELU(),
                ),
                nn.Sequential(
                    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, dilation=(1, 2), stride=(2, 2)),
                    nn.ELU(),
                ),
                nn.Sequential(
                    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, dilation=(1, 4), stride=(2, 2)),
                    nn.ELU(),
                ),
                nn.Sequential(
                    nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, stride=1),
                    nn.ELU(),
                )
            ])
        else:
            print('Not using Act in last layer')
            self.layers = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(in_channels=2, out_channels=32, kernel_size=(3, 8)),
                    nn.ELU()
                ),
                nn.Sequential(
                    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, dilation=(1, 1), stride=(2, 2)),
                    nn.ELU(),
                ),
                nn.Sequential(
                    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, dilation=(1, 2), stride=(2, 2)),
                    nn.ELU(),
                ),
                nn.Sequential(
                    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, dilation=(1, 4), stride=(2, 2)),
                    nn.ELU(),
                ),
                nn.Sequential(
                    nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, stride=1),
                )
            ])
    def forward(self, x):
        feature_map = []
        for layer in self.layers:
            x = layer(x)
            feature_map.append(x)
        return feature_map
    

if __name__ == "__main__":
    disc = STFTDiscriminator()
    nparameters = sum(p.numel() for p in disc.parameters())
    print('param num:', nparameters)
    x = torch.randn(2,220500)
    stft_x = torch.stft(x, n_fft=1024, hop_length=64, win_length=256, normalized=True, return_complex=False)
    stft_x = stft_x.permute(0, 3, 1, 2)
    print(stft_x.shape)
    out_list = disc(stft_x)
    for tensor in out_list:
        print(tensor.shape)
