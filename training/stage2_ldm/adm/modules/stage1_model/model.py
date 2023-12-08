


import torch.nn as nn
import torch

# from distribution import DiagonalGaussianDistribution

# Sound Encoder E:
# 1D Convolution with C=32 channels 
# B=4 Conv blocks , composed of 1. residual unit 2.down-sampling (strided convolution) skip-connection 
# with kernel K twice the Strides: K=(4,4,4,8)  S=(2,2,2,4) channels double
# 2-Layer LSTM  ELU Activation
# 1D Convolution with K=7, D=128

# Sound Decoder D:
# 1D Convolution 


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                # print('mean shape',self.mean.shape)
                # print('var shape', self.var.shape)
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel, padding):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels, stride=stride, kernel_size=kernel, padding=padding)
        )
    def forward(self, x):
        x = self.layers(x)
        return x

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel, padding):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, stride=stride, kernel_size=kernel, padding=padding)
        )
    def forward(self, x):
        x = self.layers(x)
        return x

class ResidualUnit(nn.Module):
    """ ResidualUnit:
        2 Convolution Layer
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=1),
            nn.ELU(),
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=1)
        )
    def forward(self, x):
        return x + self.layers(x)


class ConvDownBlock(nn.Module):
    """ ConvDownBlock: 
        1 Residual Unit,
        1 DownBlock(stride, kernel)  C --> 2 x C
    """
    def __init__(self, out_channels, stride, kernel):
        super().__init__()
        self.layers = nn.Sequential(
            ResidualUnit(in_channels=out_channels//2, out_channels=out_channels//2), # contain 2 convolution + 1 residual
            nn.ELU(),
            DownBlock(in_channels=out_channels//2, out_channels=out_channels, stride=stride, kernel=kernel, padding=(kernel - stride)//2),      # Downsampling Block
            nn.ELU())
    
    def forward(self, x):
        x = self.layers(x)
        return x


class ConvUpBlock(nn.Module):
    """ ConvUpBlock: 
        1 Residual Unit,
        1 DownBlock(stride, kernel)  2C --> C
    """
    def __init__(self, out_channels, stride, kernel):
        super().__init__()
        self.layers = nn.Sequential(
            ResidualUnit(in_channels=out_channels*2, out_channels=out_channels*2), # contain 2 convolution + 1 residual
            nn.ELU(),
            UpBlock(in_channels=out_channels*2, out_channels=out_channels, stride=stride, kernel=kernel, padding=(kernel - stride)//2),      # Downsampling Block
            nn.ELU())
    
    def forward(self, x):
        x = self.layers(x)
        return x

# H1=(H0+2×pad−kernel_size) / stride+1

class Encoder(nn.Module):      
    """ Sound Encoder Use Waveform input"""
    def __init__(self, enc_channels=32, enc_out_channels=256, enc_layer_num=4, enc_stride_list=[2,2,2,4], enc_lstm_layer=2, use_layernorm=False, remove_act=False):
        super().__init__()

        # channels = config['model']['AE']['channels']
        # out_channels = config['model']['AE']['out_channels']
        # layer_num = config['model']['AE']['layer_num']
        channels = enc_channels
        out_channels = enc_out_channels
        layer_num = enc_layer_num
        stride_list = enc_stride_list

        self.use_layernorm = use_layernorm

        assert layer_num == len(stride_list)
        self.layers = nn.ModuleList()
        # First Conv1d Layers:
        self.layers.append(nn.Conv1d(in_channels=1, out_channels=channels, kernel_size=1))
        self.layers.append(nn.ELU())

        for i in range(layer_num):  # default: 4
            blocks = ConvDownBlock(out_channels=2**(i+1)*channels, stride=stride_list[i], kernel=2*stride_list[i])
            self.layers.append(blocks)
            self.layers.append(nn.ELU())

        self.lstm = nn.Sequential(
                nn.LSTM(input_size=2**(i+1)*channels, hidden_size=2**(i+1)*channels, num_layers=enc_lstm_layer, batch_first=True))
        
        if not remove_act:
            self.last_conv = nn.Sequential(
                    nn.ELU(),
                    nn.Conv1d(in_channels=2**(i+1)*channels, out_channels=out_channels, kernel_size=1),
                    nn.ELU())
        else:
           print('Not Using ELU Act !!!! ======================> ')
           self.last_conv = nn.Sequential(
                    nn.ELU(),
                    nn.Conv1d(in_channels=2**(i+1)*channels, out_channels=out_channels, kernel_size=1)) 

    def forward(self, x):
        # x: WaveForm input: B x 1 x L
        for layer in self.layers:
            x = layer(x)        # B x C x L

        x = x.permute(0, 2, 1).contiguous()
        x, _ = self.lstm(x)
        x = x.permute(0, 2, 1).contiguous()
        # print(x.shape)
        x = self.last_conv(x)       # KL --> output: mean & var
        return x


class Decoder(nn.Module):
    """ Sound Decoder 
        Symmetric Arch:
    """
    def __init__(self, dec_channels=32, dec_out_channels=128, dec_layer_num=4, dec_stride_list=[4,2,2,2], dec_lstm_layer=2,use_layernorm=False):
        super().__init__()
        channels = dec_channels
        out_channels = dec_out_channels
        layer_num = dec_layer_num
        # stride_list = [2, 2, 2, 4]
        stride_list = dec_stride_list  # reversed
        assert layer_num == len(stride_list)
        self.layers1 = nn.ModuleList()
        self.layers2 = nn.ModuleList()
        # First Conv1d Layers:  Layers 1
        self.layers1.append(nn.Conv1d(in_channels=out_channels, out_channels=channels * (2**layer_num), kernel_size=1))
        self.layers1.append(nn.ELU())

        # LSTM
        self.lstm = nn.Sequential(
            nn.LSTM(input_size=channels * (2**layer_num), hidden_size=channels * (2**layer_num), num_layers=dec_lstm_layer, batch_first=True))

        # Layers 2:
        self.layers2.append(nn.ELU())
        for i in reversed(range(layer_num)):
            blocks = ConvUpBlock(out_channels=channels * (2**i), stride=stride_list[i], kernel=2*stride_list[i])
            self.layers2.append(blocks)
            self.layers2.append(nn.ELU())
        
        self.last_conv = nn.Sequential(
            nn.Conv1d(in_channels=channels * (2**i), out_channels=1, kernel_size=1)
        )
        
    
    def forward(self, x):
        for layer in self.layers1:
            x = layer(x)
        x = x.permute(0, 2, 1).contiguous()
        x, _ = self.lstm(x)
        x = x.permute(0, 2, 1).contiguous()
        for layer in self.layers2:
            x = layer(x)
        x = self.last_conv(x)
        return x


#### ====================================================================================================================


class ConvDownBlock_LN(nn.Module):
    """ ConvDownBlock: 
        1 Residual Unit,
        1 DownBlock(stride, kernel)  C --> 2 x C
    """
    def __init__(self, out_channels, stride, kernel):
        super().__init__()
        # self.layers = nn.Sequential(
        #     ResidualUnit(in_channels=out_channels//2, out_channels=out_channels//2), # contain 2 convolution + 1 residual
        #     nn.ELU(),
        #     DownBlock(in_channels=out_channels//2, out_channels=out_channels, stride=stride, kernel=kernel, padding=(kernel - stride)//2),      # Downsampling Block
        #     nn.ELU())
        self.residual_unit = ResidualUnit(in_channels=out_channels//2, out_channels=out_channels//2)  # contain 2 convolution + 1 residual 
        self.layer_norm1 = nn.LayerNorm(out_channels//2)
        self.downblock =  DownBlock(in_channels=out_channels//2, out_channels=out_channels, stride=stride, kernel=kernel, padding=(kernel - stride)//2) 
        self.layer_norm2 = nn.LayerNorm(out_channels)
        self.act = nn.ELU()

    def forward(self, x):
        x = self.residual_unit(x)
        x = x.permute(0, 2, 1).contiguous()
        x = self.layer_norm1(x)
        x = x.permute(0, 2, 1).contiguous()
        x = self.act(x)
        x = self.downblock(x)
        x = x.permute(0, 2, 1).contiguous() 
        x = self.layer_norm2(x)
        x = x.permute(0, 2, 1).contiguous()
        x = self.act(x) 
        return x


class ConvUpBlock_LN(nn.Module):
    """ ConvUpBlock: 
        1 Residual Unit,
        1 DownBlock(stride, kernel)  2C --> C
    """
    def __init__(self, out_channels, stride, kernel):
        super().__init__()
        # self.layers = nn.Sequential(
        #     ResidualUnit(in_channels=out_channels*2, out_channels=out_channels*2), # contain 2 convolution + 1 residual
        #     nn.ELU(),
        #     UpBlock(in_channels=out_channels*2, out_channels=out_channels, stride=stride, kernel=kernel, padding=(kernel - stride)//2),      # Downsampling Block
        #     nn.ELU())
        self.residual_unit =  ResidualUnit(in_channels=out_channels*2, out_channels=out_channels*2)  # contain 2 convolution + 1 residual
        self.layer_norm1 = nn.LayerNorm(out_channels*2)
        self.upblock = UpBlock(in_channels=out_channels*2, out_channels=out_channels, stride=stride, kernel=kernel, padding=(kernel - stride)//2) 
        self.layer_norm2 = nn.LayerNorm(out_channels)
        self.act = nn.ELU()

    def forward(self, x):
        x = self.residual_unit(x)
        x = x.permute(0, 2, 1).contiguous()
        x = self.layer_norm1(x)
        x = x.permute(0, 2, 1).contiguous()
        x = self.act(x)
        x = self.upblock(x)
        x = x.permute(0, 2, 1).contiguous()
        x = self.layer_norm2(x)
        x = x.permute(0, 2, 1).contiguous()
        x = self.act(x)
        return x



class Encoder_LN(nn.Module):      
    """ Sound Encoder Use Waveform input"""
    def __init__(self, enc_channels=32, enc_out_channels=256, enc_layer_num=4, enc_stride_list=[2,2,2,4], enc_lstm_layer=2, use_layernorm=False):
        super().__init__()

        # channels = config['model']['AE']['channels']
        # out_channels = config['model']['AE']['out_channels']
        # layer_num = config['model']['AE']['layer_num']
        channels = enc_channels
        out_channels = enc_out_channels
        layer_num = enc_layer_num
        stride_list = enc_stride_list

        self.use_layernorm = use_layernorm

        assert layer_num == len(stride_list)
        self.layers = nn.ModuleList()
        # First Conv1d Layers:
        self.layers.append(nn.Conv1d(in_channels=1, out_channels=channels, kernel_size=1))
        self.layers.append(nn.ELU())

        for i in range(layer_num):  # default: 4
            blocks = ConvDownBlock_LN(out_channels=2**(i+1)*channels, stride=stride_list[i], kernel=2*stride_list[i])
            self.layers.append(blocks)

        self.lstm = nn.Sequential(
                nn.LSTM(input_size=2**(i+1)*channels, hidden_size=2**(i+1)*channels, num_layers=enc_lstm_layer, batch_first=True))
        
        self.last_conv = nn.Sequential(
                nn.ELU(),
                nn.Conv1d(in_channels=2**(i+1)*channels, out_channels=out_channels, kernel_size=1),
                nn.ELU())
        
        self.layer_norm = nn.LayerNorm(2**(i+1)*channels)

    def forward(self, x):
        # x: WaveForm input: B x 1 x L
        for layer in self.layers:
            x = layer(x)        # B x C x L

        x = x.permute(0, 2, 1).contiguous()
        x, _ = self.lstm(x)
        x = self.layer_norm(x)
        x = x.permute(0, 2, 1).contiguous()
        # print(x.shape)
        x = self.last_conv(x)       # KL --> output: mean & var
        return x



class Decoder_LN(nn.Module):
    """ Sound Decoder 
        Symmetric Arch:
    """
    def __init__(self, dec_channels=32, dec_out_channels=128, dec_layer_num=4, dec_stride_list=[4,2,2,2], dec_lstm_layer=2,use_layernorm=False):
        super().__init__()
        channels = dec_channels
        out_channels = dec_out_channels
        layer_num = dec_layer_num
        # stride_list = [2, 2, 2, 4]
        stride_list = dec_stride_list  # reversed
        assert layer_num == len(stride_list)
        # self.layers1 = nn.ModuleList()
        self.layers2 = nn.ModuleList()
        # First Conv1d Layers:  Layers 1
        self.layers1 = nn.Sequential(nn.Conv1d(in_channels=out_channels, out_channels=channels * (2**layer_num), kernel_size=1))
        self.layer_norm1 = nn.LayerNorm(channels * (2**layer_num))
        self.layer_norm2 = nn.LayerNorm(channels * (2**layer_num))
        self.act = nn.ELU()

        # LSTM
        self.lstm = nn.Sequential(
            nn.LSTM(input_size=channels * (2**layer_num), hidden_size=channels * (2**layer_num), num_layers=dec_lstm_layer, batch_first=True))

        # Layers 2:
        self.layers2.append(nn.ELU())
        for i in reversed(range(layer_num)):
            blocks = ConvUpBlock_LN(out_channels=channels * (2**i), stride=stride_list[i], kernel=2*stride_list[i])
            self.layers2.append(blocks)

        
        self.last_conv = nn.Sequential(
            nn.Conv1d(in_channels=channels * (2**i), out_channels=1, kernel_size=1)
        )
        
    
    def forward(self, x):
        x = self.layers1(x)
        x = x.permute(0, 2, 1).contiguous()
        x = self.layer_norm1(x)
        x = self.act(x)
        x, _ = self.lstm(x)
        x = self.layer_norm2(x)
        x = x.permute(0, 2, 1).contiguous()
        for layer in self.layers2:
            x = layer(x)
        x = self.last_conv(x)
        return x




if __name__ == "__main__":
    device = torch.device("cuda:0")
    encoder = Encoder(None).to(device)
    decoder = Decoder(None).to(device)
    # x = torch.randn(2, 128, 5000).to(device)
    x = torch.randn(2, 1, 160000).to(device)   # audio waveform input
    enc = encoder(x)
    # gaussian_kl = DiagonalGaussianDistribution(enc)
    z = gaussian_kl.sample()
    print(z.shape)
    rec = decoder(z)
    print(rec.shape)
    


        

