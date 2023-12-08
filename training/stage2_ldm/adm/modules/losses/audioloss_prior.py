import torch
import torch.nn as nn
import torch.nn.functional as F
from .spec_loss import MelSpectrogramLoss, MelSpectrogramLoss_Linear_Log, MelSpectrogramLoss_Linear_Mel_Spec, MelSpectrogram_transform, MelSpectrogram_transform_loss
from .feature_match_loss import FeatureMatchLoss
from .discriminator_loss import GeneratorAdversarialLoss, DiscriminatorAdversarialLoss

from adm.modules.discriminator.model import STFTDiscriminator_wrapper

from .lpaps import LPAPS

class AudioLoss(nn.Module):
    def __init__(self, loss_type='hinge', disc_num=5, disc_last_act=True, time_weight=1, freq_weight=1, feat_weight=1, g_weight=1, kl_weight=1, 
                        disc_start=50001, mel_spec_win_list=range(5,12), stft_win_list=range(9,12), 
                        d_weight=1, fft_size=2048, fs=22050, num_mels=80, fmax=7600, fmin=80, normalized=True):
        super().__init__()

        # Multi Discriminator:
        self.discriminator_wrapper = STFTDiscriminator_wrapper(disc_num=disc_num, disc_last_act=disc_last_act)

        # Feat Match Loss:
        self.feat_match_loss = FeatureMatchLoss()
        # Adversarial Loss:
        self.generator_adv_loss = GeneratorAdversarialLoss(loss_type=loss_type)     # hinge
        self.discriminator_adv_loss = DiscriminatorAdversarialLoss(loss_type=loss_type) # hinge

        # disc_start:
        self.disc_start = disc_start

        # Spec param:
        self.mel_spec_win_list = mel_spec_win_list
        self.stft_win_list = stft_win_list
        self.fft_size = fft_size
        self.fs = fs
        self.num_mels = num_mels
        self.fmax = fmax
        self.fmin = fmin
        self.normalized = normalized

        # Loss Weight:
        self.time_loss_weight = time_weight
        self.freq_loss_weight = freq_weight
        self.feat_match_weight = feat_weight
        self.g_weight = g_weight
        self.d_weight = d_weight
        self.kl_weight = kl_weight


    def get_melspec_loss(self, inputs_wav, rec_wav):
        """ Spec L1 & L2 Loss with multi windows size """
        l1_loss = 0
        l2_loss = 0
        for i in self.mel_spec_win_list:
            # win_length: [32, 64, 128, ..., 2048]
            # hop_length: [8, 16, 32, 64, ...., 512]
            spec_loss = MelSpectrogramLoss(fs=self.fs, fft_size=self.fft_size, hop_size=2 ** (i-2), win_length=2 ** i, num_mels=self.num_mels, normalized=self.normalized, fmax=self.fmax, fmin=self.fmin)
            mel_loss_l1, mel_loss_l2 = spec_loss(inputs_wav, rec_wav)
            # cal loss:
            l1_loss += mel_loss_l1
            l2_loss += mel_loss_l2
        l1_loss = l1_loss / len(self.mel_spec_win_list)
        l2_loss = l2_loss / len(self.mel_spec_win_list)
        melspec_loss = l1_loss + l2_loss
        return melspec_loss
    

    def get_stft_list(self, inputs_wav, rec_wav):
        """ Get STFT List"""
        inputs_stft_list = []
        rec_stft_list = []
        # win_length: 128, 256, 512, 1024, 2048
        for i in self.stft_win_list:
            inputs_stft = torch.stft(inputs_wav.squeeze(1), n_fft=2048, hop_length=2 ** (i-2), win_length=2 ** i, normalized=True, return_complex=False).permute(0, 3, 1, 2).contiguous()
            rec_stft =  torch.stft(rec_wav.squeeze(1), n_fft=2048, hop_length=2 ** (i-2), win_length=2 ** i, normalized=True, return_complex=False).permute(0, 3, 1, 2).contiguous()
            inputs_stft_list.append(inputs_stft)
            rec_stft_list.append(rec_stft)
        return inputs_stft_list, rec_stft_list

    def get_disc_output(self, inputs_stft_list, rec_stft_list):
        # Get Discriminator Output:
        inputs_disc_output = self.discriminator_wrapper(inputs_stft_list)   # [[layer1, layer2, ...,], ...]
        rec_disc_output = self.discriminator_wrapper(rec_stft_list)         # [[layer1, layer2, ....], ...]
        return inputs_disc_output, rec_disc_output


    def forward(self, inputs_wav, rec_wav, posteriors, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train", predicted_indices=None):

        # inputs_wav: input waveform
        # rec_wav: output waveform
        time_domain_loss = (inputs_wav - rec_wav).abs().mean()  # Lt
        frequency_domain_loss = self.get_melspec_loss(inputs_wav, rec_wav)  # Lf
         
        # GET STFT List:
        inputs_stft_list, rec_stft_list = self.get_stft_list(inputs_wav, rec_wav)   # Multi scale stft
        # GET Disc Output:
        inputs_disc_output, rec_disc_output = self.get_disc_output(inputs_stft_list, rec_stft_list)
        
        # Feat Match Loss:
        feat_match_loss = self.feat_match_loss(rec_disc_output, inputs_disc_output)

        # KL Loss:
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        if optimizer_idx == 0:
            # generator update:
            g_loss = self.generator_adv_loss(rec_disc_output)

            # Prevent Discriminator collpase
            if global_step < self.disc_start:
                disc_factor = 0
            else:
                disc_factor = 1

            loss = self.time_loss_weight * time_domain_loss + self.freq_loss_weight * frequency_domain_loss + self.feat_match_weight * feat_match_loss + disc_factor * self.g_weight * g_loss + self.kl_weight * kl_loss
            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/g_loss".format(split): g_loss.detach().mean() * self.g_weight,
                   "{}/time_domain_loss".format(split): time_domain_loss.detach().mean() * self.time_loss_weight,
                   "{}/freq_domain_loss".format(split): frequency_domain_loss.detach().mean() * self.freq_loss_weight,
                   "{}/feat_match_loss".format(split): feat_match_loss.detach().mean() * self.feat_match_weight,
                   "{}/kl_loss".format(split): kl_loss.detach().mean() * self.kl_weight}
            return loss, log

        if optimizer_idx == 1:
            real_loss, fake_loss = self.discriminator_adv_loss(rec_disc_output, inputs_disc_output)
            if global_step < self.disc_start:
                disc_factor = 0
            else:
                disc_factor = 1
            d_loss = disc_factor * self.d_weight * 0.5 * (real_loss + fake_loss)
            log = {"{}/d_loss".format(split): d_loss.clone().detach().mean()
                   }
            return d_loss, log






class AudioLoss_Linear_Log(nn.Module):
    def __init__(self, loss_type='hinge', disc_num=5, time_weight=1, freq_weight=1, feat_weight=1, g_weight=1, kl_weight=1, disc_start=50001, mel_spec_win_list=range(5,12), stft_win_list=range(9,12), d_weight=1, fft_size=2048, fs=22050, num_mels=80, fmax=7600, fmin=80):
        super().__init__()

        # Multi Discriminator:
        self.discriminator_wrapper = STFTDiscriminator_wrapper(disc_num=disc_num)

        # Feat Match Loss:
        self.feat_match_loss = FeatureMatchLoss()
        # Adversarial Loss:
        self.generator_adv_loss = GeneratorAdversarialLoss(loss_type=loss_type)     # hinge
        self.discriminator_adv_loss = DiscriminatorAdversarialLoss(loss_type=loss_type) # hinge

        # disc_start:
        self.disc_start = disc_start

        # Spec param:
        self.mel_spec_win_list = mel_spec_win_list
        self.stft_win_list = stft_win_list
        self.fft_size = fft_size
        self.fs = fs
        self.num_mels = num_mels
        self.fmax = fmax
        self.fmin = fmin

        # Loss Weight:
        self.time_loss_weight = time_weight
        self.freq_loss_weight = freq_weight
        self.feat_match_weight = feat_weight
        self.g_weight = g_weight
        self.d_weight = d_weight
        self.kl_weight = kl_weight
        print('Using AudioLoss Linear Log ======> ')


    def get_melspec_loss(self, inputs_wav, rec_wav):
        """ Spec L1 & L2 Loss with multi windows size """
        log_l1_loss = 0
        log_l2_loss = 0
        linear_l1_loss = 0
        linear_l2_loss = 0
        for i in self.mel_spec_win_list:
            # win_length: [32, 64, 128, ..., 2048]
            # hop_length: [8, 16, 32, 64, ...., 512]
            spec_loss = MelSpectrogramLoss_Linear_Log(fs=self.fs, fft_size=self.fft_size, hop_size=2 ** (i-2), win_length=2 ** i, num_mels=self.num_mels, normalized=True, fmax=self.fmax, fmin=self.fmin)
            log_mel_loss_l1, log_mel_loss_l2, linear_mel_loss_l1, linear_mel_loss_l2  = spec_loss(inputs_wav, rec_wav)
            # cal loss:
            log_l1_loss += log_mel_loss_l1
            log_l2_loss += log_mel_loss_l2
            linear_l1_loss += linear_mel_loss_l1
            linear_l2_loss += linear_mel_loss_l2

        log_l1_loss = log_l1_loss / len(self.mel_spec_win_list)
        log_l2_loss = log_l2_loss / len(self.mel_spec_win_list)
        linear_l1_loss = linear_l1_loss / len(self.mel_spec_win_list)
        linear_l2_loss = linear_l2_loss / len(self.mel_spec_win_list)
        log_melspec_loss = (log_l1_loss + log_l2_loss + linear_l1_loss + linear_l2_loss) / 2
        return log_melspec_loss
    

    def get_stft_list(self, inputs_wav, rec_wav):
        """ Get STFT List"""
        inputs_stft_list = []
        rec_stft_list = []
        # win_length: 128, 256, 512, 1024, 2048
        for i in self.stft_win_list:
            inputs_stft = torch.stft(inputs_wav.squeeze(1), n_fft=2048, hop_length=2 ** (i-2), win_length=2 ** i, normalized=True, return_complex=False).permute(0, 3, 1, 2).contiguous()
            rec_stft =  torch.stft(rec_wav.squeeze(1), n_fft=2048, hop_length=2 ** (i-2), win_length=2 ** i, normalized=True, return_complex=False).permute(0, 3, 1, 2).contiguous()
            inputs_stft_list.append(inputs_stft)
            rec_stft_list.append(rec_stft)
        return inputs_stft_list, rec_stft_list

    def get_disc_output(self, inputs_stft_list, rec_stft_list):
        # Get Discriminator Output:
        inputs_disc_output = self.discriminator_wrapper(inputs_stft_list)   # [[layer1, layer2, ...,], ...]
        rec_disc_output = self.discriminator_wrapper(rec_stft_list)         # [[layer1, layer2, ....], ...]
        return inputs_disc_output, rec_disc_output


    def forward(self, inputs_wav, rec_wav, posteriors, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train", predicted_indices=None):

        # inputs_wav: input waveform
        # rec_wav: output waveform
        time_domain_loss = (inputs_wav - rec_wav).abs().mean()
        frequency_domain_loss = self.get_melspec_loss(inputs_wav, rec_wav)
         
        # GET STFT List:
        inputs_stft_list, rec_stft_list = self.get_stft_list(inputs_wav, rec_wav)
        # GET Disc Output:
        inputs_disc_output, rec_disc_output = self.get_disc_output(inputs_stft_list, rec_stft_list)
        
        # Feat Match Loss:
        feat_match_loss = self.feat_match_loss(rec_disc_output, inputs_disc_output)

        # KL Loss:
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        if optimizer_idx == 0:
            # generator update:
            g_loss = self.generator_adv_loss(rec_disc_output)

            # Prevent Discriminator collpase
            if global_step < self.disc_start:
                disc_factor = 0
            else:
                disc_factor = 1

            loss = self.time_loss_weight * time_domain_loss + self.freq_loss_weight * frequency_domain_loss + self.feat_match_weight * feat_match_loss + disc_factor * self.g_weight * g_loss + self.kl_weight * kl_loss
            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/g_loss".format(split): g_loss.detach().mean() * self.g_weight,
                   "{}/time_domain_loss".format(split): time_domain_loss.detach().mean() * self.time_loss_weight,
                   "{}/freq_domain_loss".format(split): frequency_domain_loss.detach().mean() * self.freq_loss_weight,
                   "{}/feat_match_loss".format(split): feat_match_loss.detach().mean() * self.feat_match_weight,
                   "{}/kl_loss".format(split): kl_loss.detach().mean() * self.kl_weight}
            return loss, log

        if optimizer_idx == 1:
            real_loss, fake_loss = self.discriminator_adv_loss(rec_disc_output, inputs_disc_output)
            if global_step < self.disc_start:
                disc_factor = 0
            else:
                disc_factor = 1
            d_loss = disc_factor * 0.5 * (real_loss + fake_loss)
            log = {"{}/d_loss".format(split): d_loss.clone().detach().mean()
                   }
            return d_loss, log







class AudioLoss_Linear_Log_Spec(nn.Module):
    def __init__(self, loss_type='hinge', disc_num=5, time_weight=1, freq_weight=1, feat_weight=1, g_weight=1, kl_weight=1, disc_start=50001, mel_spec_win_list=range(5,12), stft_win_list=range(9,12), d_weight=1, fft_size=2048, fs=22050, num_mels=80, fmax=7600, fmin=80):
        super().__init__()

        # Multi Discriminator:
        self.discriminator_wrapper = STFTDiscriminator_wrapper(disc_num=disc_num)

        # Feat Match Loss:
        self.feat_match_loss = FeatureMatchLoss()
        # Adversarial Loss:
        self.generator_adv_loss = GeneratorAdversarialLoss(loss_type=loss_type)     # hinge
        self.discriminator_adv_loss = DiscriminatorAdversarialLoss(loss_type=loss_type) # hinge

        # disc_start:
        self.disc_start = disc_start

        # Spec param:
        self.mel_spec_win_list = mel_spec_win_list
        self.stft_win_list = stft_win_list
        self.fft_size = fft_size
        self.fs = fs
        self.num_mels = num_mels
        self.fmax = fmax
        self.fmin = fmin

        # Loss Weight:
        self.time_loss_weight = time_weight
        self.freq_loss_weight = freq_weight
        self.feat_match_weight = feat_weight
        self.g_weight = g_weight
        self.d_weight = d_weight
        self.kl_weight = kl_weight
        print('Using AudioLoss Linear Log ======> ')


    def get_melspec_loss(self, inputs_wav, rec_wav):
        """ Spec L1 & L2 Loss with multi windows size """
        log_l1_loss = 0
        log_l2_loss = 0
        linear_l1_loss = 0
        linear_l2_loss = 0
        for i in self.mel_spec_win_list:
            # win_length: [32, 64, 128, ..., 2048]
            # hop_length: [8, 16, 32, 64, ...., 512]
            spec_loss = MelSpectrogramLoss_Linear_Log(fs=self.fs, fft_size=self.fft_size, hop_size=2 ** (i-2), win_length=2 ** i, num_mels=self.num_mels, normalized=True, fmax=self.fmax, fmin=self.fmin)
            log_mel_loss_l1, log_mel_loss_l2, linear_mel_loss_l1, linear_mel_loss_l2  = spec_loss(inputs_wav, rec_wav)
            # cal loss:
            log_l1_loss += log_mel_loss_l1
            log_l2_loss += log_mel_loss_l2
            linear_l1_loss += linear_mel_loss_l1
            linear_l2_loss += linear_mel_loss_l2

        log_l1_loss = log_l1_loss / len(self.mel_spec_win_list)
        log_l2_loss = log_l2_loss / len(self.mel_spec_win_list)
        linear_l1_loss = linear_l1_loss / len(self.mel_spec_win_list)
        linear_l2_loss = linear_l2_loss / len(self.mel_spec_win_list)
        log_melspec_loss = (log_l1_loss + log_l2_loss + linear_l1_loss + linear_l2_loss) / 2
        return log_melspec_loss
    

    def get_stft_list(self, inputs_wav, rec_wav):
        """ Get STFT List"""
        inputs_stft_list = []
        rec_stft_list = []
        # win_length: 128, 256, 512, 1024, 2048
        for i in self.stft_win_list:
            inputs_stft = torch.stft(inputs_wav.squeeze(1), n_fft=2048, hop_length=2 ** (i-2), win_length=2 ** i, normalized=True, return_complex=False).permute(0, 3, 1, 2).contiguous()
            rec_stft =  torch.stft(rec_wav.squeeze(1), n_fft=2048, hop_length=2 ** (i-2), win_length=2 ** i, normalized=True, return_complex=False).permute(0, 3, 1, 2).contiguous()
            inputs_stft_list.append(inputs_stft)
            rec_stft_list.append(rec_stft)
        return inputs_stft_list, rec_stft_list

    def get_disc_output(self, inputs_stft_list, rec_stft_list):
        # Get Discriminator Output:
        inputs_disc_output = self.discriminator_wrapper(inputs_stft_list)   # [[layer1, layer2, ...,], ...]
        rec_disc_output = self.discriminator_wrapper(rec_stft_list)         # [[layer1, layer2, ....], ...]
        return inputs_disc_output, rec_disc_output


    def forward(self, inputs_wav, rec_wav, posteriors, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train", predicted_indices=None):

        # inputs_wav: input waveform
        # rec_wav: output waveform
        time_domain_loss = (inputs_wav - rec_wav).abs().mean()
        frequency_domain_loss = self.get_melspec_loss(inputs_wav, rec_wav)
         
        # GET STFT List:
        inputs_stft_list, rec_stft_list = self.get_stft_list(inputs_wav, rec_wav)
        # GET Disc Output:
        inputs_disc_output, rec_disc_output = self.get_disc_output(inputs_stft_list, rec_stft_list)
        
        # Feat Match Loss:
        feat_match_loss = self.feat_match_loss(rec_disc_output, inputs_disc_output)

        # KL Loss:
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        if optimizer_idx == 0:
            # generator update:
            g_loss = self.generator_adv_loss(rec_disc_output)

            # Prevent Discriminator collpase
            if global_step < self.disc_start:
                disc_factor = 0
            else:
                disc_factor = 1

            loss = self.time_loss_weight * time_domain_loss + self.freq_loss_weight * frequency_domain_loss + self.feat_match_weight * feat_match_loss + disc_factor * self.g_weight * g_loss + self.kl_weight * kl_loss
            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/g_loss".format(split): g_loss.detach().mean() * self.g_weight,
                   "{}/time_domain_loss".format(split): time_domain_loss.detach().mean() * self.time_loss_weight,
                   "{}/freq_domain_loss".format(split): frequency_domain_loss.detach().mean() * self.freq_loss_weight,
                   "{}/feat_match_loss".format(split): feat_match_loss.detach().mean() * self.feat_match_weight,
                   "{}/kl_loss".format(split): kl_loss.detach().mean() * self.kl_weight}
            return loss, log

        if optimizer_idx == 1:
            real_loss, fake_loss = self.discriminator_adv_loss(rec_disc_output, inputs_disc_output)
            if global_step < self.disc_start:
                disc_factor = 0
            else:
                disc_factor = 1
            d_loss = disc_factor * 0.5 * (real_loss + fake_loss)
            log = {"{}/d_loss".format(split): d_loss.clone().detach().mean()
                   }
            return d_loss, log








class AudioLoss_wo_KL(nn.Module):
    def __init__(self, loss_type='hinge', disc_num=5, disc_last_act=True, time_weight=1, freq_weight=1, feat_weight=1, g_weight=1, kl_weight=1, disc_start=50001, mel_spec_win_list=range(5,12), stft_win_list=range(9,12), d_weight=1, fft_size=2048, fs=22050, num_mels=80, fmax=7600, fmin=80, normalized=True):
        super().__init__()

        # Multi Discriminator:
        self.discriminator_wrapper = STFTDiscriminator_wrapper(disc_num=disc_num, disc_last_act=disc_last_act)

        # Feat Match Loss:
        self.feat_match_loss = FeatureMatchLoss()
        # Adversarial Loss:
        self.generator_adv_loss = GeneratorAdversarialLoss(loss_type=loss_type)     # hinge
        self.discriminator_adv_loss = DiscriminatorAdversarialLoss(loss_type=loss_type) # hinge

        # disc_start:
        self.disc_start = disc_start

        # Spec param:
        self.mel_spec_win_list = mel_spec_win_list
        self.stft_win_list = stft_win_list
        self.fft_size = fft_size
        self.fs = fs
        self.num_mels = num_mels
        self.fmax = fmax
        self.fmin = fmin
        self.normalized = normalized

        # Loss Weight:
        self.time_loss_weight = time_weight
        self.freq_loss_weight = freq_weight
        self.feat_match_weight = feat_weight
        self.g_weight = g_weight
        self.d_weight = d_weight
        self.kl_weight = kl_weight


    def get_melspec_loss(self, inputs_wav, rec_wav):
        """ Spec L1 & L2 Loss with multi windows size """
        l1_loss = 0
        l2_loss = 0
        for i in self.mel_spec_win_list:
            # win_length: [32, 64, 128, ..., 2048]
            # hop_length: [8, 16, 32, 64, ...., 512]
            spec_loss = MelSpectrogramLoss(fs=self.fs, fft_size=self.fft_size, hop_size=2 ** (i-2), win_length=2 ** i, num_mels=self.num_mels, normalized=self.normalized, fmax=self.fmax, fmin=self.fmin)
            mel_loss_l1, mel_loss_l2 = spec_loss(inputs_wav, rec_wav)
            # cal loss:
            l1_loss += mel_loss_l1
            l2_loss += mel_loss_l2
        l1_loss = l1_loss / len(self.mel_spec_win_list)
        l2_loss = l2_loss / len(self.mel_spec_win_list)
        melspec_loss = l1_loss + l2_loss
        return melspec_loss
    

    def get_stft_list(self, inputs_wav, rec_wav):
        """ Get STFT List"""
        inputs_stft_list = []
        rec_stft_list = []
        # win_length: 128, 256, 512, 1024, 2048
        for i in self.stft_win_list:
            inputs_stft = torch.stft(inputs_wav.squeeze(1), n_fft=2048, hop_length=2 ** (i-2), win_length=2 ** i, normalized=True, return_complex=False).permute(0, 3, 1, 2).contiguous()
            rec_stft =  torch.stft(rec_wav.squeeze(1), n_fft=2048, hop_length=2 ** (i-2), win_length=2 ** i, normalized=True, return_complex=False).permute(0, 3, 1, 2).contiguous()
            inputs_stft_list.append(inputs_stft)
            rec_stft_list.append(rec_stft)
        return inputs_stft_list, rec_stft_list

    def get_disc_output(self, inputs_stft_list, rec_stft_list):
        # Get Discriminator Output:
        inputs_disc_output = self.discriminator_wrapper(inputs_stft_list)   # [[layer1, layer2, ...,], ...]
        rec_disc_output = self.discriminator_wrapper(rec_stft_list)         # [[layer1, layer2, ....], ...]
        return inputs_disc_output, rec_disc_output


    def forward(self, inputs_wav, rec_wav, posteriors, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train", predicted_indices=None):

        # inputs_wav: input waveform
        # rec_wav: output waveform
        time_domain_loss = (inputs_wav - rec_wav).abs().mean()  # Lt
        frequency_domain_loss = self.get_melspec_loss(inputs_wav, rec_wav)  # Lf
         
        # GET STFT List:
        inputs_stft_list, rec_stft_list = self.get_stft_list(inputs_wav, rec_wav)   # Multi scale stft
        # GET Disc Output:
        inputs_disc_output, rec_disc_output = self.get_disc_output(inputs_stft_list, rec_stft_list)
        
        # Feat Match Loss:
        feat_match_loss = self.feat_match_loss(rec_disc_output, inputs_disc_output)

        # # KL Loss:
        # kl_loss = posteriors.kl()
        # kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        if optimizer_idx == 0:
            # generator update:
            g_loss = self.generator_adv_loss(rec_disc_output)

            # Prevent Discriminator collpase
            if global_step < self.disc_start:
                disc_factor = 0
            else:
                disc_factor = 1

            loss = self.time_loss_weight * time_domain_loss + self.freq_loss_weight * frequency_domain_loss + self.feat_match_weight * feat_match_loss + disc_factor * self.g_weight * g_loss 
            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/g_loss".format(split): g_loss.detach().mean() * self.g_weight,
                   "{}/time_domain_loss".format(split): time_domain_loss.detach().mean() * self.time_loss_weight,
                   "{}/freq_domain_loss".format(split): frequency_domain_loss.detach().mean() * self.freq_loss_weight,
                   "{}/feat_match_loss".format(split): feat_match_loss.detach().mean() * self.feat_match_weight,
                    }
                #    "{}/kl_loss".format(split): kl_loss.detach().mean() * self.kl_weight}
            return loss, log

        if optimizer_idx == 1:
            real_loss, fake_loss = self.discriminator_adv_loss(rec_disc_output, inputs_disc_output)
            if global_step < self.disc_start:
                disc_factor = 0
            else:
                disc_factor = 1
            d_loss = disc_factor * self.d_weight * 0.5 * (real_loss + fake_loss)
            log = {"{}/d_loss".format(split): d_loss.clone().detach().mean()
                   }
            return d_loss, log
        





class AudioLoss_Pretrained_Feat(nn.Module):
    def __init__(self, loss_type='hinge', disc_num=5, disc_last_act=True, time_weight=1, freq_weight=1, feat_weight=1, g_weight=1, kl_weight=1, 
                        disc_start=50001, mel_spec_win_list=range(5,12), stft_win_list=range(9,12), 
                        d_weight=1, fft_size=2048, fs=22050, num_mels=80, fmax=7600, fmin=80, normalized=True):
        super().__init__()

        # Multi Discriminator:
        self.discriminator_wrapper = STFTDiscriminator_wrapper(disc_num=disc_num, disc_last_act=disc_last_act)

        # Feat Match Loss:
        self.feat_match_loss = LPAPS().eval()
        # Adversarial Loss:
        self.generator_adv_loss = GeneratorAdversarialLoss(loss_type=loss_type)     # hinge
        self.discriminator_adv_loss = DiscriminatorAdversarialLoss(loss_type=loss_type) # hinge

        # disc_start:
        self.disc_start = disc_start

        # Spec param:
        self.mel_spec_win_list = mel_spec_win_list
        self.stft_win_list = stft_win_list
        self.fft_size = fft_size
        self.fs = fs
        self.num_mels = num_mels
        self.fmax = fmax
        self.fmin = fmin
        self.normalized = normalized

        # Loss Weight:
        self.time_loss_weight = time_weight
        self.freq_loss_weight = freq_weight
        self.feat_match_weight = feat_weight
        self.g_weight = g_weight
        self.d_weight = d_weight
        self.kl_weight = kl_weight


    def get_melspec_loss(self, inputs_wav, rec_wav):
        """ Spec L1 & L2 Loss with multi windows size """
        l1_loss = 0
        l2_loss = 0
        for i in self.mel_spec_win_list:
            # win_length: [32, 64, 128, ..., 2048]
            # hop_length: [8, 16, 32, 64, ...., 512]
            spec_loss = MelSpectrogramLoss(fs=self.fs, fft_size=self.fft_size, hop_size=2 ** (i-2), win_length=2 ** i, num_mels=self.num_mels, normalized=self.normalized, fmax=self.fmax, fmin=self.fmin)
            mel_loss_l1, mel_loss_l2 = spec_loss(inputs_wav, rec_wav)
            # cal loss:
            l1_loss += mel_loss_l1
            l2_loss += mel_loss_l2
        l1_loss = l1_loss / len(self.mel_spec_win_list)
        l2_loss = l2_loss / len(self.mel_spec_win_list)
        melspec_loss = l1_loss + l2_loss
        return melspec_loss
    
    def get_featmatch_loss(self, inputs_wav, rec_wav):
        """ Feat Match Loss """
        # First Get Spec:
        spec_transform = MelSpectrogram_transform(fs=self.fs, fft_size=1024, hop_size=1024 // 4, num_mels=80, normalized=False, fmax=self.fmax, fmin=self.fmin)
        inputs_spec = spec_transform(inputs_wav)
        rec_sepc = spec_transform(rec_wav)
        featmatch_loss = self.feat_match_loss(inputs_spec, rec_sepc)
        return featmatch_loss.mean()


    def get_stft_list(self, inputs_wav, rec_wav):
        """ Get STFT List"""
        inputs_stft_list = []
        rec_stft_list = []
        # win_length: 128, 256, 512, 1024, 2048
        for i in self.stft_win_list:
            inputs_stft = torch.stft(inputs_wav.squeeze(1), n_fft=2048, hop_length=2 ** (i-2), win_length=2 ** i, normalized=True, return_complex=False).permute(0, 3, 1, 2).contiguous()
            rec_stft =  torch.stft(rec_wav.squeeze(1), n_fft=2048, hop_length=2 ** (i-2), win_length=2 ** i, normalized=True, return_complex=False).permute(0, 3, 1, 2).contiguous()
            inputs_stft_list.append(inputs_stft)
            rec_stft_list.append(rec_stft)
        return inputs_stft_list, rec_stft_list

    def get_disc_output(self, inputs_stft_list, rec_stft_list):
        # Get Discriminator Output:
        inputs_disc_output = self.discriminator_wrapper(inputs_stft_list)   # [[layer1, layer2, ...,], ...]
        rec_disc_output = self.discriminator_wrapper(rec_stft_list)         # [[layer1, layer2, ....], ...]
        return inputs_disc_output, rec_disc_output


    def forward(self, inputs_wav, rec_wav, posteriors, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train", predicted_indices=None):

        # inputs_wav: input waveform
        # rec_wav: output waveform
        time_domain_loss = (inputs_wav - rec_wav).abs().mean()  # Lt
        frequency_domain_loss = self.get_melspec_loss(inputs_wav, rec_wav)  # Lf
         
        # GET STFT List:
        inputs_stft_list, rec_stft_list = self.get_stft_list(inputs_wav, rec_wav)   # Multi scale stft
        # GET Disc Output:
        inputs_disc_output, rec_disc_output = self.get_disc_output(inputs_stft_list, rec_stft_list)
        
        # Feat Match Loss:
        # feat_match_loss = self.feat_match_loss(rec_disc_output, inputs_disc_output)
        feat_match_loss = self.get_featmatch_loss(inputs_wav, rec_wav)

        # KL Loss:
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        if optimizer_idx == 0:
            # generator update:
            g_loss = self.generator_adv_loss(rec_disc_output)

            # Prevent Discriminator collpase
            if global_step < self.disc_start:
                disc_factor = 0
            else:
                disc_factor = 1

            loss = self.time_loss_weight * time_domain_loss + self.freq_loss_weight * frequency_domain_loss + self.feat_match_weight * feat_match_loss + disc_factor * self.g_weight * g_loss + self.kl_weight * kl_loss
            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/g_loss".format(split): g_loss.detach().mean() * self.g_weight,
                   "{}/time_domain_loss".format(split): time_domain_loss.detach().mean() * self.time_loss_weight,
                   "{}/freq_domain_loss".format(split): frequency_domain_loss.detach().mean() * self.freq_loss_weight,
                   "{}/feat_match_loss".format(split): feat_match_loss.detach().mean() * self.feat_match_weight,
                   "{}/kl_loss".format(split): kl_loss.detach().mean() * self.kl_weight}
            return loss, log

        if optimizer_idx == 1:
            real_loss, fake_loss = self.discriminator_adv_loss(rec_disc_output, inputs_disc_output)
            if global_step < self.disc_start:
                disc_factor = 0
            else:
                disc_factor = 1
            d_loss = disc_factor * self.d_weight * 0.5 * (real_loss + fake_loss)
            log = {"{}/d_loss".format(split): d_loss.clone().detach().mean()
                   }
            return d_loss, log





class AudioLoss_spec_revise(nn.Module):
    def __init__(self, loss_type='hinge', disc_num=5, disc_last_act=True, time_weight=1, freq_weight=1, feat_weight=1, g_weight=1, kl_weight=1, 
                        disc_start=50001, mel_spec_win_list=range(5,12), stft_win_list=range(9,12), 
                        d_weight=1, fft_size=2048, fs=22050, num_mels=80, fmax=7600, fmin=80, normalized=True):
        super().__init__()

        # Multi Discriminator:
        self.discriminator_wrapper = STFTDiscriminator_wrapper(disc_num=disc_num, disc_last_act=disc_last_act)

        # Feat Match Loss:
        self.feat_match_loss = FeatureMatchLoss()
        # Adversarial Loss:
        self.generator_adv_loss = GeneratorAdversarialLoss(loss_type=loss_type)     # hinge
        self.discriminator_adv_loss = DiscriminatorAdversarialLoss(loss_type=loss_type) # hinge

        # disc_start:
        self.disc_start = disc_start

        # Spec param:
        self.mel_spec_win_list = mel_spec_win_list
        self.stft_win_list = stft_win_list
        self.fft_size = fft_size
        self.fs = fs
        self.num_mels = num_mels
        self.fmax = fmax
        self.fmin = fmin
        self.normalized = normalized

        # Loss Weight:
        self.time_loss_weight = time_weight
        self.freq_loss_weight = freq_weight
        self.feat_match_weight = feat_weight
        self.g_weight = g_weight
        self.d_weight = d_weight
        self.kl_weight = kl_weight


    def get_melspec_loss(self, inputs_wav, rec_wav):
        """ Spec L1 & L2 Loss with multi windows size """
        l1_loss = 0
        l2_loss = 0
        for i in self.mel_spec_win_list:
            # win_length: [32, 64, 128, ..., 2048]
            # hop_length: [8, 16, 32, 64, ...., 512]
            spec_loss = MelSpectrogram_transform_loss(fs=self.fs, fft_size=self.fft_size, hop_size=2 ** (i-2), win_length=2 ** i, num_mels=self.num_mels, normalized=self.normalized, fmax=self.fmax, fmin=self.fmin)
            mel_loss_l1, mel_loss_l2 = spec_loss(inputs_wav, rec_wav)
            # cal loss:
            l1_loss += mel_loss_l1
            l2_loss += mel_loss_l2
        l1_loss = l1_loss / len(self.mel_spec_win_list)
        l2_loss = l2_loss / len(self.mel_spec_win_list)
        melspec_loss = l1_loss + l2_loss
        return melspec_loss
    

    def get_stft_list(self, inputs_wav, rec_wav):
        """ Get STFT List"""
        inputs_stft_list = []
        rec_stft_list = []
        # win_length: 128, 256, 512, 1024, 2048
        for i in self.stft_win_list:
            inputs_stft = torch.stft(inputs_wav.squeeze(1), n_fft=2048, hop_length=2 ** (i-2), win_length=2 ** i, normalized=True, return_complex=False).permute(0, 3, 1, 2).contiguous()
            rec_stft =  torch.stft(rec_wav.squeeze(1), n_fft=2048, hop_length=2 ** (i-2), win_length=2 ** i, normalized=True, return_complex=False).permute(0, 3, 1, 2).contiguous()
            inputs_stft_list.append(inputs_stft)
            rec_stft_list.append(rec_stft)
        return inputs_stft_list, rec_stft_list

    def get_disc_output(self, inputs_stft_list, rec_stft_list):
        # Get Discriminator Output:
        inputs_disc_output = self.discriminator_wrapper(inputs_stft_list)   # [[layer1, layer2, ...,], ...]
        rec_disc_output = self.discriminator_wrapper(rec_stft_list)         # [[layer1, layer2, ....], ...]
        return inputs_disc_output, rec_disc_output


    def forward(self, inputs_wav, rec_wav, posteriors, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train", predicted_indices=None):

        # inputs_wav: input waveform
        # rec_wav: output waveform
        time_domain_loss = (inputs_wav - rec_wav).abs().mean()  # Lt
        frequency_domain_loss = self.get_melspec_loss(inputs_wav, rec_wav)  # Lf
         
        # GET STFT List:
        inputs_stft_list, rec_stft_list = self.get_stft_list(inputs_wav, rec_wav)   # Multi scale stft
        # GET Disc Output:
        inputs_disc_output, rec_disc_output = self.get_disc_output(inputs_stft_list, rec_stft_list)
        
        # Feat Match Loss:
        feat_match_loss = self.feat_match_loss(rec_disc_output, inputs_disc_output)

        # KL Loss:
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        if optimizer_idx == 0:
            # generator update:
            g_loss = self.generator_adv_loss(rec_disc_output)

            # Prevent Discriminator collpase
            if global_step < self.disc_start:
                disc_factor = 0
            else:
                disc_factor = 1

            loss = self.time_loss_weight * time_domain_loss + self.freq_loss_weight * frequency_domain_loss + self.feat_match_weight * feat_match_loss + disc_factor * self.g_weight * g_loss + self.kl_weight * kl_loss
            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/g_loss".format(split): g_loss.detach().mean() * self.g_weight,
                   "{}/time_domain_loss".format(split): time_domain_loss.detach().mean(),
                   "{}/freq_domain_loss".format(split): frequency_domain_loss.detach().mean() * self.freq_loss_weight,
                   "{}/feat_match_loss".format(split): feat_match_loss.detach().mean() * self.feat_match_weight,
                   "{}/kl_loss".format(split): kl_loss.detach().mean() * self.kl_weight}
            return loss, log

        if optimizer_idx == 1:
            real_loss, fake_loss = self.discriminator_adv_loss(rec_disc_output, inputs_disc_output)
            if global_step < self.disc_start:
                disc_factor = 0
            else:
                disc_factor = 1
            d_loss = disc_factor * self.d_weight * 0.5 * (real_loss + fake_loss)
            log = {"{}/d_loss".format(split): d_loss.clone().detach().mean()
                   }
            return d_loss, log