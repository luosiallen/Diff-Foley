
import torch
import torch.nn as nn

import pytorch_lightning as pl

from adm.modules.stage1_model.model import Encoder, Decoder, DiagonalGaussianDistribution, Encoder_LN, Decoder_LN
from adm.util import instantiate_from_config




# Based on Pylightning
class Sound_AutoencoderKL(pl.LightningModule):
    def __init__(self, ddconfig, lossconfig, ckpt_path=None, monitor=None, ignore_keys=[]):
        super().__init__()
        self.encoder = Encoder(**ddconfig.encoder)
        self.decoder = Decoder(**ddconfig.decoder)
        self.loss = instantiate_from_config(lossconfig)

        if monitor is not None:
            self.monitor = monitor

        if ckpt_path is not None:
            print("load ckpt from: ", ckpt_path)
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    
    def encode(self, x):
        h = self.encoder(x)
        # print(h.shape)
        posterior = DiagonalGaussianDistribution(h)
        return posterior
    
    def decode(self, z):
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)  # sound posterior 
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        # print(z.shape)
        dec = self.decode(z)
        return dec, posterior
    

    def training_step(self, batch, batch_idx, optimizer_idx):
        # pl training step:
        inputs = batch['audio']
        reconstruction, posterior = self(inputs)    # waveform

        if optimizer_idx == 0:
            # train encoder + decoder + KL Prior
            aeloss, log_dict_ae = self.loss(inputs, reconstruction, posterior, optimizer_idx, self.global_step, split='train')
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss
        
        if optimizer_idx == 1:
            # train the discriminator:
            discloss, log_dict_disc = self.loss(inputs, reconstruction, posterior, optimizer_idx, self.global_step, split='train')
            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss
    
    def validation_step(self, batch, batch_idx):
        # inputs = self.get_input(batch, self.image_key)
        inputs = batch['audio']
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step, split="val")
        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step, split="val")
        self.log("val/time_domain_loss", log_dict_ae["val/time_domain_loss"])
        self.log("val/freq_domain_loss", log_dict_ae["val/freq_domain_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict
    


    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")


    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator_wrapper.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []
    

    # Sound Logger:
    @torch.no_grad()
    def log_sound(self, batch, max_sound_num, **kwargs):
        log = dict()
        x = batch['audio']
        x = x.to(self.device)
        # rec, posterior
        xrec, posterior = self(x)
        gt = batch['audio']                  # B x 1 x L
        xrec = xrec.detach().cpu().numpy()   # B x 1 x L
        gt_list = []
        rec_list = []

        bs = gt.shape[0]
        for i in range(min(max_sound_num, bs)):
            gt_sound = gt[i].squeeze(0)
            rec_sound = xrec[i].squeeze(0)
            gt_list.append(gt_sound)
            rec_list.append(rec_sound)
        log['gt_sound_list'] = gt_list
        log['rec_sound_list'] = rec_list
        return log



# Based on Pylightning
class Sound_AutoencoderKL_LN(pl.LightningModule):
    def __init__(self, ddconfig, lossconfig, ckpt_path=None, ignore_keys=[]):
        super().__init__()
        self.encoder = Encoder_LN(**ddconfig.encoder)
        self.decoder = Decoder_LN(**ddconfig.decoder)
        self.loss = instantiate_from_config(lossconfig)
        
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    
    def encode(self, x):
        h = self.encoder(x)
        posterior = DiagonalGaussianDistribution(h)
        return posterior
    
    def decode(self, z):
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)  # sound posterior 
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        # print(z.shape)
        dec = self.decode(z)
        return dec, posterior
    

    def training_step(self, batch, batch_idx, optimizer_idx):
        # pl training step:
        inputs = batch['audio']
        reconstruction, posterior = self(inputs)    # waveform


        if optimizer_idx == 0:
            # train encoder + decoder + KL Prior
            aeloss, log_dict_ae = self.loss(inputs, reconstruction, posterior, optimizer_idx, self.global_step, split='train')
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss
        
        if optimizer_idx == 1:
            # train the discriminator:
            discloss, log_dict_disc = self.loss(inputs, reconstruction, posterior, optimizer_idx, self.global_step, split='train')
            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss
    
    def validation_step(self, batch, batch_idx):
        # inputs = self.get_input(batch, self.image_key)
        inputs = batch['audio']
        reconstructions, posterior = self(inputs, sample_posterior=False)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step, split="val")
        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step, split="val")
        self.log("val/time_domain_loss", log_dict_ae["val/time_domain_loss"])
        self.log("val/freq_domain_loss", log_dict_ae["val/freq_domain_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict
    


    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")


    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator_wrapper.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []
    

    # Sound Logger:
    @torch.no_grad()
    def log_sound(self, batch, max_sound_num, **kwargs):
        log = dict()
        x = batch['audio']
        x = x.to(self.device)
        # rec, posterior
        xrec, posterior = self(x, sample_posterior=False)
        gt = batch['audio']                  # B x 1 x L
        xrec = xrec.detach().cpu().numpy()   # B x 1 x L
        gt_list = []
        rec_list = []

        bs = gt.shape[0]
        for i in range(min(max_sound_num, bs)):
            gt_sound = gt[i].squeeze(0)
            rec_sound = xrec[i].squeeze(0)
            gt_list.append(gt_sound)
            rec_list.append(rec_sound)
        log['gt_sound_list'] = gt_list
        log['rec_sound_list'] = rec_list
        return log 




# Based on Pylightning
class Sound_Autoencoder_wo_KL(pl.LightningModule):
    def __init__(self, ddconfig, lossconfig, ckpt_path=None, ignore_keys=[]):
        super().__init__()
        self.encoder = Encoder(**ddconfig.encoder)
        self.decoder = Decoder(**ddconfig.decoder)
        self.loss = instantiate_from_config(lossconfig)
        
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    
    def encode(self, x):
        h = self.encoder(x)
        # posterior = DiagonalGaussianDistribution(h)
        return h
    
    def decode(self, z):
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        z = self.encode(input)  # sound posterior 
        # print(z.shape)
        dec = self.decode(z)
        return dec
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        # pl training step:
        inputs = batch['audio']
        reconstruction = self(inputs)    # waveform


        if optimizer_idx == 0:
            # train encoder + decoder + KL Prior
            aeloss, log_dict_ae = self.loss(inputs, reconstruction, None, optimizer_idx, self.global_step, split='train')
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss
        
        if optimizer_idx == 1:
            # train the discriminator:
            discloss, log_dict_disc = self.loss(inputs, reconstruction, None, optimizer_idx, self.global_step, split='train')
            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss
    
    def validation_step(self, batch, batch_idx):
        # inputs = self.get_input(batch, self.image_key)
        inputs = batch['audio']
        reconstructions = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, None, 0, self.global_step, split="val")
        discloss, log_dict_disc = self.loss(inputs, reconstructions, None, 1, self.global_step, split="val")
        self.log("val/time_domain_loss", log_dict_ae["val/time_domain_loss"])
        self.log("val/freq_domain_loss", log_dict_ae["val/freq_domain_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict
    


    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")


    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator_wrapper.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []
    

    # Sound Logger:
    @torch.no_grad()
    def log_sound(self, batch, max_sound_num, **kwargs):
        log = dict()
        x = batch['audio']
        x = x.to(self.device)
        # rec, posterior
        xrec = self(x)
        gt = batch['audio']                  # B x 1 x L
        xrec = xrec.detach().cpu().numpy()   # B x 1 x L
        gt_list = []
        rec_list = []

        bs = gt.shape[0]
        for i in range(min(max_sound_num, bs)):
            gt_sound = gt[i].squeeze(0)
            rec_sound = xrec[i].squeeze(0)
            gt_list.append(gt_sound)
            rec_list.append(rec_sound)
        log['gt_sound_list'] = gt_list
        log['rec_sound_list'] = rec_list
        return log

