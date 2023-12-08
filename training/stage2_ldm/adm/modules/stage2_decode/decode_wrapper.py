import torch
import torch.nn as nn



import pytorch_lightning as pl


import importlib

from torch.optim.lr_scheduler import LambdaLR


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


class Decoder_Wrapper(pl.LightningModule):

    def __init__(self,
                 first_stage_config,
                 decoder_config,
                 scheduler_config,
                 monitor,
                 first_stage_ckpt=None,
                 first_stage_key="spec",
                 normalize = True,
                 avg = False,
                 pool = False,
                 lossconfig = None,
                 *args, **kwargs):
    
        super().__init__()

        self.instantiate_first_stage(first_stage_config)
        self.first_stage_ckpt = first_stage_ckpt
        if self.first_stage_ckpt is not None:
            self.init_first_from_ckpt(self.first_stage_ckpt)

        self.model = instantiate_from_config(decoder_config)

        self.mse_loss = torch.nn.MSELoss()
        self.first_stage_key = first_stage_key
        self.monitor = monitor
        self.normalize = normalize
        self.avg = avg
        self.pool = pool

        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config
        
        if lossconfig:
            self.loss = instantiate_from_config(lossconfig)


    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False
    
    def init_first_from_ckpt(self, path):
        model = torch.load(path, map_location="cpu")
        if "state_dict" in list(model.keys()):
            model = model["state_dict"]
        # Remove: module prefix
        new_model = {}
        for key in model.keys():
            new_key = key.replace("module.","")
            new_model[new_key] = model[key]
        missing, unexpected = self.first_stage_model.load_state_dict(new_model, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")
    


    def get_input(self, batch, k, bs=None):
        x = batch[k]
        x = x.to(memory_format=torch.contiguous_format).float()
        if bs is not None:
            x = x[:bs]
        x = x.to(self.device)
        return x    # spec

    @torch.no_grad()
    def encode_first_stage_spec(self, x):
        out = self.first_stage_model.encode_spec(x, normalize=self.normalize, avg=self.avg)
        return out


    @torch.no_grad()
    def encode_first_stage_video(self, x):
        out = self.first_stage_model.encode_video(x, normalize=self.normalize, avg=self.avg)
        return out
    

    @torch.no_grad()
    def encode_first_stage_spec_intra(self, x):
        out = self.first_stage_model.encode_spec(x, normalize=self.normalize, pool=self.pool)
        return out


    @torch.no_grad()
    def encode_first_stage_video_intra(self, x):
        out = self.first_stage_model.encode_video(x, normalize=self.normalize, pool=self.pool)
        return out

    def get_x_rec(self, x):
        z = self.first_stage_model.encode_spec(x, normalize=self.normalize, avg=self.avg)
        x_rec = self(z)
        return x_rec
    
    def reconstruct_spec(self, x):
        x_rec = self(x)
        return x_rec

    
    def shared_step(self, batch):
        x = self.get_input(batch, self.first_stage_key)
        # Get Spec Encode:
        z = self.encode_first_stage_spec(x).detach()
        x_rec = self(z)
        loss = self.mse_loss(x_rec, x)  # L2 Loss
        loss_dict = {}
        prefix = "train" if self.training else 'val'
        loss_dict.update({f'{prefix}/l2_loss': loss})
        return loss, loss_dict


    def forward(self, x):
        # Forward spec encode:
        # input x: B x T x C
        x = x.permute(0, 2, 1).unsqueeze(2) 
        x_rec = self.model(x)
        bs, c, h, t = x_rec.shape
        x_rec = x_rec.reshape(bs, c*h, t)
        return x_rec


    # def training_step(self, batch, batch_idx):
    #     loss, loss_dict = self.shared_step(batch)
    #     self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)
    #     self.log("global_step", self.global_step,
    #              prog_bar=True, logger=True, on_step=True, on_epoch=False)
    #     if self.use_scheduler:
    #         lr = self.optimizers().param_groups[0]['lr']
    #         self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)
    #     return loss
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        x = self.get_input(batch, self.first_stage_key)
        z = self.encode_first_stage_spec(x).detach()
        x_rec = self(z)                #   x_reconstruction

        if optimizer_idx == 0:
            # Decoder Loss
            decode_loss, log_dict_decode = self.loss(inputs=x, reconstructions=x_rec, optimizer_idx=optimizer_idx, global_step=self.global_step, last_layer=self.get_last_layer(), split="train")
            self.log_dict(log_dict_decode, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return decode_loss

        if optimizer_idx == 1:
            # Discriminator
            discloss, log_dict_disc = self.loss(inputs=x, reconstructions=x_rec, optimizer_idx=optimizer_idx, global_step=self.global_step, last_layer=self.get_last_layer(), split="train")
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss


    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.first_stage_key)
        z = self.encode_first_stage_spec(x).detach()
        x_rec = self(z)

        decode_loss, log_dict_decode = self.loss(inputs=x, reconstructions=x_rec, optimizer_idx=0, global_step=self.global_step, last_layer=self.get_last_layer(), split="val")
        discloss, log_dict_disc = self.loss(inputs=x, reconstructions=x_rec, optimizer_idx=1, global_step=self.global_step, last_layer=self.get_last_layer(), split="val")

        self.log("val/rec_loss", log_dict_decode["val/rec_loss"])
        self.log_dict(log_dict_decode)
        self.log_dict(log_dict_disc)

        return self.log_dict


    def get_last_layer(self):
        return self.model.conv_out.weight


    # @torch.no_grad()
    # def validation_step(self, batch, batch_idx):
    #     _, loss_dict = self.shared_step(batch)
    #     self.log_dict(loss_dict, prog_bar=False, logger=True, on_step=False, on_epoch=True)


    @torch.no_grad()
    def log_sound(self, batch, N=4, split=None):
        log = dict()
        x = self.get_input(batch, self.first_stage_key, bs=N)
        x_rec = self.get_x_rec(x)
        log["inputs_spec"] = x
        log["reconstruction_spec"] = x_rec
        try:
            log["video_time"] = batch["video_time"]
            log["video_frame_path"] = batch["video_frame_path"]
        except:
            pass
        return log
    

    # # Configure optimizers:
    # def configure_optimizers(self):
    #     lr = self.learning_rate
    #     params = list(self.model.parameters())
    #     opt = torch.optim.Adam(params, lr=lr)
    #     if self.use_scheduler:
    #         assert 'target' in self.scheduler_config
    #         scheduler = instantiate_from_config(self.scheduler_config)
    #         print("Setting up LambdaLR scheduler...")
    #         scheduler = [
    #             {
    #                 'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
    #                 'interval': 'step',
    #                 'frequency': 1
    #             }]
    #         return [opt], scheduler
    #     return opt

    
    # GAN Loss:
    def configure_optimizers(self):
        lr = self.learning_rate

        opt_ae = torch.optim.Adam(list(self.model.parameters()), lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(list(self.loss.discriminator.parameters()), lr=lr, betas=(0.5, 0.9))

        return [opt_ae, opt_disc], []



