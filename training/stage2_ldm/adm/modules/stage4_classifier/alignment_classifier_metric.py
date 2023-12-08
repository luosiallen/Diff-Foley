import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import importlib
from torch.optim.lr_scheduler import LambdaLR
from inspect import isfunction
import numpy as np
from functools import partial

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

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


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == "linear":
        betas = (
                torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
        )

    elif schedule == "cosine":
        timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    elif schedule == "sqrt":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 0.5
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas.numpy()


class Alignment_Classifier_metric(pl.LightningModule):

    def __init__(self,
                 classifier_config,
                 first_stage_config,
                 cond_stage_config,
                 monitor,
                 first_stage_ckpt=None,
                 first_stage_key="spec",
                 scale_factor = 1.0,
                 timesteps = 2,
                 given_betas=None,
                 beta_schedule = "linear",
                 linear_start=1e-4,
                 linear_end=2e-2,
                 cosine_s=8e-3,
                 v_posterior=0.,
                 parameterization="eps",
                 *args, **kwargs):
    
        super().__init__()

        self.instantiate_first_stage(first_stage_config)
        self.first_stage_ckpt = first_stage_ckpt
        if self.first_stage_ckpt is not None:
            self.init_first_from_ckpt(self.first_stage_ckpt)

        # Init Model
        self.model = instantiate_from_config(classifier_config)
        self.cond_model = instantiate_from_config(cond_stage_config)


        self.first_stage_key = first_stage_key
        self.monitor = monitor

                
        self.register_buffer('scale_factor', torch.tensor(scale_factor))
        self.bce_loss = nn.BCELoss()
        self.v_posterior = v_posterior
        self.parameterization = parameterization
        
        # Register Schedule:
        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)

    
    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                        2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()


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


    @torch.no_grad()
    def encode_first_stage(self, x):
        x = self.first_stage_model.encode(x)
        return x
    
    @torch.no_grad()
    def get_first_stage_encoding(self, encoder_posterior):
        z = encoder_posterior.sample()
        return self.scale_factor * z
    
    @torch.no_grad()
    def encode_spec_z(self, x):
        encoder_posterior = self.encode_first_stage(x)
        z = self.get_first_stage_encoding(encoder_posterior).detach()
        return z

    @torch.no_grad()
    def encode_cond(self, x):
        encode_feat = self.cond_model(x)
        return encode_feat

    @torch.no_grad()
    def get_input(self, batch, k, bs=None):
        x = batch[k]
        x = x.to(memory_format=torch.contiguous_format).float()
        if bs is not None:
            x = x[:bs]
        x = x.to(self.device)

        if k == "mix_spec":
            encoder_posterior = self.encode_first_stage(x)
            z = self.get_first_stage_encoding(encoder_posterior).detach()
            return z
        return x    # spec


    def get_q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        return x_noisy
    
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)


    def training_step(self, batch, batch_idx):
        spec = self.get_input(batch, k="mix_spec")
        video_feat = self.get_input(batch, k="mix_video_feat")
        labels = self.get_input(batch, k="labels")
        # Noisy Spec Classifier:
        # t = torch.randint(0, self.num_timesteps, (spec.shape[0], ), device=self.device).long()
        t = torch.tensor(0).reshape(1,).repeat(spec.shape[0]).to(self.device).long()
        # spec_noisy = self.get_q_sample(spec, t)
        video_feat_encode = self.cond_model(video_feat)
        logits = self.model(spec, context=video_feat_encode, timesteps=t)
        loss = self.bce_loss(logits, labels.float().unsqueeze(1))

        # loss Dict:
        loss_dict = {}
        predicted = torch.round(logits)
        acc = ((predicted == labels.float().unsqueeze(1)).sum() / predicted.shape[0]).item()
        prefix = "train" if self.training else "val"
        loss_dict.update({f'{prefix}/bce_loss': loss.item()})
        loss_dict.update({f'{prefix}/acc': acc})
        self.log_dict(loss_dict, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return loss
    

    def validation_step(self, batch, batch_idx):
        spec = self.get_input(batch, k="mix_spec")
        video_feat = self.get_input(batch, k="mix_video_feat")
        labels = self.get_input(batch, k="labels")
        # Noisy Spec Classifier:
        # t = torch.randint(0, self.num_timesteps, (spec.shape[0], ), device=self.device).long()
        t = torch.tensor(0).reshape(1,).repeat(spec.shape[0]).to(self.device).long()
        video_feat_encode = self.cond_model(video_feat)
        logits = self.model(spec, context=video_feat_encode, timesteps=t)
        loss = self.bce_loss(logits, labels.float().unsqueeze(1))
        # loss Dict:
        loss_dict = {}
        predicted = torch.round(logits)
        acc = ((predicted == labels.float().unsqueeze(1)).sum() / predicted.shape[0]).item()
        prefix = "train" if self.training else "val"
        loss_dict.update({f'{prefix}/bce_loss': loss.item()})
        loss_dict.update({f'{prefix}/acc': acc})
        self.log_dict(loss_dict, prog_bar=False, logger=True, on_step=True, on_epoch=True)


    def forward(self, spec_noisy, video_feat, t):
        logits = self.model(spec_noisy, video_feat, t)
        return logits


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


    def configure_optimizers(self):
        lr = self.learning_rate
        params1 = list(self.model.parameters())
        params2 = list(self.cond_model.parameters())
        params = params1 + params2
        opt = torch.optim.AdamW(params, lr=lr)
        return opt








class Alignment_Classifier_wo_Encoder(pl.LightningModule):

    def __init__(self,
                 classifier_config,
                 cond_stage_config,
                 monitor,
                 first_stage_ckpt=None,
                 first_stage_key="spec",
                 scale_factor = 1.0,
                 timesteps = 1000,
                 given_betas=None,
                 beta_schedule = "linear",
                 linear_start=1e-4,
                 linear_end=2e-2,
                 cosine_s=8e-3,
                 v_posterior=0.,
                 parameterization="eps",
                 *args, **kwargs):
    
        super().__init__()

        # Init Model
        self.model = instantiate_from_config(classifier_config)
        self.cond_model = instantiate_from_config(cond_stage_config)


        self.first_stage_key = first_stage_key
        self.monitor = monitor

                
        self.register_buffer('scale_factor', torch.tensor(scale_factor))
        self.bce_loss = nn.BCELoss()
        self.v_posterior = v_posterior
        self.parameterization = parameterization
        
        # Register Schedule:
        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)

    
    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                        2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()


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


    @torch.no_grad()
    def encode_first_stage(self, x):
        x = self.first_stage_model.encode(x)
        return x
    
    @torch.no_grad()
    def get_first_stage_encoding(self, encoder_posterior):
        z = encoder_posterior.sample()
        return self.scale_factor * z

    @torch.no_grad()
    def get_input(self, batch, k, bs=None):
        x = batch[k]
        x = x.to(memory_format=torch.contiguous_format).float()
        if bs is not None:
            x = x[:bs]
        x = x.to(self.device)

        if k == "mix_spec":
            encoder_posterior = self.encode_first_stage(x)
            z = self.get_first_stage_encoding(encoder_posterior).detach()
            return z
        return x    # spec


    def get_q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        return x_noisy
    
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)


    def training_step(self, batch, batch_idx):
        spec = self.get_input(batch, k="mix_spec")
        video_feat = self.get_input(batch, k="mix_video_feat")
        labels = self.get_input(batch, k="labels")
        # Noisy Spec Classifier:
        t = torch.randint(0, self.num_timesteps, (spec.shape[0], ), device=self.device).long()
        spec_noisy = self.get_q_sample(spec, t)
        video_feat_encode = self.cond_model(video_feat)
        logits = self.model(spec_noisy, context=video_feat_encode, timesteps=t)
        loss = self.bce_loss(logits, labels.float().unsqueeze(1))

        # loss Dict:
        loss_dict = {}
        predicted = torch.round(logits)
        acc = ((predicted == labels.float().unsqueeze(1)).sum() / predicted.shape[0]).item()
        prefix = "train" if self.training else "val"
        loss_dict.update({f'{prefix}/bce_loss': loss.item()})
        loss_dict.update({f'{prefix}/acc': acc})
        self.log_dict(loss_dict, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return loss
    

    def validation_step(self, batch, batch_idx):
        spec = self.get_input(batch, k="mix_spec")
        video_feat = self.get_input(batch, k="mix_video_feat")
        labels = self.get_input(batch, k="labels")
        # Noisy Spec Classifier:
        t = torch.randint(0, self.num_timesteps, (spec.shape[0], ), device=self.device).long()
        spec_noisy = self.get_q_sample(spec, t)
        video_feat_encode = self.cond_model(video_feat)
        logits = self.model(spec_noisy, context=video_feat_encode, timesteps=t)
        loss = self.bce_loss(logits, labels.float().unsqueeze(1))
        # loss Dict:
        loss_dict = {}
        predicted = torch.round(logits)
        acc = ((predicted == labels.float().unsqueeze(1)).sum() / predicted.shape[0]).item()
        prefix = "train" if self.training else "val"
        loss_dict.update({f'{prefix}/bce_loss': loss.item()})
        loss_dict.update({f'{prefix}/acc': acc})
        self.log_dict(loss_dict, prog_bar=False, logger=True, on_step=True, on_epoch=True)


    def forward(self, spec_noisy, video_feat, t):
        logits = self.model(spec_noisy, context=video_feat, timesteps=t)
        return logits


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


    def configure_optimizers(self):
        lr = self.learning_rate
        params1 = list(self.model.parameters())
        params2 = list(self.cond_model.parameters())
        params = params1 + params2
        opt = torch.optim.AdamW(params, lr=lr)
        return opt