import os
import torch
import importlib
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from tqdm import tqdm
from einops import rearrange, repeat, reduce
# from admtransformer_utils import Transformer
from adm.modules.stage3_diffusion.transformer_utils import Transformer


"""
Diffusion Model:
    Goal: Video Feat --> Audio Feat
    Shape: (T x C) --> (T x C)

Architecture: 
   (Denoised Transformer): B x T x 2C --> B x L x C 

DiffusionPriorNetwork: 
DiffusionPrior: (Wrapper, Diffusion Loss) 
"""


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



def exists(val):
    return val is not None

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def first(arr, d = None):
    if len(arr) == 0:
        return d
    return arr[0]

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / first(alphas_cumprod)
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)


def quadratic_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps, dtype = torch.float64) ** 2


def sigmoid_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    betas = torch.linspace(-6, 6, timesteps, dtype = torch.float64)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


def l2norm(t):
    return F.normalize(t, dim = -1)

class NoiseScheduler(nn.Module):
    def __init__(self, *, beta_schedule, timesteps, loss_type, p2_loss_weight_gamma = 0., p2_loss_weight_k = 1):
        super().__init__()

        if beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        elif beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == "quadratic":
            betas = quadratic_beta_schedule(timesteps)
        elif beta_schedule == "jsd":
            betas = 1.0 / torch.linspace(timesteps, 1, timesteps)
        elif beta_schedule == "sigmoid":
            betas = sigmoid_beta_schedule(timesteps)
        else:
            raise NotImplementedError()

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis = 0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        if loss_type == 'l1':
            loss_fn = F.l1_loss
        elif loss_type == 'l2':
            loss_fn = F.mse_loss
        elif loss_type == 'huber':
            loss_fn = F.smooth_l1_loss
        else:
            raise NotImplementedError()

        self.loss_type = loss_type
        self.loss_fn = loss_fn

        # register buffer helper function to cast double back to float
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # p2 loss reweighting
        self.has_p2_loss_reweighting = p2_loss_weight_gamma > 0.
        register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

    def sample_random_times(self, batch):
        return torch.randint(0, self.num_timesteps, (batch,), device = self.betas.device, dtype = torch.long)

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_sample(self, x_start, t, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def calculate_v(self, x_start, t, noise = None):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def q_sample_from_to(self, x_from, from_t, to_t, noise = None):
        shape = x_from.shape
        noise = default(noise, lambda: torch.randn_like(x_from))

        alpha = extract(self.sqrt_alphas_cumprod, from_t, shape)
        sigma = extract(self.sqrt_one_minus_alphas_cumprod, from_t, shape)
        alpha_next = extract(self.sqrt_alphas_cumprod, to_t, shape)
        sigma_next = extract(self.sqrt_one_minus_alphas_cumprod, to_t, shape)

        return x_from * (alpha_next / alpha) + noise * (sigma_next * alpha - sigma * alpha_next) / alpha

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def p2_reweigh_loss(self, loss, times):
        if not self.has_p2_loss_reweighting:
            return loss
        return loss * extract(self.p2_loss_weight, times, loss.shape)



# classifier free guidance functions
def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob



class DiffusionPriorNetwork(nn.Module):

    def __init__(self, dim=512, in_dim=1536, num_timesteps=250, T=16, **kwargs):
        
        super().__init__()
        
        self.dim = dim
        self.T = T       # Feature Len T

        # Embeddings:
        self.null_video_embeds = nn.Parameter(torch.randn(1, self.T, dim))
        self.null_spec_embeds = nn.Parameter(torch.randn(1, self.T, dim))

        # Network:
        self.to_time_embeds = nn.Embedding(num_timesteps, dim)       # Time Embeddings
        self.transformer = Transformer(in_dim=in_dim, **kwargs)       # concat               # Transformer Non-Casual
        


    def forward(self, spec_embed, diffusion_timesteps, video_cond_drop_prob=0., spec_cond_drop_prob=0., video_embed=None):

        bs, dim, device, dtype = spec_embed.shape[0], spec_embed.shape[1], spec_embed.device, spec_embed.dtype


        # classifier free guidance masks
        video_keep_mask = prob_mask_like((bs, ), 1 - video_cond_drop_prob, device=device)
        video_keep_mask = rearrange(video_keep_mask, 'b -> b 1 1')

        spec_keep_mask = prob_mask_like((bs, ), 1 - spec_cond_drop_prob, device=device)
        spec_keep_mask = rearrange(spec_keep_mask, 'b -> b 1 1')

        # Mask Out Video Embeds:
        null_video_embeds = self.null_video_embeds.to(video_embed.dtype)
        video_embed = torch.where(video_keep_mask, video_embed, null_video_embeds)

        # Mask Out Spec Embeds:
        null_spec_embeds = self.null_spec_embeds.to(spec_embed.dtype)
        spec_embed = torch.where(spec_keep_mask, spec_embed, null_spec_embeds)

        # Time Embeds:
        time_embed = self.to_time_embeds(diffusion_timesteps)               # B x C
        time_embed = time_embed.unsqueeze(1).repeat(1, self.T, 1)           # B x T x C

        # Input tokens:
        tokens = torch.cat([spec_embed, video_embed, time_embed], dim=-1)   # B x T x (3C)
        out = self.transformer(tokens)                                      # B x T x C
        return out


    def forward_with_cond_scale(self,*args, cond_scale = 1., **kwargs):
        logits = self.forward(*args, **kwargs)
        if cond_scale == 1:
            return logits
        null_logits = self.forward(*args, text_cond_drop_prob = 1., image_cond_drop_prob = 1, **kwargs)
        return null_logits + (logits - null_logits) * cond_scale




class DiffusionPrior(nn.Module):
    """Diffusion Prior Wrapper
    self.diff_net:  DiffusionPriorNetwork
    Method:
        p_loss , q_forward , p_forward
    """

    def __init__(self, 
                 timesteps = 250,
                 sample_time_steps = 50,
                 cond_drop_prob = 0.,
                 loss_type = "l2",
                 predict_x_start = True,
                 predict_v = False,
                 beta_schedule = "cosine",
                 clamp_l2norm = False,
                 video_cond_drop_prob = None,
                 spec_cond_drop_prob = None,
                 sampling_clamp_l2norm = False,
                 sampling_final_clamp_l2norm = False,
                 spec_embed_scale = None,
                 clip_embed_dim = 512,
                 diff_net_config = None,
                 ):
        
        super().__init__()

        # Noise Scheduler:
        self.noise_scheduler = NoiseScheduler(
            beta_schedule = beta_schedule,
            timesteps = timesteps,
            loss_type = loss_type,
        )

        self.predict_x_start = predict_x_start
        self.predict_v = predict_v
        self.cond_drop_prob = cond_drop_prob

        # clamp:
        self.clamp_l2norm = clamp_l2norm
        self.sampling_final_clamp_l2norm = sampling_final_clamp_l2norm
        self.sampling_clamp_l2norm = sampling_clamp_l2norm
        self.spec_embed_scale = default(spec_embed_scale, clip_embed_dim ** 0.5)

        # Cond Drop:
        self.spec_cond_drop_prob = default(spec_cond_drop_prob, cond_drop_prob)
        self.video_cond_drop_prob = default(video_cond_drop_prob, cond_drop_prob)


        # Diffusion Model:
        self.diff_net = DiffusionPriorNetwork(**diff_net_config)    


    
    def l2norm_clamp_embed(self, embed):
        return l2norm(embed) * self.spec_embed_scale


    def p_losses(self, video_embed, times, spec_embed, noise = None, prefix=None):
        noise = default(noise, lambda: torch.randn_like(spec_embed))   # Sample Noise
        spec_embed_noisy = self.noise_scheduler.q_sample(x_start=spec_embed, t=times, noise=noise)  # Add Noise to Spec
        # Use Classifier Free Gudiance:
        pred = self.diff_net(spec_embed_noisy, times, video_embed=video_embed, video_cond_drop_prob = self.video_cond_drop_prob, spec_cond_drop_prob = self.spec_cond_drop_prob)
        
        if self.predict_x_start and self.clamp_l2norm:
            pred = self.l2norm_clamp_embed(pred)

        if self.predict_v:
            target = self.noise_scheduler.calculate_v(spec_embed, times, noise)
        elif self.predict_x_start:
            target = spec_embed
        else:
            target = noise

        loss = self.noise_scheduler.loss_fn(pred, target)
        loss_dict = {"{}/l2_loss".format(prefix): loss.item()}
        return loss, loss_dict
        

    def forward(self, video_feat, spec_feat, *args, **kwargs):
        # timestep conditioning from ddpm:
        bs, device = video_feat.shape[0], video_feat.device
        times = self.noise_scheduler.sample_random_times(bs)

        # Scale Spec Embed:
        spec_feat *= self.spec_embed_scale

        # Calculate Forward loss:
        p_loss, loss_dict = self.p_losses(video_feat, times, spec_feat, *args, **kwargs)
        return p_loss, loss_dict
    

    def p_mean_variance(self, x, t, video_embed, clip_denoised = False, cond_scale=1.):
        assert not (cond_scale != 1. and not self.can_classifier_guidance), 'the model was not trained with conditional dropout, and thus one cannot use classifier free guidance (cond_scale anything other than 1)'

        pred = self.diff_net.forward_with_cond_scale(x, t, cond_scale=cond_scale, video_embed=video_embed)

        if self.predict_v:
            x_start = self.noise_scheduler.predict_start_from_v(x, t = t, v = pred)
        elif self.predict_x_start:
            x_start = pred
        else:
            x_start = self.noise_scheduler.predict_start_from_noise(x, t = t, noise = pred)

        if clip_denoised and not self.predict_x_start:
            x_start.clamp_(-1., 1.)

        if self.predict_x_start and self.sampling_clamp_l2norm:
            x_start = l2norm(x_start) * self.spec_embed_scale

        model_mean, posterior_variance, posterior_log_variance = self.noise_scheduler.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start



    @torch.no_grad()
    def p_sample(self, x, t, video_embed = None, clip_denoised = True, cond_scale = 1.):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x=x, t=t, video_embed=video_embed, clip_denoised=clip_denoised, cond_scale=cond_scale)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        pred = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred, x_start



    @torch.no_grad()
    def p_sample_loop_ddpm(self, video_feat, cond_scale=1.):
        bs, device = video_feat.shape[0], self.device
        spec_embed = torch.randn(video_feat.shape, device=device)

        for i in tqdm(reversed(range(0, self.noise_scheduler.num_timesteps))):
            times = torch.full((bs, ), i , device=device, dtype=torch.long)
            spec_embed, x_start = self.p_sample(spec_embed, times, video_embed=video_feat, cond_scale=cond_scale)

        if self.sampling_final_clamp_l2norm and self.predict_x_start:
            spec_embed = self.l2norm_clamp_embed(spec_embed)

        return spec_embed
            


    @torch.no_grad()
    def p_sample_loop_ddim(self, video_feat, ddim_steps, eta=1., cond_scale=1.):

        bs, device, alphas, total_timesteps = video_feat.shape[0], video_feat.device, self.noise_scheduler.alphas_cumprod_prev, self.noise_scheduler.num_timesteps

        times = torch.linspace(-1, total_timesteps, steps=ddim_steps + 1)[:-1]
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        spec_embed = torch.randn(video_feat.shape, device=device)

        # DDIM Loops:
        for time, time_next in tqdm(time_pairs, desc="Using DDIM Sampler step"):
            alpha = alphas[time]
            alpha_next = alphas[time_next]
            time_cond = torch.full((bs,) , time, device=device, dtype=torch.long)
            pred = self.diff_net.forward_with_cond_scale(spec_embed, time_cond, cond_scale, video_embed=video_feat)

            # derive x0:
            if self.predict_v:
                x_start = self.noise_scheduler.predict_start_from_v(spec_embed, t=time_cond, v=pred)
            elif self.predict_x_start:
                x_start = pred
            else:
                x_start = self.noise_scheduler.predict_start_from_noise(spec_embed, t=time_cond, noise=pred)

            # clip x0 before maybe predicting noise
            # if not self.predict_x_start:
            #     x_start.clamp_(-1., 1.)

            if self.predict_x_start and self.sampling_clamp_l2norm:
                x_start = self.l2norm_clamp_embed(x_start)

            # predict noise
            if self.predict_x_start or self.predict_v:
                pred_noise = self.noise_scheduler.predict_noise_from_start(spec_embed, t=time_cond, x0=x_start)
            else:
                pred_noise = pred

            if time_next < 0:
                spec_embed = x_start
                continue

            c1 = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c2 = ((1 - alpha_next) - torch.square(c1)).sqrt()
            noise = torch.randn_like(spec_embed) if time_next > 0 else 0.

            spec_embed = x_start * alpha_next.sqrt() + c1 * noise + c2 * pred_noise

        if self.predict_x_start and self.sampling_final_clamp_l2norm:
            spec_embed = self.l2norm_clamp_embed(spec_embed)

        return spec_embed



    @torch.no_grad()
    def p_sample_loop(self, video_feat, timesteps, ddim_steps, *args, **kwargs):

        timesteps = default(timesteps, self.noise_scheduler.num_timesteps)
        assert timesteps <= self.noise_scheduler.num_timesteps

        if ddim_steps:
            normalized_spec_embed = self.p_sample_loop_ddim(video_feat, ddim_steps=ddim_steps, *args, **kwargs)
        else:
            normalized_spec_embed = self.p_sample_loop_ddpm(video_feat, *args, **kwargs)
        
        spec_embed = normalized_spec_embed / self.spec_embed_scale
        return spec_embed



    # Sampling:
    @torch.no_grad()
    def sample(self, video_feat, bs, ddim_steps=50, cond_scale = 1., timesteps=None):
        timesteps = default(timesteps, ddim_steps)
        spec_embeds_sample = self.p_sample_loop(video_feat, cond_scale=cond_scale, timesteps=timesteps, ddim_steps=ddim_steps)
        return spec_embeds_sample


class Diffusion_and_CLIP_Wrapper(pl.LightningModule):
    """
    Diffusion and Stage1 CLIP Wrapper:
    self.stage1_model: CLIP
    self.diff_prior: Diffusion Prior
    """

    def __init__(self, 
                monitor,
                first_stage_config,
                diff_prior_config,
                scheduler_config,
                decoder_config,
                decoder_ckpt=None,
                first_stage_ckpt=None,
                video_key = "video",
                spec_key = "spec",
                normalize = True,
                avg = False,
                lossconfig = None,
                *args, **kwargs):
        
        super().__init__()

        # Initalize Fist Stage Model: (Video, Audio Encoder)
        self.instantiate_first_stage(first_stage_config)
        self.first_stage_ckpt = first_stage_ckpt
        if self.first_stage_ckpt is not None:
            self.init_first_from_ckpt(self.first_stage_ckpt)

        # Initalize Decoder:
        if decoder_ckpt:
            self.instantiate_decoder_stage(decoder_config)
            self.init_decoder_from_ckpt(decoder_ckpt)


        self.video_key = video_key
        self.spec_key = spec_key
        self.normalize = normalize
        self.avg = avg
        
        # Initalize Diffusion Prior Model:
        self.diff_prior = instantiate_from_config(diff_prior_config)

        # Scheduler:
        self.monitor = monitor
        self.use_scheduler = scheduler_config
        if self.use_scheduler:
            self.scheduler_config = scheduler_config
        
    # First Stage Model:
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
    

    # Decoder:
    def instantiate_decoder_stage(self, config):
        model = instantiate_from_config(config)
        self.decoder = model.eval()
        self.decoder.train = disabled_train
        for param in self.decoder.parameters():
            param.requires_grad = False
    

    def init_decoder_from_ckpt(self, path):
        model = torch.load(path, map_location="cpu")
        if "state_dict" in list(model.keys()):
            model = model["state_dict"]
        # Remove: module prefix
        new_model = {}
        for key in model.keys():
            new_key = key.replace("module.","")
            new_key = new_key.replace("model.", "")
            new_model[new_key] = model[key]
        missing, unexpected = self.decoder.load_state_dict(new_model, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")



    # def get_single_input(self, batch, k, bs=None):
    #     x = batch[k]
    #     x = x.to(memory_format=torch.contiguous_format).float()
    #     return x

    def get_input(self, batch, video_key="video", spec_key="spec", bs=None, return_video_name=False):
        """Get Input"""
        spec, video, video_name, time_list = batch
        spec = spec.to(memory_format=torch.contiguous_format).float()
        video = video.to(memory_format=torch.contiguous_format).float()
        # video = self.get_single_input(batch, video_key)
        # spec = self.get_single_input(batch, spec_key)
        if bs is not None:
            video = video[:bs]
            spec = spec[:bs]
        video = video.to(self.device)
        spec = spec.to(self.device)

        if return_video_name:
            return video, spec, video_name, time_list

        return video, spec

    @torch.no_grad()
    def get_first_stage_encode(self, video=None, spec=None, video_key=None, spec_key=None, normalize=True, avg=False):
        video_feat, spec_feat = None, None
        if video_key:
            video_feat = self.first_stage_model.encode_video(video, normalize=normalize, avg=avg)
        if spec_key:
            spec_feat = self.first_stage_model.encode_spec(spec, normalize=normalize, avg=avg)
        return video_feat, spec_feat


    # Training Steps:
    def training_step(self, batch, batch_idx):
        # Get Input Data:
        video, spec = self.get_input(batch, self.video_key, self.spec_key)
        # Get Video, Spec Feat:
        video_feat, spec_feat = self.get_first_stage_encode(video, spec, video_key=self.video_key, spec_key=self.spec_key, normalize=self.normalize, avg=self.avg)
        # Diffusion Prior Loss:
        loss, loss_dict = self.diff_prior(video_feat, spec_feat, prefix="train")

        # Logger:
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("global_step", self.global_step, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return loss
    

    def validation_step(self, batch, batch_idx):
        # Get Input Data:
        video, spec = self.get_input(batch, self.video_key, self.spec_key)
        # # Get Video, Spec Feat:
        video_feat, spec_feat = self.get_first_stage_encode(video, spec, video_key=self.video_key, spec_key=self.spec_key, normalize=self.normalize, avg=self.avg)
        # Get Loss:
        _, loss_dict = self.diff_prior(video_feat, spec_feat, prefix="val")
        self.log_dict(loss_dict, prog_bar=False, logger=True, on_step=False, on_epoch=True)



    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.diff_prior.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt
    


    ## Sampling:
    @torch.no_grad()
    def log_sound(self, batch, N=4, split=None, ddim_steps=50, cond_scale=1.,):
        log = dict()
        video, spec, video_name, video_time = self.get_input(batch, self.video_key, self.spec_key, bs=N, return_video_name=True)
        video_feat, spec_feat = self.get_first_stage_encode(video, spec, video_key=self.video_key, spec_key=self.spec_key, normalize=self.normalize, avg=self.avg)
        log["inputs_spec"] = spec

        # Get Reconstruction:
        spec_feat = spec_feat.permute(0, 2, 1).unsqueeze(2)
        spec_rec = self.decoder(spec_feat)
        bs, c, h, t = spec_rec.shape
        spec_rec = spec_rec.reshape(bs, c*h, t)
        log["reconstruction_spec"] = spec_rec

        # Get Diffusion Sampling:
        sample_spec_embed = self.diff_prior.sample(video_feat, bs=N, ddim_steps=ddim_steps, cond_scale=cond_scale)
        # sample_spec = self.decoder.reconstruct_spec(sample_spec_embed)
        sample_spec_embed = sample_spec_embed.permute(0, 2, 1).unsqueeze(2)
        sample_spec = self.decoder(sample_spec_embed)
        bs, c, h, t = sample_spec.shape
        sample_spec = sample_spec.reshape(bs, c*h, t)


        log["diff_sample_spec"] = sample_spec

        try:
            log["video_name"] = video_name
            log["video_time"] = video_time
        except:
            pass

        return log

