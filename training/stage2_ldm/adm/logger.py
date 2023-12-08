

import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.distributed import rank_zero_only
import torchvision
import pytorch_lightning as pl
import numpy as np
import soundfile as sf
from PIL import Image
import imageio
import matplotlib.pyplot as plt
import librosa
import os
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip, concatenate_videoclips



class SoundLogger2(Callback):
    def __init__(self, train_batch_frequency,val_batch_frequency, max_sound_num, sr=22050, clamp=True, increase_log_steps=False,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False, ddim_step=250):
        super().__init__()
        self.rescale = rescale
        self.train_batch_freq = train_batch_frequency
        self.val_batch_freq = val_batch_frequency
        self.max_sound_num = max_sound_num
        self.log_steps = [2 ** n for n in range(int(np.log2(self.train_batch_freq)) + 1)]
        self.log_steps_val = [2 ** n for n in range(int(np.log2(self.val_batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.train_batch_freq]
            self.log_steps_val = [self.val_batch_freq] 
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_first_step = log_first_step
        self.sr = sr
        self.ddim_step = ddim_step

        self.train_vis_frequency = train_batch_frequency

    @rank_zero_only
    def log_local(self, save_dir, split, log_dict,
                  global_step, current_epoch, batch_idx, show_curve=True, scale_factor=1):

        # root:
        root = os.path.join(save_dir, "sound_eval", split, "epoch_{}_step_{}".format(current_epoch, global_step))

        gt_sound_list = log_dict['inputs'].detach().cpu().numpy()
        rec_sound_list = log_dict['reconstruction'].detach().cpu().numpy()
        video_path_list= log_dict['video_frame_path']
        video_time_list = log_dict['video_time']
        diff_sample_list = log_dict['samples'].detach().cpu().numpy()

        # root = os.path.join(save_dir, "sound", split, "epoch_{}_step_{}".format(current_epoch, global_step))

        os.makedirs(root,exist_ok=True)
        

        for i in range(len(gt_sound_list)):
            print('Gen examples ===========> {}'.format(i))
            sample_folder = os.path.join(root, "sample_{}".format(i))
            os.makedirs(sample_folder, exist_ok=True)
            gt_sound = gt_sound_list[i][0]
            rec_sound = rec_sound_list[i][0]
            sample = diff_sample_list[i][0]
            sf.write(os.path.join(sample_folder, "sample_{}_gt.wav".format(i)), gt_sound, self.sr)
            sf.write(os.path.join(sample_folder, "sample_{}_stage1_rec_clamp.wav".format(i)), rec_sound, self.sr)
            sf.write(os.path.join(sample_folder, "sample_{}_diff_sample_clamp.wav".format(i)), sample, self.sr) 
            video = self.concat_frame_video(video_path_list[i], video_time_list[i])
            video_save_path = os.path.join(sample_folder, "origin_video.mp4")
            imageio.mimsave(video_save_path, video, fps=21.5)

                # show curve:
        if show_curve: 
            try: 
                pred_x0_range_min = np.array(log_dict["intermediates"]["pred_x0_range_min"]).squeeze(2).transpose(1,0)     #  sample_num x T
                pred_x0_range_max = np.array(log_dict["intermediates"]["pred_x0_range_max"]).squeeze(2).transpose(1,0)     #  sample_num x T 
                # import pdb
                # pdb.set_trace()
                # x_prev_range_min = np.array(log_dict["intermediates"]['x_prev_range_min']).squeeze(2).transpose(1,0)
                # x_prev_range_max = np.array(log_dict["intermediates"]['x_prev_range_max']).squeeze(2).transpose(1,0)
                x_prev_range_min = np.array(log_dict["intermediates"]['x_prev_range_min']).squeeze(2).transpose(1,0)
                x_prev_range_max = np.array(log_dict["intermediates"]['x_prev_range_max']).squeeze(2).transpose(1,0)

                print('pred_x0 range shape', pred_x0_range_max.shape)
                print('x_prev range shape', x_prev_range_min.shape)


                latent_max = 6 * scale_factor
                latent_min = -6 * scale_factor

                plt.figure(figsize=(20,10))
                for i in range(pred_x0_range_min.shape[0]):
                    pred_x0_min_list = pred_x0_range_min[i]
                    pred_x0_max_list = pred_x0_range_max[i]
                    x_prev_min_list = x_prev_range_min[i]
                    x_prev_max_list = x_prev_range_max[i]

                    plt.subplot(121)
                    plt.plot(range(len(pred_x0_min_list)), pred_x0_min_list, label="Ours Diffusion pred_x0 Min Value {}".format(i))
                    plt.plot(range(len(pred_x0_max_list)), pred_x0_max_list, label="Ours Diffusion pred_x0 Max Value {}".format(i)) 

                    plt.subplot(122)
                    plt.plot(range(len(x_prev_min_list)), x_prev_min_list, label="Ours Diffusion x_prev Min Value {}".format(i))
                    plt.plot(range(len(x_prev_max_list)), x_prev_max_list, label="Ours Diffusion x_prev Max Value {}".format(i))

                plt.subplot(121)
                plt.hlines(latent_max,xmin=0,xmax=len(x_prev_min_list),color='red', label='Latent Max Range / scale_factor:{}'.format(scale_factor),linewidth=5)
                plt.hlines(latent_min,xmin=0,xmax=len(x_prev_min_list),color='red', label='Latent min Range / scale_factor:{}'.format(scale_factor),linewidth=5)
                # plt.ylim((-10,10))
                plt.legend()
                plt.subplot(122)
                plt.hlines(latent_max,xmin=0,xmax=len(x_prev_min_list),color='red', label='Latent Max Range / scale_factor:{}'.format(scale_factor),linewidth=5)
                plt.hlines(latent_min,xmin=0,xmax=len(x_prev_min_list),color='red', label='Latent min Range / scale_factor:{}'.format(scale_factor),linewidth=5)
                # plt.ylim((-10,10))
                plt.legend()
                plt.savefig(os.path.join(root, "sampling_range_new1.png"))
            except Exception as e:
                print(e)

    def concat_frame_video(self, video_path, video_time):
        # start_frame, end_frame = video_time[0], video_time[1]
        start_frame, end_frame = int(video_time.split('_')[0]), int(video_time.split('_')[1])
        frame_list = []
        for i in range(start_frame, end_frame):
            try:
                frame_path = os.path.join(video_path, '{}.png'.format(i+1))
                frame = np.array(Image.open(frame_path).convert('RGB'))
                frame_list.append(frame)
            except:
                pass
        return frame_list


    def log_sound(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        
        if (self.check_frequency(check_idx, split) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_sound") and
                callable(pl_module.log_sound) and
                self.max_sound_num > 0):

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                log_dict = pl_module.log_sound(batch,N=self.max_sound_num,ddim_steps=self.ddim_step, split=split)

            # gt_sound_list = log_dict['inputs']
            # rec_sound_list = log_dict['reconstruction']
            # video_path_list= log_dict['video_frame_path']
            # video_time_list = log_dict['video_time']

            self.log_local(pl_module.logger.save_dir, split, log_dict, pl_module.global_step, pl_module.current_epoch, batch_idx, scale_factor=pl_module.scale_factor)

            if is_train:
                pl_module.train()
        

    def log_sound_steps(self, pl_module, batch, batch_idx, split="train"):        
        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        with torch.no_grad():
            log_dict = pl_module.log_sound(batch,N=self.max_sound_num, ddim_steps=self.ddim_step, split=split)

        # gt_sound_list = log_dict['inputs']
        # rec_sound_list = log_dict['reconstruction']
        # video_path_list= log_dict['video_frame_path']
        # video_time_list = log_dict['video_time']

        self.log_local(pl_module.logger.save_dir, split, log_dict, pl_module.global_step, pl_module.current_epoch, batch_idx)

        if is_train:
            pl_module.train()



    def check_frequency(self, check_idx, split):
        
        if split == "train":
            if ((check_idx % self.train_batch_freq) == 0 or (check_idx in self.log_steps)) and (
                    check_idx > 0 or self.log_first_step):
                try:
                    self.log_steps.pop(0)
                except IndexError as e:
                    print(e)
                    pass
                return True
            return False
        else:
            if (check_idx % self.val_batch_freq) == 0:
                return True
            else:
                return False


    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        # if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
        if not self.disabled and (pl_module.global_step % self.train_vis_frequency == 0):
            self.log_sound_steps(pl_module, batch, batch_idx, split="train")


    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled and pl_module.global_step > 0:
            self.log_sound(pl_module, batch, batch_idx, split="val")
        if hasattr(pl_module, 'calibrate_grad_norm'):
            if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)





class SoundLogger(Callback):
    def __init__(self, train_batch_frequency,val_batch_frequency, max_sound_num, sr=22050, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False):
        super().__init__()
        self.rescale = rescale
        self.train_batch_freq = train_batch_frequency
        self.val_batch_freq = val_batch_frequency
        self.max_sound_num = max_sound_num
        self.log_steps = [2 ** n for n in range(int(np.log2(self.train_batch_freq)) + 1)]
        self.log_steps_val = [2 ** n for n in range(int(np.log2(self.val_batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.train_batch_freq]
            self.log_steps_val = [self.val_batch_freq] 
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_first_step = log_first_step
        self.sr = sr

        self.train_vis_frequency = train_batch_frequency
        self.val_vis_frequency = val_batch_frequency

    @rank_zero_only
    def log_local(self, save_dir, split, gt_sound_list, rec_sound_list,
                  global_step, current_epoch, batch_idx):

        root = os.path.join(save_dir, "sound", split, "epoch_{}_step_{}".format(current_epoch, global_step))

        os.makedirs(root,exist_ok=True)

        for i in range(len(gt_sound_list)):
            gt_sound = gt_sound_list[i]
            rec_sound = rec_sound_list[i]
            sf.write(os.path.join(root, "sample_{}_gt.wav".format(i)), gt_sound, self.sr)
            sf.write(os.path.join(root, "sample_{}_rec.wav".format(i)), rec_sound, self.sr)



    def log_sound(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        
        if (self.check_frequency(check_idx, split) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_sound") and
                callable(pl_module.log_sound) and
                self.max_sound_num > 0):

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                sound_dict = pl_module.log_sound(batch,max_sound_num=self.max_sound_num, split=split)

            gt_sound_list = sound_dict['gt_sound_list']
            rec_sound_list = sound_dict['rec_sound_list']
            self.log_local(pl_module.logger.save_dir, split, gt_sound_list, rec_sound_list, pl_module.global_step, pl_module.current_epoch, batch_idx)

            if is_train:
                pl_module.train()
        


    def log_sound_steps(self, pl_module, batch, batch_idx, split="train"):        
        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        with torch.no_grad():
            sound_dict = pl_module.log_sound(batch,max_sound_num=self.max_sound_num, split=split)

        gt_sound_list = sound_dict['gt_sound_list']
        rec_sound_list = sound_dict['rec_sound_list']
        self.log_local(pl_module.logger.save_dir, split, gt_sound_list, rec_sound_list, pl_module.global_step, pl_module.current_epoch, batch_idx)

        if is_train:
            pl_module.train()



    def check_frequency(self, check_idx, split):
        
        if split == "train":
            if ((check_idx % self.train_batch_freq) == 0 or (check_idx in self.log_steps)) and (
                    check_idx > 0 or self.log_first_step):
                try:
                    self.log_steps.pop(0)
                except IndexError as e:
                    print(e)
                    pass
                return True
            return False
        else:
            if (check_idx % self.val_batch_freq) == 0:
                return True
            else:
                return False


    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        # if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
        #     self.log_sound(pl_module, batch, batch_idx, split="train")
        # pass    # temp
        if not self.disabled and (pl_module.global_step % self.train_vis_frequency == 0):
            self.log_sound_steps(pl_module, batch, batch_idx, split="train")



    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled and (pl_module.global_step % self.val_vis_frequency == 0):
            self.log_sound_steps(pl_module, batch, batch_idx, split="val")
        # if not self.disabled and pl_module.global_step > 0:
        #     self.log_sound(pl_module, batch, batch_idx, split="val")
        # if hasattr(pl_module, 'calibrate_grad_norm'):
        #     if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
        #         self.log_gradients(trainer, pl_module, batch_idx=batch_idx)
        # pass    # temp
    





class SoundLogger3(Callback):
    def __init__(self, train_batch_frequency,val_batch_frequency, max_sound_num, sr=22050, clamp=True, increase_log_steps=False,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False, ddim_step=250, size_len=64, guidance_scale=1.0, uncond_cond=None):
        super().__init__()
        self.rescale = rescale
        self.train_batch_freq = train_batch_frequency
        self.val_batch_freq = val_batch_frequency
        self.max_sound_num = max_sound_num
        self.log_steps = [2 ** n for n in range(int(np.log2(self.train_batch_freq)) + 1)]
        self.log_steps_val = [2 ** n for n in range(int(np.log2(self.val_batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.train_batch_freq]
            self.log_steps_val = [self.val_batch_freq] 
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_first_step = log_first_step
        self.sr = sr
        print('sr: ', self.sr)
        self.ddim_step = ddim_step

        self.train_vis_frequency = train_batch_frequency
        self.val_vis_frequency = val_batch_frequency
        print('val vis freq:', self.val_vis_frequency)
        self.size_len = size_len
        self.guidance_scale = guidance_scale
        self.uncond_cond = uncond_cond
        print("Guidance Scale: ", self.guidance_scale)
        print("Uncond cond: ", self.uncond_cond)
    
    
    def inverse_op(self, spec):
        sr = self.sr
        n_fft = 1024
        fmin = 125
        fmax = 7600
        nmels = 128
        hoplen = 1024 // 4
        spec_power = 1

        # Inverse Transform
        spec = spec * 100 - 100
        spec = (spec + 20) / 20
        spec = 10 ** spec
        spec_out = librosa.feature.inverse.mel_to_stft(spec, sr = sr, n_fft = n_fft, fmin=fmin, fmax=fmax, power=spec_power)
        wav = librosa.griffinlim(spec_out, hop_length=hoplen)
        return wav

    @rank_zero_only
    def log_local(self, save_dir, split, log_dict,
                  global_step, current_epoch, batch_idx, show_curve=False, scale_factor=1):

        # root:
        root = os.path.join(save_dir, "sound_eval", split, "epoch_{}_step_{}".format(current_epoch, global_step))

        gt_sound_list = log_dict['inputs_spec'].detach().cpu().numpy()
        rec_sound_list = log_dict['reconstruction_spec'].detach().cpu().numpy()
        video_path_list= log_dict['video_frame_path']
        video_time_list = log_dict['video_time']
        diff_sample_list = log_dict['samples'].detach().cpu().numpy()

        # root = os.path.join(save_dir, "sound", split, "epoch_{}_step_{}".format(current_epoch, global_step))

        os.makedirs(root,exist_ok=True)
        

        for i in range(len(gt_sound_list)):
            print('Gen examples ===========> {}'.format(i))
            sample_folder = os.path.join(root, "sample_{}".format(i))
            os.makedirs(sample_folder, exist_ok=True)
            gt_sound = self.inverse_op(gt_sound_list[i])
            rec_sound = self.inverse_op(rec_sound_list[i])
            sf.write(os.path.join(sample_folder, "sample_{}_gt.wav".format(i)), gt_sound, self.sr)
            sf.write(os.path.join(sample_folder, "sample_{}_stage1_rec_clamp.wav".format(i)), rec_sound, self.sr)
            sample = self.inverse_op(diff_sample_list[i])
            sf.write(os.path.join(sample_folder, "sample_{}_diff_sample_clamp.wav".format(i)), sample, self.sr) 
            video = self.concat_frame_video(video_path_list[i], video_time_list[i])
            video_save_path = os.path.join(sample_folder, "origin_video.mp4")
            imageio.mimsave(video_save_path, video, fps=21.5)

                # show curve:
        if show_curve: 
            try: 
                pred_x0_range_min = np.array(log_dict["intermediates"]["pred_x0_range_min"]).squeeze(2).transpose(1,0)     #  sample_num x T
                pred_x0_range_max = np.array(log_dict["intermediates"]["pred_x0_range_max"]).squeeze(2).transpose(1,0)     #  sample_num x T 
                # import pdb
                # pdb.set_trace()
                # x_prev_range_min = np.array(log_dict["intermediates"]['x_prev_range_min']).squeeze(2).transpose(1,0)
                # x_prev_range_max = np.array(log_dict["intermediates"]['x_prev_range_max']).squeeze(2).transpose(1,0)
                x_prev_range_min = np.array(log_dict["intermediates"]['x_prev_range_min']).squeeze(2).transpose(1,0)
                x_prev_range_max = np.array(log_dict["intermediates"]['x_prev_range_max']).squeeze(2).transpose(1,0)

                print('pred_x0 range shape', pred_x0_range_max.shape)
                print('x_prev range shape', x_prev_range_min.shape)


                latent_max = 6 * scale_factor
                latent_min = -6 * scale_factor

                plt.figure(figsize=(20,10))
                for i in range(pred_x0_range_min.shape[0]):
                    pred_x0_min_list = pred_x0_range_min[i]
                    pred_x0_max_list = pred_x0_range_max[i]
                    x_prev_min_list = x_prev_range_min[i]
                    x_prev_max_list = x_prev_range_max[i]

                    plt.subplot(121)
                    plt.plot(range(len(pred_x0_min_list)), pred_x0_min_list, label="Ours Diffusion pred_x0 Min Value {}".format(i))
                    plt.plot(range(len(pred_x0_max_list)), pred_x0_max_list, label="Ours Diffusion pred_x0 Max Value {}".format(i)) 

                    plt.subplot(122)
                    plt.plot(range(len(x_prev_min_list)), x_prev_min_list, label="Ours Diffusion x_prev Min Value {}".format(i))
                    plt.plot(range(len(x_prev_max_list)), x_prev_max_list, label="Ours Diffusion x_prev Max Value {}".format(i))

                plt.subplot(121)
                plt.hlines(latent_max,xmin=0,xmax=len(x_prev_min_list),color='red', label='Latent Max Range / scale_factor:{}'.format(scale_factor),linewidth=5)
                plt.hlines(latent_min,xmin=0,xmax=len(x_prev_min_list),color='red', label='Latent min Range / scale_factor:{}'.format(scale_factor),linewidth=5)
                # plt.ylim((-10,10))
                plt.legend()
                plt.subplot(122)
                plt.hlines(latent_max,xmin=0,xmax=len(x_prev_min_list),color='red', label='Latent Max Range / scale_factor:{}'.format(scale_factor),linewidth=5)
                plt.hlines(latent_min,xmin=0,xmax=len(x_prev_min_list),color='red', label='Latent min Range / scale_factor:{}'.format(scale_factor),linewidth=5)
                # plt.ylim((-10,10))
                plt.legend()
                plt.savefig(os.path.join(root, "sampling_range_new1.png"))
            except Exception as e:
                print(e)

    def concat_frame_video(self, video_path, video_time):
        # start_frame, end_frame = video_time[0], video_time[1]
        start_frame, end_frame = int(video_time.split('_')[0]), int(video_time.split('_')[1])
        frame_list = []
        for i in range(start_frame, end_frame):
            try:
                frame_path = os.path.join(video_path, '{}.png'.format(i+1))
                frame = np.array(Image.open(frame_path).convert('RGB'))
                frame_list.append(frame)
            except:
                pass
        return frame_list

        

    def log_sound_steps(self, pl_module, batch, batch_idx, split="train"):        
        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        with torch.no_grad():
            log_dict = pl_module.log_sound(batch,N=self.max_sound_num, ddim_steps=self.ddim_step, split=split, size_len=self.size_len, guidance_scale=self.guidance_scale, uncond_cond=self.uncond_cond)

        # gt_sound_list = log_dict['inputs']
        # rec_sound_list = log_dict['reconstruction']
        # video_path_list= log_dict['video_frame_path']
        # video_time_list = log_dict['video_time']

        self.log_local(pl_module.logger.save_dir, split, log_dict, pl_module.global_step, pl_module.current_epoch, batch_idx)

        if is_train:
            pl_module.train()



    def check_frequency(self, check_idx, split):
        
        if split == "train":
            if ((check_idx % self.train_batch_freq) == 0 or (check_idx in self.log_steps)) and (
                    check_idx > 0 or self.log_first_step):
                try:
                    self.log_steps.pop(0)
                except IndexError as e:
                    print(e)
                    pass
                return True
            return False
        else:
            if (check_idx % self.val_batch_freq) == 0:
                return True
            else:
                return False


    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        # if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
        if not self.disabled and (pl_module.global_step % self.train_vis_frequency == 0) and pl_module.global_step > 0:
            self.log_sound_steps(pl_module, batch, batch_idx, split="train")
            # pass


    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled and (pl_module.global_step % self.val_vis_frequency == 0) and pl_module.global_step >= 0:
            self.log_sound_steps(pl_module, batch, batch_idx, split="val")
            # pass

        if hasattr(pl_module, 'calibrate_grad_norm'):
            if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)





# Concat Mix Logger:

class SoundLogger_concat(Callback):
    def __init__(self, train_batch_frequency,val_batch_frequency, max_sound_num, sr=22050, clamp=True, increase_log_steps=False,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False, ddim_step=250, size_len=64, guidance_scale=1.0, uncond_cond=None):
        super().__init__()
        self.rescale = rescale
        self.train_batch_freq = train_batch_frequency
        self.val_batch_freq = val_batch_frequency
        self.max_sound_num = max_sound_num
        self.log_steps = [2 ** n for n in range(int(np.log2(self.train_batch_freq)) + 1)]
        self.log_steps_val = [2 ** n for n in range(int(np.log2(self.val_batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.train_batch_freq]
            self.log_steps_val = [self.val_batch_freq] 
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_first_step = log_first_step
        self.sr = sr
        print('sr: ', self.sr)
        self.ddim_step = ddim_step

        self.train_vis_frequency = train_batch_frequency
        self.val_vis_frequency = val_batch_frequency
        self.size_len = size_len
        self.guidance_scale = guidance_scale
        self.uncond_cond = uncond_cond
        print("Guidance Scale: ", self.guidance_scale)
        print("Uncond cond: ", self.uncond_cond)
    
    
    def inverse_op(self, spec):
        sr = self.sr
        n_fft = 1024
        fmin = 125
        fmax = 7600
        nmels = 128
        hoplen = 1024 // 4
        spec_power = 1

        # Inverse Transform
        spec = spec * 100 - 100
        spec = (spec + 20) / 20
        spec = 10 ** spec
        spec_out = librosa.feature.inverse.mel_to_stft(spec, sr = sr, n_fft = n_fft, fmin=fmin, fmax=fmax, power=spec_power)
        wav = librosa.griffinlim(spec_out, hop_length=hoplen)
        return wav

    @rank_zero_only
    def log_local(self, save_dir, split, log_dict,
                  global_step, current_epoch, batch_idx, show_curve=False, scale_factor=1):

        # root:
        root = os.path.join(save_dir, "sound_eval", split, "epoch_{}_step_{}".format(current_epoch, global_step))

        gt_sound_list = log_dict['inputs_spec'].detach().cpu().numpy()
        rec_sound_list = log_dict['reconstruction_spec'].detach().cpu().numpy()
        # video_path_list= log_dict['video_frame_path']
        # video_time_list = log_dict['video_time']

        diff_sample_list = log_dict['samples'].detach().cpu().numpy()

        # root = os.path.join(save_dir, "sound", split, "epoch_{}_step_{}".format(current_epoch, global_step))

        os.makedirs(root,exist_ok=True)
        
        mix_info_dict = log_dict["mix_info_dict"]

        for i in range(len(gt_sound_list)):
            
            if mix_info_dict['audio_name2'][i] == "":
                video_path_list = mix_info_dict['video_path1']
                video_time_list = mix_info_dict['video_time1'] 
                print('Gen examples ===========> {}'.format(i))
                sample_folder = os.path.join(root, "sample_{}".format(i))
                os.makedirs(sample_folder, exist_ok=True)
                gt_sound = self.inverse_op(gt_sound_list[i])
                rec_sound = self.inverse_op(rec_sound_list[i])
                sf.write(os.path.join(sample_folder, "sample_{}_gt.wav".format(i)), gt_sound, self.sr)
                sf.write(os.path.join(sample_folder, "sample_{}_stage1_rec_clamp.wav".format(i)), rec_sound, self.sr)
                sample = self.inverse_op(diff_sample_list[i])
                sf.write(os.path.join(sample_folder, "sample_{}_diff_sample_clamp.wav".format(i)), sample, self.sr) 
                video = self.concat_frame_video(video_path_list[i], video_time_list[i])
                video_save_path = os.path.join(sample_folder, "origin_video.mp4")
                imageio.mimsave(video_save_path, video, fps=21.5)

                with open(os.path.join(sample_folder, "video_path.txt"), "w") as f:
                    txt = "Video 1:" + '  ' + str(video_path_list[i]) + "    " + str(video_time_list[i])
                    f.writelines(txt)
            
            else:

                video_path_list1, video_path_list2 = mix_info_dict['video_path1'], mix_info_dict['video_path2']
                video_time_list1, video_time_list2 = mix_info_dict['video_time1'], mix_info_dict['video_time2'] 

                print('Gen examples ===========> {}'.format(i))
                sample_folder = os.path.join(root, "sample_{}".format(i))
                os.makedirs(sample_folder, exist_ok=True)
                gt_sound = self.inverse_op(gt_sound_list[i])
                rec_sound = self.inverse_op(rec_sound_list[i])
                sf.write(os.path.join(sample_folder, "sample_{}_concat_gt.wav".format(i)), gt_sound, self.sr)
                sf.write(os.path.join(sample_folder, "sample_{}_stage1_concat_rec_clamp.wav".format(i)), rec_sound, self.sr)
                sample = self.inverse_op(diff_sample_list[i])
                sf.write(os.path.join(sample_folder, "sample_{}_diff_concat_sample_clamp.wav".format(i)), sample, self.sr) 
                video1 = self.concat_frame_video(video_path_list1[i], video_time_list1[i])
                video2 = self.concat_frame_video(video_path_list2[i], video_time_list2[i])
                video = video1 + video2
                video_save_path = os.path.join(sample_folder, "origin_concat_video.mp4")
                imageio.mimsave(video_save_path, video, fps=21.5)

                with open(os.path.join(sample_folder, "video_path.txt"), "w") as f:
                    txt = "Video 1:" + '  ' + str(video_path_list1[i]) + "    " + str(video_time_list1[i]) + '\n' + "Video 2:" + '  ' + str(video_path_list2[i]) + "    " + str(video_time_list2[i])
                    f.writelines(txt)
        

                # show curve:
        if show_curve: 
            try: 
                pred_x0_range_min = np.array(log_dict["intermediates"]["pred_x0_range_min"]).squeeze(2).transpose(1,0)     #  sample_num x T
                pred_x0_range_max = np.array(log_dict["intermediates"]["pred_x0_range_max"]).squeeze(2).transpose(1,0)     #  sample_num x T 
                # import pdb
                # pdb.set_trace()
                # x_prev_range_min = np.array(log_dict["intermediates"]['x_prev_range_min']).squeeze(2).transpose(1,0)
                # x_prev_range_max = np.array(log_dict["intermediates"]['x_prev_range_max']).squeeze(2).transpose(1,0)
                x_prev_range_min = np.array(log_dict["intermediates"]['x_prev_range_min']).squeeze(2).transpose(1,0)
                x_prev_range_max = np.array(log_dict["intermediates"]['x_prev_range_max']).squeeze(2).transpose(1,0)

                print('pred_x0 range shape', pred_x0_range_max.shape)
                print('x_prev range shape', x_prev_range_min.shape)


                latent_max = 6 * scale_factor
                latent_min = -6 * scale_factor

                plt.figure(figsize=(20,10))
                for i in range(pred_x0_range_min.shape[0]):
                    pred_x0_min_list = pred_x0_range_min[i]
                    pred_x0_max_list = pred_x0_range_max[i]
                    x_prev_min_list = x_prev_range_min[i]
                    x_prev_max_list = x_prev_range_max[i]

                    plt.subplot(121)
                    plt.plot(range(len(pred_x0_min_list)), pred_x0_min_list, label="Ours Diffusion pred_x0 Min Value {}".format(i))
                    plt.plot(range(len(pred_x0_max_list)), pred_x0_max_list, label="Ours Diffusion pred_x0 Max Value {}".format(i)) 

                    plt.subplot(122)
                    plt.plot(range(len(x_prev_min_list)), x_prev_min_list, label="Ours Diffusion x_prev Min Value {}".format(i))
                    plt.plot(range(len(x_prev_max_list)), x_prev_max_list, label="Ours Diffusion x_prev Max Value {}".format(i))

                plt.subplot(121)
                plt.hlines(latent_max,xmin=0,xmax=len(x_prev_min_list),color='red', label='Latent Max Range / scale_factor:{}'.format(scale_factor),linewidth=5)
                plt.hlines(latent_min,xmin=0,xmax=len(x_prev_min_list),color='red', label='Latent min Range / scale_factor:{}'.format(scale_factor),linewidth=5)
                # plt.ylim((-10,10))
                plt.legend()
                plt.subplot(122)
                plt.hlines(latent_max,xmin=0,xmax=len(x_prev_min_list),color='red', label='Latent Max Range / scale_factor:{}'.format(scale_factor),linewidth=5)
                plt.hlines(latent_min,xmin=0,xmax=len(x_prev_min_list),color='red', label='Latent min Range / scale_factor:{}'.format(scale_factor),linewidth=5)
                # plt.ylim((-10,10))
                plt.legend()
                plt.savefig(os.path.join(root, "sampling_range_new1.png"))
            except Exception as e:
                print(e)

    def concat_frame_video(self, video_path, video_time):
        # start_frame, end_frame = video_time[0], video_time[1]
        start_frame, end_frame = int(video_time.split('_')[0]), int(video_time.split('_')[1])
        frame_list = []
        for i in range(start_frame, end_frame):
            try:
                frame_path = os.path.join(video_path, '{}.png'.format(i+1))
                frame = np.array(Image.open(frame_path).convert('RGB'))
                frame_list.append(frame)
            except:
                pass
        return frame_list

        

    def log_sound_steps(self, pl_module, batch, batch_idx, split="train"):        
        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        with torch.no_grad():
            log_dict = pl_module.log_sound(batch,N=self.max_sound_num, ddim_steps=self.ddim_step, split=split, size_len=self.size_len, guidance_scale=self.guidance_scale, uncond_cond=self.uncond_cond)

        # gt_sound_list = log_dict['inputs']
        # rec_sound_list = log_dict['reconstruction']
        # video_path_list= log_dict['video_frame_path']
        # video_time_list = log_dict['video_time']

        self.log_local(pl_module.logger.save_dir, split, log_dict, pl_module.global_step, pl_module.current_epoch, batch_idx)

        if is_train:
            pl_module.train()



    def check_frequency(self, check_idx, split):
        if split == "train":
            if ((check_idx % self.train_batch_freq) == 0 or (check_idx in self.log_steps)) and (
                    check_idx > 0 or self.log_first_step):
                try:
                    self.log_steps.pop(0)
                except IndexError as e:
                    print(e)
                    pass
                return True
            return False
        else:
            if (check_idx % self.val_batch_freq) == 0:
                return True
            else:
                return False


    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        # if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
        if not self.disabled and (pl_module.global_step % self.train_vis_frequency == 0) and pl_module.global_step > 0:
            self.log_sound_steps(pl_module, batch, batch_idx, split="train")
            # pass


    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled and (pl_module.global_step % self.val_vis_frequency == 0) and pl_module.global_step > 0:
            self.log_sound_steps(pl_module, batch, batch_idx, split="val")
            # pass

        if hasattr(pl_module, 'calibrate_grad_norm'):
            if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)




class SoundLogger_fullset(Callback):
    def __init__(self, train_batch_frequency,val_batch_frequency, max_sound_num, sr=22050, clamp=True, increase_log_steps=False,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False, ddim_step=250, size_len=64, guidance_scale=1.0, uncond_cond=None, fps=21.5):
        super().__init__()
        self.rescale = rescale
        self.train_batch_freq = train_batch_frequency
        self.val_batch_freq = val_batch_frequency
        self.max_sound_num = max_sound_num
        self.log_steps = [2 ** n for n in range(int(np.log2(self.train_batch_freq)) + 1)]
        self.log_steps_val = [2 ** n for n in range(int(np.log2(self.val_batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.train_batch_freq]
            self.log_steps_val = [self.val_batch_freq] 
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_first_step = log_first_step
        self.sr = sr
        print('sr: ', self.sr)
        self.ddim_step = ddim_step

        self.train_vis_frequency = train_batch_frequency
        self.val_vis_frequency = val_batch_frequency
        self.size_len = size_len
        self.guidance_scale = guidance_scale
        self.uncond_cond = uncond_cond
        self.fps = fps
        print("Guidance Scale: ", self.guidance_scale)
        print("Uncond cond: ", self.uncond_cond)
    
    
    def inverse_op(self, spec):
        sr = self.sr
        n_fft = 1024
        fmin = 125
        fmax = 7600
        nmels = 128
        hoplen = 1024 // 4
        spec_power = 1

        # Inverse Transform
        spec = spec * 100 - 100
        spec = (spec + 20) / 20
        spec = 10 ** spec
        spec_out = librosa.feature.inverse.mel_to_stft(spec, sr = sr, n_fft = n_fft, fmin=fmin, fmax=fmax, power=spec_power)
        wav = librosa.griffinlim(spec_out, hop_length=hoplen)
        return wav

    @rank_zero_only
    def log_local(self, save_dir, split, log_dict,
                  global_step, current_epoch, batch_idx, show_curve=False, scale_factor=1):

        # root:
        root = os.path.join(save_dir, "sound_eval", split, "epoch_{}_step_{}".format(current_epoch, global_step))

        gt_sound_list = log_dict['inputs_spec'].detach().cpu().numpy()
        rec_sound_list = log_dict['reconstruction_spec'].detach().cpu().numpy()
        video_path_list= log_dict['video_frame_path']
        video_time_list = log_dict['video_time']
        diff_sample_list = log_dict['samples'].detach().cpu().numpy()

        # root = os.path.join(save_dir, "sound", split, "epoch_{}_step_{}".format(current_epoch, global_step))

        os.makedirs(root,exist_ok=True)
        

        for i in range(len(gt_sound_list)):
            print('Gen examples ===========> {}'.format(i))
            sample_folder = os.path.join(root, "sample_{}".format(i))
            os.makedirs(sample_folder, exist_ok=True)
            gt_sound = self.inverse_op(gt_sound_list[i])
            rec_sound = self.inverse_op(rec_sound_list[i])
            sf.write(os.path.join(sample_folder, "sample_{}_gt.wav".format(i)), gt_sound, self.sr)
            sf.write(os.path.join(sample_folder, "sample_{}_stage1_rec_clamp.wav".format(i)), rec_sound, self.sr)
            sample = self.inverse_op(diff_sample_list[i])
            sf.write(os.path.join(sample_folder, "sample_{}_diff_sample_clamp.wav".format(i)), sample, self.sr) 
            try:
                video = self.extract_frame_video(video_path_list[i], video_time_list[i], sample_folder)
            except:
                pass
            # video_save_path = os.path.join(sample_folder, "origin_video.mp4")
            # imageio.mimsave(video_save_path, video, fps=21.5)

                # show curve:
        if show_curve: 
            try: 
                pred_x0_range_min = np.array(log_dict["intermediates"]["pred_x0_range_min"]).squeeze(2).transpose(1,0)     #  sample_num x T
                pred_x0_range_max = np.array(log_dict["intermediates"]["pred_x0_range_max"]).squeeze(2).transpose(1,0)     #  sample_num x T 
                # import pdb
                # pdb.set_trace()
                # x_prev_range_min = np.array(log_dict["intermediates"]['x_prev_range_min']).squeeze(2).transpose(1,0)
                # x_prev_range_max = np.array(log_dict["intermediates"]['x_prev_range_max']).squeeze(2).transpose(1,0)
                x_prev_range_min = np.array(log_dict["intermediates"]['x_prev_range_min']).squeeze(2).transpose(1,0)
                x_prev_range_max = np.array(log_dict["intermediates"]['x_prev_range_max']).squeeze(2).transpose(1,0)

                print('pred_x0 range shape', pred_x0_range_max.shape)
                print('x_prev range shape', x_prev_range_min.shape)


                latent_max = 6 * scale_factor
                latent_min = -6 * scale_factor

                plt.figure(figsize=(20,10))
                for i in range(pred_x0_range_min.shape[0]):
                    pred_x0_min_list = pred_x0_range_min[i]
                    pred_x0_max_list = pred_x0_range_max[i]
                    x_prev_min_list = x_prev_range_min[i]
                    x_prev_max_list = x_prev_range_max[i]

                    plt.subplot(121)
                    plt.plot(range(len(pred_x0_min_list)), pred_x0_min_list, label="Ours Diffusion pred_x0 Min Value {}".format(i))
                    plt.plot(range(len(pred_x0_max_list)), pred_x0_max_list, label="Ours Diffusion pred_x0 Max Value {}".format(i)) 

                    plt.subplot(122)
                    plt.plot(range(len(x_prev_min_list)), x_prev_min_list, label="Ours Diffusion x_prev Min Value {}".format(i))
                    plt.plot(range(len(x_prev_max_list)), x_prev_max_list, label="Ours Diffusion x_prev Max Value {}".format(i))

                plt.subplot(121)
                plt.hlines(latent_max,xmin=0,xmax=len(x_prev_min_list),color='red', label='Latent Max Range / scale_factor:{}'.format(scale_factor),linewidth=5)
                plt.hlines(latent_min,xmin=0,xmax=len(x_prev_min_list),color='red', label='Latent min Range / scale_factor:{}'.format(scale_factor),linewidth=5)
                # plt.ylim((-10,10))
                plt.legend()
                plt.subplot(122)
                plt.hlines(latent_max,xmin=0,xmax=len(x_prev_min_list),color='red', label='Latent Max Range / scale_factor:{}'.format(scale_factor),linewidth=5)
                plt.hlines(latent_min,xmin=0,xmax=len(x_prev_min_list),color='red', label='Latent min Range / scale_factor:{}'.format(scale_factor),linewidth=5)
                # plt.ylim((-10,10))
                plt.legend()
                plt.savefig(os.path.join(root, "sampling_range_new1.png"))
            except Exception as e:
                print(e)

    def extract_frame_video(self, video_path, video_time, out_folder):
        # start_frame, end_frame = video_time[0], video_time[1]
        start_frame, end_frame = int(video_time.split('_')[0]), int(video_time.split('_')[1])
        start_time, end_time = start_frame / self.fps, end_frame / self.fps
        src_path = video_path
        out_path = os.path.join(out_folder, "origin.mp4")
        ffmpeg_extract_subclip(src_path, start_time, end_time, out_path)
        return True

        

    def log_sound_steps(self, pl_module, batch, batch_idx, split="train"):        
        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        with torch.no_grad():
            log_dict = pl_module.log_sound(batch,N=self.max_sound_num, ddim_steps=self.ddim_step, split=split, size_len=self.size_len, guidance_scale=self.guidance_scale, uncond_cond=self.uncond_cond)

        # gt_sound_list = log_dict['inputs']
        # rec_sound_list = log_dict['reconstruction']
        # video_path_list= log_dict['video_frame_path']
        # video_time_list = log_dict['video_time']

        self.log_local(pl_module.logger.save_dir, split, log_dict, pl_module.global_step, pl_module.current_epoch, batch_idx)

        if is_train:
            pl_module.train()



    def check_frequency(self, check_idx, split):
        
        if split == "train":
            if ((check_idx % self.train_batch_freq) == 0 or (check_idx in self.log_steps)) and (
                    check_idx > 0 or self.log_first_step):
                try:
                    self.log_steps.pop(0)
                except IndexError as e:
                    print(e)
                    pass
                return True
            return False
        else:
            if (check_idx % self.val_batch_freq) == 0:
                return True
            else:
                return False


    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        # if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
        if not self.disabled and (pl_module.global_step % self.train_vis_frequency == 0) and pl_module.global_step > 0:
            self.log_sound_steps(pl_module, batch, batch_idx, split="train")
            # pass


    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled and (pl_module.global_step % self.val_vis_frequency == 0) and pl_module.global_step > 0:
            self.log_sound_steps(pl_module, batch, batch_idx, split="val")
            # pass

        if hasattr(pl_module, 'calibrate_grad_norm'):
            if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)



class SoundLogger_movie_full(Callback):
    def __init__(self, train_batch_frequency,val_batch_frequency, max_sound_num, sr=22050, clamp=True, increase_log_steps=False,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False, ddim_step=250, size_len=64, guidance_scale=1.0, uncond_cond=None, fps=23.98):
        super().__init__()
        self.rescale = rescale
        self.train_batch_freq = train_batch_frequency
        self.val_batch_freq = val_batch_frequency
        self.max_sound_num = max_sound_num
        self.log_steps = [2 ** n for n in range(int(np.log2(self.train_batch_freq)) + 1)]
        self.log_steps_val = [2 ** n for n in range(int(np.log2(self.val_batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.train_batch_freq]
            self.log_steps_val = [self.val_batch_freq] 
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_first_step = log_first_step
        self.sr = sr
        print('sr: ', self.sr)
        self.ddim_step = ddim_step

        self.train_vis_frequency = train_batch_frequency
        self.val_vis_frequency = val_batch_frequency
        self.size_len = size_len
        self.guidance_scale = guidance_scale
        self.uncond_cond = uncond_cond
        self.fps = fps
        print("Guidance Scale: ", self.guidance_scale)
        print("Uncond cond: ", self.uncond_cond)
    
    
    def inverse_op(self, spec):
        sr = self.sr
        n_fft = 1024
        fmin = 125
        fmax = 7600
        nmels = 128
        hoplen = 1024 // 4
        spec_power = 1

        # Inverse Transform
        spec = spec * 100 - 100
        spec = (spec + 20) / 20
        spec = 10 ** spec
        spec_out = librosa.feature.inverse.mel_to_stft(spec, sr = sr, n_fft = n_fft, fmin=fmin, fmax=fmax, power=spec_power)
        wav = librosa.griffinlim(spec_out, hop_length=hoplen)
        return wav

    @rank_zero_only
    def log_local(self, save_dir, split, log_dict,
                  global_step, current_epoch, batch_idx, show_curve=False, scale_factor=1):

        # root:
        root = os.path.join(save_dir, "sound_eval", split, "epoch_{}_step_{}".format(current_epoch, global_step))

        gt_sound_list = log_dict['inputs_spec'].detach().cpu().numpy()
        rec_sound_list = log_dict['reconstruction_spec'].detach().cpu().numpy()
        video_path_list= log_dict['video_frame_path']
        video_time_list = log_dict['video_time']
        diff_sample_list = log_dict['samples'].detach().cpu().numpy()

        # root = os.path.join(save_dir, "sound", split, "epoch_{}_step_{}".format(current_epoch, global_step))

        os.makedirs(root,exist_ok=True)
        

        for i in range(len(gt_sound_list)):
            print('Gen examples ===========> {}'.format(i))
            sample_folder = os.path.join(root, "sample_{}".format(i))
            os.makedirs(sample_folder, exist_ok=True)
            gt_sound = self.inverse_op(gt_sound_list[i])
            rec_sound = self.inverse_op(rec_sound_list[i])
            sf.write(os.path.join(sample_folder, "sample_{}_gt.wav".format(i)), gt_sound, self.sr)
            sf.write(os.path.join(sample_folder, "sample_{}_stage1_rec_clamp.wav".format(i)), rec_sound, self.sr)
            sample = self.inverse_op(diff_sample_list[i])
            sf.write(os.path.join(sample_folder, "sample_{}_diff_sample_clamp.wav".format(i)), sample, self.sr) 
            try:
                video = self.extract_frame_video(video_path_list[i], video_time_list[i], sample_folder)
            except:
                pass
            # video_save_path = os.path.join(sample_folder, "origin_video.mp4")
            # imageio.mimsave(video_save_path, video, fps=21.5)

                # show curve:
        if show_curve: 
            try: 
                pred_x0_range_min = np.array(log_dict["intermediates"]["pred_x0_range_min"]).squeeze(2).transpose(1,0)     #  sample_num x T
                pred_x0_range_max = np.array(log_dict["intermediates"]["pred_x0_range_max"]).squeeze(2).transpose(1,0)     #  sample_num x T 
                # import pdb
                # pdb.set_trace()
                # x_prev_range_min = np.array(log_dict["intermediates"]['x_prev_range_min']).squeeze(2).transpose(1,0)
                # x_prev_range_max = np.array(log_dict["intermediates"]['x_prev_range_max']).squeeze(2).transpose(1,0)
                x_prev_range_min = np.array(log_dict["intermediates"]['x_prev_range_min']).squeeze(2).transpose(1,0)
                x_prev_range_max = np.array(log_dict["intermediates"]['x_prev_range_max']).squeeze(2).transpose(1,0)

                print('pred_x0 range shape', pred_x0_range_max.shape)
                print('x_prev range shape', x_prev_range_min.shape)


                latent_max = 6 * scale_factor
                latent_min = -6 * scale_factor

                plt.figure(figsize=(20,10))
                for i in range(pred_x0_range_min.shape[0]):
                    pred_x0_min_list = pred_x0_range_min[i]
                    pred_x0_max_list = pred_x0_range_max[i]
                    x_prev_min_list = x_prev_range_min[i]
                    x_prev_max_list = x_prev_range_max[i]

                    plt.subplot(121)
                    plt.plot(range(len(pred_x0_min_list)), pred_x0_min_list, label="Ours Diffusion pred_x0 Min Value {}".format(i))
                    plt.plot(range(len(pred_x0_max_list)), pred_x0_max_list, label="Ours Diffusion pred_x0 Max Value {}".format(i)) 

                    plt.subplot(122)
                    plt.plot(range(len(x_prev_min_list)), x_prev_min_list, label="Ours Diffusion x_prev Min Value {}".format(i))
                    plt.plot(range(len(x_prev_max_list)), x_prev_max_list, label="Ours Diffusion x_prev Max Value {}".format(i))

                plt.subplot(121)
                plt.hlines(latent_max,xmin=0,xmax=len(x_prev_min_list),color='red', label='Latent Max Range / scale_factor:{}'.format(scale_factor),linewidth=5)
                plt.hlines(latent_min,xmin=0,xmax=len(x_prev_min_list),color='red', label='Latent min Range / scale_factor:{}'.format(scale_factor),linewidth=5)
                # plt.ylim((-10,10))
                plt.legend()
                plt.subplot(122)
                plt.hlines(latent_max,xmin=0,xmax=len(x_prev_min_list),color='red', label='Latent Max Range / scale_factor:{}'.format(scale_factor),linewidth=5)
                plt.hlines(latent_min,xmin=0,xmax=len(x_prev_min_list),color='red', label='Latent min Range / scale_factor:{}'.format(scale_factor),linewidth=5)
                # plt.ylim((-10,10))
                plt.legend()
                plt.savefig(os.path.join(root, "sampling_range_new1.png"))
            except Exception as e:
                print(e)

    def extract_frame_video(self, video_path, video_time, out_folder):
        # start_frame, end_frame = video_time[0], video_time[1]
        start_frame, end_frame = int(video_time.split('_')[0]), int(video_time.split('_')[1])
        start_time, end_time = start_frame / self.fps, end_frame / self.fps
        src_path = video_path
        out_path = os.path.join(out_folder, "origin.mp4")
        ffmpeg_extract_subclip(src_path, start_time, end_time, out_path)
        return True

        
    @rank_zero_only
    def log_sound_steps(self, pl_module, batch, batch_idx, split="train"):        
        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        with torch.no_grad():
            log_dict = pl_module.log_sound(batch,N=self.max_sound_num, ddim_steps=self.ddim_step, split=split, size_len=self.size_len, guidance_scale=self.guidance_scale, uncond_cond=self.uncond_cond)

        # gt_sound_list = log_dict['inputs']
        # rec_sound_list = log_dict['reconstruction']
        # video_path_list= log_dict['video_frame_path']
        # video_time_list = log_dict['video_time']

        self.log_local(pl_module.logger.save_dir, split, log_dict, pl_module.global_step, pl_module.current_epoch, batch_idx)

        if is_train:
            pl_module.train()



    def check_frequency(self, check_idx, split):
        
        if split == "train":
            if ((check_idx % self.train_batch_freq) == 0 or (check_idx in self.log_steps)) and (
                    check_idx > 0 or self.log_first_step):
                try:
                    self.log_steps.pop(0)
                except IndexError as e:
                    print(e)
                    pass
                return True
            return False
        else:
            if (check_idx % self.val_batch_freq) == 0:
                return True
            else:
                return False


    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        # if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
        if not self.disabled and (pl_module.global_step % self.train_vis_frequency == 0) and pl_module.global_step > 0:
            self.log_sound_steps(pl_module, batch, batch_idx, split="train")
            # pass


    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled and (pl_module.global_step % self.val_vis_frequency == 0) and pl_module.global_step > 0:
            self.log_sound_steps(pl_module, batch, batch_idx, split="val")
            # pass

        if hasattr(pl_module, 'calibrate_grad_norm'):
            if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)

    




class SoundLogger_concat_fullset(Callback):
    def __init__(self, train_batch_frequency,val_batch_frequency, max_sound_num, sr=22050, clamp=True, increase_log_steps=False,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False, ddim_step=250, size_len=64, guidance_scale=1.0, uncond_cond=None, fps=21.5):
        super().__init__()
        self.fps = fps
        self.rescale = rescale
        self.train_batch_freq = train_batch_frequency
        self.val_batch_freq = val_batch_frequency
        self.max_sound_num = max_sound_num
        self.log_steps = [2 ** n for n in range(int(np.log2(self.train_batch_freq)) + 1)]
        self.log_steps_val = [2 ** n for n in range(int(np.log2(self.val_batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.train_batch_freq]
            self.log_steps_val = [self.val_batch_freq] 
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_first_step = log_first_step
        self.sr = sr
        print('sr: ', self.sr)
        self.ddim_step = ddim_step

        self.train_vis_frequency = train_batch_frequency
        self.val_vis_frequency = val_batch_frequency
        self.size_len = size_len
        self.guidance_scale = guidance_scale
        self.uncond_cond = uncond_cond
        print("Guidance Scale: ", self.guidance_scale)
        print("Uncond cond: ", self.uncond_cond)
    
    
    def inverse_op(self, spec):
        sr = self.sr
        n_fft = 1024
        fmin = 125
        fmax = 7600
        nmels = 128
        hoplen = 1024 // 4
        spec_power = 1

        # Inverse Transform
        spec = spec * 100 - 100
        spec = (spec + 20) / 20
        spec = 10 ** spec
        spec_out = librosa.feature.inverse.mel_to_stft(spec, sr = sr, n_fft = n_fft, fmin=fmin, fmax=fmax, power=spec_power)
        wav = librosa.griffinlim(spec_out, hop_length=hoplen)
        return wav

    @rank_zero_only
    def log_local(self, save_dir, split, log_dict,
                  global_step, current_epoch, batch_idx, show_curve=False, scale_factor=1):

        # root:
        root = os.path.join(save_dir, "sound_eval", split, "epoch_{}_step_{}".format(current_epoch, global_step))

        gt_sound_list = log_dict['inputs_spec'].detach().cpu().numpy()
        rec_sound_list = log_dict['reconstruction_spec'].detach().cpu().numpy()
        # video_path_list= log_dict['video_frame_path']
        # video_time_list = log_dict['video_time']

        diff_sample_list = log_dict['samples'].detach().cpu().numpy()

        # root = os.path.join(save_dir, "sound", split, "epoch_{}_step_{}".format(current_epoch, global_step))

        os.makedirs(root,exist_ok=True)
        
        mix_info_dict = log_dict["mix_info_dict"]

        for i in range(len(gt_sound_list)):
            
            if mix_info_dict['audio_name2'][i] == "":
                video_path_list = mix_info_dict['video_path1']
                video_time_list = mix_info_dict['video_time1'] 
                print('Gen examples ===========> {}'.format(i))
                sample_folder = os.path.join(root, "sample_{}".format(i))
                os.makedirs(sample_folder, exist_ok=True)
                gt_sound = self.inverse_op(gt_sound_list[i])
                rec_sound = self.inverse_op(rec_sound_list[i])
                sf.write(os.path.join(sample_folder, "sample_{}_gt.wav".format(i)), gt_sound, self.sr)
                sf.write(os.path.join(sample_folder, "sample_{}_stage1_rec_clamp.wav".format(i)), rec_sound, self.sr)
                sample = self.inverse_op(diff_sample_list[i])
                sf.write(os.path.join(sample_folder, "sample_{}_diff_sample_clamp.wav".format(i)), sample, self.sr) 
                
                # video = self.concat_frame_video(video_path_list[i], video_time_list[i])
                # video_save_path = os.path.join(sample_folder, "origin_video.mp4")
                # imageio.mimsave(video_save_path, video, fps=21.5)

                try:
                    video = self.extract_concat_frame_video(video_path_list[i], video_time_list[i], out_folder=sample_folder)
                except Exception as e:
                    print(e)
                    pass

                with open(os.path.join(sample_folder, "video_path.txt"), "w") as f:
                    txt = "Video 1:" + '  ' + str(video_path_list[i]) + "    " + str(video_time_list[i])
                    f.writelines(txt)
            
            else:

                video_path_list1, video_path_list2 = mix_info_dict['video_path1'], mix_info_dict['video_path2']
                video_time_list1, video_time_list2 = mix_info_dict['video_time1'], mix_info_dict['video_time2'] 

                print('Gen examples ===========> {}'.format(i))
                sample_folder = os.path.join(root, "sample_{}".format(i))
                os.makedirs(sample_folder, exist_ok=True)
                gt_sound = self.inverse_op(gt_sound_list[i])
                rec_sound = self.inverse_op(rec_sound_list[i])
                sf.write(os.path.join(sample_folder, "sample_{}_concat_gt.wav".format(i)), gt_sound, self.sr)
                sf.write(os.path.join(sample_folder, "sample_{}_stage1_concat_rec_clamp.wav".format(i)), rec_sound, self.sr)
                sample = self.inverse_op(diff_sample_list[i])
                sf.write(os.path.join(sample_folder, "sample_{}_diff_concat_sample_clamp.wav".format(i)), sample, self.sr) 
                
                # video1 = self.concat_frame_video(video_path_list1[i], video_time_list1[i])
                # video2 = self.concat_frame_video(video_path_list2[i], video_time_list2[i])
                # video = video1 + video2
                # video_save_path = os.path.join(sample_folder, "origin_concat_video.mp4")
                # imageio.mimsave(video_save_path, video, fps=21.5)
                try:
                    video = self.extract_concat_frame_video(video_path_list1[i], video_time_list1[i], video_path_list2[i], video_time_list2[i], out_folder=sample_folder)
                except:
                    pass

                with open(os.path.join(sample_folder, "video_path_cat.txt"), "w") as f:
                    txt = "Video 1:" + '  ' + str(video_path_list1[i]) + "    " + str(video_time_list1[i]) + '\n' + "Video 2:" + '  ' + str(video_path_list2[i]) + "    " + str(video_time_list2[i])
                    f.writelines(txt)
        

    def extract_concat_frame_video(self, video_path1, video_time1, video_path2=None, video_time2=None, out_folder=None):
        # start_frame, end_frame = video_time[0], video_time[1]
        start_frame1, end_frame1 = int(video_time1.split('_')[0]), int(video_time1.split('_')[1])
        start_time1, end_time1 = start_frame1 / self.fps, end_frame1 / self.fps
        src_path1 = video_path1
        out_path = os.path.join(out_folder, "origin.mp4")

        video1 = VideoFileClip(src_path1).subclip(start_time1, end_time1)

        if video_path2 is not None:
            start_frame2, end_frame2 = int(video_time2.split('_')[0]), int(video_time2.split('_')[1])
            start_time2, end_time2 = start_frame2 / self.fps, end_frame2 / self.fps
            src_path2 = video_path2
            out_path = os.path.join(out_folder, "origin_cat.mp4") 
            video2 = VideoFileClip(src_path2).subclip(start_time2, end_time2)
            finalclip = concatenate_videoclips([video1, video2], method="compose")

            finalclip.write_videofile(out_path)
        else:
            video1.write_videofile(out_path)

        # ffmpeg_extract_subclip(src_path, start_time, end_time, out_path)
        return True

        
    @rank_zero_only
    def log_sound_steps(self, pl_module, batch, batch_idx, split="train"):        
        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        with torch.no_grad():
            log_dict = pl_module.log_sound(batch,N=self.max_sound_num, ddim_steps=self.ddim_step, split=split, size_len=self.size_len, guidance_scale=self.guidance_scale, uncond_cond=self.uncond_cond)

        # gt_sound_list = log_dict['inputs']
        # rec_sound_list = log_dict['reconstruction']
        # video_path_list= log_dict['video_frame_path']
        # video_time_list = log_dict['video_time']

        self.log_local(pl_module.logger.save_dir, split, log_dict, pl_module.global_step, pl_module.current_epoch, batch_idx)

        if is_train:
            pl_module.train()



    def check_frequency(self, check_idx, split):
        if split == "train":
            if ((check_idx % self.train_batch_freq) == 0 or (check_idx in self.log_steps)) and (
                    check_idx > 0 or self.log_first_step):
                try:
                    self.log_steps.pop(0)
                except IndexError as e:
                    print(e)
                    pass
                return True
            return False
        else:
            if (check_idx % self.val_batch_freq) == 0:
                return True
            else:
                return False


    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        # if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
        if not self.disabled and (pl_module.global_step % self.train_vis_frequency == 0) and pl_module.global_step > 0:
            self.log_sound_steps(pl_module, batch, batch_idx, split="train")
            # pass

    @rank_zero_only
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled and (pl_module.global_step % self.val_vis_frequency == 0) and pl_module.global_step > 0:
            self.log_sound_steps(pl_module, batch, batch_idx, split="val")
            # pass

        if hasattr(pl_module, 'calibrate_grad_norm'):
            if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)





class SoundLogger_fullset_ar(Callback):
    def __init__(self, train_batch_frequency,val_batch_frequency, max_sound_num, sr=22050, clamp=True, increase_log_steps=False,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False, ddim_step=250, size_len=64, guidance_scale=1.0, uncond_cond=None, fps=21.5):
        super().__init__()
        self.rescale = rescale
        self.train_batch_freq = train_batch_frequency
        self.val_batch_freq = val_batch_frequency
        self.max_sound_num = max_sound_num
        self.log_steps = [2 ** n for n in range(int(np.log2(self.train_batch_freq)) + 1)]
        self.log_steps_val = [2 ** n for n in range(int(np.log2(self.val_batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.train_batch_freq]
            self.log_steps_val = [self.val_batch_freq] 
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_first_step = log_first_step
        self.sr = sr
        print('sr: ', self.sr)
        self.ddim_step = ddim_step

        self.train_vis_frequency = train_batch_frequency
        self.val_vis_frequency = val_batch_frequency
        self.size_len = size_len
        self.guidance_scale = guidance_scale
        self.uncond_cond = uncond_cond
        self.fps = fps
        print("Guidance Scale: ", self.guidance_scale)
        print("Uncond cond: ", self.uncond_cond)
    
    
    def inverse_op(self, spec):
        sr = self.sr
        n_fft = 1024
        fmin = 125
        fmax = 7600
        nmels = 128
        hoplen = 1024 // 4
        spec_power = 1

        # Inverse Transform
        spec = spec * 100 - 100
        spec = (spec + 20) / 20
        spec = 10 ** spec
        spec_out = librosa.feature.inverse.mel_to_stft(spec, sr = sr, n_fft = n_fft, fmin=fmin, fmax=fmax, power=spec_power)
        wav = librosa.griffinlim(spec_out, hop_length=hoplen)
        return wav

    @rank_zero_only
    def log_local(self, save_dir, split, log_dict,
                  global_step, current_epoch, batch_idx, show_curve=False, scale_factor=1):

        # root:
        root = os.path.join(save_dir, "sound_eval", split, "epoch_{}_step_{}".format(current_epoch, global_step))

        gt_sound_list = log_dict['inputs_spec'].detach().cpu().numpy()
        rec_sound_list = log_dict['reconstruction_spec'].detach().cpu().numpy()
        video_path_list= log_dict['video_frame_path']
        video_time_list = log_dict['video_time']
        diff_sample_list = log_dict['samples'].detach().cpu().numpy()

        # Prev Spec list:
        prev_spec_flag_list = log_dict["prev_spec_flag_list"]
        prev_spec_list = log_dict["prev_spec"].detach().cpu().numpy()

        # root = os.path.join(save_dir, "sound", split, "epoch_{}_step_{}".format(current_epoch, global_step))

        os.makedirs(root,exist_ok=True)
        

        for i in range(len(gt_sound_list)):
            print('Gen examples ===========> {}'.format(i))
            sample_folder = os.path.join(root, "sample_{}".format(i))
            os.makedirs(sample_folder, exist_ok=True)
            gt_sound = self.inverse_op(gt_sound_list[i])
            rec_sound = self.inverse_op(rec_sound_list[i])
            sf.write(os.path.join(sample_folder, "sample_{}_gt.wav".format(i)), gt_sound, self.sr)

            if prev_spec_flag_list[i]:
                prev_gt_sound = self.inverse_op(prev_spec_list[i])
                sf.write(os.path.join(sample_folder, "sample_{}_prev_gt.wav".format(i)), prev_gt_sound, self.sr)

            sf.write(os.path.join(sample_folder, "sample_{}_stage1_rec_clamp.wav".format(i)), rec_sound, self.sr)
            sample = self.inverse_op(diff_sample_list[i])
            sf.write(os.path.join(sample_folder, "sample_{}_diff_sample_clamp.wav".format(i)), sample, self.sr) 
            try:
                video = self.extract_frame_video(video_path_list[i], video_time_list[i], sample_folder)
            except:
                pass
            # video_save_path = os.path.join(sample_folder, "origin_video.mp4")
            # imageio.mimsave(video_save_path, video, fps=21.5)

                # show curve:
        if show_curve: 
            try: 
                pred_x0_range_min = np.array(log_dict["intermediates"]["pred_x0_range_min"]).squeeze(2).transpose(1,0)     #  sample_num x T
                pred_x0_range_max = np.array(log_dict["intermediates"]["pred_x0_range_max"]).squeeze(2).transpose(1,0)     #  sample_num x T 
                # import pdb
                # pdb.set_trace()
                # x_prev_range_min = np.array(log_dict["intermediates"]['x_prev_range_min']).squeeze(2).transpose(1,0)
                # x_prev_range_max = np.array(log_dict["intermediates"]['x_prev_range_max']).squeeze(2).transpose(1,0)
                x_prev_range_min = np.array(log_dict["intermediates"]['x_prev_range_min']).squeeze(2).transpose(1,0)
                x_prev_range_max = np.array(log_dict["intermediates"]['x_prev_range_max']).squeeze(2).transpose(1,0)

                print('pred_x0 range shape', pred_x0_range_max.shape)
                print('x_prev range shape', x_prev_range_min.shape)


                latent_max = 6 * scale_factor
                latent_min = -6 * scale_factor

                plt.figure(figsize=(20,10))
                for i in range(pred_x0_range_min.shape[0]):
                    pred_x0_min_list = pred_x0_range_min[i]
                    pred_x0_max_list = pred_x0_range_max[i]
                    x_prev_min_list = x_prev_range_min[i]
                    x_prev_max_list = x_prev_range_max[i]

                    plt.subplot(121)
                    plt.plot(range(len(pred_x0_min_list)), pred_x0_min_list, label="Ours Diffusion pred_x0 Min Value {}".format(i))
                    plt.plot(range(len(pred_x0_max_list)), pred_x0_max_list, label="Ours Diffusion pred_x0 Max Value {}".format(i)) 

                    plt.subplot(122)
                    plt.plot(range(len(x_prev_min_list)), x_prev_min_list, label="Ours Diffusion x_prev Min Value {}".format(i))
                    plt.plot(range(len(x_prev_max_list)), x_prev_max_list, label="Ours Diffusion x_prev Max Value {}".format(i))

                plt.subplot(121)
                plt.hlines(latent_max,xmin=0,xmax=len(x_prev_min_list),color='red', label='Latent Max Range / scale_factor:{}'.format(scale_factor),linewidth=5)
                plt.hlines(latent_min,xmin=0,xmax=len(x_prev_min_list),color='red', label='Latent min Range / scale_factor:{}'.format(scale_factor),linewidth=5)
                # plt.ylim((-10,10))
                plt.legend()
                plt.subplot(122)
                plt.hlines(latent_max,xmin=0,xmax=len(x_prev_min_list),color='red', label='Latent Max Range / scale_factor:{}'.format(scale_factor),linewidth=5)
                plt.hlines(latent_min,xmin=0,xmax=len(x_prev_min_list),color='red', label='Latent min Range / scale_factor:{}'.format(scale_factor),linewidth=5)
                # plt.ylim((-10,10))
                plt.legend()
                plt.savefig(os.path.join(root, "sampling_range_new1.png"))
            except Exception as e:
                print(e)

    def extract_frame_video(self, video_path, video_time, out_folder):
        # start_frame, end_frame = video_time[0], video_time[1]
        start_frame, end_frame = int(video_time.split('_')[0]), int(video_time.split('_')[1])
        start_time, end_time = start_frame / self.fps, end_frame / self.fps
        src_path = video_path
        out_path = os.path.join(out_folder, "origin.mp4")
        # ffmpeg_extract_subclip(src_path, start_time, end_time, out_path)
        video1 = VideoFileClip(src_path).subclip(start_time, end_time)
        video1.write_videofile(out_path)
        return True

        

    def log_sound_steps(self, pl_module, batch, batch_idx, split="train"):        
        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        with torch.no_grad():
            log_dict = pl_module.log_sound(batch,N=self.max_sound_num, ddim_steps=self.ddim_step, split=split, size_len=self.size_len, guidance_scale=self.guidance_scale, uncond_cond=self.uncond_cond)

        # gt_sound_list = log_dict['inputs']
        # rec_sound_list = log_dict['reconstruction']
        # video_path_list= log_dict['video_frame_path']
        # video_time_list = log_dict['video_time']

        self.log_local(pl_module.logger.save_dir, split, log_dict, pl_module.global_step, pl_module.current_epoch, batch_idx)

        if is_train:
            pl_module.train()



    def check_frequency(self, check_idx, split):
        
        if split == "train":
            if ((check_idx % self.train_batch_freq) == 0 or (check_idx in self.log_steps)) and (
                    check_idx > 0 or self.log_first_step):
                try:
                    self.log_steps.pop(0)
                except IndexError as e:
                    print(e)
                    pass
                return True
            return False
        else:
            if (check_idx % self.val_batch_freq) == 0:
                return True
            else:
                return False


    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        # if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
        if not self.disabled and (pl_module.global_step % self.train_vis_frequency == 0) and pl_module.global_step > 0:
            self.log_sound_steps(pl_module, batch, batch_idx, split="train")
            # pass


    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled and (pl_module.global_step % self.val_vis_frequency == 0) and pl_module.global_step > 0:
            self.log_sound_steps(pl_module, batch, batch_idx, split="val")
            # pass

        if hasattr(pl_module, 'calibrate_grad_norm'):
            if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)




class SoundLogger_fullset_contrastive(Callback):
    def __init__(self, train_batch_frequency,val_batch_frequency, max_sound_num, sr=22050, clamp=True, increase_log_steps=False,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False, ddim_step=250, size_len=64, guidance_scale=1.0, uncond_cond=None, fps=21.5):
        super().__init__()
        self.rescale = rescale
        self.train_batch_freq = train_batch_frequency
        self.val_batch_freq = val_batch_frequency
        self.max_sound_num = max_sound_num
        self.log_steps = [2 ** n for n in range(int(np.log2(self.train_batch_freq)) + 1)]
        self.log_steps_val = [2 ** n for n in range(int(np.log2(self.val_batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.train_batch_freq]
            self.log_steps_val = [self.val_batch_freq] 
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_first_step = log_first_step
        self.sr = sr
        print('sr: ', self.sr)
        self.ddim_step = ddim_step

        self.train_vis_frequency = train_batch_frequency
        self.val_vis_frequency = val_batch_frequency
        self.size_len = size_len
        self.guidance_scale = guidance_scale
        self.uncond_cond = uncond_cond
        self.fps = fps


    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        # # if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
        # if not self.disabled and (pl_module.global_step % self.train_vis_frequency == 0) and pl_module.global_step > 0:
        #     self.log_sound_steps(pl_module, batch, batch_idx, split="train")
        #     # pass
        pass


    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        # if not self.disabled and (pl_module.global_step % self.val_vis_frequency == 0) and pl_module.global_step > 0:
        #     self.log_sound_steps(pl_module, batch, batch_idx, split="val")
        #     # pass

        # if hasattr(pl_module, 'calibrate_grad_norm'):
        #     if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
        #         self.log_gradients(trainer, pl_module, batch_idx=batch_idx)

        pass

    # def on_train_epoch_end(self, trainer, pl_module, outputs):
    #     pass
    



class SoundLogger_decode_fullset(Callback):
    def __init__(self, train_batch_frequency,val_batch_frequency, max_sound_num, sr=22050, clamp=True, increase_log_steps=False,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False, ddim_step=250, size_len=64, guidance_scale=1.0, uncond_cond=None, fps=21.5):
        super().__init__()
        self.fps = fps
        self.rescale = rescale
        self.train_batch_freq = train_batch_frequency
        self.val_batch_freq = val_batch_frequency
        self.max_sound_num = max_sound_num
        self.log_steps = [2 ** n for n in range(int(np.log2(self.train_batch_freq)) + 1)]
        self.log_steps_val = [2 ** n for n in range(int(np.log2(self.val_batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.train_batch_freq]
            self.log_steps_val = [self.val_batch_freq] 
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_first_step = log_first_step
        self.sr = sr
        print('sr: ', self.sr)

        self.train_vis_frequency = train_batch_frequency
        self.val_vis_frequency = val_batch_frequency

    
    def inverse_op(self, spec):
        sr = self.sr
        n_fft = 1024
        fmin = 125
        fmax = 7600
        nmels = 128
        hoplen = 250
        spec_power = 1

        # Inverse Transform
        spec = spec * 100 - 100
        spec = (spec + 20) / 20
        spec = 10 ** spec
        spec_out = librosa.feature.inverse.mel_to_stft(spec, sr = sr, n_fft = n_fft, fmin=fmin, fmax=fmax, power=spec_power)
        wav = librosa.griffinlim(spec_out, hop_length=hoplen)
        return wav

    @rank_zero_only
    def log_local(self, save_dir, split, log_dict,
                  global_step, current_epoch, batch_idx, show_curve=False, scale_factor=1):

        # root:
        root = os.path.join(save_dir, "sound_eval", split, "epoch_{}_step_{}".format(current_epoch, global_step))

        gt_sound_list = log_dict['inputs_spec'].detach().cpu().numpy()
        rec_sound_list = log_dict['reconstruction_spec'].detach().cpu().numpy()

        os.makedirs(root,exist_ok=True)
        
        video_path_list = log_dict["video_frame_path"]
        video_time_list = log_dict["video_time"]

        for i in range(len(gt_sound_list)):
            print('Gen examples ===========> {}'.format(i))
            sample_folder = os.path.join(root, "sample_{}".format(i))
            os.makedirs(sample_folder, exist_ok=True)
            gt_sound = self.inverse_op(gt_sound_list[i])
            rec_sound = self.inverse_op(rec_sound_list[i])
            sf.write(os.path.join(sample_folder, "sample_{}_gt.wav".format(i)), gt_sound, self.sr)
            sf.write(os.path.join(sample_folder, "sample_{}_stage1_rec.wav".format(i)), rec_sound, self.sr)
            # try:
            #     video = self.extract_concat_frame_video(video_path_list[i], video_time_list[i], out_folder=sample_folder)
            # except Exception as e:
            #     print(e)
            #     pass
                
            with open(os.path.join(sample_folder, "video_path.txt"), "w") as f:
                txt = "Video 1:" + '  ' + str(video_path_list[i]) + "    " + str(video_time_list[i])
                f.writelines(txt)

        
    def extract_concat_frame_video(self, video_path1, video_time1, video_path2=None, video_time2=None, out_folder=None):
        # start_frame, end_frame = video_time[0], video_time[1]
        start_frame1, end_frame1 = int(video_time1.split('_')[0]), int(video_time1.split('_')[1])
        start_time1, end_time1 = start_frame1 / self.fps, end_frame1 / self.fps
        src_path1 = video_path1
        out_path = os.path.join(out_folder, "origin.mp4")

        video1 = VideoFileClip(src_path1).subclip(start_time1, end_time1)

        if video_path2 is not None:
            start_frame2, end_frame2 = int(video_time2.split('_')[0]), int(video_time2.split('_')[1])
            start_time2, end_time2 = start_frame2 / self.fps, end_frame2 / self.fps
            src_path2 = video_path2
            out_path = os.path.join(out_folder, "origin_cat.mp4") 
            video2 = VideoFileClip(src_path2).subclip(start_time2, end_time2)
            finalclip = concatenate_videoclips([video1, video2], method="compose")

            finalclip.write_videofile(out_path)
        else:
            video1.write_videofile(out_path)

        return True

        
    @rank_zero_only
    def log_sound_steps(self, pl_module, batch, batch_idx, split="train"):        
        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        with torch.no_grad():
            log_dict = pl_module.log_sound(batch,N=self.max_sound_num, split=split)

        self.log_local(pl_module.logger.save_dir, split, log_dict, pl_module.global_step, pl_module.current_epoch, batch_idx)

        if is_train:
            pl_module.train()



    def check_frequency(self, check_idx, split):
        if split == "train":
            if ((check_idx % self.train_batch_freq) == 0 or (check_idx in self.log_steps)) and (
                    check_idx > 0 or self.log_first_step):
                try:
                    self.log_steps.pop(0)
                except IndexError as e:
                    print(e)
                    pass
                return True
            return False
        else:
            if (check_idx % self.val_batch_freq) == 0:
                return True
            else:
                return False


    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        # if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
        if not self.disabled and (pl_module.global_step % self.train_vis_frequency == 0) and pl_module.global_step > 0:
            self.log_sound_steps(pl_module, batch, batch_idx, split="train")
            # pass


    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled and (pl_module.global_step % self.val_vis_frequency == 0) and pl_module.global_step > 0:
            self.log_sound_steps(pl_module, batch, batch_idx, split="val")
            # pass

        if hasattr(pl_module, 'calibrate_grad_norm'):
            if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)





class SoundLogger_stage3_diff_fullset(Callback):
    def __init__(self, video_root_dir, train_batch_frequency,val_batch_frequency, max_sound_num, sr=22050, clamp=True, increase_log_steps=False,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False, ddim_steps=250, size_len=64, cond_scale=1.0, uncond_cond=None, fps=4):
        super().__init__()
        self.fps = fps
        self.rescale = rescale
        self.train_batch_freq = train_batch_frequency
        self.val_batch_freq = val_batch_frequency
        self.max_sound_num = max_sound_num
        self.log_steps = [2 ** n for n in range(int(np.log2(self.train_batch_freq)) + 1)]
        self.log_steps_val = [2 ** n for n in range(int(np.log2(self.val_batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.train_batch_freq]
            self.log_steps_val = [self.val_batch_freq] 
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_first_step = log_first_step
        self.sr = sr
        print('sr: ', self.sr)

        self.train_vis_frequency = train_batch_frequency
        self.val_vis_frequency = val_batch_frequency

        # Video Root Dir:
        self.video_root_dir = video_root_dir

        # Sampling Params:
        self.cond_scale = cond_scale
        self.ddim_steps = ddim_steps
    
    def inverse_op(self, spec):
        sr = self.sr
        n_fft = 1024
        fmin = 125
        fmax = 7600
        nmels = 128
        hoplen = 250
        spec_power = 1

        # Inverse Transform
        spec = spec * 100 - 100
        spec = (spec + 20) / 20
        spec = 10 ** spec
        spec_out = librosa.feature.inverse.mel_to_stft(spec, sr = sr, n_fft = n_fft, fmin=fmin, fmax=fmax, power=spec_power)
        wav = librosa.griffinlim(spec_out, hop_length=hoplen)
        return wav

    @rank_zero_only
    def log_local(self, save_dir, split, log_dict,
                  global_step, current_epoch, batch_idx, show_curve=False, scale_factor=1):

        if split == "train":
            split = "Train"
        elif split == "val":
            split = "Test"

        # root:
        root = os.path.join(save_dir, "sound_eval", split, "epoch_{}_step_{}_cond_scale{}".format(current_epoch, global_step, self.cond_scale))

        gt_sound_list = log_dict['inputs_spec'].detach().cpu().numpy()
        rec_sound_list = log_dict['reconstruction_spec'].detach().cpu().numpy()
        diff_sample_list = log_dict["diff_sample_spec"].detach().cpu().numpy()

        os.makedirs(root,exist_ok=True)
        
        video_path_list = log_dict["video_name"]
        # Revise the Video Name:
        video_path_list = list(map(lambda x: os.path.join(self.video_root_dir, split, "video_fps21.5", x + ".mp4"), video_path_list))
        video_time_list = log_dict["video_time"]

        for i in range(len(gt_sound_list)):
            print('Gen examples ===========> {}'.format(i))
            sample_folder = os.path.join(root, "sample_{}".format(i))
            os.makedirs(sample_folder, exist_ok=True)
            gt_sound = self.inverse_op(gt_sound_list[i])
            rec_sound = self.inverse_op(rec_sound_list[i])
            sample = self.inverse_op(diff_sample_list[i])

            sf.write(os.path.join(sample_folder, "sample_{}_gt.wav".format(i)), gt_sound, self.sr)
            sf.write(os.path.join(sample_folder, "sample_{}_stage1_rec.wav".format(i)), rec_sound, self.sr)
            sf.write(os.path.join(sample_folder, "sample_{}_diff_sample.wav".format(i)), sample, self.sr) 

            try:
                video = self.extract_concat_frame_video(video_path_list[i], video_time_list[i], out_folder=sample_folder)
            except Exception as e:
                print(e)
                pass
            with open(os.path.join(sample_folder, "video_path.txt"), "w") as f:
                txt = "Video 1:" + '  ' + str(video_path_list[i]) + "    " + str(video_time_list[i])
                f.writelines(txt)

        
    def extract_concat_frame_video(self, video_path1, video_time1, video_path2=None, video_time2=None, out_folder=None):
        # start_frame, end_frame = video_time[0], video_time[1]
        start_frame1, end_frame1 = int(video_time1.split('_')[0]), int(video_time1.split('_')[1])
        start_time1, end_time1 = start_frame1 / self.fps, end_frame1 / self.fps
        src_path1 = video_path1
        out_path = os.path.join(out_folder, "origin.mp4")

        video1 = VideoFileClip(src_path1).subclip(start_time1, end_time1)

        if video_path2 is not None:
            start_frame2, end_frame2 = int(video_time2.split('_')[0]), int(video_time2.split('_')[1])
            start_time2, end_time2 = start_frame2 / self.fps, end_frame2 / self.fps
            src_path2 = video_path2
            out_path = os.path.join(out_folder, "origin_cat.mp4") 
            video2 = VideoFileClip(src_path2).subclip(start_time2, end_time2)
            finalclip = concatenate_videoclips([video1, video2], method="compose")

            finalclip.write_videofile(out_path)
        else:
            video1.write_videofile(out_path)

        return True

        
    @rank_zero_only
    def log_sound_steps(self, pl_module, batch, batch_idx, split="train"):        
        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        with torch.no_grad():
            log_dict = pl_module.log_sound(batch, N=self.max_sound_num, split=split, ddim_steps=self.ddim_steps, cond_scale=self.cond_scale)

        self.log_local(pl_module.logger.save_dir, split, log_dict, pl_module.global_step, pl_module.current_epoch, batch_idx)

        if is_train:
            pl_module.train()



    def check_frequency(self, check_idx, split):
        if split == "train":
            if ((check_idx % self.train_batch_freq) == 0 or (check_idx in self.log_steps)) and (
                    check_idx > 0 or self.log_first_step):
                try:
                    self.log_steps.pop(0)
                except IndexError as e:
                    print(e)
                    pass
                return True
            return False
        else:
            if (check_idx % self.val_batch_freq) == 0:
                return True
            else:
                return False


    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        # if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
        if not self.disabled and (pl_module.global_step % self.train_vis_frequency == 0) and pl_module.global_step > 0:
            self.log_sound_steps(pl_module, batch, batch_idx, split="train")
            # pass


    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled and (pl_module.global_step % self.val_vis_frequency == 0) and pl_module.global_step >= 0:
            self.log_sound_steps(pl_module, batch, batch_idx, split="val")
            # pass

        if hasattr(pl_module, 'calibrate_grad_norm'):
            if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)



# Contrastive:
class SoundLogger_fullset_contrastive(Callback):
    def __init__(self, train_batch_frequency,val_batch_frequency, max_sound_num, sr=22050, clamp=True, increase_log_steps=False,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False, ddim_step=250, size_len=64, guidance_scale=1.0, uncond_cond=None, fps=21.5):
        super().__init__()
        self.rescale = rescale
        self.train_batch_freq = train_batch_frequency
        self.val_batch_freq = val_batch_frequency
        self.max_sound_num = max_sound_num
        self.log_steps = [2 ** n for n in range(int(np.log2(self.train_batch_freq)) + 1)]
        self.log_steps_val = [2 ** n for n in range(int(np.log2(self.val_batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.train_batch_freq]
            self.log_steps_val = [self.val_batch_freq] 
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_first_step = log_first_step
        self.sr = sr
        print('sr: ', self.sr)
        self.ddim_step = ddim_step

        self.train_vis_frequency = train_batch_frequency
        self.val_vis_frequency = val_batch_frequency
        self.size_len = size_len
        self.guidance_scale = guidance_scale
        self.uncond_cond = uncond_cond
        self.fps = fps
        print("Guidance Scale: ", self.guidance_scale)
        print("Uncond cond: ", self.uncond_cond)
    
    
    def inverse_op(self, spec):
        sr = self.sr
        n_fft = 1024
        fmin = 125
        fmax = 7600
        nmels = 128
        hoplen = 1024 // 4
        spec_power = 1

        # Inverse Transform
        spec = spec * 100 - 100
        spec = (spec + 20) / 20
        spec = 10 ** spec
        spec_out = librosa.feature.inverse.mel_to_stft(spec, sr = sr, n_fft = n_fft, fmin=fmin, fmax=fmax, power=spec_power)
        wav = librosa.griffinlim(spec_out, hop_length=hoplen)
        return wav

    @rank_zero_only
    def log_local(self, save_dir, split, log_dict,
                  global_step, current_epoch, batch_idx, show_curve=False, scale_factor=1):

        # root:
        root = os.path.join(save_dir, "sound_eval", split, "epoch_{}_step_{}".format(current_epoch, global_step))

        gt_sound_list = log_dict['inputs_spec'].detach().cpu().numpy()
        rec_sound_list = log_dict['reconstruction_spec'].detach().cpu().numpy()
        video_path_list= log_dict['video_frame_path']
        video_time_list = log_dict['video_time']
        diff_sample_list = log_dict['samples'].detach().cpu().numpy()

        # root = os.path.join(save_dir, "sound", split, "epoch_{}_step_{}".format(current_epoch, global_step))

        os.makedirs(root,exist_ok=True)
        

        for i in range(len(gt_sound_list)):
            print('Gen examples ===========> {}'.format(i))
            sample_folder = os.path.join(root, "sample_{}".format(i))
            os.makedirs(sample_folder, exist_ok=True)
            gt_sound = self.inverse_op(gt_sound_list[i])
            rec_sound = self.inverse_op(rec_sound_list[i])
            sf.write(os.path.join(sample_folder, "sample_{}_gt.wav".format(i)), gt_sound, self.sr)
            sf.write(os.path.join(sample_folder, "sample_{}_stage1_rec_clamp.wav".format(i)), rec_sound, self.sr)
            sample = self.inverse_op(diff_sample_list[i])
            sf.write(os.path.join(sample_folder, "sample_{}_diff_sample_clamp.wav".format(i)), sample, self.sr) 
            try:
                video = self.extract_concat_frame_video(video_path_list[i], video_time_list[i], out_folder=sample_folder)
            except:
                pass
            # video_save_path = os.path.join(sample_folder, "origin_video.mp4")
            # imageio.mimsave(video_save_path, video, fps=21.5)
            with open(os.path.join(sample_folder, "video_path.txt"), "w") as f:
                txt = "Video 1:" + '  ' + str(video_path_list[i]) + "    " + str(video_time_list[i])
                f.writelines(txt)



    def extract_concat_frame_video(self, video_path1, video_time1, video_path2=None, video_time2=None, out_folder=None):
        # start_frame, end_frame = video_time[0], video_time[1]
        start_frame1, end_frame1 = int(video_time1.split('_')[0]), int(video_time1.split('_')[1])
        start_time1, end_time1 = start_frame1 / self.fps, end_frame1 / self.fps
        src_path1 = video_path1
        out_path = os.path.join(out_folder, "origin.mp4")

        video1 = VideoFileClip(src_path1).subclip(start_time1, end_time1)

        if video_path2 is not None:
            start_frame2, end_frame2 = int(video_time2.split('_')[0]), int(video_time2.split('_')[1])
            start_time2, end_time2 = start_frame2 / self.fps, end_frame2 / self.fps
            src_path2 = video_path2
            out_path = os.path.join(out_folder, "origin_cat.mp4") 
            video2 = VideoFileClip(src_path2).subclip(start_time2, end_time2)
            finalclip = concatenate_videoclips([video1, video2], method="compose")

            finalclip.write_videofile(out_path)
        else:
            video1.write_videofile(out_path)

        return True

        
    @rank_zero_only
    def log_sound_steps(self, pl_module, batch, batch_idx, split="train"):        
        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        with torch.no_grad():
            log_dict = pl_module.log_sound(batch,N=self.max_sound_num, ddim_steps=self.ddim_step, split=split, size_len=self.size_len, guidance_scale=self.guidance_scale, uncond_cond=self.uncond_cond)

        # gt_sound_list = log_dict['inputs']
        # rec_sound_list = log_dict['reconstruction']
        # video_path_list= log_dict['video_frame_path']
        # video_time_list = log_dict['video_time']
 
        self.log_local(pl_module.logger.save_dir, split, log_dict, pl_module.global_step, pl_module.current_epoch, batch_idx)

        if is_train:
            pl_module.train()



    def check_frequency(self, check_idx, split):
        
        if split == "train":
            if ((check_idx % self.train_batch_freq) == 0 or (check_idx in self.log_steps)) and (
                    check_idx > 0 or self.log_first_step):
                try:
                    self.log_steps.pop(0)
                except IndexError as e:
                    print(e)
                    pass
                return True
            return False
        else:
            if (check_idx % self.val_batch_freq) == 0:
                return True
            else:
                return False


    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        # if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
        if not self.disabled and (pl_module.global_step % self.train_vis_frequency == 0) and pl_module.global_step > 0:
            self.log_sound_steps(pl_module, batch, batch_idx, split="train")
            # pass


    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled and (pl_module.global_step % self.val_vis_frequency == 0) and pl_module.global_step > 0:
            self.log_sound_steps(pl_module, batch, batch_idx, split="val")
            # pass

        if hasattr(pl_module, 'calibrate_grad_norm'):
            if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)