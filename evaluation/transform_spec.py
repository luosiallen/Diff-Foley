import numpy as np
import librosa
import soundfile as sf
import os
import argparse
from tqdm import tqdm
from multiprocessing import Pool



"""
Goal: 

Transform the 16000hz 128 mel spec -> 22050 80 mel spec

"""


def inverse_op(spec, sr=16000):
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

def log_spec_to_linear_spec(spec):
    spec = spec * 100 - 100
    spec = (spec + 20) / 20
    spec = 10 ** spec
    return spec


def transform_spec(origin_spec, origin_n_mels, origin_sr, new_n_mels, new_sr, n_fft=1024):
    # Log Mel to Linear Spec:
    linear_spec = log_spec_to_linear_spec(origin_spec)

    # Sr Resampling:
    linear_spec_resampled = librosa.resample(linear_spec, origin_sr, new_sr)

    # Mel Basis Change:
    mel_basis = librosa.filters.mel(sr=origin_sr, n_fft=n_fft, n_mels=origin_n_mels)
    inv_mel_basis = np.linalg.pinv(mel_basis)
    linear_spec_resampled = np.dot(inv_mel_basis, linear_spec_resampled)

    # Mel Num Change:
    spec_new = librosa.feature.melspectrogram(y=None, S=linear_spec_resampled, sr=new_sr, n_fft=n_fft, n_mels=new_n_mels)

    # To Log Spec:
    log_spec_new = np.log10(np.maximum(1e-5, spec_new))
    log_spec_new = (((log_spec_new * 20) - 20) + 100) / 100
    log_spec_new = np.clip(log_spec_new, 0, 1.0)
    return log_spec_new



def process_data(root_path, file_name, save_path):
    # origin_n_mels = 80
    # origin_sr = 22050
    # new_n_mels = 128
    # new_sr = 16000

    origin_n_mels = 128
    origin_sr = 16000
    new_n_mels = 80
    new_sr = 22050

    try:
        spec = np.load(os.path.join(root_path, file_name))
        new_spec = transform_spec(spec, origin_n_mels=origin_n_mels, origin_sr=origin_sr, new_n_mels=new_n_mels, new_sr=new_sr)
        np.save(os.path.join(save_path, file_name), new_spec)
        return file_name, True
    except Exception as e:
        print(e)
        return file_name, False


success_list = []
err_list = []





if __name__ == "__main__":
    paser = argparse.ArgumentParser()
    paser.add_argument("--split", default=1, type=int)
    paser.add_argument("--node", type=int, default=0)
    args = paser.parse_args()


    def callback_fn(res):
        audio_name, flag = res
        if flag:
            success_list.append(audio_name)
        else:
            err_list.append(audio_name)
        pbar.update()

    
    root_path = "./generate_folder"
    save_path = "./save_folder"
    
    
    os.makedirs(save_path, exist_ok=True)
    data_list = sorted(os.listdir(root_path))

    # split:
    split_num = args.split
    node = args.node

    split_len = len(data_list) // split_num + 1

    start = node * split_len 
    end = (node + 1) * split_len

    if end >= len(data_list):
        end = len(data_list)

    data_list = data_list[start:end]

    pbar = tqdm(total=len(data_list))


    with Pool(32) as p:
        for i in range(len(data_list)):
            res = p.apply_async(process_data, (root_path, data_list[i], save_path), callback=callback_fn)
        p.close()
        p.join()



    success_list = sorted(success_list)
    err_list = sorted(err_list)

    success_txt = "\n".join(success_list)
    err_txt = "\n".join(err_list)

    with open("./success.txt", "w") as f:
        f.writelines(success_txt)

    with open("./fail.txt", "w") as f:
        f.writelines(err_txt)

    print('success list',success_list[:10])
    print('err list', err_list[:10])
    print("done!")

