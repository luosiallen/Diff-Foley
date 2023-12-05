import ast
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from multiprocessing import Value

import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
import webdataset as wds
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from webdataset.filters import _shuffle
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample

import torchvision.transforms.functional as transform_f


import functools

import io
import numpy 

import torch.nn as nn

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, sep="\t", tokenizer=None):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        logging.debug('Done loading data.')

        self.tokenize = tokenizer

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        texts = self.tokenize([str(self.captions[idx])])[0]
        return images, texts


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def get_dataset_size(shards):
    shards_list = wds.shardlists.expand_urls(shards)
    dir_path = os.path.dirname(shards_list[0])
    sizes_filename = os.path.join(dir_path, 'sizes.json')
    len_filename = os.path.join(dir_path, '__len__')
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, 'r'))
        total_size = sum([int(sizes[os.path.basename(shard)]) for shard in shards_list])
    elif os.path.exists(len_filename):
        # FIXME this used to be eval(open(...)) but that seemed rather unsafe
        total_size = ast.literal_eval(open(len_filename, 'r').read())
    else:
        total_size = None  # num samples undefined
        # some common dataset sizes (at time of authors last download)
        # CC3M (train): 2905954
        # CC12M: 10968539
        # LAION-400M: 407332084
        # LAION-2B (english): 2170337258
    num_shards = len(shards_list)
    return total_size, num_shards


def get_imagenet(args, preprocess_fns, split):
    assert split in ["train", "val", "v2"]
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns

    if split == "v2":
        from imagenetv2_pytorch import ImageNetV2Dataset
        dataset = ImageNetV2Dataset(location=args.imagenet_v2, transform=preprocess_val)
    else:
        if is_train:
            data_path = args.imagenet_train
            preprocess_fn = preprocess_train
        else:
            data_path = args.imagenet_val
            preprocess_fn = preprocess_val
        assert data_path

        dataset = datasets.ImageFolder(data_path, transform=preprocess_fn)

    if is_train:
        idxs = np.zeros(len(dataset.targets))
        target_array = np.array(dataset.targets)
        k = 50
        for c in range(1000):
            m = target_array == c
            n = len(idxs[m])
            arr = np.zeros(n)
            arr[:k] = 1
            np.random.shuffle(arr)
            idxs[m] = arr

        idxs = idxs.astype('int')
        sampler = SubsetRandomSampler(np.where(idxs)[0])
    else:
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler,
    )

    return DataInfo(dataloader=dataloader, sampler=sampler)


def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches


def filter_no_caption_or_no_image(sample):
    has_caption = ('txt' in sample)
    has_image = ('png' in sample or 'jpg' in sample or 'jpeg' in sample or 'webp' in sample)
    return has_caption and has_image


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


try:
    import os
    gpu_num = len(os.environ["CUDA_VISIBLE_DEVICES"])
except:
    gpu_num = 8

if gpu_num <= 4:
    _SHARD_SHUFFLE_SIZE = 50
    _SHARD_SHUFFLE_INITIAL = 50
    _SAMPLE_SHUFFLE_SIZE = 50
    _SAMPLE_SHUFFLE_INITIAL = 50

else:
    _SHARD_SHUFFLE_SIZE = 1000
    _SHARD_SHUFFLE_INITIAL = 1000
    _SAMPLE_SHUFFLE_SIZE = 1000
    _SAMPLE_SHUFFLE_INITIAL = 1000



class detshuffle2(wds.PipelineStage):
    def __init__(
            self,
            bufsize=1000,
            initial=100,
            seed=0,
            epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            # If seed is negative, we use the worker's seed, this will be different across all nodes/workers
            seed = pytorch_worker_seed(epoch)
        else:
            # This seed to be deterministic AND the same across all nodes/workers in each epoch
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        nshards=sys.maxsize,
        worker_seed=None,
        deterministic=False,
        epoch=-1,
    ):
        """Sample shards from the shard list with replacement.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls = wds.shardlists.expand_urls(urls)
        self.urls = urls
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        if self.deterministic:
            # reset seed w/ epoch if deterministic
            if self.worker_seed is None:
                # pytorch worker seed should be deterministic due to being init by arg.seed + rank + worker id
                seed = pytorch_worker_seed(epoch)
            else:
                seed = self.worker_seed() + epoch
            self.rng.seed(seed)
        for _ in range(self.nshards):
            yield dict(url=self.rng.choice(self.urls))


def get_wds_dataset(args, preprocess_img, is_train, epoch=0, floor=False, tokenizer=None):
    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None
    resampled = getattr(args, 'dataset_resampled', False) and is_train

    num_samples, num_shards = get_dataset_size(input_shards)
    if not num_samples:
        if is_train:
            num_samples = args.train_num_samples
            if not num_samples:
                raise RuntimeError(
                    'Currently, number of dataset samples must be specified for training dataset. '
                    'Please specify via `--train-num-samples` if no dataset length info present.')
        else:
            num_samples = args.val_num_samples or 0  # eval will just exhaust the iterator if not specified

    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc
    
    if resampled:
        pipeline = [ResampledShards2(input_shards, deterministic=True, epoch=shared_epoch)]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    # at this point we have an iterator over all the shards
    if is_train:
        if not resampled:
            pipeline.extend([
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ])
        pipeline.extend([
            # at this point, we have an iterator over the shards assigned to each worker at each node
            tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ])
    else:
        pipeline.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),
        ])
    pipeline.extend([
        wds.select(filter_no_caption_or_no_image),
        wds.decode("pilrgb", handler=log_and_continue),
        wds.rename(image="jpg;png;jpeg;webp", text="txt"),
        wds.map_dict(image=preprocess_img, text=lambda text: tokenizer(text)[0]),
        wds.to_tuple("image", "text"),
        wds.batched(args.batch_size, partial=not is_train),
    ])

    dataset = wds.DataPipeline(*pipeline)
    if is_train:
        if not resampled:
            assert num_shards >= args.workers * args.world_size, 'number of shards must be >= total workers'
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=True,
    )

    # FIXME not clear which approach is better, with_epoch before vs after dataloader?
    # hoping to resolve via https://github.com/webdataset/webdataset/issues/169
    # if is_train:
    #     # roll over and repeat a few samples to get same number of full batches on each node
    #     global_batch_size = args.batch_size * args.world_size
    #     num_batches = math.ceil(num_samples / global_batch_size)
    #     num_workers = max(1, args.workers)
    #     num_batches = math.ceil(num_batches / num_workers) * num_workers
    #     num_samples = num_batches * global_batch_size
    #     dataloader = dataloader.with_epoch(num_batches)
    # else:
    #     # last batches are partial, eval is done on single (master) node
    #     num_batches = math.ceil(num_samples / args.batch_size)

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples
    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def get_csv_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = CsvDataset(
        input_filename,
        preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        sep=args.csv_separator,
        tokenizer=tokenizer
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


class SyntheticDataset(Dataset):

    def __init__(self, transform=None, image_size=(224, 224), caption="Dummy caption", dataset_size=100, tokenizer=None):
        self.transform = transform
        self.image_size = image_size
        self.caption = caption
        self.image = Image.new('RGB', image_size)
        self.dataset_size = dataset_size

        self.preprocess_txt = lambda text: tokenizer(text)[0]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if self.transform is not None:
            image = self.transform(self.image)
        return image, self.preprocess_txt(self.caption)


def get_synthetic_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    image_size = preprocess_fn.transforms[0].size
    dataset = SyntheticDataset(
        transform=preprocess_fn, image_size=image_size, dataset_size=args.train_num_samples, tokenizer=tokenizer)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


"""
    Revise DataLoader:
"""
class VGGSound_audio_spec_Dataset(torch.utils.data.Dataset):
    # Only Load audio dataset: for training Stage1: Audio Npy Dataset
    def __init__(self, split, data_dir, split_txt_path, transforms=None, sr=16000, duration=10, truncate_sec=4, fps=4, subset_num=False, fix_frames=False, video_len=40, hop_size=250, shape=(224, 224)):
        super().__init__()
        self.data_dir = data_dir

        if split == "train":
            self.split_path = os.path.join(self.data_dir, "Train")  # spec dir
            self.split = "Train"
        elif split == "valid" or split == 'test':
            self.split_path = os.path.join(self.data_dir, "Test")   # spec dir
            self.split = "Test"

        # Default params:
        self.sr = sr                        # 22050
        self.duration = duration            # 10
        self.truncate_sec = truncate_sec    # 8
        self.fps = fps
        self.fix_frames = fix_frames
        self.video_len = video_len          # default: 40 Frames
        self.hop_size = hop_size            # hop_size: 250
        self.shape = shape
        print("Fix Frames: {}".format(self.fix_frames))

        # Update dir:
        self.video_dir = os.path.join(self.data_dir, self.split, "video_fps3.9")
        self.video_npy_dir = os.path.join(self.data_dir, self.split, "video_fps3.9_npy")
        self.spec_dir = os.path.join(self.data_dir, self.split, "audio_npy_spec_640spec")

        # Data list:
        with open(os.path.join(split_txt_path, '{}_clean_success.txt'.format(self.split)), "r") as f:
            data_list = f.readlines()
            data_list = list(map(lambda x: x.strip(), data_list))
            data_list = list(map(lambda x: x + '_mel.npy', data_list))      # spec

        # data_list = data_list[:32]
        random.shuffle(data_list)
        if subset_num:
            if self.split == "Train":
                self.sample_list = data_list[:subset_num]
            else:
                self.sample_list = data_list[: 2000]
        else:
            self.sample_list = data_list
        print('Split: {}  Sample Num: {}'.format(split, len(self.sample_list)))


    def __len__(self):
        return len(self.sample_list)
    

    def transform_video(self, video_tensor):
        # T x 3 x H x W
        video_tensor = transform_f.resize(video_tensor, self.shape)
        # Normalize:
        video_tensor = video_tensor.div(255)
        return video_tensor


    def load_video_and_spec(self, video_npy_path, spec_path):

        spec_raw = np.load(spec_path).astype(np.float32)
        video_npy = np.load(video_npy_path).astype(np.uint8)

        # Random Truncate:
        if not self.fix_frames:
            start_frame = random.randint(0, self.video_len - self.truncate_sec * self.fps - 1)
        else:
            start_frame = 0
        
        truncate_frame = self.truncate_sec * self.fps
        end_frame = start_frame + truncate_frame

        # Spec Start & Truncate:
        spec_start = int(start_frame / self.fps * self.sr / self.hop_size)
        spec_truncate = int(self.truncate_sec * self.sr / self.hop_size)

        # Check spec_raw:
        if spec_raw.shape[-1] < spec_start + spec_truncate:
            repeat_num = int((spec_start + spec_truncate) // spec_raw.shape[-1]) + 1
            spec_raw = np.tile(spec_raw, repeat_num)               # repeat 2 
            spec_raw = spec_raw[:, spec_start : spec_start + spec_truncate]
        else:
            spec_raw = spec_raw[:, spec_start : spec_start + spec_truncate]
        
        # Check Video npy:
        if video_npy.shape[0] < start_frame + truncate_frame:
            repeat_num = int((start_frame + truncate_frame) // video_npy.shape[0]) + 1
            video_npy = np.tile(video_npy, (repeat_num,1))
            video_npy = video_npy[start_frame: start_frame + truncate_frame]
        else:
            video_npy = video_npy[start_frame: start_frame + truncate_frame]

        # Video Tensor Transforms:
        video_npy = self.transform_video(torch.from_numpy(video_npy))
        spec_raw = torch.from_numpy(spec_raw)
        return video_npy, spec_raw, start_frame, end_frame



    def __getitem__(self, idx):
        audio_name = self.sample_list[idx]             

        # video_name:
        video_name = os.path.basename(audio_name)[:-8]      # remove _mel.npy
        video_npy_path = os.path.join(self.video_npy_dir, video_name + '.npy')
        spec_npy_path = os.path.join(self.spec_dir, audio_name)

        # Load Video & Spec npy:
        video, spec, start_frame, end_frame = self.load_video_and_spec(video_npy_path, spec_npy_path)

        # video_path:
        video_path = os.path.join(self.video_dir, video_name + '.mp4')
        
        data_dict = {}
        data_dict['video'] = video
        data_dict['spec'] = spec
        data_dict['audio_name'] = audio_name
        data_dict['video_path'] =  video_path
        data_dict['video_time'] = str(start_frame) + '_' + str(end_frame)
        return data_dict



def get_spec_and_audio_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):

    # VGGSound audio spec Dataset
    if is_train:
        split = "train"
    else:
        split = "test"
    dataset = VGGSound_audio_spec_Dataset(split, args.data_dir, args.split_txt_path, sr=args.sr, fps=args.fps, truncate_sec=args.truncate_sec, subset_num=args.subset_num)

    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )

    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_spec_and_audio_dataset_bias(args, preprocess_fn, is_train, epoch=0, tokenizer=None):

    # VGGSound audio spec Dataset
    if is_train:
        split = "train"
    else:
        split = "test"
    dataset = VGGSound_audio_spec_Dataset_Bias(split, args.data_dir, args.split_txt_path, sr=args.sr, fps=args.fps, truncate_sec=args.truncate_sec, subset_num=args.subset_num)

    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )

    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def transform_video(video_tensor):
    # T x 3 x H x W
    video_tensor = transform_f.resize(video_tensor, (224, 224))
    # Normalize:
    video_tensor = video_tensor.div(255)
    return video_tensor


def cut_video_and_spec(video, spec, fps=4, video_len=40, truncate_frame=16):

    spec_raw = spec
    video_npy = video

    start_frame = random.randint(0, video_len - truncate_frame - 1)
    truncate_sec = 4
    # fps = 4
    sr = 16000
    hop_size = 250
    shape_h = 224

    truncate_frame = truncate_sec * fps
    end_frame = start_frame + truncate_frame

    # Spec Start & Truncate:
    spec_start = int(start_frame / fps * sr / hop_size)
    spec_truncate = int(truncate_sec * sr / hop_size)


    stream = io.BytesIO(spec_raw)
    spec_raw = numpy.lib.format.read_array(stream)

    stream = io.BytesIO(video_npy)
    # video_npy = numpy.lib.format.read_array(stream)
    video_npy = Image.open(stream)
    video_npy = np.array(video_npy)
    # video transpose:
    video_npy = video_npy.reshape(shape_h, -1, shape_h, 3).transpose(1,3,0,2)    # T x 3 x H x W

        # return numpy.lib.format.read_array(stream)
    
    # Check spec_raw:
    if spec_raw.shape[-1] < spec_start + spec_truncate:
        repeat_num = int((spec_start + spec_truncate) // spec_raw.shape[-1]) + 1
        spec_raw = np.tile(spec_raw, repeat_num)               # repeat 2 
        spec_raw = spec_raw[:, spec_start : spec_start + spec_truncate]
    else:
        spec_raw = spec_raw[:, spec_start : spec_start + spec_truncate]
    
    # Check Video npy:
    if video_npy.shape[0] < start_frame + truncate_frame:
        repeat_num = int((start_frame + truncate_frame) // video_npy.shape[0]) + 1
        video_npy = np.tile(video_npy, (repeat_num,1))
        video_npy = video_npy[start_frame: start_frame + truncate_frame]
    else:
        video_npy = video_npy[start_frame: start_frame + truncate_frame]

    # Video Tensor Transforms:
    video_npy = transform_video(torch.from_numpy(video_npy))
    spec_raw = torch.from_numpy(spec_raw)
    return video_npy, spec_raw, start_frame, end_frame



def cut_video_and_spec_filter(video, spec):

    spec_raw = spec
    video_npy = video

    start_frame = random.randint(0, 40 - 4 * 4 - 1)
    truncate_sec = 4
    fps = 4
    sr = 16000
    hop_size = 250

    truncate_frame = truncate_sec * fps
    end_frame = start_frame + truncate_frame

    # Spec Start & Truncate:
    spec_start = int(start_frame / fps * sr / hop_size)
    spec_truncate = int(truncate_sec * sr / hop_size)


    stream = io.BytesIO(spec_raw)
    spec_raw = numpy.lib.format.read_array(stream)

    stream = io.BytesIO(video_npy)
    video_npy = numpy.lib.format.read_array(stream)

        # return numpy.lib.format.read_array(stream)
    
    # Check spec_raw:
    if spec_raw.shape[-1] < spec_start + spec_truncate:
        repeat_num = int((spec_start + spec_truncate) // spec_raw.shape[-1]) + 1
        spec_raw = np.tile(spec_raw, repeat_num)               # repeat 2 
        spec_raw = spec_raw[:, spec_start : spec_start + spec_truncate]
    else:
        spec_raw = spec_raw[:, spec_start : spec_start + spec_truncate]
    
    # Check spec_raw amplitude: < threshold ?
    amplitude = np.mean(spec_raw)
    if amplitude < 0.08:
        flag = False
    else:
        flag = True

    # Check Video npy:
    if video_npy.shape[0] < start_frame + truncate_frame:
        repeat_num = int((start_frame + truncate_frame) // video_npy.shape[0]) + 1
        video_npy = np.tile(video_npy, (repeat_num,1))
        video_npy = video_npy[start_frame: start_frame + truncate_frame]
    else:
        video_npy = video_npy[start_frame: start_frame + truncate_frame]

    # Video Tensor Transforms:
    video_npy = transform_video(torch.from_numpy(video_npy))
    spec_raw = torch.from_numpy(spec_raw)
    return video_npy, spec_raw, start_frame, end_frame, flag


# Augment:
def cut_video_and_spec_filter(video, spec):

    spec_raw = spec
    video_npy = video

    start_frame = random.randint(0, 40 - 4 * 4 - 1)
    truncate_sec = 4
    fps = 4
    sr = 16000
    hop_size = 250

    truncate_frame = truncate_sec * fps
    end_frame = start_frame + truncate_frame

    # Spec Start & Truncate:
    spec_start = int(start_frame / fps * sr / hop_size)
    spec_truncate = int(truncate_sec * sr / hop_size)


    stream = io.BytesIO(spec_raw)
    spec_raw = numpy.lib.format.read_array(stream)

    stream = io.BytesIO(video_npy)
    video_npy = numpy.lib.format.read_array(stream)

        # return numpy.lib.format.read_array(stream)
    
    # Check spec_raw:
    if spec_raw.shape[-1] < spec_start + spec_truncate:
        repeat_num = int((spec_start + spec_truncate) // spec_raw.shape[-1]) + 1
        spec_raw = np.tile(spec_raw, repeat_num)               # repeat 2 
        spec_raw = spec_raw[:, spec_start : spec_start + spec_truncate]
    else:
        spec_raw = spec_raw[:, spec_start : spec_start + spec_truncate]
    
    # Check spec_raw amplitude: < threshold ?
    amplitude = np.mean(spec_raw)
    if amplitude < 0.08:
        flag = False
    else:
        flag = True

    # Check Video npy:
    if video_npy.shape[0] < start_frame + truncate_frame:
        repeat_num = int((start_frame + truncate_frame) // video_npy.shape[0]) + 1
        video_npy = np.tile(video_npy, (repeat_num,1))
        video_npy = video_npy[start_frame: start_frame + truncate_frame]
    else:
        video_npy = video_npy[start_frame: start_frame + truncate_frame]

    # Video Tensor Transforms:
    video_npy = transform_video(torch.from_numpy(video_npy))
    spec_raw = torch.from_numpy(spec_raw)
    return video_npy, spec_raw, start_frame, end_frame, flag



def cut_video_and_spec_bias(video, spec, bias):

    spec_raw = spec
    video_npy = video
    video_len = 40
    start_frame = random.randint(0, 40 - 4 * 4 - 1)
    truncate_sec = 4
    fps = 4
    sr = 16000
    hop_size = 250

    truncate_frame = truncate_sec * fps
    end_frame = start_frame + truncate_frame

    # Decode:
    stream = io.BytesIO(spec_raw)
    spec_raw = numpy.lib.format.read_array(stream)

    stream = io.BytesIO(video_npy)
    video_npy = numpy.lib.format.read_array(stream)

    # Check Video npy:
    if video_npy.shape[0] < start_frame + truncate_frame:
        repeat_num = int((start_frame + truncate_frame) // video_npy.shape[0]) + 1
        video_npy = np.tile(video_npy, (repeat_num,1))
        video_npy = video_npy[start_frame: start_frame + truncate_frame]
    else:
        video_npy = video_npy[start_frame: start_frame + truncate_frame]    


    # Bias Frames:
    # print("shift bias: ", bias)
    left_bound  = min(start_frame, bias)
    right_bound = min(video_len - 1 - start_frame - truncate_sec * fps , bias)
    
    bias_frames_start = start_frame + random.randint(-left_bound, right_bound)
    bias_frames_end = bias_frames_start + truncate_frame

    # Start Bias Index:
    # trunc_len = int(self.truncate_sec * self.fps)
    if bias_frames_start > start_frame:
        start_bias_index = torch.tensor([bias_frames_start - start_frame, 0])        # (v1, s1)
        end_bias_index = torch.tensor([truncate_frame - 1, truncate_frame - 1 - (bias_frames_start - start_frame)]) # (v2, s2)
    else:
        start_bias_index = torch.tensor([0, start_frame - bias_frames_start])   # (v1, s1)
        end_bias_index = torch.tensor([truncate_frame - 1 - (start_frame - bias_frames_start), truncate_frame - 1])                                       # (v2, s2)

    # Spec Start & Truncate:
    spec_start = int(bias_frames_start / fps * sr / hop_size)
    spec_truncate = int(truncate_sec * sr / hop_size)

    # Check spec_raw:
    if spec_raw.shape[-1] < spec_start + spec_truncate:
        repeat_num = int((spec_start + spec_truncate) // spec_raw.shape[-1]) + 1
        spec_raw = np.tile(spec_raw, repeat_num)               # repeat 2 
        spec_raw = spec_raw[:, spec_start : spec_start + spec_truncate]
    else:
        spec_raw = spec_raw[:, spec_start : spec_start + spec_truncate]
    
    # Video Tensor Transforms:
    video_npy = transform_video(torch.from_numpy(video_npy))
    spec_raw = torch.from_numpy(spec_raw)
    return video_npy, spec_raw, start_bias_index, end_bias_index







def preprocess(sample, fps=4, video_len=40, truncate_frame=16):
    # image, json = sample
    # print(src)
    # print(sample.keys())
    spec, video = sample["spec.npy"], sample["video.jpg"]
    video, spec, start_frame, end_frame = cut_video_and_spec(video, spec, fps=fps, video_len=video_len, truncate_frame=truncate_frame)
    # data_dict = {}
    # data_dict["spec"] = spec
    # data_dict["video"] = video
    # data_dict["video_time"] = str(start_frame) + "_" + str(end_frame)
    video_time = str(start_frame) + "_" + str(end_frame)
    return spec, video, video_time


def preprocess_filter(sample):
    spec, video = sample["spec.npy"], sample["video.npy"]
    video, spec, start_frame, end_frame, flag = cut_video_and_spec_filter(video, spec)
    video_time = str(start_frame) + "_" + str(end_frame)
    if flag:
        return spec, video, video_time
    else:
        return None

# preprocess_spec_augmentation:
def preprocess_bias(sample, bias=None):
    spec, video = sample["spec.npy"], sample["video.npy"]
    video, spec, start_bias_index, end_bias_index = cut_video_and_spec_bias(video, spec, bias)
    return spec, video, start_bias_index, end_bias_index




def get_wds_dataset_vggsound(args, preprocess_img, is_train, epoch=0, floor=False, tokenizer=None):
    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None
    resampled = getattr(args, 'dataset_resampled', False) and is_train

    num_samples, num_shards = get_dataset_size(input_shards)
    if not num_samples:
        if is_train:
            num_samples = args.train_num_samples
            if not num_samples:
                raise RuntimeError(
                    'Currently, number of dataset samples must be specified for training dataset. '
                    'Please specify via `--train-num-samples` if no dataset length info present.')
        else:
            num_samples = args.val_num_samples or 0  # eval will just exhaust the iterator if not specified

    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc
    
    if resampled:
        pipeline = [ResampledShards2(input_shards, deterministic=True, epoch=shared_epoch)]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    # at this point we have an iterator over all the shards
    if is_train:
        if not resampled:
            pipeline.extend([
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ])
        pipeline.extend([
            # at this point, we have an iterator over the shards assigned to each worker at each node
            tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ])
    else:
        pipeline.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),
        ])
    
    if is_train:
        pipeline.extend([
            # wds.map(preprocess_filter),
            wds.map(preprocess),
            wds.batched(args.batch_size, partial=not is_train),
        ])
    else:
        pipeline.extend([
            wds.map(preprocess),
            wds.batched(args.batch_size, partial=not is_train),
        ])



    dataset = wds.DataPipeline(*pipeline)
    if is_train:
        if not resampled:
            assert num_shards >= args.workers * args.world_size, 'number of shards must be >= total workers'
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=True,
    )

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples
    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)




def get_wds_dataset_vggsound_bias(args, preprocess_img, is_train, epoch=0, floor=False, tokenizer=None):
    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None
    resampled = getattr(args, 'dataset_resampled', False) and is_train

    num_samples, num_shards = get_dataset_size(input_shards)
    if not num_samples:
        if is_train:
            num_samples = args.train_num_samples
            if not num_samples:
                raise RuntimeError(
                    'Currently, number of dataset samples must be specified for training dataset. '
                    'Please specify via `--train-num-samples` if no dataset length info present.')
        else:
            num_samples = args.val_num_samples or 0  # eval will just exhaust the iterator if not specified

    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc
    
    if resampled:
        pipeline = [ResampledShards2(input_shards, deterministic=True, epoch=shared_epoch)]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    # at this point we have an iterator over all the shards
    if is_train:
        if not resampled:
            pipeline.extend([
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ])
        pipeline.extend([
            # at this point, we have an iterator over the shards assigned to each worker at each node
            tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ])
    else:
        pipeline.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),
        ])
    pipeline.extend([
        wds.map(lambda x: preprocess_bias(x, bias=args.shift)),
        wds.batched(args.batch_size, partial=not is_train),
    ])

    dataset = wds.DataPipeline(*pipeline)
    if is_train:
        if not resampled:
            assert num_shards >= args.workers * args.world_size, 'number of shards must be >= total workers'
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=True,
    )

    # FIXME not clear which approach is better, with_epoch before vs after dataloader?
    # hoping to resolve via https://github.com/webdataset/webdataset/issues/169
    # if is_train:
    #     # roll over and repeat a few samples to get same number of full batches on each node
    #     global_batch_size = args.batch_size * args.world_size
    #     num_batches = math.ceil(num_samples / global_batch_size)
    #     num_workers = max(1, args.workers)
    #     num_batches = math.ceil(num_batches / num_workers) * num_workers
    #     num_samples = num_batches * global_batch_size
    #     dataloader = dataloader.with_epoch(num_batches)
    # else:
    #     # last batches are partial, eval is done on single (master) node
    #     num_batches = math.ceil(num_samples / args.batch_size)

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples
    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)




def get_dataset_fn(data_path, dataset_type):

    if dataset_type == "webdataset":
        return get_wds_dataset
    
    elif dataset_type == "vggsound":
        return get_spec_and_audio_dataset
    
    elif dataset_type == "vggsound_bias":
        return get_spec_and_audio_dataset_bias

    elif dataset_type == "vggsound_webdataset":
        return get_wds_dataset_vggsound
    
    elif dataset_type == "vggsound_webdataset_intra_contrast":
        return get_wds_dataset_vggsound_intra_contrast
    
    elif dataset_type == "vggsound_audioset_webdataset":
        return get_wds_dataset_vggsound_audioset
    
    elif dataset_type == "vggsound_audioset_webdataset_intra_contrast":
        return get_wds_dataset_vggsound_audioset_intra_contrast

    elif dataset_type == "vggsound_audioset_music_webdataset_intra_contrast":
        return get_wds_dataset_vggsound_audioset_music_intra_contrast

    elif dataset_type == "vggsound_audioset_music_soundnet_webdataset_intra_contrast":
        return get_wds_dataset_vggsound_audioset_music_soundnet_intra_contrast
    
    elif dataset_type == "vggsound_audioset_music_soundnet_filter_webdataset_intra_contrast":
        return get_wds_dataset_vggsound_audioset_music_soundnet_filter_intra_contrast

    elif dataset_type == "vggsound_audioset_music_soundnet_filter_entropy_webdataset_intra_contrast":
        return get_wds_dataset_vggsound_audioset_music_soundnet_filter_entropy_intra_contrast
    
    elif dataset_type == "vggsound_fps8_intra_contrast":
        return get_wds_dataset_vggsound_fps8_intra_contrast

    elif dataset_type == "vggsound_audioset_music_fps8_intra_contrast":
        return get_wds_dataset_vggsound_audioset_music_fps8_intra_contrast

    elif dataset_type == "vggsound_webdataset_bias":
        return get_wds_dataset_vggsound_bias


    elif dataset_type == "csv":
        return get_csv_dataset
    elif dataset_type == "synthetic":
        return get_synthetic_dataset
    elif dataset_type == "auto":
        ext = data_path.split('.')[-1]
        if ext in ['csv', 'tsv']:
            return get_csv_dataset
        elif ext in ['tar']:
            return get_wds_dataset
        else:
            raise ValueError(
                f"Tried to figure out dataset type, but failed for extension {ext}.")
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    

def get_data(args, preprocess_fns, epoch=0, tokenizer=None):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if args.train_data or args.dataset_type == "synthetic":
        data["train"] = get_dataset_fn(args.train_data, args.dataset_type)(
            args, preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer)

    if args.val_data:
        data["val"] = get_dataset_fn(args.val_data, args.dataset_type)(
            args, preprocess_val, is_train=False, tokenizer=tokenizer)

    if args.imagenet_val is not None:
        data["imagenet-val"] = get_imagenet(args, preprocess_fns, "val")

    if args.imagenet_v2 is not None:
        data["imagenet-v2"] = get_imagenet(args, preprocess_fns, "v2")

    return data





## Bias Dataset:
class VGGSound_audio_spec_Dataset_Bias(torch.utils.data.Dataset):
    # Only Load audio dataset: for training Stage1: Audio Npy Dataset
    def __init__(self, split, data_dir, split_txt_path, transforms=None, sr=16000, duration=10, truncate_sec=4, fps=4, subset_num=False, fix_frames=False, video_len=40, hop_size=250, shape=(224, 224)):
        super().__init__()
        self.data_dir = data_dir

        if split == "train":
            self.split_path = os.path.join(self.data_dir, "Train")  # spec dir
            self.split = "Train"
        elif split == "valid" or split == 'test':
            self.split_path = os.path.join(self.data_dir, "Test")   # spec dir
            self.split = "Test"

        # Default params:
        self.sr = sr                        # 22050
        self.duration = duration            # 10
        self.truncate_sec = truncate_sec    # 8
        self.fps = fps
        self.fix_frames = fix_frames
        self.video_len = video_len          # default: 40 Frames
        self.hop_size = hop_size            # hop_size: 250
        self.shape = shape
        print("Fix Frames: {}".format(self.fix_frames))

        # Update dir:
        self.video_dir = os.path.join(self.data_dir, self.split, "video_fps3.9")
        self.video_npy_dir = os.path.join(self.data_dir, self.split, "video_fps3.9_npy")
        self.spec_dir = os.path.join(self.data_dir, self.split, "audio_npy_spec_640spec")

        # Data list:
        with open(os.path.join(split_txt_path, '{}_clean_success.txt'.format(self.split)), "r") as f:
            data_list = f.readlines()
            data_list = list(map(lambda x: x.strip(), data_list))
            data_list = list(map(lambda x: x + '_mel.npy', data_list))      # spec

        # data_list = data_list[:32]
        random.shuffle(data_list)
        if subset_num:
            if self.split == "Train":
                self.sample_list = data_list[:subset_num]
            else:
                self.sample_list = data_list[: 2000]
        else:
            self.sample_list = data_list
        print('Split: {}  Sample Num: {}'.format(split, len(self.sample_list)))


    def __len__(self):
        return len(self.sample_list)
    

    def transform_video(self, video_tensor):
        # T x 3 x H x W
        video_tensor = transform_f.resize(video_tensor, self.shape)
        # Normalize:
        video_tensor = video_tensor.div(255)
        return video_tensor


    def load_video_and_spec_bias(self, video_npy_path, spec_path):

        spec_raw = np.load(spec_path).astype(np.float32)
        video_npy = np.load(video_npy_path).astype(np.uint8)

        # Start Frame:
        start_frame = random.randint(0, self.video_len - self.truncate_sec * self.fps - 1) 
        truncate_frame = self.truncate_sec * self.fps
        end_frame = start_frame + truncate_frame

        # Check Video npy:
        if video_npy.shape[0] < start_frame + truncate_frame:
            repeat_num = int((start_frame + truncate_frame) // video_npy.shape[0]) + 1
            video_npy = np.tile(video_npy, (repeat_num,1))
            video_npy = video_npy[start_frame: start_frame + truncate_frame]
        else:
            video_npy = video_npy[start_frame: start_frame + truncate_frame]    


        # Bias Frames:
        left_bound  = min(start_frame, self.truncate_sec * self.fps // 2)
        right_bound = min(self.video_len - 1 - start_frame - self.truncate_sec * self.fps , self.truncate_sec * self.fps // 2)
        
        bias_frames_start = start_frame + random.randint(-left_bound, right_bound)
        bias_frames_end = bias_frames_start + truncate_frame

        # Start Bias Index:
        # trunc_len = int(self.truncate_sec * self.fps)
        if bias_frames_start > start_frame:
            start_bias_index = torch.tensor([bias_frames_start - start_frame, 0])        # (v1, s1)
            end_bias_index = torch.tensor([truncate_frame - 1, truncate_frame - 1 - (bias_frames_start - start_frame)]) # (v2, s2)
        else:
            start_bias_index = torch.tensor([0, start_frame - bias_frames_start])   # (v1, s1)
            end_bias_index = torch.tensor([truncate_frame - 1 - (start_frame - bias_frames_start), truncate_frame - 1])                                       # (v2, s2)

        # Spec Start & Truncate:
        spec_start = int(bias_frames_start / self.fps * self.sr / self.hop_size)
        spec_truncate = int(self.truncate_sec * self.sr / self.hop_size)

        # Check spec_raw:
        if spec_raw.shape[-1] < spec_start + spec_truncate:
            repeat_num = int((spec_start + spec_truncate) // spec_raw.shape[-1]) + 1
            spec_raw = np.tile(spec_raw, repeat_num)               # repeat 2 
            spec_raw = spec_raw[:, spec_start : spec_start + spec_truncate]
        else:
            spec_raw = spec_raw[:, spec_start : spec_start + spec_truncate]
        
        # Video Tensor Transforms:
        video_npy = self.transform_video(torch.from_numpy(video_npy))
        spec_raw = torch.from_numpy(spec_raw)
        return video_npy, spec_raw, start_bias_index, end_bias_index


    def __getitem__(self, idx):
        audio_name = self.sample_list[idx]             

        # video_name:
        video_name = os.path.basename(audio_name)[:-8]      # remove _mel.npy
        video_npy_path = os.path.join(self.video_npy_dir, video_name + '.npy')
        spec_npy_path = os.path.join(self.spec_dir, audio_name)

        # Load Video & Spec npy:
        video, spec, start_bias_index, end_bias_index = self.load_video_and_spec_bias(video_npy_path, spec_npy_path)

        # video_path:
        video_path = os.path.join(self.video_dir, video_name + '.mp4')
        
        data_dict = {}
        data_dict['video'] = video
        data_dict['spec'] = spec
        data_dict['audio_name'] = audio_name
        data_dict['video_path'] =  video_path
        # data_dict['video_time'] = str(start_frame) + '_' + str(end_frame)
        data_dict["start_bias_index"] = start_bias_index
        data_dict["end_bias_index"] = end_bias_index
        return data_dict







def get_wds_dataset_vggsound_audioset(args, preprocess_img, is_train, epoch=0, floor=False, tokenizer=None):
    input_shards = args.train_data if is_train else args.val_data
    print(input_shards)
    if is_train and args.train_data == "audioset_vggsound":
        input_shards = ["/localdata_ssd/lsm/contrastive_pretrain_webdataset/vggsound_contrastive_lightweight_webdataset/Train/vggsound-{}.tar".format(str(i).zfill(6)) for i in range(32)]
        input_shards.extend(["/localdata_ssd/lsm/contrastive_pretrain_webdataset/Audioset_contrastive_webdataset/Train/audioset-{}.tar".format(str(i).zfill(6)) for i in range(32)])
        print("Shard List: ", input_shards)    

    assert input_shards is not None
    resampled = getattr(args, 'dataset_resampled', False) and is_train

    num_samples, num_shards = get_dataset_size(input_shards)
    if not num_samples:
        if is_train:
            num_samples = args.train_num_samples
            if not num_samples:
                raise RuntimeError(
                    'Currently, number of dataset samples must be specified for training dataset. '
                    'Please specify via `--train-num-samples` if no dataset length info present.')
        else:
            num_samples = args.val_num_samples or 0  # eval will just exhaust the iterator if not specified

    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc
    
    if resampled:
        pipeline = [ResampledShards2(input_shards, deterministic=True, epoch=shared_epoch)]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    # at this point we have an iterator over all the shards
    if is_train:
        if not resampled:
            pipeline.extend([
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ])
        pipeline.extend([
            # at this point, we have an iterator over the shards assigned to each worker at each node
            tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ])
    else:
        pipeline.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),
        ])
    pipeline.extend([
        wds.map(preprocess_vggsound_audioset),
        wds.batched(args.batch_size, partial=not is_train),
    ])

    dataset = wds.DataPipeline(*pipeline)
    if is_train:
        if not resampled:
            assert num_shards >= args.workers * args.world_size, 'number of shards must be >= total workers'
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=True,
    )

    # FIXME not clear which approach is better, with_epoch before vs after dataloader?
    # hoping to resolve via https://github.com/webdataset/webdataset/issues/169
    # if is_train:
    #     # roll over and repeat a few samples to get same number of full batches on each node
    #     global_batch_size = args.batch_size * args.world_size
    #     num_batches = math.ceil(num_samples / global_batch_size)
    #     num_workers = max(1, args.workers)
    #     num_batches = math.ceil(num_batches / num_workers) * num_workers
    #     num_samples = num_batches * global_batch_size
    #     dataloader = dataloader.with_epoch(num_batches)
    # else:
    #     # last batches are partial, eval is done on single (master) node
    #     num_batches = math.ceil(num_samples / args.batch_size)n

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples
    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


# VGGSound + AudioSet (+ ~ 50K) + AudioSet_Music
def get_wds_dataset_vggsound_audioset_music_intra_contrast(args, preprocess_img, is_train, epoch=0, floor=False, tokenizer=None):
    input_shards = args.train_data if is_train else args.val_data
    print(input_shards)
    if is_train and args.train_data == "audioset_vggsound_music":
        input_shards = ["/localdata_ssd/lsm/Stage1_Pretrained_Packed/VGGSound/Train/vggsound-{}.tar".format(str(i).zfill(6)) for i in range(32)]
        input_shards.extend(["/localdata_ssd/lsm/Stage1_Pretrained_Packed/AudioSet/Train/audioset-{}.tar".format(str(i).zfill(6)) for i in range(32)])
        input_shards.extend(["/localdata_ssd/lsm/Stage1_Pretrained_Packed/AudioSet_Music/Train/audioset_music-{}.tar".format(str(i).zfill(6)) for i in range(8)])
        print("Shard List: ", input_shards)    

    assert input_shards is not None
    resampled = getattr(args, 'dataset_resampled', False) and is_train

    num_samples, num_shards = get_dataset_size(input_shards)
    if not num_samples:
        if is_train:
            num_samples = args.train_num_samples
            if not num_samples:
                raise RuntimeError(
                    'Currently, number of dataset samples must be specified for training dataset. '
                    'Please specify via `--train-num-samples` if no dataset length info present.')
        else:
            num_samples = args.val_num_samples or 0  # eval will just exhaust the iterator if not specified

    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc
    
    if resampled:
        pipeline = [ResampledShards2(input_shards, deterministic=True, epoch=shared_epoch)]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    # at this point we have an iterator over all the shards
    if is_train:
        if not resampled:
            pipeline.extend([
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ])
        pipeline.extend([
            # at this point, we have an iterator over the shards assigned to each worker at each node
            tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ])
    else:
        pipeline.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),
        ])

    sample_clip_num = args.intra_clip_num
    shift_lb = args.shift_lb


    if is_train:
        pipeline.extend([
            # wds.map(preprocess_filter),
            # functools.partial(marginal_prob_std, sigma=sigma)
            wds.map(functools.partial(preprocess_vggsound_audioset_temporal_contrast, sample_num=sample_clip_num, shift_lb=shift_lb)),
            # wds.map(preprocess_temporal_contrast),
            wds.batched(args.batch_size, partial=not is_train),
        ])
    else:
        pipeline.extend([
            wds.map(preprocess_vggsound_audioset),
            wds.batched(args.batch_size, partial=not is_train),
        ])
        
    # pipeline.extend([
    #     wds.map(preprocess_vggsound_audioset),
    #     wds.batched(args.batch_size, partial=not is_train),
    # ])

    dataset = wds.DataPipeline(*pipeline)
    if is_train:
        if not resampled:
            assert num_shards >= args.workers * args.world_size, 'number of shards must be >= total workers'
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=True,
    )

    # FIXME not clear which approach is better, with_epoch before vs after dataloader?
    # hoping to resolve via https://github.com/webdataset/webdataset/issues/169
    # if is_train:
    #     # roll over and repeat a few samples to get same number of full batches on each node
    #     global_batch_size = args.batch_size * args.world_size
    #     num_batches = math.ceil(num_samples / global_batch_size)
    #     num_workers = max(1, args.workers)
    #     num_batches = math.ceil(num_batches / num_workers) * num_workers
    #     num_samples = num_batches * global_batch_size
    #     dataloader = dataloader.with_epoch(num_batches)
    # else:
    #     # last batches are partial, eval is done on single (master) node
    #     num_batches = math.ceil(num_samples / args.batch_size)n

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples
    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)





# VGGSound + AudioSet (+ ~ 50K) + AudioSet_Music + SoundNet
def get_wds_dataset_vggsound_audioset_music_soundnet_intra_contrast(args, preprocess_img, is_train, epoch=0, floor=False, tokenizer=None):
    input_shards = args.train_data if is_train else args.val_data
    print(input_shards)
    if is_train and args.train_data == "audioset_vggsound_music_soundnet":
        input_shards = ["/localdata_ssd/lsm/Stage1_Pretrained_Packed/VGGSound/Train/vggsound-{}.tar".format(str(i).zfill(6)) for i in range(32)]
        input_shards.extend(["/localdata_ssd/lsm/Stage1_Pretrained_Packed/AudioSet/Train/audioset-{}.tar".format(str(i).zfill(6)) for i in range(32)])
        input_shards.extend(["/localdata_ssd/lsm/Stage1_Pretrained_Packed/AudioSet_Music/Train/audioset_music-{}.tar".format(str(i).zfill(6)) for i in range(8)])
        input_shards.extend(["/localdata_ssd/lsm/Stage1_Pretrained_Packed/SoundNet/Train/soundnet-{}.tar".format(str(i).zfill(6)) for i in range(48)])
        print("Shard List: ", input_shards)    

    assert input_shards is not None
    resampled = getattr(args, 'dataset_resampled', False) and is_train

    num_samples, num_shards = get_dataset_size(input_shards)
    if not num_samples:
        if is_train:
            num_samples = args.train_num_samples
            if not num_samples:
                raise RuntimeError(
                    'Currently, number of dataset samples must be specified for training dataset. '
                    'Please specify via `--train-num-samples` if no dataset length info present.')
        else:
            num_samples = args.val_num_samples or 0  # eval will just exhaust the iterator if not specified

    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc
    
    if resampled:
        pipeline = [ResampledShards2(input_shards, deterministic=True, epoch=shared_epoch)]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    # at this point we have an iterator over all the shards
    if is_train:
        if not resampled:
            pipeline.extend([
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ])
        pipeline.extend([
            # at this point, we have an iterator over the shards assigned to each worker at each node
            tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ])
    else:
        pipeline.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),
        ])

    sample_clip_num = args.intra_clip_num
    shift_lb = args.shift_lb


    if is_train:
        pipeline.extend([
            # wds.map(preprocess_filter),
            # functools.partial(marginal_prob_std, sigma=sigma)
            wds.map(functools.partial(preprocess_vggsound_audioset_temporal_contrast, sample_num=sample_clip_num, shift_lb=shift_lb)),
            # wds.map(preprocess_temporal_contrast),
            wds.batched(args.batch_size, partial=not is_train),
        ])
    else:
        pipeline.extend([
            wds.map(preprocess_vggsound_audioset),
            wds.batched(args.batch_size, partial=not is_train),
        ])
        
    # pipeline.extend([
    #     wds.map(preprocess_vggsound_audioset),
    #     wds.batched(args.batch_size, partial=not is_train),
    # ])

    dataset = wds.DataPipeline(*pipeline)
    if is_train:
        if not resampled:
            assert num_shards >= args.workers * args.world_size, 'number of shards must be >= total workers'
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=True,
    )

    # FIXME not clear which approach is better, with_epoch before vs after dataloader?
    # hoping to resolve via https://github.com/webdataset/webdataset/issues/169
    # if is_train:
    #     # roll over and repeat a few samples to get same number of full batches on each node
    #     global_batch_size = args.batch_size * args.world_size
    #     num_batches = math.ceil(num_samples / global_batch_size)
    #     num_workers = max(1, args.workers)
    #     num_batches = math.ceil(num_batches / num_workers) * num_workers
    #     num_samples = num_batches * global_batch_size
    #     dataloader = dataloader.with_epoch(num_batches)
    # else:
    #     # last batches are partial, eval is done on single (master) node
    #     num_batches = math.ceil(num_samples / args.batch_size)n

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples
    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)



# VGGSound + AudioSet (+ ~ 50K) + AudioSet_Music + SoundNet + filter
def get_wds_dataset_vggsound_audioset_music_soundnet_filter_intra_contrast(args, preprocess_img, is_train, epoch=0, floor=False, tokenizer=None):
    input_shards = args.train_data if is_train else args.val_data
    print(input_shards)
    if is_train and args.train_data == "audioset_vggsound_music_soundnet_filter":
        input_shards = ["/localdata_ssd/lsm/Stage1_Pretrained_Packed/VGGSound/Train/vggsound-{}.tar".format(str(i).zfill(6)) for i in range(32)]
        input_shards.extend(["/localdata_ssd/lsm/Stage1_Pretrained_Packed/AudioSet/Train/audioset-{}.tar".format(str(i).zfill(6)) for i in range(32)])
        input_shards.extend(["/localdata_ssd/lsm/Stage1_Pretrained_Packed/AudioSet_Music/Train/audioset_music-{}.tar".format(str(i).zfill(6)) for i in range(8)])
        input_shards.extend(["/localdata_ssd/lsm/Stage1_Pretrained_Packed/SoundNet_filter/Train/soundnet-{}.tar".format(str(i).zfill(6)) for i in range(32)])
        print("Shard List: ", input_shards)    

    assert input_shards is not None
    resampled = getattr(args, 'dataset_resampled', False) and is_train

    num_samples, num_shards = get_dataset_size(input_shards)
    if not num_samples:
        if is_train:
            num_samples = args.train_num_samples
            if not num_samples:
                raise RuntimeError(
                    'Currently, number of dataset samples must be specified for training dataset. '
                    'Please specify via `--train-num-samples` if no dataset length info present.')
        else:
            num_samples = args.val_num_samples or 0  # eval will just exhaust the iterator if not specified

    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc
    
    if resampled:
        pipeline = [ResampledShards2(input_shards, deterministic=True, epoch=shared_epoch)]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    # at this point we have an iterator over all the shards
    if is_train:
        if not resampled:
            pipeline.extend([
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ])
        pipeline.extend([
            # at this point, we have an iterator over the shards assigned to each worker at each node
            tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ])
    else:
        pipeline.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),
        ])

    sample_clip_num = args.intra_clip_num
    shift_lb = args.shift_lb


    if is_train:
        pipeline.extend([
            # wds.map(preprocess_filter),
            # functools.partial(marginal_prob_std, sigma=sigma)
            wds.map(functools.partial(preprocess_vggsound_audioset_temporal_contrast, sample_num=sample_clip_num, shift_lb=shift_lb)),
            # wds.map(preprocess_temporal_contrast),
            wds.batched(args.batch_size, partial=not is_train),
        ])
    else:
        pipeline.extend([
            wds.map(preprocess_vggsound_audioset),
            wds.batched(args.batch_size, partial=not is_train),
        ])
        
    # pipeline.extend([
    #     wds.map(preprocess_vggsound_audioset),
    #     wds.batched(args.batch_size, partial=not is_train),
    # ])

    dataset = wds.DataPipeline(*pipeline)
    if is_train:
        if not resampled:
            assert num_shards >= args.workers * args.world_size, 'number of shards must be >= total workers'
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=True,
    )

    # FIXME not clear which approach is better, with_epoch before vs after dataloader?
    # hoping to resolve via https://github.com/webdataset/webdataset/issues/169
    # if is_train:
    #     # roll over and repeat a few samples to get same number of full batches on each node
    #     global_batch_size = args.batch_size * args.world_size
    #     num_batches = math.ceil(num_samples / global_batch_size)
    #     num_workers = max(1, args.workers)
    #     num_batches = math.ceil(num_batches / num_workers) * num_workers
    #     num_samples = num_batches * global_batch_size
    #     dataloader = dataloader.with_epoch(num_batches)
    # else:
    #     # last batches are partial, eval is done on single (master) node
    #     num_batches = math.ceil(num_samples / args.batch_size)n

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples
    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)




# VGGSound + AudioSet (+ ~ 50K) + AudioSet_Music + SoundNet + filter + entropy
def get_wds_dataset_vggsound_audioset_music_soundnet_filter_entropy_intra_contrast(args, preprocess_img, is_train, epoch=0, floor=False, tokenizer=None):
    input_shards = args.train_data if is_train else args.val_data
    print(input_shards)
    if is_train and args.train_data == "audioset_vggsound_music_soundnet_filter_entropy":
        input_shards = ["/localdata_ssd/lsm/Stage1_Pretrained_Packed/VGGSound/Train/vggsound-{}.tar".format(str(i).zfill(6)) for i in range(32)]
        input_shards.extend(["/localdata_ssd/lsm/Stage1_Pretrained_Packed/AudioSet/Train/audioset-{}.tar".format(str(i).zfill(6)) for i in range(32)])
        input_shards.extend(["/localdata_ssd/lsm/Stage1_Pretrained_Packed/AudioSet_Music/Train/audioset_music-{}.tar".format(str(i).zfill(6)) for i in range(8)])
        input_shards.extend(["/localdata_ssd/lsm/Stage1_Pretrained_Packed/SoundNet_filter_speech_entropy/Train/soundnet-{}.tar".format(str(i).zfill(6)) for i in range(32)])
        print("Shard List: ", input_shards)    

    assert input_shards is not None
    resampled = getattr(args, 'dataset_resampled', False) and is_train

    num_samples, num_shards = get_dataset_size(input_shards)
    if not num_samples:
        if is_train:
            num_samples = args.train_num_samples
            if not num_samples:
                raise RuntimeError(
                    'Currently, number of dataset samples must be specified for training dataset. '
                    'Please specify via `--train-num-samples` if no dataset length info present.')
        else:
            num_samples = args.val_num_samples or 0  # eval will just exhaust the iterator if not specified

    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc
    
    if resampled:
        pipeline = [ResampledShards2(input_shards, deterministic=True, epoch=shared_epoch)]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    # at this point we have an iterator over all the shards
    if is_train:
        if not resampled:
            pipeline.extend([
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ])
        pipeline.extend([
            # at this point, we have an iterator over the shards assigned to each worker at each node
            tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ])
    else:
        pipeline.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),
        ])

    sample_clip_num = args.intra_clip_num
    shift_lb = args.shift_lb


    if is_train:
        pipeline.extend([
            # wds.map(preprocess_filter),
            # functools.partial(marginal_prob_std, sigma=sigma)
            wds.map(functools.partial(preprocess_vggsound_audioset_temporal_contrast, sample_num=sample_clip_num, shift_lb=shift_lb)),
            # wds.map(preprocess_temporal_contrast),
            wds.batched(args.batch_size, partial=not is_train),
        ])
    else:
        pipeline.extend([
            wds.map(preprocess_vggsound_audioset),
            wds.batched(args.batch_size, partial=not is_train),
        ])
        
    # pipeline.extend([
    #     wds.map(preprocess_vggsound_audioset),
    #     wds.batched(args.batch_size, partial=not is_train),
    # ])

    dataset = wds.DataPipeline(*pipeline)
    if is_train:
        if not resampled:
            assert num_shards >= args.workers * args.world_size, 'number of shards must be >= total workers'
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=True,
    )

    # FIXME not clear which approach is better, with_epoch before vs after dataloader?
    # hoping to resolve via https://github.com/webdataset/webdataset/issues/169
    # if is_train:
    #     # roll over and repeat a few samples to get same number of full batches on each node
    #     global_batch_size = args.batch_size * args.world_size
    #     num_batches = math.ceil(num_samples / global_batch_size)
    #     num_workers = max(1, args.workers)
    #     num_batches = math.ceil(num_batches / num_workers) * num_workers
    #     num_samples = num_batches * global_batch_size
    #     dataloader = dataloader.with_epoch(num_batches)
    # else:
    #     # last batches are partial, eval is done on single (master) node
    #     num_batches = math.ceil(num_samples / args.batch_size)n

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples
    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)



def get_wds_dataset_vggsound_audioset_intra_contrast(args, preprocess_img, is_train, epoch=0, floor=False, tokenizer=None):
    input_shards = args.train_data if is_train else args.val_data
    print(input_shards)
    if is_train and args.train_data == "audioset_vggsound":
        input_shards = ["/localdata_ssd/lsm/contrastive_pretrain_webdataset/vggsound_contrastive_lightweight_webdataset/Train/vggsound-{}.tar".format(str(i).zfill(6)) for i in range(32)]
        input_shards.extend(["/localdata_ssd/lsm/contrastive_pretrain_webdataset/Audioset_contrastive_webdataset/Train/audioset-{}.tar".format(str(i).zfill(6)) for i in range(32)])
        print("Shard List: ", input_shards)    

    assert input_shards is not None
    resampled = getattr(args, 'dataset_resampled', False) and is_train

    num_samples, num_shards = get_dataset_size(input_shards)
    if not num_samples:
        if is_train:
            num_samples = args.train_num_samples
            if not num_samples:
                raise RuntimeError(
                    'Currently, number of dataset samples must be specified for training dataset. '
                    'Please specify via `--train-num-samples` if no dataset length info present.')
        else:
            num_samples = args.val_num_samples or 0  # eval will just exhaust the iterator if not specified

    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc
    
    if resampled:
        pipeline = [ResampledShards2(input_shards, deterministic=True, epoch=shared_epoch)]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    # at this point we have an iterator over all the shards
    if is_train:
        if not resampled:
            pipeline.extend([
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ])
        pipeline.extend([
            # at this point, we have an iterator over the shards assigned to each worker at each node
            tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ])
    else:
        pipeline.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),
        ])

    sample_clip_num = args.intra_clip_num
    shift_lb = args.shift_lb


    if is_train:
        pipeline.extend([
            # wds.map(preprocess_filter),
            # functools.partial(marginal_prob_std, sigma=sigma)
            wds.map(functools.partial(preprocess_vggsound_audioset_temporal_contrast, sample_num=sample_clip_num, shift_lb=shift_lb)),
            # wds.map(preprocess_temporal_contrast),
            wds.batched(args.batch_size, partial=not is_train),
        ])
    else:
        pipeline.extend([
            wds.map(preprocess_vggsound_audioset),
            wds.batched(args.batch_size, partial=not is_train),
        ])
        
    # pipeline.extend([
    #     wds.map(preprocess_vggsound_audioset),
    #     wds.batched(args.batch_size, partial=not is_train),
    # ])

    dataset = wds.DataPipeline(*pipeline)
    if is_train:
        if not resampled:
            assert num_shards >= args.workers * args.world_size, 'number of shards must be >= total workers'
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=True,
    )

    # FIXME not clear which approach is better, with_epoch before vs after dataloader?
    # hoping to resolve via https://github.com/webdataset/webdataset/issues/169
    # if is_train:
    #     # roll over and repeat a few samples to get same number of full batches on each node
    #     global_batch_size = args.batch_size * args.world_size
    #     num_batches = math.ceil(num_samples / global_batch_size)
    #     num_workers = max(1, args.workers)
    #     num_batches = math.ceil(num_batches / num_workers) * num_workers
    #     num_samples = num_batches * global_batch_size
    #     dataloader = dataloader.with_epoch(num_batches)
    # else:
    #     # last batches are partial, eval is done on single (master) node
    #     num_batches = math.ceil(num_samples / args.batch_size)n

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples
    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)



def preprocess_vggsound_audioset_temporal_contrast(sample, sample_num=4, shift_lb=8):
    # image, json = sample
    # print(src)
    # print(sample.keys())
    spec, video = sample["spec.npy"], sample["video.jpg"]
    video, spec, start_frame, end_frame = cut_video_and_spec_vggsound_audioset_temporal_contrast(video, spec, sample_num=sample_num, shift_lb=shift_lb)
    # data_dict = {}
    # data_dict["spec"] = spec
    # data_dict["video"] = video
    # data_dict["video_time"] = str(start_frame) + "_" + str(end_frame)
    video_time = str(start_frame) + "_" + str(end_frame)
    return spec, video, video_time




def preprocess_vggsound_audioset(sample):
    # image, json = sample
    # print(src)
    # print(sample.keys())
    spec, video = sample["spec.npy"], sample["video.jpg"]
    video, spec, start_frame, end_frame = cut_video_and_spec_vggsound_audioset(video, spec)
    # data_dict = {}
    # data_dict["spec"] = spec
    # data_dict["video"] = video
    # data_dict["video_time"] = str(start_frame) + "_" + str(end_frame)
    video_time = str(start_frame) + "_" + str(end_frame)
    return spec, video, video_time


def cut_video_and_spec_vggsound_audioset(video, spec):

    spec_raw = spec
    video_npy = video

    start_frame = random.randint(0, 40 - 4 * 4 - 1)
    truncate_sec = 4
    fps = 4
    sr = 16000
    hop_size = 250
    shape_h = 224

    truncate_frame = truncate_sec * fps
    end_frame = start_frame + truncate_frame

    # Spec Start & Truncate:
    spec_start = int(start_frame / fps * sr / hop_size)
    spec_truncate = int(truncate_sec * sr / hop_size)


    stream = io.BytesIO(spec_raw)
    spec_raw = numpy.lib.format.read_array(stream)

    stream = io.BytesIO(video_npy)
    video_npy = Image.open(stream)
    video_npy = np.array(video_npy)
    # video_npy = numpy.lib.format.read_array(stream)

        # return numpy.lib.format.read_array(stream)
    
    # Check spec_raw:
    if spec_raw.shape[-1] < spec_start + spec_truncate:
        repeat_num = int((spec_start + spec_truncate) // spec_raw.shape[-1]) + 1
        spec_raw = np.tile(spec_raw, repeat_num)               # repeat 2 
        spec_raw = spec_raw[:, spec_start : spec_start + spec_truncate]
    else:
        spec_raw = spec_raw[:, spec_start : spec_start + spec_truncate]
    

    # transpose:
    video_npy = video_npy.reshape(shape_h, -1, shape_h, 3).transpose(1,3,0,2)    # T x 3 x H x W

    # Check Video npy:
    if video_npy.shape[0] < start_frame + truncate_frame:
        repeat_num = int((start_frame + truncate_frame) // video_npy.shape[0]) + 1
        video_npy = np.tile(video_npy, (repeat_num,1))
        video_npy = video_npy[start_frame: start_frame + truncate_frame]
    else:
        video_npy = video_npy[start_frame: start_frame + truncate_frame]

    # Video Tensor Transforms:
    video_npy = transform_video(torch.from_numpy(video_npy))
    spec_raw = torch.from_numpy(spec_raw)
    return video_npy, spec_raw, start_frame, end_frame



def cut_video_and_spec_vggsound_audioset_temporal_contrast(video, spec, sample_num=4, shift_lb=8):
    """
    For Temporal Contrast Intra Contrastive:
    Sample Different segments from the same data
    Output:
        video: sample_num x T x C x H x W
        spec:  sample_num x Mel_num x T'
    Tempora Shift >= 8 Frames (2s)
    """

    spec_raw = spec
    video_npy = video
    assert sample_num == 2 or sample_num == 3 or sample_num == 4 , "sample num must be [2,3,4]"
    start_frame_index_list, end_frame_index_list = sample_temporal_index(sample_num=sample_num, shift_lb=shift_lb)
    start_spec_list = []

    truncate_sec = 4
    fps = 4
    sr = 16000
    hop_size = 250
    shape_h = 224
    truncate_frame = truncate_sec * fps

    # Get Spec Index list:
    spec_truncate = int(truncate_sec * sr / hop_size)
    for i in range(len(start_frame_index_list)):
        start_frame = start_frame_index_list[i]
        spec_start = int(start_frame / fps * sr / hop_size)
        start_spec_list.append(spec_start)
    
    # Stream:
    stream = io.BytesIO(spec_raw)
    spec_raw = numpy.lib.format.read_array(stream)

    stream = io.BytesIO(video_npy)
    # video_npy = numpy.lib.format.read_array(stream)
    video_npy = Image.open(stream)
    video_npy = np.array(video_npy)
    # video transpose:
    video_npy = video_npy.reshape(shape_h, -1, shape_h, 3).transpose(1,3,0,2)    # T x 3 x H x W

    # Check Spec_Raw and Crop Spec:
    sample_spec_list = []
    for i in range(sample_num):
        spec_start = start_spec_list[i]
        if spec_raw.shape[-1] < spec_start + spec_truncate:
            repeat_num = int((spec_start + spec_truncate) // spec_raw.shape[-1]) + 1
            spec_raw = np.tile(spec_raw, repeat_num)               # repeat 2 
            sample_spec = spec_raw[:, spec_start : spec_start + spec_truncate]
        else:
            sample_spec = spec_raw[:, spec_start : spec_start + spec_truncate]
        sample_spec_list.append(sample_spec[None])
    
    # Check Video npy: 
    sample_video_list = []
    for i in range(sample_num):
        start_frame, end_frame = start_frame_index_list[i], end_frame_index_list[i]
        if video_npy.shape[0] < end_frame:
            repeat_num = int((start_frame + truncate_frame) // video_npy.shape[0]) + 1
            sample_video = np.tile(video_npy, (repeat_num, 1))
            sample_video = video_npy[start_frame: end_frame]
        else:
            sample_video = video_npy[start_frame: end_frame]
        # Video Tensor Transforms:
        sample_video = transform_video(torch.from_numpy(sample_video))
        sample_video_list.append(sample_video.unsqueeze(0))
    
    sample_spec_list = torch.from_numpy(np.concatenate(sample_spec_list, 0))    # sample_num x H x W
    sample_video_list = torch.cat(sample_video_list, 0)                         # sample_num x T x C x H x W
    return sample_video_list, sample_spec_list, start_frame, end_frame





"""Intra Contrast Data Loader"""
def get_wds_dataset_vggsound_intra_contrast(args, preprocess_img, is_train, epoch=0, floor=False, tokenizer=None):
    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None
    resampled = getattr(args, 'dataset_resampled', False) and is_train

    num_samples, num_shards = get_dataset_size(input_shards)
    if not num_samples:
        if is_train:
            num_samples = args.train_num_samples
            if not num_samples:
                raise RuntimeError(
                    'Currently, number of dataset samples must be specified for training dataset. '
                    'Please specify via `--train-num-samples` if no dataset length info present.')
        else:
            num_samples = args.val_num_samples or 0  # eval will just exhaust the iterator if not specified

    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc
    
    if resampled:
        pipeline = [ResampledShards2(input_shards, deterministic=True, epoch=shared_epoch)]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    # at this point we have an iterator over all the shards
    if is_train:
        if not resampled:
            pipeline.extend([
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ])
        pipeline.extend([
            # at this point, we have an iterator over the shards assigned to each worker at each node
            tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ])
    else:
        pipeline.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),
        ])
    
    sample_clip_num = args.intra_clip_num   # For Intra Contrastive
    shift_lb = args.shift_lb

    if is_train:
        pipeline.extend([
            # wds.map(preprocess_filter),
            # functools.partial(marginal_prob_std, sigma=sigma)
            wds.map(functools.partial(preprocess_temporal_contrast, sample_num=sample_clip_num, shift_lb=shift_lb)),
            # wds.map(preprocess_temporal_contrast),
            wds.batched(args.batch_size, partial=not is_train),
        ])
    else:
        pipeline.extend([
            wds.map(preprocess),
            wds.batched(args.batch_size, partial=not is_train),
        ])


    dataset = wds.DataPipeline(*pipeline)
    if is_train:
        if not resampled:
            assert num_shards >= args.workers * args.world_size, 'number of shards must be >= total workers'
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=True,
    )

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples
    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)




"""Intra Contrast Data Loader"""
def get_wds_dataset_vggsound_fps8_intra_contrast(args, preprocess_img, is_train, epoch=0, floor=False, tokenizer=None):
    input_shards = args.train_data if is_train else args.val_data
    if is_train and args.train_data == "vggsound_fps8":
        input_shards = ["/localdata_ssd/lsm/Stage1_Pretrained_Packed/VGGSound_fps8/Train/vggsound-{}.tar".format(str(i).zfill(6)) for i in range(32)]
        # input_shards.extend(["/localdata_ssd/lsm/contrastive_pretrain_webdataset/Audioset_contrastive_webdataset/Train/audioset-{}.tar".format(str(i).zfill(6)) for i in range(32)])
        print("Shard List: ", input_shards)    

    assert input_shards is not None
    resampled = getattr(args, 'dataset_resampled', False) and is_train

    num_samples, num_shards = get_dataset_size(input_shards)
    if not num_samples:
        if is_train:
            num_samples = args.train_num_samples
            if not num_samples:
                raise RuntimeError(
                    'Currently, number of dataset samples must be specified for training dataset. '
                    'Please specify via `--train-num-samples` if no dataset length info present.')
        else:
            num_samples = args.val_num_samples or 0  # eval will just exhaust the iterator if not specified

    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc
    
    if resampled:
        pipeline = [ResampledShards2(input_shards, deterministic=True, epoch=shared_epoch)]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    # at this point we have an iterator over all the shards
    if is_train:
        if not resampled:
            pipeline.extend([
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ])
        pipeline.extend([
            # at this point, we have an iterator over the shards assigned to each worker at each node
            tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ])
    else:
        pipeline.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),
        ])
    
    sample_clip_num = args.intra_clip_num   # For Intra Contrastive
    shift_lb = args.shift_lb

    if is_train:
        pipeline.extend([
            # wds.map(preprocess_filter),
            # functools.partial(marginal_prob_std, sigma=sigma)
            wds.map(functools.partial(preprocess_temporal_contrast, sample_num=sample_clip_num, shift_lb=shift_lb, fps=8, truncate_frame=32, video_len=80)),
            # wds.map(preprocess_temporal_contrast),
            wds.batched(args.batch_size, partial=not is_train),
        ])
    else:
        pipeline.extend([
            wds.map(functools.partial(preprocess, fps=8, video_len=80, truncate_frame=32)),
            wds.batched(args.batch_size, partial=not is_train),
        ])


    dataset = wds.DataPipeline(*pipeline)
    if is_train:
        if not resampled:
            assert num_shards >= args.workers * args.world_size, 'number of shards must be >= total workers'
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=True,
    )

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples
    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


# get_wds_dataset_vggsound_audioset_music_fps8_intra_contrast

"""Intra Contrast Data Loader"""
def get_wds_dataset_vggsound_audioset_music_fps8_intra_contrast(args, preprocess_img, is_train, epoch=0, floor=False, tokenizer=None):
    input_shards = args.train_data if is_train else args.val_data
    if is_train and args.train_data == "vggsound_audioset_music_fps8":
        input_shards = ["/localdata_ssd/lsm/Stage1_Pretrained_Packed/VGGSound_fps8/Train/vggsound-{}.tar".format(str(i).zfill(6)) for i in range(32)]
        input_shards.extend(["/localdata_ssd/lsm/Stage1_Pretrained_Packed/AudioSet_fps8/Train/vggsound-{}.tar".format(str(i).zfill(6)) for i in range(32)])
        input_shards.extend(["/localdata_ssd/lsm/Stage1_Pretrained_Packed/AudioSet_Music_fps8/Train/audioset_music-{}.tar".format(str(i).zfill(6)) for i in range(8)])
        print("Shard List: ", input_shards)    

    assert input_shards is not None
    resampled = getattr(args, 'dataset_resampled', False) and is_train

    num_samples, num_shards = get_dataset_size(input_shards)
    if not num_samples:
        if is_train:
            num_samples = args.train_num_samples
            if not num_samples:
                raise RuntimeError(
                    'Currently, number of dataset samples must be specified for training dataset. '
                    'Please specify via `--train-num-samples` if no dataset length info present.')
        else:
            num_samples = args.val_num_samples or 0  # eval will just exhaust the iterator if not specified

    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc
    
    if resampled:
        pipeline = [ResampledShards2(input_shards, deterministic=True, epoch=shared_epoch)]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    # at this point we have an iterator over all the shards
    if is_train:
        if not resampled:
            pipeline.extend([
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ])
        pipeline.extend([
            # at this point, we have an iterator over the shards assigned to each worker at each node
            tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ])
    else:
        pipeline.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),
        ])
    
    sample_clip_num = args.intra_clip_num   # For Intra Contrastive
    shift_lb = args.shift_lb

    if is_train:
        pipeline.extend([
            # wds.map(preprocess_filter),
            # functools.partial(marginal_prob_std, sigma=sigma)
            wds.map(functools.partial(preprocess_temporal_contrast, sample_num=sample_clip_num, shift_lb=shift_lb, fps=8, truncate_frame=32, video_len=80)),
            # wds.map(preprocess_temporal_contrast),
            wds.batched(args.batch_size, partial=not is_train),
        ])
    else:
        pipeline.extend([
            wds.map(functools.partial(preprocess, fps=8, video_len=80, truncate_frame=32)),
            wds.batched(args.batch_size, partial=not is_train),
        ])


    dataset = wds.DataPipeline(*pipeline)
    if is_train:
        if not resampled:
            assert num_shards >= args.workers * args.world_size, 'number of shards must be >= total workers'
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=True,
    )

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples
    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)




def sample_temporal_index(sample_num, truncate_frame=16, video_len=40, shift_lb=8):
    """
    shift_lb: Temporal Shift Lower Bound >= 8
    """
    start_index_list = []
    end_index_list = []
    if sample_num == 2:
        start_frame1 = random.randint(0, video_len - truncate_frame - shift_lb)                 # 0 ~ 16
        start_frame2 = random.randint(start_frame1 + shift_lb, video_len - truncate_frame)      # start_frame + 8 ~ 24
        start_index_list.extend([start_frame1, start_frame2])
        end_index_list.extend([start_frame1 + truncate_frame, start_frame2 + truncate_frame])
    elif sample_num == 3:
        start_frame1 = random.randint(0, video_len - truncate_frame - 2 * shift_lb)                         # 0 ~ 8
        start_frame2 = random.randint(start_frame1 + shift_lb, video_len - truncate_frame - shift_lb)       #  start_frame1 + 8 ~ 16
        start_frame3 = random.randint(start_frame2 + shift_lb, video_len - truncate_frame)                  #  start_frame2 + 8 ~ 24
        start_index_list.extend([start_frame1, start_frame2, start_frame3])
        end_index_list.extend([start_frame1 + truncate_frame, start_frame2 + truncate_frame, start_frame3 + truncate_frame])
    elif sample_num == 4:
        start_frame1 = random.randint(0, video_len - truncate_frame - 3 * shift_lb)                         # 0 ~ 0
        start_frame2 = random.randint(start_frame1 + shift_lb, video_len - truncate_frame - 2 * shift_lb)   # start_frame1 + 8 ~ 8
        start_frame3 = random.randint(start_frame2 + shift_lb, video_len - truncate_frame - 1 * shift_lb)   # start_frame2 + 8 ~ 16
        start_frame4 = random.randint(start_frame3 + shift_lb, video_len - truncate_frame)                  # start_frame3 + 8 ~ 24
        start_index_list.extend([start_frame1, start_frame2, start_frame3, start_frame4])
        end_index_list.extend([start_frame1 + truncate_frame, start_frame2 + truncate_frame, start_frame3 + truncate_frame, start_frame4 + truncate_frame])
    assert len(start_index_list) != 0
    return start_index_list, end_index_list



def cut_video_and_spec_temporal_contrast(video, spec, sample_num=4, shift_lb=8, fps=4, truncate_frame=16, video_len=40):
    """
    For Temporal Contrast Intra Contrastive:
    Sample Different segments from the same data
    Output:
        video: sample_num x T x C x H x W
        spec:  sample_num x Mel_num x T'
    Tempora Shift >= 8 Frames (2s)
    """

    spec_raw = spec
    video_npy = video
    assert sample_num == 2 or sample_num == 3 or sample_num == 4 , "sample num must be [2,3,4]"
    start_frame_index_list, end_frame_index_list = sample_temporal_index(sample_num=sample_num, shift_lb=shift_lb, truncate_frame=truncate_frame, video_len=video_len)
    start_spec_list = []

    truncate_sec = 4
    # fps = fps
    sr = 16000
    hop_size = 250
    shape_h = 224
    truncate_frame = truncate_sec * fps

    # Get Spec Index list:
    spec_truncate = int(truncate_sec * sr / hop_size)
    for i in range(len(start_frame_index_list)):
        start_frame = start_frame_index_list[i]
        spec_start = int(start_frame / fps * sr / hop_size)
        start_spec_list.append(spec_start)
    
    # Stream:
    stream = io.BytesIO(spec_raw)
    spec_raw = numpy.lib.format.read_array(stream)

    stream = io.BytesIO(video_npy)
    # video_npy = numpy.lib.format.read_array(stream)
    video_npy = Image.open(stream)
    video_npy = np.array(video_npy)
    # video transpose:
    video_npy = video_npy.reshape(shape_h, -1, shape_h, 3).transpose(1,3,0,2)    # T x 3 x H x W

    # Check Spec_Raw and Crop Spec:
    sample_spec_list = []
    for i in range(sample_num):
        spec_start = start_spec_list[i]
        if spec_raw.shape[-1] < spec_start + spec_truncate:
            repeat_num = int((spec_start + spec_truncate) // spec_raw.shape[-1]) + 1
            spec_raw = np.tile(spec_raw, repeat_num)               # repeat 2 
            sample_spec = spec_raw[:, spec_start : spec_start + spec_truncate]
        else:
            sample_spec = spec_raw[:, spec_start : spec_start + spec_truncate]
        sample_spec_list.append(sample_spec[None])
    
    # Check Video npy: 
    sample_video_list = []
    for i in range(sample_num):
        start_frame, end_frame = start_frame_index_list[i], end_frame_index_list[i]
        if video_npy.shape[0] < end_frame:
            repeat_num = int((start_frame + truncate_frame) // video_npy.shape[0]) + 1
            sample_video = np.tile(video_npy, (repeat_num, 1))
            sample_video = video_npy[start_frame: end_frame]
        else:
            sample_video = video_npy[start_frame: end_frame]
        # Video Tensor Transforms:
        sample_video = transform_video(torch.from_numpy(sample_video))
        sample_video_list.append(sample_video.unsqueeze(0))
    
    sample_spec_list = torch.from_numpy(np.concatenate(sample_spec_list, 0))    # sample_num x H x W
    sample_video_list = torch.cat(sample_video_list, 0)                         # sample_num x T x C x H x W
    return sample_video_list, sample_spec_list, start_frame, end_frame


def preprocess_temporal_contrast(sample, sample_num=4, shift_lb=8, fps=4, truncate_frame=16, video_len=40):
    """
    Sampling different audio segments from the same sample for temporal contrast:
    Output:
        Video: Sample_num x T x C x H x W
        Spec:  Sample_num x T x C x H x W
    """
    spec, video = sample["spec.npy"], sample["video.jpg"]
    video, spec, start_frame, end_frame = cut_video_and_spec_temporal_contrast(video, spec, sample_num=sample_num, shift_lb=shift_lb, fps=fps, truncate_frame=truncate_frame, video_len=video_len)
    video_time = str(start_frame) + "_" + str(end_frame)
    return spec, video, video_time