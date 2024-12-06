import os
import subprocess
from glob import glob
import torch
import numpy as np
from torch.utils.data import Sampler
import shutil
from tqdm import tqdm


class CategoriesSampler(Sampler):
    def __init__(self, label, n_batch, n_cls, n_per, data_source=None):
        super().__init__(data_source)
        self.n_batch = n_batch  # epoch
        self.n_cls = n_cls  # batch
        self.n_per = n_per  # shot + query
        label = np.array(label)  # label
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for _ in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]
            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch


def modify_ssv2_full():
    root_path = '/home/cjx/data/ssv2_full/20bn-something-something-v2'
    folder_path = '/home/cjx/data/ssv2_full/videos'
    os.mkdir(folder_path)

    wc = os.path.join('/home/cjx/ufsar/code/utils/split/ssv2_full', '*.txt')
    num = 0
    for fn in glob(wc):
        classes = []  # 类别
        vids = []  # 视频的id

        with open(fn, 'r') as f:
            data = f.readlines()
            c = [x.split(os.sep)[-2].strip() for x in data]
            v = [x.split(os.sep)[-1].strip() for x in data]
            vids.extend(v)
            classes.extend(c)

        unique_classes = list(set(classes))
        for cla in unique_classes:
            os.mkdir(os.path.join(folder_path, cla))
            for id in range(len(vids)):
                if classes[id] == cla:
                    num += 1
                    shutil.copy(
                        os.path.join(root_path, f'{vids[id]}.webm'), 
                        os.path.join(folder_path, cla, f'{vids[id]}.webm')
                    )
        print(num)


def modify_ssv2_small():
    root_path = '/home/cjx/data/ssv2_small/20bn-something-something-v2'
    folder_path = '/home/cjx/data/ssv2_small/videos'
    os.makedirs(folder_path, exist_ok=True)

    wc = os.path.join('/home/cjx/ufsar/code/utils/split/ssv2_small', '*.txt')
    num = 0
    for fn in glob(wc):
        with open(fn, 'r') as f:
            data = f.readlines()
            classes = [x.split(os.sep)[-2].strip() for x in data]
            vids = [x.split(os.sep)[-1].strip() for x in data]

        unique_classes = list(set(classes))
        num_videos = len(vids)
        for cla in unique_classes:
            os.makedirs(os.path.join(folder_path, cla))
            for id in range(num_videos):
                if classes[id] == cla:
                    num += 1
                    shutil.copy(
                        os.path.join(root_path, f'{vids[id]}.webm'), 
                        os.path.join(folder_path, cla, f'{vids[id]}.webm')
                    )
    print(num)


def modify_k100():
    root_path = '/home/cjx/data/k100'
    folder_path = '/home/cjx/data/k100/videos'
    os.makedirs(folder_path, exist_ok=True)

    wc = os.path.join('/home/cjx/ufsar/code/utils/split/k100', '*.txt')
    num = 0
    for fn in glob(wc):
        classes = []  # 类别
        vids = []  # 视频的id

        if 'train' in fn:
            cur_split = 'train'
        elif 'val' in fn:
            cur_split = 'val'
        elif 'test' in fn:
            cur_split = 'test'
        else:
            raise ValueError('split error')

        with open(fn, 'r') as f:
            data = f.readlines()
            c = [x.split(os.sep)[-2].strip() for x in data]
            v = [x.split(os.sep)[-1].strip() for x in data]
            vids.extend(v)
            classes.extend(c)

        unique_classes = list(set(classes))
        for cla in unique_classes:
            os.makedirs(os.path.join(folder_path, cla), exist_ok=True)
            for id in range(len(vids)):
                if classes[id] == cla:
                    num += 1
                    shutil.copy(
                        os.path.join(root_path, cur_split, f'{vids[id]}.mp4'), 
                        os.path.join(folder_path, cla, f'{vids[id]}.mp4')
                    )
        print(num)


def extract_frames(dataset='hmdb51'):
    in_folder = f'/home/cjx/data/{dataset}/videos/'
    out_folder = f'/home/cjx/data/{dataset}/{dataset}_256x256'
    split_dir = f'/home/cjx/ufsar/code/utils/split/{dataset}'

    suffix = 'avi'
    if 'k100' in dataset:
        suffix = 'mp4'
    elif 'ssv2' in dataset:
        suffix = 'webm'

    os.makedirs(out_folder, exist_ok=True)
    wc = os.path.join(split_dir, '*.txt')
    for fn in glob(wc):
        classes = []
        vids = []

        if 'train' in fn:
            cur_split = 'train'
        elif 'val' in fn:
            cur_split = 'val'
        elif 'test' in fn:
            cur_split = 'test'
        else:
            raise ValueError('split error')

        with open(fn, 'r') as f:
            data = f.readlines()
            c = [x.split(os.sep)[-2].strip() for x in data]
            v = [x.split(os.sep)[-1].strip() for x in data]
            vids.extend(v)
            classes.extend(c)

        os.makedirs(os.path.join(out_folder, cur_split), exist_ok=True)
        for c in list(set(classes)):
            os.makedirs(os.path.join(out_folder, cur_split, c), exist_ok=True)

        pbar = tqdm(total=len(vids), unit='video')
        for v, c in zip(vids, classes):
            source_vid = os.path.join(in_folder, c, f'{v}.{suffix}')
            extract_dir = os.path.join(out_folder, cur_split, c, v)
            os.makedirs(extract_dir, exist_ok=True)
            out_wc = os.path.join(extract_dir, '%08d.jpg')
            cmd = [
                'ffmpeg', '-i', source_vid, 
                '-vf', 'fps=30,scale=256:256', 
                '-q:v', '5', 
                out_wc, 
                '-loglevel', 'quiet'
            ]
            subprocess.call(cmd)
            pbar.update()
        pbar.close()
