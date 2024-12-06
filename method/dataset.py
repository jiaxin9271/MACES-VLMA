import os
import json
import numpy as np
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from utils import select_transform
from vcache import extract_feature, run_cluster


class VideoDataset(Dataset):
    
    def __init__(self, args, split='train', train_mode='Sup'):
        # basic
        self.dataset = args.dataset
        self.split = split
        self.shot = args.shot
        self.query = args.query
        self.data_path = f'/home/cjx/data/{self.dataset}/{self.dataset}_256x256'
        self.train_mode = train_mode
        self.cluster_type = args.cluster_type
        self.frames = args.frames
        self.video_size = args.video_size
        self.aug = args.train_aug if self.split == 'train' else 'aug0'
        self.video_tf = select_transform(self.aug, args.video_size)
        self.prompt_type = args.prompt_type
 
        # backbone
        self.use_video_encoder = args.use_video_encoder
        self.use_text_encoder = args.use_text_encoder
        self.video_encoder = args.video_encoder
        self.text_encoder = args.text_encoder
        self.use_video_cache = args.use_video_cache
        self.use_text_cache = args.use_text_cache
        self.use_video_zero = args.use_video_zero   
        self.use_text_zero = args.use_text_zero   

        # cache
        self.video_zero_cache = None
        self.text_zero_cache = None
        self.video_cache = None
        self.text_cache = None
        self.video_cache_file = f'/home/cjx/ufsar/cache/{self.dataset}/feature/{self.video_encoder}_{self.split}_{self.aug}_{self.frames}.npy'
        self.text_cache_file = f'/home/cjx/ufsar/cache/{self.dataset}/feature/{self.text_encoder}_{self.split}_{self.prompt_type}.npy'
        self.crash_video_cache_file = f'/home/cjx/ufsar/cache/crash/video_{self.video_encoder}.npy'
        self.crash_text_cache_file = f'/home/cjx/ufsar/cache/crash/text_{self.text_encoder}.npy'
        self.get_cache()
        self.get_zero_cache()

        # prompt
        self.prompt_map = None
        if self.use_text_encoder:
            self.get_prompt()

        # data
        self.videos_list = []  # 每个视频的地址
        self.videos_frames_list = []  # 每个视频帧的地址 
        self.videos_labels_list = []  # 每个视频的标签
        self.videos_labels = []  # 真实标签
        self.get_data()
    
    def get_zero_cache(self):
        if self.use_video_zero:
            if not os.path.exists(self.video_cache_file):
                extract_feature(
                    dataset=self.dataset, 
                    mode='video', 
                    split=self.split, 
                    aug=self.aug,
                    video_encoder=self.video_encoder, 
                    frames=self.frames,
                    video_size=self.video_size
                )
            features = np.load(self.video_cache_file)
            self.video_zero_cache = torch.from_numpy(features)
        
        if self.use_text_zero:
            if not os.path.exists(self.text_cache_file):
                extract_feature(
                    dataset=self.dataset, 
                    mode='text', 
                    aug=self.aug,
                    split=self.split, 
                    text_encoder=self.text_encoder,
                    prompt_type=self.prompt_type
                )
            features = np.load(self.text_cache_file)
            self.text_zero_cache = torch.from_numpy(features)

    def get_cache(self):
        if self.use_video_encoder and self.use_video_cache:
            self.get_video_cache()
        if self.use_text_encoder and self.use_text_cache:
            self.get_text_cahce()

    def get_video_cache(self):
        if not os.path.exists(self.video_cache_file):
            extract_feature(
                dataset=self.dataset, 
                mode='video', 
                split=self.split, 
                aug=self.aug,
                video_encoder=self.video_encoder, 
                frames=self.frames,
                video_size=self.video_size
            )
        features = np.load(self.video_cache_file)
        self.video_cache = torch.from_numpy(features)

    def get_text_cahce(self):
        if not os.path.exists(self.text_cache_file):
            extract_feature(
                dataset=self.dataset, 
                mode='text', 
                aug=self.aug,
                split=self.split, 
                text_encoder=self.text_encoder,
                prompt_type=self.prompt_type
            )
        features = np.load(self.text_cache_file)
        self.text_cache = torch.from_numpy(features)

    def get_test_cache(self):
        if self.use_video_encoder:
            features = np.load(self.crash_video_cache_file)
            self.video_cache = torch.from_numpy(features)   
        if self.use_text_encoder:
            features = np.load(self.crash_text_cache_file)
            self.text_cache = torch.from_numpy(features) 

    def get_prompt(self):
        prompt_path = f'/home/cjx/ufsar/prompt/{self.prompt_type}/{self.dataset}/{self.split}.json'
        if not os.path.exists(prompt_path):
            raise ValueError('prompt do not exist')
        else:
            with open(prompt_path, 'r') as f:
                self.prompt_map = json.load(f)

    def get_data(self):
        if self.train_mode in ['Sup', 'Aug'] or self.split in ['test', 'val']:
            self.get_sup_data()
        elif self.train_mode == 'Cluster' and self.split == 'train':
            self.get_cluster_data()
        else:
            raise ValueError('train_mode error')

    def get_sup_data(self):
        split_path = os.path.join(self.data_path, self.split)
        class_list = os.listdir(split_path)  # class list
        class_list = [f for f in class_list if '.' not in f]  # .DS_Store error
        class_list.sort()
        # print(class_list)
        for class_name in class_list:
            video_folders = os.listdir(os.path.join(split_path, class_name))  # vide folders
            video_folders = [f for f in video_folders if '.' not in f]  # .DS_Store error
            video_folders.sort()
            video_folders = [os.path.join(os.path.join(split_path, class_name, v)) for v in video_folders]
            video_folders.sort()
            for frames_folder in video_folders:
                self.videos_list.append(frames_folder)
                frames_list = os.listdir(frames_folder)  # frames list of a video
                frames_list = [i for i in frames_list if (('.jpg' in i) or ('.png' in i))]  # is a picture
                if len(frames_list) < self.frames:
                    continue
                frames_list.sort()
                frames_list = [os.path.join(frames_folder, frame) for frame in frames_list]
                frames_list.sort()
                self.videos_frames_list.append(frames_list)
                class_id = class_list.index(class_name)  # class number
                self.videos_labels_list.append(class_id)
                self.videos_labels.append(class_name)
    
    def get_cluster_data(self):
        # k.json 文件
        k_path = f'/home/cjx/ufsar/cache/{self.dataset}/cluster/k.json'
        if not os.path.exists(k_path):
            run_cluster(
                dataset=self.dataset, 
                split=self.split,
                aug='aug0',
                max_iters=100, 
                dist_method=self.cluster_type, 
                init_method='kmeans++', 
                video_encoder=self.video_encoder, 
                text_encoder=self.text_encoder,
                prompt_type=self.prompt_type,
                frames=self.frames
            )
        # 是否聚类
        with open(k_path, 'r') as f:
            k_map = json.load(f)
        if self.cluster_type == 'clip':
            cluster_key = f'{self.video_encoder}_{self.text_encoder}_train_aug0_{self.frames}_{self.prompt_type}'
        elif self.cluster_type == 'video':
            cluster_key = f'{self.video_encoder}_train_aug0_{self.frames}'
        elif self.cluster_type == 'text':
            cluster_key = f'{self.text_encoder}_train_{self.prompt_type}'
        if cluster_key not in k_map:
            run_cluster(
                dataset=self.dataset, 
                split=self.split,
                aug='aug0',
                max_iters=100, 
                dist_method=self.cluster_type, 
                init_method='kmeans++', 
                video_encoder=self.video_encoder, 
                text_encoder=self.text_encoder,
                prompt_type=self.prompt_type,
                frames=self.frames
            )
            with open(k_path, 'r') as f:
                k_map = json.load(f)
        # 找到k值
        cluster_k = k_map[cluster_key]
        videos_labels_list_path = f'/home/cjx/ufsar/cache/{self.dataset}/cluster/{cluster_key}_{cluster_k}.npy'
        print(f'cluster data: {videos_labels_list_path}')
        self.videos_labels_list = np.load(videos_labels_list_path)
        videos_list_path = f'/home/cjx/ufsar/cache/{self.dataset}/feature/videos_list_{self.split}.json'
        videos_frames_list_path = f'/home/cjx/ufsar/cache/{self.dataset}/feature/videos_frames_list_{self.split}.json'
        with open(videos_list_path, 'r') as f1, open(videos_frames_list_path, 'r') as f2:
            tmp1 = json.load(f1)
            tmp2 = json.load(f2)
            for i in range(len(self.videos_labels_list)):
                if self.videos_labels_list[i] != -1:
                    self.videos_list.append(tmp1[i])
                    self.videos_frames_list.append(tmp2[i])
        mask = self.videos_labels_list != -1 # 排除离散点
        self.videos_labels_list = self.videos_labels_list[mask]
        self.videos_labels_list = self.videos_labels_list.tolist()
        if self.use_video_cache and self.use_video_encoder:
            self.video_cache = self.video_cache[mask]
        if self.use_text_cache and self.use_text_encoder:
            self.text_cache = self.text_cache[mask]

    def distant_sampling_idx(self, total_frames, frames):
        """Distant Sampling
        """
        gap = total_frames // frames
        res = [random.randint(i * gap, i * gap + gap - 1) for i in range(frames)]
        return res

    def __len__(self):
        return len(self.videos_labels_list)

    def __getitem__(self, index):
        frames = self.videos_frames_list[index]

        # video
        if self.use_video_encoder:  
            if self.train_mode == 'Aug':
                video_list = []
                for _ in range(self.shot + self.query):
                    idxs = self.distant_sampling_idx(len(frames), self.frames)
                    image_list = [Image.open(frames[int(i)]).convert('RGB') for i in idxs]
                    video_list.append(self.video_tf(image_list))
                video = video_list
            else:
                if self.use_video_cache:
                    video = self.video_cache[index] 
                else:
                    idxs = self.distant_sampling_idx(len(frames), self.frames)
                    image_list = [Image.open(frames[int(i)]).convert('RGB') for i in idxs]
                    video = self.video_tf(image_list)
        else:
            video = torch.tensor([1.0])

        # text
        if self.use_text_encoder:
            if self.use_text_cache:
                text = self.text_cache[index]
            else:
                text = self.prompt_map[self.videos_list[index]]
        else:
            text = 'good luck'

        # label
        label = self.videos_labels_list[index]
        # label = self.videos_labels[index]

        if self.use_video_zero:
            zero_video = self.video_zero_cache[index]
        else:
            zero_video = torch.tensor([1.0])
        
        if self.use_text_zero:
            zero_text = self.text_zero_cache[index]
        else:
            zero_text = torch.tensor([1.0])
        
        return video, text, label, zero_video, zero_text
