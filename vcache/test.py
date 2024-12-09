import os
import json
import random
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from tqdm import tqdm
from utils import select_transform


class TestVideoDataset(Dataset):

    def __init__(self, dataset, frames, video_size, use_video_encoder, use_text_encoder, prompt_type, use_template):
        # basic
        self.dataset = dataset
        self.prompt_type = prompt_type
        self.data_path = f'/home/cjx/data/{dataset}/{dataset}_256x256'
        self.frames = frames
        self.video_tf = select_transform('aug0', video_size)
        self.use_template = use_template

        # backbone
        self.use_video_encoder = use_video_encoder
        self.use_text_encoder = use_text_encoder
        
        # prompt
        self.prompt_map = None
        if self.use_text_encoder:
            self.get_prompt()

        # data
        self.videos_list = []  
        self.videos_frames_list = []  
        self.videos_labels_list = [] 
        self.videos_labels = [] 
        self.get_sup_data()

    def get_prompt(self):
        prompt_path = f'/home/cjx/ufsar/prompt/{self.prompt_type}/{self.dataset}/test.json'
        if not os.path.exists(prompt_path):
            raise ValueError('prompt do not exist')
        else:
            with open(prompt_path, 'r') as f:
                self.prompt_map = json.load(f)

    def get_sup_data(self):
        split_path = os.path.join(self.data_path, 'test')
        class_list = os.listdir(split_path)  # class list
        class_list = [f for f in class_list if '.' not in f]  # .DS_Store error
        class_list.sort()
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
            idxs = self.distant_sampling_idx(len(frames), self.frames)
            image_list = [Image.open(frames[int(i)]).convert('RGB') for i in idxs]
            video = self.video_tf(image_list)
        else:
            video = torch.tensor([1.0])

        # text
        if self.use_text_encoder:
            if self.use_template:
                text = self.videos_labels[index]
            else:
                text = self.prompt_map[self.videos_list[index]]
        else:
            text = 'good luck'

        # label
        label = self.videos_labels_list[index]
        
        return video, text, label


def extract_feature_for_test_stage(args, model, logger, verbose=True):
    os.makedirs('/home/cjx/ufsar/cache/crash', exist_ok=True)
    model.eval()

    if verbose:
        logger.log('extract feature for test stage', prefix='Feature')
    test_dataset = TestVideoDataset(
        dataset=args.dataset,
        frames=args.frames, 
        video_size=args.video_size, 
        use_video_encoder=args.use_video_encoder, 
        use_text_encoder=args.use_text_encoder,
        prompt_type=args.prompt_type,
        use_template = args.use_template
    )
    data_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
        shuffle=False  # 顺序不能乱
    )
    
    video_features = []
    text_features = []
    with torch.no_grad():
        pbar = tqdm(total = len(data_loader), leave=False)
        for video, text,  _ in data_loader:
            video_embs, text_embs = model.get_feature(video, text, args.use_template)
            if args.use_video_encoder:
                video_features.append(video_embs)
            if args.use_text_encoder:
                text_features.append(text_embs)
            pbar.update()
        pbar.close()

    if args.use_video_encoder:
        video_features = torch.cat(video_features)
        if verbose:
            logger.log(f'video_features: {video_features.shape}', prefix='Feature')
        cache_path = f'/home/cjx/ufsar/cache/crash/video_{args.video_encoder}.npy'
        video_features = video_features.cpu().numpy()
        np.save(cache_path, video_features)
    if args.use_text_encoder:
        if args.use_template:
            text_features = torch.stack(text_features)
        else:
            text_features = torch.cat(text_features)
        if verbose:
            logger.log(f'text_features: {text_features.shape}', prefix='Feature')
        text_features = text_features.cpu().numpy()
        cache_path = f'/home/cjx/ufsar/cache/crash/text_{args.text_encoder}.npy'
        np.save(cache_path, text_features)
