import os
import json
import random
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from utils import select_transform
from tqdm import tqdm


class V2TDataset(Dataset):

    def __init__(self, data_path, split='train', aug='aug0', frames=8, video_size=224, prompt_type='git'):
        self.split = split
        self.frames = frames
        self.prompt_type = prompt_type
        self.split_path = os.path.join(data_path, self.split)
        self.video_tf = select_transform(aug, video_size)
        self.videos_list = []  # 每个视频的地址
        self.videos_frames_list = []  # 每个视频帧的地址 
        self.videos_labels_list = []  # 每个视频的标签
        self.get_sup_data()

    def get_sup_data(self):
        class_list = os.listdir(self.split_path)  # class list
        class_list = [f for f in class_list if '.' not in f]  # .DS_Store error
        class_list.sort()
        for class_name in class_list:
            video_folders = os.listdir(os.path.join(self.split_path, class_name))  # vide folders
            video_folders = [f for f in video_folders if '.' not in f]  # .DS_Store error
            video_folders.sort()
            video_folders = [os.path.join(os.path.join(self.split_path, class_name, v)) for v in video_folders]
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
    
    def distant_sampling_idx(self, total_frames, frames):
        gap = total_frames // frames
        res = [random.randint(i * gap, i * gap + gap - 1) for i in range(frames)]
        return res

    def __len__(self):
        return len(self.videos_labels_list)

    def __getitem__(self, index):
        frames = self.videos_frames_list[index]

        if self.prompt_type == 'git':
            idxs = self.distant_sampling_idx(len(frames), self.frames)
            image_list = [Image.open(frames[int(i)]).convert('RGB') for i in idxs]
            data = self.video_tf(image_list)
            label = self.videos_list[index] 
            return data, label
        elif self.prompt_type == 'blip':
            data = frames[int(len(frames) // 2)]
            label = self.videos_list[index] 
            return data, label


def generate_prompt_git(dataset='hmdb51', split='train', frames=6, batch_size=1):
    os.makedirs(f'/home/cjx/ufsar/prompt', exist_ok=True)
    video_dataset = V2TDataset(
        data_path=f'/home/cjx/data/{dataset}/{dataset}_256x256', 
        split=split, 
        aug='aug0', 
        frames=frames
    )
    video_loader = DataLoader(
        dataset=video_dataset,
        batch_size=batch_size,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
        shuffle=False  # 顺序不能乱
    )
    
    from transformers import AutoProcessor, AutoModelForCausalLM
    processor = AutoProcessor.from_pretrained('microsoft/git-large-vatex', local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained('microsoft/git-large-vatex', local_files_only=True)
    model.eval()
    model.cuda()
    
    try:
        with open(f'/home/cjx/ufsar/prompt/git/{dataset}/{split}.json', 'r') as f:
            prompt_dict = json.load(f)
    except FileNotFoundError:
        prompt_dict = {}
    
    pbar = tqdm(total=len(video_loader), unit='bs')
    for data, label in video_loader:
        generated_ids = model.generate(pixel_values=data.cuda(), max_length=50)
        res = processor.batch_decode(generated_ids, skip_special_tokens=True)
        for i in range(len(res)):
            prompt_dict[label[i]] = res[i]
        with open(f'/home/cjx/ufsar/prompt/git/{dataset}/{split}.json', 'w') as f:
            json.dump(prompt_dict, f)
        pbar.update()
    pbar.close()


def generate_prompt_blip(dataset='hmdb51', split='train'):
    os.makedirs(f'/home/cjx/ufsar/prompt', exist_ok=True)
    video_dataset = V2TDataset(
        data_path=f'/home/cjx/data/{dataset}/{dataset}_256x256', 
        split=split, 
        aug='aug0', 
        frames=1,
        prompt_type='blip'
    )
    video_loader = DataLoader(
        dataset=video_dataset,
        batch_size=1,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
        shuffle=False  # 顺序不能乱
    )
    
    from transformers import BlipProcessor, BlipForConditionalGeneration
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large", local_files_only=True)
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", local_files_only=True)
    model.eval()
    model.cuda()
    
    try:
        with open(f'/home/cjx/ufsar/prompt/blip/{dataset}/{split}.json', 'r') as f:
            prompt_dict = json.load(f)
    except FileNotFoundError:
        prompt_dict = {}
    
    pbar = tqdm(total=len(video_loader), unit='bs')
    for data, label in video_loader:
        # inputs = processor(raw_image, "a photography of", return_tensors="pt").to("cuda")
        inputs = processor(Image.open(data[0]).convert('RGB'), return_tensors="pt").to("cuda")
        generated_ids = model.generate(**inputs, max_new_tokens=30)
        res = processor.decode(generated_ids[0], skip_special_tokens=True)
        prompt_dict[label[0]] = res
        with open(f'/home/cjx/ufsar/prompt/blip/{dataset}/{split}.json', 'w') as f:
            json.dump(prompt_dict, f)
        pbar.update()
    pbar.close()


def generate_prompt_gpt2(dataset='hmdb51', split='train'):
    os.makedirs(f'/home/cjx/ufsar/prompt', exist_ok=True)
    video_dataset = V2TDataset(
        data_path=f'/home/cjx/data/{dataset}/{dataset}_256x256', 
        split=split, 
        aug='aug0', 
        frames=1,
        prompt_type='blip'
    )
    video_loader = DataLoader(
        dataset=video_dataset,
        batch_size=8,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
        shuffle=False  # 顺序不能乱
    )
    
    from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning", local_files_only=True)
    feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning", local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning", local_files_only=True)
    model.cuda()
    model.eval()
    
    try:
        with open(f'/home/cjx/ufsar/prompt/gpt2/{dataset}/{split}.json', 'r') as f:
            prompt_dict = json.load(f)
    except FileNotFoundError:
        prompt_dict = {}
    
    pbar = tqdm(total=len(video_loader), unit='bs')
    for data, label in video_loader:
        images = []
        for i in data:
            image = Image.open(data[0]).convert('RGB')
            images.append(image)
        pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
        output_ids = model.generate(pixel_values=pixel_values.cuda(), max_length=16, num_beams=4)
        preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]
        for i in range(len(preds)):
            prompt_dict[label[i]] = preds[i]
        with open(f'/home/cjx/ufsar/prompt/gpt2/{dataset}/{split}.json', 'w') as f:
            json.dump(prompt_dict, f)
        pbar.update()
    pbar.close()


def generate_prompt_blip2(dataset='hmdb51', split='train'):
    os.makedirs(f'/home/cjx/ufsar/prompt', exist_ok=True)
    video_dataset = V2TDataset(
        data_path=f'/home/cjx/data/{dataset}/{dataset}_256x256', 
        split=split, 
        aug='aug0', 
        frames=1,
        prompt_type='blip'
    )
    video_loader = DataLoader(
        dataset=video_dataset,
        batch_size=1,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
        shuffle=False  # 顺序不能乱
    )
    
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b", local_files_only=True)
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", local_files_only=True)
    model.eval()
    model.cuda()
    
    try:
        with open(f'/home/cjx/ufsar/prompt/blip2/{dataset}/{split}.json', 'r') as f:
            prompt_dict = json.load(f)
    except FileNotFoundError:
        prompt_dict = {}
    
    pbar = tqdm(total=len(video_loader), unit='bs')
    for data, label in video_loader:
        # inputs = processor(raw_image, "a photography of", return_tensors="pt").to("cuda")
        inputs = processor(Image.open(data[0]).convert('RGB'), return_tensors="pt").to("cuda")
        generated_ids = model.generate(**inputs, max_new_tokens=30)
        res = processor.decode(generated_ids[0], skip_special_tokens=True)
        prompt_dict[label[0]] = res
        with open(f'/home/cjx/ufsar/prompt/blip2/{dataset}/{split}.json', 'w') as f:
            json.dump(prompt_dict, f)
        pbar.update()
    pbar.close()
    