import os
import json
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from tqdm import tqdm
from utils import select_transform, init_seeds, Logger
from backbone import select_video_backbone, select_text_backbone
from sklearn.metrics.cluster import pair_confusion_matrix


class CacheVideoDataset(Dataset):

    def __init__(self, dataset='hmdb51', split='train', mode='video', aug='test', frames=8, video_size=224, prompt_type='git'):
        # basic
        self.dataset = dataset
        self.split = split
        self.mode = mode
        self.data_path = f'/home/cjx/data/{dataset}/{dataset}_256x256'
        self.frames = frames
        self.video_tf = select_transform(aug, video_size)
        
        # prompt
        self.prompt_type = prompt_type
        self.prompt_map = None
        if self.mode == 'text' or self.mode == 'clip':
            self.get_prompt()

        # data
        self.videos_list = []  # 每个视频的地址
        self.videos_frames_list = []  # 每个视频帧的地址 
        self.videos_labels_list = []  # 每个视频的标签
        self.get_sup_data()

    def get_prompt(self):
        prompt_path = f'/home/cjx/ufsar/prompt/{self.prompt_type}/{self.dataset}/{self.split}.json'
        if not os.path.exists(prompt_path):
            raise ValueError('prompt do not exist')
        else:
            with open(prompt_path, 'r') as f:
                self.prompt_map = json.load(f)

    def get_sup_data(self):
        split_path = os.path.join(self.data_path, self.split)
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
        if self.mode == 'video' or self.mode == 'clip':
            idxs = self.distant_sampling_idx(len(frames), self.frames)
            image_list = [Image.open(frames[int(i)]).convert('RGB') for i in idxs]
            video = self.video_tf(image_list)
        else:
            video = torch.tensor([1.0])

        # text
        if self.mode == 'text' or self.mode == 'clip':
            text = self.prompt_map[self.videos_list[index]]
        else:
            text = 'good luck'

        # label
        label = self.videos_labels_list[index]
        
        return video, text, label


def extract_feature(
        dataset='hmdb51', 
        mode='clip', 
        split='train', 
        aug='aug0',
        video_encoder='ClipRN50', 
        text_encoder='TextRN50',
        frames=8,
        video_size=224,
        prompt_type='git'
    ):
    output_dir = f'/home/cjx/ufsar/cache/{dataset}/feature' 
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir + '/logs', exist_ok=True)
    logger = Logger(output_dir, 'logs')

    # 1. 选择预训练模型
    if mode == 'video':     
        logger.log(f'extract {dataset}/{split} video feature by [{video_encoder}]', prefix='Feature')
        save_name = f'{video_encoder}_{split}_{aug}_{frames}'
        model, _ = select_video_backbone(video_encoder)
    elif mode == 'text':
        logger.log(f'extract {dataset}/{split} text feature by [{text_encoder}]', prefix='Feature')
        save_name = f'{text_encoder}_{split}_{prompt_type}'
        model,  _ = select_text_backbone(text_encoder)
    elif mode == 'clip':
        logger.log(f'extract {dataset}/{split} video+text feature by [{video_encoder}+{text_encoder}]', prefix='Feature')
        save_name = f'{video_encoder}_{text_encoder}_{split}_{aug}_{frames}_{prompt_type}'
        v_model, _ = select_video_backbone(video_encoder)
        t_model, _ = select_text_backbone(text_encoder)
    else:
        raise ValueError('mode error')
    
    if mode == 'clip':
        v_model.cuda()
        v_model.eval()
        t_model.cuda()
        t_model.eval()
    else:
        model.cuda()
        model.eval()

    # 2. 准备数据
    train_dataset = CacheVideoDataset(
        dataset=dataset, 
        split=split, 
        mode=mode, 
        aug=aug, 
        frames=frames, 
        video_size=video_size,
        prompt_type=prompt_type
    )
    data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=1,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
        shuffle=False
    )
    with open(output_dir + f'/videos_list_{split}.json', 'w') as f:
        json.dump(train_dataset.videos_list, f)
    with open(output_dir + f'/videos_frames_list_{split}.json', 'w') as f:
        json.dump(train_dataset.videos_frames_list, f)

    # 3. 特征提取
    features = []
    labels = []
    with torch.no_grad():
        pbar = tqdm(total=len(data_loader))
        for video, text, label in data_loader:
            if mode == 'video':
                x = video.cuda()
                x = x.reshape(-1, *x.shape[-3:])
                video_embs = model(x)
                video_embs = video_embs.reshape(1, frames, -1)
                features.append(video_embs)

                # videomae
                # x = video.cuda()
                # x = x.reshape(-1, *x.shape[-4:])  # [bs x (s + q) x 8, 3, 224, 224]  
                # video_embs = model(pixel_values=x).last_hidden_state
                # video_embs = video_embs.mean(dim=1)
                # features.append(video_embs)  
            elif mode == 'text':
                text_embs = model(text)
                features.append(text_embs)
            elif mode == 'clip':
                # video
                x = video.cuda()
                x = x.reshape(-1, *x.shape[-3:])  # [bs x (s + q) x 8, 3, 224, 224]        
                video_embs = v_model(x)
                video_embs = video_embs.reshape(1, frames, -1)
                # text
                text_embs = t_model(text)
                features.append(torch.cat([video_embs, text_embs.unsqueeze(1)], dim=1))
            else:
                raise ValueError('mode error')
            labels.append(label)
            pbar.update()
        pbar.close()

    # 4. 将特征和标签保存到本地文件
    features = torch.cat(features)
    features = features.cpu().numpy()
    np.save(output_dir + f'/{save_name}.npy', features)
    logger.log(f'features: {features.shape}', prefix='Feature')
    labels = torch.cat(labels)
    labels = labels.cpu().numpy()
    np.save(output_dir + f'/labels_{split}.npy', labels)


def get_distance(x, y, dist_method='video', eps=10e-3, frames=8):
    if dist_method == 'video':
        # n, m = x.size(0), y.size(0)
        # x = x.unsqueeze(1).expand(n, m, -1)
        # y = y.unsqueeze(0).expand(n, m, -1)
        # return torch.norm(x - y, dim=2)
        x = F.normalize(x, dim=-1)  # [n, 8, 2048]
        y = F.normalize(y, dim=-1)  # [k, 8, 2048]
        sim = torch.einsum('nid,kjd->nkij', x, y)
        sim = sim.max(dim=-1)[0].sum(dim=-1) + sim.max(dim=-2)[0].sum(dim=-1)  # [n, k]
        sim = sim / (2 * frames)
        return 1 - sim + eps
    elif dist_method == 'text':
        # n, m = x.size(0), y.size(0)
        # x = x.unsqueeze(1).expand(n, m, -1)
        # y = y.unsqueeze(0).expand(n, m, -1)
        # return torch.norm(x - y, dim=2)
        x = F.normalize(x, dim=-1)  # [n, 8, 2048]
        y = F.normalize(y, dim=-1)  # [k, 8, 2048]
        sim = torch.einsum('nd,kd->nk', x, y)
        return 1 - sim + eps   # [n, k]
    elif dist_method == 'clip':
        video_x = x[:, :-1] # [6400, 8, 1024]
        text_x = x[:, -1] # [6400, 1024] 
        video_y = y[:, :-1] # [64, 8, 1024]
        text_y = y[:, -1] # [64, 1024]
        video_x = F.normalize(video_x, dim=-1)  # [6400, 8, 1024]
        text_x = F.normalize(text_x, dim=-1)  # [6400, 1024] 
        video_y = F.normalize(video_y, dim=-1)  # [64, 8, 1024]
        text_y = F.normalize(text_y, dim=-1)  # [64, 1024]

        sim_v = torch.einsum('nid,kjd->nkij', video_x, video_y)
        sim_v = (sim_v.max(dim=-1)[0].sum(dim=-1) + sim_v.max(dim=-2)[0].sum(dim=-1)) / (2 * frames)  # [n, k]
        sim_t = torch.einsum('nd,kd->nk', text_x, text_y) # [n, k]
        sim_v2t = torch.einsum('nd,kfd->nkf', text_x, video_y).mean(dim=-1)
        sim_t2v = torch.einsum('nfd,kd->nkf', video_x, text_y).mean(dim=-1)
        sim_vt = (sim_v2t + sim_t2v) / 2

        return (1 - sim_t + eps) * (1 - sim_v + eps) * (1 - sim_vt + eps)
    else:
        raise ValueError('dist_method error')


def get_rand_index_and_f_measure(labels_true, labels_pred, beta=1.):
    labels_true = labels_true.cpu().numpy()
    labels_pred = labels_pred.cpu().numpy()
    (tn, fp), (fn, tp) = pair_confusion_matrix(labels_true, labels_pred)

    ri = (tp + tn) / (tp + tn + fp + fn)
    ari = 2. * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
    p, r = tp / (tp + fp), tp / (tp + fn)
    f_beta = (1 + beta**2) * (p * r / ((beta ** 2) * p + r))
    return ri, ari, f_beta


def calculate_purity(labels_true, labels_pred):
    labels_true = labels_true.cpu().numpy()
    labels_pred = labels_pred.cpu().numpy()

    clusters = np.unique(labels_pred)
    labels_true = np.reshape(labels_true, (-1, 1))
    labels_pred = np.reshape(labels_pred, (-1, 1))
    count = []
    for c in clusters:
        idx = np.where(labels_pred == c)[0]
        labels_tmp = labels_true[idx, :].reshape(-1)
        count.append(np.bincount(labels_tmp).max())
    return np.sum(count) / labels_true.shape[0]


def kmeans_init(X, k, init_method='kmeans++', dist_method='mhm', frames=8):
    n = X.shape[0]
    if init_method == 'random':
        return X[np.random.choice(n, k, replace=False)]
    elif init_method == 'kmeans++':
        # 1. 从数据集中随机选择一个数据点作为第一个聚类中心
        centers = [X[np.random.randint(n)]]
        for i in range(1, k):
            # 2. 对于每个数据点，计算它到最近的聚类中心的距离
            P = torch.cat(centers).reshape(i, *X.shape[1:])
            dist = get_distance(X, P, dist_method, frames=frames)  # [n, i]
            dist = dist.min(dim=-1)[0]
            dist = dist.cpu().numpy()
            # 3. 根据权重随机选择下一个聚类中心
            prob = dist / dist.sum()
            # prob = prob.cpu().numpy()
            centers.append(X[np.random.choice(n, p=prob)])
        return torch.cat(centers).reshape(k, *X.shape[1:])
    else:
        raise ValueError('init_method error')


def get_truth_info(features, labels, dist_method, frames=8):
    unique_tensor, _ = torch.unique(labels, sorted=True, return_inverse=True)
    truth_k = unique_tensor.shape[0]
    best_centroids = torch.zeros((truth_k, *features.shape[1:])).cuda(non_blocking=True)
    best_centroids = best_centroids.to(features.dtype)
    for i in range(truth_k):
        cluster_points = features[labels == i]
        if len(cluster_points) > 0:
            best_centroids[i] = cluster_points.mean(dim=0)
    truth_dist = get_distance(features, best_centroids, dist_method, frames=frames)
    truth_sse = truth_dist.min(dim=-1)[0].sum()
    return truth_k, truth_sse
    

def kmeans(k, features, labels, logger, init_method, dist_method, max_iters, verbose=True, frames=8):
    cluster_labels = torch.zeros(labels.shape, dtype=torch.long).cuda(non_blocking=True)
    
    # 1. 初始化聚类中心
    centroids = kmeans_init(features, k, init_method, dist_method, frames=frames)

    sse = 0.0
    for it in range(1, max_iters + 1):
        easy_stop = True

        # 2. 计算每个数据点到聚类中心的距离 
        dist = get_distance(features, centroids, dist_method, frames=frames) # [n, k]

        # 3. 分配每个数据点到最近的聚类中心
        new_cluster_assignments = torch.argmin(dist, dim=-1)

        # 聚类中心不发生改变可以退出聚类
        for i in range(features.shape[0]):
            if cluster_labels[i] != new_cluster_assignments[i]:
                easy_stop = False
                break
        if easy_stop:
            break
        
        # 4. 更新每个样本的聚类中心
        cluster_labels = new_cluster_assignments

        # 5. 更新聚类中心为每个簇的均值
        for i in range(k):
            cluster_points = features[cluster_labels == i]
            if len(cluster_points) > 0:
                centroids[i] = cluster_points.mean(dim=0)
        
        # 计算sse
        sse = dist.min(dim=-1)[0].sum()
        # if verbose:
            # logger.log(f'kmeans it[{it}], sse: {sse}', prefix='Cluster')
    
    # 计算纯度
    # purity = calculate_purity(labels, cluster_labels)
    # ri, ari, f_beta = get_rand_index_and_f_measure(labels, cluster_labels, beta=1.)
    # logger.log(f'k: {k}, purity: {purity}, ri: {ri}, ari: {ari}, f_measure: {f_beta}', prefix='Cluster')
    
    # sse = silhouette_score(X, labels)
    if not verbose:
        return sse
    
    # 聚类结果
    # logger.log(f'best_sse: {sse}', prefix='Cluster')
    for i in range(k):
        if len(cluster_labels[cluster_labels == i]) < 6:
            for j in range(features.shape[0]):
                if cluster_labels[j] == i:
                    cluster_labels[j] = -1
    
    # 消除离散点
    max_label = k - 1
    i = 0
    while True:
        if i == max_label:
            break
        
        # 标签i没有样本
        if len(cluster_labels[cluster_labels == i]) == 0:
            for j in range(features.shape[0]):
                if cluster_labels[j] == max_label:
                    cluster_labels[j] = i
            max_label = max_label - 1
        else:
            i = i + 1

    # for i in range(k):
    #     logger.log(f'{i}: {len(cluster_labels[cluster_labels == i])}', prefix='Cluster')
    
    # 计算纯度
    purity = calculate_purity(labels, cluster_labels)
    ri, ari, f_beta = get_rand_index_and_f_measure(labels, cluster_labels, beta=1.)
    # logger.log(f'k: {k}, purity: {purity}, ri: {ri}, ari: {ari}, f_measure: {f_beta}', prefix='Cluster')
    logger.log(f'k: {k}, purity: {purity}, ari: {ari}, f_measure: {f_beta}', prefix='Cluster')

    # 保存
    cluster_labels = cluster_labels.cpu().numpy()
    return cluster_labels
    

def log_means(features, labels, logger, init_method, dist_method, max_iters, low_k=None, high_k=None, frames=8):
    if low_k is None:
        low_k = 2
        # low_k = 5
        # low_k = 20
    if high_k is None:
        # high_k = features.shape[0] // 6  # 每一类至少有6个
        # high_k = 500
        high_k = min(1000, features.shape[0] // 6)
    
    low_sse = kmeans(low_k, features, labels, logger, init_method, dist_method, max_iters, False, frames=frames)
    logger.log(f'k: {low_k}, sse: {low_sse}', prefix='Cluster')
    high_sse = kmeans(high_k, features, labels, logger, init_method, dist_method, max_iters, False, frames=frames)
    logger.log(f'k: {high_k}, sse: {high_sse}', prefix='Cluster')
    log_means_map = {
        low_k: low_sse,
        high_k: high_sse
    }

    while high_k - low_k > 2:
        mid_k = (low_k + high_k) // 2
        mid_sse = kmeans(mid_k, features, labels, logger, init_method, dist_method, max_iters, False, frames=frames)
        log_means_map[mid_k] = mid_sse
        ratio_left = low_sse / mid_sse
        ratio_right = mid_sse / high_sse
        logger.log(f'k: {mid_k}, sse: {mid_sse}, left: {ratio_left}, right: {ratio_right}', prefix='Cluster')
        if ratio_left > ratio_right:
            high_k = mid_k
            # high_sse = mid_sse
        else:
            low_k = mid_k
            # low_sse = mid_sse
    
    mid_k = (low_k + high_k) // 2
    mid_sse = kmeans(mid_k, features, labels, logger, init_method, dist_method, max_iters, False, frames=frames)
    log_means_map[mid_k] = mid_sse
    low_sse = log_means_map[low_k]
    high_sse = log_means_map[high_k]
    ratio_left = low_sse / mid_sse
    ratio_right = mid_sse / high_sse
    logger.log(f'k: {mid_k}, sse: {mid_sse}, left: {ratio_left}, right: {ratio_right}', prefix='Cluster')
    if ratio_left > ratio_right:
        # logger.log(f'bend: {mid_k}', prefix='Cluster')
        return mid_k
    else:
        # logger.log(f'bend: {high_k}', prefix='Cluster')
        return high_k


def run_cluster(
        dataset='hmdb51', 
        split='train',
        aug='aug0',
        max_iters=100, 
        dist_method='clip', 
        init_method='kmeans++', 
        video_encoder='ClipRN50', 
        text_encoder='TextRN50',
        prompt_type='git',
        frames=8
    ):

    output_dir = f'/home/cjx/ufsar/cache/{dataset}/cluster' 
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir + '/logs', exist_ok=True)
    logger = Logger(output_dir, 'logs')

    # seeds
    init_seeds(42)
    
    # 1. 提取特征
    # logger.log(f'step1: get feature', prefix='Cluster')
    if dist_method == 'clip':
        features_cache_file = f'{video_encoder}_{text_encoder}_{split}_{aug}_{frames}_{prompt_type}'
    elif dist_method == 'video':
        features_cache_file = f'{video_encoder}_{split}_{aug}_{frames}'
    elif dist_method == 'text':
        features_cache_file = f'{text_encoder}_{split}_{prompt_type}'
    if not os.path.exists(f'/home/cjx/ufsar/cache/{dataset}/feature/{features_cache_file}.npy'):
        extract_feature(
            dataset=dataset,
            mode=dist_method,
            split=split, 
            aug=aug,
            video_encoder=video_encoder, 
            text_encoder=text_encoder,
            prompt_type=prompt_type,
            frames=frames
        )
    # 2. 读取文件，读取一次就够了
    # logger.log(f'step2: check infomation', prefix='Cluster')
    logger.log(f'dataset: {dataset}', prefix='Cluster')
    # logger.log(f'init: {init_method}', prefix='Cluster')
    logger.log(f'dist: {dist_method}', prefix='Cluster')
    features = np.load(f'/home/cjx/ufsar/cache/{dataset}/feature/{features_cache_file}.npy')  # [4280, 2048]
    features = torch.from_numpy(features).cuda()
    # features = features.mean(dim=-2)
    logger.log(f'feature: {features.shape}', prefix='Cluster')
    labels = np.load(f'/home/cjx/ufsar/cache/{dataset}/feature/labels_{split}.npy')  # [4280]
    labels = torch.from_numpy(labels).cuda()
    logger.log(f'label: {labels.shape}', prefix='Cluster')
    truth_k, truth_sse = get_truth_info(features, labels, dist_method, frames=frames)
    logger.log(f'truth_k: {truth_k}, truth_sse: {truth_sse}', prefix='Cluster')

    # [25, 50, 75, 100, 125, 150, 175, 200]
    # [357, 179, 90, 46, 68, 79, 84, 81, 80]
    # for res_k in [357, 179, 90, 46, 68, 79, 84, 81, 80]:
    #     cluster_labels = kmeans(res_k, features, labels, logger, init_method, dist_method, max_iters, False, frames=frames) 
        # cluster_labels = kmeans(res_k, features, labels, logger, init_method, dist_method, max_iters, True, frames=frames) 
        # np.save(output_dir + f'/{features_cache_file}_{res_k}.npy', cluster_labels)

    # 3. log_means
    # logger.log(f'step3: runing log_means', prefix='Cluster')
    try:
        with open(f'/home/cjx/ufsar/cache/{dataset}/cluster/k.json', 'r') as f:
            k_map = json.load(f)
    except FileNotFoundError:
        k_map = {}
    res_k = log_means(features, labels, logger, init_method, dist_method, max_iters, frames=frames)
    # res_k = 31
    k_map[f'{features_cache_file}'] = res_k
    with open(f'/home/cjx/ufsar/cache/{dataset}/cluster/k.json', 'w') as f:
        json.dump(k_map, f)
    
    # 4. kmeans
    # logger.log(f'step4: runing k_means', prefix='Cluster')
    cluster_labels = kmeans(res_k, features, labels, logger, init_method, dist_method, max_iters, True, frames=frames) 
    np.save(output_dir + f'/{features_cache_file}_{res_k}.npy', cluster_labels)
