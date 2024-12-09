import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from method import *
from vcache import *
from utils import *


if __name__ == '__main__':
    # 1. data process
    # extract_frames(dataset='hmdb51')

    # 2. video caption generation
    # generate_prompt_git(dataset='ssv2_small', split='train', frames=6, batch_size=8)

    # 3. multimodal adaptive clustering
    # run_cluster(
    #     dataset='hmdb51', 
    #     split='train',
    #     aug='aug0',
    #     max_iters=100, 
    #     dist_method='clip', 
    #     init_method='kmeans++', 
    #     video_encoder='ClipRN50', 
    #     text_encoder='TextRN50',
    #     prompt_type='git',
    #     frames=8
    # )

    # 4. meta-training
    run('config/hmdb51/ClipRN50_TextRN50.yaml', train=True)
    # run('config/hmdb51/ClipViT16B_TextViT16B.yaml', train=True)

    # 5. meta-testing
    # run('config/hmdb51/ClipRN50_TextRN50.yaml', train=False)
    # run('config/hmdb51/ClipViT16B_TextViT16B.yaml', train=False)
