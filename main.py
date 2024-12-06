import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from method import *
from vcache import *


if __name__ == '__main__':
    # generate_prompt_blip2(dataset='ssv2_small', split='train')
    # generate_prompt_git(dataset='ssv2_small', split='train', frames=6, batch_size=8)
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

    # run('config/hmdb51/CoOp.yaml', train=True)
    # run('config/hmdb51/MAXI.yaml', train=False)
    # run_5x5('config/hmdb51/ClipRN50_TextRN50.yaml')
    # run('config/hmdb51/ClipRN50.yaml', train=True)
    # run('config/hmdb51/ClipViT16B.yaml', train=False)
    # run_frames('config/hmdb51/ClipRN50_TextRN50.yaml', train=True)
    # run('config/hmdb51/UML.yaml', train=True)
    # run('config/hmdb51/MoLo.yaml', train=True)
    # run('config/hmdb51/VideoMAEv1.yaml', train=False)
    # run('config/hmdb51/VFS.yaml', train=False)

    # HMDB51
    # run('config/hmdb51/BaselineRN50.yaml', train=False)
    # run('config/hmdb51/BaselineViT.yaml', train=False)
    # run('config/hmdb51/ClipRN50_TextRN50.yaml', train=True)
    run('config/hmdb51/ClipViT16B_TextViT16B.yaml', train=True)

    # UCF101
    # run('config/ucf101/ClipRN50_TextRN50.yaml', train=True)
    # run('config/ucf101/ClipViT16B_TextViT16B.yaml', train=True)

    # Kinetics100
    # run('config/k100/ClipRN50_TextRN50.yaml', train=True)
    # run('config/k100/ClipViT16B_TextViT16B.yaml', train=True)

