import torch
from backbone.slowfast.utils.parser import load_config, parse_args
from backbone.slowfast.config.defaults import assert_and_infer_cfg
from backbone.slowfast.models.video_model_builder import ResNet


config_file_dict = {
    'VideoMoCoSlowR50': '/home/cjx/ufsar/code/backbone/slowfast/MoCo_SlowR50_8x8.yaml',
    'VideoSimCLRSlowR50': '/home/cjx/ufsar/code/backbone/slowfast/SimCLR_SlowR50_8x8.yaml',
    'VideoBYOLSlowR50': '/home/cjx/ufsar/code/backbone/slowfast/BYOL_SlowR50_8x8.yaml',
    'VideoSwAVSlowR50': '/home/cjx/ufsar/code/backbone/slowfast/SwAV_Slow_R50_8x8.yaml',
}


pth_dict = {
    'VideoMoCoSlowR50': '/home/cjx/pretrained/MoCo_SlowR50_8x8_T4_epoch_00200.pyth',
    'VideoSimCLRSlowR50': '/home/cjx/pretrained/SimCLR_SlowR50_8x8_T2_epoch_00200.pyth',
    'VideoBYOLSlowR50': '/home/cjx/pretrained/BYOL_SlowR50_8x8_T4_epoch_00200.pyth',
    'VideoSwAVSlowR50': '/home/cjx/pretrained/SwAV_SlowR50_8x8_T2_epoch_00200.pyth',
}


def slow_only(video_model='VideoMoCoSlowR50'):
    args = parse_args(config_file_dict[video_model])
    cfg = load_config(args, args.cfg_files)
    cfg = assert_and_infer_cfg(cfg)
    slow_model = ResNet(cfg)
    pth = torch.load(pth_dict[video_model], map_location='cuda')
    new_state_dict = {}
    for k, v in pth['model_state'].items():
        if 'backbone' not in k:
            continue
        if 'backbone_hist' in k:
            continue
        if 'projection' in k:
            continue
        new_state_dict[k[9:]] = v
    slow_model.load_state_dict(new_state_dict)
    return slow_model
