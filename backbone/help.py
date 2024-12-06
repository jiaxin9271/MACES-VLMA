import clip
import torch.nn as nn
import torchvision.models as models

pth_dict = {
    'Jigsaw': 'jigsaw_r2d50_104epoch_imagenet.pth',
    'RotNet': 'rotnet_r2d50_105ep_imagenet.pth',
    'NPID': 'npid_r2d50_200ep_imagenet.pth',
    'NPID++': 'npid++_r2d50_200ep_imagenet.pth',
    'PIRL': 'pirl_r2d50_200ep_imagenet.pth',
    'SimCLR': 'simclr_r2d50_200ep_imagenet.pth',
    'MoCo': 'moco_r2d50_800ep_imagenet.pth',
    'SwAV': 'swav_r2d50_200ep_imagenet.pth',
    'VFS': 'vfs_r2d50_100ep_k400.pth',
    'BarlowTwins': 'barlow_twins_r2d50_300ep_imagenet.pth',
    'SimSiam': 'simsiam_r2d50_100ep_imagenet.pth',
    'MoCoViT': 'moco_vit_b_16_300ep_imagenet.pth',
    'DINO': 'dino_r2d50_800ep_imagenet.pth'
}

def select_video_backbone(encoder_name='ClipRN50', adapter=False):
    if encoder_name == 'RN50':
        encoder = models.__dict__['resnet50'](weights=models.ResNet50_Weights.IMAGENET1K_V1)
        encoder.fc = nn.Identity()
        encoder_dim = 2048
    elif encoder_name == 'MAXI':
        encoder = clip.load('ViT-B/16', device='cpu')[0].visual
        import torch
        pth = torch.load('/home/cjx/pretrained/maxi_ckpt.pth')
        new_state_dict = {}
        for k, v in pth['model'].items():
            if 'image_encoder' not in k:
                continue
            new_state_dict[k[21:]] = v
        encoder.load_state_dict(new_state_dict)
        encoder_dim = 512  
    elif encoder_name == 'ClipRN50':
        if adapter:
            from backbone.adapter import AdapterCLIPRN50
            encoder = AdapterCLIPRN50()
            # from backbone.molo import MoLoRN50
            # encoder = MoLoRN50()
        else:
            # from backbone.clip_adapter import AdapterVisual
            # encoder = AdapterVisual('RN50')
            encoder = clip.load('RN50', device='cpu')[0].visual
        encoder_dim = 1024
    elif encoder_name == 'ClipViT16B':
        if adapter:
            from backbone.adapter import AdapterCLIPViT
            encoder = AdapterCLIPViT('ViT-B/16')
            # from backbone.molo import MoLoViT
            # encoder = MoLoViT('ViT-B/16')
        else:
            # from backbone.clip_adapter import AdapterVisual
            # encoder = AdapterVisual('ViT-B/16')
            encoder = clip.load('ViT-B/16', device='cpu')[0].visual
        encoder_dim = 512
    elif encoder_name == 'SimMIM':
        from backbone.simmim import build_simmim_vit
        encoder = build_simmim_vit()
        encoder_dim = 768
    elif encoder_name == 'MoCov3':
        from backbone.moco_v3 import vit_base
        encoder = vit_base()
        encoder_dim = 768
    elif encoder_name == 'CIM':
        from backbone.cim import vit_base_patch16
        encoder = vit_base_patch16()
        encoder_dim = 768
    elif encoder_name == 'DINOViT16B':
        from transformers import ViTModel
        encoder = ViTModel.from_pretrained('facebook/dino-vitb16', local_files_only=True)
        encoder_dim = 768
    elif encoder_name == 'MAEViT16B':
        from transformers import ViTMAEModel
        encoder = ViTMAEModel.from_pretrained('facebook/vit-mae-base', local_files_only=True)
        encoder_dim = 768
    elif encoder_name == 'VideoMAEv1':
        # from backbone.videomae_v1 import vit_base_patch16_224
        # encoder = vit_base_patch16_224()
        from transformers import VideoMAEModel
        encoder = VideoMAEModel.from_pretrained('MCG-NJU/videomae-base', local_files_only=True)
        encoder_dim = 768
    elif encoder_name == 'VideoMAEv2':
        from backbone.videomae_v2 import vit_base_patch16_224
        encoder = vit_base_patch16_224()
        encoder_dim = 768
    elif 'Slow' in encoder_name:
        from backbone.slowfast import slow_only
        encoder = slow_only(encoder_name)
        encoder_dim = 2048
    elif encoder_name == 'VideoR3D50':
        from backbone.r3d import ResNet, Bottleneck
        import torch
        encoder = ResNet(Bottleneck, [3, 4, 6, 3])
        pth_path = f'/home/cjx/pretrained/r3d50_k700.pth'
        pth = torch.load(pth_path, map_location='cuda')
        encoder.load_state_dict(pth['state_dict'])
        encoder_dim = 2048
    else:
        import torch
        encoder = models.__dict__['resnet50']()
        encoder.fc = nn.Identity()
        pth_path = f'/home/cjx/pretrained/{pth_dict[encoder_name]}'
        pth = torch.load(pth_path, map_location='cuda')
        encoder.load_state_dict(pth['state_dict'])
        encoder_dim = 2048
    return encoder, encoder_dim


def select_text_backbone(encoder_name='TextRN50', adapter=False):
    if encoder_name == 'TextRN50':
        if adapter:
            from backbone.adapter import AdapterText
            encoder = AdapterText('RN50') 
        else:
            # from backbone.coop import CoOpText
            # encoder = CoOpText('RN50')
            # from backbone.clip_adapter import AdapterText
            # encoder = AdapterText('RN50')
            from backbone.adapter import TextCLIP
            encoder = TextCLIP('RN50')
        encoder_dim = 1024
    elif encoder_name == 'TextViT16B':
        if adapter:
            from backbone.adapter import AdapterText
            encoder = AdapterText('ViT-B/16')  
        else:
            # from backbone.coop import CoOpText
            # encoder = CoOpText('ViT-B/16')
            # from backbone.clip_adapter import AdapterText
            # encoder = AdapterText('ViT-B/16')
            from backbone.adapter import TextCLIP
            encoder = TextCLIP('ViT-B/16')
        encoder_dim = 512
    else:
        raise ValueError('text_backbone error')
    return encoder, encoder_dim
