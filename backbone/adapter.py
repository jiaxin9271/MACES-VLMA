import clip
import torch
import torch.nn as nn


class Adapter(nn.Module):
    def __init__(self, dim, active='relu', skip_connect=True, reduction=4):
        super(Adapter, self).__init__()
        self.down_project = nn.Linear(dim, dim // reduction)
        self.up_project = nn.Linear(dim // reduction, dim)
        if active == 'relu':
            self.active = nn.ReLU(inplace=True)
        elif active == 'gelu':
            self.active = nn.GELU()
        else:
            raise ValueError('active error')
        self.skip_connect = skip_connect
    def forward(self, x):
        x_residual = x
        x = self.down_project(x)
        x = self.active(x)
        x = self.up_project(x)
        if self.skip_connect:
            return x + x_residual
        else:
            return x


class ConvAdapter(nn.Module):
    def __init__(self, dim, reduction=4):
        super(ConvAdapter, self).__init__()
        self.down_project = nn.Conv2d(dim, dim // reduction, 1)
        self.up_project = nn.Conv2d(dim // reduction, dim, 1)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x_residual = x
        x = self.down_project(x)
        x = self.relu(x)
        x = self.up_project(x)
        return x + x_residual


class AdapterCLIPRN50(nn.Module):
    def __init__(self, ):
        super(AdapterCLIPRN50, self).__init__()
        self.origin_model = clip.load('RN50', device='cpu')[0].visual
        self.adapter1 = ConvAdapter(256)
        self.adapter2 = ConvAdapter(512)
        self.adapter3 = ConvAdapter(1024)
        self.adapter4 = ConvAdapter(2048)
        self.adapter5 = Adapter(1024)

    def forward(self, x):
        def stem(x):
            x = self.origin_model.relu1(self.origin_model.bn1(self.origin_model.conv1(x)))
            x = self.origin_model.relu2(self.origin_model.bn2(self.origin_model.conv2(x)))
            x = self.origin_model.relu3(self.origin_model.bn3(self.origin_model.conv3(x)))
            x = self.origin_model.avgpool(x)
            return x
        
        x = stem(x)
        x = self.adapter1(self.origin_model.layer1(x))
        x = self.adapter2(self.origin_model.layer2(x))
        x = self.adapter3(self.origin_model.layer3(x))
        x = self.adapter4(self.origin_model.layer4(x))
        x = self.adapter5(self.origin_model.attnpool(x))

        return x


class AdapterCLIPViT(nn.Module):
    def __init__(self, name):
        super(AdapterCLIPViT, self).__init__()
        self.origin_model = clip.load(name, device='cpu')[0].visual
        self.msa_adapters = []
        self.mlp_adapters = []
        self.block_adapters = []
        for _ in range(len(self.origin_model.transformer.resblocks)):
            # self.msa_adapters.append(Adapter(768, active='gelu', skip_connect=True, reduction=4))
            # self.mlp_adapters.append(Adapter(768, active='gelu', skip_connect=True, reduction=4))
            self.block_adapters.append(Adapter(768, active='gelu', skip_connect=True, reduction=4))
        # self.msa_adapters = nn.Sequential(*self.msa_adapters)
        # self.mlp_adapters = nn.Sequential(*self.mlp_adapters)
        self.block_adapters = nn.Sequential(*self.block_adapters)
        
    def forward(self, x):
        x = self.origin_model.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.origin_model.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.origin_model.positional_embedding.to(x.dtype)
        x = self.origin_model.ln_pre(x) 

        x = x.permute(1, 0, 2)  # NLD -> LND [197, N, 768]
        for i, resblock in enumerate(self.origin_model.transformer.resblocks):
            # MSA
            x = x + resblock.attention(resblock.ln_1(x))
            # x = x + self.msa_adapters[i](resblock.attention(resblock.ln_1(x)))
            # MLP
            x = x + resblock.mlp(resblock.ln_2(x))
            # x = x + self.mlp_adapters[i](resblock.mlp(resblock.ln_2(x)))
            # Block
            x = self.block_adapters[i](x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        
        x = self.origin_model.ln_post(x[:, 0, :])
        x = x @ self.origin_model.proj
        
        return x


class TextCLIP(nn.Module):
    def __init__(self, name):
        super(TextCLIP, self).__init__()
        clip_model = clip.load(name, device='cpu')[0]
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.tokenizer = clip.tokenize
        del clip_model

    def forward(self, text):
        text = self.tokenizer(text).cuda()
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x


class AdapterText(nn.Module):
    def __init__(self, name):
        super(AdapterText, self).__init__()
        clip_model = clip.load(name, device='cpu')[0]
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.tokenizer = clip.tokenize
        del clip_model

        layers = len(self.transformer.resblocks)
        self.msa_adapters = []
        self.mlp_adapters = []
        self.block_adapters = []
        for _ in range(layers):
            # self.msa_adapters.append(Adapter(512, active='gelu', skip_connect=True, reduction=4))
            # self.mlp_adapters.append(Adapter(512, active='gelu', skip_connect=True, reduction=4))
            self.block_adapters.append(Adapter(512, active='gelu', skip_connect=True, reduction=4))
        # self.msa_adapters = nn.Sequential(*self.msa_adapters)
        # self.mlp_adapters = nn.Sequential(*self.mlp_adapters)
        self.block_adapters = nn.Sequential(*self.block_adapters)

    def forward(self, text):
        text = self.tokenizer(text).cuda()
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding
        
        x = x.permute(1, 0, 2)  # NLD -> LND
        for i, resblock in enumerate(self.transformer.resblocks):
            # MSA
            x = x + resblock.attention(resblock.ln_1(x))
            # x = x + self.msa_adapters[i](resblock.attention(resblock.ln_1(x)))
            # MLP
            x = x + resblock.mlp(resblock.ln_2(x))
            # x = x + self.mlp_adapters[i](resblock.mlp(resblock.ln_2(x)))
            # Block
            x = self.block_adapters[i](x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x
