import clip
import torch
import torch.nn as nn

class CLIPAdapter(nn.Module):
    def __init__(self, c_in, reduction=4, ratio=0.2):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )
        self.ratio = ratio

    def forward(self, x):
        x = self.ratio * self.fc(x) + (1 - self.ratio) * x
        return x


class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class AdapterVisual(nn.Module):
    def __init__(self, name='RN50', ratio=0.2):
        super(AdapterVisual, self).__init__()
        self.origin_model = clip.load(name, device='cpu')[0].visual
        dim = 1024 if name == 'RN50' else 512
        self.adapter = Adapter(dim)
        self.ratio = ratio

    def forward(self, x):
        x = self.origin_model(x)
        x = self.ratio * self.adapter(x) + (1 - self.ratio) * x
        return x


class AdapterText(nn.Module):
    def __init__(self, name='RN50', ratio=0.2):
        super(AdapterText, self).__init__()
        clip_model = clip.load(name, device='cpu')[0]
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        del clip_model
        dim = 1024 if name == 'RN50' else 512
        self.adapter = Adapter(dim)
        self.ratio = ratio
        self.tokenizer = clip.tokenize

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
        
        x = self.ratio * self.adapter(x) + (1 - self.ratio) * x
        return x
