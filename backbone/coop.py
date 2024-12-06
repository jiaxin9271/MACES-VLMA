import clip
import torch
import torch.nn as nn


class CoOpText(nn.Module):
    def __init__(self, name='RN50'):
        super(CoOpText, self).__init__()
        clip_model = clip.load(name, device='cpu')[0]
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.tokenizer = clip.tokenize
        del clip_model
        self.prompt_learner = PromptLearner()

    def forward(self, text):
        tokenizer_text = self.tokenizer(text).cuda()
        embedding = self.token_embedding(tokenizer_text)  # [batch_size, n_ctx, d_model]
        x = self.prompt_learner(embedding) + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenizer_text.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_ctx = 16
        self.dim = 512
        ctx_vectors = torch.empty(self.n_ctx, self.dim)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)  # [16, 1024]

    def forward(self, embedding):
        prefix = embedding[:, :1, :] # SOS
        suffix = embedding[:, 1:embedding.shape[1] - self.n_ctx , :]
        ctx = self.ctx.unsqueeze(0).expand(embedding.shape[0], -1, -1)
        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                suffix,  # (n_cls, *, dim)
                ctx,
            ],
            dim=1,
        )
        return prompts
