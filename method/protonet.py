import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone import select_text_backbone, select_video_backbone


class ProtoNet(nn.Module):

    def __init__(self, args):
        super(ProtoNet, self).__init__()
        # basic
        self.frames = args.frames
        self.batch_size = args.batch_size
        self.num_tasks = args.num_tasks
        self.way = args.way
        self.shot = args.shot
        self.query = args.query
        self.eval_way = args.eval_way
        self.eval_shot = args.eval_shot
        self.eval_query = args.eval_query
        self.use_template = args.use_template

        # backbone
        self.use_video_encoder = args.use_video_encoder
        self.use_text_encoder = args.use_text_encoder
        self.video_encoder_dim = 0
        self.text_encoder_dim = 0
        self.use_video_cache = args.use_video_cache
        self.use_text_cache = args.use_text_cache
        self.use_video_zero = args.use_video_zero
        self.use_text_zero = args.use_text_zero
        # self.vtc_tau = nn.parameter.Parameter(torch.ones([]) * args.vtc_tau)
        self.vtc_tau = args.vtc_tau
        self.bkd_lambda = args.bkd_lambda

        if self.use_video_encoder:
            self.video_encoder, self.video_encoder_dim = select_video_backbone(args.video_encoder, args.use_video_adapter)
        else:
            self.video_encoder = nn.Identity()
        if self.use_text_encoder:
            self.text_encoder, self.text_encoder_dim = select_text_backbone(args.text_encoder, args.use_text_adapter)
        else:
            self.text_encoder = nn.Identity()
      
    def forward(self, video, text, label, zero_video, zero_text):
        video_embs, text_embs = self.get_feature(video, text)
        if not self.use_video_encoder and self.use_text_encoder:
            return self.single_forward(text_embs)
        
        if self.use_video_encoder and not self.use_text_encoder:
            return self.single_forward(video_embs)
        
        return self.video_text_forward_mm(video_embs, text_embs, zero_video, zero_text)

    def video_text_forward_mm(self, video_embs, text_embs, zero_video, zero_text):
        support_idx, query_idx = self.split_instances()
        bs = self.num_tasks if self.training else 1
        way = self.way if self.training else self.eval_way
        query = self.query if self.training else self.eval_query

        # => video
        if self.use_video_encoder:
            support_video_embs = video_embs[support_idx].mean(dim=1) # [256, way, 8, 1024]
            query_video_embs = video_embs[query_idx].reshape(bs, query * way, self.frames, -1) # [256, query*way, 8, 1024]
            support_video_embs = F.normalize(support_video_embs, dim=-1) 
            query_video_embs = F.normalize(query_video_embs, dim=-1)
            sim_v = torch.einsum('nijk,nxyk->nxiyj', support_video_embs, query_video_embs)  # [256, query*way, way, 8, 8]
            sim_v = ((sim_v.max(dim=-1)[0].sum(dim=-1) + sim_v.max(dim=-2)[0].sum(dim=-1)) / 16).reshape(-1, way)

        # => text
        if self.use_text_encoder:
            support_text_embs = text_embs[support_idx].mean(dim=1) # [256, way, 1024]
            query_text_embs = text_embs[query_idx].reshape(bs, query * way, -1) # [256, query*way, 1024]
            support_text_embs = F.normalize(support_text_embs, dim=-1)
            query_text_embs = F.normalize(query_text_embs, dim=-1)
            sim_t = torch.einsum('ijk,imk->imj', support_text_embs, query_text_embs).reshape(-1, way)

        # => clip
        if self.video_encoder_dim == self.text_encoder_dim:
            sim_v2t = torch.einsum('njk,nxyk->nxyj', support_text_embs.detach(), query_video_embs).mean(dim=-2).reshape(-1, way)
            sim_t2v = torch.einsum('nijk,nxk->nxij', support_video_embs.detach(), query_text_embs).mean(dim=-1).reshape(-1, way)
            # sim_v = sim_v + sim_v2t
            # sim_t = sim_t + sim_t2v

        # => zero-shot video
        if self.use_video_zero:
            zero_video = zero_video.cuda(non_blocking=True)
            zero_support_video_embs = zero_video[support_idx].mean(dim=1) # [256, way, 8, 1024]
            zero_query_video_embs = zero_video[query_idx].reshape(bs, query * way, self.frames, -1) # [256, query*way, 8, 1024]
            zero_support_video_embs = F.normalize(zero_support_video_embs, dim=-1) 
            zero_query_video_embs = F.normalize(zero_query_video_embs, dim=-1)
            zero_sim_v = torch.einsum('nijk,nxyk->nxiyj', zero_support_video_embs, zero_query_video_embs)  # [256, query*way, way, 8, 8]
            zero_sim_v = ((zero_sim_v.max(dim=-1)[0].sum(dim=-1) + zero_sim_v.max(dim=-2)[0].sum(dim=-1)) / 16).reshape(-1, way)
        
        # => zero-shot text
        if self.use_text_zero:
            zero_text = zero_text.cuda(non_blocking=True)
            zero_support_text_embs = zero_text[support_idx].mean(dim=1) # [256, way, 1024]
            zero_query_text_embs = zero_text[query_idx].reshape(bs, query * way, -1) # [256, query*way, 1024]
            zero_support_text_embs = F.normalize(zero_support_text_embs, dim=-1)
            zero_query_text_embs = F.normalize(zero_query_text_embs, dim=-1)
            zero_sim_t = torch.einsum('ijk,imk->imj', zero_support_text_embs, zero_query_text_embs).reshape(-1, way)

        
        labels = torch.arange(way, dtype=torch.long).repeat(query * bs).cuda(non_blocking=True)
        if self.use_template:
            acc = self.acc(sim_v + sim_t2v, labels)
        else:
            acc = self.acc(self.ami_acc(sim_v + sim_v2t, sim_t + sim_t2v), labels)
        acc_v = self.acc(sim_v + sim_v2t, labels)
        acc_t = self.acc(sim_t + sim_t2v, labels)
        if self.training:
            ce_v = torch.tensor(0)
            ce_t = torch.tensor(0)
            vtc = torch.tensor(0)
            bkd = torch.tensor(0)

            ce_v = F.cross_entropy(sim_v, labels)
            ce_t = F.cross_entropy(sim_t, labels)
            # vtc = self.vtc_loss_mm(video_embs, text_embs)
            vtc = self.loss_clip_w_class(video_embs, text_embs)
            bkd = self.bkd_loss_mm(sim_v + sim_v2t, sim_t + sim_t2v, zero_sim_v, zero_sim_t)
            loss = vtc + self.bkd_lambda * bkd + ce_v + ce_t
            return loss, vtc, bkd, ce_v, ce_t, acc, acc_v, acc_t
        return acc, acc_v, acc_t

    def vtc_loss_mm(self, video_embs, text_embs):
        video_embs = F.normalize(video_embs, dim=-1) # [64, 8, 512]
        text_embs = F.normalize(text_embs, dim=-1) # [64, 512]
        sim_v2t = torch.einsum('nk,mjk->mnj', text_embs, video_embs).mean(dim=-1) / self.vtc_tau
        sim_t2v = sim_v2t.T

        target = torch.zeros_like(sim_v2t)
        for i in range(target.shape[0]):
            for j in range(self.shot + self.query):
                target[i][i % self.batch_size + j * self.batch_size] = 1
        
        loss_v2t = - (F.log_softmax(sim_v2t, dim=-1) * target).sum(1).mean(0)
        loss_t2v = - (F.log_softmax(sim_t2v, dim=-1) * target).sum(1).mean(0)
        return (loss_v2t + loss_t2v) / 2

    def loss_clip_w_class(self, video_embs, text_embs):
        video_embs = F.normalize(video_embs, dim=-1)  # [64, 8, 512]
        text_embs = F.normalize(text_embs, dim=-1)  # [64, 512]
        sim_v2t = torch.einsum('nk,mjk->mnj', text_embs, video_embs).mean(dim=-1) / self.vtc_tau
        sim_t2v = sim_v2t.T
        pseudo_label = torch.arange(self.batch_size, dtype=torch.long).repeat(self.query + self.shot).cuda(non_blocking=True)

        def get_loss(logits_per_data, pseudo_label):
            # 正对
            mask = torch.eye(logits_per_data.shape[0], dtype=torch.bool).to(logits_per_data.device)
            # 负对
            if pseudo_label is not None:
                pseudo_label = pseudo_label.contiguous().view(-1, 1)
                class_mask = (1 - torch.eq(pseudo_label, pseudo_label.T).float().to(logits_per_data.device) * 1.0)
                class_mask.diagonal().fill_(1.0)
            else:
                class_mask = torch.ones_like(logits_per_data)
            
            exp_logits = torch.exp(logits_per_data) * class_mask
            log_prob = torch.log(exp_logits.sum(dim=1) + 1e-5)
            loss = - (logits_per_data[mask] - log_prob)
            loss = torch.mean(loss)
            return loss
        
        loss_all = (get_loss(sim_v2t, pseudo_label) + get_loss(sim_t2v, pseudo_label)) / 2

        return loss_all

    def bkd_loss_mm(self, sim_v, sim_t, zero_sim_v, zero_sim_t):
        p_z_v = F.softmax(zero_sim_v, dim=-1) 
        p_z_t = F.softmax(zero_sim_t, dim=-1) 
        p_v = F.softmax(sim_v, dim=-1) 
        p_t = F.softmax(sim_t, dim=-1) 
        p_tea_v = (p_v + p_z_v) / 2
        p_tea_t = (p_t + p_z_t) / 2
        confident_v =  p_tea_v.max(1)[0]
        confident_t = p_tea_t.max(1)[0]
        kl_v = torch.tensor(0).to(p_v.device).float()
        num_v = torch.tensor(0).to(p_v.device).float()
        kl_t = torch.tensor(0).to(p_v.device).float()
        num_t = torch.tensor(0).to(p_v.device).float()
        for i in range(self.num_tasks):
            if confident_v[i] > confident_t[i]:
                kl_v += confident_v[i] * (p_tea_v[i].detach() * torch.log(p_tea_v[i].detach() / p_t[i])).sum(0)
                num_v += confident_v[i]
            else:
                kl_t += confident_t[i] * (p_tea_t[i].detach() * torch.log(p_tea_t[i].detach() / p_v[i])).sum(0)
                num_t += confident_t[i]  
        kl_v = kl_v / num_v if kl_v != 0 else kl_v
        kl_t = kl_t / num_t if kl_t != 0 else kl_t
        kl = kl_v + kl_t
        return kl


    def ami_acc(self, sim_v, sim_t):
        p_v = F.softmax(sim_v, dim=-1)
        p_t = F.softmax(sim_t, dim=-1)
        confident_v = p_v.max(1)[0]
        confident_t = p_t.max(1)[0]
        w_v = confident_v / (confident_v + confident_t)
        w_t = confident_t / (confident_v + confident_t)
        return sim_v * w_v[:, None] + sim_t * w_t[:, None]


    def mai_forward(self, video_embs, text_embs):
        support_idx, query_idx = self.split_instances()
        bs = self.num_tasks if self.training else 1
        way = self.way if self.training else self.eval_way
        query = self.query if self.training else self.eval_query

        # # => video
        # if self.use_video_encoder:
        #     support_video_embs = video_embs[support_idx].mean(dim=1) # [256, way, 8, 1024]
        #     query_video_embs = video_embs[query_idx].reshape(bs, query * way, self.frames, -1) # [256, query*way, 8, 1024]
        #     support_video_embs = F.normalize(support_video_embs, dim=-1) 
        #     query_video_embs = F.normalize(query_video_embs, dim=-1)
        #     sim_v = torch.einsum('nijk,nxyk->nxiyj', support_video_embs, query_video_embs)  # [256, query*way, way, 8, 8]
        #     sim_v = ((sim_v.max(dim=-1)[0].sum(dim=-1) + sim_v.max(dim=-2)[0].sum(dim=-1)) / 16).reshape(-1, way)

        # # => text
        # if self.use_text_encoder:
        #     support_text_embs = text_embs[support_idx].mean(dim=1) # [256, way, 1024]
        #     query_text_embs = text_embs[query_idx].reshape(bs, query * way, -1) # [256, query*way, 1024]
        #     support_text_embs = F.normalize(support_text_embs, dim=-1)
        #     query_text_embs = F.normalize(query_text_embs, dim=-1)
        #     sim_t = torch.einsum('ijk,imk->imj', support_text_embs, query_text_embs).reshape(-1, way)

        # => video
        if self.use_video_encoder:
            support_video_embs = video_embs[support_idx].mean(dim=1) # [256, way, 8, 1024]
            query_video_embs = video_embs[query_idx].reshape(bs, query * way, self.frames, -1) # [256, query*way, 8, 1024]
            support_video_embs = F.normalize(support_video_embs, dim=-1) 
            query_video_embs = F.normalize(query_video_embs, dim=-1)
            sim_v = torch.einsum('nijk,nxyk->nxiyj', support_video_embs, query_video_embs)  # [256, query*way, way, 8, 8]
            sim_v = ((sim_v.max(dim=-1)[0].sum(dim=-1) + sim_v.max(dim=-2)[0].sum(dim=-1)) / 16).reshape(-1, way)

        # => clip
        if self.video_encoder_dim == self.text_encoder_dim:
            query_text_embs = text_embs[query_idx].reshape(bs, query * way, -1) # [256, query*way, 1024]
            query_text_embs = F.normalize(query_text_embs, dim=-1)
            sim_t2v = torch.einsum('nijk,nxk->nxij', support_video_embs.detach(), query_text_embs).mean(dim=-1).reshape(-1, way)
        
        labels = torch.arange(way, dtype=torch.long).repeat(query * bs).cuda(non_blocking=True)
        # return self.acc(sim_v, labels)
        # return self.acc(sim_t2v, labels)
        return self.acc(sim_v + sim_t2v, labels)
        # return self.acc(sim_v + sim_t, labels)

    def video_text_forward(self, video_embs, text_embs):
        support_idx, query_idx = self.split_instances()
        bs = self.num_tasks if self.training else 1
        way = self.way if self.training else self.eval_way
        query = self.query if self.training else self.eval_query

        # => video
        if self.use_video_encoder:
            support_video_embs = video_embs[support_idx].mean(dim=1) # [256, way, 8, 1024]
            query_video_embs = video_embs[query_idx].reshape(bs, query * way, self.frames, -1) # [256, query*way, 8, 1024]
            support_video_embs = F.normalize(support_video_embs, dim=-1) 
            query_video_embs = F.normalize(query_video_embs, dim=-1)
            sim_v = torch.einsum('nijk,nxyk->nxiyj', support_video_embs, query_video_embs)  # [256, query*way, way, 8, 8]
            sim_v = ((sim_v.max(dim=-1)[0].sum(dim=-1) + sim_v.max(dim=-2)[0].sum(dim=-1)) / 16).reshape(-1, way)

        # => text
        if self.use_text_encoder:
            support_text_embs = text_embs[support_idx].mean(dim=1) # [256, way, 1024]
            query_text_embs = text_embs[query_idx].reshape(bs, query * way, -1) # [256, query*way, 1024]
            support_text_embs = F.normalize(support_text_embs, dim=-1)
            query_text_embs = F.normalize(query_text_embs, dim=-1)
            sim_t = torch.einsum('ijk,imk->imj', support_text_embs, query_text_embs).reshape(-1, way)

        # => clip
        if self.video_encoder_dim == self.text_encoder_dim:
            sim_v2t = torch.einsum('njk,nxyk->nxyj', support_text_embs.detach(), query_video_embs).mean(dim=-2).reshape(-1, way)
            sim_t2v = torch.einsum('nijk,nxk->nxij', support_video_embs.detach(), query_text_embs).mean(dim=-1).reshape(-1, way)
            sim_v = self.alpha * sim_v + sim_v2t
            sim_t = self.beta * sim_t + sim_t2v
        
        labels = torch.arange(way, dtype=torch.long).repeat(query * bs).cuda(non_blocking=True)
        # acc = self.acc(sim_v + sim_t + sim_v2t + sim_t2v, labels)
        # acc = self.acc(sim_v + sim_t, labels)
        acc = self.acc(self.ami_acc(sim_v, sim_t), labels)
        acc_v = self.acc(sim_v, labels)
        acc_t = self.acc(sim_t, labels)
        if self.training:
            # vtc = self.vtc_loss_2(video_embs, text_embs, support_idx, query_idx)
            vtc = self.vtc_loss_1(video_embs, text_embs)   
            # vtc = torch.tensor(0)

            kl = self.kl_loss(sim_v, sim_t)
            ce_v = F.cross_entropy(sim_v, labels)
            ce_t = F.cross_entropy(sim_t, labels)
            # kl = torch.tensor(0)
            # ce_v = torch.tensor(0)
            # ce_t = torch.tensor(0)
            loss = self.vtc_lambda * vtc + self.kl_lambda * kl + ce_v + ce_t
            return loss, vtc, kl, ce_v, ce_t, acc, acc_v, acc_t
        return acc, acc_v, acc_t
        # return acc
    
    def video_text_forward_5x5(self, video, text):
        video_embs, text_embs = self.get_feature(video, text)
        support_video_embs = F.normalize(video_embs[:5], dim=-1)
        query_video_embs = F.normalize(video_embs[5:], dim=-1)
        support_text_embs = F.normalize(text_embs[:5], dim=-1)
        query_text_embs = F.normalize(text_embs[5:], dim=-1)

        if self.use_video_encoder:
            sim_v = torch.einsum('ijk,xyk->xiyj', support_video_embs, query_video_embs)  # [256, query*way, way, 8, 8]
            sim_v = ((sim_v.max(dim=-1)[0].sum(dim=-1) + sim_v.max(dim=-2)[0].sum(dim=-1)) / 16).reshape(-1, 5)
        
        if self.use_text_encoder:
            sim_t = torch.einsum('jk,mk->mj', support_text_embs, query_text_embs).reshape(-1, 5)

        if self.video_encoder_dim == self.text_encoder_dim:
            sim_v2t = torch.einsum('jk,xyk->xyj', support_text_embs.detach(), query_video_embs).mean(dim=-2).reshape(-1, 5)
            sim_t2v = torch.einsum('ijk,xk->xij', support_video_embs.detach(), query_text_embs).mean(dim=-1).reshape(-1, 5)
            sim_v = self.alpha * sim_v + sim_v2t
            sim_t = self.beta * sim_t + sim_t2v
        
        labels = torch.arange(5, dtype=torch.long).repeat(1).cuda(non_blocking=True)
        # logits = self.ami_acc(sim_v, sim_t)
        logits = sim_v + sim_t
        # logits = sim_t
        acc = self.acc(logits, labels)
        return F.softmax(logits * 2, dim=-1), acc

    def ami_acc(self, sim_v, sim_t):
        p_v = F.softmax(sim_v, dim=-1)
        p_t = F.softmax(sim_t, dim=-1)
        confident_v = p_v.max(1)[0]
        confident_t = p_t.max(1)[0]
        w_v = confident_v / (confident_v + confident_t)
        w_t = confident_t / (confident_v + confident_t)
        return sim_v * w_v[:, None] + sim_t * w_t[:, None]

    def vtc_loss_2(self, video_embs, text_embs, support_idx, query_idx):
        support_video_embs = video_embs[support_idx] # [128, 1, 5, 8, 1024]
        query_video_embs = video_embs[query_idx] # [128, 5, 5, 8, 1024]
        support_text_embs = text_embs[support_idx] # [128, 1, 5, 1024]
        query_text_embs = text_embs[query_idx] # [128, 5, 5, 1024]
        
        video_embs = torch.cat([support_video_embs, query_video_embs], dim=1)
        text_embs = torch.cat([support_text_embs, query_text_embs], dim=1)
        video_embs = F.normalize(video_embs, dim=-1) # [128, 6, 5, 8, 1024]
        text_embs = F.normalize(text_embs, dim=-1) # [128, 6, 5, 1024]
        num = (self.query + self.shot) * self.way
        video_embs = video_embs.reshape(self.num_tasks, num, self.frames, -1) # [128, 30, 8, 1024]
        text_embs = text_embs.reshape(self.num_tasks, num, -1) # [128, 30, 1024]

        sim_v2t = torch.einsum('znk,zmjk->zmnj', text_embs, video_embs).mean(dim=-1) / self.vtc_tau
        sim_t2v = sim_v2t.permute(0, 2, 1)
        target = torch.zeros((num, num)).cuda()
        for i in range(target.shape[0]):
            for j in range(self.shot + self.query):
                target[i][i % self.way + j * self.way] = 1
        target = target.unsqueeze(0).repeat(self.num_tasks, 1, 1)

        loss_v2t = F.cross_entropy(sim_v2t, target)
        loss_t2v = F.cross_entropy(sim_t2v, target)

        return (loss_v2t + loss_t2v) / 2

    def vtc_loss_1(self, video_embs, text_embs):
        video_embs = F.normalize(video_embs, dim=-1) # [64, 8, 512]
        text_embs = F.normalize(text_embs, dim=-1) # [64, 512]
        sim_v2t = torch.einsum('nk,mjk->mnj', text_embs, video_embs).mean(dim=-1) / self.vtc_tau
        sim_t2v = sim_v2t.T

        target = torch.zeros_like(sim_v2t)
        for i in range(target.shape[0]):
            for j in range(self.shot + self.query):
                target[i][i % self.batch_size + j * self.batch_size] = 1
        
        loss_v2t = - (F.log_softmax(sim_v2t, dim=-1) * target).sum(1).mean(0)
        loss_t2v = - (F.log_softmax(sim_t2v, dim=-1) * target).sum(1).mean(0)
        return (loss_v2t + loss_t2v) / 2

    def kl_loss(self, sim_v, sim_t):
        # v -> t
        # return F.kl_div(
        #     F.log_softmax(sim_t, dim=1), 
        #     F.softmax(sim_v.detach(), dim=1),
        #     reduction='batchmean'
        # )

        # t -> v
        # return F.kl_div(
        #     F.log_softmax(sim_v, dim=1), 
        #     F.softmax(sim_t.detach(), dim=1),
        #     reduction='batchmean'
        # )

        p_v = F.softmax(sim_v / self.kl_tau, dim=-1) 
        p_t = F.softmax(sim_t / self.kl_tau, dim=-1) 
        confident_v = p_v.max(1)[0]
        confident_t = p_t.max(1)[0]
        kl_v = torch.tensor(0).to(p_v.device).float()
        num_v = torch.tensor(0).to(p_v.device).float()
        kl_t = torch.tensor(0).to(p_v.device).float()
        num_t = torch.tensor(0).to(p_v.device).float()
        for i in range(self.num_tasks):
            if confident_v[i] > confident_t[i]:
                kl_v += confident_v[i] * (p_v[i].detach() * torch.log(p_v[i].detach() / p_t[i])).sum(0)
                num_v += confident_v[i]
            else:
                kl_t += confident_t[i] * (p_t[i].detach() * torch.log(p_t[i].detach() / p_v[i])).sum(0)
                num_t += confident_t[i]  
        kl_v = kl_v / num_v if kl_v != 0 else kl_v
        kl_t = kl_t / num_t if kl_t != 0 else kl_t
        kl = kl_v + kl_t
        return kl

    def single_forward(self, instance_embs):
        support_idx, query_idx = self.split_instances()
        bs = self.num_tasks if self.training else 1
        way = self.way if self.training else self.eval_way
        query = self.query if self.training else self.eval_query
        support_embs = instance_embs[support_idx].mean(dim=1)  # [256, way, (8), 2048]
        query_embs = instance_embs[query_idx]  # [256, query, way, (8), 2048]
        
        # support_embs = support_embs.mean(dim=-2)
        # query_embs = query_embs.mean(dim=-2)
        # query_embs = query_embs.reshape(bs, query * way, -1)  # [256, query*way, 2048]
        # support_embs = F.normalize(support_embs, dim=-1)
        # query_embs = F.normalize(query_embs, dim=-1)
        # logits = torch.einsum('ijk,imk->imj', support_embs, query_embs).reshape(-1, way)
        if self.use_video_encoder:
            query_embs = query_embs.reshape(bs, query * way, self.frames, -1)  # [256, query*way, 8, 2048]
            support_embs = F.normalize(support_embs, dim=-1)
            query_embs = F.normalize(query_embs, dim=-1)
            sim = torch.einsum('nijk,nxyk->nxiyj', support_embs, query_embs)  # [256, query*way, way, 8, 8]
            logits = (sim.max(dim=-1)[0].sum(dim=-1) + sim.max(dim=-2)[0].sum(dim=-1)).reshape(-1, way)
        else:
            query_embs = query_embs.reshape(bs, query * way, -1)  # [256, query*way, 2048]
            support_embs = F.normalize(support_embs, dim=-1)
            query_embs = F.normalize(query_embs, dim=-1)
            logits = torch.einsum('ijk,imk->imj', support_embs, query_embs).reshape(-1, way)
        
        labels = torch.arange(way, dtype=torch.long).repeat(query * bs).cuda(non_blocking=True)
        acc = self.acc(logits, labels)
        if self.training:
            loss = F.cross_entropy(logits, labels)
            return loss, acc
        return acc

    def single_forward_5x5(self, x):
        instance_embs = self.get_video_feature(x) # [10, 8, 1024]
        support_embs = F.normalize(instance_embs[:5], dim=-1)
        query_embs = F.normalize(instance_embs[5:], dim=-1)
        sim = torch.einsum('ijk,xyk->xiyj', support_embs, query_embs)  # [256, query*way, way, 8, 8]
        logits = (sim.max(dim=-1)[0].sum(dim=-1) + sim.max(dim=-2)[0].sum(dim=-1)).reshape(-1, 5)
        labels = torch.arange(5, dtype=torch.long).repeat(1).cuda(non_blocking=True)
        acc = self.acc(logits, labels)
        return F.softmax(logits, dim=-1), acc

    def get_feature(self, video, text, use_template=False):
        # video_embs
        if self.use_video_encoder:
            if self.use_video_cache:
                video_embs = video.cuda(non_blocking=True)
            else:
                video_embs = self.get_video_feature(video)
        else:
            video_embs = None

        # text_embs
        if self.use_text_encoder:
            if self.use_text_cache:
                text_embs = text.cuda(non_blocking=True)
            else:
                text_embs = self.get_text_feature(text, use_template)
        else:
            text_embs = None
        
        return video_embs, text_embs
        
    def get_video_feature(self, x):
        x = x.cuda()
        x = x.reshape(-1, *x.shape[-3:])  # [bs x (s + q) x 8, 3, 224, 224]
        video_embs = self.video_encoder(x)
        video_embs = video_embs.reshape(-1, self.frames, self.video_encoder_dim)
        return video_embs

        # x = x.cuda()
        # x = x.reshape(-1, *x.shape[-4:])  # [bs x (s + q) x 8, 3, 224, 224]  
        # video_embs = self.video_encoder(pixel_values=x).last_hidden_state
        # video_embs = video_embs.mean(dim=1)
        # return video_embs

    def get_text_feature(self, text, use_template=False):
        if not use_template:
            return self.text_encoder(text)

        prompt_template = [
            f"a photo of action {{}}",
            f"a picture of action {{}}",
            f"Human action of {{}}",
            f"{{}}, an action",
            f"{{}} this is an action",
            f"{{}}, a video of action",
            f"Playing action of {{}}",
            f"{{}}",
            f"Playing a kind of action, {{}}",
            f"Doing a kind of action, {{}}",
            f"Look, the human is {{}}",
            f"Can you recognize the action of {{}}?",
            f"Video classification of {{}}",
            f"A video of {{}}",
            f"The man is {{}}",
            f"The woman is {{}}",
        ]
        # prompt_dict = {}
        texts = [template.format(text[0]) for template in prompt_template] # format with class
        # print(texts)
        # exit(0)
        texts = [te.replace("_", " ") for te in texts]
        class_embeddings = self.text_encoder(texts) #embed with text encoder
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm()  
        return class_embedding

    def split_instances(self):
        if self.training:
            return self.split_instances_normal(self.num_tasks, self.shot, self.query, self.way, self.batch_size)
        else:
            return self.split_instances_normal(1, self.eval_shot, self.eval_query, self.eval_way, self.eval_way)

    def split_instances_normal(self, num_tasks, num_shot, num_query, num_way, num_class=None):
        num_class = num_way if (num_class is None or num_class < num_way) else num_class
        permuted_ids = torch.zeros(num_tasks, num_shot + num_query, num_way).long()
        for i in range(num_tasks):
            clsmap = torch.randperm(num_class)[:num_way]
            for j, clsid in enumerate(clsmap):
                permuted_ids[i, :, j].copy_(torch.randperm((num_shot + num_query)) * num_class + clsid)
        if torch.cuda.is_available():
            permuted_ids = permuted_ids.cuda()
        support_idx, query_idx = torch.split(permuted_ids, [num_shot, num_query], dim=1)
        return support_idx, query_idx

    def acc(self, logits, label):
        pred = torch.argmax(logits, dim=-1)
        if torch.cuda.is_available():
            return (pred == label).type(torch.cuda.FloatTensor).mean().item()
        else:
            return (pred == label).type(torch.FloatTensor).mean().item()

    def save(self, path, epoch, best):
        torch.save({
            'epoch': epoch,
            'best': best,
            'video_encoder': self.video_encoder.state_dict(),
            'text_encoder': self.text_encoder.state_dict(),
            'vtc_tau': self.vtc_tau
        }, path)

    def load(self, checkpoint_dict):
        self.video_encoder.load_state_dict(checkpoint_dict['video_encoder'])
        self.text_encoder.load_state_dict(checkpoint_dict['text_encoder'])
        self.vtc_tau = checkpoint_dict['vtc_tau']
