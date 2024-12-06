import os 
import argparse
import yaml
import time
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from method.protonet import ProtoNet
from method.dataset import VideoDataset
from utils import Logger, CategoriesSampler, AverageMeter, compute_confidence_interval, init_seeds
from vcache import extract_feature_for_test_stage


def few_shot_train_mm(args):
    # model
    model = ProtoNet(args)
    model.cuda()
    # log
    logger = Logger(args.save_path, 'train') 
    logger.log(f'train_mode: {args.train_mode}')
    logger.log(f'save_path: {args.save_path}')
    if args.use_video_encoder:
        logger.log(f'video_encoder: {args.video_encoder}')
    if args.use_text_encoder:
        logger.log(f'text_encoder: {args.text_encoder}')
    if args.use_video_zero and args.use_text_zero:
        logger.log(f'zero_shot: True')
    logger.log(f'few-shot: {args.batch_size}-bs {args.num_tasks}-task {args.way}-way {args.shot}-shot {args.query}-query')
    logger.log(f'prompt: {args.prompt_type}')
    logger.log_args(args)

    # data
    train_dataset = VideoDataset(args, split='train', train_mode=args.train_mode)
    if args.use_video_cache and args.use_video_encoder:
        logger.log(f'load video cache {train_dataset.video_cache.shape} from {train_dataset.video_cache_file}')
    if args.use_text_cache and args.use_text_encoder:
        logger.log(f'load text cache {train_dataset.text_cache.shape} from {train_dataset.text_cache_file}')
    logger.log(f'train dataset: {len(train_dataset)} videos')
    train_sampler = CategoriesSampler(
        label=train_dataset.videos_labels_list,
        n_batch=args.num_train_tasks,
        n_cls=args.batch_size,
        n_per=args.shot + args.query
    )
    if args.train_mode == 'Aug':
        def examplar_collate_2d(batch):
            X, Y, Z = [], [], []
            for v, t, l in batch:
                X.append(torch.stack(v))
                Y.append(t)
                Z.append(l)
            video = torch.stack(X)
            label = torch.LongTensor(Z)  
            video = torch.cat(tuple(video.permute(1, 0, 2, 3, 4, 5)), dim=0)
            return video, Y, label
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            collate_fn=examplar_collate_2d,
            pin_memory=True,
            drop_last=True
        )
    else:
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=True
        )
    val_dataset = VideoDataset(args, split='test', train_mode='sup')
    if args.use_video_cache and args.use_video_encoder:
        logger.log(f'load video cache {train_dataset.video_cache.shape} from {train_dataset.video_cache_file}')
    if args.use_text_cache and args.use_text_encoder:
        logger.log(f'load text cache {val_dataset.text_cache.shape} from {val_dataset.text_cache_file}')
    logger.log(f'val dataset: {len(val_dataset)} videos')
    val_sampler = CategoriesSampler(
        label=val_dataset.videos_labels_list,
        n_batch=args.num_val_tasks,
        n_cls=args.eval_way,
        n_per=args.eval_shot + args.eval_query
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # optimizer
    if args.use_video_encoder and args.use_video_adapter:
        for name, param in model.named_parameters():
            if 'video_encoder' not in name:
                continue
            if 'origin_model.ln_post' not in name \
                and 'adapter' not in name \
                and 'origin_model.proj' not in name:
                param.requires_grad = False
    if args.use_text_encoder and args.use_text_adapter:
        for name, param in model.named_parameters():
            if 'text_encoder' not in name:
                continue
            if 'ln_final' not in name \
                and 'adapter' not in name \
                and 'text_projection' not in name:
                param.requires_grad = False
    params_all = sum(p.numel() for p in model.parameters()) / 1000 / 1000
    params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000 / 1000
    logger.log(f'trainable params: {params_trainable:.2f} / {params_all:.2f} M')
    logger.log_parameters(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.total_epoch * len(train_loader))

    # main loop
    scaler = GradScaler()
    best = 0
    start_time = time.time()
    for epoch in range(1, args.total_epoch + 1):
        if epoch == 5:
            break
        best = train_epoch_mm(args, model, train_loader, val_loader, optimizer, lr_schedule, epoch, scaler, best, logger)
    model.save(args.save_path + '/last.pth', args.total_epoch, best)
    logger.log(f'save last.pth')
    logger.log(f'training time: {(time.time() - start_time):.2f}')


def train_epoch_mm(args, model, train_loader, val_loader, optimizer, lr_schedule, epoch, scaler, best, logger):
    model.train()
    losses = AverageMeter('loss:')
    vtces = AverageMeter('vtc:')
    bkdes = AverageMeter('bkd:')
    ce_ves = AverageMeter('ce_v:')
    ce_tes = AverageMeter('ce_t:')
    acces = AverageMeter('acc:')
    accves = AverageMeter('acc_v:')
    acctes = AverageMeter('acc_t:')
    start_time = time.time()
    logger.log(f"epoch {epoch}: lr={optimizer.param_groups[0]['lr']}")
    pbar = tqdm(total=len(train_loader), unit='task')
    for video, text, label, zero_video, zero_text in train_loader:
        if args.amp:
            with autocast():
                loss, vtc, bkd, ce_v, ce_t, acc, acc_v, acc_t = model(video, text, label, zero_video, zero_text)
            losses.update(loss.item())
            vtces.update(vtc.item())
            bkdes.update(bkd.item())
            ce_ves.update(ce_v.item())
            ce_tes.update(ce_t.item())
            acces.update(acc)
            accves.update(acc_v)
            acctes.update(acc_t)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss, vtc, bkd, ce_v, ce_t, acc, acc_v, acc_t = model(label, zero_video, zero_text)
            losses.update(loss.item())
            vtces.update(vtc.item())
            bkdes.update(bkd.item())
            ce_ves.update(ce_v.item())
            ce_tes.update(ce_t.item())
            acces.update(acc)
            accves.update(acc_v)
            acctes.update(acc_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        lr_schedule.step()
        pbar.set_description(
            f'[Train]  loss={losses.val:.4f}, vtc={vtces.val:.4f}, bkd={bkdes.val:.4f}, ce_v={ce_ves.val:.4f}, ce_t={ce_tes.val:.4f}, '
            f'acc={acces.avg * 100:.2f}, acc_v={accves.avg * 100:.2f}, acc_t={acctes.avg * 100:.2f}'
        )
        pbar.update()
    pbar.close()
    logger.log(
        f'loss={losses.avg:.4f}, vtc={vtces.avg:.4f}, bkd={bkdes.avg:.4f}, ce_v={ce_ves.avg:.4f}, ce_t={ce_tes.avg:.4f}, '
        f'acc={acces.avg * 100:.2f}, acc_v={accves.avg * 100:.2f}, acc_t={acctes.avg * 100:.2f}, '
        f'time={(time.time() - start_time):.2f}'
    )
    if epoch % args.val_epoch == 0:
        best = val_epoch_mm(args, model, val_loader, epoch, best, logger)
    return best


def val_epoch_mm(args, model, val_loader, epoch, best, logger):
    model.eval()
    start_time = time.time()
    acces = AverageMeter('acc:')
    accves = AverageMeter('acc_v:')
    acctes = AverageMeter('acc_t:')
    pbar = tqdm(total=len(val_loader), unit='task')
    for video, text, label, zero_video, zero_text in val_loader:
        if args.amp:
            with autocast():
                acc, acc_v, acc_t = model(video, text, label, zero_video, zero_text)
        else:
            acc, acc_v, acc_t = model(video, text, label, zero_video, zero_text)
        acces.update(acc)
        accves.update(acc_v)
        acctes.update(acc_t)
        pbar.set_description(f'[ Val ]  acc={acces.avg * 100:.2f}, acc_v={accves.avg * 100:.2f}, acc_t={acctes.avg * 100:.2f}')
        pbar.update()
    pbar.close()
    va = acces.avg
    if va < best:
        logger.log(
            f'val_acc={va * 100:.2f}, val_best={best * 100:.2f}, '
            f'val_acc_v={accves.avg * 100:.2f}, val_acc_t={acctes.avg * 100:.2f}, '
            f'time={(time.time() - start_time):.2f}', prefix=' Val '
        )
        return best
    else:
        best = va
        logger.log(
            f'val_acc={va * 100:.2f}, val_best={best * 100:.2f}, '
            f'val_acc_v={accves.avg * 100:.2f}, val_acc_t={acctes.avg * 100:.2f}, '
            f'time={(time.time() - start_time):.2f}', prefix=' Val '
        )
        model.save(args.save_path + '/best.pth', epoch, best)
        logger.log(f'save best.pth', prefix=' Val ')
        return best


def few_shot_test_mm(args):
    # log
    logger = Logger(args.save_path, 'test')
    logger.log(f'checkpoint_path: {args.save_path}')
    if args.use_video_encoder:
        logger.log(f'video_encoder: {args.video_encoder}')
    if args.use_text_encoder:
        logger.log(f'text_encoder: {args.text_encoder}')
    if args.use_template:
        logger.log(f'prompt: use_template')
    else:
        logger.log(f'prompt: {args.prompt_type}')
    
    # model
    model = ProtoNet(args)
    if args.eval_checkpoint: 
        print(args.save_path + f'/{args.eval_checkpoint}')
        checkpoint_dict = torch.load(args.save_path + f'/{args.eval_checkpoint}')
        model.load(checkpoint_dict)
        logger.log(f"load {args.eval_checkpoint}, epoch: {checkpoint_dict['epoch']}, best: {checkpoint_dict['best']}")
    model.cuda()
    model.eval()
    logger.log(f'params: {sum(p.numel() for p in model.parameters()) / 1000 / 1000:.2f}M')

    args.use_video_cache = False
    args.use_text_cache = False
    model.use_video_cache = False
    model.use_text_cache = False
    test_dataset = VideoDataset(args, 'test', 'sup')
    extract_feature_for_test_stage(args, model, logger)
    test_dataset.get_test_cache()
    if args.use_video_encoder:
        model.use_video_cache = True
        test_dataset.use_video_cache = True
        logger.log(f'load video cache {test_dataset.video_cache.shape} from {test_dataset.crash_video_cache_file}')
    if args.use_text_encoder:
        model.use_text_cache = True
        test_dataset.use_text_cache = True
        logger.log(f'load text cache {test_dataset.text_cache.shape} from {test_dataset.crash_text_cache_file}')

    # main loop
    if args.eval_all:
        # eval_setting = [(5, 1), (6, 1), (7, 1), (8, 1), (9, 1), (10, 1)]
        # eval_setting = [(5, 1), (5, 3), (5, 5)]
        eval_setting = [(5, 1), (5, 5)]
        # eval_setting = [(5, 1), (5, 2), (5, 3), (5, 4), (5, 5)]
        # eval_setting = [(5, 1)]
    else:
        eval_setting = [(args.eval_way, args.eval_shot)]
    for eval_way, eval_shot in eval_setting:
        model.eval_way = eval_way
        model.eval_shot = eval_shot
        test_sampler = CategoriesSampler(
            label=test_dataset.videos_labels_list,
            n_batch=args.num_eval_tasks,
            n_cls=eval_way,
            n_per=eval_shot + args.eval_query
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_sampler=test_sampler,
            num_workers=args.num_workers,
            pin_memory=True
        )
        acces = AverageMeter('acc:')
        record = np.zeros((args.num_eval_tasks, 2))
        start_time = time.time()
        pbar = tqdm(total=len(test_loader), unit='task')
        for i, (video, text, label, zero_video, zero_text) in enumerate(test_loader, 1):
            if args.amp:
                with autocast():
                    result = model(video, text, label, zero_video, zero_text)
            else:
                result = model(video, text, label, zero_video, zero_text)
            record[i - 1, 1] = result[0]
            acces.update(result[0])
            pbar.set_description(f'[Test]  acc={acces.avg * 100:.2f}')
            pbar.update()
        pbar.close()
        va, vap = compute_confidence_interval(record[:, 1])
        logger.log(
            f'{eval_way}-way {eval_shot}-shot, {args.num_eval_tasks} tasks, '
            f'acc={va * 100:.2f}+-{vap * 100:.2f}, time={(time.time() - start_time):.2f}'
        )


def few_shot_train(args):
    # log
    logger = Logger(args.save_path, 'train')

    # model
    model = ProtoNet(args)
    model.cuda()

    # hyperparameters
    logger.log(f'train_mode: {args.train_mode}')
    logger.log(f'save_path: {args.save_path}')
    if args.use_video_encoder:
        logger.log(f'video_encoder: {args.video_encoder}')
    if args.use_text_encoder:
        logger.log(f'text_encoder: {args.text_encoder}')
    logger.log(f'few-shot: {args.batch_size}-bs {args.num_tasks}-task {args.way}-way {args.shot}-shot {args.query}-query')
    logger.log(f'prompt: {args.prompt_type}')
    logger.log_args(args)

    # data
    train_dataset = VideoDataset(args, split='train', train_mode=args.train_mode)
    if args.use_video_cache and args.use_video_encoder:
        logger.log(f'load video cache {train_dataset.video_cache.shape} from {train_dataset.video_cache_file}')
    if args.use_text_cache and args.use_text_encoder:
        logger.log(f'load text cache {train_dataset.text_cache.shape} from {train_dataset.text_cache_file}')
    logger.log(f'train dataset: {len(train_dataset)} videos')
    train_sampler = CategoriesSampler(
        label=train_dataset.videos_labels_list,
        n_batch=args.num_train_tasks,
        n_cls=args.batch_size,
        n_per=args.shot + args.query
    )
    if args.train_mode == 'Aug':
        def examplar_collate_2d(batch):
            X, Y, Z = [], [], []
            for v, t, l in batch:
                X.append(torch.stack(v))
                Y.append(t)
                Z.append(l)
            video = torch.stack(X)
            label = torch.LongTensor(Z)  
            video = torch.cat(tuple(video.permute(1, 0, 2, 3, 4, 5)), dim=0)
            return video, Y, label
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            collate_fn=examplar_collate_2d,
            pin_memory=True,
            drop_last=True
        )
    else:
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=True
        )
    val_dataset = VideoDataset(args, split='test', train_mode='sup')
    if args.use_video_cache and args.use_video_encoder:
        logger.log(f'load video cache {train_dataset.video_cache.shape} from {train_dataset.video_cache_file}')
    if args.use_text_cache and args.use_text_encoder:
        logger.log(f'load text cache {val_dataset.text_cache.shape} from {val_dataset.text_cache_file}')
    logger.log(f'val dataset: {len(val_dataset)} videos')
    val_sampler = CategoriesSampler(
        label=val_dataset.videos_labels_list,
        n_batch=args.num_val_tasks,
        n_cls=args.eval_way,
        n_per=args.eval_shot + args.eval_query
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # CLIP-Adapter
    # for name, param in model.named_parameters():
    #     if 'adapter' not in name:
    #         param.requires_grad = False

    # CoOp
    # for name, param in model.named_parameters():
    #     if 'ctx' not in name:
    #         param.requires_grad = False

    # if args.use_video_encoder and args.use_video_adapter:
    #     for name, param in model.named_parameters():
    #         if 'video_encoder' not in name:
    #             continue
    #         if 'origin_model.ln_post' not in name \
    #             and 'adapter' not in name \
    #             and 'origin_model.proj' not in name \
    #             and 'time_transformer' not in name:
    #             param.requires_grad = False


    # optimizer
    if args.use_video_encoder and args.use_video_adapter:
        for name, param in model.named_parameters():
            if 'video_encoder' not in name:
                continue
            if 'origin_model.ln_post' not in name \
                and 'adapter' not in name \
                and 'origin_model.proj' not in name:
                param.requires_grad = False
    if args.use_text_encoder and args.use_text_adapter:
        for name, param in model.named_parameters():
            if 'text_encoder' not in name:
                continue
            if 'ln_final' not in name \
                and 'adapter' not in name \
                and 'text_projection' not in name:
                param.requires_grad = False
    params_all = sum(p.numel() for p in model.parameters()) / 1000 / 1000
    params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000 / 1000
    logger.log(f'trainable params: {params_trainable:.2f} / {params_all:.2f} M')
    logger.log_parameters(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.total_epoch * len(train_loader))

    # main loop
    scaler = GradScaler()
    best = 0
    start_time = time.time()
    for epoch in range(1, args.total_epoch + 1):
        if args.use_video_encoder and args.use_text_encoder:
            best = train_epoch_2(args, model, train_loader, val_loader, optimizer, lr_schedule, epoch, scaler, best, logger)
        else:
            best = train_epoch_1(args, model, train_loader, val_loader, optimizer, lr_schedule, epoch, scaler, best, logger)
    model.save(args.save_path + '/last.pth', args.total_epoch, best)
    logger.log(f'save last.pth')
    logger.log(f'training time: {(time.time() - start_time):.2f}')


def train_epoch_2(args, model, train_loader, val_loader, optimizer, lr_schedule, epoch, scaler, best, logger):
    model.train()
    losses = AverageMeter('loss:')
    vtces = AverageMeter('vtc:')
    kles = AverageMeter('kl:')
    ce_ves = AverageMeter('ce_v:')
    ce_tes = AverageMeter('ce_t:')
    acces = AverageMeter('acc:')
    accves = AverageMeter('acc_v:')
    acctes = AverageMeter('acc_t:')
    start_time = time.time()
    logger.log(f"epoch {epoch}: lr={optimizer.param_groups[0]['lr']}")
    pbar = tqdm(total=len(train_loader), unit='task')
    for video, text, _ in train_loader:
        if args.amp:
            with autocast():
                loss, vtc, kl, ce_v, ce_t, acc, acc_v, acc_t = model(video, text)
            losses.update(loss.item())
            vtces.update(vtc.item())
            kles.update(kl.item())
            ce_ves.update(ce_v.item())
            ce_tes.update(ce_t.item())
            acces.update(acc)
            accves.update(acc_v)
            acctes.update(acc_t)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss, vtc, kl, ce_v, ce_t, acc, acc_v, acc_t = model(video, text)
            losses.update(loss.item())
            vtces.update(vtc.item())
            kles.update(kl.item())
            ce_ves.update(ce_v.item())
            ce_tes.update(ce_t.item())
            acces.update(acc)
            accves.update(acc_v)
            acctes.update(acc_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        lr_schedule.step()
        pbar.set_description(
            f'[Train]  loss={losses.val:.4f}, vtc={vtces.val:.4f}, kl={kles.val:.4f}, ce_v={ce_ves.val:.4f}, ce_t={ce_tes.val:.4f}, '
            f'acc={acces.avg * 100:.2f}, acc_v={accves.avg * 100:.2f}, acc_t={acctes.avg * 100:.2f}'
        )
        pbar.update()
    pbar.close()
    logger.log(
        f'loss={losses.avg:.4f}, vtc={vtces.avg:.4f}, kl={kles.avg:.4f}, ce_v={ce_ves.avg:.4f}, ce_t={ce_tes.avg:.4f}, '
        f'acc={acces.avg * 100:.2f}, acc_v={accves.avg * 100:.2f}, acc_t={acctes.avg * 100:.2f}, '
        f'time={(time.time() - start_time):.2f}'
    )
    if epoch % args.val_epoch == 0:
        best = val_epoch_2(args, model, val_loader, epoch, best, logger)
    return best


def val_epoch_2(args, model, val_loader, epoch, best, logger):
    model.eval()
    start_time = time.time()
    acces = AverageMeter('acc:')
    accves = AverageMeter('acc_v:')
    acctes = AverageMeter('acc_t:')
    pbar = tqdm(total=len(val_loader), unit='task')
    for video, text, _ in val_loader:
        if args.amp:
            with autocast():
                acc, acc_v, acc_t = model(video, text)
        else:
            acc, acc_v, acc_t = model(video, text)
        acces.update(acc)
        accves.update(acc_v)
        acctes.update(acc_t)
        pbar.set_description(f'[ Val ]  acc={acces.avg * 100:.2f}, acc_v={accves.avg * 100:.2f}, acc_t={acctes.avg * 100:.2f}')
        pbar.update()
    pbar.close()
    va = acces.avg
    if va < best:
        logger.log(
            f'val_acc={va * 100:.2f}, val_best={best * 100:.2f}, '
            f'val_acc_v={accves.avg * 100:.2f}, val_acc_t={acctes.avg * 100:.2f}, '
            f'time={(time.time() - start_time):.2f}', prefix=' Val '
        )
        return best
    else:
        best = va
        logger.log(
            f'val_acc={va * 100:.2f}, val_best={best * 100:.2f}, '
            f'val_acc_v={accves.avg * 100:.2f}, val_acc_t={acctes.avg * 100:.2f}, '
            f'time={(time.time() - start_time):.2f}', prefix=' Val '
        )
        model.save(args.save_path + '/best.pth', epoch, best)
        logger.log(f'save best.pth', prefix=' Val ')
        return best


def train_epoch_1(args, model, train_loader, val_loader, optimizer, lr_schedule, epoch, scaler, best, logger):
    model.train()
    losses = AverageMeter('loss:')
    acces = AverageMeter('acc:')
    start_time = time.time()
    logger.log(f"epoch {epoch}: lr={optimizer.param_groups[0]['lr']}")
    pbar = tqdm(total=args.num_train_tasks, unit='task')
    iter_task = 1
    for video, text, _ in train_loader:
        if args.amp:
            with autocast():
                loss, acc = model(video, text)
            losses.update(loss.item())
            acces.update(acc)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss, acc = model(video, text)
            losses.update(loss.item())
            acces.update(acc)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        lr_schedule.step()
        if args.train_mode == 'Aug' and iter_task % (args.num_train_tasks + 1) == 0:
            break
        iter_task += 1
        
        pbar.set_description(f'[Train]  loss={losses.val:.4f}, acc={acces.avg * 100:.2f}')
        pbar.update()
    pbar.close()
    logger.log(f'loss={losses.avg:.4f}, acc={acces.avg * 100:.2f}, time={(time.time() - start_time):.2f}')
    if epoch % args.val_epoch == 0:
        best = val_epoch_1(args, model, val_loader, epoch, best, logger)
    return best


def val_epoch_1(args, model, val_loader, epoch, best, logger):
    model.eval()
    start_time = time.time()

    acces = AverageMeter('acc:')
    pbar = tqdm(total=len(val_loader), unit='task')
    for video, text, _ in val_loader:
        if args.amp:
            with autocast():
                acc = model(video, text)
        else:
            acc = model(video, text)
        acces.update(acc)
        pbar.set_description(f'[ Val ]  acc={acces.avg * 100:.2f}')
        pbar.update()
    pbar.close()
    va = acces.avg
    if va < best:
        logger.log(f'val_acc={va * 100:.2f}, val_best={best * 100:.2f}, time={(time.time() - start_time):.2f}', prefix=' Val ')
        return best
    else:
        best = va
        logger.log(f'val_acc={va * 100:.2f}, val_best={best * 100:.2f}, time={(time.time() - start_time):.2f}', prefix=' Val ')
        model.save(args.save_path + '/best.pth', epoch, best)
        logger.log(f'save best.pth', prefix=' Val ')
        return best


def few_shot_test(args):
    # log
    logger = Logger(args.save_path, 'test')
    logger.log(f'checkpoint_path: {args.save_path}')
    if args.use_video_encoder:
        logger.log(f'video_encoder: {args.video_encoder}')
    if args.use_text_encoder:
        logger.log(f'text_encoder: {args.text_encoder}')
    if args.use_template:
        logger.log(f'prompt: use_template')
    else:
        logger.log(f'prompt: {args.prompt_type}')
    
    # model
    model = ProtoNet(args)
    if args.eval_checkpoint: 
        print(args.save_path + f'/{args.eval_checkpoint}')
        checkpoint_dict = torch.load(args.save_path + f'/{args.eval_checkpoint}')
        model.load(checkpoint_dict)
        logger.log(f"load {args.eval_checkpoint}, epoch: {checkpoint_dict['epoch']}, best: {checkpoint_dict['best']}")
    model.cuda()
    model.eval()
    logger.log(f'params: {sum(p.numel() for p in model.parameters()) / 1000 / 1000:.2f}M')

    args.use_video_cache = False
    args.use_text_cache = False
    model.use_video_cache = False
    model.use_text_cache = False
    test_dataset = VideoDataset(args, 'test', 'sup')
    extract_feature_for_test_stage(args, model, logger)
    test_dataset.get_test_cache()
    if args.use_video_encoder:
        model.use_video_cache = True
        test_dataset.use_video_cache = True
        logger.log(f'load video cache {test_dataset.video_cache.shape} from {test_dataset.crash_video_cache_file}')
    if args.use_text_encoder:
        model.use_text_cache = True
        test_dataset.use_text_cache = True
        logger.log(f'load text cache {test_dataset.text_cache.shape} from {test_dataset.crash_text_cache_file}')

    # main loop
    if args.eval_all:
        # eval_setting = [(5, 1), (6, 1), (7, 1), (8, 1), (9, 1), (10, 1)]
        # eval_setting = [(5, 1), (5, 3), (5, 5)]
        eval_setting = [(5, 1), (5, 5)]
        # eval_setting = [(5, 1), (5, 2), (5, 3), (5, 4), (5, 5)]
        # eval_setting = [(5, 1)]
    else:
        eval_setting = [(args.eval_way, args.eval_shot)]
    for eval_way, eval_shot in eval_setting:
        model.eval_way = eval_way
        model.eval_shot = eval_shot
        test_sampler = CategoriesSampler(
            label=test_dataset.videos_labels_list,
            n_batch=args.num_eval_tasks,
            n_cls=eval_way,
            n_per=eval_shot + args.eval_query
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_sampler=test_sampler,
            num_workers=args.num_workers,
            pin_memory=True
        )
        acces = AverageMeter('acc:')
        record = np.zeros((args.num_eval_tasks, 2))
        start_time = time.time()
        pbar = tqdm(total=len(test_loader), unit='task')
        for i, (video, text, _) in enumerate(test_loader, 1):
            if args.amp:
                with autocast():
                    acc, _, _ = model(video, text)
            else:
                acc, _, _ = model(video, text)
            # if isinstance(acc, tuple):
            #     acc = acc[0]
            record[i - 1, 1] = acc
            acces.update(acc)
            pbar.set_description(f'[Test]  acc={acces.avg * 100:.2f}')
            pbar.update()
        pbar.close()
        va, vap = compute_confidence_interval(record[:, 1])
        logger.log(
            f'{eval_way}-way {eval_shot}-shot, {args.num_eval_tasks} tasks, '
            f'acc={va * 100:.2f}+-{vap * 100:.2f}, time={(time.time() - start_time):.2f}'
        )


def few_shot_test_2(args):
    # log
    logger = Logger(args.save_path, 'test')
    logger.log(f'checkpoint_path: {args.save_path}')
    if args.use_video_encoder:
        logger.log(f'video_encoder: {args.video_encoder}')
    if args.use_text_encoder:
        logger.log(f'text_encoder: {args.text_encoder}')
    logger.log(f'prompt: {args.prompt_type}')
    
    # model
    model = ProtoNet(args)
    if args.eval_checkpoint: 
        print(args.save_path + f'/{args.eval_checkpoint}')
        checkpoint_dict = torch.load(args.save_path + f'/{args.eval_checkpoint}')
        model.load(checkpoint_dict)
        logger.log(f"load {args.eval_checkpoint}, epoch: {checkpoint_dict['epoch']}, best: {checkpoint_dict['best']}")
    model.cuda()
    model.eval()
    logger.log(f'params: {sum(p.numel() for p in model.parameters()) / 1000 / 1000:.2f}M')

    args.use_video_cache = False
    args.use_text_cache = False
    model.use_video_cache = False
    model.use_text_cache = False
    test_dataset = VideoDataset(args, 'test', 'sup')
    extract_feature_for_test_stage(args, model, logger)
    test_dataset.get_test_cache()
    if args.use_video_encoder:
        model.use_video_cache = True
        test_dataset.use_video_cache = True
        logger.log(f'load video cache {test_dataset.video_cache.shape} from {test_dataset.crash_video_cache_file}')
    if args.use_text_encoder:
        model.use_text_cache = True
        test_dataset.use_text_cache = True
        logger.log(f'load text cache {test_dataset.text_cache.shape} from {test_dataset.crash_text_cache_file}')
    # if args.eval_checkpoint:
    #     args.use_video_cache = False
    #     args.use_text_cache = False
    #     model.use_video_cache = False
    #     model.use_text_cache = False
    #     test_dataset = VideoDataset(args, 'test', 'sup')
    #     extract_feature_for_test_stage(args, model, logger)
    #     test_dataset.get_test_cache()
    #     if args.use_video_encoder:
    #         model.use_video_cache = True
    #         test_dataset.use_video_cache = True
    #         logger.log(f'load video cache {test_dataset.video_cache.shape} from {test_dataset.crash_video_cache_file}')
    #     if args.use_text_encoder:
    #         model.use_text_cache = True
    #         test_dataset.use_text_cache = True
    #         logger.log(f'load text cache {test_dataset.text_cache.shape} from {test_dataset.crash_text_cache_file}')
    # else:
    #     test_dataset = VideoDataset(args, 'test', 'sup')
    #     if args.use_video_encoder and args.use_video_cache:
    #         logger.log(f'load video cache {test_dataset.video_cache.shape} from {test_dataset.video_cache_file}')
    #     if args.use_text_encoder and args.use_text_cache:
    #         logger.log(f'load text cache {test_dataset.text_cache.shape} from {test_dataset.text_cache_file}')

    # main loop
    if args.eval_all:
        # eval_setting = [(5, 1), (6, 1), (7, 1), (8, 1), (9, 1), (10, 1)]
        # eval_setting = [(5, 1), (5, 3), (5, 5)]
        eval_setting = [(5, 1), (5, 5)]
        # eval_setting = [(5, 1), (5, 2), (5, 3), (5, 4), (5, 5)]
        # eval_setting = [(5, 1)]
    else:
        eval_setting = [(args.eval_way, args.eval_shot)]
    for eval_way, eval_shot in eval_setting:
        model.eval_way = eval_way
        model.eval_shot = eval_shot
        test_sampler = CategoriesSampler(
            label=test_dataset.videos_labels_list,
            n_batch=args.num_eval_tasks,
            n_cls=eval_way,
            n_per=eval_shot + args.eval_query
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_sampler=test_sampler,
            num_workers=args.num_workers,
            pin_memory=True
        )
        acces = AverageMeter('acc:')
        accves = AverageMeter('acc_v:')
        acctes = AverageMeter('acc_t:')
        record = np.zeros((args.num_eval_tasks, 2))
        start_time = time.time()
        pbar = tqdm(total=len(test_loader), unit='task')
        for i, (video, text, _) in enumerate(test_loader, 1):
            if args.amp:
                with autocast():
                    acc, acc_v, acc_t = model(video, text)
            else:
                acc, acc_v, acc_t = model(video, text)
            # if isinstance(acc, tuple):
            #     acc = acc[0]
            record[i - 1, 1] = acc
            acces.update(acc)
            accves.update(acc_v)
            acctes.update(acc_t)
            pbar.set_description(f'[Test]  acc={acces.avg * 100:.2f}, acc_v={accves.avg * 100:.2f}, acc_t={acctes.avg * 100:.2f}')
            pbar.update()
        pbar.close()
        va, vap = compute_confidence_interval(record[:, 1])
        logger.log(
            f'{eval_way}-way {eval_shot}-shot, {args.num_eval_tasks} tasks, '
            f'acc={va * 100:.2f}+-{vap * 100:.2f}, time={(time.time() - start_time):.2f}'
        )


def few_shot_test_1(args):
    # log
    logger = Logger(args.save_path, 'test')
    logger.log(f'checkpoint_path: {args.save_path}')
    if args.use_video_encoder:
        logger.log(f'video_encoder: {args.video_encoder}')
    if args.use_text_encoder:
        logger.log(f'text_encoder: {args.text_encoder}')
    logger.log(f'prompt: {args.prompt_type}')
    
    # model
    model = ProtoNet(args)
    if args.eval_checkpoint: 
        print(args.save_path + f'/{args.eval_checkpoint}')
        checkpoint_dict = torch.load(args.save_path + f'/{args.eval_checkpoint}')
        model.load(checkpoint_dict)
        logger.log(f"load {args.eval_checkpoint}, epoch: {checkpoint_dict['epoch']}, best: {checkpoint_dict['best']}")
    model.cuda()
    model.eval()
    logger.log(f'params: {sum(p.numel() for p in model.parameters()) / 1000 / 1000:.2f}M')


    args.use_video_cache = False
    args.use_text_cache = False
    model.use_video_cache = False
    model.use_text_cache = False
    test_dataset = VideoDataset(args, 'test', 'sup')
    extract_feature_for_test_stage(args, model, logger)
    test_dataset.get_test_cache()
    if args.use_video_encoder:
        model.use_video_cache = True
        test_dataset.use_video_cache = True
        logger.log(f'load video cache {test_dataset.video_cache.shape} from {test_dataset.crash_video_cache_file}')
    if args.use_text_encoder:
        model.use_text_cache = True
        test_dataset.use_text_cache = True
        logger.log(f'load text cache {test_dataset.text_cache.shape} from {test_dataset.crash_text_cache_file}')

    # if args.eval_checkpoint:
    #     args.use_video_cache = False
    #     args.use_text_cache = False
    #     model.use_video_cache = False
    #     model.use_text_cache = False
    #     test_dataset = VideoDataset(args, 'test', 'sup')
    #     extract_feature_for_test_stage(args, model, logger)
    #     test_dataset.get_test_cache()
    #     if args.use_video_encoder:
    #         model.use_video_cache = True
    #         test_dataset.use_video_cache = True
    #         logger.log(f'load video cache {test_dataset.video_cache.shape} from {test_dataset.crash_video_cache_file}')
    #     if args.use_text_encoder:
    #         model.use_text_cache = True
    #         test_dataset.use_text_cache = True
    #         logger.log(f'load text cache {test_dataset.text_cache.shape} from {test_dataset.crash_text_cache_file}')
    # else:
    #     test_dataset = VideoDataset(args, 'test', 'sup')
    #     if args.use_video_encoder and args.use_video_cache:
    #         logger.log(f'load video cache {test_dataset.video_cache.shape} from {test_dataset.video_cache_file}')
    #     if args.use_text_encoder and args.use_text_cache:
    #         logger.log(f'load text cache {test_dataset.text_cache.shape} from {test_dataset.text_cache_file}')

    # main loop
    if args.eval_all:
        # eval_setting = [(5, 1), (6, 1), (7, 1), (8, 1), (9, 1), (10, 1)]
        # eval_setting = [(5, 1), (5, 3), (5, 5)]
        eval_setting = [(5, 1), (5, 5)]
        # eval_setting = [(5, 1), (5, 2), (5, 3), (5, 4), (5, 5)]
        # eval_setting = [(5, 1)]
    else:
        eval_setting = [(args.eval_way, args.eval_shot)]
    for eval_way, eval_shot in eval_setting:
        model.eval_way = eval_way
        model.eval_shot = eval_shot
        test_sampler = CategoriesSampler(
            label=test_dataset.videos_labels_list,
            n_batch=args.num_eval_tasks,
            n_cls=eval_way,
            n_per=eval_shot + args.eval_query
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_sampler=test_sampler,
            num_workers=args.num_workers,
            pin_memory=True
        )
        acces = AverageMeter('acc:')
        record = np.zeros((args.num_eval_tasks, 2))
        start_time = time.time()
        pbar = tqdm(total=len(test_loader), unit='task')
        for i, (video, text, _) in enumerate(test_loader, 1):
            if args.amp:
                with autocast():
                    acc = model(video, text)
            else:
                acc = model(video, text)
            # if isinstance(acc, tuple):
            #     acc = acc[0]
            record[i - 1, 1] = acc
            acces.update(acc)
            pbar.set_description(f'[Test]  acc={acces.avg * 100:.2f}')
            pbar.update()
        pbar.close()
        va, vap = compute_confidence_interval(record[:, 1])
        logger.log(
            f'{eval_way}-way {eval_shot}-shot, {args.num_eval_tasks} tasks, '
            f'acc={va * 100:.2f}+-{vap * 100:.2f}, time={(time.time() - start_time):.2f}'
        )


def run(yaml_file='config.yaml', train=False):
    parser = argparse.ArgumentParser()
    with open(yaml_file, encoding='utf-8') as f:
        yaml_config = yaml.safe_load(f)
        for k, v in yaml_config.items():
            parser.add_argument(f'--{k}', default=v, type=type(v))
    args = parser.parse_args()
    args.num_workers = 8
    args.save_path = f'/home/cjx/ufsar/checkpoints/{args.dataset}/'

    # seed
    init_seeds(args.seed)
    
    model_list = []
    if args.use_video_encoder:
        model_list.append(f'Adapter{args.video_encoder}' if args.use_video_adapter else args.video_encoder)
    if args.use_text_encoder:
        model_list.append(f'Adapter{args.text_encoder}' if args.use_text_adapter else args.text_encoder)
        model_list.append(args.prompt_type)
    args.save_path = args.save_path + '_'.join(
        [
            args.train_mode,
            '_'.join(model_list),
            f'{args.batch_size}bs',
            f'{args.way}w{args.shot}s{args.query}q',
            f'{args.total_epoch}ep',
            f'{args.lr}lr',
            f'{args.frames}f',
            f'{args.vtc_tau}tau',
            f'{args.bkd_lambda}lambda',
        ]
    )
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.save_path + '/train', exist_ok=True)
    os.makedirs(args.save_path + '/test', exist_ok=True)

    if train:
        few_shot_train_mm(args)
    few_shot_test_mm(args)


def run_vtc(yaml_file='config.yaml', train=False):
    parser = argparse.ArgumentParser()
    with open(yaml_file, encoding='utf-8') as f:
        yaml_config = yaml.safe_load(f)
        for k, v in yaml_config.items():
            parser.add_argument(f'--{k}', default=v, type=type(v))
    args = parser.parse_args()
    args.num_workers = 8
    init_seeds(args.seed)
    vtc_list = [0.1, 0.3, 0.5, 0.8, 1, 1.2, 1.5, 1.8, 2.0]
    for vtc in vtc_list:
        args.save_path = f'/home/cjx/ufsar/checkpoints/{args.dataset}/'
        args.vtc_lambda = vtc
        model_list = []
        if args.use_video_encoder:
            model_list.append(f'Adapter{args.video_encoder}' if args.use_video_adapter else args.video_encoder)
        if args.use_text_encoder:
            model_list.append(f'Adapter{args.text_encoder}' if args.use_text_adapter else args.text_encoder)
            model_list.append(args.prompt_type)
        args.save_path = args.save_path + '_'.join(
            [
                args.train_mode,
                '_'.join(model_list),
                f'{args.batch_size}bs',
                f'{args.way}w{args.shot}s',
                f'{args.total_epoch}ep',
                f'{args.lr}lr',
                f'{args.frames}f',
                f'{args.vtc_lambda}vtc',
                f'{args.kl_lambda}kl',
            ]
        )
        os.makedirs(args.save_path, exist_ok=True)
        os.makedirs(args.save_path + '/train', exist_ok=True)
        os.makedirs(args.save_path + '/test', exist_ok=True)

        if train:
            few_shot_train(args)
        
        if args.use_video_encoder and args.use_text_encoder:
            few_shot_test_2(args)
        else:
            few_shot_test_1(args)


def run_kl(yaml_file='config.yaml', train=False):
    parser = argparse.ArgumentParser()
    with open(yaml_file, encoding='utf-8') as f:
        yaml_config = yaml.safe_load(f)
        for k, v in yaml_config.items():
            parser.add_argument(f'--{k}', default=v, type=type(v))
    args = parser.parse_args()
    args.num_workers = 8
    init_seeds(args.seed)
    kl_list = [0.1, 0.3, 0.5, 0.8, 1, 1.2, 1.5, 1.8, 2.0]
    for kl in kl_list:
        args.save_path = f'/home/cjx/ufsar/checkpoints/{args.dataset}/'
        args.kl_lambda = kl
        model_list = []
        if args.use_video_encoder:
            model_list.append(f'Adapter{args.video_encoder}' if args.use_video_adapter else args.video_encoder)
        if args.use_text_encoder:
            model_list.append(f'Adapter{args.text_encoder}' if args.use_text_adapter else args.text_encoder)
            model_list.append(args.prompt_type)
        args.save_path = args.save_path + '_'.join(
            [
                args.train_mode,
                '_'.join(model_list),
                f'{args.batch_size}bs',
                f'{args.way}w{args.shot}s',
                f'{args.total_epoch}ep',
                f'{args.lr}lr',
                f'{args.frames}f',
                f'{args.vtc_lambda}vtc',
                f'{args.kl_lambda}kl',
            ]
        )
        os.makedirs(args.save_path, exist_ok=True)
        os.makedirs(args.save_path + '/train', exist_ok=True)
        os.makedirs(args.save_path + '/test', exist_ok=True)

        if train:
            few_shot_train(args)
        
        if args.use_video_encoder and args.use_text_encoder:
            few_shot_test_2(args)
        else:
            few_shot_test_1(args)


def run_5x5(yaml_file='config.yaml', seed=42):
    parser = argparse.ArgumentParser()
    with open(yaml_file, encoding='utf-8') as f:
        yaml_config = yaml.safe_load(f)
        for k, v in yaml_config.items():
            parser.add_argument(f'--{k}', default=v, type=type(v))
    args = parser.parse_args()
    args.save_path = f'/home/cjx/ufsar/checkpoints/{args.dataset}/' + '5x5'
    init_seeds(seed)
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.save_path + '/train', exist_ok=True)
    os.makedirs(args.save_path + '/test', exist_ok=True)

    # 数据集，抽取数据，打印每个类别
    args.use_video_cache = False
    args.use_text_cache = False
    test_dataset = VideoDataset(args, 'test', 'sup')
    test_sampler = CategoriesSampler(
        label=test_dataset.videos_labels_list,
        n_batch=1,
        n_cls=5,
        n_per=1 + 1
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_sampler=test_sampler,
        num_workers=8,
        pin_memory=True
    )
    # class_list = ['fencing', 'kick', 'kick_ball', 'pick', 'pour', 'pushup', 'run', 'sit', 'smoke', 'talk']
    # class_list = [
    #     'blasting sand', 'busking', 'cutting watermelon', 'dancing ballet', 'dancing charleston', 
    #     'dancing macarena', 'diving cliff', 'filling eyebrows', 'folding paper', 'hula hooping', 'hurling (sport)', 
    #     'ice skating', 'paragliding', 'playing drums', 'playing monopoly', 'playing trumpet', 'pushing car', 
    #     'riding elephant', 'shearing sheep', 'side kick', 'stretching arm', 'tap dancing', 'throwing axe', 'unboxing'
    # ]
    for video, text, label in test_loader:
        # label_list = [class_list[i] for i in label]
        # print(label_list[:5])
        with torch.no_grad():
            # CLIP-Freeze
            args.use_video_encoder = True
            args.use_text_encoder = False
            args.use_video_adapter = False
            args.use_text_adapter = False
            model1 = ProtoNet(args)
            model1.cuda()
            model1.eval()
            logits_1, acc_1 = model1.single_forward_5x5(video)
            print(logits_1)
            print(acc_1)
            del model1

            # Ours
            args.use_video_encoder = True
            args.use_text_encoder = True
            args.use_video_adapter = True
            args.use_text_adapter = True
            model2 = ProtoNet(args)
            checkpoint_dict = torch.load('/home/cjx/ufsar/checkpoints/hmdb51/Cluster_AdapterClipRN50_AdapterTextRN50_git_16bs_5w1s_10ep_0.0001lr_8f_1vtc_1kl/best.pth')
            # checkpoint_dict = torch.load('/home/cjx/ufsar/checkpoints/hmdb51/Cluster_AdapterClipRN50_AdapterTextRN50_git_16bs_5w1s_10ep_0.0001lr_1.2vtc_1kl/best.pth')
            model2.load(checkpoint_dict)
            model2.cuda()
            model2.eval()
            logits_2, acc_2 = model2.video_text_forward_5x5(video, text)
            print(logits_2)
            print(acc_2)
            del model2

            return logits_1.cpu().numpy(), logits_2.cpu().numpy(), acc_1, acc_2


def run_frames(yaml_file='config.yaml', train=False):
    parser = argparse.ArgumentParser()
    with open(yaml_file, encoding='utf-8') as f:
        yaml_config = yaml.safe_load(f)
        for k, v in yaml_config.items():
            parser.add_argument(f'--{k}', default=v, type=type(v))
    args = parser.parse_args()
    args.num_workers = 8
    init_seeds(args.seed)
    frame_list = [2, 4, 6, 8, 10, 12, 14, 16]
    for frame in frame_list:
        args.save_path = f'/home/cjx/ufsar/checkpoints/{args.dataset}/'
        args.frames = frame 
        model_list = []
        if args.use_video_encoder:
            model_list.append(f'Adapter{args.video_encoder}' if args.use_video_adapter else args.video_encoder)
        if args.use_text_encoder:
            model_list.append(f'Adapter{args.text_encoder}' if args.use_text_adapter else args.text_encoder)
            model_list.append(args.prompt_type)
        args.save_path = args.save_path + '_'.join(
            [
                args.train_mode,
                '_'.join(model_list),
                f'{args.batch_size}bs',
                f'{args.way}w{args.shot}s',
                f'{args.total_epoch}ep',
                f'{args.lr}lr',
                f'{args.frames}f',
                f'{args.vtc_lambda}vtc',
                f'{args.kl_lambda}kl',
            ]
        )
        os.makedirs(args.save_path, exist_ok=True)
        os.makedirs(args.save_path + '/train', exist_ok=True)
        os.makedirs(args.save_path + '/test', exist_ok=True)
        if train:
            few_shot_train(args)
        
        if args.use_video_encoder and args.use_text_encoder:
            few_shot_test_2(args)
        else:
            few_shot_test_1(args)
