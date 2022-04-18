import os
import time
import json
import random
import pprint
import numpy as np
from tqdm import tqdm, trange
from collections import defaultdict

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from lib.modeling.model import build_model
from lib.modeling.loss import build_loss
from lib.dataset.vidgr_dataset import build_dataset, collate_fn_feat, collate_fn_raw, prepare_batch_inputs
from lib.utils.misc import cur_time, AverageMeter
from lib.utils.model_utils import count_parameters
from lib.utils.logger import setup_logger
from lib.configs import args
from test import eval_epoch, test


def set_seed(seed, use_cuda=True):
    # fix seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)


def train_setup(logger):
    if torch.cuda.is_available() and args.use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device
        use_cuda = True
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if args.seed:
        set_seed(args.seed, use_cuda)

    if args.debug: # keep the model run deterministically
        # 'cudnn.benchmark = True' enabled auto finding the best algorithm for a specific input/net config.
        # Enable this only when input size is fixed.
        cudnn.benchmark = False
        cudnn.deterministic = True

    model = build_model(args)
    criterion = build_loss(args)
    model.to(device)
    criterion.to(device)

    # param_dicts = [{'params': [param for name, param in model.named_parameters() if param.requires_grad]}]
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'backbone' in name and args.data_type == 'raw':
                # param.requires_grad = False
                backbone_params.append(param)
            if 'head' in name:
                head_params.append(param)

    if len(backbone_params) > 0:
        param_dicts = [{'params':backbone_params}, {'params':head_params}]
    else:
        param_dicts = [{'params':head_params}]

    # optimizer
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9, weight_decay=args.wd)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(param_dicts, lr=args.lr, weight_decay=args.wd)
    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.wd)
    
    if len(backbone_params) > 0:
        optimizer.param_groups[0]['lr'] = args.lr/(10**1)
        optimizer.param_groups[1]['lr'] = args.lr

    # scheduler
    if args.scheduler == 'steplr':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_drop_step)
    if args.scheduler == 'multisteplr':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_drop_step)
    if args.scheduler == 'reducelronplateau':
        # TODO
        lr_scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.1,
            patience=1,
            threshold=0.5,
            verbose=True
        )

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        if args.resume_all:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
        logger.info(f'Loaded model saved at epoch {checkpoint["epoch"]} from checkpoint: {args.resume}')

    return model, criterion, optimizer, lr_scheduler


def train_epoch(model, dataloader, criterion, optimizer, epoch_i):
    model.train()
    criterion.train()

    time_meters = defaultdict(AverageMeter)
    loss_meters = defaultdict(AverageMeter)

    tictoc = time.time()
    for idx, batch in tqdm(enumerate(dataloader),
                           desc='Training Iteration',
                           total=len(dataloader)):
        time_meters['dataloading_time'].update(time.time() - tictoc)

        tictoc = time.time()
        device = 'cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu'
        model_inputs, targets = prepare_batch_inputs(batch[1], device, non_blocking=args.pin_memory)
        # inputs: src_txt [batch_size, num_input_sentences, dim]
        #         src_txt_mask [batch_size, num_input_sentences]
        #         src_vid [batch_size, num_input_frames, dim]
        #         src_vid_mask [batch_size, num_input_frames]
        # targets: target_spans [batch_size]
        time_meters['prepare_inputs_time'].update(time.time() - tictoc)

        # predict video-wisely
        # if args.num_input_sentences == 1:
        #     optimizer.zero_grad()
        #     model_inputs['src_txt'] = model_inputs['src_txt'].permute(1,0,2)
        #     model_inputs['src_txt_mask'] = model_inputs['src_txt_mask'].permute(1,0)
        #     for src_txt, src_txt_mask, tgt in zip(model_inputs['src_txt'],
        #                                           model_inputs['src_txt_mask'],
        #                                           targets['target_spans']):
        #         tictoc = time.time()
        #         src_txt = src_txt.unsqueeze(1) # bxd->bx1xd
        #         src_txt_mask = src_txt_mask.unsqueeze(1) # bx1
        #         target = defaultdict(list)
        #         target['target_spans'].append(tgt)

        #         outputs = model(src_txt=src_txt,
        #                         src_txt_mask=src_txt_mask,
        #                         src_vid=model_inputs['src_vid'],
        #                         src_vid_mask=model_inputs['src_vid_mask'],
        #                         att_visualize=args.att_visualize,
        #                         corr_visualize=args.corr_visualize,
        #                         epoch_i=epoch_i,
        #                         idx=idx)
        #         # outputs: pred_logits [batch_size, num_proposals, num_classes]
        #         #          pred_spans [batch_size, num_proposals, num_classes]
        #         #          aux_outputs (pred_logits, pred_spans)
        #         loss_dict = criterion(outputs, target)
        #         weight_dict = criterion.weight_dict
        #         losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        #         time_meters['model_forward_time'].update(time.time() - tictoc)

        #         tictoc = time.time()
        #         losses.backward()
        #         time_meters['model_backward_time'].update(time.time() - tictoc)
        #     optimizer.step()

        # predict fixed-sentences per video
        # else:
        tictoc = time.time()
        outputs = model(**model_inputs,
                        att_visualize=args.att_visualize,
                        corr_visualize=args.corr_visualize,
                        epoch_i=epoch_i,
                        idx=idx)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        time_meters['model_forward_time'].update(time.time() - tictoc)

        tictoc = time.time()
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        time_meters['model_backward_time'].update(time.time() - tictoc)

        loss_dict['loss_overall'] = float(losses)
        for k, v in loss_dict.items():
            loss_meters[k].update(float(v) * weight_dict[k] if k in weight_dict else float(v))

        tictoc = time.time()
        if args.debug and idx == 3:
            break

    return time_meters, loss_meters


def train(logger, run=None):
    model, criterion, optimizer, lr_scheduler = train_setup(logger)
    logger.info(f'Model {model}')
    n_all, n_trainable = count_parameters(model)
    if run:
        run[f"num_params"].log(n_all)
        run[f"num_trainable_params"].log(n_trainable) 
    logger.info(f'Start Training...')

    args.phase = 'train'
    train_dataset = build_dataset(args)
    train_loader = DataLoader(
        train_dataset,
        collate_fn=collate_fn_feat if 'features' in args.data_type else collate_fn_raw,
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )
    
    # create checkpoint
    if not os.path.exists(args.checkpoint):
        os.makedirs(args.checkpoint)

    if args.start_epoch is None:
        start_epoch = -1 if args.eval_untrained else 0
    else:
        start_epoch = start_epoch

    for epoch_i in trange(start_epoch, args.end_epoch, desc='Training Epoch'):
        if start_epoch > -1:
            time_meters, loss_meters = train_epoch(model, train_loader, criterion, optimizer, epoch_i)
            
            # train log
            if run:
                for k, v in loss_meters.items():
                    run[f"Train/{k}"].log(v.avg)  

            logger.info(
                "Training Logs\n"
                "[Epoch] {epoch:03d}\n"
                "[Time]\n{time_stats}\n"
                "[Loss]\n{loss_str}\n".format(
                    time_str=time.strftime("%Y-%m-%d %H:%M:%S"),
                    epoch=epoch_i+1,
                    time_stats="\n".join("\t> {} {:.4f}".format(k, v.avg) for k, v in time_meters.items()),
                    loss_str="\n".join(["\t> {} {:.4f}".format(k, v.avg) for k, v in loss_meters.items()])
                )
            )

            lr_scheduler.step()

        if (epoch_i + 1) % args.save_interval == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch_i,
                'args': args
            }
            torch.save(
                checkpoint,
                os.path.join(
                    args.checkpoint,
                    f'{epoch_i:04d}_model_{args.dataset}_{args.backbone}_' \
                    f'{args.bs}b_{args.enc_layers}l_{args.num_input_frames}f_{args.num_proposals}q_' \
                    f'{args.pred_label}_{args.set_cost_span}_{args.set_cost_giou}_{args.set_cost_query}.ckpt'
                )
            )
        if args.debug:
            break


def train_val(logger, run=None):
    model, criterion, optimizer, lr_scheduler = train_setup(logger)
    logger.info(f'Model {model}')
    n_all, n_trainable = count_parameters(model)
    if run:
        run[f"num_params"].log(n_all)
        run[f"num_trainable_params"].log(n_trainable) 
    logger.info(f'Start Training...')

    args.phase = 'train'
    train_dataset = build_dataset(args)
    train_loader = DataLoader(
        train_dataset,
        collate_fn=collate_fn_feat if 'features' in args.data_type else collate_fn_raw,
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )

    if args.dataset in ['activitynet']:
        args.phase = 'val'
        val_dataset = build_dataset(args)
        val_loader = DataLoader(
            val_dataset,
            collate_fn=collate_fn_feat if 'features' in args.data_type else collate_fn_raw,
            batch_size=args.eval_bs,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory
        )
    else:
        args.phase = 'test'
        val_dataset = build_dataset(args)
        val_loader = DataLoader(
            val_dataset,
            collate_fn=collate_fn_feat if 'features' in args.data_type else collate_fn_raw,
            batch_size=args.eval_bs,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory
        )

    # for early stop purpose
    best_loss = np.inf
    early_stop_count = 0
    
    # create checkpoint
    if not os.path.exists(args.checkpoint):
        os.makedirs(args.checkpoint)

    if args.start_epoch is None:
        start_epoch = -1 if args.eval_untrained else 0
    else:
        start_epoch = start_epoch

    for epoch_i in trange(start_epoch, args.end_epoch, desc='Training Epoch'):
        if start_epoch > -1:
            time_meters, loss_meters = train_epoch(model, train_loader, criterion, optimizer, epoch_i)
            
            # train log
            if run:
                for k, v in loss_meters.items():
                    run[f"Train/{k}"].log(v.avg)  

            logger.info(
                "Training Logs\n"
                "[Epoch] {epoch:03d}\n"
                "[Time]\n{time_stats}\n"
                "[Loss]\n{loss_str}\n".format(
                    time_str=time.strftime("%Y-%m-%d %H:%M:%S"),
                    epoch=epoch_i+1,
                    time_stats="\n".join("\t> {} {:.4f}".format(k, v.avg) for k, v in time_meters.items()),
                    loss_str="\n".join(["\t> {} {:.4f}".format(k, v.avg) for k, v in loss_meters.items()])
                )
            )

            lr_scheduler.step()

        if (epoch_i + 1) % args.val_interval == 0:
            with torch.no_grad():
                results_filename = f'{cur_time()}_{args.dataset}_{args.backbone}_' \
                                   f'{args.bs}b_{args.enc_layers}l_{args.num_input_frames}f_{args.num_proposals}q_' \
                                   f'{args.pred_label}_{args.set_cost_span}_{args.set_cost_giou}_{args.set_cost_query}_val.jsonl'
                metrics_no_nms, metrics_nms, eval_loss_meters, latest_file_paths = \
                    eval_epoch(model, val_loader, results_filename, criterion, logger=logger)

            cur_loss = eval_loss_meters['loss_overall'].avg # TODO

            # val log
            if run:
                for k, v in eval_loss_meters.items():
                    run[f"Val/{k}"].log(v.avg) 

                for k, v in metrics_no_nms["brief"].items():
                    run[f"Val/{k}"].log(float(v))

                if metrics_nms is not None:
                    for k, v in metrics_nms["brief"].items():
                        run[f"Val/{k}"].log(float(v))

            logger.info(
                "\n>>>>> Evalutation\n"
                "[Epoch] {epoch:03d}\n"
                "[Loss]\n{loss_str}\n"
                "[Metrics_No_NMS]\n{metrics}\n".format(
                    time_str=time.strftime("%Y-%m-%d %H:%M:%S"),
                    epoch=epoch_i+1,
                    loss_str="\n".join(["\t> {} {:.4f}".format(k, v.avg) for k, v in eval_loss_meters.items()]),
                    metrics=pprint.pformat(metrics_no_nms["brief"], indent=4)
                )
            )

            if metrics_nms is not None:
                logger.info("metrics_nms {}".format(pprint.pformat(metrics_nms["brief"], indent=4)))

            # early stop
            if cur_loss < best_loss:
                early_stop_count = 0
                best_loss = cur_loss
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch_i,
                    'args': args
                }
                torch.save(
                    checkpoint,
                    os.path.join(
                        args.checkpoint,
                        f'best_model_{args.dataset}_{args.backbone}_' \
                        f'{args.bs}b_{args.enc_layers}l_{args.num_input_frames}f_{args.num_proposals}q_' \
                        f'{args.pred_label}_{args.set_cost_span}_{args.set_cost_giou}_{args.set_cost_query}.ckpt'
                    )
                )
            else:
                if args.early_stop_patience != -1:
                    early_stop_count += 1
                    if args.early_stop_patience and early_stop_count > args.early_stop_patience:
                        logger.info(f'\n>>>>> Early Stop at Epoch {epoch_i+1} (best val loss: {best_loss})\n')
                        break

        if (epoch_i + 1) % args.save_interval == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch_i,
                'args': args
            }
            torch.save(
                checkpoint,
                os.path.join(
                    args.checkpoint,
                    f'{epoch_i:04d}_model_{args.dataset}_{args.backbone}_' \
                    f'{args.bs}b_{args.enc_layers}l_{args.num_input_frames}f_{args.num_proposals}q_' \
                    f'{args.pred_label}_{args.set_cost_span}_{args.set_cost_giou}_{args.set_cost_query}.ckpt'
                )
            )
        if args.debug:
            break


if __name__ == '__main__':
    logger = setup_logger('LVTR', args.log_dir, distributed_rank=0, filename=cur_time()+"_train.txt")
    train_val(logger, run=run)