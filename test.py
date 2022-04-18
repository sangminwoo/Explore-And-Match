import os
import time
import pprint
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import more_itertools as mit
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict

from lib.modeling.model import build_model
from lib.modeling.loss import build_loss
from lib.modeling.bipartite_matcher import build_bipartite_matcher
from lib.evaluate.eval import eval_results
from lib.dataset.vidgr_dataset import build_dataset, collate_fn_feat, collate_fn_raw, prepare_batch_inputs
from lib.utils.misc import cur_time, save_jsonl, save_json, AverageMeter
from lib.utils.span_utils import span_cw_to_xx
from lib.utils.temporal_nms import temporal_nms
from lib.utils.logger import setup_logger
from lib.configs import args


def post_processing_vg_nms(vg_res, nms_thd, max_before_nms, max_after_nms):
    vg_res_after_nms = []
    for e in vg_res:
        e["pred_timespan"] = temporal_nms(
            e["pred_timespan"][:max_before_nms],
            nms_thd=nms_thd,
            max_after_nms=max_after_nms
        )
        vg_res_after_nms.append(e)
    return vg_res_after_nms


def eval_epoch_post_processing(args, results, ground_truth, results_filename, logger):
    logger.info('Saving/Evaluating no nms results')
    results_path = os.path.join(args.results_dir, results_filename)
    save_jsonl(results, results_path)

    metrics = None
    latest_file_paths = [results_path, ]
    if args.phase in ['val', 'test']:
        metrics = eval_results(
            results,
            ground_truth,
            verbose=args.debug
        )
        save_metrics_path = results_path.replace('.jsonl', '_metrics.json')
        save_json(metrics, save_metrics_path, save_pretty=True, sort_keys=False)
        latest_file_paths = [results_path, save_metrics_path]

    metrics_nms = None
    if args.nms_thd != -1:
        logger.info(f'[VG] Performing nms with nms_thd {args.nms_thd}')
        results_after_nms = post_processing_vg_nms(
            results,
            nms_thd=args.nms_thd,
            max_before_nms=args.max_before_nms,
            max_after_nms=args.max_after_nms
        )

        logger.info('Saving/Evaluating nms results')
        results_nms_path = results_path.replace('.jsonl', f'_nms_thd_{args.nms_thd}.jsonl')
        save_jsonl(results_after_nms, results_nms_path)
        latest_file_paths = [results_nms_path, ]
        if args.phase in ['val', 'test']:
            metrics_nms = eval_results(
                results_after_nms,
                ground_truth,
                verbose=args.debug
            )
            save_metrics_nms_path = results_nms_path.replace('.jsonl', '_metrics.json')
            save_json(metrics_nms, save_metrics_nms_path, save_pretty=True, sort_keys=False)
            latest_file_paths += [results_nms_path, save_metrics_nms_path]
        
    return metrics, metrics_nms, latest_file_paths


@torch.no_grad()
def get_eval_res(model, eval_loader, criterion, dist_visualize=False):
    '''compute and save query and video proposal embeddings'''
    model.eval()
    criterion.eval()

    time_meters = defaultdict(AverageMeter)
    loss_meters = defaultdict(AverageMeter)
    vg_res = []
    test_all_samples = True

    if dist_visualize:
        x = torch.empty(size=(args.eval_bs*len(eval_loader), args.num_proposals))
        y = torch.empty(size=(args.eval_bs*len(eval_loader), args.num_proposals))
        durations = []

    for b_idx, batch in tqdm(enumerate(eval_loader),
                             desc='Evaluation',
                             total=len(eval_loader)):
        annotations = batch[0]
        device = 'cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu'
        model_inputs, targets = prepare_batch_inputs(batch[1], device, non_blocking=args.pin_memory)
        
        if test_all_samples:
            src_txt = model_inputs['src_txt']
            src_txt_mask = model_inputs['src_txt_mask']
            src_vid = model_inputs['src_vid']
            src_vid_mask = model_inputs['src_vid_mask']
            split_src_txt = src_txt.split(args.num_input_sentences, dim=1) # (bs, #query, #pred)
            split_src_txt_mask = src_txt_mask.split(args.num_input_sentences, dim=1) # (bs, #query, #pred)
            split_targets = [
                {'spans': split} \
                for target_spans in targets['target_spans'] \
                for split in target_spans['spans'].split(args.num_input_sentences)
            ]

            for idx, (annos, src_txt, src_txt_mask, tgt) in enumerate(zip(annotations,
                                                                          split_src_txt,
                                                                          split_src_txt_mask,
                                                                          split_targets)):
                tictoc = time.time()
                outputs = model(
                    src_txt=src_txt,
                    src_txt_mask=src_txt_mask,
                    src_vid=src_vid,
                    src_vid_mask=src_vid_mask,
                    att_visualize=args.att_visualize,
                    corr_visualize=args.corr_visualize,
                    epoch_i=b_idx,
                    idx=idx
                )
                time_meters['model_forward_time'].update(time.time() - tictoc)

                targets = {}
                targets['target_spans'] = [tgt]
                loss_dict = criterion(outputs, targets)
                weight_dict = criterion.weight_dict
                losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
                loss_dict['loss_overall'] = float(losses)  # for logging only
                for k, v in loss_dict.items():
                    loss_meters[k].update(float(v) * weight_dict[k] if k in weight_dict else float(v))

                timespans = outputs['pred_spans']  # (batch_size, #queries, 2)
                label_prob = F.softmax(outputs['pred_logits'], -1)  # (batch_size, #queries, #classes)
                scores, labels = label_prob.max(-1)  # (batch_size, #queries)

                if dist_visualize:
                    x[args.eval_bs*b_idx+idx] = timespans[:, :, 0] # <- [#queries]
                    y[args.eval_bs*b_idx+idx] = timespans[:, :, 1] # <- [#queries]
                    durations.append(annos['duration'])

                # compose predictions
                for span, score, label in zip(timespans.cpu(),
                                              scores.cpu(),
                                              labels.cpu()):
                    if args.span_type == 'cw':
                        duration = annos['duration'] if 'duration' in annos else annos['num_frames']
                        spans = torch.clamp(span_cw_to_xx(span), min=0, max=1) * duration

                    # (#queries, 4), [label(int), start(float), end(float), score(float)]
                    sorted_preds = torch.cat([label[:, None], spans, score[:, None]], dim=1).tolist()
                    if not args.no_sort_results:
                        sorted_preds = sorted(sorted_preds, key=lambda x: x[3], reverse=True)

                    sorted_preds = torch.tensor(sorted_preds)
                    sorted_labels = sorted_preds[:, 0].int().tolist()
                    sorted_spans = sorted_preds[:, 1:].tolist()
                    sorted_spans = [[float(f'{e:.4f}') for e in row] for row in sorted_spans]

                    for idx, query in enumerate(annos['sentences']):
                        pred_spans = [pred_span for pred_label, pred_span in zip(sorted_labels, sorted_spans) if pred_label == idx]
                        if len(pred_spans) == 0:
                            continue
                        cur_query_pred = dict(
                            video_id=annos['video_id'],
                            query=query,
                            pred_timespan=pred_spans,
                        )
                        vg_res.append(cur_query_pred)

                if args.debug:
                    break

        else:
            tictoc = time.time()
            outputs = model(**model_inputs)
            time_meters['model_forward_time'].update(time.time() - tictoc)

            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            loss_dict['loss_overall'] = float(losses)  # for logging only
            for k, v in loss_dict.items():
                loss_meters[k].update(float(v) * weight_dict[k] if k in weight_dict else float(v))

            timespans = outputs['pred_spans']  # (batch_size, #queries, 2)
            label_prob = F.softmax(outputs['pred_logits'], -1)  # (batch_size, #queries, #classes)
            scores, labels = label_prob.max(-1)  # (batch_size, #queries)

            # compose predictions
            for annos, span, score, label in zip(annotations,
                                                 timespans.cpu(),
                                                 scores.cpu(),
                                                 labels.cpu()):
                if args.span_type == 'cw':
                    duration = annos['duration'] if 'duration' in annos else annos['num_frames']
                    spans = torch.clamp(span_cw_to_xx(span), min=0, max=1) * duration

                # (#queries, 4), [label(int), start(float), end(float), score(float)]
                sorted_preds = torch.cat([label[:, None], spans, score[:, None]], dim=1).tolist()
                if not args.no_sort_results:
                    sorted_preds = sorted(sorted_preds, key=lambda x: x[3], reverse=True)

                sorted_preds = torch.tensor(sorted_preds)
                sorted_labels = sorted_preds[:, 0].int().tolist()
                sorted_spans = sorted_preds[:, 1:].tolist()
                sorted_spans = [[float(f'{e:.4f}') for e in row] for row in sorted_spans]

                for idx, query in enumerate(annos['sentences']):
                    pred_spans = [pred_span for pred_label, pred_span in zip(sorted_labels, sorted_spans) if pred_label == idx]
                    if len(pred_spans) == 0:
                        continue
                    cur_query_pred = dict(
                        video_id=annos['video_id'],
                        query=query,
                        pred_timespan=pred_spans,
                    )
                    vg_res.append(cur_query_pred)

            if args.debug:
                break

    if dist_visualize:
        col = 5
        row = args.num_proposals // col
        fig, _ = plt.subplots(row, col, figsize=(col*3, row*3))

        x = x.transpose(0, 1) # [#queries, #samples]
        y = y.transpose(0, 1) # [#queries, #samples]
        marker_size = [10] * x.shape[1] # #queries
        for i, (x_, y_) in enumerate(zip(x, y)):
            marker_color = np.log(y_)
            plt.subplot(row, col, i+1)
            plt.scatter(x_, y_, s=marker_size, c=marker_color,
                        cmap='GnBu', marker='o', alpha=0.5)
            plt.tick_params(
                top=False,
                bottom=False,
                left=False,
                right=False,
                labelleft=False,
                labelbottom=False
            )

        plt.tight_layout()
        plt.savefig('pred_dist.png')

    # logger.info(
    #     "Training Logs\n"
    #     "[Time]\n{time_stats}\n".format(
    #         time_str=time.strftime("%Y-%m-%d %H:%M:%S"),
    #         time_stats="\n".join("\t> {} {:.4f}".format(k, v.avg) for k, v in time_meters.items()),
    #     )
    # )

    return vg_res, loss_meters


def eval_epoch(model, eval_loader, results_filename, criterion, logger=None):
    model.eval()
    criterion.eval()

    results, loss_meters = get_eval_res(model, eval_loader, criterion, args.dist_visualize)
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    if args.no_sort_results:
        results_filename = results_filename.replace(".jsonl", "_unsorted.jsonl")

    pred_vids = [e["video_id"] for e in results]
    ground_truth = eval_loader.dataset.get_gt_with_vids(pred_vids)
    metrics_no_nms, metrics_nms, latest_file_paths = eval_epoch_post_processing(
        args, results, ground_truth, results_filename, logger)
    return metrics_no_nms, metrics_nms, loss_meters, latest_file_paths


def eval_setup(logger):
    if torch.cuda.is_available() and args.use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device
        use_cuda = True
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # if torch.cuda.is_available() and args.device.type == 'cuda':
    #     model.to('cuda')
    #     criterion.to('cuda')

    cudnn.benchmark = True
    cudnn.deterministic = False

    model = build_model(args)
    criterion = build_loss(args)
    model.to(device)
    criterion.to(device)

    param_dicts = [{'params': [param for name, param in model.named_parameters() if param.requires_grad]}]

    # optimizer
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9, weight_decay=args.wd)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(param_dicts, lr=args.lr, weight_decay=args.wd)
    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.wd)
    
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
        state_dict = checkpoint['model']

        if 'module' in list(state_dict.keys())[0]:
            keys = state_dict.keys()
            values = state_dict.values()
            new_keys = []
            for key in keys:
                new_key = key[7:]    # remove the 'module.'
                new_keys.append(new_key)

            from collections import OrderedDict
            new_dict = OrderedDict(list(zip(new_keys, values)))
            model.load_state_dict(new_dict)
        else:
            model.load_state_dict(checkpoint['model'])

        if args.resume_all:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
        logger.info(f'Loaded model saved at epoch {checkpoint["epoch"]} from checkpoint: {args.resume}')
    else:
        logger.warning('If you intend to evaluate the model, please specify --resume with ckpt path')

    return model, criterion, optimizer, lr_scheduler


def eval_zero_shot(logger, run=None, visualize=False):
    model, _, _, _ = eval_setup(logger)

    args.phase = 'test'
    test_dataset = build_dataset(args)
    test_loader = DataLoader(
        test_dataset,
        collate_fn=collate_fn_feat if 'features' in args.data_type else collate_fn_raw,
        batch_size=args.eval_bs,
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )
   
    logger.info("Start zero-shot evaluation...")
    results = []
    for batch in tqdm(test_loader, desc='Zero-shot Evaluation'):
        annotations = batch[0]
        device = 'cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu'
        model_inputs, _ = prepare_batch_inputs(batch[1], device, non_blocking=args.pin_memory)
        similarity = model(**model_inputs).cpu() # BxNxM

        # visualize
        if visualize:
            for i, (annos, sim_per_video) in enumerate(zip(annotations, similarity.tolist())):
                duration = annos['duration'] if 'duration' in annos else annos['num_frames']
                timestamps = annos['timestamps']

                print(annos['video_id'])
                print(annos['sentences'])
                print(annos['timestamps'])
                for (start, end), sim, sen, gt_time in zip(timestamps, sim_per_video,
                                                           annos['sentences'], annos['timestamps']):
                    gt_timespan = [0] * (args.num_input_frames+1)
                    start *= args.num_input_frames / duration
                    end *= args.num_input_frames / duration
                    for i in range(round(start), round(end)):
                        gt_timespan[i] = 1

                    sim = np.asarray(sim)
                    sim = (sim - min(sim)) / (max(sim) - min(sim))
                    sns.set()
                    plt.plot(gt_timespan, label=f'GT {str(gt_time[0])}~{str(gt_time[1])}')
                    plt.plot(sim, label='Pred')
                    plt.title(sen)
                    plt.xlabel('time')
                    plt.ylabel('prob')
                    plt.legend()
                    plt.show()

        # image-text matched if (similarity) > (threshold)
        matched_thd = 1/args.num_input_frames
        matched_idxs = (similarity > matched_thd).nonzero()
        matched_idxs = np.asarray(matched_idxs)
        match_dict = defaultdict(dict) # video_idx_in_bacth -> sentence_idx -> matched_idx
        for idx in matched_idxs:
            match_dict[idx[0]][idx[1]] = []

        for idx in matched_idxs:
            match_dict[idx[0]][idx[1]].append(idx[2])

        durations = [anno['duration'] for anno in annotations] if 'duration' in annotations[0] \
                    else [anno['num_frames'] for anno in annotations]

        timesteps = []
        for duration in durations:
            timesteps.append(np.linspace(0, duration, int(args.num_input_frames)+1))

        pred_spans = []
        for (vid_idx, matches_in_video), timestep in zip(match_dict.items(), timesteps):
            for sen_idx, matches in matches_in_video.items():
                groups = [list(group) for group in mit.consecutive_groups(matches)]
                pred_spans_per_sentence = []
                for group in groups:
                    start = group[0]
                    end = group[-1]
                    score = sum([float(similarity[vid_idx, sen_idx, match]) for match in range(start, end+1)]) / (end-start+1)
                    pred_spans_per_sentence.append([timestep[start], timestep[end+1], score])
            pred_spans.append(pred_spans_per_sentence)
        # pred_spans = np.asarray(pred_spans)

        # compose predictions
        for idx, (annos, spans) in enumerate(zip(annotations, pred_spans)):
            # pred_spans = [[float(f'{e:.4f}') for e in row] for row in pred_spans]
            cur_query_pred = dict(
                video_id=annos['video_id'],
                query=annos['sentences'],
                pred_timespan=spans,
            )
            results.append(cur_query_pred)

    pred_vids = [e["video_id"] for e in results]
    ground_truth = test_loader.dataset.get_gt_with_vids(pred_vids)

    results_filename = f'{cur_time()}_{args.dataset}_{args.backbone}_' \
                       f'{args.bs}b_{args.enc_layers}l_{args.num_input_frames}f_{args.num_proposals}q_' \
                       f'{args.pred_label}_{args.set_cost_span}_{args.set_cost_giou}_{args.set_cost_query}_zero_shot.jsonl'
    metrics_no_nms, metrics_nms, latest_file_paths = eval_epoch_post_processing(
        args, results, ground_truth, results_filename, logger)

    # test log
    if run:
        for k, v in metrics_no_nms["brief"].items():
            run[f"Zeroshot/{k}"].log(float(v))
        if metrics_nms is not None:
            for k, v in metrics_nms["brief"].items():
                run[f"Zeroshot/{k}"].log(float(v))

    logger.info(f'metrics_no_nms {pprint.pformat(metrics_no_nms["brief"], indent=4)}')
    if metrics_nms is not None:
        logger.info(f'metrics_nms {pprint.pformat(metrics_nms["brief"], indent=4)}')


def test(logger, run=None):
    model, criterion, _, _ = eval_setup(logger)

    args.phase = 'test'
    test_dataset = build_dataset(args)
    test_loader = DataLoader(
        test_dataset,
        collate_fn=collate_fn_feat if 'features' in args.data_type else collate_fn_raw,
        batch_size=args.eval_bs,
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )
    
    results_filename = f'{cur_time()}_{args.dataset}_{args.backbone}_' \
                       f'{args.bs}b_{args.enc_layers}l_{args.num_input_frames}f_{args.num_proposals}q_' \
                       f'{args.pred_label}_{args.set_cost_span}_{args.set_cost_giou}_{args.set_cost_query}_test.jsonl'
    logger.info("Start inference...")
    with torch.no_grad():
        metrics_no_nms, metrics_nms, eval_loss_meters, latest_file_paths = \
            eval_epoch(model, test_loader, results_filename, criterion, logger=logger)

    # test log
    if run:
        for k, v in eval_loss_meters.items():
            run[f"Test/{k}"].log(v.avg) 

        for k, v in metrics_no_nms["brief"].items():
            run[f"Test/{k}"].log(float(v))

        if metrics_nms is not None:
            for k, v in metrics_nms["brief"].items():
                run[f"Test/{k}"].log(float(v))

    logger.info(f'metrics_no_nms {pprint.pformat(metrics_no_nms["brief"], indent=4)}')
    if metrics_nms is not None:
        logger.info(f'metrics_nms {pprint.pformat(metrics_nms["brief"], indent=4)}')


if __name__ == '__main__':
    logger = setup_logger('LVTR_eval', args.log_dir, distributed_rank=0, filename=cur_time()+"_eval.txt")
    if args.zero_shot:
        eval_zero_shot(logger, run=run, visualize=False)
    else:
        test(logger, run=run)