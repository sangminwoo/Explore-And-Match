import numpy as np
from collections import OrderedDict, defaultdict
import json
import time
import copy
import multiprocessing as mp
from functools import partial
from lib.evaluate.utils import compute_average_precision_detection, \
    compute_temporal_iou_batch_cross, compute_temporal_iou_batch_paired


def compute_average_precision_detection_wrapper(
        input_triple, tiou_thresholds=np.linspace(0.05, 0.95, 10)):
    query, ground_truth, prediction = input_triple
    scores = compute_average_precision_detection(
        ground_truth, prediction, tiou_thresholds=tiou_thresholds)
    return query, scores


def compute_ap(results, ground_truth, iou_thds=np.linspace(0.05, 0.95, 10),
               num_workers=8, chunksize=50):
    iou_thds = [float(f"{e:.2f}") for e in iou_thds]
    pred_query2data = defaultdict(list)
    for d in results:
        pred_spans = d["pred_timespan"]
        query = d["query"]
        for w in pred_spans:
            pred_query2data[query].append({
                "video-id": d["query"],  # in order to use the API
                "t-start": w[0],
                "t-end": w[1],
                "score": w[2]
            })

    gt_query2data = defaultdict(list)
    for d in ground_truth:
        gt_spans = d["gt_timespan"]
        query = d["query"]
        for w in gt_spans:
            gt_query2data[query].append({
                "video-id": d["query"],
                "t-start": w[0],
                "t-end": w[1]
            })

    query2ap_list = {}
    data_triples = [[query, gt_query2data[query], pred_query2data[query]] for query in pred_query2data]
    compute_ap_from_triple = partial(
        compute_average_precision_detection_wrapper, tiou_thresholds=iou_thds)

    if num_workers > 1:
        with mp.Pool(num_workers) as pool:
            for query, scores in pool.imap_unordered(compute_ap_from_triple, data_triples, chunksize=chunksize):
                query2ap_list[query] = scores
    else:
        for data_triple in data_triples:
            query, scores = compute_ap_from_triple(data_triple)
            query2ap_list[query] = scores

    # print(f"compute_average_precision_detection {time.time() - start_time:.2f} seconds.")
    ap_array = np.array(list(query2ap_list.values()))  # (#queries, #thd)
    ap_thds = ap_array.mean(0)  # mAP at different IoU thresholds.

    iou_thd2ap = dict(zip([str(e) for e in iou_thds], ap_thds))
    iou_thd2ap["average"] = np.mean(ap_thds)
    # formatting
    iou_thd2ap = {k: float(f"{100 * v:.2f}") for k, v in iou_thd2ap.items()}
    return iou_thd2ap


def compute_recall_at_k(results, ground_truth, iou_thds=np.linspace(0.1, 0.9, 9), k=1):
    # if predicted timespan has IoU >= iou_thd with GT timespan, we define it positive
    gt_query2span = {gt["query"]:gt["gt_timespan"] for gt in ground_truth}
    pred_query2span = {pred["query"]:pred["pred_timespan"][:k] for pred in results}

    max_ious = []
    for gt_query in gt_query2span:
        if gt_query in pred_query2span:
            iou, _ = compute_temporal_iou_batch_cross(
                np.array(gt_query2span[gt_query]),
                np.array(pred_query2span[gt_query])
            )
            max_iou = iou.max()
            max_ious.append(max_iou)

    max_ious = np.asarray(max_ious)
    iou_thd2recall_at_k = {}
    iou_thds = [float(f"{e:.2f}") for e in iou_thds]
    for thd in iou_thds:
        iou_thd2recall_at_k[str(thd)] = \
        float(f"{np.mean(max_ious >= thd) * 100:.2f}")
    miou = np.mean(max_ious)
    return iou_thd2recall_at_k, np.round(miou, 4)


def get_timespan_len(timespan, duration):
    return (timespan[1] - timespan[0]) * 100 / duration


def get_data_by_range(results, ground_truth, len_range):
    """ keep queries with ground truth timespan length in the specified length range.
    Args:
        results:
        ground_truth:
        len_range: [min_l (int), max_l (int)]. the range is (min_l, max_l], i.e., min_l < l <= max_l
    """
    min_l, max_l = len_range
    if min_l == 0 and max_l == 100:  # min and max l in dataset
        return results, ground_truth

    # only keep ground truth with timespans in the specified length range
    # if multiple GT timespans exists, we only keep the ones in the range
    ground_truth_in_range = []
    gt_queries_in_range = set()
    for gt in ground_truth:
        gt_timespans_in_range = [
            timespan for timespan in gt["gt_timespan"]
            if min_l < get_timespan_len(timespan, gt['duration']) <= max_l
        ]
        if len(gt_timespans_in_range) > 0:
            gt = copy.deepcopy(gt)
            gt["gt_timespan"] = gt_timespans_in_range
            ground_truth_in_range.append(gt)
            gt_queries_in_range.add(gt["query"])

    # keep only results for ground_truth_in_range
    results_in_range = []
    for result in results:
        if result["query"] in gt_queries_in_range:
            results_in_range.append(copy.deepcopy(result))

    return results_in_range, ground_truth_in_range


def eval_video_grounding(results, ground_truth, verbose=True):
    length_ranges = [[0, 100], [0, 30], [30, 70], [70, 100]]  # proportion
    range_names = ["full", "short", "middle", "long"]

    ret_metrics = {}
    for l_range, name in zip(length_ranges, range_names):
        if verbose:
            start_time = time.time()
        _results, _ground_truth = get_data_by_range(results, ground_truth, l_range)
        iou_thd2average_precision = compute_ap(_results, _ground_truth, num_workers=8, chunksize=50)
        iou_thd2recall_at_one, miou_at_one = compute_recall_at_k(_results, _ground_truth, k=1)
        iou_thd2recall_at_five, miou_at_five = compute_recall_at_k(_results, _ground_truth, k=5)
        ret_metrics[name] = {
            "VG-mAP": iou_thd2average_precision,
            "VG-R1": iou_thd2recall_at_one,
            "VG-R5": iou_thd2recall_at_five,
            "mIoU@R1": miou_at_one,
            "mIoU@R5": miou_at_five
        }
        if verbose:
            print(f"[eval_video_grounding] [{name}] {time.time() - start_time:.2f} seconds")
    return ret_metrics


def eval_results(results, ground_truth, verbose=True, match_number=False):
    """
    Args:
        results: list(dict), each dict is {
            video_id: str,
            query: str,
            pred_timespan: list([st, ed])
        }
        ground_truth: list(dict), each dict is {
          "video_id": "RoripwjYFp8_360.0_510.0",
          "query": "Man in gray top walks from outside to inside.",
          "gt_timespan": list([st, ed]),
          "duration": 150,
        }
    """
    pred_queries = set([e["query"] for e in results])
    gt_queries = set([e["query"] for e in ground_truth])
    if match_number:
        assert pred_queries == gt_queries, \
            f"queries in ground_truth and results must match. " \
            f"use `match_number=False` if you wish to disable this check"
    else:  # only leave the items that exists in both results and ground_truth
        shared_queries = pred_queries.intersection(gt_queries)
        results = [e for e in results if e["query"] in shared_queries]
        ground_truth = [e for e in ground_truth if e["query"] in shared_queries]

    eval_metrics = {}
    eval_metrics_brief = OrderedDict()
    if "pred_timespan" in results[0]:
        vg_scores = eval_video_grounding(
            results, ground_truth, verbose=verbose)
        eval_metrics.update(vg_scores)
        vg_scores_brief = {
            # mAP with IoU 0.5/0.75 
            "VG-full-mAP": vg_scores["full"]["VG-mAP"]["average"],
            # recall@1 short/middle/long
            "VG-short-R1@0.5": vg_scores["short"]["VG-R1"]["0.5"],
            "VG-middle-R1@0.5": vg_scores["middle"]["VG-R1"]["0.5"],
            "VG-long-R1@0.5": vg_scores["long"]["VG-R1"]["0.5"],
            # recall@5 short/middle/long
            "VG-short-R5@0.5": vg_scores["short"]["VG-R5"]["0.5"],
            "VG-middle-R5@0.5": vg_scores["middle"]["VG-R5"]["0.5"],
            "VG-long-R5@0.5": vg_scores["long"]["VG-R5"]["0.5"],
            # mAP@short/middle/long
            "VG-short-mAP": vg_scores["short"]["VG-mAP"]["average"],
            "VG-middle-mAP": vg_scores["middle"]["VG-mAP"]["average"],
            "VG-long-mAP": vg_scores["long"]["VG-mAP"]["average"],
            # recall@1 with IoU 0.1/0.3/0.5/0.7
            "VG-full-R1@0.1": vg_scores["full"]["VG-R1"]["0.1"],
            "VG-full-R1@0.3": vg_scores["full"]["VG-R1"]["0.3"],
            "VG-full-R1@0.5": vg_scores["full"]["VG-R1"]["0.5"],
            "VG-full-R1@0.7": vg_scores["full"]["VG-R1"]["0.7"],
            # recall@5 with IoU 0.1/0.3/0.5/0.7
            "VG-full-R5@0.1": vg_scores["full"]["VG-R5"]["0.1"],
            "VG-full-R5@0.3": vg_scores["full"]["VG-R5"]["0.3"],
            "VG-full-R5@0.5": vg_scores["full"]["VG-R5"]["0.5"],
            "VG-full-R5@0.7": vg_scores["full"]["VG-R5"]["0.7"],
            # mIoU
            "VG-full-mIoU@R1": vg_scores["full"]["mIoU@R1"],
            "VG-full-mIoU@R5": vg_scores["full"]["mIoU@R5"]
        }
        eval_metrics_brief.update(
            sorted([(k, v) for k, v in vg_scores_brief.items()], key=lambda x: x[0]))

    # sort by keys
    final_eval_metrics = OrderedDict()
    final_eval_metrics["brief"] = eval_metrics_brief
    final_eval_metrics.update(sorted([(k, v) for k, v in eval_metrics.items()], key=lambda x: x[0]))
    return final_eval_metrics