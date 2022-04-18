import torch
import torch.nn.functional as F

from torch import nn
from lib.modeling.bipartite_matcher import build_bipartite_matcher
from lib.modeling.hungarian_matcher import build_hungarian_matcher
from lib.utils.span_utils import generalized_span_iou, span_cw_to_xx
from lib.utils.model_utils import accuracy
from lib.utils.loss_utils import LabelSmoothingLoss


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, losses, pred_label, span_type):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes  # sentence_idxs
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.pred_label = pred_label
        self.span_type = span_type
        self.nll_label_smoothing = LabelSmoothingLoss(smoothing=0.1, reduction='mean', weight=None)

    def loss_labels(self, outputs, targets, indices, log=True):
        """
        Classification loss (NLL)
            targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
       
            indices: [(src_idx, tgt_idx), (src_idx, tgt_idx), (src_idx, tgt_idx), ...]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']  # (batch_size, #queries, #sentence_slots)
        
        idx = self._get_src_permutation_idx(indices)
        if self.pred_label == 'pred':
            target_classes_o = torch.cat(
                [torch.arange(len(t['spans']))[i] \
                for t, (_, i) in zip(targets['target_spans'], indices)]
            ).to(src_logits.device)

            target_classes = torch.full(
                src_logits.shape[:2],
                self.num_classes-1,
                dtype=torch.int64,
                device=src_logits.device
            )
            target_classes[idx] = target_classes_o
        else:
            num_sentences = [len(target['spans']) for target in targets['target_spans']]
            num_proposals = src_logits.shape[1]
            target_classes = []
            for num_sentence in num_sentences:
                target_class = torch.zeros(
                    num_proposals,
                    dtype=torch.int64,
                    device=src_logits.device
                )
                target_groups = target_class.chunk(num_sentence)
                for i, target_group in enumerate(target_groups):
                    target_group.fill_(i)
                target_classes.append(target_class)
            target_classes = torch.stack(target_classes)


        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2),  # (batch_size, #sentence_slots, #queries)
            target_classes,  # (batch_size, #queries)
            reduction='mean'
        )
        # loss_ce = self.nll_label_smoothing(
        #     src_logits.transpose(1, 2),  # (batch_size, #sentence_slots, #queries)
        #     target_classes,  # (batch_size, #queries)
        # )

        losses = {'loss_label': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes[idx])[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([num_tgts["num"] for num_tgts in targets["num_targets"]], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_spans(self, outputs, targets, indices):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "spans" containing a tensor of dim [nb_tgt_spans, 2]
           The target spans are expected in format (center, width), normalized by the image size.
        """
        assert 'pred_spans' in outputs
        targets = targets["target_spans"]
        idx = self._get_src_permutation_idx(indices)
        src_spans = outputs['pred_spans'][idx]  # (#spans, 2)
        tgt_spans = torch.cat([t['spans'][i] for t, (_, i) in zip(targets, indices)], dim=0)  # (#spans, 2)

        losses = {}
        loss_span = F.l1_loss(src_spans, tgt_spans, reduction='none')
        losses['loss_span'] = loss_span.mean()

        if self.span_type == 'cw':
            loss_giou = 1 - torch.diag(
                generalized_span_iou(
                    span_cw_to_xx(src_spans),
                    span_cw_to_xx(tgt_spans)
                )
            )
        elif self.span_type == 'xx':
            loss_giou = 1 - torch.diag(
                generalized_span_iou(
                    src_spans,
                    tgt_spans
                )
            )
        losses['loss_giou'] = loss_giou.mean()
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx  # two 1D tensors of the same length

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "spans": self.loss_spans,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        # list(tuples), each tuple is (pred_span_indices, tgt_span_indices)
        indices = self.matcher(outputs_without_aux, targets)

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


def build_loss(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    
    matcher = build_hungarian_matcher(args)
    # matcher = build_bipartite_matcher(args)

    weight_dict = {"loss_span": args.set_cost_span,
                   "loss_giou": args.set_cost_giou,
                   "loss_label": args.set_cost_query}
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'spans']

    return SetCriterion(
        num_classes=args.num_input_sentences,
        matcher=matcher,
        weight_dict=weight_dict,
        losses=losses,
        pred_label=args.pred_label,
        span_type=args.span_type
    )