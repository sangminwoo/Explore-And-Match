import torch
import torch.nn as nn
from lib.modeling.backbone import build_backbone
from lib.modeling.lvtr import build_lvtr

class VideoGroundingModel(nn.Module):

	def __init__(self, backbone, head, skip_backbone):
		super(VideoGroundingModel, self).__init__()
		self.backbone = backbone
		self.head = head
		self.skip_backbone = skip_backbone

	def forward(self, src_txt, src_vid, src_txt_mask=None, src_vid_mask=None,
				att_visualize=False, corr_visualize=False, epoch_i=None, idx=None):
		if not self.skip_backbone:
			src_txt, src_txt_mask, src_vid, src_vid_mask = \
				self.backbone(src_txt, src_vid)  # BxNx512, BxMx512

		# add no_class token
		# B, N, D = src_txt.shape
		# no_class_token = torch.zeros((B, 1, D), device=src_txt.device)  # Bx1x512
		# no_class_mask = torch.zeros((B, 1), dtype=torch.float32, device=src_txt_mask.device)  # Bx1
		# src_txt = torch.cat([src_txt, no_class_token], dim=1)  # Bx(N+1)x512
		# src_txt_mask = torch.cat([src_txt_mask, no_class_mask], dim=1)  # Bx(N+1)

		outputs = self.head(
			src_txt, src_txt_mask,
			src_vid, src_vid_mask,
			att_visualize, corr_visualize, epoch_i, idx
		)
		return outputs


def build_model(args):
	backbone = build_backbone(args)
	head = build_lvtr(args)

	model = VideoGroundingModel(
		backbone,
		head,
		skip_backbone=args.data_type=='features'
	)
	return model