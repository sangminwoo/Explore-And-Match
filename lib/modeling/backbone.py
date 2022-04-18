import os
import json
import torch
import torch.nn as nn
from lib.modeling.models import clip
from lib.modeling.models.model_clip import build_image_clip, build_text_clip
from lib.modeling.models.model_c3d import build_c3d
from lib.modeling.models.model_lstm import build_lstm
from lib.utils.tensor_utils import pad_sequences_2d


class Backbone(nn.Module):

    def __init__(self, text_backbone, video_backbone, backbone_type, oom=False):
        super(Backbone, self).__init__()
        self.backbone_t = text_backbone
        self.backbone_v = video_backbone
        self.backbone_type = backbone_type
        self.txt_fc = nn.Linear(512, 512)
        self.vid_fc = nn.Linear(8192, 512)
        self.oom = oom

    def forward(self, txt, vid):
        '''
        input:
            txt: sequence of sentences
                clip: (batch x num_sentences x 77)
                c3d_lstm:(batch x num_sentences x len_longest_seq x dim)

            vid: sequence of frames (or video clip)
                clip: (batch x num_frames x C x H x W)
                c3d_lstm: (batch x num_clips x C x T x H x W)
        '''
        if self.backbone_t:
            if self.backbone_type == 'clip':
                B, N, L = txt.shape 
                txt = txt.view(-1, L) # B*NxL
                txt = self.backbone_t(txt.int()) # B*NxD
                txt = txt.view(B, N, -1) # BxNxD
            elif self.backbone_type == 'c3d_lstm':
                if self.oom:
                    out = []
                    for txt_ in txt: # NxLxD
                        txt_ = txt_.permute(1, 0, 2) # NxLxD -> LxNxD
                        txt_ = self.backbone_t(txt_) # LxNxD -> NxD
                        txt_ = self.txt_fc(txt_) # NxD -> NxD'
                        out.append(txt_)
                    txt = torch.stack(out) # BxNxD'
                else:
                    B, N, L, D = txt.shape
                    txt = txt.view(-1, L, D) # B*NxLxD
                    txt = txt.permute(1, 0, 2) # LxB*NxD
                    txt = self.backbone_t(txt) # LxB*NxD -> B*NxD
                    txt = self.txt_fc(txt) # B*NxD -> B*NxD'
                    txt = txt.view(B, N, -1)

        if self.backbone_v:
            if self.backbone_type == 'clip':
                if self.oom:
                    out = []
                    for vid_ in vid:
                        vid_ = self.backbone_v(vid_)
                        out.append(vid_)
                    vid = torch.stack(out) # BxMxD
                else:
                    B, N, C, H, W = vid.shape
                    vid = vid.view(-1, C, H, W)
                    vid = self.backbone_v(vid)
                    vid = vid.view(B, N, -1)
                
            elif self.backbone_type == 'c3d_lstm':
                if self.oom:
                    out = []
                    for vid_ in vid:
                        vid_ = self.backbone_v(vid_) # Mx512x1x4x4
                        vid_ = vid_.view(vid_.shape[0], -1) # Mx8192
                        vid_ = self.vid_fc(vid_) # MxD'
                        out.append(vid_)
                    vid = torch.stack(out) # BxMxD'
                else:
                    B, N, C, T, H, W = vid.shape
                    vid = vid.view(-1, C, T, H, W) # B*NxCxTxHxW
                    vid = self.backbone_v(vid) # B*Nx512x1x4x4
                    vid = vid.view(vid.shape[0], -1) # B*Nx8192
                    vid = self.vid_fc(vid) # B*NxD'
                    vid = vid.view(B, N, -1)

        txt_mask = torch.ones(txt.shape[:2]).to(txt.device) # BxN
        vid_mask = torch.ones(vid.shape[:2]).to(vid.device) # BxM

        return txt, txt_mask, vid, vid_mask


def build_backbone(args):
    if 'features' in args.data_type:
        text_backbone = None
        video_backbone = None
    else: # raw
        if args.backbone == 'clip':
            model, _ = clip.load("ViT-B/32", device='cuda')
            text_backbone = build_text_clip(model)
            video_backbone = build_image_clip(model)
        elif args.backbone == 'c3d_lstm':
            text_backbone = build_lstm(args)
            video_backbone = build_c3d(args)
        else:
            raise NotImplementedError

    backbone = Backbone(
        text_backbone,
        video_backbone,
        backbone_type=args.backbone
    )
    return backbone