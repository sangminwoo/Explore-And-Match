# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class TransformerStepwise(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        # TransformerEncoderLayerThin
        vid_encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                    dropout, activation, normalize_before)
        vid_encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.vid_encoder = TransformerEncoder(vid_encoder_layer, num_encoder_layers, vid_encoder_norm)

        # TransformerEncoderLayerThin
        txt_encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                    dropout, activation, normalize_before)
        txt_encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.txt_encoder = TransformerEncoder(txt_encoder_layer, num_encoder_layers, txt_encoder_norm)

        # TransformerDecoderLayerThin
        decoder_layer = TransformerFusionDecoderLayer(d_model, nhead, dim_feedforward,
                                                      dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerFusionDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                                return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src_vid, src_txt, vid_mask, txt_mask, query_embed, vid_pos_embed, txt_pos_embed):
        """
        Args:
            src: (batch_size, L, d)
            mask: (batch_size, L)
            query_embed: (#queries, d)
            pos_embed: (batch_size, L, d) the same as src

        Returns:

        """
        # flatten NxCxHxW to HWxNxC
        bs, l, d = src_vid.shape
        bs, l, d = src_txt.shape
        src_vid = src_vid.permute(1, 0, 2)  # (L, batch_size, d)
        src_txt = src_txt.permute(1, 0, 2)  # (L, batch_size, d)
        vid_pos_embed = vid_pos_embed.permute(1, 0, 2)   # (L, batch_size, d)
        txt_pos_embed = txt_pos_embed.permute(1, 0, 2)   # (L, batch_size, d)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)  # (#queries, batch_size, d)

        tgt = torch.zeros_like(query_embed)
        vid_memory = self.vid_encoder(src_vid, src_key_padding_mask=vid_mask, pos=vid_pos_embed)  # (L, batch_size, d)
        txt_memory = self.txt_encoder(src_txt, src_key_padding_mask=txt_mask, pos=txt_pos_embed)  # (L, batch_size, d)
        hs, vid_att, txt_att = self.decoder(tgt, vid_memory, txt_memory,
                                            vid_memory_key_padding_mask=vid_mask,
                                            txt_memory_key_padding_mask=txt_mask,
                                            vid_pos=vid_pos_embed,
                                            txt_pos=txt_pos_embed,
                                            query_pos=query_embed)  # (#layers, #queries, batch_size, d)
        hs = hs.transpose(1, 2) # (#layers, batch_size, #qeries, d)
        # memory = memory.permute(1, 2, 0)  # (batch_size, d, L)
        vid_memory = vid_memory.transpose(0, 1)  # (batch_size, L, d)
        txt_memory = txt_memory.transpose(0, 1)  # (batch_size, L, d)
        return hs, vid_memory, txt_memory, vid_att, txt_att


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        intermediate = []

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
            if self.return_intermediate:
                intermediate.append(output)

        if self.norm is not None:
            output = self.norm(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerFusionDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, vid_memory, txt_memory,
                tgt_mask: Optional[Tensor] = None,
                vid_memory_mask: Optional[Tensor] = None,
                txt_memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                vid_memory_key_padding_mask: Optional[Tensor] = None,
                txt_memory_key_padding_mask: Optional[Tensor] = None,
                vid_pos: Optional[Tensor] = None,
                txt_pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []
        vid_att_weights = []
        txt_att_weights = []
        for layer in self.layers:
            output, vid_att, txt_att = layer(output, vid_memory, txt_memory,
                                             tgt_mask=tgt_mask,
                                             vid_memory_mask=vid_memory_mask,
                                             txt_memory_mask=txt_memory_mask,
                                             tgt_key_padding_mask=tgt_key_padding_mask,
                                             vid_memory_key_padding_mask=vid_memory_key_padding_mask,
                                             txt_memory_key_padding_mask=txt_memory_key_padding_mask,
                                             vid_pos=vid_pos,
                                             txt_pos=txt_pos,
                                             query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
                vid_att_weights.append(vid_att)
                txt_att_weights.append(txt_att)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate), \
                   torch.stack(vid_att_weights), \
                   torch.stack(txt_att_weights)

        return output.unsqueeze(0)


class TransformerFusionDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.vid_multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.txt_multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, vid_memory, txt_memory,
                     tgt_mask: Optional[Tensor] = None,
                     vid_memory_mask: Optional[Tensor] = None,
                     txt_memory_mask: Optional[Tensor] = None,                     
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     vid_memory_key_padding_mask: Optional[Tensor] = None,
                     txt_memory_key_padding_mask: Optional[Tensor] = None,
                     vid_pos: Optional[Tensor] = None,
                     txt_pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, vid_att = self.vid_multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                       key=self.with_pos_embed(vid_memory, vid_pos),
                                       value=vid_memory, attn_mask=vid_memory_mask,
                                       key_padding_mask=vid_memory_key_padding_mask)
        tgt3, txt_att = self.txt_multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                       key=self.with_pos_embed(txt_memory, txt_pos),
                                       value=txt_memory, attn_mask=txt_memory_mask,
                                       key_padding_mask=txt_memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt3)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, vid_att, txt_att

    def forward_pre(self, tgt, vid_memory, txt_memory,
                    tgt_mask: Optional[Tensor] = None,
                    vid_memory_mask: Optional[Tensor] = None,
                    txt_memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    vid_memory_key_padding_mask: Optional[Tensor] = None,
                    txt_memory_key_padding_mask: Optional[Tensor] = None,
                    vid_pos: Optional[Tensor] = None,
                    txt_pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2, vid_att = self.vid_multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                       key=self.with_pos_embed(vid_memory, vid_pos),
                                       value=vid_memory, attn_mask=vid_memory_mask,
                                       key_padding_mask=vid_memory_key_padding_mask)
        tgt3, txt_att = self.txt_multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                       key=self.with_pos_embed(txt_memory, txt_pos),
                                       value=txt_memory, attn_mask=txt_memory_mask,
                                       key_padding_mask=txt_memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt3)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, vid_memory, txt_memory,
                tgt_mask: Optional[Tensor] = None,
                vid_memory_mask: Optional[Tensor] = None,
                txt_memory_mask: Optional[Tensor] = None,                
                tgt_key_padding_mask: Optional[Tensor] = None,
                vid_memory_key_padding_mask: Optional[Tensor] = None,
                txt_memory_key_padding_mask: Optional[Tensor] = None,
                vid_pos: Optional[Tensor] = None,
                txt_pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, vid_memory, txt_memory, tgt_mask, vid_memory_mask, txt_memory_mask,
                                    tgt_key_padding_mask, vid_memory_key_padding_mask, txt_memory_key_padding_mask,
                                    vid_pos, txt_pos, query_pos)
        return self.forward_post(tgt, vid_memory, txt_memory, tgt_mask, vid_memory_mask, txt_memory_mask,
                                 tgt_key_padding_mask, vid_memory_key_padding_mask, txt_memory_key_padding_mask,
                                 vid_pos, txt_pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer_stepwise(args):
    return TransformerStepwise(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")