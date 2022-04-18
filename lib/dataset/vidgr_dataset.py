import os
import glob
import json
import random
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from nltk.tokenize import word_tokenize
from lib.modeling.models import clip
from lib.utils.tensor_utils import pad_sequences_1d, pad_sequences_2d
from lib.utils.span_utils import span_xx_to_cw
from lib.utils.misc import l2_normalize_tensor


class VideoGroundingDataset(Dataset):
    '''
    Video Grounding Dataset: ActivityNet and Charades
    '''
    EXCLUDE_FILES = {
        'activitynet':{
            'train': [],
            'val': ['v_0dkIbKXXFzI', 'v_j73Wh1olDsA'] # num_frames = 32, 29
        },
        'charades':{
            'train': [], 'val': []
        }
     }

    def __init__(self, root, dataset='activitynet', data_type='features', backbone='clip',
                 phase='train', num_input_frames=32, num_input_sentences=16,
                 normalize_txt_feats=True, normalize_vid_feats=True,
                 txt_drop_ratio=0., exlude_short=False, glove_type='glove.6B.300d.txt'):
        super().__init__()
        self.root = root
        self.dataset = dataset
        self.data_type = data_type
        self.backbone = backbone
        self.phase = phase
        assert self.phase in ['train', 'val', 'test'], \
            'phase should be one of train/val/test.'
        self.num_input_frames = num_input_frames
        self.num_input_sentences = num_input_sentences
        self.normalize_input_txt_feats = normalize_txt_feats
        self.normalize_input_vid_feats = normalize_vid_feats
        self.txt_drop_ratio = txt_drop_ratio if phase=='train' else 0
        self.glove_dir = os.path.join(root, 'glove', glove_type)

        # get video annotations
        dir_to_annotations = os.path.join(self.root, self.dataset, 'annotations')
        with open(os.path.join(dir_to_annotations, self.phase + '.json')) as j:
            self.annotations = json.load(j)

        if 'features' in self.data_type: # pre-trained features
            dir_to_feats = os.path.join(self.root, self.dataset, 'clip_features', self.phase)
            video_ids = os.listdir(dir_to_feats)
            self.vid2data = {vid: os.path.join(dir_to_feats, vid) for vid in video_ids}
        else: # raw images-texts
            dir_to_frames = os.path.join(self.root, self.dataset, 'frames', str(num_input_frames))
            video_ids = list(self.annotations.keys())
            self.vid2data = {vid: os.path.join(dir_to_frames, vid) for vid in video_ids}
            if 'clip' in self.backbone:
                _, self.clip_preprocess = clip.load("ViT-B/32")

        if exlude_short: # exclude short videos (not necessary?)
            for exclude in self.EXCLUDE_FILES[self.dataset][self.phase]:
                self.annotations.pop(exclude)
                self.vid2data.pop(exclude)

    def __len__(self):
        return len(self.vid2data)

    def __getitem__(self, idx):
        video_id = list(self.annotations.keys())[idx]
        annos = self.annotations[video_id]
        annos['video_id'] = video_id
        # fps = annos['fps']
        assert len(annos['timestamps']) == len(annos['sentences']), \
            "The number of target spans and input sentences does not matches."

        if self.phase=='train' and len(annos['sentences']) > self.num_input_sentences: # always shuffled
            random_idxs = np.random.choice(
                range(len(annos['sentences'])),
                self.num_input_sentences,
                replace=False
            )
            random_idxs = sorted(random_idxs)
            annos['sentences'] = [annos['sentences'][i] for i in random_idxs]
            annos['timestamps'] = [annos['timestamps'][i] for i in random_idxs]
        else:
            random_idxs = None

        sorted_by_time = sorted(zip(annos['sentences'], annos['timestamps']), key=lambda x: x[1][0] + x[1][1])
        annos['sentences'], annos['timestamps'] = map(list, zip(*sorted_by_time))

        model_inputs = dict()
        if 'features' in self.data_type: # pre-trained features
            dir_to_feats = self.vid2data[video_id]
            model_inputs['input_txt'] = self._get_txt_feats(dir_to_feats, random_idxs=random_idxs) 
            model_inputs['input_vid'] = self._get_vid_feats(dir_to_feats)
        else: # raw image-text
            dir_to_frames = self.vid2data[video_id]
            model_inputs['input_txt'] = self._get_txt(annos['sentences']) 
            model_inputs['input_vid'] = self._get_vid(dir_to_frames)

        if self.dataset in ['activitynet', 'charades']:
            duration = annos['duration']
        else:
            raise NotImplementedError(f'{self.dataset} is not supported.')

        model_inputs['target_spans'] = self._get_target_spans(
            spans=annos['timestamps'],
            duration=duration,
        )
        return dict(annos=annos, model_inputs=model_inputs)

    def _get_txt_feats(self, dir_to_feats, random_idxs=None):
        feature_dir = os.path.join(dir_to_feats, 'txt_feats.pt')
        txt_feats = torch.load(feature_dir).float()
        if random_idxs is not None:
            txt_feats = txt_feats[random_idxs]
        if self.normalize_input_txt_feats:
            txt_feats = l2_normalize_tensor(txt_feats)
        if self.txt_drop_ratio > 0:
            txt_feats = self._random_drop_rows(txt_feats)
        return txt_feats

    def _get_vid_feats(self, dir_to_feats):
        feature_dir = os.path.join(dir_to_feats, 'vid_feats_' + str(self.num_input_frames) + '.pt')
        vid_feats = torch.load(feature_dir).float()
        if self.normalize_input_vid_feats:
            vid_feats = l2_normalize_tensor(vid_feats)
        return vid_feats

    def _get_txt(self, sentences):
        if 'lstm' in self.backbone:
            return self._to_glove_embeddings(sentences)
        elif 'clip' in self.backbone:
            return clip.tokenize(sentences, truncate=True)
        else:
            NotImplementedError

    def _to_glove_embeddings(self, sentences):
        word2glove = self._load_glove(self.dataset, self.glove_dir)

        sentences_glove = []
        for words in sentences:
            words_glove = []
            words = word_tokenize(words)
            for word in words:
                if '/' in word:
                    word = word.replace('/', ' / ') # '/' as a delimter
                    for chars in word.split():
                        glove = word2glove[chars.lower()]
                        words_glove.append(glove) # lower case
                else:
                    glove = word2glove[word.lower()]
                    words_glove.append(glove) # lower case
            sentences_glove.append(torch.tensor(words_glove))
        return sentences_glove

    def _load_glove(self, dataset, dir_to_glove, use_compact_glove=True):
        if use_compact_glove: # need to generate compact glove before use
            with open(dir_to_glove.replace('.txt', f'.{dataset}.json')) as j:
                glove = json.load(j)
        else:
            glove = get_raw_glove(dir_to_glove)
        return glove

    def _get_vid(self, dir_to_frames):
        frames = sorted(glob.glob(os.path.join(dir_to_frames+'*', '*.png')))
        if len(frames) == 0:
            raise ValueError(f'No valid frames exist in {dir_to_frame}.')
        frames = [self._transform(Image.open(frame).convert('RGB')) for frame in frames]
        if self.backbone == 'clip':
            return torch.stack(frames, dim=0)
        elif self.backbone == 'c3d_lstm':
            # group by 16 frames when num_frames < 16 or num_frames%16!=0
            if len(frames) % 16 != 0:
                C, H, W = frames[0].shape
                frames = torch.cat([
                    torch.stack(frames),
                    torch.stack([torch.zeros(3, 112, 112)] * (16-(len(frames)%16)))
                ]) # 58x3x112x112 + 6x3x112x112 -> 64x3x112x112
                clips = torch.stack([
                    frames[16*i:16*(i+1)]
                    for i in range(len(frames)//16)
                ]) # 4x16x3x112x112
                clips = clips.permute(0, 2, 1, 3, 4) 
            else:
                clips = torch.stack([
                    torch.stack(frames[16*i:16*(i+1)], dim=1)
                    for i in range(len(frames)//16)
                ]) # Mx3x16x112x112
            return clips
        else:
            raise NotImplementedError

    def _transform(self, img):
        if 'c3d' in self.backbone:
            transform = transforms.Compose([
                    transforms.Resize(112),
                    transforms.RandomResizedCrop(112),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ])
            return transform(img)
        elif 'clip' in self.backbone:
            return self.clip_preprocess(img)
        else:
            raise NotImplementedError

    def _get_target_spans(self, spans, duration):
        spans = torch.Tensor(spans) / duration  # normalized spans in xx
        spans = span_xx_to_cw(spans)  # normalized spans in cxw
        return spans

    def _random_drop_rows(self, feats):
        # randomly mask num_drop rows in features to be zero.
        num_drop = round(len(feats) * self.txt_drop_ratio)
        if num_drop > 0:
            row_indices = np.random.choice(
                len(feats), size=num_drop, replace=False)
            feats[row_indices] = 0
        return feats

    def get_gt_with_vids(self, video_ids):
        '''
        return ground truth for evaluation.

        return:
            ground_truth: list(dict), each dict is {
              "video_id": "RoripwjYFp8_360.0_510.0",
              "query": "Man in gray top walks from outside to inside.",
              "duration": 150
            }
        '''
        ground_truth = []
        for vid in video_ids:
            annos = self.annotations[vid]
            for query, timespan in zip(annos['sentences'], annos['timestamps']):
                gt = {}
                gt['video_id'] = vid
                gt['query'] = query
                gt['gt_timespan'] = [timespan]
                if self.dataset in ['activitynet', 'charades']:
                    gt['duration'] = annos['duration']
                else:
                    raise NotImplementedError(f'{self.dataset} is not supported.')
                ground_truth.append(gt)
        return ground_truth


def build_dataset(args):
    return VideoGroundingDataset(
                root=args.root,
                dataset=args.dataset,
                data_type=args.data_type,
                backbone=args.backbone,
                phase=args.phase,
                num_input_frames=args.num_input_frames,
                num_input_sentences=args.num_input_sentences,
                normalize_txt_feats=args.norm_tfeat,
                normalize_vid_feats=args.norm_vfeat,
                txt_drop_ratio=args.txt_drop_ratio
            )


def collate_fn_feat(batch):
    batch_annos = [d['annos'] for d in batch]
    model_inputs_keys = batch[0]['model_inputs'].keys()
    batched_data = dict()
    for k in model_inputs_keys:
        if k == 'target_spans':
            batched_data[k] = [
                dict(spans=d['model_inputs']['target_spans']) for d in batch
            ]
            continue

        batched_data[k] = pad_sequences_1d(
            [d['model_inputs'][k] for d in batch],
            dtype=torch.float32,
        )
    return batch_annos, batched_data


def collate_fn_raw(batch):
    batch_annos = [d['annos'] for d in batch]
    model_inputs_keys = batch[0]['model_inputs'].keys()
    batched_data = dict()
    for k in model_inputs_keys:
        if k == 'target_spans':
            batched_data[k] = [
                dict(spans=d['model_inputs']['target_spans']) for d in batch
            ]
            continue

        # enable this when backbone=='c3d_lstm'
        if k == 'input_txt':
            batched_data[k] = pad_sequences_2d(
                [d['model_inputs'][k] for d in batch],
                dtype=torch.float32,
            )
            continue

        batched_data[k] = pad_sequences_1d(
            [d['model_inputs'][k] for d in batch],
            dtype=torch.float32,
        )
    return batch_annos, batched_data


def prepare_batch_inputs(batched_inputs, device, non_blocking=False):
    model_inputs = dict(
        src_txt=batched_inputs['input_txt'][0].to(device, non_blocking=non_blocking),
        src_txt_mask=batched_inputs['input_txt'][1].to(device, non_blocking=non_blocking),
        src_vid=batched_inputs['input_vid'][0].to(device, non_blocking=non_blocking),
        src_vid_mask=batched_inputs['input_vid'][1].to(device, non_blocking=non_blocking)
    )
    targets = {}
    if 'target_spans' in batched_inputs:
        targets['target_spans'] = [
            dict(spans=d['spans'].to(device, non_blocking=non_blocking))
            for d in batched_inputs['target_spans']
        ]

    targets = None if len(targets) == 0 else targets
    return model_inputs, targets