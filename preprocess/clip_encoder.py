import os
import sys
import glob
import json
import argparse
import torch
import torch.nn as nn
import time
import clip
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
sys.path.append(os.path.dirname(os.path.abspath(
                os.path.dirname((__file__)))))
from lib.modeling.models.clip import build_image_clip, build_text_clip
from lib.utils.misc import AverageMeter


def encode_image_text_with_clip(dataset, dir_to_data, num_frames,
                                clip_model="ViT-B/32", image_only=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    time_meters = defaultdict(AverageMeter)
    tictoc = time.time()

    # load clip model
    model, preprocess = clip.load(clip_model, device=device)
    model_text = build_text_clip(model)
    model_image = build_image_clip(model)
    time_meters['load_model'].update(time.time()-tictoc)
    tictoc = time.time()

    dir_to_anno = os.path.join(dir_to_data, 'annotations')

    phases = ['train', 'val', 'test'] if dataset in ['activitynet'] else ['train', 'test']
    for phase in phases:
        # load annotations
        with open(os.path.join(dir_to_anno, phase + '.json')) as j:
            annos = json.load(j)
        time_meters['load_annotations'].update(time.time()-tictoc)
        tictoc = time.time()

        for video_id in tqdm(list(annos.keys()), desc=phase):
            save_dir = os.path.join(dir_to_data, 'clip_features', phase, video_id, )
            if os.path.exists(save_dir):
                if os.path.exists(os.path.join(save_dir, f'vid_feats_{str(num_frames)}.pt')):
                    continue
                if os.path.exists(os.path.join(save_dir, 'txt_feats.pt')):
                    # if text features already exists, 'image_only' is set to True
                    # only image features are extracted
                    image_only = True
            else:
                os.makedirs(save_dir)

            # prepare data
            annotations = annos[video_id]
            if not image_only:
                video_captions = annotations['sentences']

            dir_to_frame = os.path.join(dir_to_data, 'frames', str(args.num_frames), video_id+'*')
            if not os.path.exists(dir_to_frame):
                ValueError(f'The directory {dir_to_frame} does not exists.')

            frames = sorted(glob.glob(os.path.join(dir_to_frame, '*.png')))
            if len(frames) == 0:
                print(f'No valid frames exist in {dir_to_frame}.')
                continue 
            video_frames = [Image.open(frame).convert('RGB') for frame in frames]
            time_meters['prepare_text_image'].update(time.time()-tictoc)
            tictoc = time.time()

            # preprocess
            if not image_only:
                text = clip.tokenize(video_captions, truncate=True).to(device)
            frames = torch.cat([preprocess(video_frame).unsqueeze(0).to(device) for video_frame in video_frames], dim=0)
            time_meters['preprocess_text_image'].update(time.time()-tictoc)
            tictoc = time.time()

            # encode
            with torch.no_grad():
                if not image_only:
                    text_features = model_text(text) # Mx512
                video_features = model_image(frames) # Nx512
            time_meters['encode_text_image'].update(time.time()-tictoc)
            tictoc = time.time()

            # save features
            if not image_only:
                torch.save(text_features.cpu(), os.path.join(save_dir, 'txt_feats') + '.pt')
            torch.save(video_features.cpu(), os.path.join(save_dir, 'vid_feats_' + str(num_frames))+ '.pt')
            time_meters['save_features'].update(time.time()-tictoc)
            tictoc = time.time()

    print('Time stats:')
    for name, meter in time_meters.items():
        d = {k: f'{getattr(meter, k):.4f}' for k in ['max', 'min', 'avg']}
        print(f'{name} ==> {d}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CLIP encoder')
    parser.add_argument('--root', type=str, default='/ROOT_DIR',
                        help='root directory of dataset')
    parser.add_argument('--dataset', type=str, default='activitynet',
                        choices=['activitynet', 'charades'],
                        help='the name of dataset')
    parser.add_argument('--num_frames', type=int, default=64,
                        help='number of input frames.')
    parser.add_argument('--clip_model', type=str, default='ViT-B/32',
                        choices=['ViT-B/32', 'ViT-B/16', 'ViT-L/14'],
                        help='variants of clip model')
    parser.add_argument('--image_only', action='store_true',
                        help='if image_only == True, only image features are extracted.')
    args = parser.parse_args()

    print(f'Extracting features from {args.num_frames} frames of {args.dataset} video dataset...')
    start_time = time.time()
    dir_to_data = os.path.join(args.root, args.dataset)
    encode_image_text_with_clip(args.dataset, dir_to_data, args.num_frames, args.clip_model, args.image_only)
    end_time = time.time()
    print(f'Extraction Finished! Time taken to process: {time.strftime("%X", time.gmtime(end_time - start_time))}')