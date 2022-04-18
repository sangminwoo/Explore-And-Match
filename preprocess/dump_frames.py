import os
import sys
import shutil
import glob
import time
import argparse
import warnings
import cv2
import imutils
import numpy as np
from tqdm import tqdm, trange
from collections import defaultdict
sys.path.append(os.path.dirname(os.path.abspath(
                os.path.dirname((__file__)))))
from lib.utils.misc import AverageMeter


def extract_N_frames_from_all_videos(data_dir, save_dir, resize, scale=224, num_frames=64):
    '''Return constant frames for any length videos
    args:
        data_dir: folder containing videos to process
        save_dir: folder containing frames to be dumped
        resize: if True, frames will be resized
        scale: the shorter side of frame will be rescaled to 'scale'
        num_frames: number of constant frames to extract
    '''
    video_dirs = os.listdir(data_dir)
    time_meters = defaultdict(AverageMeter)

    # For each video, dump frames
    for video_dir in tqdm(video_dirs):
        tictoc = time.time()
        frame_dir = os.path.join(save_dir, str(num_frames), video_dir)
        if os.path.exists(frame_dir):
            continue
        else:
            os.makedirs(frame_dir)

        # count the total number of frames in video
        vidcap = cv2.VideoCapture(os.path.join(data_dir, video_dir))
        if resize:
            width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            if width < height:
                size = (scale, round(scale*height/width))
            else:
                size = (round(scale*width/height), scale)
        total_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
        time_meters['count_frames_and_get_size'].update(time.time() - tictoc)
        tictoc = time.time()

        # save frame indices to extract into the list 
        # [first, first+skip_step, ... , last-skip_step, last] (first & last inclusive)
        first_frame = 0
        last_frame = total_frames - 1
        skip_step = last_frame / (num_frames - 1)
        idxs_to_extract = np.arange(first_frame, last_frame + skip_step, skip_step).round() 
        idxs_to_extract = idxs_to_extract.astype('int').tolist()
        time_meters['frame_indices_to_list'].update(time.time() - tictoc)
        tictoc = time.time()

        # extract constant frames from video
        frame_idx = 0
        ret, frame = vidcap.read()
        if not ret:
            print(f'No valid frames exist in video: {video_dir}, skipping the process')

        while ret:
            if frame_idx in idxs_to_extract:
                if resize:
                    frame = cv2.resize(frame, dsize=size, interpolation=cv2.INTER_CUBIC)
                    time_meters['resize_frame'].update(time.time() - tictoc)
                    tictoc = time.time()
                cv2.imwrite(os.path.join(frame_dir, f'{frame_idx:05d}.png'), frame)
                time_meters['save_frame'].update(time.time() - tictoc)
                tictoc = time.time()
            frame_idx += 1
            ret, frame = vidcap.read()
        # vidcap.release()

    print('Time stats:')
    for name, meter in time_meters.items():
        d = {k: f'{getattr(meter, k):.4f}' for k in ['max', 'min', 'avg']}
        print(f'{name} ==> {d}')


def extract_N_frames_from_single_video(data_dir, video_name, save_dir, resize, scale=224, num_frames=64):
    video_dir = os.path.join(data_dir, video_name)
    frame_dir = os.path.join(save_dir, str(num_frames), video_name)
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir)

    # count the total frames in video
    vidcap = cv2.VideoCapture(video_dir)
    if resize:
        width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if width < height:
            size = (scale, round(scale*height/width))
        else:
            size = (round(scale*width/height), scale)
    total_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)

    # save frame indices to extract into the list (first & last inclusive)
    # [first, first+skip_step, ... , last-skip_step, last]
    first_frame = 0
    last_frame = total_frames - 1
    skip_step = last_frame / (num_frames-1)
    idxs_to_extract = np.arange(first_frame, last_frame + skip_step, skip_step).round() 
    idxs_to_extract = idxs_to_extract.astype('int').tolist()

    # extract constant frames from video
    frame_idx = 0
    ret, frame = vidcap.read()
    if not ret:
        print(f'No valid frames exist in video: {video_dir}, skipping the process')

    while ret:
        if frame_idx in idxs_to_extract:
            if resize:
                frame = cv2.resize(frame, dsize=size, interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(frame_dir, f'{frame_idx:05d}.png'), frame)
        frame_idx += 1
        ret, frame = vidcap.read()
    vidcap.release()

    return int(total_frames)


def rematch_non_matched_videos(root_dir, resize, scale, verbose=False):
    list_dataset = ['activitynet', 'charades']
    list_num_frames = [16, 32, 64, 128, 256]

    for dataset in list_dataset:
        for num_frames in list_num_frames:
            dir_to_frames = f'/ROOT_DIR/{dataset}/frames/{num_frames}'
            list_vid = os.listdir(dir_to_frames)
            len_frames = [len(os.listdir(os.path.join(dir_to_frames, vid))) for vid in list_vid]
            non_matched_idxs = list((np.asarray(len_frames) != num_frames).nonzero()[0])
            print(f'Non-matched videos in {dataset}/{num_frames}: {len(non_matched_idxs)}')

            for non_matched in non_matched_idxs:
                non_matched_vid = list_vid[non_matched]
                data_dir = os.path.join(root_dir, dataset, 'videos')
                save_dir = os.path.join(root_dir, dataset, 'frames')

                if verbose:
                    len_before_process = len(os.listdir(os.path.join(dir_to_frames, non_matched_vid)))
                    print(f'number of frames before process in {dataset}/{non_matched_vid}: {len_before_process}')

                # remove non-matched videos
                shutil.rmtree(os.path.join(save_dir, str(num_frames), non_matched_vid))

                # re-extract videos
                total_frames = extract_N_frames_from_single_video(
                    data_dir,
                    non_matched_vid,
                    save_dir,
                    resize, scale, num_frames=num_frames
                )
            
                len_after_process = len(os.listdir(os.path.join(dir_to_frames, non_matched_vid)))
                if verbose:
                    print(f'number of frames after process in {dataset}/{non_matched_vid}: {len_after_process}/{total_frames}')
                    
                if (len_after_process != num_frames) or \
                    (total_frames < num_frames and len_after_process != total_frames):
                    print(f'not matched! {dataset}/{non_matched_vid} '
                          f'(num_frames:{len_after_process}/{num_frames}, total_frames:{total_frames})')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dump frames')
    parser.add_argument('--root_dir', type=str, default='/ROOT_DIR',
                        help='Folder containing dataset')
    parser.add_argument('--dataset', type=str, default='charades',
                        choices=['activitynet' , 'charades'],
                        help='Dataset to process')
    parser.add_argument('--no_resize', dest='resize', action='store_false',
                        help='Disable resizing the frames to fixed scale')
    parser.add_argument('--scale', type=int, default=224,
                        help='Scale to resize')
    parser.add_argument('--num_frames', type=int, default=64,
                        help='Number of frames to extract from a video')
    args = parser.parse_args()

    print(f'Dumping {args.num_frames} frames from {args.dataset} dataset...')
    start_time = time.time()
    extract_N_frames_from_all_videos(
        data_dir=os.path.join(args.root_dir, args.dataset, 'videos'),
        save_dir=os.path.join(args.root_dir, args.dataset, 'frames'),
        resize=args.resize,
        scale=args.scale,
        num_frames=args.num_frames
    )
    end_time = time.time()
    print(f'Dumping Finished! Time taken to process: {time.strftime("%X", time.gmtime(end_time - start_time))}')

    # rematch_non_matched_videos(
    #     root_dir=args.root_dir,
    #     resize=args.resize,
    #     scale=args.scale,
    #     verbose=False
    # )