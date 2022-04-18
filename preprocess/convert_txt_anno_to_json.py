import os
import sys
import json
import argparse
import time
import glob
import cv2
from collections import defaultdict
sys.path.append(os.path.dirname(os.path.abspath(
                os.path.dirname((__file__)))))
from lib.utils.misc import AverageMeter


def get_meta_info(filename):
    video = cv2.VideoCapture(filename)

    # duration = video.get(cv2.CAP_PROP_POS_MSEC)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = round(frame_count/fps, 2)

    return duration, frame_count, fps


def prepare_annos(dir_to_video):
	vids = os.listdir(dir_to_video)
	video_files = glob.glob(os.path.join(dir_to_video, '*'))
	vid2annos = {vid[:-4]:{} for vid in vids}

	for video_file in video_files:
		vid = video_file.split('/')[-1][:-4] # remove .mp4
		vid2annos[vid]['duration'], vid2annos[vid]['num_frames'], \
			vid2annos[vid]['fps'] = get_meta_info(video_file)

	return vid2annos


def txt_to_json(dir_to_video, dir_to_anno, remove_unused=True):
	'''
	convert txt format annotations to json format
	(only applicable for charades-sta annotations)
	'''
	time_meters = defaultdict(AverageMeter)
	for phase in ['train', 'test']:
		anno_file = os.path.join(dir_to_anno, f'{phase}.txt')

		tictoc = time.time()
		vid2annos = prepare_annos(dir_to_video)
		time_meters['cache_meta_annos'].update(time.time() - tictoc)

		tictoc = time.time()
		with open(anno_file, 'r') as txt_file:
			lines = txt_file.readlines()
			for line in lines:
				vid_span, sentence = line.split('##')
				vid, start, end = vid_span.split()

				if 'timestamps' not in vid2annos[vid]:
					vid2annos[vid]['timestamps'] = []

				if 'sentences' not in vid2annos[vid]:
					vid2annos[vid]['sentences'] = []

				vid2annos[vid]['timestamps'].append([float(start), float(end)])
				vid2annos[vid]['sentences'].append(sentence.replace('\n', ''))
		time_meters['cache_span_sentence'].update(time.time() - tictoc)

		tictoc = time.time()
		if remove_unused:
			unused_vids = []
			for vid, annos in vid2annos.items():
				if 'sentences' not in annos:
					unused_vids.append(vid)

			for vid in unused_vids:
				vid2annos.pop(vid)

		with open(anno_file.replace('txt', 'json'), 'w') as json_file:
			json.dump(vid2annos, json_file)
		time_meters['save_vid2annos'].update(time.time() - tictoc)

	print('Time stats:')
	for name, meter in time_meters.items():
		d = {k: f'{getattr(meter, k):.4f}' for k in ['max', 'min', 'avg']}
		print(f'{name} ==> {d}')


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='txt annotations to json')
	parser.add_argument('--root_dir', type=str, default='/ROOT_DIR',
						help='Folder containing dataset')
	parser.add_argument('--dataset', type=str, default='charades',
						help='Dataset to process')
	args = parser.parse_args()

	dir_to_video = os.path.join(args.root_dir, args.dataset, 'videos')
	dir_to_anno = os.path.join(args.root_dir, args.dataset, 'annotations')
	txt_to_json(dir_to_video, dir_to_anno)