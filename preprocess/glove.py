import os
import sys
import numpy as np
import json
import time
import argparse
from collections import defaultdict
from nltk.tokenize import word_tokenize
sys.path.append(os.path.dirname(os.path.abspath(
                os.path.dirname((__file__)))))
from lib.utils.misc import AverageMeter


def txt_to_json(dir_to_glove):
	glove = {}
	txt = open(dir_to_glove, 'r')
	while True:
	    line = txt.readline()
	    if not line:
	        break
	    glove[line.split(' ')[0]] = [float(i) for i in line.split(' ')[1:-1]] + [float(line.split(' ')[-1][:-1])]

	with open('glove.42B.300d.json', 'w') as j:
	    json.dump(glove, j)


def sentence_to_glove_embeddings(dir_to_glove, sentence):
	with open(dir_to_glove) as j:
		glove_dict = json.load(j)

	sentence = sentence.lower() # lowercase
	if sentence[-1] == '.':
		sentence = sentence[:-1] # remove '.'
	words = sentence.split(' ')
	
	glove_embeddings = []
	for word in words:
		glove_embeddings.append(glove_dict[word])
	return glove_embeddings


def get_raw_glove(dir_to_glove):
    tictoc = time.time()
    glove_model = {}
    with open(dir_to_glove,'r') as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = [float(v) for v in split_line[1:]] # np.asarray(split_line[1:], dtype='float32')
            glove_model[word] = embedding
    print(f'{len(glove_model)} words loaded!'
          f' Time taken to load: {time.strftime("%X", time.gmtime(time.time() - tictoc))}')
    return glove_model


def get_compact_glove(dir_to_data, dataset, dir_to_glove):
	time_meters = defaultdict(AverageMeter)

	# tokenize all unique words (lower-cased) in annotation files
	unique_words = set()

	phases = ['train', 'val', 'test'] if dataset in ['activitynet'] else ['train', 'test']
	for phase in phases:
		tictoc = time.time()
		dir_to_annotations = os.path.join(dir_to_data, dataset, 'annotations', phase+'.json')
		with open(dir_to_annotations) as j:
			annotations = json.load(j)
		time_meters['load_annotations'].update(time.time() - tictoc)
		
		tictoc = time.time()
		for anno in annotations.values():
			for sentence in anno['sentences']:
				tokens = word_tokenize(sentence)
				for token in tokens:
					if '/' in token:
						token = token.replace('/', ' / ')
						for t in token.split():
							unique_words.add(t.lower())
					else:
						unique_words.add(token.lower())
		time_meters['tokenize_unique_words'].update(time.time() - tictoc)

	# load glove model
	tictoc = time.time()
	glove = get_raw_glove(dir_to_glove)
	time_meters['load_glove'].update(time.time() - tictoc)

	# save dictionary {unique_word:glove embeddings}
	tictoc = time.time()
	word2glove = dict()

	for word in unique_words:
		if word not in glove:
		# unseen word (e.g., typos, misspellings) to zero vector
			word2glove[word] = [0] * 300 # np.zeros(300, dtype='float32')
		else:
			word2glove[word] = glove[word]

	with open(os.path.join(dir_to_glove.replace('.txt', f'.{dataset}.json')), 'w') as j:
	    json.dump(word2glove, j)
	time_meters['save_word2glove'].update(time.time() - tictoc)

	print('Time stats:')
	for name, meter in time_meters.items():
		d = {k: f'{getattr(meter, k):.4f}' for k in ['max', 'min', 'avg']}
		print(f'{name} ==> {d}')

	return word2glove


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Glove Embeddings')
    parser.add_argument('--root_dir', type=str, default='/ROOT_DIR',
                        help='Folder containing dataset')
    parser.add_argument('--glove_dir', type=str, default='/ROOT_DIR/glove/glove.6B.300d.txt',
                        help='Folder containing glove')
    parser.add_argument('--dataset', type=str, default='activitynet',
                        choices=['activitynet' , 'charades'],
                        help='Dataset to process')
    args = parser.parse_args()

    get_compact_glove(args.root_dir, args.dataset, args.glove_dir)