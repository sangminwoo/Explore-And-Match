import logging
import os
import sys
from datetime import datetime

def setup_logger(name, save_dir, distributed_rank, filename="log.txt"):
	logger = logging.getLogger(name)
	logger.setLevel(logging.DEBUG) # DEBUG, INFO, ERROR, WARNING
	# don't log results for the non-master process
	if distributed_rank > 0:
		return logger

	stream_handler = logging.StreamHandler(stream=sys.stdout)
	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	stream_handler.setFormatter(formatter)
	logger.addHandler(stream_handler)

	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	file_handler = logging.FileHandler(os.path.join(save_dir, filename))
	file_handler.setFormatter(formatter)
	logger.addHandler(file_handler)

	return logger


def get_timestamp():
	now = datetime.now()
	timestamp = datetime.timestamp(now)
	st = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
	return st