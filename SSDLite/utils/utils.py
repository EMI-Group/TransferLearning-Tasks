import os
import logging
import time
from pathlib import Path

import torch
import torch.optim as optim

def create_logger(save_path, is_train):
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = 'train_{}.log'.format(time_str) if is_train else 'test_{}.log'.format(time_str)
    final_log_file = '{}/{}'.format(save_path, log_file)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger