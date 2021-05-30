import os
import json
import random
import torch
import time
import numpy as np


def load_json(input_file):
    with open(input_file, 'r') as f:
        samples = json.load(f)
    return samples


def load_dict(dict_path):
    """load_dict"""
    vocab = {}
    for line in open(dict_path, 'r', encoding='utf-8'):
        key, value = line.strip('\n').split('\t')
        vocab[int(key)] = value
    return vocab


def write_dict(dict_path, dict_data):
    with open(dict_path, "w", encoding="utf-8") as f:
        for key, value in dict_data.items():
            f.writelines("{}\t{}\n".format(key, value))


def str_q2b(text):
    ustring = text
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:
            inside_code = 32
        elif 65281 <= inside_code <= 65374:
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring


class ProgressBar(object):
    """
    custom progress bar
    Example:
        >>> pbar = ProgressBar(n_total=30,desc='Training')
        >>> step = 2
        >>> pbar(step=step)
    """
    def __init__(self, n_total,width=30,desc = 'Training'):
        self.width = width
        self.n_total = n_total
        self.start_time = time.time()
        self.desc = desc

    def __call__(self, step, info={}):
        now = time.time()
        current = step + 1
        recv_per = current / self.n_total
        bar = f'[{self.desc}] {current}/{self.n_total} ['
        if recv_per >= 1:
            recv_per = 1
        prog_width = int(self.width * recv_per)
        if prog_width > 0:
            bar += '=' * (prog_width - 1)
            if current< self.n_total:
                bar += ">"
            else:
                bar += '='
        bar += '.' * (self.width - prog_width)
        bar += ']'
        show_bar = f"\r{bar}"
        time_per_unit = (now - self.start_time) / current
        if current < self.n_total:
            eta = time_per_unit * (self.n_total - current)
            if eta > 3600:
                eta_format = ('%d:%02d:%02d' %
                              (eta // 3600, (eta % 3600) // 60, eta % 60))
            elif eta > 60:
                eta_format = '%d:%02d' % (eta // 60, eta % 60)
            else:
                eta_format = '%ds' % eta
            time_info = f' - ETA: {eta_format}'
        else:
            if time_per_unit >= 1:
                time_info = f' {time_per_unit:.1f}s/step'
            elif time_per_unit >= 1e-3:
                time_info = f' {time_per_unit * 1e3:.1f}ms/step'
            else:
                time_info = f' {time_per_unit * 1e6:.1f}us/step'

        show_bar += time_info
        if len(info) != 0:
            show_info = f'{show_bar} ' + \
                        "-".join([f' {key}: {value:.4f} ' for key, value in info.items()])
            print(show_info, end='')
        else:
            print(show_bar, end='')


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# class EarlyStopper(object):
#     def __init__(self, patience, logger):
#         self.patience = patience
#         self.counter = 0
#         self.best_score = None
#         self.best_step = None
#         self.early_stop = False
#         self.logger = logger
#
#     def step(self, score, model, args, tokenizer, global_step):
#         if self.best_score is None:
#             self.best_score = score
#             self.best_step = global_step
#             self.save(model, args, tokenizer, global_step)
#         elif score > self.best_score:
#             self.best_score = score
#             self.best_step = global_step
#             self.save(model, args, tokenizer, global_step)
#             self.counter = 0
#         else:
#             self.counter += 1
#             self.logger.info("Earlystopper counter: %s out of %s", self.counter, self.patience)
#             if self.counter >= self.patience:
#                 self.early_stop = True
#
#     def save(self, model, args, tokenizer, global_step):
#         output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
#         if not os.path.exists(output_dir):
#             os.makedirs(output_dir)
#         model.save_pretrained(output_dir)
#         torch.save(args, os.path.join(output_dir, 'training_args.bin'))
#         self.logger.info("Saving model checkpoint to %s", output_dir)
#         tokenizer.save_vocabulary(save_directory=output_dir)
