import os
import re
import time
import yaml
import random
import logging
import colorlog
from colorama import init
from datetime import datetime, timezone, timedelta

import numpy as np
import torch

'''
Enviroment Utils
'''
def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable

def init_seed(seed=None, reproducibility=None, cuda_id=0):
    r""" init random seed for random functions in numpy, torch, cuda and cudnn
    Args:
        seed (int): random seed
        reproducibility (bool): Whether to require reproducibility
    """
    if seed is None:
        seed = int(time.time() * 1000 // 1000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    torch.cuda.set_device(cuda_id) # only for gpu_torch



'''
Logger Utils
'''
log_colors_config = {
    'DEBUG': 'cyan',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'red',
}

def get_local_time():
    r"""Get current time
    Returns:
        str: current time
    """
    tz = timezone(timedelta(hours=+8)) # the east 8th time zone
    cur = datetime.now(tz)
    cur = cur.strftime('%b-%d-%Y_%H-%M-%S')

    return cur


def ensure_dir(dir_path):
    r"""Make sure the directory exists, if it does not exist, create it
    Args:
        dir_path (str): directory path
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


class RemoveColorFilter(logging.Filter):

    def filter(self, record):
        if record:
            ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
            record.msg = ansi_escape.sub('', str(record.msg))
        return True


def set_color(log, color, highlight=True):
    color_set = ['black', 'red', 'green', 'yellow', 'blue', 'pink', 'cyan', 'white']
    try:
        index = color_set.index(color)
    except:
        index = len(color_set) - 1
    prev_log = '\033['
    if highlight:
        prev_log += '1;3'
    else:
        prev_log += '0;3'
    prev_log += str(index) + 'm'
    return prev_log + log + '\033[0m'


def init_logger(config, state=None):
    """
    A logger that can show a message on standard output and write it into the
    file named `filename` simultaneously.
    All the message that you want to log MUST be str.
    Args:
        entry_name: the name of sub-folder, e.g., model_name by default
    Example:
        >>> logger = logging.getLogger(config)
        >>> logger.debug(train_state)
        >>> logger.info(train_result)
    """
    init(autoreset=True)
    LOGROOT = config['log_root'] if config['log_root'] else './log/'
    LOGROOT += generate_model_dir(config)
    dir_name = os.path.dirname(LOGROOT)
    ensure_dir(dir_name)
    config['log_dir'] = dir_name
    logfilename = 'logging.log'

    logfilepath = os.path.join(LOGROOT, logfilename)

    filefmt = "%(asctime)-15s %(levelname)s  %(message)s"
    filedatefmt = "%a %d %b %Y %H:%M:%S"
    fileformatter = logging.Formatter(filefmt, filedatefmt)

    sfmt = "%(log_color)s%(asctime)-15s %(levelname)s  %(message)s"
    sdatefmt = "%d %b %H:%M"
    sformatter = colorlog.ColoredFormatter(sfmt, sdatefmt, log_colors=log_colors_config)
    if state is None or state.lower() == 'info':
        level = logging.INFO
    elif state.lower() == 'debug':
        level = logging.DEBUG
    elif state.lower() == 'error':
        level = logging.ERROR
    elif state.lower() == 'warning':
        level = logging.WARNING
    elif state.lower() == 'critical':
        level = logging.CRITICAL
    else:
        level = logging.INFO

    fh = logging.FileHandler(logfilepath)
    fh.setLevel(level)
    fh.setFormatter(fileformatter)
    remove_color_filter = RemoveColorFilter()
    fh.addFilter(remove_color_filter)

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(sformatter)

    logging.basicConfig(level=level, handlers=[sh, fh])
    return config


'''
Config Utils
'''
class Config(object):
    def __init__(self, model=None, dataset=None, config_file_list=None, config_dict=None):
        self.yaml_loader = self._build_yaml_loader()
        self.final_config_dict = self._load_config_files(config_file_list)
        self.final_config_dict['model_name'], self.final_config_dict['dataset'] = model, dataset
        self.parameters = self.final_config_dict

    def _load_config_files(self, file_list):
        file_config_dict = dict()
        if file_list:
            for file in file_list:
                with open(file, 'r', encoding='utf-8') as f:
                    file_config_dict.update(yaml.load(f.read(), Loader=self.yaml_loader))
        return file_config_dict
    
    def _build_yaml_loader(self):
        loader = yaml.FullLoader
        loader.add_implicit_resolver(
            u'tag:yaml.org,2002:float',
            re.compile(
                u'''^(?:
             [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$''', re.X
            ), list(u'-+0123456789.')
        )
        return loader

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        self.final_config_dict[key] = value

    def __getitem__(self, item):
        if item in self.final_config_dict:
            return self.final_config_dict[item]
        else:
            return None

    def __contains__(self, key):
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        return key in self.final_config_dict

    def __str__(self):
        args_info = '\n'
        for category in self.parameters:
            args_info += set_color(category + ' Hyper Parameters:\n', 'pink')
            args_info += '\n'.join([(set_color("{}", 'cyan') + " =" + set_color(" {}", 'yellow')).format(arg, value)
                                    for arg, value in self.final_config_dict.items()
                                    if arg in self.parameters[category]])
            args_info += '\n\n'

        args_info += set_color('Other Hyper Parameters: \n', 'pink')
        args_info += '\n'.join([
            (set_color("{}", 'cyan') + " = " + set_color("{}", 'yellow')).format(arg, value)
            for arg, value in self.final_config_dict.items()
            if arg not in {
                _ for args in self.parameters.values() for _ in args
            }.union({'model', 'dataset', 'config_files'})
        ])
        args_info += '\n\n'
        return args_info

    def __repr__(self):
        return self.__str__()

'''
Other Utils
'''
def get_dataset_folder(config, dataset): # TODO
    dataset_folder = None
    if config['dataset_type'] == 'dataset_cf':
        pass
    elif config['dataset_type'] == 'dataset_sr':
        dataset_folder = 'thre-{}_filt-{}_core-{}_split-[{}]_ratio-{}'.format(
            config['rating_threshold'],
            config['filter_strategy'],
            config['core'],
            config['train_test_strategy'],
            config['split_ratio'],
        )
        dataset_folder = os.path.join(config['data_folder'][dataset], config['dataset_type'], dataset_folder)
    elif config['dataset_type'] == 'dataset_mmsr':
        dataset_folder = 'thre-{}_filt-{}_core-{}_split-[{}]_ratio-{}__imagec-{}_textc-{}_k-{}-{}-{}'.format(
            config['rating_threshold'],
            config['filter_strategy'],
            config['core'],
            config['train_test_strategy'],
            config['split_ratio'],
            config['image_cluster_num'],
            config['text_cluster_num'],
            config['link_k'],
            config['image_extractor'],
            config['text_extractor'],
        )
        dataset_folder = os.path.join(config['data_folder'][dataset], config['dataset_type'], dataset_folder)
    return dataset_folder



def generate_model_dir(config):
    model_name = config['model_name']
    if model_name=='GCE-GNN':
        return model_name + '/{}/bs{}-{}-lr{}-h{}/'.format(
            config['dataset'], config['batch_size'], 
            get_local_time(), config['lr'], config['hiddenSize'])
    elif model_name=='SR-GNN':
        return model_name + '/{}/bs{}-{}-lr{}-h{}/'.format(
            config['dataset'], config['batch_size'], 
            get_local_time(), config['lr'], config['hiddenSize'])
    elif model_name=='GC-SAN':
        return model_name + '/{}/bs{}-{}-lr{}-h{}/'.format(
            config['dataset'], config['batch_size'], 
            get_local_time(), config['lr'], config['hiddenSize'])
    else:
        return model_name + '/{}/bs{}-{}-lr{}-h{}/'.format(
            config['dataset'], config['batch_size'], 
            get_local_time(), config['lr'], config['hiddenSize'])
