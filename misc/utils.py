import os
import numpy as np
import gym
import glob
import json
from collections import deque, OrderedDict
import psutil
import re
import csv
import pandas as pd
import ntpath
import re
import random

def set_global_seeds(myseed):

    import torch
    torch.manual_seed(myseed)
    np.random.seed(myseed)
    random.seed(myseed)

def get_fname_from_path(f):
    '''
     input:
           '/Users/user/logs/check_points/mmmxm_dummy_B32_H5_D1_best.pt'
     output:
           'mmmxm_dummy_B32_H5_D1_best.pt'
    '''
    return ntpath.basename(f)

def get_action_info(action_space, obs_space = None):
    '''
        This fucntion returns info about type of actions.
    '''
    space_type = action_space.__class__.__name__

    if action_space.__class__.__name__ == "Discrete":
            num_actions = action_space.n

    elif action_space.__class__.__name__ == "Box":
            num_actions = action_space.shape[0]

    elif action_space.__class__.__name__ == "MultiBinary":
            num_actions = action_space.shape[0]
    
    else:
        raise NotImplementedError
    
    return num_actions, space_type

def create_dir(log_dir, ext = '*.monitor.csv', cleanup = False):

    '''
        Setup checkpoints dir
    '''

    try:
        os.makedirs(log_dir)

    except OSError:
        if cleanup == True:
            files = glob.glob(os.path.join(log_dir, '*.'))

            for f in files:
                os.remove(f)

def dump_to_json(path, data):
    '''
      Write json file
    '''
    with open(path, 'w') as f:
        json.dump(data, f)

def read_json(input_json):
    ## load the json file
    file_info = json.load(open(input_json, 'r'))

    return file_info

class CSVWriter:

    def __init__(self, fname, fieldnames):

        self.fname = fname
        self.fieldnames = fieldnames
        self.csv_file = open(fname, mode='w')
        self.writer = None

    def write(self, data_stats):

        if self.writer == None:
            self.writer = csv.DictWriter(self.csv_file, fieldnames=self.fieldnames)
            self.writer.writeheader()

        self.writer.writerow(data_stats)
        self.csv_file.flush()

    def close(self):
        self.csv_file.close()

def safemean(xs):
    '''
        Avoid division error when calculate the mean (in our case if
        epinfo is empty returns np.nan, not return an error)
    '''
    return np.nan if len(xs) == 0 else np.mean(xs)