import gym
import time
import os
import copy
import torch
import json
import numpy as np
import pandas as pd
from collections import deque
from pathlib import Path
from torch import nn
import torch.optim as optim
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete

from utils.logger import *

class TxtLogger:
  def __init__(self, experiment_folder):
    self.metrics_file = Path(experiment_folder) / "metrics.json"
    self.metrics_file.touch()

    self._keep_n_episodes = 5
    self.exploration_episode_lengths = deque(maxlen=self._keep_n_episodes)
    self.exploration_episode_returns = deque(maxlen=self._keep_n_episodes)
    self.exploration_episode_number = 0

  def log(self, metrics):
    metrics['Exploration episodes number'] = self.exploration_episode_number
    for name, d in zip(['episode length', 'episode return'], [self.exploration_episode_lengths, self.exploration_episode_returns]):
      metrics[f'Exploration {name}, mean'] = np.mean(d)
      metrics[f'Exploration {name}, std'] = np.std(d)
    with open(self.metrics_file, 'a') as out_metrics:
      json.dump(metrics, out_metrics)
      out_metrics.write('\n')

  def update_evaluation_statistics(self, episode_length, episode_return):
    self.exploration_episode_number += 1
    self.exploration_episode_lengths.append(episode_length)
    self.exploration_episode_returns.append(episode_return)

class BaseAgent(object):
  def __init__(self, cfg):
    self.logger = Logger(cfg['logs_dir'])
    self.txt_log_freq = cfg['txt_log_freq']
    self.txt_logger = TxtLogger(cfg['logs_dir'])