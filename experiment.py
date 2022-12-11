import os
import glob
import sys
import copy
import time
import json
import torch
import numpy as np
import pandas as pd

import agents
from utils.helper import *


class Experiment(object):
  '''
  Train the agent to play the game.
  '''
  def __init__(self, cfg):
    self.cfg = copy.deepcopy(cfg)
    if torch.cuda.is_available() and 'cuda' in cfg['device']:
      self.device = cfg['device']
    else:
      self.cfg['device'] = 'cpu'
      self.device = 'cpu'
    self.config_idx = cfg['config_idx']
    self.env_name = cfg['env']['name']
    self.agent_name = cfg['agent']['name']
    if self.cfg['generate_random_seed']:
      self.cfg['seed'] = np.random.randint(int(1e6))
    self.model_path = self.cfg['model_path']
    self.cfg_path = self.cfg['cfg_path']
    self.save_config()

  def init_agent(self):
    set_one_thread()
    set_random_seed(self.cfg['seed'])
    self.agent = getattr(agents, self.agent_name)(self.cfg)
    self.agent.env['Train'].seed(self.cfg['seed'])
    self.agent.env['Train'].action_space.np_random.seed(self.cfg['seed'])
    self.agent.env['Test'].seed(self.cfg['seed'])
    self.agent.env['Test'].action_space.np_random.seed(self.cfg['seed'])

  def run(self):
    '''
    Run the game for multiple times
    '''
    self.start_time = time.time()
    self.init_agent()
    # Train && Test
    self.agent.run_steps(render=self.cfg['render'])
    # Save model
    # self.save_model()
    self.end_time = time.time()
    self.agent.logger.info(f'Memory usage: {rss_memory_usage():.2f} MB')
    self.agent.logger.info(f'Time elapsed: {(self.end_time-self.start_time)/60:.2f} minutes')

  def eval_bias(self, checkpoint_dir, checkpoint_prefix, metrics, logger):
    self.init_agent()
    checkpoints = glob.glob(f"{checkpoint_dir}/{checkpoint_prefix}*")
    checkpoints = sorted(checkpoints)
    eval_timestamps = [int(checkpoint.split('/')[-1].split('_')[-1]) for checkpoint in checkpoints]
    eval_num_nets = [metrics['nets/NumNets'][metrics['Total_timesteps'] == ts].values[0] for ts in eval_timestamps]
    assert len(eval_num_nets) == len(checkpoints), "Length mismatch!"
    for checkpoint, num_nets, eval_timestamp in zip(checkpoints, eval_num_nets, eval_timestamps):
      metrics = self.eval_single_checkpoint(checkpoint, eval_timestamp, num_nets)
      metrics = {'evals': metrics}
      metrics['checkpoint'] = checkpoint
      metrics['Timestamp'] = str(datetime.datetime.now())
      logger.log(metrics)
  
  def eval_single_checkpoint(self, checkpoint, eval_timestamp, num_nets):
    self.load_model(checkpoint)
    self.agent.step_count = eval_timestamp
    self.agent.nets_to_use = num_nets
    for i in range(self.cfg['eta']['Q_G_n_episodes']):
      self.agent.run_eval_episode()
    res_list = list()
    for i in range(100):
        threshold_metrics = self.agent.eval_thresholds(self.agent.replay,
                                                       self.cfg['eta']['Q_G_n_per_episode'],
                                                       n_episodes=self.cfg['eta']['Q_G_n_episodes_eval'])
        res_list.append(threshold_metrics)
    return res_list

  
  def save_model(self):
    self.agent.save_model(self.model_path)
  
  def load_model(self, model_path):
    self.agent.load_model(model_path)
    for i in range(self.agent.k):
        self.agent.Q_net_target[i].load_state_dict(self.agent.Q_net[i].state_dict())

  def save_config(self):
    cfg_json = json.dumps(self.cfg, indent=2)
    f = open(self.cfg_path, 'w')
    f.write(cfg_json)
    f.close()