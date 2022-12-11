import os
import json
import pandas as pd
import sys
import argparse

from pathlib import Path

from utils.sweeper import Sweeper
from utils.helper import make_dir
from experiment import Experiment

class Logger:
    def __init__(self, experiment_folder):
        experiment_folder = Path(experiment_folder)
        self.metrics_file = experiment_folder / "eval_metrics.json"
        self.metrics_file.touch()

    def log(self, metrics):
        with open(self.metrics_file, 'a') as out_metrics:
            json.dump(metrics, out_metrics)
            out_metrics.write('\n')

def read_metrics(metric_file):
    dicts = [json.loads(x.strip()) for x in open(metric_file)]
    columns = list(dicts[0].keys())
    df_dict = dict()
    for c in columns:
        df_dict[c] = [d[c] for d in dicts]
    return pd.DataFrame.from_dict(df_dict)

def main(argv):
  parser = argparse.ArgumentParser(description="Config file")
  parser.add_argument('--logs_dir', type=str, help='Directory with config, checkpoints and metrics.json')
  parser.add_argument('--store_N_episodes', type=int, default="Number of episodes to collect with each checkpoint")
  parser.add_argument('--checkpoint_prefix', type=str, default='iter', help='Prefix for checkpoints')
  parser.add_argument('--rollout_length', type=int, default=500)
  args = parser.parse_args()
  
  logger = Logger(args.logs_dir)
  
  metric_file = os.path.join(args.logs_dir, 'metrics.json')
  metrics = read_metrics(metric_file)[['Total_timesteps', 'nets/NumNets']]
  cfg = json.load(open(os.path.join(args.logs_dir, 'config.json')))
  cfg['eta']['Q_G_n_episodes_eval'] = cfg['eta']['Q_G_n_episodes']
  cfg['eta']['Q_G_n_episodes'] = args.store_N_episodes
  cfg['eta']['Q_G_rollout_length'] = args.rollout_length
  # Set eval_bias_metrics

  exp = Experiment(cfg)
  exp.eval_bias(args.logs_dir, args.checkpoint_prefix, metrics, logger)

if __name__=='__main__':
  main(sys.argv)