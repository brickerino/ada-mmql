# Adaptive Maxmin DQN

The Implementation is based on [Explorer](https://github.com/qlan3/Explorer).

## Implemented modifications

- Adaptive bias correction technique is implemented for Maxmin Deep Q-learning (MaxminDQN).

## Requirements

- Python (>=3.6)
- [PyTorch](https://pytorch.org/)
- [Gym && Gym Games](https://github.com/qlan3/gym-games): You may only install part of Gym (`classic_control, box2d`) by command `pip install 'gym[classic_control, box2d]'`.
- Optional: 
  - [Gym Atari](https://github.com/openai/gym/blob/master/docs/environments.md#atari)
  - [Gym Mujoco](https://github.com/openai/gym/blob/master/docs/environments.md#mujoco)
  - [PyBullet](https://pybullet.org/): `pip install pybullet`
- Others: Please check `requirements.txt`.


## Experiments

### Train && Test

All hyperparameters including parameters for grid search are stored in a configuration file in directory `configs`. To run an experiment, a configuration index is first used to generate a configuration dict corresponding to this specific configuration index. Then we run an experiment defined by this configuration dict. All results including log files and the model file are saved in directory `logs`. Please refer to the code for details.

For example, run the experiment with configuration file `minatar/AdaMMQL_adaptive_minatar.json` and configuration index `1`:

```python main.py --config_file ./configs/minatar/AdaMMQL_adaptive_minatar.json --config_idx 1```

The models are tested for one episode after every `test_per_episodes` training episodes which can be set in the configuration file.

### Bias evaluation
To evaluate the on policy bias we first need to run an experiment with `save_checkpoints` set to `true`. The checkpoints will be stored in `logs/exp_name/config_idx/`. To run the bias evaluation.

```python eval_bias_on_policy.py --logs_dir ./logs/exp_name/config_idx/ --store_N_episodes 300```

To reproduce the results from the paper run `run.sh`.

For more info refer to [Explorer](https://github.com/qlan3/Explorer).