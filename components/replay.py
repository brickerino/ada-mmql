import torch
import numpy as np
from collections import namedtuple, deque
from utils.helper import to_tensor


class InfiniteReplay(object):
  '''
  Infinite replay buffer to store experiences
  '''
  def __init__(self, keys=None):
    if keys is None:
      keys = ['action', 'reward', 'mask']
    self.keys = keys
    self.clear()

  def add(self, data):
    for k, v in data.items():
      if k not in self.keys:
        raise RuntimeError('Undefined key')
      getattr(self, k).append(v)

  def placeholder(self, data_size):
    for k in self.keys:
      v = getattr(self, k)
      if len(v) == 0:
        setattr(self, k, [None] * data_size)

  def clear(self):
    for key in self.keys:
      setattr(self, key, [])

  def get(self, keys, data_size):
    data = [getattr(self, k)[:data_size] for k in keys]
    data = map(lambda x: torch.stack(x), data)
    Entry = namedtuple('Entry', keys)
    return Entry(*list(data))


class FiniteReplay(object):
  '''
  Finite replay buffer to store experiences
  '''
  def __init__(self, memory_size, keys=None):
    if keys is None:
      keys = []
    self.keys = keys + ['action', 'reward', 'mask']
    self.memory_size = int(memory_size)
    self.clear()

  def clear(self):
    self.pos = 0
    self.full = False
    for key in self.keys:
      setattr(self, key, [None] * self.memory_size)

  def add(self, data):
    for k, v in data.items():
      if k not in self.keys:
        raise RuntimeError('Undefined key')
      getattr(self, k)[self.pos] = v
    self.pos = (self.pos + 1) % self.memory_size
    if self.pos == 0:
      self.full = True

  def get(self, keys, data_size, detach=False):
    data = [getattr(self, k)[:data_size] for k in keys]
    data = map(lambda x: torch.stack(x), data)
    if detach:
      data = map(lambda x: x.detach(), data)
    Entry = namedtuple('Entry', keys)
    return Entry(*list(data))

  def sample(self, keys, batch_size, detach=False):
    '''
    if self.size() < batch_size:
      return None
    '''
    idxs = np.random.randint(0, self.size(), size=batch_size)
    # data = [getattr(self, k)[idxs] for k in keys]
    data = [[getattr(self, k)[idx] for idx in idxs] for k in keys]
    data = map(lambda x: torch.stack(x), data)
    if detach:
      data = map(lambda x: x.detach(), data)
    Entry = namedtuple('Entry', keys)
    return Entry(*list(data))

  def is_empty(self):
    if self.pos == 0 and not self.full:
      return True
    else:
      return False
  
  def is_full(self):
    return self.full

  def size(self):
    if self.full:
      return self.memory_size
    else:
      return self.pos


class BiasControlReplayBuffer(object):
    def __init__(self, state_dim, action_dim, device, max_size=int(1e6), gamma=0.99, n_episodes_to_store=50, q_g_rollout_length=None):
        self.device = device
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.gamma = gamma
        self.n_episodes_to_store = n_episodes_to_store
        self.q_g_rollout_length = q_g_rollout_length

        self.transition_names = ('state', 'action', 'next_state', 'reward', 'mask',
                                 'ep_end', 'returns', 'ep_length', 'bs_multiplier')
        sizes = (state_dim, [action_dim], state_dim, [1], [1], [1], [1], [1], [1])
        for name, size in zip(self.transition_names, sizes):
            setattr(self, name, np.empty((max_size, *size)))

        self.last_episodes = deque()

    def add(self, state, action, next_state, reward, done, ep_end):
        values = (state, action, next_state, reward, 1. - done, ep_end)
        for name, value in zip(self.transition_names, values):
            getattr(self, name)[self.ptr] = value

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        if ep_end:
          res_idx = []
          running_return = 0
          running_tail_return = 0
          was_timelimit = self.mask[(self.ptr - 1) % self.max_size, 0] > 0.5
          for t in range(self.size):
            idx = (self.ptr - 1 - t) % self.max_size
            if t > 0 and (self.ep_end[idx, 0] > 0.5):
              break
            running_return = self.reward[idx] + self.gamma * running_return
            if t >= self.q_g_rollout_length:
              running_tail_return = self.reward[(idx + self.q_g_rollout_length) % self.max_size] + self.gamma * running_tail_return
            self.returns[idx] = running_return - np.power(self.gamma, self.q_g_rollout_length) * running_tail_return
            self.ep_length[idx] = t + 1

            is_enough_rollout = t + 1 > self.q_g_rollout_length
            if (was_timelimit and t > is_enough_rollout) or not was_timelimit:
              res_idx.append(idx)
              if is_enough_rollout:
                self.bs_multiplier[idx] = 1.0
              else:
                self.bs_multiplier[idx] = 0.0

          if len(res_idx) > 5:
              self.last_episodes.append(np.array(res_idx, dtype='int32'))
              if len(self.last_episodes) > self.n_episodes_to_store:
                  self.last_episodes.popleft()

    def sample(self, keys=None, batch_size=256):
        ind = np.random.randint(0, self.size, size=batch_size)
        if keys is None:
          names = self.transition_names[:-3]
        else:
          names = keys
        data = (torch.FloatTensor(getattr(self, name)[ind]).to(self.device) for name in names)
        Entry = namedtuple('Entry', keys)
        return Entry(*list(data))

    def gather_returns(self, n_per_episode, n_episodes):
        selected_idx = []
        if n_episodes < len(self.last_episodes):
            indices = np.random.choice(range(len(self.last_episodes)), replace=False, size=n_episodes)
            selected_episodes = [self.last_episodes[i] for i in indices]
        else:
            selected_episodes = self.last_episodes
        for ep_lst in selected_episodes:
            selected_idx.append(np.random.choice(ep_lst, replace=True, size=n_per_episode))
        selected_idx = np.concatenate(selected_idx)
        return self.get_returns_by_idx(selected_idx)

    def gather_returns_uniform(self, n_per_episode, n_episodes):
        if n_episodes < len(self.last_episodes):
            indices = np.random.choice(range(len(self.last_episodes)), replace=False, size=n_episodes)
            selected_episodes = [self.last_episodes[i] for i in indices]
        else:
            selected_episodes = self.last_episodes
        all_idx = np.concatenate(selected_episodes)
        selected_idx = np.random.choice(all_idx, replace=True, size=n_per_episode * len(self.last_episodes))
        return self.get_returns_by_idx(selected_idx)

    def get_returns_by_idx(self, selected_idx):
        result_states = self.state[selected_idx]
        result_actions = self.action[selected_idx]
        result_returns = self.returns[selected_idx]
        result_bs_multiplier = self.bs_multiplier[selected_idx]
        result_bs_states = self.state[(selected_idx + self.q_g_rollout_length) % self.max_size]
        return (torch.tensor(arr, dtype=torch.float32, device=self.device) for arr in
                [result_states, result_actions, result_returns, result_bs_states, result_bs_multiplier])