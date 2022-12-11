from agents.VanillaDQN import *


class MaxminDQN(VanillaDQN):
  '''
  Implementation of Maxmin DQN with target network and replay buffer
  We can update all Q_nets for every update. However, this makes training really slow.
  Instead, we randomly choose one to update.
  '''
  def __init__(self, cfg):
    super().__init__(cfg)
    self.save_checkpoints = cfg["save_checkpoints"]
    self.checkpoint_freq = cfg["checkpoint_freq"]
    self.logs_dir = cfg['logs_dir']
    self.k = cfg['agent']['total_networks_num'] # number of target networks
    self.update_nets_num = cfg['agent']['update_networks_num']
    self.adaptive = cfg['agent']['adaptive']
    if self.adaptive:
      self.nets_to_use = cfg['eta']['init_d']
    else:
      self.nets_to_use = cfg['agent']['networks_num']
    self.min_num_nets_to_use = cfg['eta']['min_num_nets']
    self.update_d_iterval = cfg['eta']['update_d_interval']
    self.update_d_gamma = cfg['eta']['update_d_gamma']
    self.q_g_eval_interval = cfg['eta']['Q_G_eval_interval']
    self.q_g_n_per_episode = cfg['eta']['Q_G_n_per_episode']
    self.sampling_scheme = cfg['eta']['sampling_scheme']
    self.Q_G_delta = 0
    # tmp action size
    self.replay = BiasControlReplayBuffer(action_dim=1, state_dim=self.state_size, device=self.device, max_size=int(cfg['memory_size']),
                                          gamma=cfg['discount'], n_episodes_to_store=cfg['eta']['Q_G_n_episodes'],
                                          q_g_rollout_length=cfg['eta']['Q_G_rollout_length'])
    # Remove TimeLimit from env
    self.max_episode_length = cfg['env']['max_episode_steps']
    self.env = {
      'Train': make_env(cfg['env']['name'], max_episode_steps=int(self.max_episode_length), no_timelimit=True),
      'Test': make_env(cfg['env']['name'], max_episode_steps=int(self.max_episode_length), no_timelimit=True)
    }
    # Create k different: Q value network, Target Q value network and Optimizer
    self.Q_net = [None] * self.k
    self.Q_net_target = [None] * self.k
    self.optimizer = [None] * self.k
    for i in range(self.k):
      self.Q_net[i] = self.createNN(cfg['env']['input_type']).to(self.device)
      self.Q_net_target[i] = self.createNN(cfg['env']['input_type']).to(self.device)
      self.optimizer[i] = getattr(torch.optim, cfg['optimizer']['name'])(self.Q_net[i].parameters(), **cfg['optimizer']['kwargs'])
      # Load target Q value network
      self.Q_net_target[i].load_state_dict(self.Q_net[i].state_dict())
      self.Q_net_target[i].eval()

  def run_episode(self, mode, render):
    while True:
      self.action[mode] = self.get_action(mode)
      if render:
        self.env[mode].render()
      # Take a step
      self.next_state[mode], self.reward[mode], self.done[mode], _ = self.env[mode].step(self.action[mode])
      self.next_state[mode] = self.state_normalizer(self.next_state[mode])
      self.reward[mode] = self.reward_normalizer(self.reward[mode])
      self.episode_return[mode] += self.reward[mode]
      self.episode_step_count[mode] += 1
      ep_end = self.done[mode] or self.episode_step_count[mode] >= self.max_episode_length
      if mode == 'Train':
        # Save experience
        self.save_experience(ep_end)
        # Update policy
        if self.time_to_learn():
          step_metrics = self.learn()
        # Log metrics
        if (self.step_count + 1) > 10000 and (self.step_count + 1)% self.txt_log_freq == 0:
          res = self.eval_thresholds(self.replay, self.q_g_n_per_episode)
          step_metrics.update(res)
          step_metrics['Total_timesteps'] = self.step_count + 1
          step_metrics['Train_Average_Return'] = self.rolling_score['Train']
          self.eval_policy()
          step_metrics['Evaluation_returns'] = self.rolling_score['Test']
          self.txt_logger.log(step_metrics)
        if self.save_checkpoints and (self.step_count + 1) % self.checkpoint_freq == 0:
          trainer_save_name = f'{self.logs_dir}/iter_{self.step_count + 1}'
          self.save_model(trainer_save_name)
        self.step_count += 1
      # Update state
      self.state[mode] = self.next_state[mode]
      if self.done[mode] or self.episode_step_count[mode] >= self.max_episode_length:
        break
    # End of one episode
    self.save_episode_result(mode)
    # Update evaluation statistics
    if mode == 'Train':
      self.txt_logger.update_evaluation_statistics(self.episode_step_count[mode], self.episode_return[mode])
    # Reset environment
    self.reset_game(mode)
    if mode == 'Train':
      self.episode_count += 1

  def run_eval_episode(self):
    mode = 'Train'
    # Reset environment
    self.reset_game(mode)
    while True:
      self.action[mode] = self.get_action(mode)
      # Take a step
      self.next_state[mode], self.reward[mode], self.done[mode], _ = self.env[mode].step(self.action[mode])
      self.next_state[mode] = self.state_normalizer(self.next_state[mode])
      self.reward[mode] = self.reward_normalizer(self.reward[mode])
      self.episode_step_count[mode] += 1
      ep_end = self.done[mode] or self.episode_step_count[mode] >= self.max_episode_length
      # Save experience
      self.save_experience(ep_end)
      # Update state
      self.state[mode] = self.next_state[mode]
      if self.done[mode] or self.episode_step_count[mode] >= self.max_episode_length:
        break

  def save_experience(self, ep_end):
    mode = 'Train'
    self.replay.add(self.state[mode], self.action[mode], self.next_state[mode],
                    self.reward[mode], self.done[mode], ep_end)

  def learn(self):
    # Choose update_nets_num to update
    self.update_Q_net_indices = np.random.choice(self.k, self.update_nets_num, replace=False)
    # Choose a Q_net from active networks to update
    #active_update_index = np.random.choice(list(range(self.nets_to_use)))
    # Choose Q netwroks from inactive networks to match the update rate
    #inactive_update_indices = self.nets_to_use + np.argwhere(np.random.uniform(0, 1, self.k - self.nets_to_use) < (1 / self.nets_to_use))
    # Compose a list of indices to update
    #self.update_Q_net_indices = [active_update_index] + inactive_update_indices.squeeze(1).tolist()
    step_metrics = super().learn()
    # Update target network
    if (self.step_count // self.cfg['network_update_frequency']) % self.cfg['target_network_update_frequency'] == 0:
      for i in range(self.k):
        self.Q_net_target[i].load_state_dict(self.Q_net[i].state_dict())
    step_metrics['nets/NumUpdatedNets'] = len(self.update_Q_net_indices)
    step_metrics['nets/NumNets'] = self.nets_to_use
    if self.adaptive:
      # Update Q_G_delta
      if self.step_count > 10000 and self.step_count % self.q_g_eval_interval == 0:
        self.eval_thresholds(self.replay, self.q_g_n_per_episode)
      # Update Number of Q networks
      if self.step_count > 10000 and self.step_count % self.update_d_iterval == 0:
        self.update_d()
    return step_metrics
  
  def compute_q_target(self, batch):
    with torch.no_grad():
      nets_indices = np.random.choice(self.k, self.nets_to_use, replace=False)
      q_min, _ = torch.min(torch.stack([self.Q_net_target[i](batch.next_state) for i in nets_indices], 1), 1)
      q_next = q_min.max(1, keepdim=True)[0]
      q_target = batch.reward + self.discount * q_next * batch.mask
    return q_target.squeeze()
  
  def get_action_selection_q_values(self, state):
    nets_indices = np.random.choice(self.k, self.nets_to_use, replace=False)
    q_min, _ = torch.min(torch.stack([self.Q_net[i](state) for i in nets_indices], 1), 1)
    q_min = to_numpy(q_min).flatten()
    return q_min

  def update_d(self):
    if self.Q_G_delta < 0:
      self.nets_to_use = max(self.nets_to_use - 1, self.min_num_nets_to_use)
    if self.Q_G_delta > 0:
      self.nets_to_use = min(self.nets_to_use + 1, self.k)

  def eval_thresholds_by_type(self, replay_buffer, n_per_episode, sampling_scheme, n_episodes):
    if sampling_scheme == 'uniform':
      states, actions, returns, bs_states, bs_multiplier = replay_buffer.gather_returns_uniform(n_per_episode, n_episodes)
    elif sampling_scheme == 'episodes':
      states, actions, returns, bs_states, bs_multiplier= replay_buffer.gather_returns(n_per_episode, n_episodes)
    else:
      raise Exception("No such sampling scheme")
    with torch.no_grad():
      tail_nets_indices = np.random.choice(self.k, self.nets_to_use, replace=False)
      tail_q_min, _ = torch.min(torch.stack([self.Q_net_target[i](bs_states) for i in tail_nets_indices], 1), 1)
      tail_q, _ = torch.max(tail_q_min, 1) # Select Q values with respect to optimal policy Q*
      tail_q = tail_q * bs_multiplier * np.power(replay_buffer.gamma, replay_buffer.q_g_rollout_length)
      nets_indices = np.random.choice(self.k, self.nets_to_use, replace=False)
      q_min, _ = torch.min(torch.stack([self.Q_net[i](states) for i in nets_indices], 1), 1)
      q = q_min.gather(1, actions.long()).squeeze()
    res = {f'LastReplay_{sampling_scheme}/Q_value': q.mean().__float__(),
           f'LastReplay_{sampling_scheme}/Returns': (returns + tail_q).mean().__float__()}
    return res
  
  def eval_thresholds(self, replay_buffer, n_per_episode, n_episodes=None):
    if n_episodes is None:
      n_episodes = len(self.replay.last_episodes)
    res_uniform = self.eval_thresholds_by_type(replay_buffer, n_per_episode, 'uniform', n_episodes)
    res_episodes = self.eval_thresholds_by_type(replay_buffer, n_per_episode, 'episodes', n_episodes)
    res = dict()
    res.update(res_uniform)
    res.update(res_episodes)
    last_Q_G_delta = res[f'LastReplay_{self.sampling_scheme}/Q_value'] - \
              res[f'LastReplay_{self.sampling_scheme}/Returns']
    self.Q_G_delta = self.Q_G_delta * self.update_d_gamma + last_Q_G_delta * (1 - self.update_d_gamma)
    return res
