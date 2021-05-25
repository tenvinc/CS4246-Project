import torch
import collections
import collections
import math
import random
import numpy as np

from itertools import compress
import torch.nn.functional as F

from my_logging import *

"""
Observations:
With HER, DQfD:
Without shaping: ~ 5 avg reward
With Shaping(Both): ~4.5 avg reward
With Shaping(Crash): ~5 avg reward
Exploration(0.3): ~5.7 avg reward

Conclusion: Reward shaping not useful

Curriculum (Easy first with exploration, then hard with little exploration)
Reward: ~7.3 avg reward

No Curriculum
Reward: Max around ~5.7 avg reward

No Dqfd:
No reward for a long time
Consistent nonzero reward: After 12700 episodes
Avg reward: ~6 after 24220

Dqfd with little exploration:
Consistent nonzero reward: After 4800 episodes
End result: ~ 5.7 avg reward

Dqfd with adjusting expert samples
End result: ~6. after 27260 episodes
"""

'''
Definitions
'''
FAST_DOWNWARD_PATH = "/fast_downward/"
Transition = collections.namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
INPUT_SHAPE = (4, 10, 50)
NUM_ACTIONS = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
Hyperparameters for training
'''
learning_rate = 1e-3  # learning rate reduced to 1e-4 for training model after 5170 episodes
max_episodes  = 20000000
pretrain_epochs = 5000
t_max         = 600
print_interval= 20
target_update = 20 # episode(s)
train_steps   = 10
gamma = 0.98
max_epsilon   = 1 # 1st round is 0.1, After 5170 episodes, reduce to 0.01
min_epsilon   = 0.01
epsilon_decay = 5000
batch_size    = 32
buffer_limit  = 15000  # For non RNN
min_buffer    = 1000
dqfd_margin_loss = 0.8
crash_penalty = -5   # penalty for crashing
total_potential_reward = 2   # potential based reward based on distance
her_constant = 4   # Constant k used in HER

expert_ratio_prob_decay = 500
min_expert_ratio_prob = 1
max_expert_ratio_prob = 1
success_failure_probability = 1  # Probability of success trials being added to the replay memory

'''
Hindsight Experience Replay
To help the agent learn even when the goal is not reached, 
faster convergence for environments with sparse reward signals
https://arxiv.org/pdf/1707.01495.pdf
Credits to https://medium.com/@jscriptcoder/yet-another-hindsight-experience-replay-refining-the-plan-3dcf8ede6f4a
article for the high level clarification for HER
'''
def her_sample(episode, k=her_constant):

    def sample_goal(episode_len, idx, k=her_constant):
        idxes = np.random.randint(idx, episode_len, k)
        return idxes

    def reward_fun(state):
        # 2nd channel is agent pos, 3rd channel is goal position
        agent_pos = state[1].nonzero()[0], state[1].nonzero()[1]
        goal_pos = state[2].nonzero()[0], state[2].nonzero()[1]
            
        return 10 if agent_pos == goal_pos else 0

    aug_episode = []
    for i, ep in enumerate(episode):
        # HER with future goal selection
        goal_idxes = sample_goal(len(episode), i, k=k)
        
        for idx in goal_idxes:
            goal_state, goal_action, goal_reward, goal_next_state, goal_done = \
                episode[idx].state, episode[idx].action, episode[idx].reward, episode[idx].next_state, episode[idx].done

            state, action, reward, next_state, done = episode[i].state, episode[i].action, episode[i].reward, episode[i].next_state, episode[i].done
            state, next_state = np.copy(state), np.copy(next_state)
            state[2, ...] = goal_state[1, ...]
            next_state[2, ...] = goal_state[1, ...]
            reward = reward_fun(next_state)
            if reward == 10:
                done = [True]

            # print(f"New: {state[2, ...].nonzero()}, {state[1, ...].nonzero()}, {next_state[2, ...].nonzero()}, {reward}")
            aug_episode.append(Transition(state, action, [reward], next_state, done))

    return aug_episode

'''
Loss functions
'''
def compute_dqn_loss(state_action_values, target, states, actions, rewards, next_states, dones, ret_TD_error=False):
    '''
    Input:
        * `model`       : model network to optimize
        * `target`      : target network
        * `states`      (`torch.tensor` [batch_size, channel, height, width])
        * `actions`     (`torch.tensor` [batch_size, 1])
        * `rewards`     (`torch.tensor` [batch_size, 1])
        * `next_states` (`torch.tensor` [batch_size, channel, height, width])
        * `dones`       (`torch.tensor` [batch_size, 1])

    Output: scalar representing the loss.

    References:
        * MSE Loss  : https://pytorch.org/docs/stable/nn.html#torch.nn.MSELoss
        * Huber Loss: https://pytorch.org/docs/stable/nn.html#torch.nn.SmoothL1Loss
    '''
    n_batch = state_action_values.size(0)
    non_final_mask = (1 - dones.squeeze(-1).long()).to(torch.bool)

    # Compute the Q(s,a) using current network, then use Q(s', a') from the target network to get max(Q(s', a'))
    # using bellman update
    state_action_values = state_action_values.gather(1, actions.long())
    next_state_values = torch.zeros((n_batch, 1), device=device)
    non_final_next_states = next_states[non_final_mask]
    target_sav = target(non_final_next_states)
    max_values, _ = target_sav.max(dim=1)
    next_state_values[non_final_mask] = max_values.unsqueeze(-1).detach()  # detach because target tensor no need backprop
    expected_state_action_values = rewards + (next_state_values * gamma)
    
    GenericLogger.add_scalar('average predicted Q value', torch.mean(state_action_values).data)
    GenericLogger.add_scalar('average target Q value', torch.mean(expected_state_action_values).data)
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    if ret_TD_error:
        return loss, (expected_state_action_values - state_action_values)
    else:
        return loss

def compute_imitation_loss(state_action_values, target_actions, is_expert):
    ''' 
    Loss reflecting the difference between the expert demo vs the model's predictions 
    JE(Q)=max_a [Q(s,a)+ ℓ(a,aE)]−Q(s,aE)
    As formulated in Deep Q Networks from Demonstrations,
    https://arxiv.org/abs/1704.03732

    Input:
        * `state_action_values`       (`torch.tensor` [batch_size, num_actions])
        * `target_actions             (`torch.tensor` [batch_size, 1])
        * `is_expert`                 (`torch.tensor` [batch_size, 1])
    Output:
        Imitation classification loss
    '''
    n_batches = state_action_values.size(0)
    margin_losses = torch.full(state_action_values.size(), dqfd_margin_loss).to(device)   # ℓ(a,aE)
    margin_losses.scatter_(1, target_actions.long(), 0)
    augmented_sav = state_action_values + margin_losses  # Q(s,a) + ℓ(a,aE)
    max_asav, _ = torch.max(augmented_sav, dim=1)
    max_asav = max_asav.unsqueeze(-1)  # (n_batch, 1)
    ref_sav = torch.gather(state_action_values, 1, target_actions.long())
    diff = max_asav - ref_sav
    loss = torch.mean(diff * is_expert)   # Apply the is_expert mask (1 if expert and 0 otherwise)
    return loss


'''
Reward shaping
'''
def compute_crash_penalty(done, reward):
    ''' 
    Penalty given when the agent crashes 
    '''
    if done[0] and reward[0] != 10:
        return crash_penalty
    return 0

def compute_lane_potential(agent_pos, next_agent_pos, goal_pos):
    '''
    Potential based reward signal based on distance (Linear)  TODO: Need to debug this. Seems a bit wrong

    Input:
        * `agent_pos`           (`torch.tensor` [batch_size, height, width])
        * `next_agent_pos`      (`torch.tensor` [batch_size, height, width])
        * `goal_pos`            (`torch.tensor` [batch_size, height, width])
    '''
    def manhattan_distance(pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    height, width = agent_pos.shape[0], agent_pos.shape[1]
    agent_pos = agent_pos.nonzero()
    next_agent_pos = next_agent_pos.nonzero()
    goal_pos = goal_pos.nonzero()
    return (manhattan_distance(agent_pos, goal_pos) - manhattan_distance(next_agent_pos, goal_pos)) / (height + width) * total_potential_reward


'''
Exploration
'''
def compute_epsilon(episode):
    '''
    Compute epsilon used for epsilon-greedy exploration
    '''
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * math.exp(-1. * episode / epsilon_decay)
    return epsilon

'''
Other Utility functions
'''
def compute_expert_ratio_prob(episode):
    '''
    COmpute the probability of selecting expert sample to be in the lottery
    '''
    expert_ratio_prob = min_expert_ratio_prob + (max_expert_ratio_prob - min_expert_ratio_prob) * math.exp(-1. * episode / expert_ratio_prob_decay)
    return expert_ratio_prob

'''
Replay Buffers
'''
class ReplayBuffer():
    """ Stores experiences to be used for DQN during training"""
    def __init__(self, buffer_limit=buffer_limit, reward_shaping_fn=None):
        self.expert_buffer = collections.deque()
        self.buffer = collections.deque(maxlen=buffer_limit)
        if reward_shaping_fn is None:
            # assume to be 0
            self.reward_shaping_fn = lambda x : 0
        else:
            self.reward_shaping_fn = reward_shaping_fn 

        # debug
        self.rewards = collections.deque(maxlen=buffer_limit)

    def push_expert(self, transitions):
        for t in transitions:
            new_t = Transition(t.state, [t.action], [t.reward], t.next_state, [t.done])
            self.expert_buffer.append(new_t)

    def push(self, transitions):
        for t in transitions:
            self.buffer.append(t)
            self.rewards.append(t.reward[0])

    def sample_expert(self, batch_size):
        buffer_list = list(self.expert_buffer)
        return self._sample(batch_size, buffer_list, len(buffer_list))

    def sample(self, batch_size, expert_ratio_prob=1.0):
        '''
        Inputs:
            expert_ratio_prob: Probability of an expert sample chosen to be in the set to be sampled
        '''
        expert_list = list(self.expert_buffer)
        expert_sel_idx = (np.random.uniform(size=len(expert_list)) <= expert_ratio_prob).nonzero()[0]
        filtered_expert_list = [expert_list[i] for i in expert_sel_idx]
        # print("expert", len(filtered_expert_list))
        buffer_list = filtered_expert_list + list(self.buffer)
        return self._sample(batch_size, buffer_list, len(filtered_expert_list))

    def distribution(self):
        print(f"Reward distribution {np.mean(list(self.rewards))}")

    def _sample(self, batch_size, buffer_list, expcount):
        ''' 
        Arguments:
            buffer_list: list of all possible transitions in a subset of the replay buffer to choose from
            batch_size: number of transitions to pick
        '''
        sample_idxes = np.random.randint(len(buffer_list), size=batch_size)
        sample = random.choices(buffer_list, k=batch_size)

        states = np.zeros([batch_size] + list(sample[0].state.shape))
        actions = np.zeros((batch_size, 1))
        rewards = np.zeros((batch_size, 1))
        next_states = np.zeros([batch_size] + list(sample[0].next_state.shape))
        dones = np.zeros((batch_size, 1))
        is_expert = np.zeros((batch_size, 1))

        for i, idx in enumerate(sample_idxes):
            states[i, ...] = buffer_list[idx].state
            actions[i, ...] = buffer_list[idx].action
            rewards[i, ...] = buffer_list[idx].reward[0] + self.reward_shaping_fn(buffer_list[idx])
            # if not buffer_list[idx].done:
            #     rewards[i, ...] = buffer_list[idx].reward + compute_lane_potential(buffer_list[idx].state[1],
            #         buffer_list[idx].next_state[1], buffer_list[idx].state[2]) + \
            #             compute_crash_penalty(buffer_list[idx].done, buffer_list[idx].reward)
            # else:
            #     rewards[i, ...] = buffer_list[idx].reward
            next_states[i, ...] = buffer_list[idx].next_state
            dones[i, ...] = buffer_list[idx].done
            is_expert[i, ...] = (idx < expcount)

        states = torch.from_numpy(states).float().to(device)
        actions = torch.from_numpy(actions).int().to(device)
        rewards = torch.from_numpy(rewards).float().to(device)
        next_states = torch.from_numpy(next_states).float().to(device)
        dones = torch.from_numpy(dones).float().to(device)
        is_expert = torch.from_numpy(is_expert).float().to(device)

        return (states, actions, rewards, next_states, dones, is_expert)
        
    def __len__(self):
        '''
        Return the length of the replay buffer. (Inclusive of expert)
        '''
        return len(self.buffer) + len(self.expert_buffer)

    def expert_count(self):
        return len(self.expert_buffer)
