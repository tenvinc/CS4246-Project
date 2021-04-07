from models import *
from utils import *
import collections
import math
import random
import torch
import torch.nn.functional as F
import numpy as np

try:
    from runner.abstracts import Agent
except:
    class Agent(object): pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_SHAPE = (4, 10, 50)
NUM_ACTIONS = 5

""" Hyperparameters """
gamma = 0.99
max_epsilon   = 0.01
min_epsilon   = 0.01
epsilon_decay = 500
batch_size    = 128
# buffer_limit  = 20000  # For non RNN
# min_buffer    = 1000
buffer_limit = 300
min_buffer    = 86 + 20

n_timesteps = NUM_LOOKBACK_TIMESTEPS

margin_loss = 0.8 # margin loss for imitation learning (0 if a = aE else positive)

Transition = collections.namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

""" Utility functions related to DQN """
def lane_potential(agent_pos, next_agent_pos, goal_pos):
    # linear potential (small reward 0.5)
    height = agent_pos.shape[0]
    agent_pos_y = agent_pos.nonzero()[0]
    next_agent_pos_y = next_agent_pos.nonzero()[0]
    goal_pos_y = goal_pos.nonzero()[0]

    return (agent_pos_y - next_agent_pos_y) /height * 5 

def compute_epsilon(episode):
    '''
    Compute epsilon used for epsilon-greedy exploration
    '''
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * math.exp(-1. * episode / epsilon_decay)
    return epsilon

def compute_dqn_loss(state_action_values, target, states, actions, rewards, next_states, dones):
    '''
    FILL ME : This function should compute the DQN loss function for a batch of experiences.

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
    actions = actions[:, -1, ...]
    rewards = rewards[:, -1, ...]
    non_final_mask = (1 - dones[:, -1, ...].squeeze(-1).long()).to(torch.bool)

    # Compute the Q(s,a) using current network, then use Q(s', a') from the target network to get max(Q(s', a'))
    # using bellman update
    state_action_values = state_action_values.gather(1, actions.long())
    next_state_values = torch.zeros((n_batch, 1), device=device)
    non_final_next_states = next_states[non_final_mask]
    hidden_state, cell_state = target.reset_hidden_states(non_final_next_states.size(0))
    target_sav, _, _ = target(non_final_next_states, hidden_state, cell_state)
    max_values, _ = target_sav.max(dim=1)

    next_state_values[non_final_mask] = max_values.unsqueeze(-1).detach()  # detach because target tensor no need backprop

    expected_state_action_values = rewards + (next_state_values * gamma)
    
    # TFWriter.add_scalar('average rewards (replay)', torch.mean(rewards).data, global_step=TFWriter.get_num_epochs())
    TFWriter.add_scalar('average predicted Q value', torch.mean(state_action_values).data, global_step=TFWriter.get_num_epochs())
    # TFWriter.add_scalar('min predicted Q value', torch.min(state_action_values).data, TFWriter.get_num_epochs())
    # TFWriter.add_scalar('max predicted Q value', torch.max(state_action_values).data, TFWriter.get_num_epochs())
    TFWriter.add_scalar('average target Q value', torch.mean(expected_state_action_values).data, global_step=TFWriter.get_num_epochs())
    # TFWriter.add_scalar('min target Q value', torch.min(expected_state_action_values).data, TFWriter.get_num_epochs())
    # TFWriter.add_scalar('max target Q value', torch.max(expected_state_action_values).data, TFWriter.get_num_epochs())
    
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
    return loss

def optimize(model, target, memory, optimizer):
    '''
    Optimize the model for a sampled batch with a length of `batch_size`
    '''
    batch = memory.sample(batch_size)
    loss = compute_loss(model, target, *batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def compute_imitation_loss(state_action_values, target_actions, is_expert):
    """ 
    Loss reflecting the difference between the expert demo vs the model's predictions 
    JE(Q)=max_a [Q(s,a)+ ℓ(a,aE)]−Q(s,aE)
    """
    # Only take the last timestep
    target_actions = target_actions[:, -1, ...]
    n_batches = state_action_values.size(0)
    # print("===========================================")
    # print(model.features[0].weight)
    # print("SAV", state_action_values)
    # print("act", target_actions)
    margin_losses = torch.full(state_action_values.size(), margin_loss).to(device)   # ℓ(a,aE)
    margin_losses.scatter_(1, target_actions.long(), 0)
    # print("marginloss", margin_losses)
    augmented_sav = state_action_values + margin_losses  # Q(s,a) + ℓ(a,aE)
    # print("augmented_sav", augmented_sav)
    max_asav, _ = torch.max(augmented_sav, dim=1)
    max_asav = max_asav.unsqueeze(-1)  # (n_batch, 1)
    # print("max_asav", max_asav)
    ref_sav = torch.gather(state_action_values, 1, target_actions.long())
    # print("ref_sav", ref_sav)
    # print("diff_sav", max_asav - ref_sav)
    diff = max_asav - ref_sav

    loss = torch.mean(diff * is_expert)   # Apply the is_expert mask (1 if expert and 0 otherwise)
    # loss = torch.sum(max_asav - ref_sav)  # TODO: Might want to consider averaging over batch size
    # print("loss", loss)

    # print("===========================")
    # print("loss " , max_asav - ref_sav)
    # print("avg loss", loss)

    return loss


class ReplayBuffer():
    """ Stores experiences in sets based on episode"""
    def __init__(self, buffer_limit=buffer_limit):
        self.expert_buffer = collections.deque()
        self.buffer = collections.deque(maxlen=buffer_limit)

    def push_expert(self, transition):
        self.expert_buffer.append(transition)

    def push(self, transition):
        self.buffer.append(transition)

    def sample_expert(self, batch_size):
        buffer_list = list(self.expert_buffer)
        return self._sample(batch_size, buffer_list)

    def sample(self, batch_size):
        buffer_list = list(self.expert_buffer + self.buffer)
        return self._sample(batch_size, buffer_list)

    def _sample(self, batch_size, buffer_list):
        ''' 
        Arguments:
            buffer_list: list of all possible transitions in a subset of the replay buffer to choose from
            batch_size: number of transitions to pick
        '''
        sample_idxes = np.random.randint(len(buffer_list), size=batch_size)

        sample = random.choices(buffer_list, k=batch_size)
        # if self.cached_sample is not None:
        #     sample = self.cached_sample
        # else:
        #     self.cached_sample = sample

        states = np.zeros([batch_size] + list(sample[0].state.shape))
        actions = np.zeros((batch_size, 1))
        rewards = np.zeros((batch_size, 1))
        next_states = np.zeros([batch_size] + list(sample[0].next_state.shape))
        dones = np.zeros((batch_size, 1))
        is_expert = np.zeros((batch_size, 1))

        for i, idx in enumerate(sample_idxes):
            states[i, ...] = buffer_list[idx].state
            actions[i, ...] = buffer_list[idx].action
            rewards[i, ...] = buffer_list[idx].reward
            next_states[i, ...] = buffer_list[idx].next_state
            dones[i, ...] = buffer_list[idx].done
            is_expert[i, ...] = (idx < self.expert_count())

        states = torch.from_numpy(states).float().to(device)
        actions = torch.from_numpy(actions).int().to(device)
        rewards = torch.from_numpy(rewards).float().to(device)
        next_states = torch.from_numpy(next_states).float().to(device)
        dones = torch.from_numpy(dones).float().to(device)
        is_expert = torch.from_numpy(is_expert).float().to(device)

        return (states, actions, rewards, next_states, dones, is_expert)

    # def sample(self, batch_size):
    #     buffer_list = list(self.buffer)
    #     sample = random.choices(buffer_list, k=batch_size)

    #     states = np.zeros([batch_size] + list(sample[0].state.shape))
    #     actions = np.zeros((batch_size, 1))
    #     rewards = np.zeros((batch_size, 1))
    #     next_states = np.zeros([batch_size] + list(sample[0].next_state.shape))
    #     dones = np.zeros((batch_size, 1))

    #     for i in range(batch_size):
    #         states[i, ...] = sample[i].state
    #         actions[i, ...] = sample[i].action
    #         rewards[i, ...] = sample[i].reward
    #         next_states[i, ...] = sample[i].next_state
    #         dones[i, ...] = sample[i].done

    #     states = torch.from_numpy(states).float().to(device)
    #     actions = torch.from_numpy(actions).int().to(device)
    #     rewards = torch.from_numpy(rewards).float().to(device)
    #     next_states = torch.from_numpy(next_states).float().to(device)
    #     dones = torch.from_numpy(dones).float().to(device)

    #     return (states, actions, rewards, next_states, dones)

    def __len__(self):
        '''
        Return the length of the replay buffer. (Inclusive of expert)
        '''
        return len(self.buffer) + len(self.expert_buffer)

    def expert_count(self):
        return len(self.expert_buffer)

class BatchReplayBuffer(ReplayBuffer):
    """Special version of the replay buffer that returns consecutive sequences 
    of transitions. Used for LSTM Q-networks"""
    def __init__(self, buffer_limit=buffer_limit):
        super(BatchReplayBuffer, self).__init__(buffer_limit=buffer_limit)

    def push(self, transitions):
        self.buffer.append(transitions)

    def push_expert(self, transitions):
        self.expert_buffer.append(transitions)

    def sample(self, batch_size, n_timesteps=NUM_LOOKBACK_TIMESTEPS):
        buffer_list = list(self.expert_buffer) + list(self.buffer)
        return self._sample(buffer_list, batch_size, n_timesteps)

    def sample_expert(self, batch_size, n_timesteps=NUM_LOOKBACK_TIMESTEPS):
        buffer_list = list(self.expert_buffer)
        return self._sample(buffer_list, batch_size, n_timesteps)

    def _sample(self, buffer_list, batch_size, n_timesteps):
        sample_idxes = np.random.randint(len(buffer_list), size=batch_size)

        states_batch = []
        actions_batch = []
        rewards_batch = []
        next_states_batch = []
        dones_batch = []
        is_expert_batch = []

        for idx in sample_idxes:
            episode = buffer_list[idx]
            start_idx = int(np.random.randint(len(episode)-n_timesteps+1))
            states = np.zeros([n_timesteps] + list(episode[0].state.shape))
            actions = np.zeros((n_timesteps, 1))
            rewards = np.zeros((n_timesteps, 1))
            next_states = np.zeros([n_timesteps] + list(episode[0].next_state.shape))
            dones = np.zeros((n_timesteps, 1))

            for i, t_idx in enumerate(range(start_idx, start_idx+n_timesteps)):
                # debug_print_transitions(episode[t_idx])
                states[i, ...] = episode[t_idx].state
                actions[i, ...] = episode[t_idx].action
                rewards[i, ...] = episode[t_idx].reward + \
                                    lane_potential(episode[t_idx].state[1, ...], episode[t_idx].next_state[1, ...], episode[t_idx].state[2, ...])
                next_states[i, ...] = episode[t_idx].next_state
                dones[i, ...] = episode[t_idx].done

            states_batch.append(states)
            actions_batch.append(actions)
            rewards_batch.append(rewards)
            next_states_batch.append(next_states)
            dones_batch.append(dones)
            is_expert_batch.append(idx < len(self.expert_buffer))

        states_batch =  torch.from_numpy(np.stack(states_batch)).float().to(device)
        actions_batch =  torch.from_numpy(np.stack(actions_batch)).float().to(device)
        rewards_batch =  torch.from_numpy(np.stack(rewards_batch)).float().to(device)
        next_states_batch =  torch.from_numpy(np.stack(next_states_batch)).float().to(device)
        dones_batch =  torch.from_numpy(np.stack(dones_batch)).float().to(device)
        is_expert_batch =  torch.from_numpy(np.stack(is_expert_batch)).float().to(device)

        return (states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch, is_expert_batch)


class DQfDAgent(Agent):
    '''
    Agent that uses Deep Q learning from expert demonstrations to learn faster than 
    the vanilla DQN
    '''
    def __init__(self, *args, **kwargs):
        '''
        [OPTIONAL]
        Initialize the agent with the `test_case_id` (string), which might be important
        if your agent is test case dependent.
        
        For example, you might want to load the appropriate neural networks weight 
        in this method.
        '''
        test_case_id = kwargs.get('test_case_id')
        '''
        # Uncomment to help debugging
        print('>>> __INIT__ >>>')
        print('test_case_id:', test_case_id)
        '''

        # Assume input shape is (4, 50, 10) with 5 actions
        self.model = RecurrentAtariDQN(INPUT_SHAPE, NUM_ACTIONS).to(device)
        self.num_actions = 5
        self.is_train = False

        self.lambda_je = 1 
        self.lambda_dq = 1

    def initialize(self, **kwargs):
        '''
        [OPTIONAL]
        Initialize the agent.

        Input:
        * `fast_downward_path` (string): the path to the fast downward solver
        * `agent_speed_range` (tuple(float, float)): the range of speed of the agent
        * `gamma` (float): discount factor used for the task

        Output:
        * None

        This function will be called once before the evaluation.
        '''
        fast_downward_path  = kwargs.get('fast_downward_path')
        agent_speed_range   = kwargs.get('agent_speed_range')
        gamma               = kwargs.get('gamma')
        '''
        # Uncomment to help debugging
        print('>>> INITIALIZE >>>')
        print('fast_downward_path:', fast_downward_path)
        print('agent_speed_range:', agent_speed_range)
        print('gamma:', gamma)
        '''

    def reset(self, state, *args, **kwargs):
        ''' 
        [OPTIONAL]
        Reset function of the agent which is used to reset the agent internal state to prepare for a new environement.
        As its name suggests, it will be called after every `env.reset`.
        
        Input:
        * `state`: tensor of dimension `[channel, height, width]`, with 
                   `channel=[cars, agent, finish_position, occupancy_trails]`

        Output:
        * None
        '''
        '''
        # Uncomment to help debugging
        print('>>> RESET >>>')
        print('state:', state)
        '''
        pass

    def step(self, state, *args, **kwargs):
        ''' 
        [REQUIRED]
        Step function of the agent which computes the mapping from state to action.
        As its name suggests, it will be called at every step.
        
        Input:
        * `state`: tensor of dimension `[channel, height, width]`, with 
                   `channel=[cars, agent, finish_position, occupancy_trails]`

        Output:
        * `action`: `int` representing the index of an action or instance of class `Action`.
                    In this example, we only return a random action
        '''
        '''
        # Uncomment to help debugging
        print('>>> STEP >>>')
        print('state:', state)
        '''
        return self.act(state)
        pass

    def update(self, *args, **kwargs):
        '''
        [OPTIONAL]
        Update function of the agent. This will be called every step after `env.step` is called.
        
        Input:
        * `state`: tensor of dimension `[channel, height, width]`, with 
                   `channel=[cars, agent, finish_position, occupancy_trails]`
        * `action` (`int` or `Action`): the executed action (given by the agent through `step` function)
        * `reward` (float): the reward for the `state`
        * `next_state` (same type as `state`): the next state after applying `action` to the `state`
        * `done` (`int`): whether the `action` induce terminal state `next_state`
        * `info` (dict): additional information (can mostly be disregarded)

        Output:
        * None

        This function might be useful if you want to have policy that is dependant to its past.
        '''
        state       = kwargs.get('state')
        action      = kwargs.get('action')
        reward      = kwargs.get('reward')
        next_state  = kwargs.get('next_state')
        done        = kwargs.get('done')
        info        = kwargs.get('info')
        '''
        # Uncomment to help debugging
        print('>>> UPDATE >>>')
        print('state:', state)
        print('action:', action)
        print('reward:', reward)
        print('next_state:', next_state)
        print('done:', done)
        print('info:', info)
        '''

    """ Training related apis """
    def init_train(self, expert_demo_path = None):
        self.is_train = True
        self.memory = BatchReplayBuffer()

        # Load in all the expert demonstrations
        if expert_demo_path is not None:
            expert_demo = torch.load(expert_demo_path)['expert_transitions']
            for episode in expert_demo:
                self.memory.push_expert(episode)
        print(self.memory.expert_count())

        def weight_init(m): 
            if isinstance(m, nn.Conv2d):
                print("Init Conv")
                # Improvement: 
                # He initialization used due to the ReLU activation. Reported to work better than xavier initialization
                # https://arxiv.org/pdf/1502.01852v1.pdf
                torch.nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')

            if isinstance(m, nn.Linear):
                print("Init Linear")
                torch.nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')

        self.model.apply(weight_init)
        # self.model.load_state_dict(torch.load("init.pt"))

        self.target = RecurrentAtariDQN(INPUT_SHAPE, NUM_ACTIONS).to(device)
        self.target.load_state_dict(self.model.state_dict())
        self.target.eval()

    # def record_train(self, transition):
    #     self.memory.push(transition)

    def enough_memory_train(self):
        return len(self.memory) > min_buffer

    def update_tgt_train(self):
        self.target.load_state_dict(self.model.state_dict())

    def optimize(self, optimizer, expert_phase=False):
        if expert_phase:
            batch = self.memory.sample_expert(batch_size)
        else:
            batch = self.memory.sample(batch_size)

        states, actions, rewards, next_states, dones, is_expert = batch
        batch = (states, actions, rewards, next_states, dones)
        hidden_state, cell_state = self.model.reset_hidden_states(states.size(0))
        state_action_values, _, _ = self.model(states, hidden_state, cell_state)  # Only compute once
        # print(is_expert.size())
        # print("SAV: ", torch.min(state_action_values), " ", torch.max(state_action_values))
        J_E = compute_imitation_loss(state_action_values, actions, is_expert)
        J_DQ = compute_dqn_loss(state_action_values, self.target, states, actions, rewards, next_states, dones)
        loss = J_E * self.lambda_je + J_DQ * self.lambda_dq
        # loss = J_DQ * self.lambda_dq

        # if TFWriter.get_num_epochs() % 10 == 0:
        #     log_weights_biases(self.model)

        TFWriter.add_scalar('J_E loss', J_E.item(), TFWriter.get_num_epochs())
        TFWriter.add_scalar('J_DQ loss', J_DQ.item(), TFWriter.get_num_epochs())
        TFWriter.add_scalar('Total loss', loss.item(), TFWriter.get_num_epochs())
        
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)  # To fix gradient explosion
        optimizer.step()
        return loss, J_E, J_DQ

    def act(self, state, hidden_state, cell_state, epsilon=0.0, env=None, manual=False):
        if not isinstance(state, torch.FloatTensor):
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        '''
        Training version of step

        Input:
            * `state` (`torch.tensor` [batch_size, channel, height, width])
            * `epsilon` (`float`): the probability for epsilon-greedy

        Output: action (`Action` or `int`): representing the action to be taken.
                if action is of type `int`, it should be less than `self.num_actions`
        '''
        assert state.size(0) == 1
        state_action_values, hidden_state, cell_state = self.model(state, hidden_state, cell_state )
        best_action_value, best_action_idx = torch.max(state_action_values, dim=1)

        chosen_action_idx = -1
        lottery = torch.rand(1)

        """ Observations: pos of other cars (1), your pos (2), your goal (3), obstacles including occupancy trails (4) """
        if manual:
            env.render()
            action = None
            while action is None:
                user_input = input("Enter your action: ")
                if user_input.lower() == "u":
                    print("up")
                    action = 0
                elif user_input.lower() == "d":
                    print("down")
                    action = 1
                elif user_input.lower() == "f1":
                    print("forward1")
                    action = 4
                elif user_input.lower() == "f2":
                    print("forward2")
                    action = 3
                elif user_input.lower() == "f3":
                    print("forward3")
                    action = 2
            chosen_action_idx = torch.tensor([action])
        else:
            if lottery < epsilon:
                chosen_action_idx = torch.randint(0, self.num_actions, best_action_idx.size())
            else:
                chosen_action_idx = best_action_idx

        assert chosen_action_idx >= 0 and chosen_action_idx < self.num_actions

        return int(chosen_action_idx.data), hidden_state, cell_state

    def record_episode_train(self, episode):
        """
        Improvement: (Not working right now. For stochastic environment, seems to be performing badly)
        Hindsight Experience Replay. To help the agent learn even when the goal is not reached, faster
        convergence for environments with sparse reward signals
        https://arxiv.org/pdf/1707.01495.pdf
        """
        def sample_goal(episode_len, idx, k=4):
            idxes = np.random.randint(idx, episode_len, k)
            return idxes


        k = 2   # Hyperparameter for the goal sampling rate 
        self.memory.push(episode)

        # if episode[-1].reward != 10:
        #     for i, ep in enumerate(episode):
        #         if i < 2 * n_timesteps: 
        #             continue
        #         goal_idxes = sample_goal(len(episode), i, k=k)
        #         for idx in goal_idxes:
        #             goal_state, goal_action, goal_reward, goal_next_state, goal_done = \
        #                 episode[idx].state, episode[idx].action, episode[idx].reward, episode[idx].next_state, episode[idx].done
        #             goal_state, goal_next_state = np.copy(goal_state), np.copy(goal_next_state)
        #             goal_state[2, ...] = goal_state[1, ...]
        #             goal_next_state[2, ...] = goal_state[1, ...]
        #             aug_episode = []
        #             for j in range(idx):
        #                 state, action, reward, next_state, done = episode[j].state, episode[j].action, episode[j].reward, episode[j].next_state, episode[j].done
        #                 state, next_state = np.copy(state), np.copy(next_state)
        #                 state[2, ...] = goal_state[2, ...]
        #                 next_state[2, ...] = goal_state[2, ...]
        #                 aug_episode.append(Transition(state, action, reward, next_state, done))
        #             goal_reward = [10]  # Make this the new goal
        #             goal_done = [True]
        #             aug_episode.append(Transition(goal_state, goal_action, goal_reward, goal_next_state, goal_done))
        #             # self.record_train(Transition(goal_state, goal_action, goal_reward, goal_next_state, goal_done))
        #             self.memory.push(aug_episode)
            


    """ Utility methods not tied to the object """

    # def optimize(model, target, memory, optimizer):
    #     '''
    #     Optimize the model for a sampled batch with a length of `batch_size`
    #     '''
    #     batch = memory.sample(batch_size)
    #     loss = compute_loss(model, target, *batch)
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #     return loss
        


        

    

