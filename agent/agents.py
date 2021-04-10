from models import *
from utils import *
from my_logging import *
import collections
import math
import random
import torch
import torch.nn.functional as F
import numpy as np

from core import *

try:
    from runner.abstracts import Agent
except:
    class Agent(object): pass

def reward_shaping(transition):
    if not transition.done[0]:
        add_reward = compute_crash_penalty(transition.done, transition.reward)
        # add_reward = compute_lane_potential(transition.state[1], transition.next_state[1], transition.state[2]) + \
        #                 compute_crash_penalty(transition.done, transition.reward)
        return add_reward
    return 0

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
        # self.model = RecurrentAtariDQN(INPUT_SHAPE, NUM_ACTIONS).to(device)
        self.model = AtariDQN(INPUT_SHAPE, NUM_ACTIONS).to(device)
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
        # state_action_values, hidden_state, cell_state = self.model(state, hidden_state, cell_state )
        state_action_values = self.model(state)
        best_action_value, best_action_idx = torch.max(state_action_values, dim=1)
        return int(best_action_idx.data)

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
        self.memory = ReplayBuffer()
        # self.memory = ReplayBuffer(reward_shaping_fn=reward_shaping)

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

        # self.target = RecurrentAtariDQN(INPUT_SHAPE, NUM_ACTIONS).to(device)
        self.target = AtariDQN(INPUT_SHAPE, NUM_ACTIONS).to(device)
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
        # hidden_state, cell_state = self.model.reset_hidden_states(states.size(0))
        state_action_values = self.model(states)
        # state_action_values, _, _ = self.model(states, hidden_state, cell_state)  # Only compute once
        # print(is_expert.size())
        # print("SAV: ", torch.min(state_action_values), " ", torch.max(state_action_values))
        J_E = compute_imitation_loss(state_action_values, actions, is_expert)
        J_DQ = compute_dqn_loss(state_action_values, self.target, states, actions, rewards, next_states, dones)
        loss = J_E * self.lambda_je + J_DQ * self.lambda_dq
        # loss = J_DQ * self.lambda_dq

        # if TFWriter.get_num_epochs() % 10 == 0:
        #     log_weights_biases(self.model)
        
        GenericLogger.add_scalar('J_E loss', J_E.item())
        GenericLogger.add_scalar('J_DQ loss', J_DQ.item())
        GenericLogger.add_scalar('Total loss', loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)  # To fix gradient explosion
        optimizer.step()
        return loss, J_E, J_DQ

    # def act(self, state, hidden_state, cell_state, epsilon=0.0, env=None, manual=False):
    def act(self, state, epsilon=0.0, env=None, manual=False):
        # if not isinstance(state, torch.FloatTensor):
        #     state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        '''
        Training version of step

        Input:
            * `state` (`torch.tensor` [batch_size, channel, height, width])
            * `epsilon` (`float`): the probability for epsilon-greedy

        Output: action (`Action` or `int`): representing the action to be taken.
                if action is of type `int`, it should be less than `self.num_actions`
        '''
        # assert state.size(0) == 1
        # # state_action_values, hidden_state, cell_state = self.model(state, hidden_state, cell_state )
        # state_action_values = self.model(state)
        # best_action_value, best_action_idx = torch.max(state_action_values, dim=1)

        best_action_idx = self.step(state)

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
                chosen_action_idx = int(torch.randint(0, self.num_actions, (1, )).data)
            else:
                chosen_action_idx = best_action_idx

        assert chosen_action_idx >= 0 and chosen_action_idx < self.num_actions

        # return int(chosen_action_idx.data), hidden_state, cell_state
        return chosen_action_idx

    def record_episode_train(self, episode):
        self.memory.push(episode)
        self.memory.push(her_sample(episode))
            



        

    

