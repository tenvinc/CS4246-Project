try:
    from runner.abstracts import Agent
except:
    class Agent(object): pass
import random

try:
    from models import *
except: pass
try:
    from .models import *
except: pass

import torch

INPUT_SHAPE = (4, 10, 50)
NUM_ACTIONS = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQNAgent(Agent):
    '''
    Agent with the relevant code for inference. The code for training is in
    training_agents.py
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
        self.model = AtariDQN(INPUT_SHAPE, NUM_ACTIONS).to(device)
        self.model.load_state_dict(torch.load("agent/weights/trained.pt"))
        self.num_actions = 5

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
        self.model.eval()

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