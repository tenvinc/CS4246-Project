# CS4246 AI Planning and Decision Making

## Introduction
This project uses a custom OpenAI Gym environment created by the CS4246 team at NUS, which can be found at https://github.com/cs4246/gym-grid-driving. The task of this project is to navigate the car agent from start point to the end point, while avoiding other cars. 

The environment used in this project is a 50 x 10 grid, which represents 10 lanes, each having a width of 50. The speed of cars besides the agent is stochastic and discrete, having a speed between 1 and 3 cells per timestep. More details to be found in the simulator environment.

## Design
The core of this project is Deep Q-Learning which uses Deep Learning models as a function approximator for the Q-values in the MDP, due to large state space. The reward system provided by the simulator is also sparse in nature, with the agent getting a reward of 10 when reaching the goal position and reward of 0 for everything else.

Traditional mechanisms used in Deep Q-learning Networks (DQN) such as experience replay buffers and double Q networks are deployed to make DQN more stable. Due to the sparse reward setting, exploration is incredibly slow and the network takes very long to learn with a vanilla DQN approach.

Thus, to make the learning more efficient, various methods such as Hindsight Experience Replay (HER), Deep Q-learning from Demonstrations (DQfD) and reward shaping are used.

### Reward Shaping
The reward is potential based reward, that rewards the agent for getting close to the goal and penalises the opposite action.

### Hindsight Experience Replay (HER)
Hindsight Experience Replay is based on the idea, that even though the agent may not reach the goal, it has still gathered more information about the environment. Then, why not treat the last position that it has reached as the goal and adjust the rewards accordingly? This will provide a reward signal to the agent even in failed trials. This is the intuition behind HER. More details can be found in the paper https://arxiv.org/abs/1707.01495

### Deep Q-Learning from Demonstrations (DQfD)
DQfD is based on the idea, that much of the initial exploration can be skipped if we provide a somewhat decent tutor for our agent to follow. The demonstration data from a human expert is first used in a supervised training scenario, to tune the weights in the DQN to follow the actions of the human expert. The demonstrations are then kept in a separate part of the replay buffer. During the training phase where the agent explores the environment, the agent uses both the demonstration data and its experience, sampling from both with a fixed ratio. This allows the agent to incorporate both the human expert's knowledge and the agent's knowledge into its decision network. More details can be found in the paper https://arxiv.org/abs/1704.03732

## Usage
To run this project, first set up the openai gym environment by downloading the docker build already prepared using `docker pull cs4246/base`. You would need to set up docker beforehand. Next, clone the project, then run `docker run -it --rm -v $PWD:/workspace cs4246/base <command>`. The command to run will be `python <script_path>` where <script_path> is the path to the python script you want to run.

For recording of the demonstration data:

`docker run -it --rm -v $PWD:/workspace cs4246/base python agent/record.py`

For training of the agent:

`docker run -it --rm -v $PWD:/workspace cs4246/base python agent/train.py`



