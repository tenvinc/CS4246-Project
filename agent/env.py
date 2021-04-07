import gym
from gym.utils import seeding
from gym_grid_driving.envs.grid_driving import LaneSpec, Point

# def construct_task_env():
#     config = {'observation_type': 'tensor', 'agent_speed_range': [-3, -1], 'width': 50,
#               'lanes': [LaneSpec(cars=7, speed_range=[-2, -1]), 
#                         LaneSpec(cars=8, speed_range=[-2, -1]), 
#                         LaneSpec(cars=6, speed_range=[-1, -1]), 
#                         LaneSpec(cars=6, speed_range=[-3, -1]), 
#                         LaneSpec(cars=7, speed_range=[-2, -1]), 
#                         LaneSpec(cars=8, speed_range=[-2, -1]), 
#                         LaneSpec(cars=6, speed_range=[-3, -2]), 
#                         LaneSpec(cars=7, speed_range=[-1, -1]), 
#                         LaneSpec(cars=6, speed_range=[-2, -1]), 
#                         LaneSpec(cars=8, speed_range=[-2, -2])]
#             }
#     return gym.make('GridDriving-v0', **config)

# Easier env
def construct_task_env():
    config = {'observation_type': 'tensor', 'agent_speed_range': [-3, -1], 'stochasticity': 1.0, 'width': 50,
              'lanes': [LaneSpec(cars=3, speed_range=[-2, -1]), 
                        LaneSpec(cars=4, speed_range=[-2, -1]), 
                        LaneSpec(cars=3, speed_range=[-1, -1]), 
                        LaneSpec(cars=4, speed_range=[-3, -1]), 
                        LaneSpec(cars=4, speed_range=[-2, -1]), 
                        LaneSpec(cars=3, speed_range=[-2, -1]), 
                        LaneSpec(cars=2, speed_range=[-3, -2]), 
                        LaneSpec(cars=3, speed_range=[-1, -1]), 
                        LaneSpec(cars=2, speed_range=[-2, -1]), 
                        LaneSpec(cars=3, speed_range=[-2, -2])]
            }
    return gym.make('GridDriving-v0', **config)

# def construct_task_env():

#     config = {  'observation_type': 'tensor', 'agent_speed_range': [-3, -1], 'stochasticity': 1.0, 'width': 20,
#                 'lanes': [
#                     LaneSpec(cars=2, speed_range=[-2, -1]), 
#                     LaneSpec(cars=1, speed_range=[-2, -1]), 
#                     LaneSpec(cars=2, speed_range=[-1, -1]), 
#                     LaneSpec(cars=1, speed_range=[-3, -1]),
#                     LaneSpec(cars=1, speed_range=[-3, -1]),
#                     LaneSpec(cars=2, speed_range=[-2, -1]), 
#                     LaneSpec(cars=1, speed_range=[-2, -1]), 
#                     LaneSpec(cars=1, speed_range=[-1, -1]), 
#                     LaneSpec(cars=2, speed_range=[-3, -1]),
#                     LaneSpec(cars=1, speed_range=[-3, -1])
#                 ]}
#     return gym.make('GridDriving-v0', **config)