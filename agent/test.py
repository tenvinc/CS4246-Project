from models import *
from utils import *

INPUT_SHAPE = (4, 10, 20)
NUM_ACTIONS = 5
dqn = RecurrentAtariDQN(INPUT_SHAPE, NUM_ACTIONS)

record_weights_biases(dqn)

# print(dqn.features[0].weight)
# print(dqn.features[0].bias)
# print(dqn.features[2].weight)
# print(dqn.features[2].bias)

# print(dqn.recurrent.weight_ih_l0)
# print(dqn.recurrent.weight_hh_l0)
# print(dqn.recurrent.bias_ih_l0)
# print(dqn.recurrent.bias_hh_l0)

# print(dqn.layers[0].weight)
# print(dqn.layers[0].bias)