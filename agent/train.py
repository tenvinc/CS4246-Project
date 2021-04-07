"""
This file is used for training the DQN (DQfD)
"""

import torch
import numpy as np
import os

from agents import *
from env import construct_task_env

from utils import *

tfwriter = TFWriter()

FAST_DOWNWARD_PATH = "/fast_downward/"

"""
Hyperparameters used for the training
"""
learning_rate = 1e-3
max_episodes  = 10000
pretrain_epochs = 20000
t_max         = 600
print_interval= 20
target_update = 20 # episode(s)
train_steps   = 20

def create_agent(test_case_id, *args, **kwargs):
    '''
    Method that will be called to create your agent during testing.
    You can, for example, initialize different class of agent depending on test case.
    '''
    return DQfDAgent(test_case_id=test_case_id)


def train(agent, env, weights_path=None):
    agent_init = {'fast_downward_path': FAST_DOWNWARD_PATH, 'agent_speed_range': (-3,-1), 'gamma' : 1}
    agent.initialize(**agent_init)
    num_actions = len(env.actions)

    agent.init_train(expert_demo_path=os.path.join("agent", "expert_dem_combined.pt"))
    # agent.init_train()
    rewards = []
    losses = {
        'J_E': [],
        'J_DQ': [],
        'Total': []
    }
    optimizer = torch.optim.Adam(agent.model.parameters(), lr=learning_rate, weight_decay=1e-5)

    agent.model.train()
    epoch = 0
    if weights_path is None:
        print("Beginning Expert pretraining phase...")
        # Phase 1: Expert initialization phase
        TFWriter.initialize_writer()

        for epoch in range(pretrain_epochs):
            TFWriter.set_num_epochs(epoch)
            loss, J_E, J_DQ = agent.optimize(optimizer, expert_phase=True)

            if epoch % (print_interval * 10) == 0 and epoch > 0:
                print("[Epoch {}]\tavg loss: : {:.6f}\tavg J_E loss: {:.6f}\tavg J_DQ loss: {:.6f}".format(epoch, loss.item(), J_E.item(), J_DQ.item()))
                # print("[Epoch {}]\tavg loss: : {:.6f}".format(epoch, np.mean(losses[-print_interval*5:])))

            # Update target network every once in a while
            if epoch % target_update == 0:
                agent.update_tgt_train()
            # if epoch % 2 == 0 and epoch > 0:
            #     break
    else:
        print("Loading preexisting weights....")
        agent.model.load_state_dict(torch.load(weights_path))
        agent.update_tgt_train()

    # torch.save(agent.target.state_dict(), "phase1.pt")

    manual = False
    # Phase 2: Exploration with some sampling of expert data
    print("Beginning exploration and training phase...")
    episode_lens = []
    TFWriter.initialize_writer()
    save_filename = "last_model"
    for episode in range(max_episodes):
        epsilon = compute_epsilon(episode)
        state = env.reset()
        episode_rewards = 0.0
        experiences = []
        hidden_state, cell_state = agent.model.reset_hidden_states(1)
        # Try the epsiode
        for t in range(t_max):
            action, hidden_state, cell_state = agent.act(state, hidden_state, cell_state, epsilon=epsilon, env=env, manual=manual)
            next_state, reward, done, info = env.step(action)
            experiences.append(Transition(state, [action], [reward], next_state, [done]))
            episode_rewards += reward
            if done:
                break
            state = next_state
        rewards.append(episode_rewards)

        # Record down all the stuff related to episodes
        episode_lens.append(len(experiences))

        # Store all episodes into the replay buffer using the agent's store method
        if len(experiences) > NUM_LOOKBACK_TIMESTEPS + 2:
            agent.record_episode_train(experiences)

        if agent.enough_memory_train():
            manual = False
            for i in range(train_steps):
                TFWriter.set_num_epochs(epoch)
                epoch += 1
                loss, J_E, J_DQ = agent.optimize(optimizer)
                losses['Total'].append(loss.item())
                losses['J_DQ'].append(J_DQ.item())
                losses['J_E'].append(J_E.item())

        if episode % print_interval == 0 and episode > 0:
            TFWriter.add_scalar('Mean episode length', np.mean(episode_lens[-print_interval:]), epoch)
            TFWriter.add_scalar('Min episode length', np.min(episode_lens[-print_interval:]), epoch)
            TFWriter.add_scalar('Max episode length', np.sum(episode_lens[-print_interval:]), epoch)
            TFWriter.add_scalar('Mean episode reward', np.mean(rewards[-print_interval:]), epoch)
            print("[Episode {}]\tavg rewards : {:.3f},\tavg loss: : {:.6f},\tavg J_E loss {:.6f},\tavg J_DQ loss {:.6f}, \
                    \tbuffer size : {},\t epsilon: {}".format(
                            episode, np.mean(rewards[-print_interval:]), np.mean(losses['Total'][-print_interval*10:]),
                            np.mean(losses['J_E'][-print_interval*10:]), np.mean(losses['J_DQ'][-print_interval*10:]), 
                            len(agent.memory), epsilon * 100))
            
            print("TOTAL: [Episode {}]\tavg rewards : {:.3f},\tavg loss: : {:.6f},\tavg J_E loss {:.6f},\tavg J_DQ loss {:.6f}, \
                    \tbuffer size : {},\t epsilon: {}".format(
                            episode, np.mean(rewards[print_interval:]), np.mean(losses['Total'][print_interval*10:]),
                            np.mean(losses['J_E'][print_interval*10:]), np.mean(losses['J_DQ'][print_interval*10:]), 
                            len(agent.memory), epsilon * 100))

        if episode % 2000 == 0:
            save_filename = "last_model_" + str(episode)

        # Update target network every once in a while
        if episode % target_update == 0:
            agent.update_tgt_train()
            # print(">>>>>>>>>>> Saving target network to disc")
            torch.save(agent.target.state_dict(), f"{save_filename}.pt")



if __name__ == "__main__":
    import sys
    import time
    import argparse
    from env import construct_task_env

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_path', default=None, type=str, help="Path to weights file")
    opt = parser.parse_args()

    test_env = construct_task_env()
    agent = create_agent(0)
    train(agent, test_env, weights_path=opt.weights_path)

        # for run in range(runs):
        #     state = env.reset()
        #     agent.reset(state)
        #     episode_rewards = 0.0
        #     for t in range(t_max):
        #         action = agent.step(state)   
        #         next_state, reward, done, info = env.step(action)
        #         full_state = {
        #             'state': state, 'action': action, 'reward': reward, 'next_state': next_state, 
        #             'done': done, 'info': info
        #         }
        #         agent.update(**full_state)
        #         state = next_state
        #         episode_rewards += reward
        #         if done:
        #             break
        #     rewards.append(episode_rewards)
        # avg_rewards = sum(rewards)/len(rewards)
        # print("{} run(s) avg rewards : {:.1f}".format(runs, avg_rewards))
        # return avg_rewards
    




    


