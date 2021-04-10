"""
This file is used for training the DQN (DQfD)
"""
import neptune.new as neptune
import torch
import numpy as np
import os

from agents import *
from env import construct_task_env

# from utils import *
from core import *
from my_logging import *

nep_logger = GenericLogger(2000)

# FAST_DOWNWARD_PATH = "/fast_downward/"

"""
Hyperparameters used for the training
"""
# learning_rate = 1e-4 # learning rate reduced for training model after 5170 episodes
# max_episodes  = 1000000
# pretrain_epochs = 10000
# t_max         = 600
# print_interval= 20
# target_update = 10 # episode(s)
# train_steps   = 10

def create_agent(test_case_id, *args, **kwargs):
    '''
    Method that will be called to create your agent during testing.
    You can, for example, initialize different class of agent depending on test case.
    '''
    return DQfDAgent(test_case_id=test_case_id)


def train(agent, env, weights_path=None, tag=None, disable_il=False):
    agent_init = {'fast_downward_path': FAST_DOWNWARD_PATH, 'agent_speed_range': (-3,-1), 'gamma' : 1}
    agent.initialize(**agent_init)
    num_actions = len(env.actions)

    if disable_il:
        agent.init_train()
    else:
        # agent.init_train(expert_demo_path=os.path.join("agent", "expert_dem_combined_easy_10_20.pt"))
        agent.init_train(expert_demo_path=os.path.join("agent", "expert_dem_combined.pt"))
    rewards = []
    losses = {
        'J_E': [],
        'J_DQ': [],
        'Total': []
    }
    optimizer = torch.optim.Adam(agent.model.parameters(), lr=learning_rate, weight_decay=1e-4)

    GenericLogger.initialize_writer()
    GenericLogger.add_params({
        "weight_decay": 1e-4,
        "learning_rate": learning_rate,
        "target_update": target_update,
        "train_steps": train_steps,
        "epsilon_decay": epsilon_decay,
        "gamma": gamma,
        "max_epsilon": max_epsilon,
        "min_epsilon": min_epsilon,
        "batch_size": batch_size,
        "buffer_limit": buffer_limit,
        "min_buffer": min_buffer,
    })

    if opt.tag is not None:
        GenericLogger.add_tag(opt.tag)

    agent.model.train()
    epoch = 0
    if weights_path is None and not disable_il:
        print("Beginning Expert pretraining phase...")
        # Phase 1: Expert initialization phase

        for epoch in range(pretrain_epochs):
            loss, J_E, J_DQ = agent.optimize(optimizer, expert_phase=True)

            if epoch % (print_interval * 10) == 0 and epoch > 0:
                print("[Epoch {}]\tavg loss: : {:.6f}\tavg J_E loss: {:.6f}\tavg J_DQ loss: {:.6f}".format(epoch, loss.item(), J_E.item(), J_DQ.item()))
                # print("[Epoch {}]\tavg loss: : {:.6f}".format(epoch, np.mean(losses[-print_interval*5:])))

            # Update target network every once in a while
            if epoch % target_update == 0:
                agent.update_tgt_train()
            # if epoch % 2 == 0 and epoch > 0:
            #     break
    elif not weights_path is None:
        print("Loading preexisting weights....")
        agent.model.load_state_dict(torch.load(weights_path))
        agent.update_tgt_train()
    else:
        print("Imitation learning is disabled.")

    # torch.save(agent.target.state_dict(), "phase1.pt")

    manual = False
    # Phase 2: Exploration with some sampling of expert data
    print("Beginning exploration and training phase...")
    episode_lens = []
    prefix = f"last_model_{INPUT_SHAPE[1]}_{INPUT_SHAPE[2]}_"
    for episode in range(max_episodes):
        epsilon = compute_epsilon(episode)
        state = env.reset()
        episode_rewards = 0.0
        experiences = []
        # hidden_state, cell_state = agent.model.reset_hidden_states(1)
        # Try the epsiode
        for t in range(t_max):
            # action, hidden_state, cell_state = agent.act(state, hidden_state, cell_state, epsilon=epsilon, env=env, manual=manual)
            action = agent.act(state, epsilon=epsilon, env=env, manual=manual)
            next_state, reward, done, info = env.step(action)
            experiences.append(Transition(state, [action], [reward], next_state, [done]))
            episode_rewards += reward
            if done:
                if t > 50:
                    print(f"Weird value t: {t}")
                break
            state = next_state

            if len(experiences) > 50:
                print(f"Found anomaly {len(experiences)}")

        rewards.append(episode_rewards)

        # Record down all the stuff related to episodes
        episode_lens.append(len(experiences))

        # Store all episodes into the replay buffer using the agent's store method
        if len(experiences) > NUM_LOOKBACK_TIMESTEPS + 2:
            agent.record_episode_train(experiences)

        if agent.enough_memory_train():
            manual = False
            for i in range(train_steps):
                epoch += 1
                loss, J_E, J_DQ = agent.optimize(optimizer)
                losses['Total'].append(loss.item())
                losses['J_DQ'].append(J_DQ.item())
                losses['J_E'].append(J_E.item())

        if episode % print_interval == 0 and episode > 0:
            GenericLogger.add_scalar('Mean episode length', np.mean(episode_lens[-print_interval:]))
            # GenericLogger.add_scalar('Min episode length', np.min(episode_lens[-print_interval:]))
            # GenericLogger.add_scalar('Max episode length', np.sum(episode_lens[-print_interval:]))
            GenericLogger.add_scalar('Mean episode reward', np.mean(rewards[-print_interval:]))
            print("[Episode {}]\tavg rewards : {:.3f},\tavg loss: : {:.6f},\tavg J_E loss {:.6f},\tavg J_DQ loss {:.6f}, \
                    \tbuffer size : {},\t epsilon: {}".format(
                            episode, np.mean(rewards[-print_interval:]), np.mean(losses['Total'][-print_interval*10:]),
                            np.mean(losses['J_E'][-print_interval*10:]), np.mean(losses['J_DQ'][-print_interval*10:]), 
                            len(agent.memory), epsilon * 100))
            
            print("TOTAL: [Episode {}]\tavg rewards : {:.3f},\tavg loss: : {:.6f},\tavg J_E loss {:.6f},\tavg J_DQ loss {:.6f}, \
                    \tbuffer size : {},\t epsilon: {}".format(
                            episode, np.mean(rewards[-print_interval*10:]), np.mean(losses['Total'][-print_interval*10:]),
                            np.mean(losses['J_E'][-print_interval*10:]), np.mean(losses['J_DQ'][-print_interval*10:]), 
                            len(agent.memory), epsilon * 100))
            agent.memory.distribution()

        if episode % (target_update*2000) == 0:
            save_filename = prefix + str(episode)
            torch.save(agent.target.state_dict(), f"{save_filename}.pt")

        # Update target network every once in a while
        if episode % target_update == 0:
            agent.update_tgt_train()
            # print(">>>>>>>>>>> Saving target network to disc")



if __name__ == "__main__":
    import sys
    import time
    import argparse
    from env import construct_task_env

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_path', default=None, type=str, help="Path to weights file")
    parser.add_argument('--tag', default=None, type=str, help="Tag for neptune ai logging")
    parser.add_argument('--disable-il', default=False, action="store_true", help="Disable imitation learning")
    opt = parser.parse_args()

    test_env = construct_task_env()
    agent = create_agent(0)
    train(agent, test_env, weights_path=opt.weights_path, tag=opt.tag, disable_il=opt.disable_il)




    


