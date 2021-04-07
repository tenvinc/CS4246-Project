from env import construct_task_env
import numpy as np
from agents import *
from datetime import datetime

def record():
    env = construct_task_env()
    num_episodes = 10

    combined_transition_list = []

    suffix = datetime.now().strftime("%Y_%m_%d_%H_%M")

    for eps in range(num_episodes):
        transition_list = []
        episodic_rewards = []
        state = env.reset()
        while True:
            env.render()
            """ Observations: pos of other cars (1), your pos (2), your goal (3), obstacles including occupancy trails (4) """
            
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

            next_state, reward, done, info = env.step(action)
            transition = Transition(state, action, reward, next_state, done)
            transition_list.append(transition)
            episodic_rewards.append(reward)
            if done:
                print("Reward obtained: ", reward)
                break
            state = next_state
        
        if np.sum(episodic_rewards) > 0.1:
            combined_transition_list = combined_transition_list + transition_list
            torch.save({'expert_transitions': combined_transition_list}, f"expert_dem_{suffix}.pt")
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Successful run")
        else:
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Failed run")


# Transition = collections.namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--load', action='store_true', default=False, help="True if want to load expert dem for sanity checks")
    opt = parser.parse_args()

    if opt.load:
        import os

        combined_transition_list = []
        # foldername = "easy1_10_20"
        foldername = "easy1_10_20"
        for root, dir, filenames in os.walk(foldername):
            for filename in filenames:
                if "expert_dem" in filename:
                    transitions = torch.load(os.path.join(foldername, filename))['expert_transitions']
                    episode = []
                    for t in transitions:
                        episode.append(t)
                        if t.done == 1:
                            combined_transition_list.append(episode)
                            episode = []
        print(len(combined_transition_list))
        torch.save({'expert_transitions': combined_transition_list}, f"expert_dem_combined_easy_10_20.pt")


        # expert = torch.load("expert_dem_2021_04_01_04_22.pt")
        # transitions = expert['expert_transitions']

        # for t in transitions:
        #     print(t.state.shape)
        #     print(t.done)
        #     print(t.next_state.shape)
        #     print(t.action)
        #     print(t.reward)
    else:
        record()

