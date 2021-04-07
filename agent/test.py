from models import *
from agents import *

FAST_DOWNWARD_PATH = "/fast_downward/"

def create_agent(test_case_id, *args, **kwargs):
    '''
    Method that will be called to create your agent during testing.
    You can, for example, initialize different class of agent depending on test case.
    '''
    return DQfDAgent(test_case_id=test_case_id)

def test(agent, env, weights_path=None, silent=False):
    agent_init = {'fast_downward_path': FAST_DOWNWARD_PATH, 'agent_speed_range': (-3,-1), 'gamma' : 1}
    agent.initialize(**agent_init)
    num_actions = len(env.actions)

    if weights_path is not None:
        print("Loading preexisting weights....")
        agent.model.load_state_dict(torch.load(weights_path))

    agent.model.eval()    
    total_rewards = []
    for episode in range(600):
        state = env.reset()
        episode_rewards = 0.0
        for t in range(40):
            action = agent.step(state)
            next_state, reward, done, info = env.step(action)
            if not silent:
                env.render()
                import time
                time.sleep(1)
            episode_rewards += reward
            if done:
                break
            state = next_state
        total_rewards.append(episode_rewards)
    
    print(f"Avg reward after 600 runs: {np.mean(total_rewards)}")


if __name__ == "__main__":
    import sys
    import time
    import argparse
    from env import construct_task_env

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_path', default=None, type=str, help="Path to weights file")
    parser.add_argument('--silent', default=False, action="store_true", help="Whether to render or not")
    opt = parser.parse_args()

    test_env = construct_task_env()
    agent = create_agent(0)
    test(agent, test_env, weights_path=opt.weights_path, silent=opt.silent)
