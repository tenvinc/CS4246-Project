try:
    from runner.abstracts import Agent
except:
    class Agent(object): pass
import random


'''
An example to import a Python file.

Uncomment the following lines (both try-except statements) to import everything inside models.py
'''
try:
    from agents import *
except: pass
try: from .agents import *
except: pass



def create_agent(test_case_id, *args, **kwargs):
    '''
    Method that will be called to create your agent during testing.
    You can, for example, initialize different class of agent depending on test case.
    '''
    return DQNAgent(test_case_id=test_case_id)


if __name__ == '__main__':
    import sys
    import time
    from env import construct_task_env

    FAST_DOWNWARD_PATH = "/fast_downward/"

    def test(agent, env, runs=1000, t_max=100):
        rewards = []
        agent_init = {'fast_downward_path': FAST_DOWNWARD_PATH, 'agent_speed_range': (-3,-1), 'gamma' : 1}
        agent.initialize(**agent_init)
        for run in range(runs):
            state = env.reset()
            agent.reset(state)
            episode_rewards = 0.0
            for t in range(t_max):
                action = agent.step(state)   
                next_state, reward, done, info = env.step(action)
                full_state = {
                    'state': state, 'action': action, 'reward': reward, 'next_state': next_state, 
                    'done': done, 'info': info
                }
                agent.update(**full_state)
                state = next_state
                episode_rewards += reward
                if done:
                    break
            rewards.append(episode_rewards)
        avg_rewards = sum(rewards)/len(rewards)
        print("{} run(s) avg rewards : {:.1f}".format(runs, avg_rewards))
        return avg_rewards

    def timed_test(task):
        start_time = time.time()
        rewards = []
        for tc in task['testcases']:
            agent = create_agent(tc['id'])
            print("[{}]".format(tc['id']), end=' ')
            avg_rewards = test(agent, tc['env'], tc['runs'], tc['t_max'])
            rewards.append(avg_rewards)
        point = sum(rewards)/len(rewards)
        elapsed_time = time.time() - start_time

        print('Point:', point)

        for t, remarks in [(0.4, 'fast'), (0.6, 'safe'), (0.8, 'dangerous'), (1.0, 'time limit exceeded')]:
            if elapsed_time < task['time_limit'] * t:
                print("Local runtime: {} seconds --- {}".format(elapsed_time, remarks))
                print("WARNING: do note that this might not reflect the runtime on the server.")
                break

    def get_task():
        tcs = [('t2_tmax50', 50), ('t2_tmax40', 40)]
        return {
            'time_limit': 600,
            'testcases': [{ 'id': tc, 'env': construct_task_env(), 'runs': 300, 't_max': t_max } for tc, t_max in tcs]
        }

    task = get_task()
    timed_test(task)
