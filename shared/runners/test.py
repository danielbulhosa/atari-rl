import sys
import importlib
import numpy as np
import shared.agent_methods.methods as agmeth
from gym import wrappers

agent_path = sys.argv[1]
print(agent_path)
rundef = importlib.import_module(agent_path)  # Run definition

"""Model Loading (If Applicable)"""
checkpoint_dir = rundef.loading_params['test_checkpoint_dir']
model_file =  rundef.loading_params['test_model_file']
model_dir = rundef.loading_params['test_model_dir']

assert model_file is not None and checkpoint_dir is not None, "Need to define model to test"

env = rundef.environment
epsilon = lambda iter: 0.00
gamma = 1
num_episodes = 100
observation = env.reset()
render = False
increment_model = False
save_video = False
checkpoint_start = 1

# By default only perfect cube recordings get saved. Callable passed here changes that.
env = env if not save_video else wrappers.Monitor(env, model_dir + '/videos/', video_callable=lambda episode_id: True)

total_reward = 0
rewards = []
first_episode = True

for episode in range(num_episodes):
    done = False
    step = 0
    if increment_model or first_episode:
        print("\nCheckpoint #{}".format(checkpoint_start))
        rundef.model.load_weights(checkpoint_dir + model_file.format(epoch=checkpoint_start))
        checkpoint_start += 1
        first_episode = False

    observation = env.reset()

    while not done:
        if render:
          env.render()

        action = agmeth.get_action(rundef.model, env, [observation], epsilon, iter)
        observation, reward, done, info = env.step(action)
        total_reward += gamma**step * reward
        step += 1

    print("Total reward: {}".format(total_reward))
    rewards.append(total_reward)
    total_reward = 0

print("Average Reward Over All Episodes: {}".format(np.mean(np.array(rewards))))
env.close()
