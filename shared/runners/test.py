import sys
import importlib
import numpy as np
import shared.agent_methods.methods as agmeth
agent_path = sys.argv[1]
print(agent_path)
rundef = importlib.import_module(agent_path)  # Run definition

"""Model Loading (If Applicable)"""
checkpoint_dir = rundef.loading_params['test_checkpoint_dir']
model_file = rundef.loading_params['test_model_file']

assert model_file is not None and checkpoint_dir is not None, "Need to define model to test"
print("Loading Model")
rundef.model.load_weights(checkpoint_dir + model_file)

env = rundef.environment
epsilon = lambda iter: 0.00
observation = env.reset()
render = False
actions = []

for iter in range(10000):
  if render:
    env.render()
  action = agmeth.get_action(rundef.model, env, [observation], epsilon, iter)
  actions.append(action)
  observation, reward, done, info = env.step(action)

  if done:
    observation = env.reset()
env.close()

if __name__ == '__main__':
  action_groupby = {action: 0 for action in range(env.action_space.n)}

  for action in actions:
      action_groupby[action] += 1

  print(action_groupby)

