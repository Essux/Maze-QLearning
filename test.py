from math import inf
from random import random, choice

from matplotlib import pyplot as plt
import numpy as np
import time

import gym
import gym_maze

# HYPERPARAMETERS
EPSILON = 0.9
DEFAULT_VALUE = 5
LEARNING_RATE = 0.5
EPISODES = 150
ITERATIONS = 5000
DECAY_EVERY_EPISODES = 25
REDUCE_ALPHA_BY = 0.9

# VISUALIZATION PARAMETERS
WINDOW_WIDTH = 10
SLEEP_TIME = 0.25
SHOW_EVERY_EPISODES = 25
SHOWN_ITERATIONS = 40

ACTION = ["N", "S", "E", "W"]
env =  gym.make("maze-random-10x10-v0")

out_file = open('test.out', 'w')

values = {}

def get_value(observation, i):
    global values
    observation = tuple(observation.tolist())
    if (observation, i) not in values:
        values[(observation, i)] = DEFAULT_VALUE
    return values[(observation, i)]

def get_best_action(observation):
    best_value = max([get_value(observation, i) for i in range(4)])
    best_actions = [i for i in range(4) if get_value(observation, i)==best_value]
    return choice(best_actions), best_value

def set_value(observation, action, new_value):
    global values
    observation = tuple(observation.tolist())
    values[(observation, action)] = new_value

def update_value(observation, action, next_observation, reward):
    best_action, best_value = get_best_action(next_observation)
    delta = LEARNING_RATE * (reward + best_value - get_value(observation, action))
    current_value = get_value(observation, action)
    set_value(observation, action, current_value+delta)

iteration_history = []

def show_progress():
    print('SHOWING PROGRESS')
    observation = env.reset()
    for t in range(SHOWN_ITERATIONS):
        env.render()
        r = random()
        best_action, best_value = get_best_action(observation)
        action = best_action

        for i in range(4):
            val = get_value(observation, i)
        
        next_observation, reward, done, info = env.step(action)

        observation = next_observation

        if done:
            break
        
        time.sleep(SLEEP_TIME)


for i_episode in range(EPISODES):
    if i_episode % DECAY_EVERY_EPISODES == 0:
        EPSILON *= REDUCE_ALPHA_BY
        print('New epsilon {:.2f}'.format(EPSILON))
    
    

    print('Episode starts')
    observation = env.reset()
    for t in range(ITERATIONS):
        if i_episode % SHOW_EVERY_EPISODES == 0:
            env.render()
        
        r = random()

        if r < EPSILON:
            action = env.action_space.sample()
        else:
            best_action, best_value = get_best_action(observation)
            action = best_action

        next_observation, reward, done, info = env.step(action)
        
        update_value(observation, action, next_observation, reward)

        observation = next_observation.copy()
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            import json
            out_values = {str(x): y for x, y in values.items()}
            print(json.dumps(out_values, indent=4, sort_keys=True), file=out_file)
            iteration_history.append(t)
            break

env.close()


cumsum_vec = np.cumsum(np.insert(iteration_history, 0, 0)) 
ma_vec = (cumsum_vec[WINDOW_WIDTH:] - cumsum_vec[:-WINDOW_WIDTH]) / WINDOW_WIDTH

plt.plot([i for i in range(len(iteration_history))], iteration_history)
plt.show()