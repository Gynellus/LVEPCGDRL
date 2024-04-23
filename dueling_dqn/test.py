from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import numpy as np
from time import sleep

env = gym_super_mario_bros.make('SuperMarioBros-v3', rom_mode = 'vanilla') # full_action_space=False, frameskip=1
env = JoypadSpace(env, COMPLEX_MOVEMENT)

done = True
for step in range(5000):
    if done:
        observation = env.reset()
    observation, reward, done, info = env.step(step % 3 if step < 200 else 1)
    print(reward)
    env.render()
    if step > 200:
        sleep(0.1)

env.close()