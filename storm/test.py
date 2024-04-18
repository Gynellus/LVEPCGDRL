from gym_super_mario_bros import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import env_wrapper
import gymnasium

env = gym_super_mario_bros.make('SuperMarioBros-v0', render_mode='rgb_array') # full_action_space=False, frameskip=1
env = JoypadSpace(env, COMPLEX_MOVEMENT)
env = env_wrapper.SeedEnvWrapper(env, seed=None)
env = env_wrapper.MaxLast2FrameSkipWrapper(env, skip=4)
env = gymnasium.wrappers.ResizeObservation(env, shape=128)
env = env_wrapper.LifeLossInfo(env)

terminated = True
for step in range(5000):
    if terminated:
        observation = env.reset()
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
    # env.render()

env.close()