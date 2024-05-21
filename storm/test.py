from gym_super_mario_bros import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import env_wrapper
import gymnasium

# build a single environment
def build_single_env(env_name='SuperMarioBros-v3', rom_mode = "vanilla", seed=None):
    env = gym_super_mario_bros.make(env_name, rom_mode, render_mode='human') # full_action_space=False, frameskip=1
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = env_wrapper.SeedEnvWrapper(env, seed=None)
    env = env_wrapper.MaxLast2FrameSkipWrapper(env, skip=4)
    env = gymnasium.wrappers.ResizeObservation(env, shape=128)
    env = env_wrapper.LifeLossInfo(env)
    return env

# build a vectorized environment
# env = gymnasium.vector.AsyncVectorEnv(env_fns=[build_single_env])
env = build_single_env()

terminated = True
for step in range(5000):
    if terminated:
        observation = env.reset()
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())

env.close()