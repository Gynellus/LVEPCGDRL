from gym_super_mario_bros import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import stable_baselines3.common
import stable_baselines3.common.env_checker
import env_wrapper
import gymnasium
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import os
import cv2

env = gym_super_mario_bros.make('SuperMarioBros-v3', render_mode='human') # full_action_space=False, frameskip=1
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = env_wrapper.SeedEnvWrapper(env, seed=None)
env = env_wrapper.MaxLast2FrameSkipWrapper(env, skip=4)
env = gymnasium.wrappers.ResizeObservation(env, shape=128)
env = env_wrapper.LifeLossInfo(env)
stable_baselines3.common.env_checker.check_env(env)

env = DummyVecEnv([lambda: env])

model_path = "./ppo_super_mario_model"

# Initialize the PPO model.
model = PPO("CnnPolicy", env, verbose=1, ent_coef=0.01, tensorboard_log="./ppo_super_mario_tensorboard/")

# Model load
# model = PPO.load(model_path, env=env)

# # Train the model. You can adjust the total_timesteps based on how long you're willing to train and the computational resources you have.
try:
    while True:
        model.learn(total_timesteps=20490, progress_bar=True, log_interval=1)
        model.save(model_path)
except KeyboardInterrupt:
    model.save(model_path)
print("Model saved.")

# # Optionally, evaluate the model
# mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
# print(f"Mean reward: {mean_reward}, Std Reward: {std_reward}")

# Load the model
model = PPO.load(model_path, env=env)
print("model_loaded")
def process_visualize(img):
    img = img.astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (640, 640))
    return img

# Test the trained model
observation = env.reset()
for _ in range(1000):
    action, _states = model.predict(observation, deterministic=True)
    observation, reward, done, info = env.step(action)
    process_visualize(observation)

print("Testing finished.")
env.close()