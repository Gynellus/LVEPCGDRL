from gym_super_mario_bros import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import stable_baselines3.common
import stable_baselines3.common.env_checker
import env_wrapper
import gymnasium
import stable_baselines3
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import os
import cv2

env = gym_super_mario_bros.make('SuperMarioBros-v3', render_mode='human')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = env_wrapper.SeedEnvWrapper(env, seed=None)
env = env_wrapper.MaxLast2FrameSkipWrapper(env, skip=4)
env = gymnasium.wrappers.ResizeObservation(env, shape=128)
env = env_wrapper.LifeLossInfo(env)
stable_baselines3.common.env_checker.check_env(env)

env = DummyVecEnv([lambda: env])

model_path = "./dqn_super_mario_model"

# Initialize the DQN model.
model = DQN("CnnPolicy", env, verbose=1, tensorboard_log="./dqn_super_mario_tensorboard/",
            buffer_size=10000, learning_starts=1000, batch_size=32, target_update_interval=500)

# Model load (if available)
# Ensure you have a compatible DQN model saved. If not, comment out this line at first run.
# model = DQN.load(model_path, env=env)

# Train the model. Adjust the total_timesteps based on your computational resources.
try:
    while True:
        model.learn(total_timesteps=20490, progress_bar=True, log_interval=1)
        model.save(model_path)
except KeyboardInterrupt:
    model.save(model_path)
print("Model saved.")


# # Evaluate the model
# mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
# print(f"Mean reward: {mean_reward}, Std Reward: {std_reward}")

# Load the model for testing
model = DQN.load(model_path, env=env)

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
