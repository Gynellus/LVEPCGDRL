import os
import csv
import argparse
import torch
import colorama
import gymnasium
import numpy as np
import cv2
from collections import deque
from tqdm import tqdm
from einops import rearrange

from utils import seed_np_torch, Logger, load_config
from replay_buffer import ReplayBuffer
import env_wrapper
import agents
from sub_models.functions_losses import symexp
from sub_models.world_models import WorldModel, MSELoss
import gym_super_mario_bros
from gym_super_mario_bros.joypad_space import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT


def process_visualize(img):
    img = img.astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (640, 640))
    return img


def build_single_env(env_name, image_size, seed=None):
    env = gym_super_mario_bros.make(env_name, rom_mode='vanilla', render_mode='human')
    if 'SuperMarioBros' in env_name:
        env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = env_wrapper.SeedEnvWrapper(env, seed=seed)
    env = env_wrapper.MaxLast2FrameSkipWrapper(env, skip=4)
    env = gymnasium.wrappers.ResizeObservation(env, shape=image_size)
    env = env_wrapper.LifeLossInfo(env)
    return env


def build_vec_env(env_name, image_size, num_envs, seed=None):
    def lambda_generator(env_name, image_size):
        return lambda: build_single_env(env_name, image_size, seed)
    env_fns = [lambda_generator(env_name, image_size) for _ in range(num_envs)]
    vec_env = gymnasium.vector.AsyncVectorEnv(env_fns=env_fns)
    return vec_env


def eval_episodes(num_episode, env_name, max_steps, num_envs, image_size,
                  world_model: WorldModel, agent: agents.ActorCriticAgent, results_file):
    world_model.eval()
    agent.eval()
    vec_env = build_vec_env(env_name, image_size, num_envs=num_envs)
    print("Current env: " + colorama.Fore.YELLOW + f"{env_name}" + colorama.Style.RESET_ALL)
    sum_reward = np.zeros(num_envs)
    current_obs, current_info = vec_env.reset()
    context_obs = deque(maxlen=16)
    context_action = deque(maxlen=16)
    final_rewards = []

    episode_count = 0

    try:
        while episode_count < num_episode:
            with torch.no_grad():
                if len(context_action) == 0:
                    action = vec_env.action_space.sample()
                else:
                    context_latent = world_model.encode_obs(torch.cat(list(context_obs), dim=1))
                    model_context_action = np.stack(list(context_action), axis=1)
                    model_context_action = torch.Tensor(model_context_action).cuda()
                    prior_flattened_sample, last_dist_feat = world_model.calc_last_dist_feat(context_latent, model_context_action)
                    action = agent.sample_as_env_action(
                        torch.cat([prior_flattened_sample, last_dist_feat], dim=-1),
                        greedy=False
                    )

            context_obs.append(rearrange(torch.Tensor(current_obs).cuda(), "B H W C -> B 1 C H W")/255)
            context_action.append(action)

            obs, reward, done, truncated, info = vec_env.step(action)

            done_flag = np.logical_or(done, truncated)
            if done_flag.any():
                for i in range(num_envs):
                    if done_flag[i]:
                        episode_count += 1
                        final_rewards.append(sum_reward[i])
                        # Log each episode's reward before resetting sum_reward
                        with open(results_file, "a", newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([env_name, episode_count, sum_reward[i]])
                        sum_reward[i] = 0

                        if episode_count == num_episode:
                            mean_reward = np.mean(final_rewards)
                            print("Mean reward: " + colorama.Fore.YELLOW + f"{mean_reward}" + colorama.Style.RESET_ALL)
                            return mean_reward

            sum_reward += reward
            current_obs = obs
            current_info = info

    except Exception as e:
        print(f"Exception encountered: {e}")
        with open(results_file, "a", newline='') as f:
            writer = csv.writer(f)
            for i, reward in enumerate(final_rewards):
                writer.writerow([env_name, i + 1, reward])
        raise e


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    parser = argparse.ArgumentParser()
    parser.add_argument("-config_path", type=str, required=True)
    parser.add_argument("-env_name", type=str, required=True)
    parser.add_argument("-run_name", type=str, required=True)
    args = parser.parse_args()
    conf = load_config(args.config_path)
    print(colorama.Fore.RED + str(args) + colorama.Style.RESET_ALL)

    seed_np_torch(seed=conf.BasicSettings.Seed)

    import trainmario as train
    dummy_env = build_single_env(args.env_name, conf.BasicSettings.ImageSize)
    action_dim = dummy_env.action_space.n
    world_model = train.build_world_model(conf, action_dim)
    agent = train.build_agent(conf, action_dim)
    root_path = f"ckpt/{args.run_name}"

    import glob
    pathes = glob.glob(f"{root_path}/world_model_*.pth")
    steps = [int(path.split("_")[-1].split(".")[0]) for path in pathes]
    steps.sort()
    steps = steps[-1:]
    results = []

    results_dir = "eval_result"
    os.makedirs(results_dir, exist_ok=True)
    base_results_file = os.path.join(results_dir, f"{args.run_name}.csv")
    results_file = base_results_file

    counter = 1
    while os.path.exists(results_file):
        results_file = base_results_file.replace(".csv", f"_{counter}.csv")
        counter += 1

    with open(results_file, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["env_name", "episode", "episode_reward"])

    for step in tqdm(steps):
        print(step)
        world_model.load_state_dict(torch.load(f"{root_path}/world_model_{step}.pth"))
        agent.load_state_dict(torch.load(f"{root_path}/agent_{step}.pth"))
        print("Loaded models")
        try:
            episode_avg_return = eval_episodes(
                num_episode=10,
                env_name=args.env_name,
                num_envs=1,
                max_steps=conf.JointTrainAgent.SampleMaxSteps,
                image_size=conf.BasicSettings.ImageSize,
                world_model=world_model,
                agent=agent,
                results_file=results_file
            )
            results.append([step, episode_avg_return])
        except Exception as e:
            print(f"Exception during evaluation at step {step}: {e}")

        with open(results_file, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([step, episode_avg_return])
