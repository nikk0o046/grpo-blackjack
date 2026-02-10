from collections import deque
import os
import torch
import gymnasium as gym
import numpy as np 
from pathlib import Path
from dotenv import load_dotenv
import wandb

import utils as u
from grpo_blackjack_agent import Agent, Policy

load_dotenv()
wandb.login(key=os.environ["WANDB_APIKEY"])

work_dir = Path().cwd()/'results'


def setup(cfg_args={}, print_info=False):
    cfg = cfg_args
    
    # Setting library seeds
    if cfg["seed"] == None:
        seed = np.random.randint(low=1, high=1000)
    else:
        seed = cfg["seed"]
    
    print("Numpy/Torch/Random Seed: ", seed)
    u.set_seed(seed) # set seed

    # create folders if needed
    if cfg["save_model"]: 
        u.make_dir(work_dir/"model")
    if cfg["save_logging"]:
        u.make_dir(work_dir/"logging")
        L = u.Logger() # create a simple logger to record stats

    # use wandb to store stats; we aren't currently logging anything into wandb during testing (might be useful to
    # have the cfg.testing check here if someone forgets to set use_wandb=false)
    # create env
    env = gym.make(cfg["env_name"], 
                    max_episode_steps=cfg["max_episode_steps"],
                    render_mode='rgb_array')

    if cfg["save_video"]:
        # During testing, save every episode
        if cfg["testing"]:
            ep_trigger = 1
            video_path = work_dir/'video'/cfg["env_name"]/'test'
        # During training, save every 50th episode
        else:
            ep_trigger = 5000
            video_path = work_dir/'video'/cfg["env_name"]/'train'
            
        env = gym.wrappers.RecordVideo(env, video_path,
                                        episode_trigger=lambda x: x % ep_trigger == 0, # save video every 50 episode
                                        name_prefix=cfg["exp_name"], disable_logger=True)
    # Get dimensionalities of actions and observations
    action_space_dim = u.get_space_dim(env.action_space)
    observation_space_dim = 3

    # Instantiate agent and its policy
    policy = Policy(observation_space_dim, action_space_dim, cfg["hidden_size"])
    agent = Agent(policy, cfg["batch_size"], cfg["silent"], cfg["clip"])
    
    # Print some stuff
    if print_info:
        print("Configuration Settings:", cfg)
        print("Training device:", agent.train_device)
        print("Observation space dimensions:", observation_space_dim)
        print("Action space dimensions:", action_space_dim)
        print()
    return env, policy, agent, cfg
    
# Policy training function
def train_iteration(agent, env, episode_num, max_episode_steps=200, seed=None):
    # Run actual training        
    reward_sum, timesteps = 0, 0
    done = False

    # Reset the environment and observe the initial state
    observation, _  = env.reset()
    starting_state = observation

    while not done and timesteps < max_episode_steps:

        # Get action from the agent
        action, action_log_prob = agent.get_action(observation)
        previous_observation = observation

        # Perform the action on the environment, get new state and reward
        observation, reward, done, _, _ = env.step(action)

        # Store total episode reward (for blackjack this is just the final reward)
        reward_sum += reward

        # Store action's outcome (so that the agent can improve its policy)
        agent.store_outcome(episode_num, previous_observation, action, observation,
                reward, action_log_prob, done, starting_state)

        timesteps += 1

    # Return stats of training
    update_info = {'timesteps': timesteps, 'ep_reward': reward_sum}
    return update_info

# Training
def train(cfg_args={}):
    env, policy, agent, cfg = setup(cfg_args=cfg_args, print_info=True)

    run = wandb.init(project="grpo-blackjack", config=cfg)

    num_updates = 0

    if cfg["save_logging"]: 
        L = u.Logger() # create a simple logger to record stats

    # track rewards for the most recent X episodes
    reporting_interval = 5000
    recent_rewards = deque(maxlen=reporting_interval)

    for ep in range(cfg["train_episodes"]):
        train_info = train_iteration(agent, env, episode_num=ep, max_episode_steps=cfg["max_episode_steps"], seed=cfg["seed"])

        # Update the policy, if we have enough data
        if len(agent.states) > cfg["min_update_samples"]:
            state_stats = agent.update_policy()
            num_updates += 1
            state_stats = {str(k): v for k, v in state_stats.items()}
            run.log({"state_stats": state_stats}, step = ep)
        
        train_info.update({'num_updates': num_updates})
        train_info.update({'episodes': ep})
        recent_rewards.append(train_info['ep_reward'])

        if not cfg["silent"]:
            if ep % reporting_interval == 0:
                avg_reward = float(np.mean(recent_rewards))
                print(f"Episode {ep} finished. Average reward of last {reporting_interval} steps: {avg_reward}")

        if cfg["save_logging"]:
            L.log(**train_info)

    # Save the model
    if cfg["save_model"]:
        model_path = work_dir/'model'/f'{cfg["model_name"]}_params.pt'
        torch.save(policy.state_dict(), model_path)
        print("Model saved to", model_path)

    if cfg["save_logging"]:
        logging_path = work_dir/'logging'/f'{cfg["model_name"]}_{cfg["seed"]}'
        L.save(logging_path)

    print("------Training finished.------")
    
# Function to test a trained policy
    
def test(episodes, cfg_args={}):
    
    env, policy, agent, cfg  = setup(cfg_args=cfg_args, print_info=False)
    
    # Testing 
    model_path = work_dir/'model'/f'{cfg["model_name"]}_params.pt'
    print("Loading model from", model_path, "...")
 
    # load model
    state_dict = torch.load(model_path)
    policy.load_state_dict(state_dict)

    print("Testing...")
    total_test_reward, total_test_len = 0, 0
    for ep in range(episodes):
        done = False   
        observation, _ = env.reset()

        test_reward, test_len = 0, 0
        for t in range(cfg["max_episode_steps"]):
            # Similar to the training loop above -
            # get the action, act on the environment, save total reward
            # (evaluation=True makes the agent always return what it thinks to be
            # the best action - there is no exploration at this point)
            action, _ = agent.get_action(observation, evaluation=True)
            observation, reward, done, truncated, info = env.step(action)

            test_reward += reward
            test_len += 1
            if done:
                break
        total_test_reward += test_reward
        total_test_len += test_len
        if ep % 5000 == 0:
            print(f"Test episode number: {ep}")
    print("Average test reward:", total_test_reward/episodes, "episode length:", total_test_len/episodes)
    return total_test_reward/episodes