import pickle, os, random, torch
import numpy as np
from collections import defaultdict
import pandas as pd 
import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

def get_space_dim(space):
    t = type(space)
    if t is gym.spaces.Discrete:
        return space.n
    elif t is gym.spaces.Box:
        return space.shape[0]
    else:
        raise TypeError("Unknown space type:", t)

def load_object(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def save_object(obj, filename): 
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def make_dir(dir_path):
    """Create directory if it does not already exist."""
    try:
        os.makedirs(dir_path)
    except OSError:
        pass
    return dir_path

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed(seed)

class Logger(object):
    def __init__(self,):
        self.metrics = defaultdict(list)
        
    def log(self, **kwargs):
        for key, value in kwargs.items():
            self.metrics[key].append(value)
    
    def save(self, path):
        df = pd.DataFrame.from_dict(self.metrics)
        df.to_csv(f'{path}.csv')
        
def plot_reward(path, env_name):
    df = pd.read_csv(path)
    episodes = df['episodes'].to_numpy()
    rewards = df['ep_reward'].to_numpy()
    chunk_eps = []
    chunk_rewards = []
    for i in range(0, len(rewards), 1000):
        chunk = rewards[i:i+1000]
        chunk_eps.append(episodes[i + len(chunk) - 1])
        chunk_rewards.append(np.mean(chunk))
    plt.figure(figsize=(4.5,3))
    plt.plot(chunk_eps, chunk_rewards, linewidth=1.2)
    plt.xlabel('Episode', fontweight=10)
    plt.ylabel('Avg Reward per 1000 episodes', fontweight=10)
    plt.title(env_name, fontweight=12)
    plt.plot()

def create_grids_nn(policy, usable_ace=False):
    policy.eval()

    policy_action = {}

    for player_sum in range(12, 22):
        for dealer_card in range(1, 11):
            obs = np.array([player_sum, dealer_card, int(usable_ace)], dtype=np.float32)
            obs_t = torch.tensor(obs).unsqueeze(0).to(next(policy.parameters()).device)

            with torch.no_grad():
                action_dist = policy(obs_t)

            policy_action[(player_sum, dealer_card, usable_ace)] = int(action_dist.probs.argmax().item())

    player_grid, dealer_grid = np.meshgrid(
        np.arange(12, 22),
        np.arange(1, 11),
    )

    policy_matrix = np.apply_along_axis(
        lambda arr: policy_action[(arr[0], arr[1], usable_ace)],
        axis=2,
        arr=np.dstack([player_grid, dealer_grid]),
    )

    return policy_matrix


def create_hit_prob_grid(policy, usable_ace=False):
    """
    Build a grid with the model's probability of taking the Hit action.

    Args:
        policy: trained policy network.
        usable_ace (bool): whether to evaluate the grid with a usable ace.

    Returns:
        np.ndarray: probability of Hit for each (player_sum, dealer_card).
                    Shape matches the policy grid used in create_grids_nn.
    """
    policy.eval()
    hit_prob = {}

    for player_sum in range(12, 22):
        for dealer_card in range(1, 11):
            obs = np.array([player_sum, dealer_card, int(usable_ace)], dtype=np.float32)
            obs_t = torch.tensor(obs).unsqueeze(0).to(next(policy.parameters()).device)

            with torch.no_grad():
                action_dist = policy(obs_t)
                # In Gym Blackjack: action index 0 = stick, 1 = hit
                prob_hit = float(action_dist.probs[0, 1].item())

            hit_prob[(player_sum, dealer_card, usable_ace)] = prob_hit

    player_grid, dealer_grid = np.meshgrid(
        np.arange(12, 22),
        np.arange(1, 11),
    )

    prob_matrix = np.apply_along_axis(
        lambda arr: hit_prob[(arr[0], arr[1], usable_ace)],
        axis=2,
        arr=np.dstack([player_grid, dealer_grid]),
    )

    return prob_matrix


# Source: https://gymnasium.farama.org/v0.26.3/tutorials/blackjack_tutorial/
def create_plots(policy_grid, title: str):
    """Creates a plot showing the policy (which action to take in each state)."""
    # create a new figure with just the policy plot
    fig = plt.figure(figsize=(8, 6))
    fig.suptitle(title, fontsize=16)

    # plot the policy
    ax = sns.heatmap(policy_grid, linewidth=0, annot=True, cmap="Accent_r", cbar=False)
    ax.set_title(f"Policy: {title}")
    ax.set_xlabel("Player sum")
    ax.set_ylabel("Dealer showing")
    ax.set_xticklabels(range(12, 22))
    ax.set_yticklabels(["A"] + list(range(2, 11)), fontsize=12)

    # add a legend
    legend_elements = [
        Patch(facecolor="lightgreen", edgecolor="black", label="Hit"),
        Patch(facecolor="grey", edgecolor="black", label="Stick"),
    ]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.3, 1))
    return fig


def create_hit_prob_plot(prob_grid, title: str):
    """
    Plot a heatmap of the Hit action probability for each state.
    """
    fig = plt.figure(figsize=(8, 6))
    fig.suptitle(title, fontsize=16)

    ax = sns.heatmap(
        prob_grid,
        linewidth=0,
        cmap="YlGnBu",
        cbar=True,
        vmin=0.0,
        vmax=1.0,
        annot=True,
        fmt=".2f",
    )
    ax.set_title(f"Hit probability: {title}")
    ax.set_xlabel("Player sum")
    ax.set_ylabel("Dealer showing")
    ax.set_xticklabels(range(12, 22))
    ax.set_yticklabels(["A"] + list(range(2, 11)), fontsize=12)
    return fig


def load_trained_policy(model_path, state_space=3, action_space=2, hidden_size=16, device="cpu"):
    """
    Load a trained Policy from disk.

    Args:
        model_path (str or Path): path to the saved params .pt file.
        state_space (int): observation dimension (default 3 for Blackjack).
        action_space (int): action dimension (default 2 for Blackjack).
        hidden_size (int): hidden layer width used during training.
        device (str): torch device to map the weights to.

    Returns:
        Policy: initialized and loaded model in eval mode.
    """
    from grpo_blackjack_agent import Policy  # local import to avoid cycles

    policy = Policy(state_space, action_space, hidden_size=hidden_size)
    state_dict = torch.load(model_path, map_location=device)
    policy.load_state_dict(state_dict)
    policy.to(device)
    policy.eval()
    return policy
