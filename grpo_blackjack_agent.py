import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space, hidden_size=16):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space

        self.fc1_a = torch.nn.Linear(state_space, hidden_size)
        self.fc2_a = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3_a = torch.nn.Linear(hidden_size, action_space)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight, 0, 1e-1)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x_a = self.fc1_a(x)
        x_a = F.relu(x_a)
        x_a = self.fc2_a(x_a)
        x_a = F.relu(x_a)
        x_a = self.fc3_a(x_a)

        action_probs = F.softmax(x_a, dim=-1)
        action_dist = Categorical(action_probs)

        return action_dist


class Agent(object):
    def __init__(self, policy, batch_size=64, silent=False):
        self.train_device = "cuda"  # ""cuda" if torch.cuda.is_available() else "cpu"
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
        self.batch_size = batch_size
        # self.gamma = 0.98
        # self.tau = 0.95
        self.clip = 0.2
        self.epochs = 12
        self.running_mean = None
        self.episode_nums = []
        self.states = []
        self.starting_states = []
        self.actions = []
        self.next_states = []
        self.rewards = []
        self.advantages = []
        self.dones = []
        self.action_log_probs = []
        self.silent = silent

    def update_policy(self):
        if not self.silent:
            print("Updating the policy...")

        self.states = torch.stack(self.states).to(self.train_device)
        self.actions = torch.stack(self.actions).squeeze().to(self.train_device)
        self.next_states = torch.stack(self.next_states).to(self.train_device)
        self.rewards = torch.stack(self.rewards).squeeze().to(self.train_device)
        self.advantages = torch.tensor(self.compute_advantages()).to(self.train_device)
        self.dones = torch.stack(self.dones).squeeze().to(self.train_device)
        self.action_log_probs = torch.stack(self.action_log_probs).squeeze().to(self.train_device)

        for _ in range(self.epochs):
            self.grpo_epoch()

        # Clear the replay buffer
        self.episode_nums = []
        self.states = []
        self.starting_states = []
        self.actions = []
        self.next_states = []
        self.rewards = []
        self.advantages = []
        self.dones = []
        self.action_log_probs = []
        if not self.silent:
            print("Updating finished!")

    def compute_advantages(self) -> list:
        # 1. group episodes by starting state
        # 2. calculate average reward and std of rewards for every group
        # 3. loop over all final steps and calculate advantage as episode (reward - avg reward) / std
        # 4. distribute this reward to all time steps
    
        # 1
        state_rewards = {}
        for i in range(len(self.states)):
            if self.dones[i]:
                key = tuple(self.starting_states[i].tolist())
                
                final_reward = self.rewards[i].item()
                if key not in state_rewards:
                    state_rewards.update({key: [final_reward]})
                else:
                    state_rewards[key].append(final_reward)
                
        # 2
        state_stats = {}
        for starting_state, rewards in state_rewards.items():
            mean = np.mean(rewards)
            std = np.std(rewards)
            state_stats[starting_state] = {"mean": mean, "std": std}

        # 3
        episode_advantages = {}
        for i in range(len(self.states)):
            if self.dones[i]:
                key = tuple(self.starting_states[i].tolist())
                # calculate advantage, avoid zero division error
                advantage = (self.rewards[i].item() - state_stats[key]["mean"]) / max(state_stats[key]["std"], 1e-8)
                episode_advantages.update({self.episode_nums[i]: advantage})
 
        # 4
        step_advantages = []
        for i in range(len(self.states)):
            step_advantage = episode_advantages[self.episode_nums[i]]
            step_advantages.append(step_advantage)

        return step_advantages

    def grpo_epoch(self):
        indices = list(range(len(self.states)))
        while len(indices) >= self.batch_size:
            # Sample a minibatch
            batch_indices = np.random.choice(indices, self.batch_size,
                    replace=False)

            # Do the update
            self.grpo_update(
                self.states[batch_indices],
                self.actions[batch_indices],
                self.action_log_probs[batch_indices],
                self.advantages[batch_indices],
            )

            # Drop the batch indices
            indices = [i for i in indices if i not in batch_indices]

    def grpo_update(self, states, actions, old_log_probs, advantages):
        action_dists = self.policy.forward(states)
        # using log probabilities for numerical stability
        new_log_probs = action_dists.log_prob(actions)
        ratio = torch.exp(new_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1-self.clip, 1+self.clip)
        
        policy_objective = -torch.min(ratio*advantages, clipped_ratio*advantages)
        loss = policy_objective.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_action(self, observation, evaluation=False):
        x = torch.tensor(observation).float().to(self.train_device)
        action_dist = self.policy.forward(x)
        if evaluation:
            action = action_dist.probs.argmax()
        else:
            action = action_dist.sample()
        aprob = action_dist.log_prob(action)
        action = action.item()
        return action, aprob

    def store_outcome(self, episode_num, state, action, next_state, reward, action_log_prob, done, starting_state):
        self.episode_nums.append(episode_num)
        self.states.append(torch.tensor(state).float())
        self.starting_states.append(torch.tensor(starting_state).float())
        self.actions.append(torch.Tensor([action]))
        self.action_log_probs.append(action_log_prob.detach())
        self.rewards.append(torch.Tensor([reward]).float())
        self.dones.append(torch.Tensor([done]))
        self.next_states.append(torch.tensor(next_state).float())
