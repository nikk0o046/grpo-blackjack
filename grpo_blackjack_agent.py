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
    def __init__(
        self,
        policy,
        batch_size=64,
        silent=False,
        clip = 0.1
        ):
        self.train_device = "cuda"  # ""cuda" if torch.cuda.is_available() else "cpu"
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
        self.batch_size = batch_size
        # self.gamma = 0.98
        # self.tau = 0.95
        self.clip = clip
        self.epochs = 4
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

        state_stats = self.calculate_stats()

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
        
        return state_stats

    def calculate_stats(self) -> dict:
        # For all states I want: state visits, mean reward, mean advantage, hit probability
        # state visits, and reward should be available easily from compute advantages.
        # mean advantage would be for both actions? they are complements?

        # Total average reward across episodes (top-level metric)
        episode_final_rewards = [float(self.rewards[i].item()) for i in range(len(self.states)) if self.dones[i]]
        avg_reward = float(np.mean(episode_final_rewards))

        # State stats: possible states
        all_states = [(i, j, False) for i in range(12, 22) for j in range(1, 11)]
        all_states += [(i, j, True) for i in range(12, 22) for j in range(1, 11)]
        state_stats = {
            state: {
                "visits": 0,
                "reward_sum": 0.0,
                "actions": {
                    0: {"visits": 0, "reward_sum": 0.0, "adv_sum": 0.0},
                    1: {"visits": 0, "reward_sum": 0.0, "adv_sum": 0.0},
                },
            }
            for state in all_states
        }

        for i in range(len(self.states)):
            key = tuple(self.states[i].tolist())  # from tensor to hashable tuple e.g. (4, 12, True,)
            if key not in all_states:
                continue  # skips done states where no actions are possible

            action = int(self.actions[i].item())
            reward = float(self.rewards[i].item())
            advantage = float(self.advantages[i])

            # per-state aggregates
            state_stats[key]["visits"] += 1
            state_stats[key]["reward_sum"] += reward

            # per-action aggregates within the state
            state_stats[key]["actions"][action]["visits"] += 1
            state_stats[key]["actions"][action]["reward_sum"] += reward
            state_stats[key]["actions"][action]["adv_sum"] += advantage

            # store both action probabilities once per state
            if "action_probs" not in state_stats[key]:
                chosen_prob = torch.exp(self.action_log_probs[i]).item()  # p(chosen_action | state)
                if action == 0:
                    p0 = chosen_prob
                    p1 = 1.0 - chosen_prob
                else:
                    p1 = chosen_prob
                    p0 = 1.0 - chosen_prob

                state_stats[key]["action_probs"] = {0: p0, 1: p1}

        # turn sums into means
        for state, stats in state_stats.items():
            visits = stats["visits"]
            if visits > 0:
                stats["mean_reward"] = stats["reward_sum"] / visits
            else:
                stats["mean_reward"] = 0.0

            for action, a_stats in stats["actions"].items():
                a_visits = a_stats["visits"]
                if a_visits > 0:
                    a_stats["mean_reward"] = a_stats["reward_sum"] / a_visits
                    a_stats["mean_advantage"] = a_stats["adv_sum"] / a_visits
                else:
                    a_stats["mean_reward"] = 0.0
                    a_stats["mean_advantage"] = 0.0

        state_stats.update({"avg_reward": avg_reward})
        return state_stats

    def compute_advantages(self) -> list:
        # 1. group episode rewards by starting state
        # 2. calculate average reward and std of rewards for every group
        # 3. loop over all final steps and calculate advantage as episode (reward - avg reward) / std
        # 4. distribute this reward to all time steps
    
        # 1
        state_rewards = {}
        for i in range(len(self.states)):
            if self.dones[i]:
                key = tuple(self.starting_states[i].tolist())  # from tensor to hashable tuple e.g. (4, 12, True,)
                
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
                # could be optimized by calculating advantage just once for both actions
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
