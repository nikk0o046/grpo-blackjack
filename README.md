# grpo-blackjack
Implementation of DeepSeek's GRPO algorithm for Blackjack

This readme file works both as my notes for the project and as an explanation for anyone interested in the implementation.

## Goal of this project

The primary goal is hands-on learning of the Group Relative Policy Optimization (GRPO) algorithm used in the famous DeepSeek model.

## Background

A critical element of the GRPO algorithm is that even from your starting point you need a way to gain rewards in order to learn. The authors use a pre-trained LLM as a reference policy which is already capable of solving some of their training problems some of the time. I want to go for a small problem just to check if I can get this algorithm to converge to a good solution and I don't have a pre-trained model to start with or a lot of compute. In my opinion Blackjack would be a good problem for this use case as even with a random policy it will win some of the time thus enabling it to start learning. Gymnasium conveniently has a RL environment for Blackjack which I make use of.

In Deepseek's implementation they regulate the model to not drift too far away from the reference policy using KL Divergence term. Since I start with a random policy I can drop the whole KL divergence term in the formula as there is no need to keep it regulated towards a pre-trained model.

<p align="center">
  <img src="images/grpo_objective_deepseek.png" width="45%">
</p>
<p align="center">
  <sub><em> DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning: https://arxiv.org/pdf/2501.12948 </em></sub>
</p>

The remaining algorithm is just a modification from the popular PPO algorithm to reduce the resource consumption. The difference really boils down to the advantage term *A*<sub>i</sub>. While the PPO algorithm needs to have another huge model (in case of LLMs) to estimate the value of each action outcome, GRPO foregoes this altogether and just makes groups out of similar trajectories and calculates the advantage as the reward relative to the group mean.

One thing to note is that this surely is NOT the optimal solution for blackjack - GRPO algorithm is in many ways wasteful, but for some problems it is one of the only ones that works without supervision. Therefore it is not as relevant to benchmark it against the speed of some of the other algorithms, but instead focus on if it can reach the same end result through a different reward scheme.

Since GRPO is newer and not a widely used solution it makes sense to first run tests with a known solution to the problem to make sure everything in the environment etc. is running smoothly without bugs. I took an existing PPO implementation from a university RL course for the Cartpole environment, modified it for Blackjack and it converged to a reasonable policy.

---

How to modify the training loop?

It is not possible to run the same starting hand for multiple runs without setting a seed, which would make the hands deterministic and ruin the idea of the advantage averaging out. Therefore the plan is to simulate many hands, e.g. 10k and then group them by the initial state for advantage calculation. In DeepSeek this would be the equivalent of them having some prompt to start with (e.g. a math question) and then they sample multiple responses. In the case of blackjack it would also be possible to group them by state, not just starting state. This could actually be even better as it increases samples in some of the states, but it would not be in the spirit of DeepSeek's GRPO implementation where a core idea is to use the same reward and advantage for every timestep in an episode.

Therefore for implementation I need:
1) Store episode numbers
2) Store initial states
3) Count advantages and distribute them to steps
4) Make sure updates are only done after episodes not within episodes like sometimes in PPO


## Initial Results

Overall the models seems to converge to a reasonable policy, though not the optimal one. Practically the differences in performance are quite small: With this policy the reward is about -0.050 when the optimal policy yields -0.047, according to https://chisness.github.io/2020-09-21/monte-carlo-rl-and-blackjack. Their optimal policy, found with a Monte Carlo method after 10 million iterations is almost identical (there is a single difference) to the optimal policy proposed by Sutton and Barto in their RL book. Despite mean rewards being close to each other there are multiple differences in the policies.

<p align="center">
  <img src="images/run_1_actions_ace.png" width="45%">
  <img src="images/run_1_actions_no_ace.png" width="45%">
</p>

For example, when it comes to the no usable ace scenario the policy is playing it too safe when the dealer is showing an ace. When taking a deeper look, we can see that at least in those states the propability of hitting is not zero, unlike in states where the choice is obvious.

<p align="center">
  <img src="images/run_1_action_probs_no_ace.png" width="45%">
</p>

After thinking about the issue I realized that the issue could be that for the option 'hit' to be a good choice, the latter choices need to be optimal too in this scheme. Rewards are based on the actual episode rewards (or their advantages to be more precise) and there is no value function that gives credit for the earlier good choice if a later bad choice loses the reward. This should be especially bad in the beginning of the training: You might be holding 14 and hit, but after getting lucky with a 6, the policy hits again and you bust losing the reward.

<p align="center">
  <img src="images/(14, 1, False).action_probs.1.png" width="80%">
</p>
<p align="center">
  <sub><em>The graph from Weights & Biases shows the probability of hitting in a state where we have 14 and the dealer is showing an Ace. As expected, the model quickly learns to stick, but after roughly a million episodes it starts to slowly appreciate hitting more.</em></sub>
</p>

The reason for this can be seen when checking how the model operates with an obvious choice where the hand is 21. Only sane option is to stick as hitting will only get you busted. Still, initially the probability is 50% from the randomly initialized neural net. Before it learns to stick here, all the hands with a lower value e.g. 14 will get a biased signal. And as you can see from the graph below, it takes like 500k episodes before the probability of hitting in that state gets close to zero.

<p align="center">
  <img src="images/(21, 1, False).action_probs.1.png" width="80%">
</p>

At that point the model has already learned to stick in states like (14, 1, False) mentioned earlier. Since at that point the probability of hitting in such as state is low, there wont be many visits to the action of hitting in that state and the unlearning of the wrong strategy will take time. From the graph below it can be seen that such action in the state is often tried 1 or 0 times per update (25k hands). However, towards the end of the 2 million episodes the amount of visits starts to likely pick up as the action probability increases, suggesting that the speed of unlearning could accelerate.

<p align="center">
  <img src="images/(14, 1, False).actions.1.visits.png" width="80%">
</p>

This is strong evidence that 2 million episodes simply were not enough for the problem. There are also some states where the outcome of the two actions is extremely close: on this [blog post](https://chisness.github.io/2020-09-21/monte-carlo-rl-and-blackjack) the authours mention that when player has 16 and the dealer is showing a ten the difference between hitting and sticking is 0.000604 in favor of hitting, making it very difficult so get it right with a RL model. But even if all edge cases cannot be learned there is still some clear room for improvement and even the training loss seems to be slowly decreasing despite being volatile.

<p align="center">
  <img src="images/run_1_loss.png" width="40%">
</p>
