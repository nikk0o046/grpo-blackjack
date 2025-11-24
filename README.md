# grpo-blackjack
Implementation of DeepSeek's GRPO algorithm for Blackjack

This readme file works both as my notes for the project and as an explanation for anyone interested in the implementation.

## Goal of this project

The primary goal is hands-on learning of the GRPO algorithm used in the famous DeepSeek model.

## Implementation

A critical element of this group relative policy optimization is that even from your starting point you need a way to gain rewards in order to learn. They use a pre-trained LLM as a reference policy which is already capable of solving some of their training problems some of the time. I want to go for a small problem just to check if I can get this algorithm to converge to a good solution and I don't have a pre-trained model to start with or a lot of compute. In my opinion going Blackjack would be a good problem for this use case as even with a random policy it will win some of the time.

Since I start with a random policy I can drop the whole KL divergence term in the formula as there is no need to keep it regulated towards a pre-trained model.

To interact with an environment I will use an existing blackjack implemention e.g. from gymnasium package.

Also, it would make sense to first run tests with a known solution to the problem to make sure everything is running smoothly. Once I have verified I can get an existing implementation to converge to an optimal policy I can start tweaking it and implementing this GRPO approach which I'm not familiar has been implemented before.

One thing to note is that this surely is NOT the optimal solution for blackjack - GRPO algorithm is in many ways wasteful, but for some problems it is one of the only ones that works without supervision. Therefore it is not as relevant to benchmark it against the speed of some of the other algorithms, but instead focus on if it can reach the same end result through a different reward scheme.
