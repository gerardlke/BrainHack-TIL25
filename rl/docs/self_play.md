# I LOVE SELF PLAY

oh yeah time to conduct self-play (no homo)

## How to self play

Lets first make explicit our rough strategy for establishing self-play.

self play plans:

0. Instantiate empty network pool for all agents
1. To start, randomly instantiate hparams for all agents, and do 1st round of play (round = n episodes / n steps)
2. Cache weights once seemingly converged

start of self/cross play
For each agent, Loop:
3. Randomly load weights for the other 3
4. Play for one episode

5. loop 3 and 4  until stop reward, no improvement or max elapse, and take best (todo refine this choice), and add to network
6. prune old checkpoint from network pool (worst or just by time?)
7. select another agent and go back to start of self/cross play until you get agi

8. get agi


## How will we accomplish the above?

For 0-2, our existing normal training code will suffice, so just integrate that.
Onwards, we will have to create new things.


## Zeroth, the network pool

This is the most crucial part of our self-play design. The plan for this should be worked on in greater detail, but the requirements are as follows:

1. Components can retrieve and write to this pool, any particular checkpoint of any particular agent.
2. There must be two indexes; one for the pool of checkpoints currently considered for sampling (e.g ones that are newer or havent been pruned for bad performance), and another index for all checkpoints
3. may come alongside more requirements

Envrionment -> Retrieves opponent NPC policies from pool
Trainer -> Retrieves best policy for that particular agent (what is best is determined by some evaluator, block this out in your head for now)
Evaluator -> Loops over (all?) past versions of opponent NPC agents (every single permutation and combination? might be combinatorily expensive) and plays against them to truly evaluate current policy



### First, the environment. 

Yes, the base environment remains the same, but if we approach this by teaching each agent independently at its own time, the other agents are technically part of the environment. This means that several integrations and mutations have to occur.

- observation_space: Only as big as one agent's observation space
- action_space: Only as big as one agent's action space
- step: Each step only takes in one agent's action, and automatically runs prediction for other NPC models. (lets call them these for now on)
- reset: Depending on the reset strategy, we may randomly choose other weights to be loaded in, so the learning agent has more variety of opponents to play against. We could swap opponents after each train method call (i.e after learning policy's rollout buffer is completely gathered) or just every reset call of the environment. see how


### Second, the trainer.

While the environment handles loading of opponents (and everything else environment related), the training orchestrator should handle policy training and rollout collection, and any other callbacks attached. This code should remain relatively similar, and just re-initialized whenever we are done with one agent and moving on with another.


### Third, callbacks.

The main callback to cause the most issue is the evaluate callback, which I will term Evaluator, as we need to decide on what policies to play against, which implies we will need to determine what is a fair evaluation.

Either way, this evaluator should operate the environment resetting in some way, as it should tell the environment which checkpoints to play against.