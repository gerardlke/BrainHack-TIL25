## Note

If ray tune is used, any configuration provided by ray tune that conflicts with existing configurations will be overridden by ray tune. For instance, your default environment frame stacking could be 4, but if you are hyparameter tuning using ray to try and find a better frame stacking number, the number returned by ray will override it.


# Configuration Specifics

## Env

This category handles anything to do with our environment. Custom rewards, number of vectorized environments, frame stacking, novice intialization, etcetc.



# Other Notes:

1. Distinction between policy and role.
    - I define a policy as 'a model which is fed in observations and spits out an action'
    - I define a role as 'a role in the game'.

    This distinction may appear superficial but it is important.
    When training and evaluating our policies, we must know which agent is running under what policy.
    However, individual agent performance should be detached from the policy umbrella it is under.

    For instance, player_1, player_2 and player_3 may all share one guard policy (if we define it to be so), however if we want a mangifying glass on each of their individual performances, we should segregate them into their own respective roles (e.g if player_1 always starts top right, player_2 bottom left, etc etc.). We could also group multiple agents into one role. For instance, player_1 and player_2 are closer to player_0 which we fix as the scout, but player 3 is the furthest, which means we could distinguish them into 'closer' and 'further' guard roles during evaluation, if we so choose.

    Thus, this distinction emerges. We can have the macro numbers (how all agents under a policy perform), and the micro numbers (each distinct role), while keeping policy and role definitions seperate.






