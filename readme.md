# Reinforcement Learning in Videogames

## Introduction

This is my end-of-degree project about Reinforcement Learning in videogames, the objective of this project is to implement, test and compare diferent reinforcement learning algorithms ranging form simple tabular methods to more complex Deep RL methods.

# Tabular Methods

These methods consist on filling a table with values for each state-action pair, later this table will be used to obtain the optimal policy which indicates the behaviour the agent has to follow in order to have the best chance to reach the goal. 

These methods will be tested in the Frozen-Lake environment, in which a character has to reach the goal without falling into a hole, the catch is that there is that the agent doesn't always move where it wants to. For example if the agent wants to move down there is a 33% chance of going left or a 33% chance of going right instead. I will test the algorithms in two different frozen lake maps, a 4x4 and an 8x8.

## Dynamic Programming

I have implemented the value-iteration method which works by repeatedly updating the state values using the Bellman optimality equation until convergence, meaning that when the changes in the state values are smaller than a threshold we stop the algorithm since its already arrived at the optimal value function form which we can obtain the optimal policy.

### Results in 4x4

This is the policy obtained in 4x4:

![image](value_iteration/vi_4x4_policy.png)

This is the optimal policy in this map, it is impossible to reach the goal with 100% accuracy in this map, after a 1000 episodes of testing with this policy it reaches an 85% accuracy and it takes 0.04 seconds. 

### Results in 8x8

This is the policy obtained in 8x8:

![image](value_iteration/vi_4x4_policy.png)

This is the optimal policy in this map, it is possible to reach the goal with 100% accuracy in this map since the agent can keep going through the sides of the map avoiding the holes and it will evetually reach the goal, after a 1000 episodes of testing with this policy it reaches 100% accuracy and it takes 0.37 seconds. 

## Monte Carlo



## TD Learning

