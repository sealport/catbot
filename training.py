import random
import time
from typing import Dict
import numpy as np
import pygame
from utility import play_q_table
from cat_env import make_env
#############################################################################
# TODO: YOU MAY ADD ADDITIONAL IMPORTS OR FUNCTIONS HERE.                   #
#############################################################################

# Checks if nextState has matching 1st and 3rd, 2nd and 4th digits
def isGoal(nextState):
    if(nextState // 1000 == nextState % 100 // 10 and nextState // 100 % 10 == nextState % 10):
        return True
    else:
        return False




#############################################################################
# END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
#############################################################################

def train_bot(cat_name, render: int = -1):
    env = make_env(cat_type=cat_name)
    
    # Initialize Q-table with all possible states (0-9999)
    # Initially, all action values are zero.
    q_table: Dict[int, np.ndarray] = {
        state: np.zeros(env.action_space.n) for state in range(10000)
    }

    # Training hyperparameters
    episodes = 5000 # Training is capped at 5000 episodes for this project
    
    #############################################################################
    # TODO: YOU MAY DECLARE OTHER VARIABLES AND PERFORM INITIALIZATIONS HERE.   #
    #############################################################################
    # Hint: You may want to declare variables for the hyperparameters of the    #
    # training process such as learning rate, exploration rate, etc.            #
    #############################################################################
    
    # Hyperparameters (temporary for now)
    alpha = 0.4
    gamma = 0.62
    epsilon = 0.95
    epsilon_decay = 0.0019

    # Used to count how many times a cat has been caught
    wins = 0



    #############################################################################
    # END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
    #############################################################################
    
    for ep in range(1, episodes + 1):
        ##############################################################################
        # TODO: IMPLEMENT THE Q-LEARNING TRAINING LOOP HERE.                         #
        ##############################################################################
        # Hint: These are the general steps you must implement for each episode.     #
        # 1. Reset the environment to start a new episode.                           #
        # 2. Decide whether to explore or exploit.                                   #
        # 3. Take the action and observe the next state.                             #
        # 4. Since this environment doesn't give rewards, compute reward manually    #
        # 5. Update the Q-table accordingly based on agent's rewards.                #
        ############################################################################## 

		# Referenced from: https://medium.com/data-science/q-learning-for-beginners-2837b777741
		# Referenced from: https://gymnasium.farama.org/introduction/train_agent/
        # 1. Reset the environment to start a new episode. #      
        state, info = env.reset()
        done = False

        for move in range(60):
            # Generate a random number between 0 and 1
            rnd = np.random.random()
            
            # 2. Decide whether to explore or exploit. #
            # If rnd < epsilon, explore
            if rnd < epsilon:
                action = env.action_space.sample()
            # Else, exploit
            else:
                action = np.argmax(q_table[state]) 

        # 3. Take the action and observe the next state. #
            next_state, reward, done, truncated, info = env.step(action)

            if isGoal(next_state):
                reward = 1
                #################################################################################
                # Increment the win counter (Comment this out when submitting)
                wins = wins + 1 
                #################################################################################
            else:
                reward = -0.00025

	    # Update Q(state, action)
            q_table[state][action] = q_table[state][action] + \
                                alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state][action])

	    # Update current state 
            state = next_state

			# End current episode if marked found?
            if done:
                break
                
        # Update chances of exploring/exploiting
        epsilon = max(epsilon - epsilon_decay, 0)

        ###########################################################################################
        # Used for evaluating the current model
        print("Wins: " + str(wins) + "/5000 || Remaining: " + str(5000 - ep))
        ###########################################################################################























        
        
        #############################################################################
        # END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
        #############################################################################

        # If rendering is enabled, play an episode every 'render' episodes
        if render != -1 and (ep == 1 or ep % render == 0):
            viz_env = make_env(cat_type=cat_name)
            play_q_table(viz_env, q_table, max_steps=100, move_delay=0.02, window_title=f"{cat_name}: Training Episode {ep}/{episodes}")
            print('episode', ep)

    return q_table
