import gym
import numpy as np
import random
import pandas as pd
from collections import defaultdict

#hyperparameter

alpha = 0.1
gamma = 0.6
epsilon = 0.1
total_episodes=1000000

def main():
  #creating q table
  env = gym.make("Taxi-v3").env
  q_table = np.zeros([env.observation_space.n, env.action_space.n])
  for i in range(total_episodes):
      curr_state = env.reset()
      
      epochs = 0
      penalties, curr_reward, total_reward = 0, 0, 0
      while curr_reward != 20:#positive reward considered 20 
          
          curr_state, curr_reward = update_table(q_table, env, curr_state)
          total_reward += curr_reward

          if curr_reward == -10: #penalty for wrong drop offs
              penalties += 1

          epochs += 1
         

  print("Training finished.\n") 
  result=q_table.ravel()
  df=pd.DataFrame(result)
  df.columns = ['Id']
  df.index = np.arange(1, len(df)+1)
  df.to_csv("results.csv", index=True)

def update_table(q_table, env, state):
  if random.uniform(0, 1) < epsilon:
    # number<epsilon exploration in progress pick random action
    action = env.action_space.sample()
  else:
    ## If this number > epsilon = exploitation (take the action with biggest Q value for current state)
    action = np.argmax(q_table[state,:])

  #observe the action outcome 
  next_state, reward, done, info = env.step(action)
  old_q_value = q_table[state,action]

  # Maximum q_value for the actions in next state
  next_max = max(q_table[next_state,:])

  # Calculate the new q_value
  new_q_value = (1 - alpha) * old_q_value + alpha * (reward + gamma * next_max)

  # Finally, update the q_value
  q_table[state,action] = new_q_value

  return next_state, reward



if __name__ == "__main__":
    main()

