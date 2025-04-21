#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This algorithm provides a solution to the cart_pole problem based on artificial artificial intelligence.
Autor: Wittgrocket
Date: Feb 2025

https://gymnasium.farama.org/environments/classic_control/cart_pole/
Heavily based on A. Géron, Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, O’Reilly, 2019

see this comparision 
https://www.youtube.com/watch?v=kocx9DlghoM

"""



import numpy as np
import gymnasium as gym
import tensorflow as tf
import time as t
#import tf_keras as ks  # Use legacy Keras 2.16.0, because Keras ≥ 3 does not work.
import keras as ks  # Use legacy Keras 2.16.0, because Keras ≥ 3 does not work.
from keras.models import load_model
from enum import Enum
import pickle
ITERATION_COUNT = 199  # 9999Severe success is usually achieved after about 50 iterations.
EPISODES_PER_UPDATE = 10

DISCOUNT_FACTOR = 0.95
class NNMODE(Enum):
    START = 0
    TRAINING_MODE = 1
    TEST_MODE = 2
# bad style with global variables
#myMode=NNMODE.TRAINING_MODE
myMode=NNMODE.TEST_MODE
if myMode == NNMODE.TRAINING_MODE:
    MAX_STEP_COUNT = 500
elif myMode == NNMODE.TEST_MODE: 
    MAX_STEP_COUNT = 10000  
myTrainingTime=0
myEpisodes=0
start_time = t.time()
def main():
    global myMode
    global myTrainingTime
    global start_time
    environment = gym.make('CartPole-v1', render_mode='human')
    environment.reset()
    t.sleep(20)
    start_time = t.time()
    if myMode==NNMODE.TRAINING_MODE: 
      model = ks.models.Sequential([ks.layers.Dense(5, activation='elu', input_shape=[4]),                            ks.layers.Dense(1, activation='sigmoid')])
      optimizer = ks.optimizers.legacy.Adam(learning_rate=0.01)
    elif myMode==NNMODE.TEST_MODE: 
      model = load_model('my_model.h5')
      optimizer = ks.optimizers.legacy.Adam(learning_rate=0.01)
      value=0    
      with open('LearningTime', 'rb') as f:
        myTrainingTime = pickle.load(f)    
    for iteration in range(1, ITERATION_COUNT + 1):
        print(f'Iteration {iteration}')
        all_rewards, all_gradients = play_multiple_episodes(environment, EPISODES_PER_UPDATE, MAX_STEP_COUNT, model,
                                                            loss_function=ks.losses.binary_crossentropy)
        all_final_rewards = discount_and_normalize_rewards(all_rewards, DISCOUNT_FACTOR)
        all_mean_gradients = []
        for var_index in range(len(model.trainable_variables)):
            mean_gradients = tf.reduce_mean(
                [final_reward * all_gradients[episode_index][step][var_index]
                 for episode_index, final_rewards in enumerate(all_final_rewards)
                 for step, final_reward in enumerate(final_rewards)],
                axis=0)
            all_mean_gradients.append(mean_gradients)
            optimizer.apply_gradients(zip(all_mean_gradients, model.trainable_variables))
    model.save('my_model.h5')  # Saves architecture, weights and optimizer state
    with open('LearningTime', 'wb') as f:
     pickle.dump((t.time() - start_time), f)
    
def play_multiple_episodes(environment, episode_count, max_step_count, model, loss_function):
    global myEpisodes
    global myMode
    global start_time
    all_rewards = []
    all_gradients = []
    for episode in range(1, episode_count + 1):
        print(f'    Episode {episode:2}', end='')
        current_episode_rewards = []
        current_episode_gradients = []
        observation, _ = environment.reset()
        myEpisodes=myEpisodes+1
        for step in range(1, max_step_count + 1):
            observation, reward, terminated, truncated, gradients = play_one_step(
                environment, observation, model, loss_function,start_time,myEpisodes,step)
            current_episode_rewards.append(reward)
            current_episode_gradients.append(gradients)
            if myMode==NNMODE.TRAINING_MODE:
              if terminated or truncated:
                print(f' | {step:3} steps')
                break
            elif myMode==NNMODE.TEST_MODE:
              if terminated:
                print(f' | {step:3} steps')
                start_time= t.time()    
                break  
        else:
            print(f' | Finished {max_step_count + 1} steps without termination.')
        all_rewards.append(current_episode_rewards)
        all_gradients.append(current_episode_gradients)
    return all_rewards, all_gradients


def play_one_step(environment, observation, model, loss_function,start_time,episode,step):
    global myTrainingTime
    with tf.GradientTape() as tape:
        left_probability = model(observation[np.newaxis])
        action = (tf.random.uniform([1, 1]) > left_probability)
        y_target = tf.constant([[1.]]) - tf.cast(action, tf.float32)
        loss = tf.reduce_mean(loss_function(y_target, left_probability), axis=0)
        gradients = tape.gradient(loss, model.trainable_variables)
        elapsed_time = t.time() - start_time    
        timespan = round(elapsed_time,2)
        #timespan=(f"Elapsed time: {elapsed_time:.2f} seconds") 
        action=[int(action[0, 0].numpy()), timespan, myMode.value,episode,step] 
    #observation, reward, terminated, truncated, _ = environment.step(int(action[0, 0].numpy()))
    observation, reward, terminated, truncated, _ = environment.step(action)
    if myMode == NNMODE.TEST_MODE:   
        timespan =  round(myTrainingTime/60,2)  
        
           
    return observation, reward, terminated, truncated, gradients


def discount_and_normalize_rewards(all_rewards, discount_factor):
    all_discounted_rewards = [discount_rewards(rewards, discount_factor) for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean) / reward_std for discounted_rewards in all_discounted_rewards]


def discount_rewards(rewards, discount_factor):
    discounted_rewards = np.array(rewards)
    for step in range(len(rewards) - 2, -1, -1):
        discounted_rewards[step] += discounted_rewards[step + 1] * discount_factor
    return discounted_rewards


if __name__ == '__main__':
    main()
