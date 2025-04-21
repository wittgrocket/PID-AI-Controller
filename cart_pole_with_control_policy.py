#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This algorithm provides a classical solution to the cart_pole problem.
https://gymnasium.farama.org/environments/classic_control/cart_pole/
Autor: Wittgrocket
Date: Feb 2025

see this comparision 
https://www.youtube.com/watch?v=kocx9DlghoM
"""

import time as t
import math
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from enum import Enum
class CONTROLTYPE(Enum):
    UNCONTROLLED = 1
    SIMPLE_CONTROLLED = 2
    OPTIMAL_CONTROLLED = 3
    
class pid_control:
  def __init__(self, P, I, D, setpoint, controlTyp):
    self.P = P #0.5
    self.I = I #0.1
    self.D = D #0.3
    self.setpoint = setpoint 
    self.error_sum = 0
    self.prev_error = 0
    self.controlTyp = controlTyp
    self.inControl = 0
    self.StartTime = 0
    self.timeSpan = 0.5 #1s
    
        
  def controller_reset(self):
    self.error_sum = 0
    self.prev_error = 0   
  def controller_update(self, pole_angle, pole_angle_velocity,cart_velocity, cart_position):
    if self.controlTyp == CONTROLTYPE.UNCONTROLLED:
      self.P =0
      self.I =0
      self.D =0
      error = self.setpoint - pole_angle_velocity
      error1 = self.setpoint - pole_angle
      error2 = self.setpoint - cart_velocity
    elif  self.controlTyp == CONTROLTYPE.SIMPLE_CONTROLLED:
      error = self.setpoint - pole_angle_velocity
      error1 = self.setpoint - pole_angle
      error2 = 0
      error3 = 0 
    else:
      error = self.setpoint - pole_angle_velocity
      error1 = self.setpoint - pole_angle
      error2 = self.setpoint - cart_velocity
      error3 = self.setpoint - cart_position
    #solve the initialisation problem with larger pole angles
    if error1 >1.5:
      error = error1*0.5
    elif error1 <-1.5:
      error = error1*0.5 
    #solves the drift problem
    if error2 >0.15:
     error = error + error2*0.25
    elif error2 <-0.15:
     error = error + error2*0.25
    if error3 >0.15:
     error = error + error3*0.25
    elif error3 <-0.15:
     error = error + error3*0.25 
    if (abs(error) < 0.5 and abs(self.setpoint - cart_velocity)< 1):
      #debouncing
      if (t.time() - self.StartTime)> self.timeSpan:
        self.inControl = 1
    else:
      self.inControl = 0 
      self.StartTime = t.time() 
    self.error_sum += error
    d_error = error - self.prev_error
    self.prev_error = error
    output = self.P * error + self.I * self.error_sum + self.D * d_error
    return output, self.inControl   
class cart_pole:
  def __init__(self):
    self.position=[]
    self.velocity=[]
    self.pole_angle =[]
    self.pole_angle_vel=[] 
    self.control=[]
    #self.controlType=CONTROLTYPE.SIMPLE_CONTROLLED
    #self.setpoint - cart_velocityself.controlType=CONTROLTYPE.UNCONTROLLED
    self.controlType=CONTROLTYPE.OPTIMAL_CONTROLLED
    self.pole_velocity_control=pid_control(1.5,0.2,0.2,0,self.controlType)
    self.first_break=False
    self.inControl=0

  def basic_pole_angle_policy(self,observation):
    degrees = observation[2] * (180 / math.pi)
    self.pole_angle.append(degrees)
    self.pole_angle_vel.append(observation[3])
    if self.pole_angle[-1] < 1:
      return 0
    else:
      return 1 

  def basic_pole_angular_velocity_policy(self,observation):
    degrees = observation[2] * (180 / math.pi)
    self.pole_angle.append(degrees)
    self.pole_angle_vel.append(observation[3])
    self.velocity.append(observation[1])
    self.position.append(observation[0])
    retval, self.inControl=self.pole_velocity_control.controller_update(self.pole_angle[-1],self.pole_angle_vel[-1],self.velocity[-1],self.position[-1])
    self.position.append(retval)
    
    if retval < 0:
      self.control.append(1)
      return 1    #left side
    else:
      self.control.append(-1)
      return 0    #right side
  
    


  def simulate(self):
    environment = gym.make('CartPole-v1', render_mode='human')
    # Start time
    start_time = t.time()
    elapsed_time=0
    for episode in range(1, 10 + 1):
        self.pole_velocity_control.controller_reset()
        print(f'Episode {episode:2}: ', end='')
        total_reward = 0
        observation, _ = environment.reset()  
        print('Startwert')
        t.sleep(10)
        start_time = t.time()
        elapsed_time= 0
        for _ in range(500):   #Schritte pro Episode
            #action = self.basic_pole_angle_policy(observation)
            action = self.basic_pole_angular_velocity_policy(observation)
            print('<' if action == 0 else '>', end='')
            if self.pole_velocity_control.inControl == 0:
              elapsed_time = t.time() - start_time
            timespan=(f"Elapsed time: {elapsed_time:.2f} seconds")
            timespan = round(elapsed_time,2)
            action=[action, timespan,self.controlType.value,self.inControl] 
            observation, reward, terminated, truncated, _ = environment.step(action)
            total_reward += reward
            if self.controlType==CONTROLTYPE.UNCONTROLLED or self.controlType==CONTROLTYPE.SIMPLE_CONTROLLED:
               if abs(observation[0])>3:
                 break
            elif terminated or truncated:
                print(f'\nTotal Reward: {total_reward}')
                self.first_break=True
                break
            t.sleep(0.05)
        if self.first_break == True:
           pass
           #break # First break debugging    
  def plot(self):
     # Create a figure and axis
    fig, axs = plt.subplots(2, 2, figsize=(12, 6))

    axs[0,0].plot(self.pole_angle,label="pole angle")
    axs[0,1].plot(self.pole_angle_vel,label="pole angle velocity")
    axs[1,0].plot(self.position,label="pole cart")
    axs[1,1].plot(self.velocity,label="velocity cart")
    fig.set_size_inches(3, 2) 
    axs[0,0].set_title('pole angle')
    axs[0,0].set_xlabel('number of rewards')
    axs[0,0].set_ylabel('pole angle')
    axs[0,0].legend()
    axs[0,1].set_title('pole angle velocity')
    axs[0,1].set_xlabel('number of rewards')
    axs[0,1].set_ylabel('pole angle velocity in [m/s]')
    axs[0,1].legend()
    axs[1,0].set_title('position cart')
    axs[1,0].set_xlabel('number of rewards')
    axs[1,0].set_ylabel('position cart [m]')
    axs[1,0].legend()
    axs[1,1].set_title('velocity cart ')
    axs[1,1].set_xlabel('number of rewards')
    axs[1,1].set_ylabel('velocity cart [m/s]')
    axs[1,1].legend()
    fig.show()
    input()

if __name__ == '__main__':
    mycar = cart_pole()
    mycar.simulate()
    mycar.plot()
