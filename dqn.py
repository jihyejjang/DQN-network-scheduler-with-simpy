#!/usr/bin/env python
# coding: utf-8

# # DQN model

# In[ ]:


#dqn model

import pandas as pd
import numpy as np 
import random 
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from collections import deque 
from itertools import product
import math
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam 

class DQN:#모델 선언
    def __init__(self): #parameter들의 초기값
        self.epsilon = 0.999
        self.epsilon_min = 0.01
        self.step = 1
        self.tau = 0.125 #?
        self.learning_rate = 1 #할인율. 1에 가까울 수록 미래에 받는 보상도 중요, 0에 가까울수록 즉각적인 보상이 중요
        self.memory = deque()
        self.model = self.create_model() #현재 state에 대한 model
        self.target_model = self.create_model() #next state에 대한 model

    # create the neural network to train the q function 
    def create_model(self): #Q값예측모델. 
        model = Sequential()
        model.add(Dense(24, input_dim= 3, activation= 'relu')) # input dimension : source들 차원
        model.add(Dense(48, activation= 'relu'))
        model.add(Dense(24, activation= 'relu'))
        model.add(Dense(66)) #계산했을 때,(1~10)까지 세 수의 합이 10이 되는 경우의수는 66개. output에 대한 가중치는 매번 update되기 때문에 이에 mapping시키면 된다. 
        model.compile(loss= 'mean_squared_error', optimizer= Adam(lr= 1e-3)) #optimizer의 learning rate도 주의
        
        return model 



    # Action function to choose the best action given the q-function if not exploring based on epsilon p값에 의한 예측이 아닐때
    def choose_action(self, state, allowed_actions): #action을 선택 (parameter로 선택가능한 action이 들어옴)
        select = False
        if (self.step%50000 == 0):#약 10만번 step에서 3만번의 scheduling 발생. 10000번마다 epsilon이 감소
            self.epsilon = max(self.epsilon_min, pow(self.epsilon,int(self.step/10000 +1)))
        #print ("epsilon", self.epsilon)
        self.step+=1
        r = np.random.random()
        if r < self.epsilon: #p값보다 작은 경우 랜덤한 액션을 취함
            print("random action")
            return random.choice(allowed_actions),self.epsilon,select
        
        
        print ("@@action choose@@" , self.step)
        select = True
        state = np.array(state).reshape(1,len(state)) #p값보다 큰경우, state 배열 생성
        
        pred = self.model.predict(state)[0]
        #print ("q",pred)
        
        return self.maxQ_action(pred,allowed_actions),self.epsilon,select #Q예측값중 min_rate 이상으로 가장 큰 action을 선택
    

    def maxQ_action(self,pred,allowed_actions):#allowed action 생성 (min_rate 이상 조합만 남김)
        print ("max q", np.argmax(pred) )
        return allowed_actions[np.argmax(pred)]
        
        
        
    # create replay buffer memory to sample randomly #메모리에서 꺼내서 학습할 수 있게 저장, terminal이란 next_state가 없는 경우
    def remember(self, state, action, reward, next_state,terminal):
        self.memory.append([state, action, reward, next_state,terminal])


    # build the replay buffer 저장한 것을 버퍼에서 꺼내오는.? 학습단계?
    def replay(self,allowed_actions):
        
        #global mse_loss
        mse=[]
        batch_size = 32
        if len(self.memory) < batch_size: #buffer에 저장된 memory가 buffer의 총 batch_size보다 작다면 return
            return 
        
        samples = random.sample(self.memory, batch_size) #메모리에서 배치사이즈만큼 랜덤으로 선택
        
        
        for sample in samples:
            
            #print ("sample" , sample)
            
            state, action, reward, new_state, terminal = sample # sample 데이터 하나를 꺼내서
            
            state = np.array(state).reshape(1,len(state)) 
            
            new_state = np.array(new_state).reshape(1,len(new_state))
            
            target = self.target_model.predict(state) 
            
            action_id = allowed_actions.index(tuple(action)) #63개의 allowed action 중에서 state에 대한 action의 index를 추출

            if terminal :
                target[0][action_id] = reward
            else :
                next_pred = self.target_model.predict(new_state)[0] #new state에 대한 target 예측
                  
                Q_future= max(next_pred) #next state에 대한 predict 값 중 가장 큰 값이 Q값이 됨
                
                target[0][action_id] = reward + Q_future * self.learning_rate # target의 action_id번째 위치에 다음 Q값이 들어감. 맞춰야 하는 값!!!
            
            history=self.model.fit(state, target, epochs= 1, verbose= 0) 
            mse.append(history.history['loss'][0]) # loss 기록
            
        return min(mse)
        
        #print("Mean_square_error:"min(mse_loss))
        


    # update our target network 
    def train_target(self): #target network를 업데이트
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)#loss함수?
            #target_weights[i] = weights[i]
        self.target_model.set_weights(target_weights)



    # save our model 
    def save_model(self, fn):
        self.model.save(fn)

