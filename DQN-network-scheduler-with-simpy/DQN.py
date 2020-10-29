#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np 
#from collections import namedtuple
#from recordtype import recordtype
import random 
#import matplotlib
#matplotlib.use('TkAgg')
#import matplotlib.pyplot as plt
#from prettytable import PrettyTable
#get_ipython().magic('matplotlib inline')
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam 
from collections import deque 
#from itertools import product
import math


class DQN:#모델 선언
    def __init__(self): #parameter들의 초기값
        #self.gamma = 0.85
        self.epsilon = 0.999
        self.epsilon_min = 0.01
        #self.epsilon_decay = 0.95
        self.step = 1
        self.tau = 0.125 #?
        self.learning_rate = 1
        self.memory = deque(maxlen= 2000)
        self.model = self.create_model()
        self.target_model = self.create_model() #target_model이란 create_model의 return값..?



    # create the neural network to train the q function 
    def create_model(self): #Q값예측모델. 
        model = Sequential()
        model.add(Dense(24, input_dim= 3, activation= 'relu')) # input dimension : source들 차원
        model.add(Dense(48, activation= 'relu'))
        model.add(Dense(24, activation= 'relu'))
        model.add(Dense(66)) #계산했을 때,(1~10)까지 세 수의 합이 10이 되는 경우의수는 66개. output에 대한 가중치는 매번 update되기 때문에 이에 mapping시키면 된다. 
        model.compile(loss= 'mean_squared_error', optimizer= Adam(lr= self.learning_rate))
        
        return model 



    # Action function to choose the best action given the q-function if not exploring based on epsilon p값에 의한 예측이 아닐때
    def choose_action(self, state, allowed_actions): #action을 선택 (parameter로 선택가능한 action이 들어옴)
        
        if (self.step%10000 == 0):
            self.epsilon = max(self.epsilon_min, pow(self.epsilon,int(self.step/10000 +1)))
        print (self.epsilon)
        print (self.step)
        self.step+=1
        r = np.random.random()
        #print (r)
        if r < self.epsilon: #p값보다 작은 경우 랜덤한 액션을 취함
            print("random action")
            return random.choice(allowed_actions),self.step
        
        state = np.array(state).reshape(1,len(state)) #p값보다 큰경우, state 배열 생성

        pred = self.model.predict(state)[0]
        
        
        
            
        #state=0이면 filesize=0이라는 뜻인데 이 때 action을 0으로 할당해줘야 함. 만약 1개의 flow가 완료된경우는 어떻게 분배할 지 애매 
#         if (0 in state[0]):
#             #print("state",state[0])
#              #전송완료된 flow가 하나라도 있다면(근데 state가 모두 000인경우...)
#             index = [i for i, value in enumerate(state[0]) if value == 0]
            
# #             try:
# #                 if (index[2]==2):
# #                     return [3,3,4] #state가 다 0이면
                
# #             except:
# #                 pass 
            
#             q=[]
#             # indexes = [i for i, value in enumerate(a) if value[s_] = 0 for s_ in s]
#             # indexes = [i for s_ in s if value[s_]==0 ]
            
#             for a in allowed_actions:
#                 try:
#                     if (a[index[0]]==0): #0인 state가 하나면 여기까지만        
#                         try:
#                             if(a[index[1]]==0):
#                                 try:
#                                     if(a[index[2]]==0): #index가 2까지있다는것은 모든 인덱스가 0이라는것
#                                         #아무것도 없음
#                                         pass
                                        
#                                 except: 
#                                     q.append(allowed_actions.index(a))
#                                     continue
#                         except:
#                             q.append(allowed_actions.index(a))
#                             continue
#                 except:continue
#             print ("q:",q)
#             preds=[pred[q_] for q_ in q]
#             allowedacts=[allowed_actions[q_] for q_ in q ] #그 actions중에 가장 큰 Q값을 가지는 action을 가져옴
#             return allowedacts[np.argmax(preds)]
        
        return self.maxQ_action(pred,state,allowed_actions),self.step #Q예측값중 min_rate 이상으로 가장 큰 action을 선택
    

    def maxQ_action(self,pred,state,allowed_actions):#allowed action 생성 (min_rate 이상 조합만 남김)
#        print("maxQ action")
        #for p in range(len(pred)):
#             index=np.argmax(pred)
#             if ((allowed_actions[index][0]>=state[0][0])and(allowed_actions[index][1]>=state[0][1])and(allowed_actions[index][2]>=state[0][2])):#모든 action이 min_rate보다 크거나 같으면
#                 #print("선택된 action set은:",allowed_actions[index])
#                 return allowed_actions[index]
#             else:
#                 pred[index]=np.min(pred)
                
        #만약 return되지 못한 경우(즉, pred값이 가장 큰 index가 min_rate 조건을 충족시키지 못할 경우) 
        #위 코드를 주석처리하면 min_rate에 신경쓰지 않고 action을 할당하게 되는 코드 -> allowed_action이 none으로 return되는 상황 발생
        return allowed_actions[np.argmax(pred)]
        
        
        
    # create replay buffer memory to sample randomly #메모리에서 꺼내서 학습할 수 있게 저장
    def remember(self, state, action, reward, next_state):
        self.memory.append([state, action, reward, next_state])


    # build the replay buffer 저장한 것을 버퍼에서 꺼내오는.? 학습단계?
    def replay(self,allowed_actions):
        
        global mse_loss
        mse=[]
        batch_size = 32
        #list_of_next_allowed_machines = []
        if len(self.memory) < batch_size: #buffer에 저장된 memory가 buffer의 총 batch_size보다 작다면 return
            return 
        samples = random.sample(self.memory, batch_size) #메모리에서 배치사이즈만큼 랜덤으로 선택
        for sample in samples:
            state, action, reward, new_state = sample # sample 데이터 하나를 꺼내서
            #print (state)
            state = np.array(state).reshape(1,len(state)) 
            #print("state",state)
            new_state = np.array(new_state).reshape(1,len(new_state))
            #print("new_state",new_state) #????????????
            target = self.target_model.predict(state) #기존 state로 target을 예측
            
            action_id = allowed_actions.index(tuple(action)) #63개의 allowed action 중에서 state에 대한 action의 index를 추출

            target[0][action_id]=reward #63개의 action을 예측한 target에서 state에 대한 실제 action에 reward를 할당
                
            
            
            next_pred = self.target_model.predict(new_state)[0] #new state에 대한 target 예측
            
            new_action=self.maxQ_action(next_pred,new_state,allowed_actions) # new state에 대한 action예측
            
            Q_future=next_pred[allowed_actions.index(new_action)] #next state에 대한 action의 Q값은?
                
            target[0][action_id] = reward + Q_future * self.learning_rate # target의 action_id번째 위치에 다음 Q값이 들어감. 맞춰야 하는 값!!!
      
            
            history=self.model.fit(state, target, epochs= 1, verbose= 0) #1에폭으로 학습
            
            
            mse.append(history.history['loss'][0])
            
        return min(mse)
        
        #print("Mean_square_error:"min(mse_loss))
        


    # update our target network 
    def train_target(self): #학습에서 가중치를 업데이트
        print("update weights")
        weights = self.model.get_weights() 
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)#loss함수?
        self.target_model.set_weights(target_weights)



    # save our model 
    def save_model(self, fn):
        self.model.save(fn)

