#!/usr/bin/env python
# coding: utf-8

# # Reinforcement learning Simpy Env&Agent


import simpy
import pandas as pd
import numpy as np 
import random 
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
#from prettytable import PrettyTable
from collections import deque 
from itertools import product
import math
from dqn import DQN
import time

#npz
mse_loss=[] #episode의 진행에 따른 mse_loss의 변화율 graph
flow_success=[] #episode의 진행에 따른 flow_success rate
pexp=[] #action choice 진행에 따른 pexp 변화
record = []

state=[0,0,0]#초기 state
action=[0,0,0]#초기 action 
Tsc=2 #scheduling interval을 구성하는 flow update interval의 수
Tfu=1 #flow update interval의 시간단위(0.1초로 가정)
sources = ['s0', 's1', 's2']
destinations = ['d0', 'd1', 'd2']
filesize=[random.randrange(10,50) for source in range(len(sources))] #단위 Gb
deadline=[int(filesize[source]/3) for source in range(len(sources))] #sum_rmin이  9Gbps, 각각 3
request = {
    'sources' : sources,
    'destinations' : destinations, 
    'filesize' : filesize, #단위는 Mbps
    'deadline' : deadline,
    'value' : [None for source in sources] #아직 전송되지 않았으면 None, 제시간에 전송되었으면 1, 제시간에 전송되지 않았으면 0
}  

env = simpy.Environment()
dqn_agent = DQN()

#보상 함수1: deadline이 적게 남을수록 pacing rate을 많이 할당    
def reward_function(deadlines,action,value): 
    reward = 0
    drem_max=np.max(deadlines)
    value=np.array(value)
    active=np.where(value==None) #active flow index
    act_deadline=[] #active flow deadline
    
    try: #active(전송중인) flow에 대해
        for a in active[0]:
            act_deadline.append(deadlines[a])
        drem_max=np.max(act_deadline) #최대 deadline을 가지는 active flow
    except: #deactive flow만 있을 때
        drem_max=np.max(deadlines)
        
        
    for dremi in active[0]: #active한 flow, deadline이 작을수록 action이 많이 할당되어야 reward또한 커짐
        if (deadline[dremi]>=0): #active and positive deadline / deadtive, negative deadline flows get reward 0
            reward += ((drem_max - deadlines[dremi])*action[dremi])
            
    return reward

    
def episode(env,DQN,Tsc,Tfu,allowed_actions,filename): #pacing rate가 각 flow에게 할당
    global action
    global state
    global request
    global flow_success
    global record
    
    
    m=-200 # 200번의 시행 동안
    s=0.9 # 성공률이 90%이상이면
    
    cnt=1 #episode 수
    c=0 #scheduling interval의 수
   
    #episode 시작
    
    while True: #Simularion time 동안 episode를 반복한다
        print ("--------------------------------------------------")
        print("********Episode start********",cnt)
        print ("")
        
        terminal=False
        first_action=1 #new action policy -> EDF
        
        #1개의 에피소드는 모든 filesize가 0이 될때까지 실행
        while ((request['filesize'][0]!=0)or(request['filesize'][1]!=0)or(request['filesize'][2]!=0)):#모두 전송이 완료될 때 까지
            c+=1
            
            #state 결정
            
            state=[0,0,0] #state 초기화
            for s in range(len(sources)):#deadline이 0이 아닌 source는 그대로
                if ( request['value'][s] != None ) : # (filesize=0 , value =1,0) 
                    state[s]=0
                else: #active, value=none
                    if (request['deadline'][s] > 0) : #(value = None, deadline >0이면 전송중, deadline =0이면 기한 지남)
                        state[s]=math.ceil(request['filesize'][s]/request['deadline'][s])
                    elif (request['deadline'][s] == 0):# 기한 지남, 남은시간 0
                        state[s]=10 #link capacity 전체 할당
                    elif (request['deadline'][s] < 0):# 기한 지남
                        state[s]=math.ceil(request['filesize'][s]/request['deadline'][s])
            
            print ("state", state)
            print ("deadline", request['deadline'])
            print ("filesize",request['filesize'])

            #action 결정
            
            if (first_action==1): #New state-action policy를 EDF방식으로 (가장 deadline이 시급한)     
                action=[0,0,0]
                index=np.argmin(request['deadline']) #deadline 최소인 source의 index
                action[index]=10 #bottleneck capacity
                
            else: 
                action,p,select=dqn_agent.choose_action(state,allowed_actions) #DQN에 의한 액션선택
                pexp.append(p) #epsilon값
                
            print ("action", action)
                
            #Scheduling interval 시작
            print ("Tsc" , c)
            
            for i in range(Tsc): 
                for s in range(len(sources)): #각 source에 대해 
                    
                    #filesize와 deadline 감소
                    
                    request['filesize'][s]=max([request['filesize'][s]-int(action[s]),0]) #filesize는 음수 X
                    
                    if (request['value'][s]==None):
                        request['deadline'][s]= request['deadline'][s]-1 #deadline는 음수 가능->state때문, reward에서는 음수반영 x reward는 deadline안에 전송하는것이 목적이기 때문임
                    
                    # Active, Deactive flow 검사, 전송완료되면 deadline 0으로.
                    
                    if ((request['filesize'][s]==0) and (request['value'][s]==None)): #아직 완료되지 않았던 flow가 전송이 완료되면?
                        
                        if (request['deadline'][s]>=0): #기간 안에 전송되면? 남아있는 시간이 양수, 또는 0 (시간이 0에 딱 맞게 전송 되는 경우도 있음..)
                            request['value'][s]=1 #value를 1로 변경
                            request['deadline'][s] = 0 #남은시간도 0
                            flow_success.append(1)
                            #print ("s{}의 전송이 deadline 안에 완료됨".format(s))
                            
                        else: #기간안에 전송된게 아니라면(value초기값은 None)
                            request['value'][s] = 0
                            flow_success.append(0)
                            #print ("s{}의 전송이 deadline을 지나 완료됨".format(s))
                            
                yield env.timeout(Tfu)# Tfu(1초)마다 위 과정 실행
            
            
            
            #모든 전송이 완료된 후 next_state는 고려할 필요 없음: terminal=True로 하여 target에 reward를 할당
            
            if ((request['filesize'][0]==0)and(request['filesize'][1]==0)and(request['filesize'][2]==0)):
                terminal = True
            
            #Next state 결정
            
            next_state=[0,0,0]
            for s in range(len(sources)):
                if (request['value'][s] == 1) : 
                    next_state[s]=0
                else: #active, value=none
                    if (request['deadline'][s] > 0) : 
                        next_state[s]=math.ceil(request['filesize'][s]/request['deadline'][s])
                    elif (request['deadline'][s] == 0):
                        next_state[s]=10 #link capacity
                    elif (request['deadline'][s] < 0):
                        next_state[s]=math.ceil(request['filesize'][s]/request['deadline'][s])

            #print("next_state" , next_state)
                        
            reward = reward_function((request['deadline']),action,request['value'])
            cur_state = state 
            action = action
            new_state = next_state 
            reward = reward
            terminal = terminal
            
            if (first_action==0): # 첫번쨰 선택 액션이 아닌경우에만 학습
                dqn_agent.remember(cur_state, action, reward, new_state,terminal) #새로운 state로 설정해주고 기존state저장
                mse_loss.append(dqn_agent.replay(allowed_actions))#학습, loss 저장
                dqn_agent.train_target()
                record.append([cur_state, action, reward, new_state , select]) #select는 action을 random에 의해 선택했는지 dqn에 의해 선택했는지 여부
            
            first_action=0 

        for i in range(len(sources)):
            if (request['value'][i]==1):
                print ("source {} 전송완료".format(i))
            else:
                print ("source {} deadine 충족하지 못함".format(i))
                    
        if (terminal==True): #학습 종료 시 검사
            if (np.mean(np.array(flow_success)[:m]) >= s) and (np.all(np.array(select[:m])==True) ):
                #결과 저장
                
                np.savez(filename,loss = mse_loss, success = flow_success, p = pexp, record = record )
                dqn_agent.save_model("dqn_policy.h5")
                print ("성공률 90%이상, 학습 종료")
                sys.exit()

        
        print("모두 전송 완료")
        
        #다음 episode에 simulation할 flow생성

        filesize=[random.randrange(10,50)for source in range(len(sources))] #byte 단위
        deadline=[int(filesize[source]/3) for source in range(len(sources))] #sum_rmin이  9
        request = {
            'sources' : sources,
            'destinations' : destinations, 
            'filesize' : filesize, #단위는 Gbps
            'deadline' : deadline,
            'value' : [None for source in sources] #아직 전송되지 않았으면 None, 제시간에 전송되었으면 1, 제시간에 전송되지 않았으면 0
        }

        state=[0,0,0]#초기 state
        action=[0,0,0]#초기 action ->고칠것
        print ("********episode end********")
        print("--------------------------------------------------")
        cnt+=1


    
def main(filename):    
    # main함수

    A=range(11)
    B=range(11)
    C=range(11) 
    allowed_actions=[] #합이 10이 되는 0~10까지의 수 조합
    for i in product(*(A,B,C)):
        if sum(i)==10:
            allowed_actions.append(i) #d는 66개의 조합

    start = time.time()  # 시작 시간 저장
    env.process(episode(env,DQN,Tsc,Tfu,allowed_actions,filename))
    env.run(until=100000)#10만 초 동안 가동

    #결과 저장
    np.savez(filname,loss = mse_loss, success = flow_success, p = pexp, record = record )
    dqn_agent.save_model("dqn_policy.h5")

    print("종료")
    print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간
    
if __name__=="__main__":
    main(filename)

