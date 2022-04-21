import os
import sys
import optparse
import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=np.inf)
import math
import random
import matplotlib.pyplot as plt
import datetime
from DQN import DeepQNetwork
from model import EvalModel, TargetModel



if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci
from sumolib import checkBinary
import traci.constants as tc


def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options


# Environment Model
options = get_options()
if options.nogui:
    sumoBinary = checkBinary('sumo')
else:
    sumoBinary = checkBinary('sumo-gui')
# sumoBinary = checkBinary('sumo')
sumoCmd = [sumoBinary, "-c", "single_route.sumocfg"]

def reset():
    traci.start(sumoCmd)
    tls = traci.trafficlight.getIDList()
    return tls

def getLegalAction(phases):
    legal_action = np.zeros(5)-1
    i = 0
    for x in phases:
        if x>5:
            legal_action[i] = i
        if x<20:
            legal_action[i+3] = i+3
        i +=1
    legal_action[2] = 2
    return legal_action

def end():
    traci.close()

def state():
    for veh_id in traci.vehicle.getIDList():
        traci.vehicle.subscribe(veh_id, (tc.VAR_POSITION, tc.VAR_SPEED))
    p_state = np.zeros((40, 40))
    for veh_id in traci.vehicle.getIDList():
        p = traci.vehicle.getSubscriptionResults(veh_id)

        ps = p[66]
        spd= p[64]
        p_state[int(ps[0] / 5), int(ps[1] / 5)] = 1
        # print(ps)
        # if ps[0]>0 and int(ps[1])>0:
        #     p_state[0, abs(int(ps[0] / 5))] = 1
        # if ps[0]<0 and int(ps[1])<0:
        #     p_state[1, abs(int(ps[0] / 5))] = 1
        # if ps[1]>0 and int(ps[0])<0:
        #     p_state[2, abs(int(ps[1] / 5))] = 1
        # if ps[1]<0 and int(ps[0])>0:
        #     p_state[3, abs(int(ps[1] / 5))] = 1



    # for x in p:
    #     ps = p[x][tc.VAR_POSITION]
    #     spd = p[x][tc.VAR_SPEED]
    #     p_state[int(ps[0]/5), int(ps[1]/5)] = [1, int(round(spd))]
    #     v_state[int(ps[0]/5), int(ps[1]/5)] = spd
    # p_state = np.reshape(p_state, [-1, 3600, 2])
    p_state = p_state.flatten()
    return p_state #, v_state]


# def getPhaseFromAction(phases, act):
#     if act<2:
#         phases[int(act)] -= 5
#     elif act>2:
#         phases[int(act)-5] += 5
#     return phases
def getPhaseFromAction(phases, act):
    if act==0:
        phases[0] = 15
        phases[1]=25
    if act==1:
        phases[0] = 25
        phases[1] = 15
    if act==2:
        phases[0] = 20
        phases[1] = 20
    if act==3:
        phases[0] = 10
        phases[1] = 30
    if act==4:
        phases[0] = 30
        phases[1] = 10
    return phases

def action(tls, ph, wait_time):
    tls_id = tls[0]
    init_p = traci.trafficlight.getPhase(tls_id)
    prev = -1
    changed = False
    current_phases = ph
    p_state = np.zeros((40,40))
    step=0
    while traci.simulation.getMinExpectedNumber() > 0:
        c_p = traci.trafficlight.getPhase(tls_id)
        if c_p != prev:
            traci.trafficlight.setPhaseDuration(tls_id, ph[c_p]-1)
            prev = c_p
        if init_p != c_p:
            changed = True
        if changed:
            if c_p == init_p:
                break
        traci.simulationStep()
        step += 1
        global wait_time_map
        if step % 10 == 0:
            for veh_id in traci.vehicle.getIDList():
                wait_time_map[veh_id] = traci.vehicle.getAccumulatedWaitingTime(veh_id)
    wait_temp = dict(wait_time_map)
    wait_t = sum(wait_temp[x] for x in wait_temp)

    d = False
    if traci.simulation.getMinExpectedNumber() == 0:
        d = True

    r = wait_time - wait_t
    p_state = state()
    return p_state, r, d, wait_t









if __name__ == "__main__":
    eval_model = EvalModel(num_actions=5)
    target_model = TargetModel(num_actions=5)
    RL= DeepQNetwork(5, 1600, eval_model, target_model,
                   double_q=False,
                   learning_rate=0.0001,
                   reward_decay=0.9,
                   e_greedy=0.9,
                   replace_target_iter=100,
                   memory_size=20000,
                   batch_size=64,
                   # e_greedy_increment=0.0001,
                   # param_collect=self.collections
                   )
    total_steps = 0
    pre_train_steps = 2000
    num_episodes = 300
    init_phases = [20, 20]
    max_epLength = 5000
    all_reward=[]
    for i in range(1, num_episodes):
        wait_time_map = {}
        tls = reset()
        s = state()

        current_phases = list(init_phases)
        wait_time_map = {}
        wait_time = 0
        d = False
        rAll = 0
        j = 0

        while j < max_epLength:
            j += 1
            # legal_action = getLegalAction(current_phases)
            a = RL.choose_action(s)
            # print(a)
            # while legal_action[a] == -1:
            #     a = RL.choose_action(s)
            #     if legal_action[a] != -1:
            #         break
            ph = getPhaseFromAction(current_phases, a)
            # print('ph is',ph)
            s1, r, d, wait_time = action(tls, ph, wait_time)
            RL.store_transition(s, a, r, s1)
            total_steps += 1
            if total_steps > pre_train_steps and (total_steps % 5 == 0):
                RL.learn()

            rAll += r
            # print(rAll)
            s = s1

            if d == True:
                break
        end()
        all_reward.append(rAll)

    Time = range(0, len(all_reward))
    x_label = "Time"
    y_label = "Reward"
    plt.plot(Time, all_reward)
    plt.title("Simulations  and  Reward", fontsize=14)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()






