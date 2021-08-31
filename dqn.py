import os
import sys
import optparse
import tensorflow as tf
import numpy as np
import math
import pandas as pd

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci
from sumolib import checkBinary
import datetime

def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options

#-------------------------------------------SUMO environment-----------------------------------------------------------#

class sumo_env:
    def __init__(self):
        self.action_space = ['0','1','2','3']
        self.n_actions = len(self.action_space)
        self.n_features = 40000

    def reset(self):
        traci.start([sumoBinary, "-c", "single_route.sumocfg"])
        # s=self.get_state()
        # return s

    def get_reward(self):
        car_ids = traci.vehicle.getIDList()
        reward = 0
        for car_id in car_ids:
            reward =reward- traci.vehicle.getAccumulatedWaitingTime(car_id)
        return reward

    def get_state(self):
        state = np.zeros([200, 200])
        for carID in traci.vehicle.getIDList():
            position = traci.vehicle.getPosition(carID)
            position_x = math.floor(position[0])
            position_y = math.floor(position[1])
            state[position_x, position_y] = 1
        s = state.flatten()
        return s

    def getPhaseFromAction(phases, act):
        if act < 4:
            phases[int(act)] -= 5
        elif act > 4:
            phases[int(act) - 5] += 5
        return phases

    def step(self,action):
        if action == 0:
            if traci.trafficlight.getPhase('gneJ5')==0:
                traci.trafficlight.setPhaseDuration('gneJ5', '45')
        elif action == 1:
            if traci.trafficlight.getPhase('gneJ5')==2:
                traci.trafficlight.setPhaseDuration('gneJ5', '40')

        elif action == 2:
            if traci.trafficlight.getPhase('gneJ5')==0:
                traci.trafficlight.setPhaseDuration('gneJ5', '40')

        elif action == 3:
            if traci.trafficlight.getPhase('gneJ5')==2:
                traci.trafficlight.setPhaseDuration('gneJ5', '45')


        traci.simulationStep()
        s_=self.get_state()
        if (self.get_state()==np.zeros([1, 40000])).all==True:
            done = True
        else:
            done = False
        reward=self.get_reward()
        # print(reward)
        return s_,reward,done

    def close(self):
        traci.close()







if __name__ == "__main__":
    env = sumo_env()
    options = get_options()
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')
    env.reset()
    for step in range(0,3600):
        PI = traci.trafficlight.getPhaseDuration('gneJ5')
        print(PI)
        print('222')

        traci.simulationStep()
    traci.close()
