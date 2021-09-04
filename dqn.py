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

options = get_options()
if options.nogui:
    sumoBinary = checkBinary('sumo')
else:
    sumoBinary = checkBinary('sumo-gui')
sumoCmd = [sumoBinary, "-c", "single_route.sumocfg"]

#-------------------------------------------SUMO environment-----------------------------------------------------------#

class sumo_env:
    def __init__(self):
        self.action_space = ['0','1','2','3']
        self.n_actions = len(self.action_space)
        self.n_features = 40000



    def get_reward(self):
        car_ids = traci.vehicle.getIDList()
        r = 0
        for car_id in car_ids:
            r =r- traci.vehicle.getAccumulatedWaitingTime(car_id)
        return r

    def get_state(self):
        state = np.zeros([200, 200])
        for carID in traci.vehicle.getIDList():
            position = traci.vehicle.getPosition(carID)
            position_x = math.floor(position[0])
            position_y = math.floor(position[1])
            state[position_x, position_y] = 1
        s = state.flatten()
        return s

    def get_state_new(self):
        state=traci.vehicle.getIDCount()
        return state


    def getPhaseFromAction(phases, act):
        if act < 4:
            phases[int(act)] -= 5
        elif act > 4:
            phases[int(act) - 5] += 5
        return phases

    def step(self,action):
        # if action == 0:
        #     print('1')
        # #     if traci.trafficlight.getPhase('gneJ5')==0:
        # #         traci.trafficlight.setPhaseDuration('gneJ5', '45')
        # #         print('action is apply in 1')
        # elif action == 1:
        #     print('2')
        # #     if traci.trafficlight.getPhase('gneJ5')==2:
        # #         traci.trafficlight.setPhaseDuration('gneJ5', '40')
        # #         print('action is apply in 2')
        # #
        # elif action == 2:
        #     print('3')
        # #     if traci.trafficlight.getPhase('gneJ5')==0:
        # #         traci.trafficlight.setPhaseDuration('gneJ5', '40')
        # #         print('action is apply in 3')
        # #
        # elif action == 3:
        #     print('4')
        # #     if traci.trafficlight.getPhase('gneJ5')==2:
        # #         traci.trafficlight.setPhaseDuration('gneJ5', '45')
        # #         print('action is apply in 4')

        P=traci.vehicle.getIDCount()
        print('pre',P)



        traci.simulationStep()
        PP=traci.vehicle.getIDCount()
        print('next',PP)
        s_=self.get_state()
        if (self.get_state()==np.zeros([1, 40000])).all==True:
            done = True
        else:
            done = False
        reward=self.get_reward()
        print(reward)
        return s_,reward,done

    def close(self):
        traci.close()
#-------------------------------------------------DQN  Network---------------------------------------------------------#

class DQN:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=50000,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        # print(t_params)
        e_params = tf.get_collection('eval_net_params')
        # print(e_params)
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')
        with tf.variable_scope('eval_net'):
            c_names, n_l1, w_initializer, b_initializer = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))
        print('transition is ',transition)

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            # print(actions_value)
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)

        # print('******************8')
        # print(action)
        # print('**************')
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],  # fixed params
                self.s: batch_memory[:, :self.n_features],  # newest params
            })

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        # train eval network
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1






if __name__ == "__main__":
    env = sumo_env()
    RL = DQN(env.n_actions, env.n_features,
             learning_rate=0.01,
             reward_decay=0.9,
             e_greedy=0.9,
             replace_target_iter=200,
             memory_size=1000,
             # output_graph=True
             )
    step1=0
    traci.start([sumoBinary, "-c", "single_route.sumocfg"])
    for step in range(0, 3000):
        while True:
            print('-------------------------------------------------------')
            observation = env.get_state()
            print('state is', observation)
            action = RL.choose_action(observation)
            print('choose action is', action)
            env.step(action)
            P=traci.trafficlight.getPhaseDuration('gneJ5')
            print('time is',P)
            # observation_, reward, done = env.step(action)
        #     print('next state is', observation_)
        #     print('reward is', reward)
        #     print(done)
        #     P = traci.trafficlight.getPhase('gneJ5')
        #     print(P)
        #     RL.store_transition(observation, action, reward, observation_)
        #     if (step1 > 200) and (step1 % 5 == 0):
        #         print('kaishixuexi')
        #         RL.learn()
        #
        #     # swap observation
        #     observation = observation_
        #     if done:
        #         break
        #     step1 += 1
        #     print('step1 is', step1)
        #     print('-----------------------------------------------------')
        # print('simulation over')
    env.close()


