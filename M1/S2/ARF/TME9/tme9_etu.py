import gym
import numpy as np
from collections import defaultdict
from ple.games.flappybird import FlappyBird
from ple import PLE
import gym_ple
import matplotlib.pyplot as plt



def test_gym():
    for game_name in ["Taxi-v2"]:#,"FlappyBird-v0"]:
        game = gym.make(game_name)
        game.reset()
        r = 0

        #trouver politique
        nb_state = game.observation_space.n
        nb_action = game.action_space.n
        V = np.zeros(nb_state)
        state = game.env.s
        transition = game.env.P
        #transition[state][action] : { (proba,état,reward,done) }

        V_iter = V.copy()
        policies= []

        for policy in range(50):

            #TODO : reinitialiser V avec la nouvelle politique

            for iter in range(100):
                V = V_iter.copy()
                for s in range(nb_state):
                    value = 0
                    for a in range(nb_action):
                        mdp = transition[s][a]
                        if(len(mdp) == 1):
                            mdp = mdp[0]
                            # proba = mdp[0]
                            s_prim = mdp[1]
                            reward = mdp[2]
                            # done = mdp[3]
                            value += reward + V[s_prim]
                    V_iter[s] = value 
            
            #TODO: déduire politique avec le V calculé
            policies.append(V_iter)

        print(policies)
        V_opt = max( [sum(pol) for pol in policies] )
        print(V_opt)

        # for i in range(100):
        #     action = game.action_space.sample() #remplacer le random
        #     observation,reward,done,info = game.step(action)
        #     r+=reward
        #     game.render()
        #     print("iter {} : action {}, reward {}, state {} ".format(i, action, reward, observation))
        #     if done:
        #         break
        # print(" Succes : {} , reward cumulatif : {} ".format(done,r))

        

test_gym()
