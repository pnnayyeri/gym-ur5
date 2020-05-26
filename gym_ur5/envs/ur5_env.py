import sim
import numpy as np
import time
import matplotlib.pyplot as plt

import gym
from gym import error, spaces


class UR5Env(gym.Env):

    def __init__(self): # n_actions:3 (target pos), n_states:6 (3pos+3force)
        self.metadata = {'render.modes': ['human']}
        super().__init__()
        sim.simxFinish(-1)
        for _ in range(5):
            self.clientID = sim.simxStart('127.0.0.1',19997,True,True,5000,5)
            if self.clientID is not -1:
                print('[INFO] Connected to CoppeliaSim.')
                break
        if self.clientID is -1:
            raise IOError('[ERROR] Could not connect to CoppeliaSim.')

        sim.simxSynchronous(self.clientID, True)
        sim.simxStartSimulation(self.clientID, sim.simx_opmode_oneshot)
        sim.simxGetPingTime(self.clientID)
        self.stepCount = 0
        self.reward = 0
        # self.n_substeps = n_substeps #not implemented
        # self.sim_timestep = 0.5      # set in coppeliaSim (not implemented)
        n_actions = 3
        n_states = 6
        self.action_space = spaces.Box(-1., 1, shape = (n_actions,), dtype = 'float32')
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(n_states,), dtype='float32')
        self.getHandles()
        sim.simxGetPingTime(self.clientID)

    def getHandle(self, name, ignoreError = False, attempts = 5):
        for _ in range(attempts):
            res, out_handle = sim.simxGetObjectHandle(self.clientID, name, sim.simx_opmode_blocking)
            if res == sim.simx_return_ok or ignoreError:
                print('[INFO] {} handle obtained.'.format(name))
                break
            sim.simxGetPingTime(self.clientID)
        if res!=sim.simx_return_ok and not ignoreError:
            print('[WARNING] Failed to find {} with error {}.'.format(name, res))
        return out_handle

    def __del__(self):
        self.close()

    def close(self):
        sim.simxStopSimulation(self.clientID, sim.simx_opmode_oneshot)
        sim.simxGetPingTime(self.clientID)
        sim.simxFinish(self.clientID)

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self._simulation_step()
        obs = self._get_obs()
        done = self._is_done()
        info = {}
        reward = self.getReward()
        return obs, reward, done, info

    def _set_action(self, action):
        actionX = action[0]
        actionY = action[1]
        actionZ = action[2]
        # print('[DATA] X:{:.4f}, Y:{:.4f}, Z:{:.4f}'.format(actionX,actionY,actionZ))
        sim.simxSetFloatSignal(self.clientID, 'actionX', actionX, sim.simx_opmode_oneshot)
        sim.simxSetFloatSignal(self.clientID, 'actionY', actionY, sim.simx_opmode_oneshot)
        sim.simxSetFloatSignal(self.clientID, 'actionZ', actionZ, sim.simx_opmode_oneshot)
        sim.simxGetPingTime(self.clientID)

    def _simulation_step(self):
        # for i in range(self.n_substeps):
        #     sim.simxSynchronousTrigger(self.clientID)

        while self.getTargetPos(False) is not self.getFinalPos(False): # keep triggering until movement is done
            sim.simxSynchronousTrigger(self.clientID)
        sim.simxGetPingTime(self.clientID)

    def _get_obs(self):
        X, Y, Z = self.getKinectXYZ(False)
        forces = self.getForce(False)
        return [X, Y, Z, forces[0][2], forces[1][2], forces[2][2]]

    def _is_done(self):
        pathIsDone = sim.simxGetFloatSignal(self.clientID,'movePathDone',sim.simx_opmode_buffer)[1]
        sim.simxGetPingTime(self.clientID)
        return True if pathIsDone==1 else False

    def reset(self):
        self.resetSim()
        self.prepareSim()

    def render(self, mode='human'):
        raise NotImplementedError

    def getHandles(self):
        '''
        print('getting tip handle...')
        self.tipHandle = self.getHandle('UR5_link7_visible')
        print('tip handle obtained successfully.')
        '''
        self.jointHandles = []
        print('[INFO] getting joint handles...')
        for i in range(1,7):
            self.jointHandles.append(self.getHandle('UR5_joint{}'.format(i)))
        print('[INFO] joint handles obtained successfully.')
        print('[INFO] getting dummy handles...')
        self.tipDummyHandle     = self.getHandle('Tip')
        self.targetDummyHandle  = self.getHandle('TargetDummy')
        self.testDummyHandle    = self.getHandle('TestDummy')
        # print('[INFO] dummy handles obtained successfully.')
        self.targetObjectHandle = self.getHandle('TargetObject')
        #print('[INFO] target object handle obtained successfully.')
        print('[INFO] getting force sensor handles...')
        self.fsHandles = []
        for i in range(1,4):
            self.fsHandles.append(self.getHandle('JacoHand_forceSens2_finger{}'.format(i)))
        print('[INFO] force sensor handles obtained successfully.')


    def getTipPos(self, initialize = True):
        if initialize:
            ret, pos = sim.simxGetObjectPosition(self.clientID, self.tipDummyHandle, -1, sim.simx_opmode_streaming)
        else:
            ret, pos = sim.simxGetObjectPosition(self.clientID, self.tipDummyHandle, -1, sim.simx_opmode_buffer)
        if ret == sim.simx_return_ok:
            return pos
        else:
            print('[WARNING] problem in getting tip position.')
            return -1

    def getTargetPos(self, initialize = True):
        if initialize:
            ret, pos = sim.simxGetObjectPosition(self.clientID, self.targetObjectHandle, -1, sim.simx_opmode_streaming)
        else:
            ret, pos = sim.simxGetObjectPosition(self.clientID, self.targetObjectHandle, -1, sim.simx_opmode_buffer)
        if ret == sim.simx_return_ok:
            return pos
        else:
            print('[WARNING] problem in getting target position.')
            return -1

    def getFinalPos(self, initialize = True):
        if initialize:
            ret, pos = sim.simxGetObjectPosition(self.clientID, self.testDummyHandle, -1, sim.simx_opmode_streaming)
        else:
            ret, pos = sim.simxGetObjectPosition(self.clientID, self.testDummyHandle, -1, sim.simx_opmode_buffer)
        if ret == sim.simx_return_ok:
            return pos
        else:
            print('[WARNING] problem in getting tip position.')
            return -1

    def getKinectXYZ(self, initialize = True):
        if initialize:
            X = sim.simxGetFloatSignal(self.clientID,'actX',sim.simx_opmode_streaming)[1]
            Y = sim.simxGetFloatSignal(self.clientID,'actY',sim.simx_opmode_streaming)[1]
            Z = sim.simxGetFloatSignal(self.clientID,'actZ',sim.simx_opmode_streaming)[1]
        else:
            X = sim.simxGetFloatSignal(self.clientID,'actX',sim.simx_opmode_buffer)[1]
            Y = sim.simxGetFloatSignal(self.clientID,'actY',sim.simx_opmode_buffer)[1]
            Z = sim.simxGetFloatSignal(self.clientID,'actZ',sim.simx_opmode_buffer)[1]
        sim.simxGetPingTime(self.clientID)
        return X,Y,Z

    def getForce(self, initialize = True):
        if initialize:
            F1 = sim.simxReadForceSensor(self.clientID,self.fsHandles[0],sim.simx_opmode_streaming)[2]
            F2 = sim.simxReadForceSensor(self.clientID,self.fsHandles[1],sim.simx_opmode_streaming)[2]
            F3 = sim.simxReadForceSensor(self.clientID,self.fsHandles[2],sim.simx_opmode_streaming)[2]
        else:
            F1 = sim.simxReadForceSensor(self.clientID,self.fsHandles[0],sim.simx_opmode_buffer)[2]
            F2 = sim.simxReadForceSensor(self.clientID,self.fsHandles[1],sim.simx_opmode_buffer)[2]
            F3 = sim.simxReadForceSensor(self.clientID,self.fsHandles[2],sim.simx_opmode_buffer)[2]
        sim.simxGetPingTime(self.clientID)
        return [F1,F2,F3]

    def getForceMagnitude(self, initialize = True):
        if initialize:
            F1 = sim.simxReadForceSensor(self.clientID,self.fsHandles[0],sim.simx_opmode_streaming)[2]
            F2 = sim.simxReadForceSensor(self.clientID,self.fsHandles[1],sim.simx_opmode_streaming)[2]
            F3 = sim.simxReadForceSensor(self.clientID,self.fsHandles[2],sim.simx_opmode_streaming)[2]
        else:
            F1 = sim.simxReadForceSensor(self.clientID,self.fsHandles[0],sim.simx_opmode_buffer)[2]
            F2 = sim.simxReadForceSensor(self.clientID,self.fsHandles[1],sim.simx_opmode_buffer)[2]
            F3 = sim.simxReadForceSensor(self.clientID,self.fsHandles[2],sim.simx_opmode_buffer)[2]
        sim.simxGetPingTime(self.clientID)
        return [np.linalg.norm(F1),np.linalg.norm(F2),np.linalg.norm(F3)]

    def initializeFunctions(self):
        self.getKinectXYZ(True)
        self.getForce(True)
        self.getForceMagnitude(True)
        self.getTipPos(True)
        self.getTargetPos(True)
        self.getFinalPos(True)

    def getReward(self):
        tipPos    = self.getTipPos(False)
        targetPos = self.getTargetPos(False)
        dist      = np.linalg.norm(np.array(tipPos)-np.array(targetPos))
        self.stepCount += 1
        self.reward    = self.reward + (1/self.stepCount)*((-dist)-self.reward)
        return self.reward


    def prepareSim(self):
        self.initializeFunctions()
        isTracking = sim.simxGetFloatSignal(self.clientID,'isTracking',sim.simx_opmode_streaming)[1]
        while isTracking != 1:
            sim.simxSynchronousTrigger(self.clientID)
            isTracking = sim.simxGetFloatSignal(self.clientID,'isTracking',sim.simx_opmode_buffer)[1]
            sim.simxGetPingTime(self.clientID)
            X,Y,Z = self.getKinectXYZ(True)
            actionX = X
            actionY = Y
            actionZ = Z
            # print(actionX,actionY,actionZ)
            # print('[DATA] X:{:.4f}, Y:{:.4f}, Z:{:.4f}'.format(actionX,actionY,actionZ))
            sim.simxSetFloatSignal(self.clientID, 'actionX', actionX, sim.simx_opmode_oneshot)
            sim.simxSetFloatSignal(self.clientID, 'actionY', actionY, sim.simx_opmode_oneshot)
            sim.simxSetFloatSignal(self.clientID, 'actionZ', actionZ, sim.simx_opmode_oneshot)
            sim.simxGetPingTime(self.clientID)
        print('[INFO] simulation is ready for tracking.')


    def resetSim(self):
        print('[INFO] reseting sim...')
        ret = sim.simxStopSimulation(self.clientID,sim.simx_opmode_blocking)
        if ret == sim.simx_return_ok:
            print('[INFO] sim reset successfully.')
            time.sleep(1)
        else:
            raise IOError('[ERROR] problem in reseting sim...')
        sim.simxGetPingTime(self.clientID)
        print('[INFO] starting sim in synchronous mode...')
        sim.simxSynchronous(self.clientID, True)
        sim.simxGetPingTime(self.clientID)
        ret = sim.simxStartSimulation(self.clientID,sim.simx_opmode_oneshot)
        if ret == sim.simx_return_ok:
            print('[INFO] sim started successfully.')
        sim.simxGetPingTime(self.clientID)
        sim.simxClearFloatSignal(self.clientID, 'actionX', sim.simx_opmode_blocking)
        sim.simxClearFloatSignal(self.clientID, 'actionY', sim.simx_opmode_blocking)
        sim.simxClearFloatSignal(self.clientID, 'actionZ', sim.simx_opmode_blocking)
        sim.simxGetPingTime(self.clientID)
        self.stepCount = 0
        self.reward = 0

    def doTracking(self):
        self.resetSim()
        self.prepareSim()

        pathIsDone = sim.simxGetFloatSignal(self.clientID,'movePathDone',sim.simx_opmode_streaming)[1]
        posX = []
        posY = []
        posZ = []
        tactile1 = []
        reward = []

        while pathIsDone == 0:
            X,Y,Z = self.getKinectXYZ(False)
            # forces = self.getForce(False)
            # diffForce = (forces[0][0] + ((forces[1][0] + forces[2][0]) / 2))/2
            # print('[DATA] diff force: {:.4f}'.format(diffForce))
            actionX = X # + 0.005*diffForce
            actionY = Y
            actionZ = Z
            # print('[DATA] X:{:.4f}, Y:{:.4f}, Z:{:.4f}'.format(actionX,actionY,actionZ))
            sim.simxSetFloatSignal(self.clientID, 'actionX', actionX, sim.simx_opmode_oneshot)
            sim.simxSetFloatSignal(self.clientID, 'actionY', actionY, sim.simx_opmode_oneshot)
            sim.simxSetFloatSignal(self.clientID, 'actionZ', actionZ, sim.simx_opmode_oneshot)
            sim.simxGetPingTime(self.clientID)
            sim.simxSynchronousTrigger(self.clientID)
            forces = self.getForceMagnitude(False)
            posX.append(X)
            posY.append(Y)
            posZ.append(Z)
            tactile1.append(forces[0])
            reward.append(self.getReward())
            sim.simxGetPingTime(self.clientID)
            pathIsDone = sim.simxGetFloatSignal(self.clientID,'movePathDone',sim.simx_opmode_buffer)[1]
            # time.sleep(0.5)
        print('[INFO] tracking is done.')
        print('[DATA] accumulated reward: {}'.format(np.sum(np.array(reward))))
        sim.simxStopSimulation(self.clientID,sim.simx_opmode_blocking)
