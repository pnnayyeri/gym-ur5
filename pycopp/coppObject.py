# This can be used if the coppelia connection is already made and clientID is passed
# V-REP must be running in synchronuous mode to use this.
# So the trigger to continue simulation must be done outside of this class
# To ensure data is sync'ed properly. Use the following two commands outside of this class:
#		sim.simxSynchronousTrigger(clientID)
#		sim.simxGetPingTime(cliendID)

try:
	import sim
except:
	print ('--------------------------------------------------------------')
	print ('"sim.py" could not be imported. This means very probably that')
	print ('either "sim.py" or the remoteApi library could not be found.')
	print ('Make sure both are in the same folder as this file,')
	print ('or appropriately adjust the file "sim.py"')
	print ('--------------------------------------------------------------')
	print ('')

import numpy as np
import transforms3d.quaternions as quaternions
import transforms3d.euler as euler


class coppObject:
	def __init__(self, clientID):
		self.clientID = clientID

	#Wrapper for simxSetIntegerSignal
	def setIntegerSignal(self, integerString, value, ignoreError = False):
		res = sim.simxSetIntegerSignal(self.clientID, integerString, value,  sim.simx_opmode_oneshot)

		if res!=sim.simx_return_ok and not ignoreError:
			print('Failed to set integer signal {}'.format(integerString))
			print(res)

	#Wrapper for simxGetObjectParent
	def getParent(self, handle_obj, ignoreError = False, initialize = False):

		if initialize:
			sim.simxGetObjectParent(self.clientID, handle_obj, sim.simx_opmode_streaming)

		res, out = sim.simxGetObjectParent(self.clientID, handle_obj,  sim.simx_opmode_buffer)

		if res!=sim.simx_return_ok and not ignoreError:
			print('Failed to get parent from object')
			print(res)

		return out

	#Wrapper for simxSetObjectParent
	def setParent(self, handle_obj, handle_parent, retainPosition, ignoreError = False):
		res = sim.simxSetObjectParent(self.clientID, handle_obj, handle_parent, retainPosition, sim.simx_opmode_oneshot)

		if res!=sim.simx_return_ok and not ignoreError:
			print('Failed to set parent for object')
			print(res)


	#Wrapper for simxSetBooleanParameter
	# Look here to see what booleans you can modify: http://www.coppeliarobotics.com/helpFiles/en/apiConstants.htm#booleanParameters
	def setBooleanParameter(self, paramIdentifier, paramValue, ignoreError = False):
		res = sim.simxSetBooleanParameter(self.clientID, paramIdentifier, paramValue, sim.simx_opmode_blocking)

		if res!=sim.simx_return_ok and not ignoreError:
			print('Failed to set {}'.format(paramIdentifier))
			print(res)

	#Wrapper for simxGetBooleanParameter
	def getBooleanParameter(self, paramIdentifier, ignoreError = False, initialize = False):
		if initialize:
			sim.simxGetBooleanParameter(self.clientID, paramIdentifier, sim.simx_opmode_streaming)

		res, out = sim.simxGetBooleanParameter(self.clientID, paramIdentifier,  sim.simx_opmode_buffer)

		if res!=sim.simx_return_ok and not ignoreError:
			print('Failed to get {}'.format(paramIdentifier))
			print(res)

		return out

	#Wrapper for simxGetObjectHandle
	def getHandle(self, name, ignoreError = False, attempts = 5):

		for _ in range(attempts):
            res, out_handle = sim.simxGetObjectHandle(self.clientID, name, sim.simx_opmode_blocking)
            if res == sim.simx_return_ok or ignoreError:
                print('{} handle obtained'.format(name))
                break
			sim.simxGetPingTime(self.clientID)
		if res!=sim.simx_return_ok and not ignoreError:
			print('Failed to find {} with error {}'.format(name, res))

		return out_handle

	#Wrapper for simxGetVisionSensorImage
	def getVisionSensorImage(self, handle, rgb = True, ignoreError = False, initialize = False):
		if rgb:
			b = 0
		else:
			b = 1

		if initialize:
			sim.simxGetVisionSensorImage(self.clientID, handle, b, sim.simx_opmode_streaming)

		res, image_resolution, image_data = sim.simxGetVisionSensorImage(self.clientID, handle, b, sim.simx_opmode_buffer)

		if res!=sim.simx_return_ok and not ignoreError:
			print('Failed to find {} with error {}'.format(name, res))

		return image_data, image_resolution

	#Wrapper for simxGetCollisionHandle
	def getCollisionHandle(self, name, ignoreError = False):
		res, out_handle = sim.simxGetCollisionHandle(self.clientID, name, sim.simx_opmode_blocking)

		if res != sim.simx_return_ok and not ignoreError:
			print('Failed to find {} with error {}'.format(name, res))

		return out_handle

	#Wrapper for simxReadCollision
	def checkCollision(self, handle, ignoreError = False, initialize = False):
		if initialize:
			sim.simxReadCollision(self.clientID, handle, sim.simx_opmode_streaming)

		res, collision = sim.simxReadCollision(self.clientID, handle, sim.simx_opmode_buffer)

		if res!=sim.simx_return_ok and not ignoreError:
			print('Failed to check collision')
			print(res)

		return collision

	#Wrapper for simxGetJointPosition
	def getJointPosition(self, handle, ignoreError = False, initialize = False):
		if initialize:
			sim.simxGetJointPosition(self.clientID, handle,  sim.simx_opmode_streaming)

		res, out_pos = sim.simxGetJointPosition(self.clientID, handle,  sim.simx_opmode_buffer)

		if res!=sim.simx_return_ok and not ignoreError:
			print('Failed to get joint angle')
			print(res)

		return out_pos

	#Wrapper to get joint velocity
	def getJointVelocity(self, handle, ignoreError = False, initialize = False):
		if initialize:
			sim.simxGetObjectFloatParameter(self.clientID, handle, 2012, sim.simx_opmode_streaming)

		res,velocity=sim.simxGetObjectFloatParameter(self.clientID, handle, 2012, sim.simx_opmode_buffer)

		if res!=sim.simx_return_ok and not ignoreError:
			print('Failed to get joint velocity')
			print(res)

		return velocity

	#Wrapper for simxSetJointPosition
	def setJointPosition(self, handle, position, ignoreError = False, initialize = False):
		res = sim.simxSetJointPosition(self.clientID, handle, position, sim.simx_opmode_oneshot)

		if res!=sim.simx_return_ok and not ignoreError:
			print('Failed to set joint angle')
			print(res)

	#Wrapper for simxGetObjectPosition and simxGetObjectQuaternion
	def getPoseAtHandle(self, targetHandle, refHandle, ignoreError = False, initialize = False):
		if initialize:
			sim.simxGetObjectPosition(  self.clientID, targetHandle, refHandle, sim.simx_opmode_streaming)
			sim.simxGetObjectQuaternion(self.clientID, targetHandle, refHandle, sim.simx_opmode_streaming)

		res1, pos  = sim.simxGetObjectPosition(  self.clientID, targetHandle, refHandle, sim.simx_opmode_buffer)
		res2, quat = sim.simxGetObjectQuaternion(self.clientID, targetHandle, refHandle, sim.simx_opmode_buffer)

		if res1!=sim.simx_return_ok and not ignoreError:
			print('Failed to get position')
			print(res1)
		if res2!=sim.simx_return_ok and not ignoreError:
			print('Failed to get orientation')
			print(res2)

		return np.array(pos), np.array(quat)

	#Wrapper for simxGetObjectVelocity
	def getVelocityAtHandle(self, targetHandle, ignoreError = False, initialize = False):
		if initialize:
			sim.simxGetObjectVelocity(self.clientID, targetHandle, sim.simx_opmode_streaming)

		res, l_vel, r_vel = sim.simxGetObjectVelocity(self.clientID, targetHandle, sim.simx_opmode_buffer)

		if res!=sim.simx_return_ok and not ignoreError:
			print('Failed to get velocity')
			print(res)

		return np.array(l_vel), np.array(r_vel)

	#Wrapper for simxSetObjectPosition and simxSetObjectQuaternion
	def setPoseAtHandle(self, targetHandle, refHandle, pos, quat, ignoreError = False):
		res1 = sim.simxSetObjectPosition(  self.clientID, targetHandle, refHandle, pos,  sim.simx_opmode_oneshot)
		res2 = sim.simxSetObjectQuaternion(self.clientID, targetHandle, refHandle, quat, sim.simx_opmode_oneshot)

		if res1!=sim.simx_return_ok and not ignoreError:
			print('Failed to set position')
			print(res1)
		if res2!=sim.simx_return_ok and not ignoreError:
			print('Failed to set orientation')
			print(res2)


	def posquat2Matrix(self, pos, quat):
		T = np.eye(4)
		T[0:3, 0:3] = quaternions.quat2mat([quat[-1], quat[0], quat[1], quat[2]])
		T[0:3, 3] = pos

		return np.array(T)

	def matrix2posquat(self,T):
		pos = T[0:3, 3]
		quat = quaternions.mat2quat(T[0:3, 0:3])
		quat = [quat[1], quat[2], quat[3], quat[0]]

		return np.array(pos), np.array(quat)


	#Everything is defined here as the same as V-rep
	#RPY is relative
	#Quaternion is (i, j, k, w)
	#These functions are added just to make life easier and reduce number of errors...
	def euler2Quat(self, rpy):
		q = euler.euler2quat(rpy[0], rpy[1], rpy[2],  axes = 'rxyz')
		return np.array([q[1], q[2], q[3], q[0]])

	def quat2Euler(self, quat):
		return np.array(euler.quat2euler([quat[3], quat[0], quat[1], quat[2]], axes = 'rxyz'))
