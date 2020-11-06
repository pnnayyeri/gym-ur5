# gym-ur5
UR5 robot with Jaco hand as gripper for HRI project


## Setting up
1. Install gym: ```pip install gym```
2. Install pytorch (please refer to its website for customized installation command): https://pytorch.org/
2. Clone/donwload the repo.
3. cd to the cloned repo: ```cd gym-ur5``` and install it using: ```pip install -e .```
4. Open coppelia scene ```UR5-Gripper-thread-pyapi.ttt``` in coppeliaSim.
5. Use one of agent's jupyter notebook implementation to create the agent and create the gym environment.

## Source
Deep RL Agent implementations are derived from cyoon1729 github repo: https://github.com/cyoon1729/Reinforcement-learning
