from gym.envs.registration import register

register(
    id='ur5-v0',
    entry_point='gym_ur5.envs:UR5Env'
)
