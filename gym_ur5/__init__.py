from gym.envs.registration import register

register(
    id='ur5-v0',
    entry_point='gym_ur5.envs:UR5Env',
    n_actions=3,
    n_states=6
)
