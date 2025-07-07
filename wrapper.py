import gym
from gym import spaces
import numpy as np

class DMCWrapper(gym.Env):
    def __init__(self, dmc_env):
        self._env = dmc_env
        obs_spec = self._env.observation_spec()
        action_spec = self._env.action_spec()

        # Flatten dict observation
        obs_dim = sum(np.prod(v.shape) for v in obs_spec.values())
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=action_spec.minimum,
            high=action_spec.maximum,
            dtype=np.float32
        )

    def reset(self):
        ts = self._env.reset()
        self._last_time_step = ts  # Save for info dict
        return self._flatten_obs(ts.observation)

    def step(self, action):
        ts = self._env.step(action)
        self._last_time_step = ts  # Save for info dict

        reward = ts.reward or 0.0
        done = ts.last()
        obs = self._flatten_obs(ts.observation)

        info = self._build_info(ts)

        return obs, reward, done, info

    def _flatten_obs(self, obs_dict):
        return np.concatenate([v.ravel() for v in obs_dict.values()])

    def _build_info(self, ts):
        physics = self._env.physics
        info = {}

        # Example info fields you might want:
        try:
            info['to_target_distance'] = physics.nose_to_target_dist()
            info['to_target_vector'] = physics.nose_to_target()
            info['joint_angles'] = physics.joints()
            info['body_velocities'] = physics.body_velocities()
        except Exception as e:
            info['warning'] = f"Could not build some info fields: {str(e)}"

        return info


class ActionMaskWrapper(gym.Wrapper):
    def __init__(self, env,p):
        super().__init__(env)
        '''
        p is the probability that a link is disabled
        '''
        self.env = env
        self.action_space = env.action_space
        self.a_dim = self.action_space.shape[0]
        self.mask = None
        self.p = p
    def reset(self):
        obs = self.env.reset()
        # self.mask = np.random.uniform(low =-1.0, high = 1.0, size = self.action_space.shape)
        self.mask = np.random.binomial(n=1,p=1-self.p,size=self.a_dim)
        return obs
    def step(self,action):
        assert self.mask is not None #Ensures that step is not called before reset
        masked_action = self.mask* action
        obs,reward,done,info = self.env.step(masked_action)
        info["action_mask"] = self.mask
        return obs, reward, done, info 
    

class SegmentActionWrapper(gym.Wrapper):
    def __init__(self, env, n_segments):
        super().__init__(env)
        self.env = env
        self.n_segments = n_segments

        # Total number of joints (original action space dimension)
        self.action_dim = self.env.action_space.shape[0]

        # Compute segment boundaries
        base = self.action_dim // n_segments
        remainder = self.action_dim % n_segments
        self.segment_sizes = [base + 1 if i < remainder else base for i in range(n_segments)]

        # New lower-dimensional action space
        low = np.full((n_segments,), self.env.action_space.low.min())
        high = np.full((n_segments,), self.env.action_space.high.max())
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def get_low_level_actions(self, action):
        '''
        Expand a low-dimensional action into the full action space
        by repeating each segment action value over its corresponding joints.
        '''
        assert len(action) == self.n_segments, "Action size must match number of segments"
        full_action = []
        for i, size in enumerate(self.segment_sizes):
            full_action.extend([action[i]] * size)
        return np.array(full_action, dtype=np.float32)

    def step(self, action):
        print('action: ', action)
        full_action = self.get_low_level_actions(action)
        print('full_action: ', full_action)
        return self.env.step(full_action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class FrameSkipWrapper(gym.Wrapper):
    def __init__(self, env, skip=4):
        """
        Repeat the same action for `skip` frames.

        Args:
            env (gym.Env): Environment to wrap.
            skip (int): Number of times to repeat each action.
        """
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        info = {}
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    