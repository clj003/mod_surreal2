"""
This file implements a wrapper for facilitating compatibility with OpenAI gym.
This is useful when using these environments with code that assumes a gym-like 
interface.
"""

import numpy as np
from gym import spaces
from robosuite.wrappers import Wrapper


class GymWrapper(Wrapper):
    env = None

    def __init__(self, env, keys=None):
        """
        Initializes the Gym wrapper.

        Args:
            env (MujocoEnv instance): The environment to wrap.
            keys (list of strings): If provided, each observation will
                consist of concatenated keys from the wrapped environment's
                observation dictionary. Defaults to robot-state and object-state.
        """
        self.env = env

        #print("mujoco env obs size: ", self.env._get_observation().keys())
        #countingstates = 0
        #for kay in self.env._get_observation().keys():
        #    countingstates += self.env._get_observation()[kay].shape[0]
        #    print(kay, self.env._get_observation()[kay].shape[0] )
        #print("number of counted states: ", countingstates)


        if keys is None:
            assert self.env.use_object_obs, "Object observations need to be enabled."
            keys = ["robot-state", "object-state"]
        self.keys = keys

        # set up observation and action spaces
        flat_ob = self._flatten_obs(self.env.reset(), verbose=True)
        self.obs_dim = flat_ob.size

        #print("flattened dimensions of fed in obs: ",self.obs_dim)

        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low=low, high=high)

        #print("gym wrapper obs space size: ",self.observation_space.shape) # for debugging, ends up as 40

        low, high = self.env.action_spec
        self.action_space = spaces.Box(low=low, high=high)

        print("gym wrapper high and low values of env: ",high , low)

        # Set up a reward range, seed, spec and metadata for compatibility with baseline
        self.reward_range = (-float('inf'), float('inf'))
        self.metadata = {'render.modes': []} # figure out if this is useful at a later point
        self.spec = None

    def _flatten_obs(self, obs_dict, verbose=False):
        """
        Filters keys of interest out and concatenate the information.

        Args:
            obs_dict: ordered dictionary of observations
        """
        ob_lst = []
        for key in obs_dict:
            if key in self.keys:
                if verbose:
                    print("adding key: {}".format(key))
                ob_lst.append(obs_dict[key])
        return np.concatenate(ob_lst)

    def reset(self):
        ob_dict = self.env.reset()
        return self._flatten_obs(ob_dict)

    def step(self, action):
        # This part clips the action so that the range of the action space is respected.
        ob_dict, reward, done, info = self.env.step(action)
        return self._flatten_obs(ob_dict), reward, done, info
