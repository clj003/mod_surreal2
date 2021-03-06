"""
This file implements a wrapper for saving simulation states to disk.
This data collection wrapper is useful for collecting demonstrations.
"""

import os
import time
import numpy as np

from robosuite.wrappers import Wrapper
from robosuite.wrappers import IKWrapper


class DataCollectionWrapperBaseline(Wrapper):
    def __init__(self, env, directory, collect_freq=1, flush_freq=100):
        """
        Initializes the data collection wrapper.

        Args:
            env: The environment to monitor.
            directory: Where to store collected data.
            collect_freq: How often to save simulation state, in terms of environment steps.
            flush_freq: How frequently to dump data to disk, in terms of environment steps.
        """
        super().__init__(env)

        # the base directory for all logging
        self.directory = directory

        # in-memory cache for simulation states and action info
        self.states = []
        self.action_infos = []  # stores information about actions taken

        # how often to save simulation state, in terms of environment steps
        self.collect_freq = collect_freq

        # how frequently to dump data to disk, in terms of environment steps
        self.flush_freq = flush_freq

        # For simulations due to environment differences
        self.sim_states =[]

        if not os.path.exists(directory):
            print("DataCollectionWrapper: making new directory at {}".format(directory))
            os.makedirs(directory)

        # store logging directory for current episode
        self.ep_directory = None

        # remember whether any environment interaction has occurred
        self.has_interaction = False

    def _start_new_episode(self):
        """
        Bookkeeping to do at the start of each new episode.
        """

        # flush any data left over from the previous episode if any interactions have happened
        if self.has_interaction:
            self._flush()

        # timesteps in current episode
        self.t = 0
        self.has_interaction = False

    def _on_first_interaction(self):
        """
        Bookkeeping for first timestep of episode.
        This function is necessary to make sure that logging only happens after the first
        step call to the simulation, instead of on the reset (people tend to call
        reset more than is necessary in code).
        """

        self.has_interaction = True

        # create a directory with a timestamp
        t1, t2 = str(time.time()).split(".")
        self.ep_directory = os.path.join(self.directory, "ep_{}_{}".format(t1, t2))
        assert not os.path.exists(self.ep_directory)
        print("DataCollectionWrapper: making folder at {}".format(self.ep_directory))
        os.makedirs(self.ep_directory)

        # save the model xml
        xml_path = os.path.join(self.ep_directory, "model.xml")
        self.env.model.save_model(xml_path)

    def _flush(self):
        """
        Method to flush internal state to disk.
        """
        t1, t2 = str(time.time()).split(".")
        state_path = os.path.join(self.ep_directory, "state_{}_{}.npz".format(t1, t2))
        if hasattr(self.env, "unwrapped"):
            env_name = self.env.unwrapped.__class__.__name__
        else:
            env_name = self.env.__class__.__name__
        #print(np.array(self.states).shape) # for debugging, shows 34
        np.savez(
            state_path,
            obs_t=np.array(self.states),
            acs=self.action_infos,
            sims=np.array(self.sim_states),
            #env=env_name,
            ep_rets=np.zeros(len(self.states)),
            rews=np.zeros(len(self.states)),

        )
        self.states = []
        self.action_infos = []

        # For simulation rendering
        self.sim_states = []

    def reset(self):
        ret = super().reset()
        self._start_new_episode()
        return ret


    # Taken from the Gym Wrapper for correct states, not using sim in step
    # Can use if needed
    def _flatten_obs(self, obs_dict, verbose=False):
        """
        Filters keys of interest out and concatenate the information.
        Args:
            obs_dict: ordered dictionary of observations
        """
        ob_lst = []
        keys = ["robot-state", "object-state"] # added in such that gym wrapper works
        for key in obs_dict:
            if key in keys:
                if verbose:
                    print("adding key: {}".format(key))
                ob_lst.append(obs_dict[key])
        return np.concatenate(ob_lst)

    
    
    def step(self, action):
        ret = super().step(action)
        self.t += 1

        # on the first time step, make directories for logging
        if not self.has_interaction:
            self._on_first_interaction()

        # collect the current simulation state if necessary
        if self.t % self.collect_freq == 0:
            sim_state = self.env.sim.get_state().flatten()
            state = self._flatten_obs( self.env._get_observation() )  # changed so that matches the gym env for rollout
            # Check the env.sim to see which parts are extra in the gym wrapper
            
            #print("this env sim state size: ", state.shape) # changed from shape to size after state recording change
            #print("this is the env state size not simmed: ", self.env._get_observation().shape)

            self.states.append(state)
            self.sim_states.append(sim_state) # simulation state storage

            # Modify such that the specific parts of the actions are stored
            #self.action_infos.append(action)


            # Revert back because IK wrapper makes actions different
            if isinstance(self.env, IKWrapper):
                #add end effector actions in addition to the low-level joint actions
                info = {}
                info["joint_velocities"] = np.array(
                    self.controller.commanded_joint_velocities
                )
                info["right_dpos"] = np.array(action[:3])
                info["right_dquat"] = np.array(action[3:7])
                if self.env.mujoco_robot.name == "sawyer":
                    info["gripper_actuation"] = np.array(action[7:])
                elif self.env.mujoco_robot.name == "baxter":
                    info["gripper_actuation"] = np.array(action[14:])
                    info["left_dpos"] = np.array(action[7:10])  # add in second arm info
                    info["left_dquat"] = np.array(action[10:14])
            else:
                info = {}
                info["joint_velocities"] = np.array(action[: self.env.mujoco_robot.dof])
                info["gripper_actuation"] = np.array(
                    action[self.env.mujoco_robot.dof :]
                )
            
            
            self.action_infos.append( np.clip( np.concatenate([info["joint_velocities"], info["gripper_actuation"]]), -1, 1 )  ) # changed so the actual action is stored rather than a dic, clip the actions since actuation is also clipped, since envspec low high is -1,1
            #self.action_infos.append(info)

        # flush collected data to disk if necessary
        if self.t % self.flush_freq == 0:
            self._flush()

        return ret

    def close(self):
        """
        Override close method in order to flush left over data
        """
        self._start_new_episode()
        self.env.close()
