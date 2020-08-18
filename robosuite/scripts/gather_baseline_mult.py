"""
A script to collect a batch of human demonstrations that can be used
to generate a learning curriculum (see `demo_learning_curriculum.py`).

The demonstrations can be played back using the `playback_demonstrations_from_pkl.py`
script.
"""

import os
import shutil
import time
import argparse
import datetime
import h5py
from glob import glob
import numpy as np

import robosuite
import robosuite.utils.transform_utils as T
from robosuite.wrappers import IKWrapper
from robosuite.wrappers import DataCollectionWrapperBaseline

def merge_multiple_trajectories_npz(directory):
    """
    Gathers the demonstrations saved in @directory into a
    single hdf5 file, and another directory that contains the 
    raw model.xml files.

    The strucure of the hdf5 file is as follows.

    data (group)
        date (attribute) - date of collection
        time (attribute) - time of collection
        repository_version (attribute) - repository version used during collection
        env (attribute) - environment name on which demos were collected

        demo1 (group) - every demonstration has a group
            model_file (attribute) - name of corresponding model xml in `models` directory
            states (dataset) - flattened mujoco states
            joint_velocities (dataset) - joint velocities applied during demonstration
            gripper_actuations (dataset) - gripper controls applied during demonstration
            right_dpos (dataset) - end effector delta position command for
                single arm robot or right arm
            right_dquat (dataset) - end effector delta rotation command for
                single arm robot or right arm
            left_dpos (dataset) - end effector delta position command for
                left arm (bimanual robot only)
            left_dquat (dataset) - end effector delta rotation command for
                left arm (bimanual robot only)

        demo2 (group)
        ...

    Args:
        directory (str): Path to the directory containing raw demonstrations.
        out_dir (str): Path to where to store the hdf5 file and model xmls. 
            The model xmls will be stored in a subdirectory called `models`.
    """

    # store model xmls in this directory
    model_dir = os.path.join(directory,"combined")

    # May need to find way to not delete the original root directory
    if os.path.isdir(model_dir):
        shutil.rmtree(model_dir)
    os.makedirs(model_dir)


    num_eps = 0

    print("This is the directory: ", directory)

    print("Listing out all files within I think:", os.listdir(directory))

    for ep_directory in os.listdir(directory):

        print("name of episode directory found: ", ep_directory)

        state_paths = os.path.join(directory, ep_directory, "model_*.npz")
        episode_states = []
        episode_sims = []
        episode_actions = []
        episode_ep_rets = []
        episode_rews = []

        #sims = [] # For simulations, commented out since proof of concept was achieved

        max_ep_len = 0
        max_ep_ac_len = 0

        for state_file in sorted(glob(state_paths)):
            print("check if all files are read", state_file) # checking if it goes through
            dic = np.load(state_file, allow_pickle=True)

            print("shape of the current loaded episode: ", dic["obs"].shape)

            #env_name = str(dic["env"]) # removed for changed wrapper

            #Probably just duplicate the last observation and an empty action for the length of the trajectories 
            if ( dic["obs"].shape[1] > max_ep_len ):
                max_ep_len = dic["obs"].shape[1]
                #print("maximum episode length", max_ep_len)
            
            if ( dic["acs"].shape[1] > max_ep_ac_len ):
                max_ep_ac_len = dic["acs"].shape[1]
                #print("maximum action episode length", max_ep_ac_len)

            # Duplicate the last item in the second axis for all prev episodes
            for ep_ind in range(0, len(episode_states) ):
                # Episode states
                prev_episode = episode_states[ep_ind]
                duped_episode = episode_states[ep_ind] 
                
                prev_episode_sims = episode_sims[ep_ind]
                duped_episode_sims = episode_sims[ep_ind] 
                
                for dup_ind in range(0, (max_ep_len - prev_episode.shape[0]) ):
                    duped_episode = np.insert(duped_episode, duped_episode.shape[0] , duped_episode[ duped_episode.shape[0] - 1 ] , axis=0  ) # insert at the end? make sure that -1 inserts at the end for obj, replicate last state
                    duped_episode_sims = np.insert(duped_episode_sims, duped_episode_sims.shape[0] , duped_episode_sims[ duped_episode_sims.shape[0] - 1 ] , axis=0  ) # insert at the end? make sure that -1 inserts at the end for obj, replicate last state

                # Update the episode in the 3-tensor
                episode_states[ep_ind] = duped_episode
                episode_sims[ep_ind] = duped_episode_sims

            # THis part is for actions
            for ep_ac_ind in range(0, len(episode_actions) ):
                # Episode actions
                prev_action_episode = episode_actions[ep_ac_ind]
                duped_action_episode = episode_actions[ep_ac_ind]
                for dup_ac_ind in range(0, (max_ep_ac_len - prev_action_episode.shape[0]) ):
                    duped_action_episode = np.insert(duped_action_episode, duped_action_episode.shape[0] , np.zeros(duped_action_episode[ -1 ].shape[0] ) , axis=0  ) # insert at the end? make sure that -1 inserts at the end for obj, replicate last state

                # Update the episode in the 3-tensor
                episode_actions[ep_ac_ind] = duped_action_episode
            
            
            
            # Make sure the current episode is at max_length then put together on 0 axis
            current_obs = dic["obs"][0]
            pre_len = current_obs.shape[0]
            to_copy_state = current_obs[-1]

            # Current sims
            current_sims = dic["sims"][0]
            pre_len_sims = current_sims.shape[0]
            to_copy_sims = current_sims[-1]
            
            # Current actions
            current_actions = dic["acs"][0]
            #pre_acs_len = current_obs.shape[0]
            to_copy_action = np.zeros( current_actions[-1].shape[0] ) # a bunch of zeros

            #print("to copy action: ",to_copy_action)

            for cur_ep_ind in range(0, max_ep_len - pre_len):
                current_obs = np.insert(current_obs, current_obs.shape[0], to_copy_state, axis=0)
                current_sims = np.insert(current_sims, current_sims.shape[0], to_copy_sims, axis=0)
                current_actions = np.insert(current_actions, current_actions.shape[0], to_copy_action, axis=0)

            # Insert the newly modified episode into the modified 3-t
            episode_states.append(current_obs)
            episode_sims.append(current_sims)
            episode_actions.append(current_actions)




            #episode_states.append(dic["obs"]) # changed
            #sims.extend(dic["sims"]) # for simulations

            #episode_actions.append(dic["acs"])
            #episode_ep_rets.append(dic["ep_rets"])
            #episode_rews.append(dic["rews"])


        # Reshape into an np_array
        if(episode_states != []):
            #print("THis is episode states before: \n", episode_states )
            #print("THis is episode actions before: \n", episode_actions )
            
            #print("shape of the episode actions", len(episode_actions) , episode_actions[0].shape )
            
            episode_states = np.reshape(episode_states, ( len(episode_states), max_ep_len,-1 ) )
            episode_sims = np.reshape(episode_sims, ( len(episode_sims), max_ep_len,-1 ) )
            episode_actions = np.reshape(episode_actions, ( len(episode_actions), max_ep_ac_len,-1 ) )
            #print("shape of the episode states", episode_states.shape)
            #print("shape of the episode sims", episode_sims.shape)
            #print("shape of the episode actions", episode_actions.shape)

            #print("This is episode states after:  \n", episode_states )
            #print("This is episode actions after:  \n", episode_actions )


                
        #if len(states) == 0:
        #    continue

        # Set the path for the episode
        #num_eps += 1
        #print(num_eps)
        # copy over and rename model xml (npz in this case)
        ep_path = os.path.join(model_dir, "combined_{}.npz".format(num_eps))

        #print("check the dimension of the episode things and check why nothing is outputted: ", len(episode_states) )

        # Save npz files for the baseline gail training
        np.savez(
            ep_path,
            obs = episode_states,
            sims = episode_sims,
            acs = episode_actions,
            ep_rets = np.zeros(len(episode_states)),
            rews = np.zeros(( len(episode_states), max_ep_len )),
        )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        default=os.path.join(robosuite.models.assets_root, "demonstrations/120_shift5/"),
    ) # change from just demonstrations
    parser.add_argument("--environment", type=str, default="SawyerLift")
    args = parser.parse_args()

    # collect demonstrations
    merge_multiple_trajectories_npz(args.directory)
