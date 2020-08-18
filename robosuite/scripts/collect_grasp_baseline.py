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

from robosuite.wrappers import GymWrapper

from baselines.gail import mlp_policy_sawyer
from baselines.gail.adversary import TransitionClassifier

import tensorflow as tf

# ----- Stuff for loading the policy for collecting the new grasp trajectories ------



def runner_1_traj(env, pi, timesteps_per_batch,
           stochastic_policy, save=False, reuse=False):

    obs_list = []
    acs_list = []
    len_list = []
    ret_list = []

    sims_list = [] # For simulations

    traj = traj_1_generator(pi, env, timesteps_per_batch, stochastic=stochastic_policy)
    obs, acs, ep_len, ep_ret = traj['ob'], traj['ac'], traj['ep_len'], traj['ep_ret']
    #sims = traj["sims"]# for simulations
    #obs_list.append(obs)
    #sims_list.append(sims)
    #acs_list.append(acs)
    len_list.append(ep_len)
    ret_list.append(ep_ret)

    if stochastic_policy:
        print('stochastic policy:')
    else:
        print('deterministic policy:')
    
    
    avg_len = sum(len_list)/len(len_list)
    avg_ret = sum(ret_list)/len(ret_list)
    print("Average length:", avg_len)
    print("Average return:", avg_ret)

    # Get the last joint positions to load for next part
    last_jpos = env._joint_positions # just making sure its being loaded correctly
    return avg_len, avg_ret, last_jpos


# Sample one trajectory (until trajectory end)
def traj_1_generator(pi, env, horizon, stochastic):

    t = 0
    ac = env.action_space.sample()  # not used, just so we have the datatype
    new = True  # marks if we're on first timestep of an episode

    ob = env.reset()

    #env.set_robot_joint_positions([0, -1.18, 0.00, 2.18, 0.00, 0.57, 1.5708]) # to match collected trajectories

    cur_ep_ret = 0  # return in current episode
    cur_ep_len = 0  # len of current episode

    # Initialize history arrays
    obs = []
    rews = []
    news = []
    acs = []

    # Create a sim storage for simulating such trajectory
    sims = []

    while True:
        ac, vpred = pi.act(stochastic, ob)
        obs.append(ob)
        news.append(new)
        acs.append(ac)

        # For simulation playback
        sims.append( env.sim.get_state().flatten() ) # Only works with robosuite environment

        ob, rew, new, _ = env.step(ac)
        rews.append(rew)

        cur_ep_ret += rew
        cur_ep_len += 1
        if new or t >= horizon:
            break
        t += 1

    obs = np.array(obs)
    sims = np.array(sims) # for simulations
    rews = np.array(rews)
    news = np.array(news)
    acs = np.array(acs)
    traj = {"ob": obs, "rew": rews, "new": news, "ac": acs,
            "ep_ret": cur_ep_ret, "ep_len": cur_ep_len, "sims": sims,}
    return traj


# Here is the part of data collection

def collect_human_trajectory(env, pre_env, pi, device):
    """
    Use the device (keyboard or SpaceNav 3D mouse) to collect a demonstration.
    The rollout trajectory is saved to files in npz format.
    Modify the DataCollectionWrapper wrapper to add new fields or change data formats.

    Args:
        env: environment to control
        device (instance of Device class): to receive controls from the device
    """

    timesteps_per_batch = 3500

    pre_env.reset() # reset pre environment
    _, _, last_jpos = runner_1_traj(pre_env, pi, timesteps_per_batch, stochastic_policy=False, save=False, reuse=False)
    pre_env.close() # close environment
    
    obs = env.reset()

    env.set_robot_joint_positions(last_jpos)


    env.viewer.set_camera(camera_id=2)
    env.render()

    
    
    is_first = True

    # episode terminates on a spacenav reset input or if task is completed
    reset = False
    task_completion_hold_count = -1 # counter to collect 10 timesteps after reaching goal
    
    
    device.start_control()
    # reorient the device rotation so that it begins in the correct orientation set by joint position
    device.rotation = env._right_hand_orn # hard reset from environment, is hacky

    while not reset:
        state = device.get_controller_state()
        dpos, rotation, grasp, reset = (
            state["dpos"],
            state["rotation"],
            state["grasp"],
            state["reset"],
        )

        # convert into a suitable end effector action for the environment
        current = env._right_hand_orn
        drotation = current.T.dot(rotation)  # relative rotation of desired from current
        dquat = T.mat2quat(drotation)
        grasp = grasp - 1.  # map 0 to -1 (open) and 1 to 0 (closed halfway)
        action = np.concatenate([dpos, dquat, [grasp]])

        #print(action) action creation above somehow realigns the gripper but why?
        # state controller rotation is somewhat wack and reverts the angle
        #print(current)

        obs, reward, done, info = env.step(action)

        # Test if the collected state space is what we want
        if is_first:
            is_first = False

            # We grab the initial model xml and state and reload from those so that
            # we can support deterministic playback of actions from our demonstrations.
            # This is necessary due to rounding issues with the model xml and with
            # env.sim.forward(). We also have to do this after the first action is 
            # applied because the data collector wrapper only starts recording
            # after the first action has been played.
            initial_mjstate = env.sim.get_state().flatten()
            xml_str = env.model.get_xml()
            env.reset_from_xml_string(xml_str)
            env.sim.reset()
            env.sim.set_state_from_flattened(initial_mjstate)
            env.sim.forward()
            env.viewer.set_camera(camera_id=2)


        env.render()


        if task_completion_hold_count == 0:
            break

        # state machine to check for having a success for 10 consecutive timesteps
        if env._check_success():
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1 # latched state, decrement count
            else:
                task_completion_hold_count = 10 # reset count on first success timestep
        else:
            task_completion_hold_count = -1 # null the counter if there's no success

    # cleanup for end of data collection episodes
    env.close()


def gather_demonstrations_as_npz(directory, out_dir):
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
    model_dir = os.path.join(out_dir, "models")
    if os.path.isdir(model_dir):
        shutil.rmtree(model_dir)
    os.makedirs(model_dir)

    # These parts are for hdf5 storage
    #hdf5_path = os.path.join(out_dir, "demo.hdf5")
    #f = h5py.File(hdf5_path, "w")

    # store some metadata in the attributes of one group
    #grp = f.create_group("data")

    num_eps = 0
    #env_name = None  # will get populated at some point

    for ep_directory in os.listdir(directory):

        state_paths = os.path.join(directory, ep_directory, "state_*.npz")
        states = []
        actions = []
        ep_rets = []
        rews = []

        sims = [] # For simulations
        #joint_velocities = []
        #gripper_actuations = []
        #right_dpos = []
        #right_dquat = []
        #left_dpos = []
        #left_dquat = []

        for state_file in sorted(glob(state_paths)):
            dic = np.load(state_file, allow_pickle=True)
            #env_name = str(dic["env"]) # removed for changed wrapper
            #print(dic["obs"].shape)

            states.extend(dic["obs_t"]) # changed
            sims.extend(dic["sims"]) # for simulations

            actions.extend(dic["acs"])
            ep_rets.extend(dic["ep_rets"])
            rews.extend(dic["rews"])
            #for ai in dic["action_infos"]:
            #    joint_velocities.append(ai["joint_velocities"])
            #    gripper_actuations.append(ai["gripper_actuation"])
            #    right_dpos.append(ai.get("right_dpos", []))
            #    right_dquat.append(ai.get("right_dquat", []))
            #    left_dpos.append(ai.get("left_dpos", []))
            #    left_dquat.append(ai.get("left_dquat", []))
                
        if len(states) == 0:
            continue

        # Delete the first actions and the last state. This is because when the DataCollector wrapper
        # recorded the states and actions, the states were recorded AFTER playing that action.
        del states[-1]
        del sims[-1] # for simulation
        del actions[0]
        del ep_rets[0]
        del rews[0]

        # Set the path for the episode
        num_eps += 1
        print(num_eps)
        # copy over and rename model xml (npz in this case)
        ep_path = os.path.join(model_dir, "model_{}.npz".format(num_eps))
        #shutil.copy(xml_path, model_dir)
        #os.rename(
        #    os.path.join(model_dir, "model.xml"),
        #    os.path.join(model_dir, "model_{}.xml".format(num_eps)),
        #)

        #print("length of action",len(actions) )

        # Save npz files for the baseline gail training
        np.savez(
            ep_path,
            obs = np.reshape( np.array(states), (1, np.array(states).shape[0], np.array(states).shape[1] ) ) , # reshape for the sake of loading data
            sims = np.reshape( np.array(sims), (1, np.array(sims).shape[0], np.array(sims).shape[1] ) ), # for simulation
            acs = np.reshape( np.array(actions), (1, np.array(actions).shape[0], np.array(actions).shape[1]) )  , # change to have the same sort of input as obs
            ep_rets = ep_rets,
            rews = rews
        )



        # For hdf5 files
        #del joint_velocities[0]
        #del gripper_actuations[0]
        #del right_dpos[0]
        #del right_dquat[0]
        #del left_dpos[0]
        #del left_dquat[0]

        #ep_data_grp = grp.create_group("demo_{}".format(num_eps))

        # store model file name as an attribute
        #ep_data_grp.attrs["model_file"] = "model_{}.xml".format(num_eps)

        # write datasets for states and actions
        #ep_data_grp.create_dataset("states", data=np.array(states))
        #ep_data_grp.create_dataset("joint_velocities", data=np.array(joint_velocities))
        #ep_data_grp.create_dataset(
        #    "gripper_actuations", data=np.array(gripper_actuations)
        #)
        #ep_data_grp.create_dataset("right_dpos", data=np.array(right_dpos))
        #ep_data_grp.create_dataset("right_dquat", data=np.array(right_dquat))
        #ep_data_grp.create_dataset("left_dpos", data=np.array(left_dpos))
        #ep_data_grp.create_dataset("left_dquat", data=np.array(left_dquat))

        # copy over and rename model xml
        #xml_path = os.path.join(directory, ep_directory, "model.xml")
        #shutil.copy(xml_path, model_dir)
        #os.rename(
        #    os.path.join(model_dir, "model.xml"),
        #    os.path.join(model_dir, "model_{}.xml".format(num_eps)),
        #)

    # write dataset attributes (metadata)
    #now = datetime.datetime.now()
    #grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    #grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    #grp.attrs["repository_version"] = robosuite.__version__
    #grp.attrs["env"] = env_name

    #f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        default=os.path.join(robosuite.models.assets_root, "demonstrations"),
    )
    parser.add_argument("--environment", type=str, default="SawyerLift")
    parser.add_argument("--device", type=str, default="keyboard")
    args = parser.parse_args()

    # create original environment
    env = robosuite.make(
        args.environment,
        ignore_done=True,
        use_camera_obs=False,
        has_renderer=True,
        control_freq=100,
        gripper_visualization=True,
        box_pos = [0.63522776, -0.3287869, 0.82162434], # shift2
        box_quat=[0.6775825618903728, 0, 0, 0.679425538604203], # shift2
    )

    pre_env = robosuite.make(
            args.environment,
            ignore_done=True,
            use_camera_obs=False,
            has_renderer=True,
            control_freq=100,
            gripper_visualization=True,
            box_pos = [0.63522776, -0.3287869, 0.82162434], # shift2
            box_quat=[0.6775825618903728, 0, 0, 0.679425538604203], # shift2
        )


    # enable controlling the end effector directly instead of using joint velocities
    pre_env = GymWrapper(pre_env)
    env = IKWrapper(env) # note cannot disable this or things go wack

    # wrap the environment with data collection wrapper
    tmp_directory = "~/Robotics/{}".format(str(time.time()).replace(".", "_")) # Change from temp to Robotics folder
    env = DataCollectionWrapperBaseline(env, tmp_directory)

    # initialize device
    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard()
        env.viewer.add_keypress_callback("any", device.on_press)
        env.viewer.add_keyup_callback("any", device.on_release)
        env.viewer.add_keyrepeat_callback("any", device.on_press)
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse

        device = SpaceMouse()
    else:
        raise Exception(
            "Invalid device choice: choose either 'keyboard' or 'spacemouse'."
        )

    # make a new timestamped directory
    t1, t2 = str(time.time()).split(".")
    new_dir = os.path.join(args.directory, "{}_{}".format(t1, t2))
    os.makedirs(new_dir)
    
    # Setup network
    # ----------------------------------------
    policy_hidden_size = 100 # matches default


    def policy_fn(name, ob_space, ac_space, reuse=False):
        return mlp_policy_sawyer.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space, reuse=reuse, hid_size=policy_hidden_size, num_hid_layers=2)
    
    ob_space = pre_env.observation_space
    ac_space = pre_env.action_space

    pi = policy_fn("pi", ob_space, ac_space, reuse=False)
    
    init_op = tf.compat.v1.global_variables_initializer()
    saver = tf.compat.v1.train.Saver(max_to_keep=5)

    # Hack for loading policies using tensorflow
    with tf.compat.v1.Session() as sess:            
        sess.run(init_op)
        # Load Checkpoint
        ckpt = tf.compat.v1.train.get_checkpoint_state('../../../reach_shift2/trpo_gail.transition_limitation_2500.SawyerLift.g_step_1.d_step_1.policy_entcoeff_0.adversary_entcoeff_0.001.seed_0/')
        saver.restore(sess, ckpt.model_checkpoint_path)

        
        
        # collect demonstrations (place this loop within the tf session)
        while True:
            collect_human_trajectory(env, pre_env, pi, device)
            gather_demonstrations_as_npz(tmp_directory, new_dir)
