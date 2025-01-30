import isaacgym
from hydra._internal.utils import get_args_parser
from isaacgymenvs.utils.utils import set_np_formatting, set_seed

from algorithms.ppo import PPO
from tasks import load_isaacgym_env
from utils.config import get_args, load_cfg
# from utils.vis import Visualizer # use visualizer requires to install sim-web-visualizer


import torch
from isaacgym import gymapi, gymtorch
from tasks.isaacgym_utils import (
    ActionSpec,
    ObservationSpec,
    draw_axes,
    draw_boxes,
    get_action_indices,
    ik,
    orientation,
    position,
    print_action_space,
    print_asset_options,
    print_dof_properties,
    print_links_and_dofs,
    print_observation_space,
    random_orientation_within_angle,
    to_torch,
)



if __name__ == "__main__":
    set_np_formatting()

    # argparse
    parser = get_args_parser()
    parser.add_argument("--num_iterations", type=int, default=1000, help="Number of iterations to run")
    parser.add_argument("--seed", type=int, default=0, help="Seed Number")
    parser.add_argument("--run_device_id", type=int, default=0, help="Device id")

    parser.add_argument(
        "--torch_deterministic",
        action="store_true",
        default=False,
        help="Apply additional PyTorch settings for more deterministic behaviour",
    )
    parser.add_argument("--test", action="store_true", default=False, help="Run trained policy, no training")
    parser.add_argument("--con", action="store_true", default=False, help="whether continue train")
    parser.add_argument(
        "--web_visualizer_port", type=int, default=-1, help="port to visualize in web visualizer, set to -1 to disable"
    )
    parser.add_argument("--collect_demo_num", type=int, default=-1, help="collect demo num")
    parser.add_argument("--eval_times", type=int, default=5, help="Eval times for each object")
    parser.add_argument("--max_iterations", type=int, default=-1, help="Max iterations for training")

    parser.add_argument(
        "--cfg_train", type=str, default="ShadowHandFunctionalManipulationUnderarmPPO", help="Training config"
    )

    parser.add_argument("--logdir", type=str, default="", help="Log directory")
    parser.add_argument("--method", type=str, default="", help="Method name")
    parser.add_argument("--exp_name", type=str, default="", help="Exp name")
    parser.add_argument("--model_dir", type=str, default="", help="Choose a model dir")
    parser.add_argument("--eval_name", type=str, default="", help="Eval metric saving name")
    parser.add_argument("--vis_env_num", type=int, default=0, help="Number of env to visualize")

    # score matching parameter
    parser.add_argument("--t0", type=float, default=0.05, help="t0 for sample")
    parser.add_argument("--hidden_dim", type=int, default=1024, help="num of hidden dim")
    parser.add_argument("--embed_dim", type=int, default=512, help="num of embed_dim")
    parser.add_argument("--score_mode", type=str, default="target", help="score mode")
    parser.add_argument("--space", type=str, default="euler", help="angle space")
    parser.add_argument("--cond_on_arm", action="store_true", help="dual score")
    parser.add_argument("--n_obs_steps", type=int, default=2, help="observation steps")
    parser.add_argument("--n_action_steps", type=int, default=1)
    parser.add_argument("--n_prediction_steps", type=int, default=4)
    parser.add_argument("--encode_state_type", type=str, default="all", help="encode state type")
    parser.add_argument(
        "--score_action_type",
        type=str,
        default="all",
        metavar="SCORE_ACTION_TYPE",
        help="score action type: arm, hand, all",
    )
    parser.add_argument(
        "--action_mode", type=str, default="rel", metavar="ACTION_MODE", help="action mode: rel, abs, obs"
    )
    parser.add_argument(
        "--score_model_path",
        type=str,
        default="/home/thwu/Projects/func-mani/ckpt/score_all.pt",
        help="pretrain score model path",
    )

    args = parser.parse_args()

    # if args.web_visualizer_port != -1:
    #     visualizer = Visualizer(args.web_visualizer_port)

    sim_device = f"cuda:{args.run_device_id}"
    rl_device = f"cuda:{args.run_device_id}"

    cfg_train, logdir = load_cfg(args)

    # set the seed for reproducibility
    set_seed(args.seed)
    """Change for different methods."""
    action_space = ["hand_rotation"]
    
    if args.exp_name == "PPO":
        if "env_mode=orn" in args.overrides:
            obs_space = [
                "shadow_hand_dof_position",
                "shadow_hand_dof_velocity",
                "fingertip_position",
                "fingertip_orientation",
                "fingertip_linear_velocity",
                "fingertip_angular_velocity",
                "object_position",
                "object_orientation",
                "object_linear_velocity",
                "object_angular_velocity",
                "object_target_orn",
                "orientation_error",
            ]
        elif "env_mode=relpose" in args.overrides:
            obs_space = [
                "shadow_hand_dof_position",
                "shadow_hand_dof_velocity",
                "fingertip_position_wrt_palm",
                "fingertip_orientation_wrt_palm",
                "fingertip_linear_velocity",
                "fingertip_angular_velocity",
                "object_position_wrt_palm",
                "object_orientation_wrt_palm",
                "object_linear_velocity",
                "object_angular_velocity",
                "object_target_relpose",
                "orientation_error",
                "position_error",
            ]
        elif "env_mode=relposecontact" in args.overrides:
            obs_space = [
                "shadow_hand_dof_position",
                "shadow_hand_dof_velocity",
                "fingertip_position_wrt_palm",
                "fingertip_orientation_wrt_palm",
                "fingertip_linear_velocity",
                "fingertip_angular_velocity",
                "object_position_wrt_palm",
                "object_orientation_wrt_palm",
                "object_linear_velocity",
                "object_angular_velocity",
                "object_target_relposecontact",
                "orientation_error",
                "position_error",
                "fingerjoint_error",
                # "pointcloud_wrt_palm"
            ]
        elif "env_mode=pgm" in args.overrides:
            obs_space = [
                # "shadow_hand_position",
                # "shadow_hand_orientation",
                "ur_endeffector_position", # robot arm
                "ur_endeffector_orientation",
                "shadow_hand_dof_position", # shadow hand
                "shadow_hand_dof_velocity",
                "fingertip_position_wrt_palm",
                "fingertip_orientation_wrt_palm",
                "fingertip_linear_velocity",
                "fingertip_angular_velocity",
                "object_position_wrt_palm", # object
                "object_orientation_wrt_palm",
                "object_position",
                "object_orientation",
                "object_linear_velocity",
                "object_angular_velocity",
                "object_target_relposecontact",
                "position_error",
                "orientation_error",
                "fingerjoint_error",
                "object_bbox",
                # "object_category",
                # "pointcloud_wrt_palm"
            ]
            action_space = ["wrist_translation", "wrist_rotation", "hand_rotation"]
        # training parameter
        cfg_train["learn"]["nsteps"] = 8
        cfg_train["learn"]["noptepochs"] = 5
        cfg_train["learn"]["nminibatches"] = 4
        cfg_train["learn"]["desired_kl"] = 0.016
        cfg_train["learn"]["gamma"] = 0.99
        cfg_train["learn"]["clip_range"] = 0.1
    elif args.exp_name == "ppo_real":
        obs_space = [
            "ur_endeffector_position",
            "ur_endeffector_orientation",
            "shadow_hand_dof_position",
            "object_position_wrt_palm",
            "object_orientation_wrt_palm",
            "object_target_relposecontact",
            # "object_bbox",
        ]
        action_space = ["wrist_translation", "wrist_rotation", "hand_rotation"]
        # training parameter
        cfg_train["learn"]["nsteps"] = 8
        cfg_train["learn"]["noptepochs"] = 5
        cfg_train["learn"]["nminibatches"] = 4
        cfg_train["learn"]["desired_kl"] = 0.016
        cfg_train["learn"]["gamma"] = 0.99
        cfg_train["learn"]["clip_range"] = 0.1
    else:
        raise NotImplementedError(f"setting {args.exp_name} not supported") 
    """
    load env
    """
    # override env args
    args.overrides.append(f"seed={args.seed}")
    args.overrides.append(f"sim_device={sim_device}")
    args.overrides.append(f"rl_device={rl_device}")
    args.overrides.append(f"obs_space={obs_space}")
    args.overrides.append(f"action_space={action_space}")
    
    # Load and wrap the Isaac Gym environment
    env = load_isaacgym_env(
        task_name="ShadowHandFunctionalManipulationUnderarm", args=args
    )  # preview 3 and 4 use the same loader


    # load object, hand and robot arm
    # get target relative hand_obj pose
    # use absolute object pose to calculate target arm pose
    # calculate the relative movement of the hand
    # use ik to move the arm
    # grasp the object

    # input with random action
    env.reset_arm()
    
    # # time delay part
    # num = 0
    # while num < 30:
    #     # episode length 300
    #     # obs space (env, 208)
    #     # action space (env, 26)
    #     action = (torch.rand((1, 26), device=rl_device) * 2 - 1) * 0
    #     obs, reward, done, info = env.step(action)
    #     import time
    #     time.sleep(0.5)
    #     print(num)
    #     num += 1
    
    # # test finger actuator
    # num = 1
    # i = 0
    # while True:
    #     action = (torch.rand((1, 26), device=rl_device) * 2 - 1) * 0
    #     # import numpy as np
    #     # if num % 200 == 0:
    #     #     i += 1
    #     # action[0, 0+i] = np.sin(num / 10)
    #     action[0, 0] = 0.1
    #     if num > 150:
    #         action[0, 0] = 0
    #     obs, reward, done, info = env.step(action)
    #     print(num)
    #     num += 1    


    # #####################################################
    # env.object_root_positions
    # env.object_root_orientations
    # env.shadow_hand_center_positions
    # env.shadow_hand_center_orientations
    
    # env._r_target_palm_positions_wrt_object[env_ids]
    # env._r_target_palm_orientations_wrt_object[env_ids]

    # env.palm_positions_wrt_object
    # env.palm_orientations_wrt_object

    # env.endeffector_positions
    # env.endeffector_orientations

    # env.shadow_hand_dof_positions[env_ids, env.shadow_digits_actuated_dof_indices]
    # env._r_target_shadow_digits_actuated_dof_positions,

    # env._palm2forearm_quat
    # env._palm2forearm_pos
    
    while True:
        env.reset_arm()

        delta = 0.1
        max_ik_steps = 2
        env_ids = torch.tensor([0], device=rl_device)

        # dim=6
        target_palm_positions_wrt_object = env._r_target_palm_positions_wrt_object[env_ids]
        target_palm_orientations_wrt_object = env._r_target_palm_orientations_wrt_object[env_ids]
        current_palms_positions_wrt_object = env.palm_positions_wrt_object[env_ids]
        current_palms_orientations_wrt_object = env.palm_orientations_wrt_object[env_ids]
        initial_distance = torch.norm(target_palm_positions_wrt_object - current_palms_positions_wrt_object)
        
        for i in range(70):
            print(f"moving step: {i}")
            # print("target pos: {}, current pos: {}".format(target_palm_positions_wrt_object, env.palm_positions_wrt_object[env_ids]))
            # print("target ori: {}, current ori: {}".format(target_palm_orientations_wrt_object, env.palm_orientations_wrt_object[env_ids]))
            current_palms_positions_wrt_object = env.palm_positions_wrt_object[env_ids]
            current_palms_orientations_wrt_object = env.palm_orientations_wrt_object[env_ids]
            diff_pos = (target_palm_positions_wrt_object - current_palms_positions_wrt_object)# * 0
            diff_ori = target_palm_orientations_wrt_object - current_palms_orientations_wrt_object
            
            current_pos = env.endeffector_positions
            current_ori = env.endeffector_orientations
            target_pos = current_pos + delta * diff_pos
            target_ori = current_ori + delta * diff_ori

            current_distance = torch.norm(target_palm_positions_wrt_object - current_palms_positions_wrt_object)

            # if i == 34:
            #     target_pos[0, 2] -= 0.04
            # target_pos[0, 0] += 0.001
            target_pos[0, 1] += 0.0015


            for j in range(int(1 + max_ik_steps * current_distance / initial_distance)):
                delta_joint_move = ik(
                    env.j_eef,
                    # current_pos,
                    current_pos,
                    current_ori,
                    # target_pos,
                    target_pos,
                    target_ori,
                )
                delta_joint_move = delta_joint_move * env.dof_speed_scale * env.dt
                targets = env.shadow_hand_dof_positions.clone()
                ii, jj = torch.meshgrid(env_ids, env.ur_actuated_dof_indices, indexing="ij")
                env.curr_targets[ii, jj] = targets[ii, jj] + delta_joint_move
                indices = torch.unique(
                    torch.cat([env.shadow_hand_indices]).flatten().to(torch.int32)
                )
                env.gym.set_dof_position_target_tensor_indexed(
                    env.sim,
                    gymtorch.unwrap_tensor(env.curr_targets_buffer),
                    gymtorch.unwrap_tensor(indices),
                    indices.shape[0],
                )

                # step physics and render each frame
                for _ in range(env.control_freq_inv):
                    if env.force_render:
                        env.render()
                    env.gym.simulate(env.sim)

                env._refresh_sim_tensors()

        # dim=18
        target_shadow_hand_dof_positions = env._r_target_shadow_digits_actuated_dof_positions[env_ids]
        for i in range(100):
            print("grasping step: {}".format(i))
            current_shadow_hand_dof_positions = env.shadow_hand_dof_positions[env_ids, env.shadow_digits_actuated_dof_indices]
            delta_hand_joint_move = (target_shadow_hand_dof_positions - current_shadow_hand_dof_positions) * delta
            hand_targets = env.shadow_hand_dof_positions.clone()
            ii, jj = torch.meshgrid(env_ids, env.shadow_digits_actuated_dof_indices, indexing="ij")
            env.curr_targets[ii, jj] = hand_targets[ii, jj] + delta_hand_joint_move
            hand_indices = torch.unique(
                torch.cat([env.shadow_hand_indices]).flatten().to(torch.int32)
            )
            env.gym.set_dof_position_target_tensor_indexed(
                env.sim,
                gymtorch.unwrap_tensor(env.curr_targets_buffer),
                gymtorch.unwrap_tensor(hand_indices),
                hand_indices.shape[0],
            )
            if env.force_render and i % 1 == 0:
                env.render()
            env.gym.simulate(env.sim)
            env._refresh_sim_tensors()

        close_dis = 3
        env.close_dof_indices = torch.tensor([10, 11, 19, 20, 23, 24, 15, 16, 28, 29], device=rl_device)
        # env.close_dof_indices = torch.tensor([10, 19, 23, 15, 28], device=rl_device)
        for i in range(50):
            if i < 30:
                targets = env.shadow_hand_dof_positions.clone()
                # print(env.close_dof_indices)
                ii, jj = torch.meshgrid(env_ids, env.close_dof_indices, indexing="ij")
                env.curr_targets[ii, jj] = targets[ii, jj] + close_dis / 30
                indices = torch.unique(
                    torch.cat([env.shadow_hand_indices]).flatten().to(torch.int32)
                )
                env.gym.set_dof_position_target_tensor_indexed(
                    env.sim,
                    gymtorch.unwrap_tensor(env.curr_targets_buffer),
                    gymtorch.unwrap_tensor(indices),
                    indices.shape[0],
                )
            if env.force_render and i % 1 == 0:
                env.render()
            env.gym.simulate(env.sim)
            env._refresh_sim_tensors()



        current_pos = env.endeffector_positions.clone()
        target_pos = current_pos.clone()
        target_pos[:, 2] += 0.3
        for i in range(100):
            print("lifting step: {}".format(i))
            delta_joint_move = ik(
                env.j_eef,
                env.endeffector_positions,
                env.endeffector_orientations,
                target_pos,
                env.endeffector_orientations,
            )
            delta_joint_move = delta_joint_move * env.dof_speed_scale * env.dt * 0.05

            targets = env.shadow_hand_dof_positions.clone()
            ii, jj = torch.meshgrid(env_ids, env.ur_actuated_dof_indices, indexing="ij")
            env.curr_targets[ii, jj] = targets[ii, jj] + delta_joint_move
            indices = torch.unique(
                torch.cat([env.shadow_hand_indices]).flatten().to(torch.int32)
            )
            env.gym.set_dof_position_target_tensor_indexed(
                env.sim,
                gymtorch.unwrap_tensor(env.curr_targets_buffer),
                gymtorch.unwrap_tensor(indices),
                indices.shape[0],
            )
            # step physics and render each frame
            for i in range(env.control_freq_inv):
                if env.force_render:
                    env.render()
                env.gym.simulate(env.sim)
            env._refresh_sim_tensors()



    exit()

    # while True:
    #     action = torch.zeros((1, 26), device=rl_device)
    #     obs, reward, done, info = env.step(action)
    #     print(reward, done)
    #     # import time
    #     # time.sleep(0.5)
    
    # while True:
    #     current_pos = env.endeffector_positions
    #     # current_ori = env.endeffector_orientations
    #     current_shadow_hand_dof = env.shadow_hand_dof_positions[env_ids, env.shadow_digits_actuated_dof_indices]
    #     action = torch.zeros((1, 26), device=rl_device)
    #     action[0, :6] = final_joint_move
    #     action[0, 8:26] = (target_shadow_hand_dof_positions - current_shadow_hand_dof) * 0.1
    #     obs, reward, done, info = env.step(action)
    #     print(reward, done)
    #     import time
    #     time.sleep(0.5)
    
    exit()
    

    """
    load agent
    """
    learn_cfg = cfg_train["learn"]
    if "mode=eval" in args.overrides:
        learn_cfg["test"] = True
    is_testing = learn_cfg["test"]
    # Override resume and testing flags if they are passed as parameters.
    if args.model_dir != "":
        chkpt_path = args.model_dir

    runner = PPO(
        vec_env=env,
        cfg_train=cfg_train,
        device=rl_device,
        sampler=learn_cfg.get("sampler", "sequential"),
        log_dir=logdir,
        is_testing=is_testing,
        print_log=learn_cfg["print_log"],
        apply_reset=False,
        asymmetric=False,
        args=args,
    )

    if args.model_dir != "":
        if is_testing:
            runner.test(chkpt_path)
        else:
            runner.load(chkpt_path)

    iterations = cfg_train["learn"]["max_iterations"]
    if args.max_iterations > 0:
        iterations = args.max_iterations

    runner.run(num_learning_iterations=iterations, log_interval=cfg_train["learn"]["save_interval"])
