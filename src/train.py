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
    num = 0
    while num < 20:
        # episode length 300
        # obs space (env, 208)
        # action space (env, 26)
        import torch
        action = (torch.rand((1, 26), device=rl_device) * 2 - 1) * 0
        # action[:, :3] = 0.1
        obs, reward, done, info = env.step(action)
        import time
        time.sleep(0.5)
        # print(f"reward: {reward}")
        # print(f"done: {done}")
        print(num)
        # print(env.endeffector_positions)
        num += 1
    
    # print("object now: ")
    # print(env.object_root_positions, env.object_root_orientations)
    # print("hand now: ")
    # print(env.shadow_hand_center_positions, env.shadow_hand_center_orientations)
    # print("target:")
    # print(env._r_target_palm_positions_wrt_object)
    # exit()

    current_pos = env.endeffector_positions
    target_pos = current_pos.clone()
    target_pos[:, 2] += 0.3

    for i in range(100):
        delta_joint_move = ik(
            env.j_eef,
            env.endeffector_positions,
            env.endeffector_orientations,
            target_pos,
            env.endeffector_orientations,
        )
        delta_joint_move = delta_joint_move * env.dof_speed_scale * env.dt

        targets = env.shadow_hand_dof_positions.clone()
        ii, jj = torch.meshgrid(torch.tensor([0], device=rl_device), env.ur_actuated_dof_indices, indexing="ij")
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
        print(i, env.endeffector_positions)
        
    exit()
    
    
        #     targets = self.shadow_hand_dof_positions.clone()
        #     ii, jj = torch.meshgrid(env_ids, self.ur_actuated_dof_indices, indexing="ij")
        #     self.curr_targets[ii, jj] = targets[ii, jj] + delta_joint_move
        #     # apply_forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        #     # apply_forces[env_ids, self.shadow_center_index, 2] = 10
        #     # self.gym.apply_rigid_body_force_tensors(
        #     #     self.sim, gymtorch.unwrap_tensor(apply_forces), None, gymapi.ENV_SPACE
        #     # )

        #     # ii, jj = torch.meshgrid(env_ids, close_dof_indices)
        #     # self.curr_targets[ii, jj] += 0.02

        #     indices = torch.unique(
        #         torch.cat([self.shadow_hand_indices, self.target_shadow_hand_indices]).flatten().to(torch.int32)
        #     )
        #     self.gym.set_dof_position_target_tensor_indexed(
        #         self.sim,
        #         gymtorch.unwrap_tensor(self.curr_targets_buffer),
        #         gymtorch.unwrap_tensor(indices),
        #         indices.shape[0],
        #     )
        #     # step physics and render each frame
        #     for i in range(self.control_freq_inv):
        #         if self.force_render:
        #             self.render()
        #         self.gym.simulate(self.sim)

        #     self._refresh_sim_tensors()

        #     print(
        #         F.pairwise_distance(
        #             self.shadow_hand_dof_positions[0, 6:],
        #             self.curr_targets_buffer[0, self.shadow_hand_dof_start : self.shadow_hand_dof_end][6:],
        #         )
        #     )
        # print("lifted")









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
