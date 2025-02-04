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




    def slerp(q1, q2, t, DOT_THRESHOLD=0.9995):
        """
        对形状为 (B,4) 的批量四元数 q1 和 q2 进行球面线性插值（SLERP）。

        参数：
            q1 (torch.Tensor): 起始四元数，形状 (B, 4)。
            q2 (torch.Tensor): 目标四元数，形状 (B, 4)。
            t (float 或 torch.Tensor): 插值因子，范围 [0, 1]。若为标量，则对所有批次都使用同一插值因子；
                                        若为张量，其形状应为 (B,) 或 (B,1)。
            DOT_THRESHOLD (float): 当两个四元数内积大于此阈值时，采用线性插值（LERP）近似，防止数值不稳定。

        返回：
            torch.Tensor: 插值后的四元数，形状 (B, 4)，归一化为单位四元数。
        """
        # 归一化输入的四元数，保证是单位四元数
        q1 = q1 / q1.norm(dim=1, keepdim=True)
        q2 = q2 / q2.norm(dim=1, keepdim=True)

        # 计算内积，形状 (B, 1)
        dot = torch.sum(q1 * q2, dim=1, keepdim=True)

        # 如果内积为负，则翻转 q2，保证两个四元数位于同一半空间，得到最短插值路径
        q2 = torch.where(dot < 0, -q2, q2)
        # 重新计算内积（取绝对值）
        dot = torch.abs(torch.sum(q1 * q2, dim=1, keepdim=True))

        # 确保 t 为张量，并扩展成形状 (B, 1)
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=q1.dtype, device=q1.device)
        if t.dim() == 0:
            t = t.expand(q1.shape[0], 1)
        elif t.dim() == 1:
            t = t.unsqueeze(1)

        # 对于内积大于阈值的部分，使用线性插值（LERP）近似
        use_lerp = dot > DOT_THRESHOLD

        # SLERP 部分：计算两个四元数之间的夹角
        theta_0 = torch.acos(dot)          # shape: (B, 1)
        sin_theta_0 = torch.sin(theta_0)     # shape: (B, 1)

        # 插值角度 theta
        theta = theta_0 * t                # shape: (B, 1)
        sin_theta = torch.sin(theta)       # shape: (B, 1)

        # 计算插值系数
        s1 = torch.sin(theta_0 - theta) / sin_theta_0  # shape: (B, 1)
        s2 = sin_theta / sin_theta_0                   # shape: (B, 1)

        # SLERP 插值结果
        slerp_result = s1 * q1 + s2 * q2  # shape: (B, 4)

        # LERP 插值结果
        lerp_result = q1 + t * (q2 - q1)
        lerp_result = lerp_result / lerp_result.norm(dim=1, keepdim=True)

        # 使用掩码决定对于哪些批次使用 LERP 插值
        use_lerp_expanded = use_lerp.expand_as(slerp_result)
        result = torch.where(use_lerp_expanded, lerp_result, slerp_result)

        # 归一化最终结果
        result = result / result.norm(dim=1, keepdim=True)
        return result
    
    def quaternion_mul(q, r):
        """
        对形状为 (bsz, 4) 的四元数 q 和 r 进行乘法运算，
        假设四元数格式为 [x, y, z, w]。
        """
        # 提取各分量
        x1, y1, z1, w1 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        x2, y2, z2, w2 = r[:, 0], r[:, 1], r[:, 2], r[:, 3]
        
        # 根据四元数乘法公式计算各分量
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        
        return torch.stack([x, y, z, w], dim=1)

    def quaternion_conjugate(q):
        """
        计算一批四元数的共轭。
        
        参数:
            q: numpy 数组，形状 (B, 4)，格式为 [x, y, z, w]
        
        返回:
            共轭后的四元数，形状 (B, 4)，即 [-x, -y, -z, w]
        """
        x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        return torch.stack([-x, -y, -z, w], dim=1)

    def rotate_vector_by_quaternion(v, q):
        """
        利用批量单位四元数 q 对批量三维向量 v 进行旋转。
        
        参数:
            v: numpy 数组，形状为 (B, 3)，表示 B 个三维向量。
            q: numpy 数组，形状为 (B, 4)，表示 B 个单位四元数，格式为 [x, y, z, w].
        
        返回:
            rotated_v: numpy 数组，形状为 (B, 3)，表示旋转后的向量。
        """
        B = v.shape[0]
        # 将每个向量转换为纯虚四元数形式： (v_x, v_y, v_z, 0)
        v_quat = torch.cat([v, torch.zeros((B, 1), device=rl_device)], dim=1)
        
        # 计算 q 的共轭（单位四元数的逆）
        q_inv = quaternion_conjugate(q)
        
        # 计算 q * v_quat
        temp = quaternion_mul(q, v_quat)
        # 再计算 (q * v_quat) * q_inv
        rotated_v_quat = quaternion_mul(temp, q_inv)
        
        # 返回结果中的向量部分（前 3 个分量）
        return rotated_v_quat[:, :3]

    def relative_quaternion(q1, q2):
        """
        计算批量相对旋转四元数 q_rel，使得对于每个样本有： q2 = q_rel ⊗ q1.
        
        参数:
            q1: numpy 数组，形状 (B, 4)，表示第一组四元数 [x, y, z, w].
            q2: numpy 数组，形状 (B, 4)，表示第二组四元数 [x, y, z, w].
            
        返回:
            q_rel: numpy 数组，形状 (B, 4)，表示相对旋转四元数 [x, y, z, w]。
                注意 q_rel 与 -q_rel 表示相同的旋转。
        """
        # 计算 q1 的共轭，即 q1 的逆（单位四元数）
        q1_conj = quaternion_conjugate(q1)
        # 计算相对四元数： q_rel = q2 ⊗ q1_conj
        q_rel = quaternion_mul(q2, q1_conj)
        # 对每个样本归一化
        norm = torch.norm(q_rel, dim=1, keepdim=True)
        # 防止除 0
        q_rel = q_rel / (norm + 1e-8)
        return q_rel

    def success():
        success_tolerance = 0.11 #0.1
        trans_scale = 9 #10
        env.rot_dist = quat_diff_rad(env.object_orientations_wrt_palm, env._r_target_object_orientations_wrt_palm)
        env.pos_dist = F.pairwise_distance(env.object_positions_wrt_palm, env._r_target_object_positions_wrt_palm)
        env.fj_dist = F.pairwise_distance(
            env.shadow_hand_dof_positions[:, env.shadow_digits_actuated_dof_indices],
            env._r_target_shadow_digits_actuated_dof_positions,
        )
        
        env.succ_rew = torch.where(
            (torch.abs(env.rot_dist) <= success_tolerance) # 0.1
            & (torch.abs(env.pos_dist) <= (success_tolerance / trans_scale)) # 0.01
            & (torch.abs(env.fj_dist) <= 1.0)) #env.contact_eps)) # 0.2
        cost = torch.abs(env.rot_dist) + torch.abs(env.pos_dist) + torch.abs(env.fj_dist)
        success_idx = torch.ones(env.num_envs, device=rl_device)
        success_idx[torch.abs(env.rot_dist) > success_tolerance] = 0
        success_idx[torch.abs(env.pos_dist) > (success_tolerance / trans_scale)] = 0
        success_idx[torch.abs(env.fj_dist) > 1.0] = 0
        return env.succ_rew, cost, success_idx

    def record(record_data, obj_pos, obj_ori, succ_idx, cost, save=False):
        if record_data["obj_pos"] is None:
            record_data["obj_pos"] = obj_pos
            record_data["obj_ori"] = obj_ori
            record_data["succ_idx"] = succ_idx
            record_data["cost"] = cost
        else:
            record_data["obj_pos"] = torch.cat([record_data["obj_pos"], obj_pos], dim=0)
            record_data["obj_ori"] = torch.cat([record_data["obj_ori"], obj_ori], dim=0)
            record_data["succ_idx"] = torch.cat([record_data["succ_idx"], succ_idx], dim=0)
            record_data["cost"] = torch.cat([record_data["cost"], cost], dim=0)
        print(obj_pos[:10])
        print(obj_ori[:10])
        print(succ_idx[:10])
        # print(record_data["cost"])
        print("record data size: ", record_data["obj_pos"].size())
        # if save:
        #     torch.save(record_data, "record_data.pth")
        #     print("record data saved")
        
    def data_collection(env=env):
        import numpy as np
        import torch.nn.functional as F
        from tasks.torch_utils import quat_diff_rad
        record_data = {}
        record_data["obj_pos"] = None
        record_data["obj_ori"] = None
        record_data["succ_idx"] = None
        record_data["cost"] = None

        object_original_orientation = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=rl_device).repeat(env.num_envs, 1)

        import torch.nn as nn
        class SimpleNet(nn.Module):
            def __init__(self):
                super(SimpleNet, self).__init__()
                self.model = nn.Sequential(
                    nn.Linear(7, 1024),  # 输入维度为 7 (3+4)
                    nn.ReLU(),
                    nn.Linear(1024, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 512),
                    nn.ReLU(),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, 1)    # 输出 1 个数，后续使用 Sigmoid 进行二分类判断
                )

            def forward(self, x):
                return self.model(x)

        model = SimpleNet()
        model.load_state_dict(torch.load("/juno/u/suning/DexFunPreGrasp/mug_scorer.pth"))
        model.to(rl_device)
        model.eval()

        num = 0
        while True:
            # env.reset_arm()
            env.reset()
            env_ids = torch.tensor([range(env.num_envs)], device=rl_device).squeeze(0)
            palm_positions_wrt_eef = env.shadow_hand_center_positions - env.endeffector_positions
            
            # priviledge information
            object_positions = env.object_root_positions
            object_orientations = env.object_root_orientations
            
            output = model(torch.cat([object_positions, object_orientations], dim=1))
            print(output)
            output = torch.sigmoid(output)
            output = output.squeeze(1)
            output = output > 0.5
            print("predict results: ", output)

            target_palm_positions_wrt_object = env._r_target_palm_positions_wrt_object[:]
            target_palm_orientations_wrt_object = env._r_target_palm_orientations_wrt_object[:]
                    
            object_rel_orientation = relative_quaternion(object_original_orientation, object_orientations)
            target_palm_positions_wrt_object = rotate_vector_by_quaternion(target_palm_positions_wrt_object, object_rel_orientation)

            current_palms_positions_wrt_object = env.palm_positions_wrt_object[:]
            current_palms_orientations_wrt_object = env.palm_orientations_wrt_object[:]
            
            # target_hand_orientation = torch.tensor([[ 0.4857, -0.5795, -0.0983,  0.6470]], device='cuda:0')
            target_hand_orientation = quaternion_mul(object_orientations, target_palm_orientations_wrt_object)        
            relative_hand_end_wrt_start = relative_quaternion(env.shadow_hand_center_orientations, target_hand_orientation)
            v = rotate_vector_by_quaternion(palm_positions_wrt_eef, relative_hand_end_wrt_start)
            target_hand_positions = object_positions + target_palm_positions_wrt_object - v
            
            # robot initial pos [0.0200, 0.3000, 0.6000]
            for i in range(150):
                # print(f"moving step: {i}")
                current_pos = env.endeffector_positions
                # print(target_hand_positions, current_pos)
                current_ori = env.endeffector_orientations
                # print(target_hand_orientation, current_ori)
                target_pos = current_pos + (target_hand_positions - current_pos) * 0.1
                target_ori = slerp(current_ori, target_hand_orientation, 0.1)
                
                for j in range(1):
                    delta_joint_move = ik(
                        env.j_eef,
                        # current_pose,
                        current_pos,
                        current_ori,
                        # target_pose,
                        target_pos,
                        target_ori,
                    )
                    delta_joint_move = delta_joint_move * env.dof_speed_scale * env.dt
                    norm = torch.norm(delta_joint_move, dim=1, keepdim=True)
                    max_norm = 0.1
                    delta_joint_move = torch.where(norm > max_norm, delta_joint_move / norm * max_norm, delta_joint_move)
                    norm = torch.norm(delta_joint_move, dim=1, keepdim=True)
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
                    
            target_shadow_hand_dof_positions = env._r_target_shadow_digits_actuated_dof_positions[:]
            for i in range(100):
                # print("grasping step: {}".format(i))
                current_shadow_hand_dof_positions = env.shadow_hand_dof_positions[:, env.shadow_digits_actuated_dof_indices]
                delta_hand_joint_move = (target_shadow_hand_dof_positions - current_shadow_hand_dof_positions) * 0.1
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
            
            # succ_rew, cost, success_idx = success()
            # print("success_idx: ", success_idx)
            save = num % 2 == 0
            # record(record_data, object_positions, object_orientations, success_idx, cost, save)
            num += 1
       
    # data_collection(env=env)
        
    
    import wandb
    wandb.init(
        project="DexPreGrasp",     # 修改为你的 wandb 项目名
        name="PPO_Evaluator",        # 实验运行名称，可选
        config=cfg_train,                     # 记录超参数、配置等
        reinit=True
    )
    
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
        wandb=wandb,
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
