python src/train.py mode="eval" env_mode=pgm env_info=False num_envs=1 num_objects=1 num_objects_per_env=1 graphics_device_id=0 split='train' cluster=0 task.env.datasetMetainfoPath="data/oakink_filtered_metainfo.csv" task.env.datasetPoseLevelSampling=True --seed=0 --exp_name='PPO' --run_device_id=0 --web_visualizer_port=-1 --model_dir="/juno/u/suning/DexFunPreGrasp/ckpt/pose-level-specialist-0.pt" --eval_times=10



python src/train.py mode="eval" env_mode=pgm env_info=False num_envs=1 num_objects=1 num_objects_per_env=1 graphics_device_id=0 split='train' cluster=0 task.env.datasetMetainfoPath="data/oakink_filtered_metainfo.csv" task.env.datasetPoseLevelSampling=True --seed=0 --exp_name='PPO' --run_device_id=0 --web_visualizer_port=-1 --model_dir="/juno/u/suning/DexFunPreGrasp/ckpt/pose-level-specialist-0.pt" --eval_times=10

python src/train.py mode="eval" env_mode=pgm env_info=False num_envs=1 num_objects=1 num_objects_per_env=1 graphics_device_id=0 split='train' cluster=0 task.env.datasetMetainfoPath="data/oakink_filtered_metainfo.csv" task.env.datasetPoseLevelSampling=True --seed=0 --exp_name='PPO' --run_device_id=0 --web_visualizer_port=-1 --model_dir="/juno/u/suning/DexFunPreGrasp/logs/PPO/model_pose_level_full_observation_cluster_0_objtype:mug_s190_labeltype:data/oakink_shadow_dataset_valid_force_noise_accept_1/mug_s190/010510_use_s10190.json_objnum:4096_objcat:all_maxpercat:-1_geo:all_scale:all_envnum:4096_rewtype:succrew+rotrew+actionpen+mutual+fjcontact_seed0/model_24500.pt" --eval_times=10


python src/train.py mode="eval" env_mode=pgm env_info=False num_envs=16 num_objects=16 num_objects_per_env=1 graphics_device_id=0 split='train' cluster=0 task.env.datasetMetainfoPath="data/oakink_filtered_metainfo.csv" task.env.datasetPoseLevelSampling=True --seed=0 --exp_name='PPO' --run_device_id=0 --web_visualizer_port=-1 --model_dir="/juno/u/suning/DexFunPreGrasp/ckpt/pose-level-specialist-0.pt" --eval_times=10


python src/train.py mode="eval" env_mode=pgm headless=True env_info=False num_envs=4096 num_objects=4096 num_objects_per_env=1 graphics_device_id=0 split='train' cluster=0 task.env.datasetMetainfoPath="data/oakink_filtered_metainfo.csv" task.env.datasetPoseLevelSampling=True --seed=0 --exp_name='PPO' --run_device_id=0 --web_visualizer_port=-1 --model_dir="/juno/u/suning/DexFunPreGrasp/ckpt/pose-level-specialist-0.pt" --eval_times=10