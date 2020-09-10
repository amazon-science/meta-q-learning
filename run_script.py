import argparse
import os

def ant_dir(seed, log_id, gpu_id):

	cmd = " python main.py --env_name ant-dir \
	--alg_name mql --policy_freq 4 \
	--expl_noise 0.2 --enable_context  --num_train_steps 1000 \
	--cuda_deterministic  --history_length  25  --beta_clip 1.2 \
	--enable_adaptation  --num_initial_steps 1500 --main_snap_iter_nums 400 \
	--snap_iter_nums 5 --hidden_sizes  300 300  --lam_csc  0.05 \
	--snapshot_size 2000 --lr  0.0003 --sample_mult 5  --use_epi_len_steps \
	--unbounded_eval_hist  --hiddens_conext 30  --num_tasks_sample 5 --burn_in  10000 \
	--batch_size 256 --policy_noise 0.4 --eval_freq 10000 --replay_size 1000000 " + \
	' --log_id ' + log_id + ' --seed ' + str(seed) + ' --gpu_id ' + str(gpu_id)
	return cmd

def humanoid_dir(seed, log_id, gpu_id):

	cmd = " python main.py --env_name humanoid-dir \
	--alg_name mql --policy_freq 3 --expl_noise 0.2 \
	--enable_context  --num_train_steps 500 --cuda_deterministic \
	--history_length  15  --unbounded_eval_hist  --beta_clip 2 \
	--enable_adaptation  --num_initial_steps 1000 --main_snap_iter_nums 15 \
	--snap_iter_nums 5 --hidden_sizes  300 300  --lam_csc  0.1 --snapshot_size 2000 \
	--lr  0.0003  --sample_mult 5  --use_epi_len_steps  --enable_beta_obs_cxt \
	--hiddens_conext 20   --num_tasks_sample 1 --burn_in  10000 --batch_size 256 \
	--policy_noise 0.3 --eval_freq 10000  --replay_size 500000  "+ \
	' --log_id ' + log_id + ' --seed ' + str(seed) + ' --gpu_id ' + str(gpu_id)
	return cmd

def cheetah_vel(seed, log_id, gpu_id):

	cmd = " python main.py  --env_name cheetah-vel \
	--alg_name mql --policy_freq 2 \
	--expl_noise 0.3 --enable_context  --num_train_steps 200 \
	--cuda_deterministic  --history_length  20  --beta_clip 1 \
	--enable_adaptation  --num_initial_steps 500 --main_snap_iter_nums 100 \
	--snap_iter_nums 10 --hidden_sizes  300 300  --lam_csc  0.1 \
	--snapshot_size 2000 --lr  0.001 --sample_mult 5  --use_epi_len_steps  \
	--hiddens_conext 15   --num_tasks_sample 1 --burn_in  10000 --batch_size 256 \
	--policy_noise 0.3 --eval_freq 10000  --replay_size 1000000 " + \
	' --log_id ' + log_id + ' --seed ' + str(seed) + ' --gpu_id ' + str(gpu_id)
	return cmd 

def cheetah_dir(seed, log_id, gpu_id):

	cmd = " python main.py  --env_name cheetah-dir \
	--alg_name mql \
	--policy_freq 2 --expl_noise 0.3 --enable_context  --num_train_steps 500 \
	--cuda_deterministic  --history_length  20  --beta_clip 1.5 --enable_adaptation  \
	--num_initial_steps 1000 --main_snap_iter_nums 50 --snap_iter_nums 10 --hidden_sizes  300 300 \
	--lam_csc  0.5  --snapshot_size 2000 --hiddens_conext 20  --lr  0.0003 \
	--sampling_style replay --sample_mult 10  --use_epi_len_steps  --num_tasks_sample 1 \
	--burn_in  10000 --batch_size 256 --policy_noise 0.3 --eval_freq 10000  --replay_size 1000000 " + \
	' --log_id ' + log_id + ' --seed ' + str(seed) + ' --gpu_id ' + str(gpu_id)
	return cmd

def ant_goal(seed, log_id, gpu_id):

	cmd = " python main.py --env_name ant-goal --alg_name mql \
	--policy_freq 4 --expl_noise 0.3 --enable_context  \
	--num_train_steps 200 --cuda_deterministic  --history_length  40  \
	--beta_clip 2 --enable_adaptation  --num_initial_steps 1500  \
	--main_snap_iter_nums 150 --snap_iter_nums 10 --hidden_sizes  300 300  \
	--lam_csc  0.1 --snapshot_size 1000 --lr  0.0004 --sample_mult 1  \
	--use_epi_len_steps --unbounded_eval_hist --enable_beta_obs_cxt  \
	--hiddens_conext 40   --num_tasks_sample 5 --burn_in  10000 --batch_size 256 \
	--policy_noise 0.3 --eval_freq 10000 --replay_size 1000000 "+ \
	' --log_id ' + log_id + ' --seed ' + str(seed) + ' --gpu_id ' + str(gpu_id)
	return cmd

def walker_rand_params(seed, log_id, gpu_id):

	cmd = " python main.py --env_name walker-rand-params \
	--policy_freq 3 --expl_noise 0.2 --enable_context  --num_train_steps 200 \
	--history_length  10 --beta_clip 1.1 --enable_adaptation  --num_initial_steps 2000 --main_snap_iter_nums 400 \
	--snap_iter_nums 10 --hidden_sizes  300 300  --lam_csc  0.06  --snapshot_size 4000 --hiddens_conext 30  \
	--lr  0.0008  --eval_freq 10000  \
	--lr_gamma 0.7  --enable_beta_obs_cxt  --sample_mult 5  --use_epi_len_steps  \
	--num_tasks_sample 5 --burn_in  10000 --batch_size 256 --policy_noise 0.2  --replay_size 1000000 "+ \
	' --log_id ' + log_id + ' --seed ' + str(seed) + ' --gpu_id ' + str(gpu_id)
	return cmd

if __name__ == "__main__":


	parser = argparse.ArgumentParser()
	parser.add_argument('--env_name', type=str, default='ant-goal')
	parser.add_argument('--seed', type=int, default=0)
	parser.add_argument('--gpu_id', type=int, default=0)
	parser.add_argument('--log_id', type=str, default='dummy')
	args = parser.parse_args()
	print('------------')
	print(args.__dict__)
	print('------------')

	cmd = ''
	if args.env_name == 'ant-dir':
		cmd = ant_dir(args.seed, args.log_id, args.gpu_id)

	elif args.env_name  == 'ant-goal':
		cmd = ant_goal(args.seed, args.log_id, args.gpu_id)

	elif args.env_name  == 'humanoid-dir':
		cmd = humanoid_dir(args.seed, args.log_id, args.gpu_id)

	elif args.env_name  == 'cheetah-vel':
		cmd = cheetah_vel(args.seed, args.log_id, args.gpu_id)

	elif args.env_name  == 'cheetah-dir':
		cmd = cheetah_dir(args.seed, args.log_id, args.gpu_id)

	elif args.env_name  == 'walker-rand-params':
		cmd = walker_rand_params(args.seed, args.log_id, args.gpu_id)

	# run the code
	print(cmd)
	os.system(cmd)










