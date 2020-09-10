import numpy as np
from algs.MQL.buffer import Buffer
import random

class MultiTasksSnapshot(object):
	def __init__(self, max_size=1e3):
		'''	
			all task will have same size
		'''
		self.max_size = max_size
		
	def init(self, task_ids=None):
		'''
			init buffers for all tasks
		'''
		self.task_buffers = dict([(idx, Buffer(max_size = self.max_size))
									for idx in task_ids
								])

	def reset(self, task_id):

		self.task_buffers[task_id].reset()

	def list(self):

		return list(self.task_buffers.keys())

	def add(self, task_id, data):
		'''
			data ==> (state, next_state, action, reward, done, previous_action, previous_reward)
		'''
		self.task_buffers[task_id].add(data)

	def size_rb(self, task_id):

		return self.task_buffers[task_id].size_rb()

	def get_buffer(self, task_id):

		return self.task_buffers[task_id]

	def sample(self, task_ids, batch_size):
		'''
			Returns tuples of (state, next_state, action, reward, done,
							  previous_action, previous_reward, previous_state
							  )
		'''
		if len(task_ids) == 1:
			xx, _, _, _, _, pu, pr, px, _, _, _ =  self.task_buffers[task_ids[0]].sample(batch_size)

			return pu, pr, px, xx

		mb_actions = []
		mb_rewards = []
		mb_obs = []
		mb_x = []

		for tid in task_ids:

			xx, _, _, _, _, pu, pr, px, _, _, _ = self.task_buffers[tid].sample(batch_size)
			mb_actions.append(pu) # batch_size x D1
			mb_rewards.append(pr) # batch_size x D2
			mb_obs.append(px)     # batch_size x D3
			mb_x.append(xx)

		mb_actions = np.asarray(mb_actions, dtype=np.float32) # task_ids x batch_size x D1
		mb_rewards = np.asarray(mb_rewards, dtype=np.float32) # task_ids x batch_size x D2
		mb_obs     = np.asarray(mb_obs, dtype=np.float32)     # task_ids x batch_size x D2
		mb_x       = np.asarray(mb_x, dtype=np.float32)

		return mb_actions, mb_rewards, mb_obs, mb_x

	def sample_tasks(self, task_ids, batch_size):
		'''
			Returns tuples of (state, next_state, action, reward, done,
							  previous_action, previous_reward, previous_state
							  )
		'''
		mb_xx = []
		mb_yy = []
		mb_u = []
		mb_r = []
		mb_d = []
		mb_pu = []
		mb_pr = []
		mb_px = []
		mb_nu = []
		mb_nr = []
		mb_nx = []

		# shuffle task lists
		shuffled_task_ids = random.sample(task_ids, len(task_ids))

		for tid in shuffled_task_ids:

			xx, yy, u, r, d, pu, pr, px, nu, nr, nx = self.task_buffers[tid].sample(batch_size)
			mb_xx.append(xx) # batch_size x D1
			mb_yy.append(yy) # batch_size x D2
			mb_u.append(u)   # batch_size x D3
			mb_r.append(r)
			mb_d.append(d)
			mb_pu.append(pu)
			mb_pr.append(pr)
			mb_px.append(px)
			mb_nu.append(nu)
			mb_nr.append(nr)
			mb_nx.append(nx)

		mb_xx = np.asarray(mb_xx, dtype=np.float32).reshape(len(task_ids) * batch_size , -1) # task_ids x batch_size x D1
		mb_yy = np.asarray(mb_yy, dtype=np.float32).reshape(len(task_ids) * batch_size , -1) # task_ids x batch_size x D1
		mb_u = np.asarray(mb_u, dtype=np.float32).reshape(len(task_ids) * batch_size , -1) # task_ids x batch_size x D1
		mb_r = np.asarray(mb_r, dtype=np.float32).reshape(len(task_ids) * batch_size , -1) # task_ids x batch_size x D1
		mb_d = np.asarray(mb_d, dtype=np.float32).reshape(len(task_ids) * batch_size , -1) # task_ids x batch_size x D1
		mb_pu = np.asarray(mb_pu, dtype=np.float32).reshape(len(task_ids) * batch_size , -1) # task_ids x batch_size x D1
		mb_pr = np.asarray(mb_pr, dtype=np.float32).reshape(len(task_ids) * batch_size , -1) # task_ids x batch_size x D1
		mb_px = np.asarray(mb_px, dtype=np.float32).reshape(len(task_ids) * batch_size , -1) # task_ids x batch_size x D1
		mb_nu = np.asarray(mb_nu, dtype=np.float32).reshape(len(task_ids) * batch_size , -1) # task_ids x batch_size x D1
		mb_nr = np.asarray(mb_nr, dtype=np.float32).reshape(len(task_ids) * batch_size , -1) # task_ids x batch_size x D1
		mb_nx = np.asarray(mb_nx, dtype=np.float32).reshape(len(task_ids) * batch_size , -1) # task_ids x batch_size x D1

		return mb_xx, mb_yy, mb_u, mb_r, mb_d, mb_pu, mb_pr, mb_px, mb_nu, mb_nr, mb_nx