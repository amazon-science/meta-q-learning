import numpy as np

# Code based on: 
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
# and https://github.com/sfujim/TD3/blob/master/utils.py

class Buffer(object):
	def __init__(self, max_size=1e6):
		'''

		'''
		self.storage = []
		self.max_size = max_size
		self.ptr = 0

	def reset(self):
		self.storage = []
		self.ptr = 0

	def add(self, data):
		'''
			data ==> (state, next_state, action, reward, done, previous_action, previous_reward)
		'''
		if len(self.storage) == self.max_size:
			self.storage[int(self.ptr)] = data
			self.ptr = (self.ptr + 1) % self.max_size
		else:
			self.storage.append(data)

	def size_rb(self):
		if len(self.storage) == self.max_size:
			return self.max_size

		else:
			return len(self.storage)

	def sample(self, batch_size):
		'''
			Returns tuples of (state, next_state, action, reward, done,
							  previous_action, previous_reward, previous_state
							  next_actions, next_rewards, next_states
							  )
		'''
		ind = np.random.randint(0, len(self.storage), size=batch_size)
		x, y, u, r, d, pu, pr, px, nu, nr, nx = [], [], [], [], [], [], [], [], [], [], []

		for i in ind: 
			# state, next_state, action, reward, done, previous_action, previous_reward, previous_state,
			# next_actions, next_rewards, next_states
			# X ==> state, 
			# Y ==> next_state
			# U ==> action
			# r ==> reward
			# d ==> done
			# pu ==> previous action
			# pr ==> previous reward
			# px ==> previous state
			# nu ==> next actions
			# nr ==> next rewards
			# nx ==> next states

			X, Y, U, R, D, PU, PR, PX, NU, NR, NX = self.storage[i]
			x.append(np.array(X, copy=False))
			y.append(np.array(Y, copy=False))
			u.append(np.array(U, copy=False))
			r.append(np.array(R, copy=False))
			d.append(np.array(D, copy=False))
			pu.append(np.array(PU, copy=False))
			pr.append(np.array(PR, copy=False))
			px.append(np.array(PX, copy=False))
			nu.append(np.array(NU, copy=False))
			nr.append(np.array(NR, copy=False))
			nx.append(np.array(NX, copy=False))

		return np.array(x), np.array(y), np.array(u), \
			   np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1), \
			   np.array(pu), np.array(pr), np.array(px),\
			   np.array(nu), np.array(nr), np.array(nx)