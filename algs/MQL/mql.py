from __future__ import  print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from copy import deepcopy
from sklearn.linear_model import LogisticRegression as logistic

class MQL:

    def __init__(self, 
                actor,
                actor_target,
                critic,
                critic_target,
                lr=None,
                gamma=0.99,
                ptau = 0.005,
                policy_noise = 0.2,
                noise_clip = 0.5,
                policy_freq = 2,
                batch_size = 100,
                optim_method = '',
                max_action = None,
                max_iter_logistic = 2000,
                beta_clip = 1,
                enable_beta_obs_cxt = False,
                prox_coef = 1,
                device = 'cpu',
                lam_csc = 1.0,
                type_of_training = 'csc',
                use_ess_clipping = False,
                use_normalized_beta = True,
                reset_optims = False,
                ):

        '''
            actor:  actor network 
            critic: critic network 
            lr:   learning rate for RMSProp
            gamma: reward discounting parameter
            ptau:  Interpolation factor in polyak averaging  
            policy_noise: add noise to policy 
            noise_clip: clipped noise 
            policy_freq: delayed policy updates
            enable_beta_obs_cxt:  decide whether to concat obs and ctx for logistic regresstion
            lam_csc: logisitc regression reg, samller means stronger reg
        '''
        self.actor = actor
        self.actor_target = actor_target
        self.critic = critic
        self.critic_target = critic_target
        self.gamma = gamma
        self.ptau = ptau
        self.policy_noise = policy_noise
        self.policy_freq  = policy_freq
        self.noise_clip = noise_clip
        self.max_action = max_action
        self.batch_size = batch_size
        self.max_iter_logistic = max_iter_logistic
        self.beta_clip = beta_clip
        self.enable_beta_obs_cxt = enable_beta_obs_cxt
        self.prox_coef = prox_coef
        self.prox_coef_init = prox_coef
        self.device = device
        self.lam_csc = lam_csc
        self.type_of_training = type_of_training
        self.use_ess_clipping = use_ess_clipping
        self.r_eps = np.float32(1e-7)  # this is used to avoid inf or nan in calculations
        self.use_normalized_beta = use_normalized_beta
        self.set_training_style()
        self.lr = lr
        self.reset_optims = reset_optims


        # load tragtes models.
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # keep a copy of model params which will be used for proximal point
        self.copy_model_params()

        if lr:
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr = lr)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr = lr)

        else:
            self.actor_optimizer = optim.Adam(self.actor.parameters())
            self.critic_optimizer = optim.Adam(self.critic.parameters())

        print('-----------------------------')
        print('Optim Params')
        print("Actor:\n ",  self.actor_optimizer)
        print("Critic:\n ", self.critic_optimizer )
        print('********')
        print("reset_optims: ", reset_optims)
        print("use_ess_clipping: ", use_ess_clipping)
        print("use_normalized_beta: ", use_normalized_beta)
        print("enable_beta_obs_cxt: ", enable_beta_obs_cxt)
        print('********')
        print('-----------------------------')

    def copy_model_params(self):
        '''
            Keep a copy of actor and critic for proximal update
        '''
        self.ckpt = {
                        'actor': deepcopy(self.actor),
                        'critic': deepcopy(self.critic)
                    }

    def set_tasks_list(self, tasks_idx):
        '''
            Keep copy of task lists
        '''
        self.train_tasks_list = set(tasks_idx.copy())

    def select_action(self, obs, previous_action, previous_reward, previous_obs):

        '''
            return action
        '''
        obs = torch.FloatTensor(obs.reshape(1, -1)).to(self.device)
        previous_action = torch.FloatTensor(previous_action.reshape(1, -1)).to(self.device)
        previous_reward = torch.FloatTensor(previous_reward.reshape(1, -1)).to(self.device)
        previous_obs = torch.FloatTensor(previous_obs.reshape(1, -1)).to(self.device)

        # combine all other data here before send them to actor
        # torch.cat([previous_action, previous_reward], dim = -1)
        pre_act_rew = [previous_action, previous_reward, previous_obs]

        return self.actor(obs, pre_act_rew).cpu().data.numpy().flatten()

    def get_prox_penalty(self, model_t, model_target):
        '''
            This function calculates ||theta - theta_t||
        '''
        param_prox = []
        for p, q in zip(model_t.parameters(), model_target.parameters()):
            # q should ne detached
            param_prox.append((p - q.detach()).norm()**2)

        result = sum(param_prox)

        return result

    def train_cs(self, task_id = None, snap_buffer = None, train_tasks_buffer = None, adaptation_step = False):
        '''
            This function trains covariate shift correction model
        '''

        ######
        # fetch all_data
        ######
        if adaptation_step == True:
            # step 1: calculate how many samples per classes we need
            # in adaption step, all train task can be used
            task_bsize = int(snap_buffer.size_rb(task_id) / (len(self.train_tasks_list))) + 2
            neg_tasks_ids = self.train_tasks_list

        else:
            # step 1: calculate how many samples per classes we need
            task_bsize = int(snap_buffer.size_rb(task_id) / (len(self.train_tasks_list) - 1)) + 2
            neg_tasks_ids = list(self.train_tasks_list.difference(set([task_id])))

        # collect examples from other tasks and consider them as one class
        # view --> len(neg_tasks_ids),task_bsize, D ==> len(neg_tasks_ids) * task_bsize, D
        pu, pr, px, xx = train_tasks_buffer.sample(task_ids = neg_tasks_ids, batch_size = task_bsize)
        neg_actions = torch.FloatTensor(pu).view(task_bsize * len(neg_tasks_ids), -1).to(self.device)
        neg_rewards = torch.FloatTensor(pr).view(task_bsize * len(neg_tasks_ids), -1).to(self.device)
        neg_obs = torch.FloatTensor(px).view(task_bsize * len(neg_tasks_ids), -1).to(self.device)
        neg_xx = torch.FloatTensor(xx).view(task_bsize * len(neg_tasks_ids), -1).to(self.device)

        # sample cuurent task and consider it as another class
        # returns size: (task_bsize, D)
        ppu, ppr, ppx, pxx = snap_buffer.sample(task_ids = [task_id], batch_size = snap_buffer.size_rb(task_id))
        pos_actions = torch.FloatTensor(ppu).to(self.device)
        pos_rewards = torch.FloatTensor(ppr).to(self.device)
        pos_obs = torch.FloatTensor(ppx).to(self.device)
        pos_pxx = torch.FloatTensor(pxx).to(self.device)

        # combine reward and action and previous states for context network.
        pos_act_rew_obs  = [pos_actions, pos_rewards, pos_obs]
        neg_act_rew_obs  = [neg_actions, neg_rewards, neg_obs]

        ######
        # extract features: context features 
        ######
        with torch.no_grad():   

            # batch_size X context_hidden 
            # self.actor.get_conext_feats outputs, [batch_size , context_size]
            # torch.cat ([batch_size , obs_dim], [batch_size , context_size]) ==> [batch_size, obs_dim + context_size ]
            if self.enable_beta_obs_cxt == True:
                snap_ctxt = torch.cat([pos_pxx, self.actor.get_conext_feats(pos_act_rew_obs)], dim = -1).cpu().data.numpy()
                neg_ctxt = torch.cat([neg_xx, self.actor.get_conext_feats(neg_act_rew_obs)], dim = -1).cpu().data.numpy()

            else:
                snap_ctxt = self.actor.get_conext_feats(pos_act_rew_obs).cpu().data.numpy()
                neg_ctxt = self.actor.get_conext_feats(neg_act_rew_obs).cpu().data.numpy()


        ######
        # Train logistic classifiers 
        ######
        x = np.concatenate((snap_ctxt, neg_ctxt)) # [b1 + b2] X D
        y = np.concatenate((-np.ones(snap_ctxt.shape[0]), np.ones(neg_ctxt.shape[0])))

        # model params : [1 , D] wehere D is context_hidden
        model = logistic(solver='lbfgs', max_iter = self.max_iter_logistic, C = self.lam_csc).fit(x,y)
        # keep track of how good is the classifier
        predcition_score = model.score(x, y)

        info = (snap_ctxt.shape[0], neg_ctxt.shape[0],  model.score(x, y))
        #print(info)
        return model, info

    def update_prox_w_ess_factor(self, cs_model, x, beta=None):
        '''
            This function calculates effective sample size (ESS):
            ESS = ||w||^2_1 / ||w||^2_2  , w = pi / beta
            ESS = ESS / n where n is number of samples to normalize
            x: is (n, D)
        '''
        n = x.shape[0]
        if beta is not None:
            # beta results should be same as using cs_model.predict_proba(x)[:,0] if no clipping
            w = ((torch.sum(beta)**2) /(torch.sum(beta**2) + self.r_eps) )/n
            ess_factor = np.float32(w.numpy())

        else:
            # step 1: get prob class 1
            p0 = cs_model.predict_proba(x)[:,0]
            w =  p0 / ( 1 - p0 + self.r_eps)
            w = (np.sum(w)**2) / (np.sum(w**2) + self.r_eps)
            ess_factor = np.float32(w) / n

        # since we assume task_i is class -1, and replay buffer is 1, then
        ess_prox_factor = 1.0 - ess_factor

        if np.isnan(ess_prox_factor) or np.isinf(ess_prox_factor) or ess_prox_factor <= self.r_eps: # make sure that it is valid
            self.prox_coef = self.prox_coef_init

        else:
            self.prox_coef = ess_prox_factor

    def get_propensity(self, cs_model, curr_pre_act_rew, curr_obs):
        '''
            This function returns propensity for current sample of data 
            simply: exp(f(x))
        '''

        ######
        # extract features: context features 
        ######
        with torch.no_grad():

            # batch_size X context_hidden 
            if self.enable_beta_obs_cxt == True:
                ctxt = torch.cat([curr_obs, self.actor.get_conext_feats(curr_pre_act_rew)], dim = -1).cpu().data.numpy()

            else:
                ctxt = self.actor.get_conext_feats(curr_pre_act_rew).cpu().data.numpy()

        # step 0: get f(x)
        f_prop = np.dot(ctxt, cs_model.coef_.T) + cs_model.intercept_

        # step 1: convert to torch
        f_prop = torch.from_numpy(f_prop).float()

        # To make it more stable, clip it
        f_prop = f_prop.clamp(min=-self.beta_clip)

        # step 2: exp(-f(X)), f_score: N * 1
        f_score = torch.exp(-f_prop)
        f_score[f_score < 0.1]  = 0 # for numerical stability

        if self.use_normalized_beta == True:

            #get logistic regression prediction of class [-1] for current task
            lr_prob = cs_model.predict_proba(ctxt)[:,0]
            # normalize using logistic_probs
            d_pmax_pmin = np.float32(np.max(lr_prob) - np.min(lr_prob))
            f_score = ( d_pmax_pmin * (f_score - torch.min(f_score)) )/( torch.max(f_score) - torch.min(f_score) + self.r_eps ) + np.float32(np.min(lr_prob))

        # update prox coeff with ess.
        if self.use_ess_clipping == True:
            self.update_prox_w_ess_factor(cs_model, ctxt, beta=f_score)


        return f_score, None

    def do_training(self,
                    replay_buffer = None,
                    iterations = None,
                    csc_model = None,
                    apply_prox = False,
                    current_batch_size = None,
                    src_task_ids = []):

        '''
            inputs:
                replay_buffer
                iterations episode_timesteps                 
        '''
        actor_loss_out = 0.0
        critic_loss_out = 0.0
        critic_prox_out = 0.0
        actor_prox_out = 0.0
        list_prox_coefs = [self.prox_coef]

        for it in range(iterations):

            ########
            # Sample replay buffer 
            ########
            if len(src_task_ids) > 0:
                x, y, u, r, d, pu, pr, px, nu, nr, nx = replay_buffer.sample_tasks(task_ids = src_task_ids, batch_size = current_batch_size)

            else:
                x, y, u, r, d, pu, pr, px, nu, nr, nx = replay_buffer.sample(current_batch_size)

            obs = torch.FloatTensor(x).to(self.device)
            next_obs = torch.FloatTensor(y).to(self.device)
            action = torch.FloatTensor(u).to(self.device)
            reward = torch.FloatTensor(r).to(self.device)
            mask = torch.FloatTensor(1 - d).to(self.device)
            previous_action = torch.FloatTensor(pu).to(self.device)
            previous_reward = torch.FloatTensor(pr).to(self.device)
            previous_obs = torch.FloatTensor(px).to(self.device)

            # list of hist_actions and hist_rewards which are one time ahead of previous_ones
            # example:
            # previous_action = [t-3, t-2, t-1]
            # hist_actions    = [t-2, t-1, t]
            hist_actions = torch.FloatTensor(nu).to(self.device)
            hist_rewards = torch.FloatTensor(nr).to(self.device)
            hist_obs     = torch.FloatTensor(nx).to(self.device)


            # combine reward and action
            act_rew = [hist_actions, hist_rewards, hist_obs] # torch.cat([action, reward], dim = -1)
            pre_act_rew = [previous_action, previous_reward, previous_obs] #torch.cat([previous_action, previous_reward], dim = -1)

            if csc_model is None:
                # propensity_scores dim is batch_size 
                # no csc_model, so just do business as usual 
                beta_score = torch.ones((current_batch_size, 1)).to(self.device)

            else:
                # propensity_scores dim is batch_size 
                beta_score, clipping_factor = self.get_propensity(csc_model, pre_act_rew, obs)
                beta_score = beta_score.to(self.device)
                list_prox_coefs.append(self.prox_coef)

            ########
            # Select action according to policy and add clipped noise 
            # mu'(s_t) = mu(s_t | \theta_t) + N (Eq.7 in https://arxiv.org/abs/1509.02971) 
            # OR
            # Eq. 15 in TD3 paper:
            # e ~ clip(N(0, \sigma), -c, c)
            ########
            noise = (torch.randn_like(action) * self.policy_noise ).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_obs, act_rew) + noise).clamp(-self.max_action, self.max_action)

            ########
            #  Update critics
            #  1. Compute the target Q value 
            #  2. Get current Q estimates
            #  3. Compute critic loss
            #  4. Optimize the critic
            ########

            # 1. y = r + \gamma * min{Q1, Q2} (s_next, next_action)
            # if done , then only use reward otherwise reward + (self.gamma * target_Q)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action, act_rew)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (mask * self.gamma * target_Q).detach()

            # 2.  Get current Q estimates
            current_Q1, current_Q2 = self.critic(obs, action, pre_act_rew)


            # 3. Compute critic loss
            # even we picked min Q, we still need to backprob to both Qs
            critic_loss_temp = F.mse_loss(current_Q1, target_Q, reduction='none') + F.mse_loss(current_Q2, target_Q, reduction='none')
            assert critic_loss_temp.shape == beta_score.shape, ('shape critic_loss_temp and beta_score shoudl be the same', critic_loss_temp.shape, beta_score.shape)

            critic_loss = (critic_loss_temp * beta_score).mean()
            critic_loss_out += critic_loss.item()

            if apply_prox:
                # calculate proximal term
                critic_prox = self.get_prox_penalty(self.critic, self.ckpt['critic'])
                critic_loss = critic_loss + self.prox_coef * critic_prox
                critic_prox_out += critic_prox.item()

            # 4. Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            ########
            # Delayed policy updates
            ########
            if it % self.policy_freq == 0:

                # Compute actor loss
                actor_loss_temp = -1 * beta_score * self.critic.Q1(obs, self.actor(obs, pre_act_rew), pre_act_rew)
                actor_loss = actor_loss_temp.mean()
                actor_loss_out += actor_loss.item()

                if apply_prox:
                    # calculate proximal term
                    actor_prox = self.get_prox_penalty(self.actor, self.ckpt['actor'])
                    actor_loss = actor_loss + self.prox_coef * actor_prox
                    actor_prox_out += actor_prox.item()

                # Optimize the actor 
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()


                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.ptau * param.data + (1 - self.ptau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.ptau * param.data + (1 - self.ptau) * target_param.data)

        out = {}
        if iterations == 0:
            out['critic_loss'] = 0
            out['actor_loss']  = 0
            out['prox_critic'] = 0
            out['prox_actor']  = 0
            out['beta_score']  = 0

        else:
            out['critic_loss'] = critic_loss_out/iterations
            out['actor_loss']  = self.policy_freq * actor_loss_out/iterations
            out['prox_critic'] = critic_prox_out/iterations
            out['prox_actor']  = self.policy_freq * actor_prox_out/iterations
            out['beta_score']  = beta_score.cpu().data.numpy().mean()

        #if csc_model and self.use_ess_clipping == True:
        out['avg_prox_coef'] = np.mean(list_prox_coefs)

        return out

    def train_TD3(
                self,
                replay_buffer=None,
                iterations=None,
                tasks_buffer = None,
                train_iter = 0,
                task_id = None,
                nums_snap_trains = 5):

        '''
            inputs:
                replay_buffer
                iterations episode_timesteps
            outputs:

        '''
        actor_loss_out = 0.0
        critic_loss_out = 0.0

        ### if there is no eough data in replay buffer, then reduce size of iteration to 20:
        #if replay_buffer.size_rb() < iterations or replay_buffer.size_rb() <  self.batch_size * iterations:
        #    temp = int( replay_buffer.size_rb()/ (self.batch_size) % iterations ) + 1
        #    if temp < iterations:
        #        iterations = temp

        for it in range(iterations):

            ########
            # Sample replay buffer
            ########
            x, y, u, r, d, pu, pr, px, nu, nr, nx = replay_buffer.sample(self.batch_size)
            obs = torch.FloatTensor(x).to(self.device)
            next_obs = torch.FloatTensor(y).to(self.device)
            action = torch.FloatTensor(u).to(self.device)
            reward = torch.FloatTensor(r).to(self.device)
            mask = torch.FloatTensor(1 - d).to(self.device)
            previous_action = torch.FloatTensor(pu).to(self.device)
            previous_reward = torch.FloatTensor(pr).to(self.device)
            previous_obs = torch.FloatTensor(px).to(self.device)

            # list of hist_actions and hist_rewards which are one time ahead of previous_ones
            # example:
            # previous_action = [t-3, t-2, t-1]
            # hist_actions    = [t-2, t-1, t]
            hist_actions = torch.FloatTensor(nu).to(self.device)
            hist_rewards = torch.FloatTensor(nr).to(self.device)
            hist_obs     = torch.FloatTensor(nx).to(self.device)

            # combine reward and action
            act_rew = [hist_actions, hist_rewards, hist_obs] # torch.cat([action, reward], dim = -1)
            pre_act_rew = [previous_action, previous_reward, previous_obs] #torch.cat([previous_action, previous_reward], dim = -1)

            ########
            # Select action according to policy and add clipped noise
            # mu'(s_t) = mu(s_t | \theta_t) + N (Eq.7 in https://arxiv.org/abs/1509.02971)
            # OR
            # Eq. 15 in TD3 paper:
            # e ~ clip(N(0, \sigma), -c, c)
            ########
            noise = (torch.randn_like(action) * self.policy_noise ).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_obs, act_rew) + noise).clamp(-self.max_action, self.max_action)

            ########
            #  Update critics
            #  1. Compute the target Q value
            #  2. Get current Q estimates
            #  3. Compute critic loss
            #  4. Optimize the critic
            ########

            # 1. y = r + \gamma * min{Q1, Q2} (s_next, next_action)
            # if done , then only use reward otherwise reward + (self.gamma * target_Q)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action, act_rew)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (mask * self.gamma * target_Q).detach()

            # 2.  Get current Q estimates
            current_Q1, current_Q2 = self.critic(obs, action, pre_act_rew)

            # 3. Compute critic loss
            # even we picked min Q, we still need to backprob to both Qs
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            critic_loss_out += critic_loss.item()

            # 4. Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            ########
            # Delayed policy updates
            ########
            if it % self.policy_freq == 0:

                # Compute actor loss
                actor_loss = -self.critic.Q1(obs, self.actor(obs, pre_act_rew), pre_act_rew).mean()
                actor_loss_out += actor_loss.item()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()


                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.ptau * param.data + (1 - self.ptau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.ptau * param.data + (1 - self.ptau) * target_param.data)

        out = {}
        out['critic_loss'] = critic_loss_out/iterations
        out['actor_loss'] = self.policy_freq * actor_loss_out/iterations

        # keep a copy of models' params
        self.copy_model_params()
        return out, None

    def adapt(self,
            train_replay_buffer = None,
            train_tasks_buffer = None,
            eval_task_buffer = None,
            task_id = None,
            snap_iter_nums = 5,
            main_snap_iter_nums = 15,
            sampling_style = 'replay',
            sample_mult = 1
            ):
        '''
            inputs:
                replay_buffer
                iterations episode_timesteps
        '''
        #######
        # Reset optim at the beginning of the adaptation
        #######
        self.actor_optimizer = optim.Adam(self.actor.parameters())
        self.critic_optimizer = optim.Adam(self.critic.parameters())

        #######
        # Adaptaion step:
        # learn a model to correct covariate shift
        #######
        out_single = None

        # train covariate shift correction model
        csc_model, csc_info = self.train_cs(task_id = task_id,
                                            snap_buffer = eval_task_buffer,
                                            train_tasks_buffer = train_tasks_buffer,
                                            adaptation_step = True)

        # train td3 for a single task
        out_single = self.do_training(replay_buffer = eval_task_buffer.get_buffer(task_id),
                                      iterations = snap_iter_nums,
                                      csc_model = None,
                                      apply_prox = False,
                                      current_batch_size = eval_task_buffer.size_rb(task_id))
        #self.copy_model_params()

        # keep a copy of model params for task task_id
        out_single['csc_info'] = csc_info
        out_single['snap_iter'] = snap_iter_nums

        # sampling_style is based on 'replay'
        # each train task has own buffer, so sample from each of them
        out = self.do_training(replay_buffer = train_replay_buffer,
                                   iterations = main_snap_iter_nums,
                                   csc_model = csc_model,
                                   apply_prox = True,
                                   current_batch_size = sample_mult * self.batch_size)

        return out, out_single

    def rollback(self):
        '''
            This function rollback everything to state before test-adaptation
        '''

        ####### ####### ####### Super Important ####### ####### #######
        # It is very important to make sure that we rollback everything to
        # Step 0
        ####### ####### ####### ####### ####### ####### ####### #######
        self.actor.load_state_dict(self.actor_copy.state_dict())
        self.actor_target.load_state_dict(self.actor_target_copy.state_dict())
        self.critic.load_state_dict(self.critic_copy.state_dict())
        self.critic_target.load_state_dict(self.critic_target_copy.state_dict())
        self.actor_optimizer.load_state_dict(self.actor_optimizer_copy.state_dict())
        self.critic_optimizer.load_state_dict(self.critic_optimizer_copy.state_dict())

    def save_model_states(self):

        ####### ####### ####### Super Important ####### ####### #######
        # Step 0: It is very important to make sure that we save model params before
        # do anything here
        ####### ####### ####### ####### ####### ####### ####### #######
        self.actor_copy = deepcopy(self.actor)
        self.actor_target_copy = deepcopy(self.actor_target)
        self.critic_copy = deepcopy(self.critic)
        self.critic_target_copy = deepcopy(self.critic_target)
        self.actor_optimizer_copy  = deepcopy(self.actor_optimizer)
        self.critic_optimizer_copy = deepcopy(self.critic_optimizer)

    def set_training_style(self):
        '''
            This function just selects style of training
        '''
        print('**** TD3 style is selected ****')
        self.training_func = self.train_TD3

    def train(self,
              replay_buffer = None,
              iterations = None,
              tasks_buffer = None,
              train_iter = 0,
              task_id = None,
              nums_snap_trains = 5):
        '''
         This starts type of desired training
        '''
        return self.training_func(  replay_buffer = replay_buffer,
                                    iterations = iterations,
                                    tasks_buffer = tasks_buffer,
                                    train_iter = train_iter,
                                    task_id = task_id,
                                    nums_snap_trains = nums_snap_trains
                                )
