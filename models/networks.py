from __future__ import  print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from misc.utils import get_action_info

##################################
# Actor and Critic Newtork for TD3
##################################
class Actor(nn.Module):
    """
      This arch is standard based on https://github.com/sfujim/TD3/blob/master/TD3.py
    """
    def __init__(self,
                action_space,
                hidden_sizes = [400, 300],
                input_dim = None,
                hidden_activation = F.relu,
                max_action= None,
                enable_context = False,
                hiddens_dim_conext = [50],
                input_dim_context=None,
                output_conext=None,
                only_concat_context = 0,
                history_length = 1,
                obsr_dim = None,
                device = 'cpu'
                ):

        super(Actor, self).__init__()
        self.hsize_1 = hidden_sizes[0]
        self.hsize_2 = hidden_sizes[1]
        action_dim, action_space_type = get_action_info(action_space)

        self.actor = nn.Sequential(
                        nn.Linear(input_dim[0], self.hsize_1),
                        nn.ReLU(),
                        nn.Linear(self.hsize_1, self.hsize_2),
                        nn.ReLU()
                        )
        self.out = nn.Linear(self.hsize_2,  action_dim)
        self.max_action = max_action
        self.enable_context = enable_context
        self.output_conext = output_conext

        #context network
        self.context = None
        if self.enable_context == True:
            self.context = Context(hidden_sizes=hiddens_dim_conext,
                                   input_dim=input_dim_context,
                                   output_dim = output_conext,
                                   only_concat_context = only_concat_context,
                                   history_length = history_length,
                                   action_dim = action_dim,
                                   obsr_dim = obsr_dim,
                                   device = device
                                   )

    def forward(self, x, pre_act_rew = None, state = None, ret_context = False):
        '''
            input (x  : B * D where B is batch size and D is input_dim
            pre_act_rew: B * (A + 1) where B is batch size and A + 1 is input_dim
        '''
        combined = None
        if self.enable_context == True:
            combined = self.context(pre_act_rew)
            x = torch.cat([x, combined], dim = -1)

        x = self.actor(x)
        x = self.max_action * torch.tanh(self.out(x))

        if ret_context == True:
            return x, combined

        else:
            return x

    def get_conext_feats(self, pre_act_rew):
        '''
            pre_act_rew: B * (A + 1) where B is batch size and A + 1 is input_dim
            return combine features
        '''
        combined = self.context(pre_act_rew)

        return combined

class Critic(nn.Module):
    """
      This arch is standard based on https://github.com/sfujim/TD3/blob/master/TD3.py
    """
    def __init__(self,
                action_space,
                hidden_sizes = [400, 300],
                input_dim = None,
                hidden_activation = F.relu,
                enable_context = False,
                dim_others = 0,
                hiddens_dim_conext = [50],
                input_dim_context=None,
                output_conext=None,
                only_concat_context = 0,
                history_length = 1,
                obsr_dim = None,
                device = 'cpu'
                ):

        super(Critic, self).__init__()
        self.hsize_1 = hidden_sizes[0]
        self.hsize_2 = hidden_sizes[1]
        action_dim, action_space_type = get_action_info(action_space)

        # handling extra dim
        self.enable_context = enable_context

        if self.enable_context == True:
            self.extra_dim = dim_others # right now, we add reward + previous action

        else:
            self.extra_dim = 0

        # It uses two different Q networks
        # Q1 architecture
        self.q1 = nn.Sequential(
                        nn.Linear(input_dim[0] + action_dim + self.extra_dim, self.hsize_1),
                        nn.ReLU(),
                        nn.Linear(self.hsize_1, self.hsize_2),
                        nn.ReLU(),
                        nn.Linear(self.hsize_2, 1),
                        )


        # Q2 architecture
        self.q2 = nn.Sequential(
                        nn.Linear(input_dim[0] + action_dim + self.extra_dim, self.hsize_1),
                        nn.ReLU(),
                        nn.Linear(self.hsize_1, self.hsize_2),
                        nn.ReLU(),
                        nn.Linear(self.hsize_2, 1),
                        )

        if self.enable_context == True:
            self.context = Context(hidden_sizes=hiddens_dim_conext,
                                   input_dim=input_dim_context,
                                   output_dim = output_conext,
                                   only_concat_context = only_concat_context,
                                   history_length = history_length,
                                   action_dim = action_dim,
                                   obsr_dim = obsr_dim,
                                   device = device
                                   )

    def forward(self, x, u, pre_act_rew = None, ret_context = False):
        '''
            input (x): B * D where B is batch size and D is input_dim
            input (u): B * A where B is batch size and A is action_dim
            pre_act_rew: B * (A + 1) where B is batch size and A + 1 is input_dim
        '''
        xu = torch.cat([x, u], 1)
        combined = None

        if self.enable_context == True:
            combined = self.context(pre_act_rew)
            xu = torch.cat([xu, combined], dim = -1)

        # Q1
        x1 = self.q1(xu)
        # Q2
        x2 = self.q2(xu)

        if ret_context == True:
            return x1, x2, combined

        else:
            return x1, x2

    def Q1(self, x, u, pre_act_rew = None, ret_context = False):
        '''
            input (x): B * D where B is batch size and D is input_dim
            input (u): B * A where B is batch size and A is action_dim
            pre_act_rew: B * (A + 1) where B is batch size and A + 1 is input_dim
        '''

        xu = torch.cat([x, u], 1)
        combined = None

        if self.enable_context == True:
            combined = self.context(pre_act_rew)
            xu = torch.cat([xu, combined], dim = -1)

        # Q1
        x1 = self.q1(xu)

        if ret_context == True:
            return x1, combined

        else:
            return x1

    def get_conext_feats(self, pre_act_rew):
        '''
            pre_act_rew: B * (A + 1) where B is batch size and A + 1 is input_dim
            return combine features
        '''
        combined = self.context(pre_act_rew)

        return combined

class Context(nn.Module):
    """
      This layer just does non-linear transformation(s)
    """
    def __init__(self,
                 hidden_sizes = [50],
                 output_dim = None,
                 input_dim = None,
                 only_concat_context = 0,
                 hidden_activation=F.relu,
                 history_length = 1,
                 action_dim = None,
                 obsr_dim = None,
                 device = 'cpu'
                 ):

        super(Context, self).__init__()
        self.only_concat_context = only_concat_context
        self.hid_act = hidden_activation
        self.fcs = [] # list of linear layer
        self.hidden_sizes = hidden_sizes
        self.input_dim = input_dim
        self.output_dim_final = output_dim # count the fact that there is a skip connection
        self.output_dim_last_layer  = output_dim // 2
        self.hist_length = history_length
        self.device = device
        self.action_dim = action_dim
        self.obsr_dim = obsr_dim

        #### build LSTM or multi-layers FF
        if only_concat_context == 3:
            # use LSTM or GRU
            self.recurrent =nn.GRU(self.input_dim,
                               self.hidden_sizes[0],
                               bidirectional = False,
                               batch_first = True,
                               num_layers = 1)

    def init_recurrent(self, bsize = None):
        '''
            init hidden states
            Batch size can't be none
        '''
        # The order is (num_layers, minibatch_size, hidden_dim)
        # LSTM ==> return (torch.zeros(1, bsize, self.hidden_sizes[0]),
        #        torch.zeros(1, bsize, self.hidden_sizes[0]))
        return torch.zeros(1, bsize, self.hidden_sizes[0]).to(self.device)

    def forward(self, data):
        '''
            pre_x : B * D where B is batch size and D is input_dim
            pre_a : B * A where B is batch size and A is input_dim
            previous_reward: B * 1 where B is batch size and 1 is input_dim
        '''
        previous_action, previous_reward, pre_x = data[0], data[1], data[2]
        
        if self.only_concat_context == 3:
            # first prepare data for LSTM
            bsize, dim = previous_action.shape # previous_action is B* (history_len * D)
            pacts = previous_action.view(bsize, -1, self.action_dim) # view(bsize, self.hist_length, -1)
            prews = previous_reward.view(bsize, -1, 1) # reward dim is 1, view(bsize, self.hist_length, 1)
            pxs   = pre_x.view(bsize, -1, self.obsr_dim ) # view(bsize, self.hist_length, -1)
            pre_act_rew = torch.cat([pacts, prews, pxs], dim = -1) # input to LSTM is [action, reward]

            # init lstm/gru
            hidden = self.init_recurrent(bsize=bsize)

            # lstm/gru
            _, hidden = self.recurrent(pre_act_rew, hidden) # hidden is (1, B, hidden_size)
            out = hidden.squeeze(0) # (1, B, hidden_size) ==> (B, hidden_size)

            return out

        else:
            raise NotImplementedError

        return None