import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch.distributions import kl_divergence
from modules.layer.self_atten import SelfAttention
from modules.layer.diff_atten import DiffAttention
from contextlib import contextmanager

@contextmanager
def freeze_module(module):
    states = [p.requires_grad for p in module.parameters()]
    for p in module.parameters():
        p.requires_grad_(False)
    yield
    for p, s in zip(module.parameters(), states):
        p.requires_grad_(s)


class EDMACClubAgent_all_dec(nn.Module):
    def __init__(self, input_shape, args):
        super(EDMACClubAgent_all_dec, self).__init__()
        self.args = args
        self.activation_func = nn.ReLU(inplace=True)
        self.aggregated_dim = args.state_repre_dim * args.n_agents

        #get hidden state
        self.obs_encoder = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)

        #msg generate
        self.msg_encoder = nn.Sequential(
            nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim),
            self.activation_func,
            nn.Linear(args.rnn_hidden_dim, args.latent_dim),
            self.activation_func,
            nn.Linear(args.latent_dim, 2*args.msg_dim)
        )
        #dec msg aggregator
        self.diff_attn = DiffAttention(args.rnn_hidden_dim, args.msg_dim, args.msg_dim, args.attn_head_num, args.attn_embed_dim, args.diff_attn_lambda_init)
        
        #decision maker
        self.selfQ = nn.Sequential(
            nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim),
            self.activation_func,
            nn.Linear(args.rnn_hidden_dim, args.n_actions)
        )
        self.msgQ = nn.Sequential(
            nn.Linear(args.attn_head_num*args.attn_embed_dim, args.rnn_hidden_dim),
            self.activation_func,
            nn.Linear(args.rnn_hidden_dim, args.n_actions)
        )

        #loss
        self.variational_estimator = nn.Sequential(
            nn.Linear(args.rnn_hidden_dim + args.n_actions, args.rnn_hidden_dim),
            self.activation_func,
            nn.Linear(args.rnn_hidden_dim, 2*args.msg_dim)
        )

    def init_hidden(self):
        # make hidden states on same device as model
        hidden = self.obs_encoder.weight.new(1, self.args.rnn_hidden_dim).zero_()
        return hidden

    def forward(self, inputs, hidden_state, state=None, test_mode=False, calculate_loss=False):
        b, a, e = inputs.size()
        
        #get hidden state
        x = self.obs_encoder(inputs.view(-1, e))
        if hidden_state is not None:
            hidden_state = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, hidden_state)

        #msg generate
        msg_dim = self.args.msg_dim
        msg_para = self.msg_encoder(h)
        msg_dec = msg_para[:, :msg_dim] #ba,m
        if not test_mode:
            msg_para_dec = th.clamp(msg_para[:, msg_dim:], max=self.args.var_top, min=self.args.var_floor)
            gaussian_dec = D.Normal(msg_dec, th.exp(msg_para_dec * 0.5))
            msg_dec = gaussian_dec.rsample() #ba,m
        msg_dec = msg_dec.view(b, a, -1)

        #dec msg aggregator
        msg_dec_copied = msg_dec.repeat_interleave(a, dim=0)#ba,a,-1
        aggregated_dec = self.diff_attn(h.view(b*a, 1, -1), msg_dec_copied, msg_dec_copied).view(b*a, -1)

        #decision maker
        Qself = self.selfQ(h)
        Qmsg = self.msgQ(aggregated_dec)
        Qfinal =  Qself + Qmsg

        #loss
        extra_losses = None
        if calculate_loss:
            selected_action = Qfinal.max(dim=-1, keepdim=True)[1] #ba,1
            one_hot_action = Qfinal.new_zeros((b*a, self.args.n_actions)).scatter(-1, selected_action, 1)
            one_hot_action = one_hot_action.view(b*a, -1)#ba, action
            md_para = self.variational_estimator(th.cat((h,one_hot_action), dim=-1).detach())
            md_sigma = th.clamp(md_para[:,msg_dim:], max=self.args.var_top, min=self.args.var_floor)
            md_g= D.Normal(md_para[:,:msg_dim], th.exp(md_sigma * 0.5))
            extra_losses = self.get_loss(gaussian_dec, md_g,
                                         b, a)

        return Qfinal.view(b, a, -1), h.view(b, a, -1), extra_losses


    def get_loss(self, g1, g2, b, a):
        extra_losses = {}

        extra_losses["action"] = kl_divergence(g1, g2).view(b, a, -1).sum(-1).mean(-1, keepdim=True) #b,1

        extra_losses["state"] = th.zeros_like(extra_losses["action"])
        extra_losses["estimator"] = th.zeros_like(extra_losses["action"])
        extra_losses["msg"] = th.zeros_like(extra_losses["action"])

        return extra_losses