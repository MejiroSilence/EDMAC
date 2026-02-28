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


class EDMACClubAgent_all_envp(nn.Module):
    def __init__(self, input_shape, args):
        super(EDMACClubAgent_all_envp, self).__init__()
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
            nn.Linear(args.latent_dim, 4*args.msg_dim)
        )

        #env msg aggregator
        self.self_attn = SelfAttention(args.msg_dim * 2, args.attn_head_num, args.attn_embed_dim)
        self.global_state_reconstructor = nn.Linear(args.attn_head_num*args.n_agents*args.attn_embed_dim, self.aggregated_dim)

        #update hidden
        self.update_hidden = nn.GRUCell(self.aggregated_dim, args.rnn_hidden_dim)
       
        #decision maker
        self.selfQ = nn.Sequential(
            nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim),
            self.activation_func,
            nn.Linear(args.rnn_hidden_dim, args.n_actions)
        )

        #loss
        self.variational_estimator = nn.Sequential(
            nn.Linear(args.rnn_hidden_dim + args.n_actions, args.rnn_hidden_dim),
            self.activation_func,
            nn.Linear(args.rnn_hidden_dim, 2*args.msg_dim)
        )
        self.state_decoder = nn.Sequential(
            nn.Linear(self.aggregated_dim, args.rnn_hidden_dim),
            self.activation_func,
            nn.Linear(args.rnn_hidden_dim, args.state_shape)
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
        msg_env = msg_para[:, :msg_dim] #ba,m
        msg_dec = msg_para[:, msg_dim:2*msg_dim] #ba,m
        if not test_mode:
            msg_para_env = th.clamp(msg_para[:, 2*msg_dim:3*msg_dim], max=self.args.var_top, min=self.args.var_floor)
            msg_para_dec = th.clamp(msg_para[:, 3*msg_dim:], max=self.args.var_top, min=self.args.var_floor)
            gaussian_env = D.Normal(msg_env, th.exp(msg_para_env * 0.5))
            gaussian_dec = D.Normal(msg_dec, th.exp(msg_para_dec * 0.5))
            msg_env = gaussian_env.rsample() #ba,m
            msg_dec = gaussian_dec.rsample() #ba,m
        msg_env = msg_env.view(b, a, -1)
        msg_dec = msg_dec.view(b, a, -1)

        #env msg aggregator
        self_attn_out = self.self_attn(th.cat((msg_env,msg_dec), dim=-1))
        reconstructed = self.global_state_reconstructor(self_attn_out.view(b, -1))

        #update hidden
        H = self.update_hidden(reconstructed.repeat_interleave(a, dim=0), h)

        #decision maker
        Qself = self.selfQ(H)
        Qfinal =  Qself

        #loss
        extra_losses = None
        if calculate_loss:
            selected_action = Qfinal.max(dim=-1, keepdim=True)[1] #ba,1
            one_hot_action = Qfinal.new_zeros((b*a, self.args.n_actions)).scatter(-1, selected_action, 1)
            one_hot_action = one_hot_action.view(b*a, -1)#ba, action
            md_para = self.variational_estimator(th.cat((h,one_hot_action), dim=-1).detach())
            md_sigma = th.clamp(md_para[:,msg_dim:], max=self.args.var_top, min=self.args.var_floor)
            md_g= D.Normal(md_para[:,:msg_dim], th.exp(md_sigma * 0.5))
            extra_losses = self.get_loss(state, self.state_decoder(reconstructed),
                                         gaussian_dec, md_g,
                                         msg_env, msg_dec)

        return Qfinal.view(b, a, -1), H.view(b, a, -1), extra_losses


    def get_loss(self, state1, state2, g1, g2, m1, m2):
        extra_losses = {}
        b,a,_ = m1.size()

        extra_losses["state"] = (state1 - state2) ** 2 #b,-1
        extra_losses["state"] = extra_losses["state"].mean(-1, keepdim=True) #b,1

        extra_losses["action"] = kl_divergence(g1, g2).view(b, a, -1).sum(-1).mean(-1, keepdim=True) #b,1

        extra_losses["estimator"] = th.zeros_like( extra_losses["state"])
        extra_losses["msg"] = th.zeros_like( extra_losses["state"])

        return extra_losses