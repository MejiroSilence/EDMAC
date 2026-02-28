import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch.distributions import kl_divergence
from modules.layer.self_atten import SelfAttention
from modules.layer.diff_atten import DiffAttention

class EDMACAblationLateAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(EDMACAblationLateAgent, self).__init__()
        self.args = args
        self.activation_func = nn.ReLU(inplace=True)
        self.aggregated_dim = args.state_repre_dim * args.n_agents

        self.cos_loss_func = nn.CosineEmbeddingLoss(margin=args.cos_loss_margin, reduction='none')

        #get hidden state
        self.obs_encoder = nn.Sequential(
            nn.Linear(input_shape, args.rnn_hidden_dim),
            self.activation_func
        )
        self.rnn = nn.GRUCell(args.rnn_hidden_dim + self.aggregated_dim, args.rnn_hidden_dim)

        #msg generate
        self.seg_encoder = nn.Sequential(
            nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim),
            nn.LayerNorm(args.rnn_hidden_dim),
            self.activation_func,
            nn.Linear(args.rnn_hidden_dim, args.latent_dim),
            nn.Sigmoid()
        )
        self.full_infor_latent_encoder = nn.Sequential(
            nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim),
            nn.LayerNorm(args.rnn_hidden_dim),
            self.activation_func,
            nn.Linear(args.rnn_hidden_dim, args.latent_dim)
        )
        self.env_msg_encoder = nn.Linear(args.latent_dim, 2*args.msg_dim)
        self.dec_msg_encoder = nn.Linear(args.latent_dim, 2*args.msg_dim*args.n_agents)

        #env msg aggregator
        self.self_attn = SelfAttention(args.msg_dim, args.attn_head_num, args.attn_embed_dim)
        self.global_state_reconstructor = nn.Linear(args.attn_head_num*args.n_agents*args.attn_embed_dim, self.aggregated_dim)
        if not args.self_supervise:
            self.state_encoder = nn.Linear(args.state_shape, self.aggregated_dim)

        #dec msg aggregator
        self.diff_attn = DiffAttention(args.rnn_hidden_dim, args.msg_dim, args.msg_dim, args.attn_head_num, args.attn_embed_dim, args.diff_attn_lambda_init)
        
        #decision maker
        self.selfQ = nn.Sequential(
            nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim),
            nn.LayerNorm(args.rnn_hidden_dim),
            self.activation_func,
            nn.Linear(args.rnn_hidden_dim, args.n_actions)
        )
        self.msgQ = nn.Sequential(
            nn.Linear(args.attn_head_num*args.attn_embed_dim, args.rnn_hidden_dim),
            nn.LayerNorm(args.rnn_hidden_dim),
            self.activation_func,
            nn.Linear(args.rnn_hidden_dim, args.n_actions)
        )
        self.beta_encoder = nn.Sequential(
            nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim),
            self.activation_func,
            nn.Linear(args.rnn_hidden_dim, args.n_actions),
            nn.Sigmoid()
        )

        #loss
        self.teammate_action_encoder = nn.Linear(args.n_actions + args.n_agents, args.latent_dim)
        self.variational_estimator = nn.Sequential(
            nn.Linear(args.rnn_hidden_dim + args.latent_dim, args.rnn_hidden_dim),
            nn.LayerNorm(args.rnn_hidden_dim),
            self.activation_func,
            nn.Linear(args.rnn_hidden_dim, 2*args.msg_dim)
        )
        if args.self_supervise:
            self.state_decoder = nn.Sequential(
                nn.Linear(self.aggregated_dim, args.rnn_hidden_dim),
                self.activation_func,
                nn.Linear(args.rnn_hidden_dim, args.state_shape)
            )

    def init_hidden(self):
        # make hidden states on same device as model
        hidden = self.env_msg_encoder.weight.new(1, self.args.rnn_hidden_dim).zero_()
        reconstructed = self.env_msg_encoder.weight.new(1, self.aggregated_dim).zero_()
        return hidden, reconstructed

    def forward(self, inputs, alpha=0, state=None, hidden_state=None, reconstructed = None, test_mode=False, calculate_loss=False):
        b, a, e = inputs.size()
        
        #get hidden state
        x = self.obs_encoder(inputs.view(-1, e))
        if hidden_state is not None:
            hidden_state = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        if reconstructed is not None:
            reconstructed = reconstructed.reshape(-1, self.aggregated_dim)
        x = th.cat((x, reconstructed), dim=-1)
        h = self.rnn(x, hidden_state)

        #msg generate
        w = self.seg_encoder(h)
        h_msg = self.full_infor_latent_encoder(h)
        h_env = h_msg * w
        h_dec = h_msg - h_env
        msg_para_env = self.env_msg_encoder(h_env) #ba,2m
        msg_para_dec = self.dec_msg_encoder(h_dec).view(-1, 2*self.args.msg_dim) #baa, 2m
        if test_mode:
            msg_env = msg_para_env[:, :self.args.msg_dim] #ba,m
            msg_dec = msg_para_dec[:, :self.args.msg_dim] #baa,m
        else:
            msg_para_env[:, self.args.msg_dim:] = th.clamp(th.exp(msg_para_env[:, self.args.msg_dim:]), min=self.args.var_floor)
            msg_para_dec[:, self.args.msg_dim:] = th.clamp(th.exp(msg_para_dec[:, self.args.msg_dim:]), min=self.args.var_floor)
            gaussian_env = D.Normal(msg_para_env[:, :self.args.msg_dim], msg_para_env[:, self.args.msg_dim:] ** (1 / 2))
            gaussian_dec = D.Normal(msg_para_dec[:, :self.args.msg_dim], msg_para_dec[:, self.args.msg_dim:] ** (1 / 2))
            msg_env = gaussian_env.rsample() #ba,m
            msg_dec = gaussian_dec.rsample() #baa,m
        msg_env = msg_env.view(b, a, -1)
        msg_dec = msg_dec.view(b, a, a, -1).transpose(1, 2).contiguous().view(b*a, a, -1)

        #env msg aggregator
        reconstructed = self.self_attn(msg_env)
        reconstructed = self.global_state_reconstructor(reconstructed.view(b, -1))
        reconstructed_back = reconstructed
        true_state = state
        if (not self.args.self_supervise) and (state is not None) and (calculate_loss or alpha < 1):
            true_state = self.state_encoder(state)
            reconstructed = alpha * reconstructed + (1 - alpha) * true_state

        #dec msg aggregator
        aggregated_dec = self.diff_attn(h.view(b*a, 1, -1), msg_dec, msg_dec).view(b*a, -1)

        #decision maker
        Qself = self.selfQ(h)
        Qmsg = self.msgQ(aggregated_dec)
        beta = self.beta_encoder(h)
        Qfinal = beta * Qself + (1 - beta) * Qmsg

        #loss
        extra_losses = None
        if calculate_loss:
            extra_losses = self.get_loss(h_env, h_dec, reconstructed_back, true_state, h, b, a, Qfinal.view(b, a, -1), gaussian_dec)

        return Qfinal.view(b, a, -1), h.view(b, a, -1), extra_losses, reconstructed.view(b, 1, -1).repeat(1, a, 1)


    def get_loss(self, h_env, h_dec, reconstructed, state, h, b, a, q, g1): #TODO: size check
        extra_losses = {}

        cos_loss_label = -1 * h_env.new_ones(b*a)
        extra_losses["cos_loss"] = self.cos_loss_func(h_env, h_dec, cos_loss_label).view(b, a).mean(-1, keepdim=True) #b,1

        if self.args.self_supervise:
            reconstructed = self.state_decoder(reconstructed)
        extra_losses["state_loss"] = (reconstructed - state) ** 2 #b,-1
        extra_losses["state_loss"] = extra_losses["state_loss"].mean(-1, keepdim=True) #b,1

        #h: ba, -1
        h_expand = h.view(b*a, 1, -1).repeat(1, a, 1).view(b*a*a, -1)
        selected_action = q.max(dim=-1, keepdim=True)[1] #b,a,1
        one_hot_action = q.new_zeros((b, a, self.args.n_actions)).scatter(-1, selected_action, 1)
        one_hot_action = one_hot_action.view(b, 1, -1).repeat(1, a, 1).view(-1, self.args.n_actions) #baa, -1
        one_hot_agent = th.eye(a, device=h.device).view(1,-1).repeat(b*a,1).view(b*a*a,-1)#baa,-1
        teammate_action = self.teammate_action_encoder(th.cat((one_hot_agent, one_hot_action), dim=-1))
        inference = self.variational_estimator(th.cat((h_expand, teammate_action), dim=-1))
        inference[:, self.args.msg_dim:] = th.clamp(th.exp(inference[:, self.args.msg_dim:]), min=self.args.var_floor)
        g2 = D.Normal(inference[:, :self.args.msg_dim], inference[:, self.args.msg_dim:] ** (1 / 2))
        extra_losses["KL_loss"] = kl_divergence(g1, g2).view(b, a*a, -1).sum(-1).mean(-1, keepdim=True) #b,1

        return extra_losses