import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch.distributions import kl_divergence
from modules.layer.self_atten import SelfAttention
from modules.layer.diff_atten import DiffAttention

class EDMACVAEAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(EDMACVAEAgent, self).__init__()
        self.args = args
        self.activation_func = nn.ReLU(inplace=True)
        self.aggregated_dim = args.state_repre_dim * args.n_agents

        #get hidden state
        self.obs_encoder = nn.Sequential(
            nn.Linear(input_shape, args.rnn_hidden_dim),
            self.activation_func
        )
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)

        #msg generate
        self.full_infor_latent_encoder = nn.Sequential(
            nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim),
            # nn.LayerNorm(args.rnn_hidden_dim),
            self.activation_func,
            nn.Linear(args.rnn_hidden_dim, 4 * args.latent_dim)
        )
        self.env_msg_encoder = nn.Linear(args.latent_dim, args.msg_dim)
        # self.dec_msg_encoder = nn.Linear(args.latent_dim, args.msg_dim*args.n_agents)
        self.dec_msg_encoder = nn.Linear(args.latent_dim, args.msg_dim)

        #env msg aggregator
        self.self_attn = SelfAttention(args.msg_dim, args.attn_head_num, args.attn_embed_dim)
        self.global_state_reconstructor = nn.Linear(args.attn_head_num*args.n_agents*args.attn_embed_dim, self.aggregated_dim)
        if not args.self_supervise:
            self.state_encoder = nn.Linear(args.state_shape, self.aggregated_dim)

        #update hidden
        self.update_hidden = nn.GRUCell(self.aggregated_dim, args.rnn_hidden_dim)

        #dec msg aggregator
        self.diff_attn = DiffAttention(args.rnn_hidden_dim, args.msg_dim, args.msg_dim, args.attn_head_num, args.attn_embed_dim, args.diff_attn_lambda_init)
        
        #decision maker
        self.selfQ = nn.Sequential(
            nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim),
            # nn.LayerNorm(args.rnn_hidden_dim),
            # self.activation_func,
            # nn.Linear(args.rnn_hidden_dim, args.n_actions)
        )
        self.msgQ = nn.Sequential(
            nn.Linear(args.attn_head_num*args.attn_embed_dim, args.rnn_hidden_dim),
            # nn.LayerNorm(args.rnn_hidden_dim),
            # self.activation_func,
            # nn.Linear(args.rnn_hidden_dim, args.n_actions)
        )
        self.beta_encoder = nn.Sequential(
            nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim),
            # self.activation_func,
            # nn.Linear(args.rnn_hidden_dim, args.n_actions),
            # nn.Sigmoid()
        )

        #loss
        # self.teammate_action_encoder = nn.Linear(args.n_actions * args.n_agents, args.latent_dim)
        # self.variational_estimator = nn.Sequential(
        #     nn.Linear(args.rnn_hidden_dim + args.latent_dim, args.rnn_hidden_dim),
        #     # nn.LayerNorm(args.rnn_hidden_dim),
        #     self.activation_func,
        #     nn.Linear(args.rnn_hidden_dim, 2*args.latent_dim)
        # )
        # if args.self_supervise:
        #     self.state_decoder = nn.Sequential(
        #         nn.Linear(self.aggregated_dim, args.rnn_hidden_dim),
        #         self.activation_func,
        #         nn.Linear(args.rnn_hidden_dim, args.state_shape)
        #     )

    def init_hidden(self):
        # make hidden states on same device as model
        hidden = self.env_msg_encoder.weight.new(1, self.args.rnn_hidden_dim).zero_()
        return hidden

    def forward(self, inputs, alpha=0, state=None, hidden_state=None, test_mode=False, calculate_loss=False):
        b, a, e = inputs.size()
        
        #get hidden state
        x = self.obs_encoder(inputs.view(-1, e))
        if hidden_state is not None:
            hidden_state = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, hidden_state)

        #msg generate
        h_msg_para = self.full_infor_latent_encoder(h)
        mu = h_msg_para[:, :2*self.args.latent_dim]
        sigma = h_msg_para[:, 2*self.args.latent_dim:]
        if test_mode:
            h_msg = mu
        else:
            # sigma = th.clamp(th.exp(sigma), min=self.args.var_floor)
            # gaussian = D.Normal(mu, sigma ** (1 / 2))
            sigma = th.clamp(sigma / 2, min=self.args.var_floor)
            gaussian = D.Normal(mu, th.exp(sigma))
            h_msg = gaussian.rsample()
        msg_env = self.env_msg_encoder(h_msg[:,:self.args.latent_dim]).view(b, a, -1)
        msg_dec = self.dec_msg_encoder(h_msg[:,self.args.latent_dim:]).view(b, a, -1)
        # msg_dec = self.dec_msg_encoder(h_msg[:,self.args.latent_dim:]).view(b, a, a, -1).transpose(1, 2).contiguous().view(b*a, a, -1)

        #env msg aggregator
        reconstructed = self.self_attn(msg_env)
        reconstructed = self.global_state_reconstructor(reconstructed.view(b, -1))
        reconstructed_back = reconstructed
        true_state = state
        if (not self.args.self_supervise) and (state is not None) and (calculate_loss or alpha < 1):
            true_state = self.state_encoder(state)
            reconstructed = alpha * reconstructed + (1 - alpha) * true_state

        #update hidden
        h_back = h
        h = self.update_hidden(reconstructed.view(b, 1, -1).repeat(1, a, 1).view(b*a, -1), h)

        #dec msg aggregator
        # aggregated_dec = self.diff_attn(h.view(b*a, 1, -1), msg_dec, msg_dec).view(b*a, -1)
        msg_dec_copied = msg_dec.repeat_interleave(a, dim=0)#ba,a,-1
        aggregated_dec = self.diff_attn(h.view(b*a, 1, -1), msg_dec_copied, msg_dec_copied).view(b*a, -1)

        #decision maker
        Qself = self.selfQ(h)
        Qmsg = self.msgQ(aggregated_dec)
        beta = self.beta_encoder(h)
        Qfinal = beta * Qself + (1 - beta) * Qmsg

        #loss
        extra_losses = None
        if calculate_loss:
            extra_losses = self.get_loss(h_msg, gaussian, reconstructed_back, true_state, h_back, b, a, Qfinal.view(b, a, -1))

        return Qfinal.view(b, a, -1), h.view(b, a, -1), extra_losses


    def get_loss(self, h_msg, g, reconstructed, state, h, b, a, q):
        extra_losses = {}

        extra_losses["cos_loss"] = self.MWS(h_msg, g) #1

        if self.args.self_supervise:
            reconstructed = self.state_decoder(reconstructed)
        extra_losses["state_loss"] = (reconstructed - state) ** 2 #b,-1
        extra_losses["state_loss"] = extra_losses["state_loss"].mean(-1, keepdim=True) #b,1


        selected_action = q.max(dim=-1, keepdim=True)[1] #b,a,1
        one_hot_action = q.new_zeros((b, a, self.args.n_actions)).scatter(-1, selected_action, 1)
        one_hot_action = one_hot_action.view(b, -1)#b, a*action
        teammate_action = self.teammate_action_encoder(one_hot_action).view(b,1,-1).repeat(1,a,1).view(b*a,-1)#ba,-1
        inference = self.variational_estimator(th.cat((h, teammate_action), dim=-1))#ba,-1
        inference[:, self.args.latent_dim:] = th.clamp(th.exp(inference[:, self.args.latent_dim:]), min=self.args.var_floor)
        g2 = D.Normal(inference[:, :self.args.latent_dim], inference[:, self.args.latent_dim:] ** (1 / 2))
        mu = g.mean[:,self.args.latent_dim:]
        sigma = g.stddev[:,self.args.latent_dim:]
        g1 = D.Normal(mu, sigma)
        extra_losses["KL_loss"] = kl_divergence(g1, g2).view(b, a, -1).sum(-1).mean(-1, keepdim=True) #b,1

        return extra_losses
    

    def MWS(self, z, g):
        m = z.shape[0]
        e = int(z.shape[-1]/2)

        z = z.view(m,1,-1).expand(-1,m,-1)
        log_probs = g.log_prob(z)
        log_prob1 = th.sum(log_probs[:,:,:e],dim=-1,keepdim=False)#m,m
        log_prob2 = th.sum(log_probs[:,:,e:],dim=-1,keepdim=False)#m,m
        sum1 = self.logsumexp(log_prob1,-1)
        sum2 = self.logsumexp(log_prob2,-1)
        marginal_entropies = sum1 + sum2 #m
        marginal_entropies = th.sum(marginal_entropies, dim =-1, keepdim=True)
        joint_entropy = self.logsumexp(log_prob1+log_prob2,-1) #m
        joint_entropy = th.sum(joint_entropy, dim =-1, keepdim=True) #1
        
        return (joint_entropy-marginal_entropies)/m #1


        

    def logsumexp(self, value, dim, keepdim=False):
        m, _ = th.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + th.log(th.sum(th.exp(value0), dim=dim, keepdim=keepdim))


