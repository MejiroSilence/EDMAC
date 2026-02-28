import torch as th
import torch.nn as nn
from modules.layer.self_atten import SelfAttention

class BS1TardecAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(BS1TardecAgent, self).__init__()
        self.args = args
        self.activation_func = nn.ReLU(inplace=True)
        self.aggregated_dim = args.state_repre_dim * args.n_agents

        self.dec_msg_encoder = nn.Sequential(
            nn.Linear(input_shape + args.rnn_hidden_dim, args.rnn_hidden_dim),
            self.activation_func,
            nn.Linear(args.rnn_hidden_dim, args.msg_dim * args.n_agents)
        )
        self.env_msg_encoder = nn.Sequential(
            nn.Linear(input_shape, args.rnn_hidden_dim),
            self.activation_func,
            nn.Linear(args.rnn_hidden_dim, args.msg_dim)
        )

        self.self_attn = SelfAttention(2 * args.msg_dim, args.attn_head_num, args.attn_embed_dim)
        self.state_reconstructor = nn.Linear(args.attn_head_num*args.n_agents*args.attn_embed_dim, self.aggregated_dim)

        self.gru = nn.GRUCell(self.aggregated_dim + args.n_agents if self.args.gru_with_id else 0, args.rnn_hidden_dim)
        self.final_q_net = nn.Linear(args.rnn_hidden_dim, args.n_actions)


        # loss
        self.state_decoder = nn.Sequential(
                nn.Linear(self.aggregated_dim, args.rnn_hidden_dim),
                self.activation_func,
                nn.Linear(args.rnn_hidden_dim, args.state_shape)
            )


    def init_hidden(self):
        return self.final_q_net.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state, need_loss = False, state = None, test_mode = False):
        b, a, e = inputs.size()

        inputs = inputs.view(-1, e)
        hidden_state = hidden_state.view(-1, self.args.rnn_hidden_dim)

        env_msg = self.env_msg_encoder(inputs).view(b,a,1,-1).repeat(1,1,a,1)
        dec_msg = self.dec_msg_encoder(th.cat((inputs, hidden_state), dim=-1)).view(b,a,a,-1)
        msg = th.cat((env_msg, dec_msg), dim = -1).transpose(1,2).contiguous().view(b*a, a, -1)

        if self.args.noise and not test_mode:
            noise = th.randn(*msg.shape, device = msg.device) * self.args.noise_norm
            msg += noise

        self_attn_out = self.self_attn(msg)
        reconstructed_global_state = self.state_reconstructor(self_attn_out.view(b*a, -1))

        gru_input = reconstructed_global_state.view(b,a,-1)
        if self.args.gru_with_id:
            onthot_id = th.eye(a, device=gru_input.device).view(1,a,a).repeat(b,1,1)
            gru_input = th.cat((gru_input, onthot_id), dim=-1)
        gru_input = gru_input.view(b*a, -1)
        h = self.gru(gru_input, hidden_state)
        final_q = self.final_q_net(h)

        loss = None
        if need_loss:
            loss = self.get_loss(state.view(b,1,-1).repeat(1,a,1), self.state_decoder(reconstructed_global_state).view(b,a,-1))

        return final_q.view(b,a,-1), h.view(b,a,-1), loss
        
    def get_loss(self, state1, state2):
        loss = {}
        b = state1.shape[0]

        loss["state"] = (state1 - state2) ** 2
        loss["state"] = loss["state"].view(b,-1).mean(-1, keepdim=True) #b,1

        return loss
        
        