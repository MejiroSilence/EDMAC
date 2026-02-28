import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffAttention(nn.Module):
    def __init__(self, q_input_size, k_input_size, v_input_size, heads, embed_size, lambda_init=0.2):
        super().__init__()
        self.h = heads
        self.e = embed_size

        self.toqueries = nn.Linear(q_input_size, 2 * self.e * self.h, bias = False)
        self.tokeys = nn.Linear(k_input_size, 2 * self.e * self.h, bias = False)
        self.tovalues = nn.Linear(v_input_size, self.e * self.h, bias = False)

        self.labmda = nn.Parameter(self.toqueries.weight.new_zeros(4, embed_size).normal_(mean=0,std=0.1))
        self.labmda_init = lambda_init

    def forward(self, q_input, k_input, v_input):
        b, t, _ = q_input.size()
        
        h = self.h 
        e = self.e
        
        queries = self.toqueries(q_input).view(b, -1, 2 * h, e)
        keys = self.tokeys(k_input).view(b, -1, 2 * h, e)
        values = self.tovalues(v_input).view(b, -1, h, e)

        # dot-product attention
        # folding heads to batch dimensions
        queries = queries.transpose(1, 2).contiguous().view(b * 2 * h, -1, e)
        keys = keys.transpose(1, 2).contiguous().view(b * 2 * h, -1, e)
        values = values.transpose(1, 2).contiguous().view(b * h, -1, e)
        queries = queries / (e ** (1/4))
        keys = keys / (e ** (1/4))

        dot = torch.bmm(queries, keys.transpose(1, 2))

        # row wise self attention probabilities
        dot = F.softmax(dot, dim=2)
        lambda_ = self.labmda_init + torch.exp(torch.dot(self.labmda[0],self.labmda[1])) - torch.exp(torch.dot(self.labmda[2],self.labmda[3]))
        dot = dot.view(b*h, 2, t, -1) 
        dot = dot[:, 0] - lambda_ * dot[:, 1]#bh,t,-1
        out = torch.bmm(dot, values).view(b, h, -1, e)
        out = out.transpose(1, 2).contiguous().view(b, -1, h * e)
        return out