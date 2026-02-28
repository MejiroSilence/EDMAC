import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, q_input_size, k_input_size, v_input_size, heads, embed_size):
        super().__init__()
        self.h = heads
        self.e = embed_size

        self.toqueries = nn.Linear(q_input_size, self.e * self.h, bias = False)
        self.tokeys = nn.Linear(k_input_size, self.e * self.h, bias = False)
        self.tovalues = nn.Linear(v_input_size, self.e * self.h, bias = False)

    def forward(self, q_input, k_input, v_input):
        b, t, _ = q_input.size()
        
        h = self.h 
        e = self.e
        
        queries = self.toqueries(q_input).view(b, -1, h, e)
        keys = self.tokeys(k_input).view(b, -1, h, e)
        values = self.tovalues(v_input).view(b, -1, h, e)

        # dot-product attention
        # folding heads to batch dimensions
        queries = queries.transpose(1, 2).contiguous().view(b * h, -1, e)
        keys = keys.transpose(1, 2).contiguous().view(b * h, -1, e)
        values = values.transpose(1, 2).contiguous().view(b * h, -1, e)
        queries = queries / (e ** (1/4))
        keys = keys / (e ** (1/4))

        dot = torch.bmm(queries, keys.transpose(1, 2))

        # row wise self attention probabilities
        dot = F.softmax(dot, dim=2)
        out = torch.bmm(dot, values).view(b, h, -1, e)
        out = out.transpose(1, 2).contiguous().view(b, -1, h * e)
        return out