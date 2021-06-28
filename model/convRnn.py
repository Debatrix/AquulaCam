import torch
from torch import nn
import torch.nn.functional as F

from model.framework import Module


class ConvGRUCell(Module):
    def __init__(self, input_dim, hidden_dim=None, kernel_size=3, bias=True):
        super(ConvGRUCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim if hidden_dim else input_dim
        self.kernel_size = kernel_size
        self.ConvGates1 = nn.Conv2d(self.input_dim + self.hidden_dim,
                                    2 * self.hidden_dim,
                                    kernel_size,
                                    stride=1,
                                    padding=self.kernel_size // 2,
                                    bias=bias)
        self.ConvGates2 = nn.Conv2d(self.input_dim + self.hidden_dim,
                                    self.hidden_dim,
                                    kernel_size,
                                    stride=1,
                                    padding=self.kernel_size // 2,
                                    bias=bias)

        self.init_params()

    def forward(self, input, hidden):
        if hidden is None:
            size_h = [input.shape[0], self.hidden_dim] + list(input.shape[2:])
            if input.is_cuda == True:
                hidden = torch.zeros(size_h).cuda()
            else:
                hidden = torch.zeros(size_h)
        c1 = self.ConvGates1(torch.cat((input, hidden), 1))
        (rt, ut) = c1.chunk(2, 1)
        reset_gate = torch.sigmoid(rt)
        update_gate = torch.sigmoid(ut)
        gated_hidden = torch.mul(reset_gate, hidden)
        ct = torch.tanh(self.ConvGates2(torch.cat((input, gated_hidden), 1)))
        next_h = torch.mul(update_gate, hidden) + (1 - update_gate) * ct
        return next_h


class ConvLSTMCell(Module):
    def __init__(self, input_dim, hidden_dim=None, kernel_size=3, bias=True):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim if hidden_dim else input_dim
        self.kernel_size = kernel_size

        self.combine = nn.Conv2d(self.input_dim + self.hidden_dim,
                                 4 * self.hidden_dim,
                                 self.kernel_size,
                                 padding=self.kernel_size // 2,
                                 bias=bias)

        self.init_params()

    def forward(self, input, hidden):
        if hidden is None:
            h_cur = torch.zeros_like(input).to(input.device)
            c_cur = torch.zeros_like(input).to(input.device)
        else:
            h_cur, c_cur = hidden

        combined = torch.cat([input, h_cur], dim=1)
        combined = self.combine(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return (h_next, c_next)
