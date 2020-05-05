import math

import torch
import torch.nn as nn
from torch.nn import init


class GraphGRUCell(nn.Module):

    def __init__(self,
                 input_size,
                 hidden_size,
                 init_edge_weights=None,
                 bias=True,
                 batch_first=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.batch_first = batch_first

        if init_edge_weights is None:
            self.init_edge_weights = nn.Parameter(torch.Tensor(input_size[0], input_size[0]))
            self.__need_train = True
        else:
            self.init_edge_weights = nn.Parameter(init_edge_weights, requires_grad=False)
            self.__need_train = False

        self.graph_gates = nn.Linear(input_size[1] + hidden_size, 2 * hidden_size, bias=self.bias)
        self.graph_can = nn.Linear(input_size[1] + hidden_size, hidden_size, bias=self.bias)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def forward(self, input, hx_prev):
        """

        :param input: (b, n, d_i)
        :param hx_prev: (b, n, d_h)
        :return: (b, n, d_h)
        """

        if self.__need_train:
            edge_weights = torch.softmax(self.init_edge_weights, dim=-1)
        else:
            edge_weights = self.init_edge_weights

        # (b, n, d_h)
        weighted_hx_prev = torch.einsum("bid, ij -> bjd", hx_prev, edge_weights.t())

        # (b, n, d_i + d_h)
        combined = torch.cat([input, weighted_hx_prev], dim=2)

        # (b, n, d_h * 2)
        combined_gates = self.graph_gates(combined)

        gamma, beta = torch.split(combined_gates, self.hidden_size, dim=2)
        reset_gate = torch.sigmoid(gamma)
        update_gate = torch.sigmoid(beta)

        combined = torch.cat([input, reset_gate * weighted_hx_prev], dim=2)
        combined_can = self.graph_can(combined)
        can = torch.tanh(combined_can)

        hx_next = (1. - update_gate) * hx_prev + update_gate * can
        return hx_next


class GraphGRU(nn.Module):
    """门控图神经网络"""

    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 init_edge_weights=None,
                 bias=True,
                 batch_first=True,
                 dropout=0.):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = nn.Dropout(float(dropout))

        cells = []
        for i in range(self.num_layers):
            cur_input_size = input_size if i == 0 else (input_size[0], hidden_size)
            cells.append(GraphGRUCell(input_size=cur_input_size,
                                      hidden_size=hidden_size,
                                      init_edge_weights=init_edge_weights,
                                      bias=bias,
                                      batch_first=batch_first))

        self.cells = nn.ModuleList(cells)

    def _init_hidden(self, input, batch_size):
        return torch.zeros(self.num_layers,
                           batch_size, input.shape[2], self.hidden_size,
                           dtype=input.dtype, device=input.device)

    def forward(self, input, hx=None):
        """

        :param input: (b, s_i, n, d_i)
        :param hx: (n_l, b, n, d_h)
        :return:
        """

        if not self.batch_first:
            input = input.permute(1, 0, 2, 3)

        if hx is None:
            hx = self._init_hidden(input, input.shape[0])

        seq_len = input.shape[1]
        cur_layer_input = input

        last_states = []
        for layer_idx in range(self.num_layers):

            # (b, n, d_h)
            cur_hx = hx[layer_idx]

            output_inner = []
            for t in range(seq_len):
                cur_hx = self.cells[layer_idx](cur_layer_input[:, t], cur_hx)
                output_inner.append(cur_hx)

            # (b, s_o, n, d_h)
            layer_output = torch.stack(output_inner, dim=1)
            layer_output = self.dropout(layer_output)

            cur_layer_input = layer_output
            last_states.append(cur_hx)

        # (n_l, b, n, d_h)
        last_states = torch.stack(last_states, dim=0)

        return layer_output, last_states


if __name__ == '__main__':
    ggru = GraphGRU(input_size=(2, 2),
                    hidden_size=8,
                    num_layers=2)
    input = torch.randn((2, 10, 2, 2))
    output, hidden = ggru(input)
    print(output.shape)
    print(hidden.shape)
