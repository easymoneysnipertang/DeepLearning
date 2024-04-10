import torch
import torch.nn as nn
import math


class LSTMBlock(nn.Module):
    def __init__(self, num_inputs, num_hiddens):
        super().__init__()

        # 输入门参数
        self.W_xi = nn.Parameter(torch.Tensor(num_inputs,num_hiddens))
        self.W_hi = nn.Parameter(torch.Tensor(num_hiddens,num_hiddens))
        self.b_i = nn.Parameter(torch.Tensor(num_hiddens))
        
        # 遗忘门参数
        self.W_xf = nn.Parameter(torch.Tensor(num_inputs, num_hiddens))
        self.W_hf = nn.Parameter(torch.Tensor(num_hiddens, num_hiddens))
        self.b_f = nn.Parameter(torch.Tensor(num_hiddens))

        # 记忆门参数
        self.W_xc = nn.Parameter(torch.Tensor(num_inputs, num_hiddens))
        self.W_hc = nn.Parameter(torch.Tensor(num_hiddens, num_hiddens))
        self.b_c = nn.Parameter(torch.Tensor(num_hiddens))

        # 输出门参数
        self.W_xo = nn.Parameter(torch.Tensor(num_inputs, num_hiddens))
        self.W_ho = nn.Parameter(torch.Tensor(num_hiddens, num_hiddens))
        self.b_o = nn.Parameter(torch.Tensor(num_hiddens))

        self.init_weights()
    
    def init_weights(self):
        # 正态分布初始化
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
 
    def forward(self, x, h_t, c_t):
        x_t = x
        i_t = torch.sigmoid(x_t @ self.W_xi + h_t @ self.W_hi + self.b_i)
        f_t = torch.sigmoid(x_t @ self.W_xf + h_t @ self.W_hf + self.b_f)
        g_t = torch.tanh(x_t @ self.W_xc + h_t @ self.W_hc + self.b_c)
        o_t = torch.sigmoid(x_t @ self.W_xo + h_t @ self.W_ho + self.b_o)
        c_t = f_t * c_t + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        
        hidden_seq = h_t
        return hidden_seq, (h_t, c_t)


class LSTMModel(nn.Module):
    def __init__(self, num_inputs, num_hiddens, num_outputs):
        super().__init__()

        self.lstm1 = LSTMBlock(num_inputs,num_hiddens)
        self.lstm2 = LSTMBlock(num_inputs,num_hiddens)

        self.linear=nn.Sequential(
            nn.Linear(num_hiddens,num_outputs),
            nn.LogSoftmax(dim=1)
        )

    def forward(self,x,h_t,c_t):
        hidden_seq,(temp_h_t,temp_c_t) = self.lstm1(x,h_t,c_t)
        hidden_seq,(temp_h_t,temp_c_t) = self.lstm2(x,temp_h_t,temp_c_t)
        result = self.linear(hidden_seq)
        return result, (temp_h_t, temp_c_t) 
    
    
if __name__ == '__main__':
    num_inputs, num_hiddens, num_outputs = 28, 128, 10
    model = LSTMModel(num_inputs, num_hiddens, num_outputs)
    print(model)
    x = torch.randn(2, 3, 28)
    h_t = torch.zeros(3, 128)
    c_t = torch.zeros(3, 128)
    y, (h_t, c_t) = model(x, h_t, c_t)
    print(y.shape, h_t.shape, c_t.shape)
        