# Model file Author Xiangyu Meng
import torch.nn as nn
from torch.distributions.normal import Normal
import torch
import torch.fft as fft
import torch.nn.functional as F
from torch.nn.modules import activation
from torch.nn.modules.activation import Tanh
from torch.nn.modules.linear import Linear
from layers import *
import numpy as np
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from layers import *
import math
from inspect import isfunction
class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""
        # 门数量
        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0).exp()

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        # add eps to all zero values in order to avoid nans when going back to log space
        combined[combined == 0] = np.finfo(float).eps
        # back to log space
        return combined.log()

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class MLP(nn.Module):
    def __init__(self, input_size, output_size,dropout = 0):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        # self.fc2 = nn.Linear(hidden_size, output_size)
        self.act = nn.Tanh()
        self.dropout = dropout
        if self.dropout>0:
            self.dropout_layer = nn.Dropout(dropout)


    def forward(self, x):
        out = self.fc1(x)
        out = self.act(out)
        if self.dropout>0:
            out = self.dropout_layer(out)
        return out


class AttentionMoE(nn.Module):
    
    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts default 6
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(self, input_size, output_size, num_experts, noisy_gating=True, k=1):
        super(AttentionMoE, self).__init__()
        # s
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size

        self.k = k
        # instantiate experts
        # assert (num_experts % 3 ==0)
        model = []
        # for i in range(int(self.num_experts/3)):
        #     model.extend([Attention(self.input_size),Attention(self.input_size),Attention(self.input_size)])
            
        # self.experts = nn.ModuleList(
        #     model
        # )
        self.experts = nn.ModuleList([Attention(self.input_size) for i in range(self.num_experts)])
        # self.experts = nn.ModuleList([MLP(self.input_size, self.output_size) for i in range(self.num_experts)])
        self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True).cuda() #(1000,10)
        self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True).cuda() #(1000,10)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))  #self. mean
        self.register_buffer("std", torch.tensor([1.0]))    # self.std
        assert(self.k <= self.num_experts)

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)
        # 输入参数为：没有引入噪声的value，加入噪声的value，噪声的标准差，
    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1) #5
        top_values_flat = noisy_top_values.flatten() #加噪声的top k logits, 对维度为n*5的张量拍平
        # torch.arrange 0-32 对每一个batch找到其中对应的位置，找到每个batch中前k个里面最小的数据
        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        value_if_in = torch.gather(top_values_flat, 0, threshold_positions_if_in)
        threshold_if_in = torch.unsqueeze(value_if_in, 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        value_if_out = torch.gather(top_values_flat, 0, threshold_positions_if_out)
        threshold_if_out = torch.unsqueeze(value_if_out, 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev) #计算在这个值下的概率
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        clean_logits = x @ self.w_gate #输入和门函数矩阵乘法 得到logit
        if self.noisy_gating and train:  #
            raw_noise_stddev = x @ self.w_noise #x和noise矩阵乘法
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))  #softplus防止梯度爆炸
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)  #添加噪声
            logits = noisy_logits
        else:
            logits = clean_logits # (32,10)

        # calculate topk + 1 that will be needed for the noisy gates  计算前k个最大值与索引
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        # print(top_logits)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits) #对前k个

        zeros = torch.zeros_like(logits, requires_grad=True) # (32,10)
        gates = zeros.scatter(1, top_k_indices, top_k_gates) # 把模型的softmax scatter到gates

        if self.noisy_gating and self.k < self.num_experts and train:
            # 加入噪声的noise，
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        # print(gates.size())
        # print(load.size())
        return gates, load

    def forward(self, x, loss_coef=1e-2):
        """Args:
        x: tensor shape [batch_size, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses

        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        gates, load = self.noisy_top_k_gating(x, self.training)
        # calculate importance loss 
        importance = gates.sum(0)
        #
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)
        # print(x.size())
        expert_inputs = dispatcher.dispatch(x)
        # for i in range(self.num_experts):
        #     print(expert_inputs[i].size()) 
        # print(expert_inputs.size())
        gates = dispatcher.expert_to_gates()
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
        y = dispatcher.combine(expert_outputs)
        return y, loss



class MoE(nn.Module):

    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(self, input_size, output_size, num_experts, noisy_gating=True, k=6, dropout = 0):
        super(MoE, self).__init__()
        # s
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size

        self.k = k
        # instantiate expertsx
        self.experts = nn.ModuleList([MLP(self.input_size, self.output_size, dropout) for i in range(self.num_experts)])
        self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True).cuda() #(1000,10)
        self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True).cuda() #(1000,10)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))  #self. mean
        self.register_buffer("std", torch.tensor([1.0]))    # self.std
        assert(self.k <= self.num_experts)

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-9
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)
        # 输入参数为：没有引入噪声的value，加入噪声的value，噪声的标准差，
    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1) #5
        top_values_flat = noisy_top_values.flatten() #加噪声的top k logits, 对维度为n*5的张量拍平
        # torch.arrange 0-32 对每一个batch找到其中对应的位置，找到每个batch中前k个里面最小的数据
        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        value_if_in = torch.gather(top_values_flat, 0, threshold_positions_if_in)
        threshold_if_in = torch.unsqueeze(value_if_in, 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        value_if_out = torch.gather(top_values_flat, 0, threshold_positions_if_out)
        threshold_if_out = torch.unsqueeze(value_if_out, 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev) #计算在这个值下的概率
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        # print(x.device)
        # print(self.w_gate.device)
        clean_logits = x @ self.w_gate #输入和门函数矩阵乘法 得到logit
        if self.noisy_gating and train:  #
            raw_noise_stddev = x @ self.w_noise #x和noise矩阵乘法
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))  #softplus防止梯度爆炸
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)  #添加噪声
            logits = noisy_logits
        else:
            logits = clean_logits # (32,10)

        # calculate topk + 1 that will be needed for the noisy gates  计算前k个最大值与索引
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        # print(top_logits)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits) #对前k个

        zeros = torch.zeros_like(logits, requires_grad=True) # (32,10)
        gates = zeros.scatter(1, top_k_indices, top_k_gates) # 把模型的softmax scatter到gates

        if self.noisy_gating and self.k < self.num_experts and train:
            # 加入噪声的noise，
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, loss_coef=1e-2):
        """Args:
        x: tensor shape [batch_size, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses

        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        gates, load = self.noisy_top_k_gating(x, self.training)
        # calculate importance loss 
        importance = gates.sum(0)
        #
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        gates = dispatcher.expert_to_gates()
        # for i in range(self.num_experts):
        #     print(expert_inputs[i].size()) 
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
        y = dispatcher.combine(expert_outputs)
        return y, loss

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))
class Discriminator(nn.Module):
    def __init__(self,seq_length,sample_length):
        super(Discriminator,self).__init__()
        self.main=nn.Sequential(
            nn.Linear(seq_length,sample_length),
            # nn.LeakyReLU(0.2,inplace=True),
            Mish(),

            # nn.Linear(sample_length,int(sample_length/2)),

            nn.Linear(sample_length,sample_length//2),
            # nn.LeakyReLU(0.2,inplace=True),
            Mish(),

            # nn.Linear(int(sample_length/2),code_dim)
            nn.Linear(sample_length//2,sample_length//4),
            Mish(),

            nn.Linear(sample_length//4,1)
            

        )
        # print(self.modules())

        self._init_weight()
    def forward(self,x):
        return self.main(x)
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m,nn.Linear):

                nn.init.xavier_normal_(m.weight)

class VAE(nn.Module):
    def __init__(self,seq_length,sample_length,code_dim,dropout=True):
        super(VAE,self).__init__()
        
        model1=[
            nn.Linear(seq_length,sample_length),
            # nn.BatchNorm1d(sample_length),
            # nn.LeakyReLU(0.2,inplace=True),
            # Mish()
            nn.Tanh(),
            # nn.Dropout(0.5),
        ]

        self.downsample1=nn.Sequential(*model1)
        
        # self.attention=Attention(sample_length)

        self.encode_u=nn.Sequential(
            nn.Linear(sample_length,code_dim),
            # nn.BatchNorm1d(code_dim),
            # nn.LeakyReLU(0.2,inplace=True)
            
            nn.Tanh(),
            
        )

        self.encode_si = nn.Sequential(
            nn.Linear(sample_length,code_dim),
            # nn.BatchNorm1d(code_dim),
            # nn.LeakyReLU(0.2,inplace=True)
            
            nn.Tanh(),
            

        )

        self.decode = nn.Sequential(
			nn.Linear(code_dim, sample_length),
			nn.Tanh(),

			nn.Linear(sample_length, seq_length)
		)

        # self._init_weight()

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        assert not torch.isnan(std).any() and not torch.isnan(eps).any()
        return eps.mul(std).add_(mu)

    def dimention_reduction(self, x):
        h = self.downsample1(x)
        mu = self.encode_u(h)
        return mu
        
    def forward(self,x):
        x=self.downsample1(x)
        mu = self.encode_u(x)
        var = self.encode_si(x)
        z = self._reparameterize(mu,var)
        rec = self.decode(z)
        # x=self.attention(x)
        return rec, mu, var
    
    def _init_weight(self):
        # for m in self.downsample1.modules():
        #     if isinstance(m,nn.Linear):
        #         nn.init.kaiming_normal_(m.weight)
        # for m in self.downsample2.modules():
        #     if isinstance(m,nn.Linear):
        #         nn.init.kaiming_normal_(m.weight)
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.xavier_normal_(m.weight)


class MaskAttention(nn.Module):
    def __init__(self,input_dim,dropout = True):
        super(MaskAttention,self).__init__()
        model = [
            nn.Linear(input_dim,input_dim),
            nn.Sigmoid(),
        ]
        if dropout:
            model.append(nn.Dropout(0.2))
        self.atten = nn.Sequential(*model)
    def forward(self, x):
        atten = self.atten(x)
        x = atten*x
        return x
class AVAE(nn.Module):
    def __init__(self,seq_length,sample_length,code_dim,dropout=True):
        super(AVAE,self).__init__()
        
        model1=[
            nn.Linear(seq_length,sample_length),
            # nn.BatchNorm1d(sample_length),
            # nn.LeakyReLU(0.2,inplace=True),
            # Mish()
            nn.Tanh(),
            
        ]
        if dropout:
            model1.append(nn.Dropout(0.5))
        self.downsample1=nn.Sequential(*model1)
        
        self.attention = nn.Sequential(
            nn.Linear(sample_length,sample_length),
            nn.Sigmoid(),
            nn.Dropout(0.2)
            
        )
        # self.attention=Attention(sample_length)

        self.encode_u=nn.Sequential(
            nn.Linear(sample_length,code_dim),
            # nn.BatchNorm1d(code_dim),
            # nn.LeakyReLU(0.2,inplace=True)
            
            nn.Tanh(),
            nn.Dropout(0.5)
            
        )

        self.encode_si = nn.Sequential(
            nn.Linear(sample_length,code_dim),
            # nn.BatchNorm1d(code_dim),
            # nn.LeakyReLU(0.2,inplace=True)
            
            nn.Tanh(),
            nn.Dropout(0.5)
            

        )

        self.decode = nn.Sequential(
			nn.Linear(code_dim, sample_length),
			nn.Tanh(),
            nn.Dropout(0.5),
			nn.Linear(sample_length, seq_length)
		)

        self._init_weight()

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        assert not torch.isnan(std).any() and not torch.isnan(eps).any()
        return eps.mul(std).add_(mu)

    def dimention_reduction(self, x):
        h = self.downsample1(x)
        atten = self.attention(h)
        h = h*atten
        mu = self.encode_u(h)
        return mu
        
    def forward(self,x):
        x=self.downsample1(x)
        atten = self.attention(x)
        x = x*atten
        mu = self.encode_u(x)
        var = self.encode_si(x)
        z = self._reparameterize(mu,var)
        rec = self.decode(z)
        # x=self.attention(x)
        return rec, mu, var
    
    def _init_weight(self):
        # for m in self.downsample1.modules():
        #     if isinstance(m,nn.Linear):
        #         nn.init.kaiming_normal_(m.weight)
        # for m in self.downsample2.modules():
        #     if isinstance(m,nn.Linear):
        #         nn.init.kaiming_normal_(m.weight)
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.xavier_normal_(m.weight)

#######################################

MIN_EXPERT_CAPACITY = 4

# helper functions

def default(val, default_val):
    default_val = default_val() if isfunction(default_val) else default_val
    return val if val is not None else default_val

def cast_tuple(el):
    return el if isinstance(el, tuple) else (el,)

# tensor related helper functions

def top1(t):
    values, index = t.topk(k=1, dim=-1)
    values, index = map(lambda x: x.squeeze(dim=-1), (values, index))
    return values, index

def cumsum_exclusive(t, dim=-1):
    num_dims = len(t.shape)
    num_pad_dims = - dim - 1
    pre_padding = (0, 0) * num_pad_dims
    pre_slice   = (slice(None),) * num_pad_dims
    padded_t = F.pad(t, (*pre_padding, 1, 0)).cumsum(dim=dim)
    return padded_t[(..., slice(None, -1), *pre_slice)]

# pytorch one hot throws an error if there are out of bound indices.
# tensorflow, in contrast, does not throw an error
def safe_one_hot(indexes, max_length):
    max_index = indexes.max() + 1
    return F.one_hot(indexes, max(max_index + 1, max_length))[..., :max_length]

def init_(t):
    dim = t.shape[-1]
    std = 1 / math.sqrt(dim)
    return t.uniform_(-std, std)

# activations

class GELU_(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

GELU = nn.GELU if hasattr(nn, 'GELU') else GELU_

# expert class

class Experts(nn.Module):
    def __init__(self,
        dim,
        num_experts = 16,
        hidden_dim = None,
        activation = GELU):
        super().__init__()

        hidden_dim = default(hidden_dim, dim * 4)
        num_experts = cast_tuple(num_experts)

        w1 = torch.zeros(*num_experts, dim, dim)
        w2 = torch.zeros(*num_experts, dim, dim)
        w3 = torch.zeros(*num_experts, dim, dim)

        w1 = init_(w1)
        w2 = init_(w2)
        w3 = init_(w3)

        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)
        self.w3 = nn.Parameter(w3)
        # self.act = activation()
        self.dropout = nn.Dropout(0.2)
    def forward(self, x):
        q = torch.einsum('...nd,...dh->...nh', x, self.w1)
        k = torch.einsum('...nd,...dh->...nh', x, self.w2)
        v = torch.einsum('...nd,...dh->...nh', x, self.w3)
        atten = torch.einsum('...nh,...nd->...hd', q,k)
        atten = F.softmax(atten,dim=-1)
        # atten = self.dropout(atten)
        out = torch.einsum('...nh,...hh->...nh',v,atten)
    
        return out

# the below code is almost all transcribed from the official tensorflow version, from which the papers are written
# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/research/moe.py

# gating network

class Top2Gating(nn.Module):
    def __init__(
        self,
        dim,
        num_gates,
        eps = 1e-9,
        outer_expert_dims = tuple(),
        second_policy_train = 'random',
        second_policy_eval = 'random',
        second_threshold_train = 0.2,
        second_threshold_eval = 0.2,
        capacity_factor_train = 1.25,
        capacity_factor_eval = 2.):
        super().__init__()

        self.eps = eps
        self.num_gates = num_gates
        self.w_gating = nn.Parameter(torch.randn(*outer_expert_dims, dim, num_gates))

        self.second_policy_train = second_policy_train
        self.second_policy_eval = second_policy_eval
        self.second_threshold_train = second_threshold_train
        self.second_threshold_eval = second_threshold_eval
        self.capacity_factor_train = capacity_factor_train
        self.capacity_factor_eval = capacity_factor_eval

    def forward(self, x, importance = None):
        *_, b, group_size, dim = x.shape
        num_gates = self.num_gates

        if self.training:
            policy = self.second_policy_train
            threshold = self.second_threshold_train
            capacity_factor = self.capacity_factor_train
        else:
            policy = self.second_policy_eval
            threshold = self.second_threshold_eval
            capacity_factor = self.capacity_factor_eval

        raw_gates = torch.einsum('...bnd,...de->...bne', x, self.w_gating)
        raw_gates = raw_gates.softmax(dim=-1)

        # FIND TOP 2 EXPERTS PER POSITON
        # Find the top expert for each position. shape=[batch, group]

        gate_1, index_1 = top1(raw_gates)
        mask_1 = F.one_hot(index_1, num_gates).float()
        density_1_proxy = raw_gates

        if importance is not None:
            equals_one_mask = (importance == 1.).float()
            mask_1 *= equals_one_mask[..., None]
            gate_1 *= equals_one_mask
            density_1_proxy = density_1_proxy * equals_one_mask[..., None]
            del equals_one_mask

        gates_without_top_1 = raw_gates * (1. - mask_1)

        gate_2, index_2 = top1(gates_without_top_1)
        mask_2 = F.one_hot(index_2, num_gates).float()

        if importance is not None:
            greater_zero_mask = (importance > 0.).float()
            mask_2 *= greater_zero_mask[..., None]
            del greater_zero_mask

        # normalize top2 gate scores
        denom = gate_1 + gate_2 + self.eps
        gate_1 /= denom
        gate_2 /= denom

        # BALANCING LOSSES
        # shape = [batch, experts]
        # We want to equalize the fraction of the batch assigned to each expert
        density_1 = mask_1.mean(dim=-2)
        # Something continuous that is correlated with what we want to equalize.
        density_1_proxy = density_1_proxy.mean(dim=-2)
        loss = (density_1_proxy * density_1).mean() * float(num_gates ** 2)

        # Depending on the policy in the hparams, we may drop out some of the
        # second-place experts.
        if policy == "all":
            pass
        elif policy == "none":
            mask_2 = torch.zeros_like(mask_2)
        elif policy == "threshold":
            # print(gate_2.size())
            # print(threshold.size())
            # print((gate_2 > threshold).float().unsqueeze(-1).size())
            mask_2 *= (gate_2 > threshold).float().unsqueeze(-1)
        elif policy == "random":
            probs = torch.zeros_like(gate_2).uniform_(0., 1.)
            # print((probs < (gate_2 / max(threshold, self.eps))).float().unsqueeze(-1).size())
            mask_2 *= (probs < (gate_2 / max(threshold, self.eps))).float().unsqueeze(-1)
        else:
            raise ValueError(f"Unknown policy {policy}")

        # Each sequence sends (at most?) expert_capacity positions to each expert.
        # Static expert_capacity dimension is needed for expert batch sizes
        expert_capacity = min(group_size, int((group_size * capacity_factor) / num_gates))
        expert_capacity = max(expert_capacity, MIN_EXPERT_CAPACITY)
        expert_capacity_f = float(expert_capacity)

        # COMPUTE ASSIGNMENT TO EXPERTS
        # [batch, group, experts]
        # This is the position within the expert's mini-batch for this sequence
        position_in_expert_1 = cumsum_exclusive(mask_1, dim=-2) * mask_1
        # Remove the elements that don't fit. [batch, group, experts]
        mask_1 *= (position_in_expert_1 < expert_capacity_f).float()
        # [batch, experts]
        # How many examples in this sequence go to this expert
        mask_1_count = mask_1.sum(dim=-2, keepdim=True)
        # [batch, group] - mostly ones, but zeros where something didn't fit
        mask_1_flat = mask_1.sum(dim=-1)
        # [batch, group]
        position_in_expert_1 = position_in_expert_1.sum(dim=-1)
        # Weight assigned to first expert.  [batch, group]
        gate_1 *= mask_1_flat

        position_in_expert_2 = cumsum_exclusive(mask_2, dim=-2) + mask_1_count
        position_in_expert_2 *= mask_2
        mask_2 *= (position_in_expert_2 < expert_capacity_f).float()
        mask_2_flat = mask_2.sum(dim=-1)

        position_in_expert_2 = position_in_expert_2.sum(dim=-1)
        gate_2 *= mask_2_flat
        
        # [batch, group, experts, expert_capacity]
        combine_tensor = (
            gate_1[..., None, None]
            * mask_1_flat[..., None, None]
            * F.one_hot(index_1, num_gates)[..., None]
            * safe_one_hot(position_in_expert_1.long(), expert_capacity)[..., None, :] +
            gate_2[..., None, None]
            * mask_2_flat[..., None, None]
            * F.one_hot(index_2, num_gates)[..., None]
            * safe_one_hot(position_in_expert_2.long(), expert_capacity)[..., None, :]
        )

        dispatch_tensor = combine_tensor.bool().to(combine_tensor)
        return dispatch_tensor, combine_tensor, loss

# plain mixture of experts

class MoAE(nn.Module):
    def __init__(self,
        dim,
        num_experts = 16,
        hidden_dim = None,
        activation = nn.ReLU,
        second_policy_train = 'random',
        second_policy_eval = 'random',
        second_threshold_train = 0.2,
        second_threshold_eval = 0.2,
        capacity_factor_train = 1.25,
        capacity_factor_eval = 2.,
        loss_coef = 1e-2,
        experts = None):
        super().__init__()

        self.num_experts = num_experts

        gating_kwargs = {'second_policy_train': second_policy_train, 'second_policy_eval': second_policy_eval, 'second_threshold_train': second_threshold_train, 'second_threshold_eval': second_threshold_eval, 'capacity_factor_train': capacity_factor_train, 'capacity_factor_eval': capacity_factor_eval}
        self.gate = Top2Gating(dim, num_gates = num_experts, **gating_kwargs)
        self.experts = default(experts, lambda: Experts(dim, num_experts = num_experts, hidden_dim = hidden_dim, activation = activation))
        self.loss_coef = loss_coef

    def forward(self, inputs, **kwargs):
        b, n, d, e = *inputs.shape, self.num_experts
        dispatch_tensor, combine_tensor, loss = self.gate(inputs)
        expert_inputs = torch.einsum('bnd,bnec->ebcd', inputs, dispatch_tensor)

        # Now feed the expert inputs through the experts.
        orig_shape = expert_inputs.shape
        expert_inputs = expert_inputs.reshape(e, -1, d)
        expert_outputs = self.experts(expert_inputs)
        expert_outputs = expert_outputs.reshape(*orig_shape)

        output = torch.einsum('ebcd,bnec->bnd', expert_outputs, combine_tensor)
        return output, loss * self.loss_coef




class SVAE(nn.Module):
    def __init__(self,seq_length,sample_length,code_dim,dropout=True,noisy_gating=True, attention_trainmode = 'random',attention_evalmode = 'threshold'):
        super(SVAE,self).__init__()
        self.dropout = 0
        self.noisy_gating = noisy_gating
        if dropout==True:
            self.dropout = 0.5
        self.downsample1 = MoE(seq_length, sample_length, num_experts=15,dropout=self.dropout,noisy_gating=self.noisy_gating)

        # self.attention = MoAE(sample_length,
                              
        #                       num_experts=4,
        #                       second_policy_train=attention_trainmode,
        #                       second_policy_eval=attention_evalmode
                        
        #                       )
        # self.attention = Attention(sample_length)
        # self.attention = nn.Sequential(MLP(sample_length,sample_length))
        self.encode_u=nn.Sequential(
            nn.Linear(sample_length,code_dim),
            nn.Tanh(),
            # nn.Dropout(0.5)  
        )

        self.encode_si = nn.Sequential(
            nn.Linear(sample_length,code_dim),
            nn.Tanh(),
            # nn.Dropout(0.5)
        )
        self.decode1 =  MoE(code_dim, sample_length, num_experts=15,dropout=0.5,noisy_gating=self.noisy_gating)
        # nn.Sequential(
        #     nn.Linear(code_dim,sample_length),
        #     nn.Tanh(),
        #     nn.Dropout(0.5)
        # )
        self.decode2 = nn.Linear(sample_length, seq_length)
        self._init_weight()

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        assert not torch.isnan(std).any() and not torch.isnan(eps).any()
        return eps.mul(std).add_(mu)

    def dimention_reduction(self, x):
        x , loss_0 = self.downsample1(x)
        # x = x.unsqueeze(1)
        # x = self.attention(x)
        # x = x.squeeze(1)
        mu = self.encode_u(x)
        aux_loss = loss_0
        return mu, aux_loss
        
    def forward(self,x):
        # print(x.size())
        x , loss_0 = self.downsample1(x)
        # print(x.size())
        # x = x.unsqueeze(1)
        # x_atten = self.attention(x)
        # x_atten = x_atten.squeeze(1)
        mu = self.encode_u(x)
        var = self.encode_si(x)
        
        z = self._reparameterize(mu,var)
        
        rec1, loss_1 = self.decode1(z)
        # rec_cat = rec1+x_atten
        # rec_cat = torch.cat([rec1,x_atten],dim=1)
        rec = self.decode2(rec1)
        aux_loss = loss_0+loss_1+loss_at
        # x=self.attention(x)
        return rec, mu, var, aux_loss
    
    def _init_weight(self):
        # for m in self.downsample1.modules():
        #     if isinstance(m,nn.Linear):
        #         nn.init.kaiming_normal_(m.weight)
        # for m in self.downsample2.modules():
        #     if isinstance(m,nn.Linear):
        #         nn.init.kaiming_normal_(m.weight)
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.xavier_normal_(m.weight)
#######################################
class SAVAE(nn.Module):
    def __init__(self,seq_length,sample_length,code_dim,dropout=True,noisy_gating=False, attention_trainmode = 'threshold',attention_evalmode = 'threshold'):
        super(SAVAE,self).__init__()
        self.dropout = 0
        self.noisy_gating = noisy_gating
        if dropout==True:
            self.dropout = 0.5
        self.downsample1 = MoE(seq_length, sample_length, num_experts=15,dropout=self.dropout,noisy_gating=self.noisy_gating)

        self.attention = MoAE(sample_length,
                              
                              num_experts=4,
                              second_policy_train=attention_trainmode,
                              second_policy_eval=attention_evalmode
                        
                              )
        # self.attention = Attention(sample_length)
        self.encode_u=nn.Sequential(
            nn.Linear(sample_length,code_dim),
            nn.Tanh(),
            # nn.Dropout(0.5)  
        )

        self.encode_si = nn.Sequential(
            nn.Linear(sample_length,code_dim),
            nn.Tanh(),
            # nn.Dropout(0.5)
        )
        self.decode1 =  MoE(code_dim, sample_length, num_experts=15,dropout=0.5,noisy_gating=self.noisy_gating)
        # nn.Sequential(
        #     nn.Linear(code_dim,sample_length),
        #     nn.Tanh(),
        #     nn.Dropout(0.5)
        # )
        self.decode2 = nn.Linear(sample_length, seq_length)
        self._init_weight()

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        assert not torch.isnan(std).any() and not torch.isnan(eps).any()
        return eps.mul(std).add_(mu)

    def dimention_reduction(self, x):
        x , loss_0 = self.downsample1(x)
        x = x.unsqueeze(1)
        x_atten,loss_at = self.attention(x)
        x = x+x_atten
        x = x.squeeze(1)
        mu = self.encode_u(x)
        aux_loss = loss_0+loss_at
        return mu, aux_loss
        
    def forward(self,x):
        # print(x.size())
        x , loss_0 = self.downsample1(x)
        # print(x.size())
        x = x.unsqueeze(1)
        x_atten,loss_at = self.attention(x)
        x_atten = x_atten+x
        x_atten = x_atten.squeeze(1)
        mu = self.encode_u(x_atten)
        var = self.encode_si(x_atten)
        
        z = self._reparameterize(mu,var)
        
        rec1, loss_1 = self.decode1(z)
        # rec_cat = rec1+x_atten
        # rec_cat = torch.cat([rec1,x_atten],dim=1)
        rec = self.decode2(rec1)
        aux_loss = loss_0+loss_1+loss_at
        # x=self.attention(x)
        return rec, mu, var, aux_loss
    
    def _init_weight(self):
        # for m in self.downsample1.modules():
        #     if isinstance(m,nn.Linear):
        #         nn.init.kaiming_normal_(m.weight)
        # for m in self.downsample2.modules():
        #     if isinstance(m,nn.Linear):
        #         nn.init.kaiming_normal_(m.weight)
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.xavier_normal_(m.weight)
                

class SAVAE2(nn.Module):
    def __init__(self,seq_length,sample_length,code_dim,dropout=True,noisy_gating=True, attention_trainmode = 'random',attention_evalmode = 'threshold'):
        super(SAVAE2,self).__init__()
        self.dropout = 0
        self.noisy_gating = noisy_gating
        if dropout==True:
            self.dropout = 0.5
        self.downsample1 = MoE(seq_length, sample_length, num_experts=15,dropout=self.dropout,noisy_gating=self.noisy_gating)

        # self.attention = MoAE(sample_length,
                              
        #                       num_experts=4,
        #                       second_policy_train=attention_trainmode,
        #                       second_policy_eval=attention_evalmode
                        
        #                       )
        self.attention = Attention(sample_length)
        self.encode_u=nn.Sequential(
            nn.Linear(sample_length,code_dim),
            nn.Tanh(),
            # nn.Dropout(0.5)  
        )

        self.encode_si = nn.Sequential(
            nn.Linear(sample_length,code_dim),
            nn.Tanh(),
            # nn.Dropout(0.5)
        )
        self.decode1 =  MoE(code_dim, sample_length, num_experts=15,dropout=0.5,noisy_gating=self.noisy_gating)
        # nn.Sequential(
        #     nn.Linear(code_dim,sample_length),
        #     nn.Tanh(),
        #     nn.Dropout(0.5)
        # )
        self.decode2 = nn.Linear(sample_length, seq_length)
        self._init_weight()

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        assert not torch.isnan(std).any() and not torch.isnan(eps).any()
        return eps.mul(std).add_(mu)

    def dimention_reduction(self, x):
        x , loss_0 = self.downsample1(x)
        # x = x.unsqueeze(1)
        x = self.attention(x)
        # x = x.squeeze(1)
        mu = self.encode_u(x)
        aux_loss = loss_0
        return mu, aux_loss
        
    def forward(self,x):
        # print(x.size())
        x , loss_0 = self.downsample1(x)
        # print(x.size())
        # x = x.unsqueeze(1)
        x_atten = self.attention(x)
        # x_atten = x_atten.squeeze(1)
        mu = self.encode_u(x_atten)
        var = self.encode_si(x_atten)
        
        z = self._reparameterize(mu,var)
        
        rec1, loss_1 = self.decode1(z)
        # rec_cat = rec1+x_atten
        # rec_cat = torch.cat([rec1,x_atten],dim=1)
        rec = self.decode2(rec1)
        aux_loss = loss_0+loss_1
        # x=self.attention(x)
        return rec, mu, var, aux_loss
    
    def _init_weight(self):
        # for m in self.downsample1.modules():
        #     if isinstance(m,nn.Linear):
        #         nn.init.kaiming_normal_(m.weight)
        # for m in self.downsample2.modules():
        #     if isinstance(m,nn.Linear):
        #         nn.init.kaiming_normal_(m.weight)
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.xavier_normal_(m.weight)


class Coxnnet(nn.Module):
    def __init__(self, nfeat):
        super(Coxnnet, self).__init__()
        self.fc1 = nn.Linear(nfeat, int(np.ceil(nfeat ** 0.5)))
        self.fc2 = nn.Linear(int(np.ceil(nfeat ** 0.5)), 1)
        self.init_hidden()

    def forward(self, x, coo=None):
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

    def init_hidden(self):
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)



class CoxClassifierSRNAseq(nn.Module):
    def __init__(self,seq_length,sample_length,code_dim,encoder_type='basic',freeze=False):
        super(CoxClassifierSRNAseq,self).__init__()
        self.freeze=freeze
        self.encoder_type = encoder_type
        if encoder_type == 'basic':
            self.encoder=VAE(seq_length,sample_length,code_dim)
            self.encoder.load_state_dict(torch.load('saved_models/NVAEpretrain/g_a_100'))
        elif encoder_type =='attention':
            self.encoder=SAVAE(seq_length,sample_length,code_dim,noisy_gating=False)
            self.encoder.load_state_dict(torch.load('saved_models/SVAEpretrain/g_a_200'))

        self.cox=nn.Sequential(nn.Linear(code_dim,1))

        
        if self.freeze==True:
            self.set_freeze_by_names(self,'ecoder',freeze=True)
    
    def forward(self,x_rna):
        aux_loss = None
        if self.encoder_type == 'attention':
            rna_code, aux_loss=self.encoder.dimention_reduction(x_rna)
        # mirna_code=self.encoder_mirna(x_mirna)
        else:
            rna_code=self.encoder.dimention_reduction(x_rna)
        return self.cox(rna_code),rna_code, aux_loss
    
    def _init_weight(self):
        if self.transfer==True:
            for m in self.classifier.modules():
                if isinstance(m,nn.Linear):
                    nn.init.xavier_normal_(m.weight)
        else:
            for m in self.modules():
                if isinstance(m,nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    

class CoxClassifierSRNAseq2(nn.Module):
    def __init__(self,seq_length,sample_length,code_dim,encoder_type='basic',freeze=False):
        super(CoxClassifierSRNAseq2,self).__init__()
        self.freeze=freeze
        self.encoder_type = encoder_type
        if encoder_type == 'basic':
            self.encoder=VAE(seq_length,sample_length,code_dim)
            # self.encoder.load_state_dict(torch.load('saved_models/NVAEpretrain/g_a_100'))
        elif encoder_type =='attention':
            self.encoder=SAVAE(seq_length,sample_length,code_dim,noisy_gating=False)
            # self.encoder.load_state_dict(torch.load('saved_models/SVAEpretrain/g_a_200'))

        self.cox=nn.Sequential(nn.Linear(code_dim,1))

        
        if self.freeze==True:
            self.set_freeze_by_names(self,'ecoder',freeze=True)
    
    def forward(self,x_rna):
        aux_loss = None
        if self.encoder_type == 'attention':
            rna_code, aux_loss=self.encoder.dimention_reduction(x_rna)
        # mirna_code=self.encoder_mirna(x_mirna)
        else:
            rna_code=self.encoder.dimention_reduction(x_rna)
        return self.cox(rna_code),rna_code, aux_loss
    
    def _init_weight(self):
        if self.transfer==True:
            for m in self.classifier.modules():
                if isinstance(m,nn.Linear):
                    nn.init.xavier_normal_(m.weight)
        else:
            for m in self.modules():
                if isinstance(m,nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    

class CoxClassifierSRNAseq3(nn.Module):
    def __init__(self,seq_length,sample_length,code_dim,encoder_type='basic',freeze=False):
        super(CoxClassifierSRNAseq3,self).__init__()
        self.freeze=freeze
        self.encoder_type = encoder_type
        if encoder_type == 'basic':
            self.encoder=VAE(seq_length,sample_length,code_dim)
            # self.encoder.load_state_dict(torch.load('saved_models/NVAEpretrain/g_a_100'))
        elif encoder_type =='attention':
            self.encoder=SVAE(seq_length,sample_length,code_dim,noisy_gating=False)
            # self.encoder.load_state_dict(torch.load('saved_models/SVAEpretrain/g_a_200'))

        self.cox=nn.Sequential(nn.Linear(code_dim,1))

        
        if self.freeze==True:
            self.set_freeze_by_names(self,'ecoder',freeze=True)
    
    def forward(self,x_rna):
        aux_loss = None
        if self.encoder_type == 'attention':
            rna_code, aux_loss=self.encoder.dimention_reduction(x_rna)
        # mirna_code=self.encoder_mirna(x_mirna)
        else:
            rna_code=self.encoder.dimention_reduction(x_rna)
        return self.cox(rna_code),rna_code, aux_loss
    
    def _init_weight(self):
        if self.transfer==True:
            for m in self.classifier.modules():
                if isinstance(m,nn.Linear):
                    nn.init.xavier_normal_(m.weight)
        else:
            for m in self.modules():
                if isinstance(m,nn.Linear):
                    nn.init.xavier_normal_(m.weight)

class CoxClassifierSRNAseq4(nn.Module):
    def __init__(self,seq_length,sample_length,code_dim,encoder_type='basic',freeze=False):
        super(CoxClassifierSRNAseq3,self).__init__()
        self.freeze=freeze
        self.encoder_type = encoder_type
        if encoder_type == 'basic':
            self.encoder=VAE(seq_length,sample_length,code_dim)
            # self.encoder.load_state_dict(torch.load('saved_models/NVAEpretrain/g_a_100'))
        elif encoder_type =='attention':
            self.encoder=SAVAE2(seq_length,sample_length,code_dim,noisy_gating=False)
            # self.encoder.load_state_dict(torch.load('saved_models/SVAEpretrain/g_a_200'))

        self.cox=nn.Sequential(nn.Linear(code_dim,1))

        
        if self.freeze==True:
            self.set_freeze_by_names(self,'ecoder',freeze=True)
    
    def forward(self,x_rna):
        aux_loss = None
        if self.encoder_type == 'attention':
            rna_code, aux_loss=self.encoder.dimention_reduction(x_rna)
        # mirna_code=self.encoder_mirna(x_mirna)
        else:
            rna_code=self.encoder.dimention_reduction(x_rna)
        return self.cox(rna_code),rna_code, aux_loss
    
    def _init_weight(self):
        if self.transfer==True:
            for m in self.classifier.modules():
                if isinstance(m,nn.Linear):
                    nn.init.xavier_normal_(m.weight)
        else:
            for m in self.modules():
                if isinstance(m,nn.Linear):
                    nn.init.xavier_normal_(m.weight)



class Classifier(nn.Module):
    def __init__(self,seq_length,sample_length,code_dim,encoder_type='basic',freeze=False):
        super(Classifier,self).__init__()
        self.freeze=freeze
        self.encoder_type = encoder_type
        if encoder_type == 'basic':
            self.encoder=VAE(seq_length,sample_length,code_dim)
            self.encoder.load_state_dict(torch.load('saved_models/NVAEpretrain/g_a_100'))
        elif encoder_type =='attention':
            self.encoder=SAVAE(seq_length,sample_length,code_dim,noisy_gating=False,attention_trainmode='threshold')
            self.encoder.load_state_dict(torch.load('saved_models/SVAEpretrain/g_a_200'))

        # self.cox=nn.Sequential(nn.Linear(code_dim,1))
        self.classifier = nn.Sequential(nn.Linear(code_dim, 33))

        
        if self.freeze==True:
            self.set_freeze_by_names(self,'ecoder',freeze=True)
    
    def forward(self,x_rna):
        aux_loss = None
        if self.encoder_type == 'attention':
            rna_code, aux_loss=self.encoder.dimention_reduction(x_rna)
        # mirna_code=self.encoder_mirna(x_mirna)
        else:
            rna_code=self.encoder.dimention_reduction(x_rna)
        return self.classifier(rna_code),rna_code, aux_loss
    
    def _init_weight(self):
        if self.transfer==True:
            for m in self.classifier.modules():
                if isinstance(m,nn.Linear):
                    nn.init.xavier_normal_(m.weight)
        else:
            for m in self.modules():
                if isinstance(m,nn.Linear):
                    nn.init.xavier_normal_(m.weight)



def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def accuracy_cox(hazards, labels):
    # This accuracy is based on estimated survival events against true survival events
    hazardsdata = hazards.cpu().numpy().reshape(-1)
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    labels = labels.data.cpu().numpy()
    correct = np.sum(hazards_dichotomize == labels)
    return correct / len(labels)

def cox_log_rank(hazards, labels, survtime_all):
    hazardsdata = hazards.cpu().numpy().reshape(-1)
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    survtime_all = survtime_all.data.cpu().numpy().reshape(-1)
    idx = hazards_dichotomize == 0
    labels = labels.data.cpu().numpy()
    T1 = survtime_all[idx]
    T2 = survtime_all[~idx]
    E1 = labels[idx]
    E2 = labels[~idx]
    results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
    pvalue_pred = results.p_value
    return(pvalue_pred)
    
def CIndex(hazards, labels, survtime_all):
    labels = labels.data.cpu().numpy()
    concord = 0.
    total = 0.
    N_test = labels.shape[0]
    labels = np.asarray(labels, dtype=bool)
    for i in range(N_test):
        if labels[i] == 1:
            for j in range(N_test):
                if survtime_all[j] > survtime_all[i]:
                    total = total + 1
                    if hazards[j] < hazards[i]: concord = concord + 1
                    elif hazards[j] < hazards[i]: concord = concord + 0.5

    return(concord/total)
    
def CIndex_lifeline(hazards, labels, survtime_all):
    labels = labels.data.cpu().numpy()
    hazards = hazards.cpu().numpy().reshape(-1)
    return(concordance_index(survtime_all, -hazards, labels))
        
def frobenius_norm_loss(a, b):
    loss = torch.sqrt(torch.sum(torch.abs(a-b)**2))
    return loss

class BaselineC(nn.Module):
    def __init__(self,seq_length,sample_length,code_dim,encoder_type='basic',freeze=False):
        super(BaselineC,self).__init__()
        self.freeze=freeze
        self.encoder_type = encoder_type
        self.encoder = nn.Sequential(MLP(seq_length, sample_length),MLP(sample_length,code_dim))

        # self.cox=nn.Sequential(nn.Linear(code_dim,1))
        self.classifier = nn.Sequential(nn.Linear(code_dim, 33))

        
        if self.freeze==True:
            self.set_freeze_by_names(self,'ecoder',freeze=True)
    
    def forward(self,x_rna):
        aux_loss = None
        rna_code = self.encoder(x_rna)
        # if self.encoder_type == 'attention':
        #     rna_code, aux_loss=self.encoder.dimention_reduction(x_rna)
        # # mirna_code=self.encoder_mirna(x_mirna)
        # else:
        #     rna_code=self.encoder.dimention_reduction(x_rna)
        return self.classifier(rna_code),rna_code, aux_loss
    
    def _init_weight(self):
        if self.transfer==True:
            for m in self.classifier.modules():
                if isinstance(m,nn.Linear):
                    nn.init.xavier_normal_(m.weight)
        else:
            for m in self.modules():
                if isinstance(m,nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    

class ClassifierA(nn.Module):
    def __init__(self,seq_length,sample_length,code_dim,encoder_type='basic',freeze=False):
        super(ClassifierA,self).__init__()
        self.freeze=freeze
        self.encoder_type = encoder_type
        if encoder_type == 'basic':
            self.encoder=VAE(seq_length,sample_length,code_dim)
            # self.encoder.load_state_dict(torch.load('saved_models/NVAEpretrain/g_a_100'))
        elif encoder_type =='attention':
            self.encoder=SAVAE(seq_length,sample_length,code_dim,noisy_gating=False,attention_trainmode='threshold')
            # self.encoder.load_state_dict(torch.load('saved_models/SVAEpretrain/g_a_200'))

        # self.cox=nn.Sequential(nn.Linear(code_dim,1))
        self.classifier = nn.Sequential(nn.Linear(code_dim, 33))

        
        if self.freeze==True:
            self.set_freeze_by_names(self,'ecoder',freeze=True)
    
    def forward(self,x_rna):
        aux_loss = None
        if self.encoder_type == 'attention':
            rna_code, aux_loss=self.encoder.dimention_reduction(x_rna)
        # mirna_code=self.encoder_mirna(x_mirna)
        else:
            rna_code=self.encoder.dimention_reduction(x_rna)
        return self.classifier(rna_code),rna_code, aux_loss
    
    def _init_weight(self):
        if self.transfer==True:
            for m in self.classifier.modules():
                if isinstance(m,nn.Linear):
                    nn.init.xavier_normal_(m.weight)
        else:
            for m in self.modules():
                if isinstance(m,nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    
                    


class ClassifierB(nn.Module):
    def __init__(self,seq_length,sample_length,code_dim,encoder_type='basic',freeze=False):
        super(ClassifierB,self).__init__()
        self.freeze=freeze
        self.encoder_type = encoder_type
        if encoder_type == 'basic':
            self.encoder=VAE(seq_length,sample_length,code_dim)
            # self.encoder.load_state_dict(torch.load('saved_models/NVAEpretrain/g_a_100'))
        elif encoder_type =='attention':
            self.encoder=SVAE(seq_length,sample_length,code_dim,noisy_gating=False,attention_trainmode='threshold')
            # self.encoder.load_state_dict(torch.load('saved_models/SVAEpretrain/g_a_200'))

        # self.cox=nn.Sequential(nn.Linear(code_dim,1))
        self.classifier = nn.Sequential(nn.Linear(code_dim, 33))

        
        if self.freeze==True:
            self.set_freeze_by_names(self,'ecoder',freeze=True)
    
    def forward(self,x_rna):
        aux_loss = None
        if self.encoder_type == 'attention':
            rna_code, aux_loss=self.encoder.dimention_reduction(x_rna)
        # mirna_code=self.encoder_mirna(x_mirna)
        else:
            rna_code=self.encoder.dimention_reduction(x_rna)
        return self.classifier(rna_code),rna_code, aux_loss
    
    def _init_weight(self):
        if self.transfer==True:
            for m in self.classifier.modules():
                if isinstance(m,nn.Linear):
                    nn.init.xavier_normal_(m.weight)
        else:
            for m in self.modules():
                if isinstance(m,nn.Linear):
                    nn.init.xavier_normal_(m.weight)



class ClassifierC(nn.Module):
    def __init__(self,seq_length,sample_length,code_dim,encoder_type='basic',freeze=False):
        super(ClassifierC,self).__init__()
        self.freeze=freeze
        self.encoder_type = encoder_type
        if encoder_type == 'basic':
            self.encoder=VAE(seq_length,sample_length,code_dim)
            # self.encoder.load_state_dict(torch.load('saved_models/NVAEpretrain/g_a_100'))
        elif encoder_type =='attention':
            self.encoder=SAVAE2(seq_length,sample_length,code_dim,noisy_gating=False,attention_trainmode='threshold')
            # self.encoder.load_state_dict(torch.load('saved_models/SVAEpretrain/g_a_200'))

        # self.cox=nn.Sequential(nn.Linear(code_dim,1))
        self.classifier = nn.Sequential(nn.Linear(code_dim, 33))

        
        if self.freeze==True:
            self.set_freeze_by_names(self,'ecoder',freeze=True)
    
    def forward(self,x_rna):
        aux_loss = None
        if self.encoder_type == 'attention':
            rna_code, aux_loss=self.encoder.dimention_reduction(x_rna)
        # mirna_code=self.encoder_mirna(x_mirna)
        else:
            rna_code=self.encoder.dimention_reduction(x_rna)
        return self.classifier(rna_code),rna_code, aux_loss
    
    def _init_weight(self):
        if self.transfer==True:
            for m in self.classifier.modules():
                if isinstance(m,nn.Linear):
                    nn.init.xavier_normal_(m.weight)
        else:
            for m in self.modules():
                if isinstance(m,nn.Linear):
                    nn.init.xavier_normal_(m.weight)



class DNNclassifier(nn.Module):
    def __init__(self, input_feature):
        super(DNNclassifier, self).__init__()
        self.fc1 = nn.Linear(input_feature, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 200)
        self.fc4 = nn.Linear(200, 300)
        self.fc5 = nn.Linear(300, 200)
        self.fc6 = nn.Linear(200, 100)
        self.fc7 = nn.Linear(100, 33)
        self.init_hidden()

    def forward(self, x, coo=None):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        x = self.fc7(x)
        return x

    def init_hidden(self):
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        

class VanilaConv(nn.Module):
    def __init__(self, input_feature):
        super(VanilaConv, self).__init__()
        self.conv1 = nn.Conv2d(1,32,(140,1),stride=1)
        self.conv2 = nn.Conv2d(1,32,(1,200),stride=1)
        self.pool1 = nn.MaxPool2d((1,2))
        self.pool2 = nn.MaxPool2d((2,1))
        self.fc1 = nn.Linear(16*(140+200),128)
        self.fc2 = nn.Linear(128,33)
        # self.init_hidden()

    def forward(self, x, coo=None):
        y = torch.relu(self.conv1(x))
        y = self.pool1(y)
        y = torch.flatten(y,1)
        z = torch.relu(self.conv2(x))
        z = self.pool2(z)
        z = torch.flatten(z,1)
        out_feat = torch.cat((y,z),dim=1)
        x = torch.relu(self.fc1(out_feat))
        x = self.fc2(x)
        # print(x.size())
        # print(z.size())
        return x
    
# a = torch.randn((1,1,400,700))
# model = VanilaConv(3000)
# c = model(a)
# print(c.size())