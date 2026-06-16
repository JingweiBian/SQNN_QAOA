# -*- coding: utf-8 -*-

'''
这里是，在QuantumNeuronLayer这个单层类的基础之上，对于多层感知器情况下的整个模型的构建、
写成一个类class
最后，我需要他也得兼容单层模式
'''
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
import functools

from ..core.layers import (
    InputEncodedLayerAngleEncoding,
    InputEncodedLayerParallelEncoding,
    QuantumNeuronLayer,
)

class SoftQuantumNeuralNetwork1(nn.Module):
    def __init__(self, layer_dims, layers_size, noise_config, encoding_config):
        super(SoftQuantumNeuralNetwork, self).__init__()
        '''
        layer_dims里面包含的是每一层的神经元数量信息，代表着该层的输出维度和下一层的输入维度
        layer_size代表着总的层数
        '''
        if len(layer_dims)!=layers_size:
            print('config中神经元维度与层数不匹配')
        
        self.layer_dims = layer_dims
        self.layers_size = layers_size
        
        self.qubits_number=sum(layer_dims)
        
        # noise_config could be None or e.g. ['bit_flip', 0.1].
        # Treat [None, *] as no noise to avoid default.mixed (which requires huge memory for many qubits).
        self.noise_config = noise_config if noise_config and noise_config[0] is not None else None

        self.encoding_type = encoding_config[0]
        self.encoding_times = encoding_config[1]

        if self.noise_config:
            # Mixed-state simulator required for noise models (expensive for >20 qubits)
            self.dev = qml.device("default.mixed", wires=self.qubits_number, shots=None)
        else:
            # Use statevector simulator for efficiency when noise isn't applied.
            self.dev = qml.device("lightning.qubit", wires=self.qubits_number, shots=None,batch_obs=True)

        # 从第二层开始创建可训练参数（索引 i = 1, 2, ...）
        #把第一层挤占掉，所以索引0啥都没有，从索引1开始对应第1层
        self.layer_weights = nn.ParameterList([None])
        self.layer_biases = nn.ParameterList([None])

        for i in range(1, len(layer_dims)):
            in_dim = layer_dims[i-1]   # 上一层量子比特数
            out_dim = layer_dims[i]     # 当前层量子比特数
            w = nn.Parameter(torch.randn(out_dim, in_dim, 3) * np.pi)
            b = nn.Parameter(torch.randn(out_dim, 3) * 0.1 * np.pi)
            self.layer_weights.append(w)
            self.layer_biases.append(b)
        

        # 计算每层对应的量子比特线索引（假设连续编号）
        #这个wire是从编码层开始算的
        #但是权重的索引是从，隐藏层开始的
        self.layer_wires = []
        start = 0
        for dim in layer_dims:
            wires = list(range(start, start + dim))
            self.layer_wires.append(wires)
            start += dim

    def _apply_noise(self, noise_type, probability):#这里面第一个self代表函数的对象，而非输入参数
        """在量子状态下应用噪声"""
        if noise_type == 'bit_flip':
                # 比特翻转噪声
            for j in range(self.qubits_number):
                qml.BitFlip(probability, wires=j)
                   
        elif noise_type == 'phase_flip':
                # 相位翻转噪声
            for j in range(self.qubits_number):
                qml.PhaseFlip(probability, wires=j)        

    def _encoding_layer(self,sample_input):
        if self.encoding_type == 'angle':
            encoded_number = self.layer_dims[0]
            for i in range(encoded_number):
                qml.RY(sample_input[i] * np.pi, wires=i)

            if self.noise_config:
                noise_type, noise_strength = self.noise_config
                self._apply_noise(noise_type, noise_strength)

        elif self.encoding_type == 'parallel':
            """处理单个样本的量子电路"""
            # 特征编码：每个特征重复编码到不同量子比特
            input_dim = len(sample_input)
            for i in range(input_dim):
                x = sample_input[i] * np.pi
                for time in range(self.encoding_times):
                    wire = i * self.encoding_times + time
                    qml.RY(x, wires=wire)

            # 应用噪声模型
            if self.noise_config:
                noise_type, noise_strength = self.noise_config
                self._apply_noise(noise_type, noise_strength)

    def _quantum_forward(self):
        #inputs_batch是（batchsize,input_dim）的
        @qml.qnode(self.dev, interface="torch")
        def single_circuit(sample_input):
            self._encoding_layer(sample_input)
            for layer_index in range(1, self.layers_size):
                w=self.layer_weights[layer_index] 
                b=self.layer_biases[layer_index]
                in_wires=self.layer_wires[layer_index-1] #上一层的量子比特
                out_wires=self.layer_wires[layer_index] #当前层的量子比特

                input_dim=len(in_wires)
                output_dim=len(out_wires)

                for j in range(output_dim):
                    for i in range(input_dim):
                        qml.ctrl(qml.Rot,control=in_wires[i])(*w[j,i], wires=out_wires[j])
                        qml.Rot(*b[j],wires=out_wires[j])

            sample_wires=self.layer_wires[-1] #最后一层的量子比特
            return [qml.expval(qml.PauliZ(wires=sample_wires[j])) for j in range(len(sample_wires))]

        def parallel_circuit(batch_input):
            batch_results = []
            for sample in batch_input:
                sample_result = single_circuit(sample)
                batch_results.append(torch.stack(sample_result))
            return torch.stack(batch_results)
        return parallel_circuit  
    
    def forward(self, inputs_batch):
        model=self._quantum_forward()
        outputs_batch=model(inputs_batch)
        return outputs_batch

class SoftQuantumNeuralNetwork2(nn.Module):
    def __init__(self, layer_dims, layers_size, noise_config, encoding_config):
        super(SoftQuantumNeuralNetwork, self).__init__()
        '''
        layer_dims里面包含的是每一层的神经元数量信息，代表着该层的输出维度和下一层的输入维度
        layer_size代表着总的层数
        '''
        
        if len(layer_dims)!=layers_size:
            print('config中神经元维度与层数不匹配')
        
        self.layer_dims = layer_dims
        self.layers_size = layers_size
        
        self.qubits_number=sum(layer_dims)
        
        self.noise_config = noise_config

        self.encoding_type = encoding_config[0]
        self.encoding_times = encoding_config[1]

        if noise_config:
            self.dev = qml.device("default.mixed", wires=self.qubits_number, shots=None)
        else:
            self.dev = qml.device("lightning.qubit", wires=self.qubits_number, shots=None,batch_obs=True)

        # 从第二层开始创建可训练参数（索引 i = 1, 2, ...）
        #把第一层挤占掉，所以索引0啥都没有，从索引1开始对应第1层
        self.layer_weights = nn.ParameterList([None])
        self.layer_biases = nn.ParameterList([None])

        for i in range(1, len(layer_dims)):
            in_dim = layer_dims[i-1]   # 上一层量子比特数
            out_dim = layer_dims[i]     # 当前层量子比特数
            w = nn.Parameter(torch.randn(out_dim, in_dim, 3) * np.pi)
            b = nn.Parameter(torch.randn(out_dim, 3) * 0.1 * np.pi)
            self.layer_weights.append(w)
            self.layer_biases.append(b)
        

        # 计算每层对应的量子比特线索引（假设连续编号）
        #这个wire是从编码层开始算的
        #但是权重的索引是从，隐藏层开始的
        self.layer_wires = []
        start = 0
        for dim in layer_dims:
            wires = list(range(start, start + dim))
            self.layer_wires.append(wires)
            start += dim

    def _apply_noise(self, noise_type, probability):#这里面第一个self代表函数的对象，而非输入参数
        """在量子状态下应用噪声"""
        if noise_type == 'bit_flip':
                # 比特翻转噪声
            for j in range(self.qubits_number):
                qml.BitFlip(probability, wires=j)
                   
        elif noise_type == 'phase_flip':
                # 相位翻转噪声
            for j in range(self.qubits_number):
                qml.PhaseFlip(probability, wires=j)        

    def _encoding_layer(self,sample_input):
        if self.encoding_type == 'angle':

            encoded_number = self.layer_dims[0]
            for i in range(encoded_number):
                qml.RY(sample_input[i] * np.pi, wires=i)

            if self.noise_config:
                noise_type, noise_strength = self.noise_config
                self._apply_noise(noise_type, noise_strength)
                
            return [qml.sample(qml.PauliZ(wires=j)) for j in range(encoded_number)]

        elif self.encoding_type == 'parallel':
            """处理单个样本的量子电路"""
            # 特征编码：每个特征重复编码到不同量子比特
            input_dim = len(sample_input)
            for i in range(input_dim):
                x = sample_input[i] * np.pi
                for time in range(self.encoding_times):
                    wire = i * self.encoding_times + time
                    qml.RY(x, wires=wire)

            # 应用噪声模型
            if self.noise_config:
                noise_type, noise_strength = self.noise_config
                self._apply_noise(noise_type, noise_strength)

            # 返回所有量子比特的测量结果
            return [qml.sample(qml.PauliZ(wires=j)) for j in range(encoded_number)]
  
    def _hidden_layer(self,sample_input,layer_index):
        w=self.layer_weights[layer_index] 
        b=self.layer_biases[layer_index]
        in_wires=self.layer_wires[layer_index-1] #上一层的量子比特
        out_wires=self.layer_wires[layer_index] #当前层的量子比特

        input_dim=len(in_wires)
        output_dim=len(out_wires)

        for j in range(output_dim):
            for i in range(input_dim): 
                rot=sample_input[i]*w[j,i]
                qml.Rot(*rot, wires=out_wires[j])
                # 下面是原本的伪代码逻辑，仅供参考：
                # if layer_input[i]==0:
                #     continue
                # elif layer_input[i]==1:
                #     qml.Rot(*self.weight[j,i], wires=j)
        #这里给这一层的每一个神经元加偏置bias     
        for j in range(output_dim):
            qml.Rot(*b[j],wires=out_wires[j])
                    
        # 应用噪声
        if self.noise_config:
            (noise_type, noise_strength) = self.noise_config
            self._apply_noise(noise_type, noise_strength)
                
        return [qml.sample(qml.PauliZ(wires=out_wires[j])) for j in range(output_dim)]

    def _quantum_forward(self):
        #inputs_batch是（batchsize,input_dim）的
        @qml.qnode(self.dev, interface="torch")
        def single_circuit(sample_input):
            encoded_input=self._encoding_layer(sample_input)
            layer_input=encoded_input
            for idx in range(1, self.layers_size):
                layer_output=self._hidden_layer(layer_input,idx)
            # range(start, stop) 生成的序列 不包含 stop 这个值本身
            return layer_output
        def parallel_circuit(batch_input):
            batch_results = []
            for sample in batch_input:
                sample_result = single_circuit(sample)
                batch_results.append(torch.stack(sample_result))
            return torch.stack(batch_results)
        return parallel_circuit  
    
    def forward(self, inputs_batch):
        model=self._quantum_forward()
        outputs_batch=model(inputs_batch)
        return outputs_batch

class SoftQuantumNeuralNetwork(nn.Module):
    def __init__(self, layer_dims, layers_size, noise_config, encoding_config):
        super(SoftQuantumNeuralNetwork, self).__init__()
        '''
        layer_dims里面包含的是每一层的神经元数量信息，代表着该层的输出维度和下一层的输入维度
        layer_size代表着总的层数
        '''
        
        if len(layer_dims)!=layers_size:
            print('config中神经元维度与层数不匹配')
        
        self.layer_dims = layer_dims
        self.layers_size = layers_size
    
        
        self.noise_config = noise_config if noise_config and noise_config[0] is not None else None

        self.encoding_type = encoding_config[0]

        # 创建量子神经元层
        self.layers = nn.ModuleList()

        if self.encoding_type == 'angle':

            layer = InputEncodedLayerAngleEncoding(layer_dims[0], noise_config)
            print(f"创建[输入层](角度编码): 特征维度={layer_dims[0]}, 噪声配置={noise_config}")
            self.layers.append(layer)

        elif self.encoding_type == 'parallel':

            repeat_times = encoding_config[1]
            input_dim = layer_dims[0] // repeat_times

            layer = InputEncodedLayerParallelEncoding(
                repeat_times, input_dim, noise_config)
            self.layers.append(layer)

        for size in range(layers_size-1):
            # range(start, stop) 生成的序列 不包含 stop 这个值本身
            input_dim = layer_dims[size]
            output_dim = layer_dims[size + 1]
            layer = QuantumNeuronLayer(
                input_dim, 
                output_dim,
                noise_config
            )
            print(f'创建量子层 #{size+1}: [{input_dim}]→[{output_dim}]')
            self.layers.append(layer)
            
            # 添加验证
        if len(self.layers) != layers_size:
            print(f"⚠️ 警告! 创建的层数({len(self.layers)})与层大小参数({layers_size})不匹配")
            
            
    def _quantum_forward(self,inputs_batch):
        #inputs_batch是（batchsize,input_dim）的
        current_input=inputs_batch
        for i in range(self.layers_size):
            outputs_batch=self.layers[i](current_input)
            current_input=outputs_batch
            
        return outputs_batch
    
    def forward(self, inputs_batch):
        outputs_batch=self._quantum_forward(inputs_batch)
        return outputs_batch


class DataReuploadingSoftQuantumNeuralNetwork(nn.Module):
    """
    Data reuploading variant that keeps the original model untouched.

    In data reuploading, the encoded input is injected again at every trainable
    layer. Here the reuploaded data is the encoding-layer measurement result.

    Forward logic:
        encoded = encoding_layer(x)
        current = encoded
        for each later layer:
            layer_input = concat(encoded, current)
            current = layer(layer_input)

    Therefore layer i receives both the fixed encoding information and the
    previous layer output. With layer_dims=[16, 4, 4, 2], the trainable layer
    input sizes are [32, 20, 20].
    """

    def __init__(self, layer_dims, layers_size, noise_config, encoding_config):
        super().__init__()

        if len(layer_dims) != layers_size:
            print('config中神经元维度与层数不匹配')

        self.layer_dims = layer_dims
        self.layers_size = layers_size
        self.noise_config = noise_config if noise_config and noise_config[0] is not None else None
        self.encoding_type = encoding_config[0]
        self.layers = nn.ModuleList()

        if self.encoding_type == 'angle':
            layer = InputEncodedLayerAngleEncoding(layer_dims[0], noise_config)
            print(
                f"创建[输入层](角度编码): 特征维度={layer_dims[0]}, "
                f"噪声配置={noise_config}"
            )
            self.layers.append(layer)
        elif self.encoding_type == 'parallel':
            repeat_times = encoding_config[1]
            input_dim = layer_dims[0] // repeat_times
            layer = InputEncodedLayerParallelEncoding(
                repeat_times,
                input_dim,
                noise_config
            )
            self.layers.append(layer)
        else:
            raise ValueError(f"Unsupported encoding type: {self.encoding_type}")

        encoded_dim = layer_dims[0]
        for size in range(layers_size - 1):
            previous_dim = layer_dims[size]
            input_dim = encoded_dim + previous_dim
            output_dim = layer_dims[size + 1]
            layer = QuantumNeuronLayer(
                input_dim,
                output_dim,
                noise_config
            )
            print(
                f'创建data reuploading量子层 #{size + 1}: '
                f'[encoding {encoded_dim} + previous {previous_dim}] -> [{output_dim}]'
            )
            self.layers.append(layer)

        if len(self.layers) != layers_size:
            print(
                f"Warning: created {len(self.layers)} layers, "
                f"expected {layers_size}"
            )

    def _quantum_forward(self, inputs_batch):
        encoded_input = self.layers[0](inputs_batch)
        current_input = encoded_input

        for layer_index in range(1, self.layers_size):
            layer_input = torch.cat([encoded_input, current_input], dim=1)
            current_input = self.layers[layer_index](layer_input)

        return current_input

    def forward(self, inputs_batch):
        return self._quantum_forward(inputs_batch)


EncodingSkipSoftQuantumNeuralNetwork = DataReuploadingSoftQuantumNeuralNetwork


class QuantumLoss(nn.Module):
    def __init__(self):
        #mean squared error (MSE) loss
       super().__init__()
       
       self.loss_history=[]
       self.record_history = False
          
    def forward(self, predict_batch, expect_batch):
        '''
        第一个参数是模型跑出来的output
        第二个参数是数据集自带的标签
        '''
        if expect_batch.shape[0] == predict_batch.shape[0]:
            batch_size = predict_batch.shape[0]
        else:
            print('The batch size of the predicted and expected are different')
        if expect_batch.shape[1] == predict_batch.shape[1]:
            self.label_dim = predict_batch.shape[1]
        else:
            print('The label dimension of the predicted and expected are different')
           
        difference = predict_batch-expect_batch
        squared=difference**2
        #沿着dim=1，也就是行的方向求和，一行相加，每个label向量的元素平方之和，保留列相当于
        results=torch.sum(squared,dim=1)
        
        total=torch.sum(results)
        
        loss=total/batch_size
        
        if self.record_history:
            self.loss_history.append(loss.detach().cpu().item())
        
        return loss
