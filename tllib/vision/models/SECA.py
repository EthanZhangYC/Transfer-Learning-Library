# utf-8

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


__all__ = ['TSEncoder']


class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None

class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

class Classifier_domain(nn.Module):
    
    def __init__(self, n_class, input_dim, lamda=1.):
        super(Classifier_domain, self).__init__()
        self.lamda=lamda
        self.grad_reverse = GradientReversal(lambda_=self.lamda)
        
        #self.conv1 = nn.Conv2d(1024, 128, 3, 1)
        #self.conv2 = nn.Conv2d(64, 32, 3, 1)
        self.fc1 = nn.Linear(input_dim, input_dim//4)  # 5*5 from image dimension
        #self.fc2 = nn.Linear(32, 84)
        self.fc3 = nn.Linear(input_dim//4, n_class)
        
        #self.fc1 = nn.Linear(1280, 256)
        #self.fc2 = nn.Linear(256, n_class)
        #self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x): 
        #x = x.mean(3).mean(2)
        x = self.grad_reverse(x)
        #x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        #x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        #x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Classifier(nn.Module):
    
    def __init__(self, n_class, input_dim):
        super(Classifier, self).__init__()
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(input_dim, n_class)

    def forward(self, x): 
        #x = x.mean(3).mean(2)
        x = self.fc(self.dropout(x))
        return x

class SECAEncoder(nn.Module):
    def __init__(self, n_class=4):
        super().__init__()
        #self.input_fc = nn.Linear(input_dims, hidden_dims)
        
        #num_filter_ae_cls = [4, 32, 32, 64, 64, 128, 128]
        num_filter_ae_cls = [6, 32, 32, 64, 64]
        self.out_features=64

        layers=[]
        for i in range(len(num_filter_ae_cls)-1):
            layers.append(nn.Conv1d(num_filter_ae_cls[i], num_filter_ae_cls[i+1], kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU())
            # if i % 2 != 0 and i!=0:
            #     layers.append(nn.MaxPool2d(kernel_size=(1,2)))
        self.feature_extractor = nn.Sequential(*layers)
        self.repr_dropout = nn.Dropout(p=0.1)

        
    def forward(self, x):  # x: B x T x input_dims
        # conv encoder
        x = x.transpose(1, 2)  # B x 5 x T
        x = self.feature_extractor(x)  # B x Co x T
        x = x.transpose(1, 2)  # B x T x Co

        x = F.max_pool1d(
            x.transpose(1, 2),
            kernel_size = x.size(1),
        ).transpose(1, 2).squeeze(1)
        #x = torch.flatten(x, 1)
        
        return x
                       
class SECADecoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        #num_filter_ae_cls = [4, 32, 32, 64, 64, 128, 128]
        num_filter_ae_cls = [4, 32, 32, 64, 64]
        num_filter_decoder = num_filter_ae_cls[::-1]
        num_filter_ = sorted(set(num_filter_decoder), reverse=True)
        print('decoder:',num_filter_) #[64, 32, 4] 
        
        layers=[]
        for i in range(len(num_filter_)-1):
            layers.append(nn.Upsample(scale_factor=2))
            layers.append(nn.ConvTranspose1d(num_filter_[i], num_filter_[i], kernel_size=3, stride=1, padding=1))#, output_padding=1))
            layers.append(nn.ReLU())

            layers.append(nn.ConvTranspose1d(num_filter_[i], num_filter_[i+1], kernel_size=3, stride=1, padding=1))#, output_padding=1))
            if i==(len(num_filter_)-2):
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.ReLU())

        self.decoder = nn.Sequential(*layers)

        
    def forward(self, feat):  
        feat = feat.transpose(1, 2)
        x_re = self.decoder(feat)  
        x_re = x_re.transpose(1, 2)
        #print('x_re:',x_re.shape)
        return x_re
        

class SamePadConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups
        )
        self.remove = 1 if self.receptive_field % 2 == 0 else 0
        
    def forward(self, x):
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False):
        super().__init__()
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation=dilation)
        self.projector = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels or final else None
    
    def forward(self, x):
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual

class DilatedConvEncoder(nn.Module):
    def __init__(self, in_channels, channels, kernel_size):
        super().__init__()
        self.net = nn.Sequential(*[
            ConvBlock(
                channels[i-1] if i > 0 else in_channels,
                channels[i],
                kernel_size=kernel_size,
                #dilation=min(2**i,1024),
                dilation=2**i,
                final=(i == len(channels)-1)
            )
            for i in range(len(channels))
        ])
        
    def forward(self, x):
        return self.net(x)
        





def generate_continuous_mask(B, T, n=5, l=0.1):
    res = torch.full((B, T), True, dtype=torch.bool)
    if isinstance(n, float):
        n = int(n * T)
    n = max(min(n, T // 2), 1)
    
    if isinstance(l, float):
        l = int(l * T)
    l = max(l, 1)
    
    for i in range(B):
        for _ in range(n):
            t = np.random.randint(T-l+1)
            res[i, t:t+l] = False
    return res

def generate_binomial_mask(B, T, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)



# class TSEncoder(nn.Module):
#     def __init__(self, input_dims=9, output_dims=64, hidden_dims=64, depth=10, mask_mode='binomial', n_class=4):
#         super().__init__()
#         self.input_dims = input_dims
#         self.output_dims = output_dims
#         self.out_features=64
#         self.hidden_dims = hidden_dims
#         self.mask_mode = mask_mode
#         self.input_fc = nn.Linear(input_dims, hidden_dims)
#         self.feature_extractor = DilatedConvEncoder(
#             hidden_dims,
#             [hidden_dims] * depth + [output_dims],
#             kernel_size=3
#         )
#         self.repr_dropout = nn.Dropout(p=0.1)
        
#         self.fc = nn.Sequential(
#             nn.Linear(output_dims, 64),
#             nn.ReLU(),
#             nn.Dropout(p=0.5),
#             nn.Linear(64, n_class)
#         )
        
#     def forward(self, x):  # x: B x T x input_dims
#         # nan_mask = ~x.isnan().any(axis=-1)
#         pad_mask = x[:,:,0]==0 # pad -> True
#         # x[~nan_mask] = -1.
#         x[pad_mask] = 0.

#         x = self.input_fc(x)  # B x T x Ch

#         # conv encoder
#         x = x.transpose(1, 2)  # B x Ch x T
#         x = self.feature_extractor(x)  # B x Co x T
#         x = x.transpose(1, 2)  # B x T x Co

#         feat = x = F.avg_pool1d(
#             x.transpose(1, 2),
#             kernel_size = x.size(1),
#         ).transpose(1, 2).squeeze(1)

#         return feat




class TSEncoder(nn.Module):
    def __init__(self, input_dims=9, output_dims=64, hidden_dims=64, depth=10, n_class=4):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_size=3
        )
        self.repr_dropout = nn.Dropout(p=0.1)

        # self.fc = nn.Sequential(
        #     nn.Linear(output_dims, 64),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.1),
        #     nn.Linear(64, n_class)
        # )
        self.fc1 = nn.Linear(output_dims, 64)
        self.fc2 = nn.Linear(64, n_class)
        self.dropout = nn.Dropout(p=0.1)
        
        
    def forward(self, x):  

        x = self.input_fc(x)  # B x T x Ch

        # conv encoder
        x = x.transpose(1, 2)  # B x Ch x T
        x = self.feature_extractor(x)  # B x Co x T
        ori_feat = x = x.transpose(1, 2)  # B x T x Co

        x = F.avg_pool1d(
            x.transpose(1, 2),
            kernel_size = x.size(1),
        ).transpose(1, 2).squeeze(1)

        feat = x = F.relu(self.fc1(x))
        x = self.fc2(self.dropout(x))
        # logits = self.fc(feat) 
        # pred_each = self.fc(ori_feat)
        
        return x, feat#, ori_feat
