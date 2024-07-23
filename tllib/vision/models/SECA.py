# utf-8

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
from tllib.modules.grl import WarmStartGradientReverseLayer

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange
import pdb

__all__ = ['TSEncoder','Classifier_clf','ViT','AttnNet','TSEncoder_new', 'Classifier_clf_samedim', 'LabelEncoder', 'DimConverter']


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

class Classifier_clf(nn.Module):
    def __init__(self, n_class=4, input_dim=64):
        super(Classifier_clf, self).__init__()
        # self.dropout = nn.Dropout(p=0.5)
        # self.fc = nn.Linear(input_dim, n_class)
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, n_class)
        self.dropout_p=0.1
        self.dropout = nn.Dropout(p=self.dropout_p)

    def forward(self, x, aux_feat=None): 
        #x = x.mean(3).mean(2)
        # x = self.fc(self.dropout(x))
        x = self.dropout(F.relu(self.fc1(x)))
        feat = x
        x = self.fc2(x)
        return x
    
class Classifier_clf_samedim(nn.Module):
    
    def __init__(self, n_class=4, input_dim=64):
        super(Classifier_clf_samedim, self).__init__()
        self.convert = nn.Linear(input_dim*2, input_dim)
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, n_class)
        self.dropout_p = 0.1
        self.dropout = nn.Dropout(p=self.dropout_p)

    def forward(self, x, aux_feat=None): 
        x = self.convert(x)
        feat = x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
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
        

class AttnNet(nn.Module):
    def __init__(self, n_class=1, embed_dim=4, num_layers=2):
        super(AttnNet, self).__init__()
        self.fc = nn.Linear(embed_dim, n_class)
        self.lstm = nn.LSTM(input_size=1, hidden_size=embed_dim, num_layers=num_layers)
        
    def forward(self, x): 
        _,(_,x) = self.lstm(x)
        x = self.fc(x)
        return x
    
class LabelEncoder(nn.Module):
    def __init__(self, input_dim=1, embed_dim=16, num_layers=2):
        super(LabelEncoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, embed_dim)
        )
        
    def forward(self, x): 
        x = self.fc(x)
        return x

class DimConverter(nn.Module):
    def __init__(self, input_dim=64, out_dim=64):
        super(DimConverter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, out_dim),
            # nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            # nn.ReLU()
        )
        
    def forward(self, x): 
        x = self.fc(x)
        return x



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
    def __init__(self, input_dims=7, output_dims=64, hidden_dims=64, depth=10, n_class=4):
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
        self.dropout_p=0.1
        self.dropout = nn.Dropout(p=self.dropout_p)
        self.bn = nn.BatchNorm1d(64)
        
        # self.grl_layer = WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=1000, auto_step=False) 
        self.grl_layer=None
        self.adv_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, n_class)
        )
    
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.eval()
        
    def step(self):
        """
        Gradually increase :math:`\lambda` in GRL layer.
        """
        self.grl_layer.step()
        
    def forward(self, x, is_training=False):  

        x = self.input_fc(x)  # B x T x Ch

        # conv encoder
        x = x.transpose(1, 2)  # B x Ch x T
        x = self.feature_extractor(x)  # B x Co x T
        conv_feat = x = x.transpose(1, 2)  # B x T x Co

        feat = x = F.avg_pool1d(
            x.transpose(1, 2),
            kernel_size = x.size(1),
        ).transpose(1, 2).squeeze(1)
        ori_conv_feat=None

        # # x = self.dropout(F.relu(self.bn(self.fc1(x))))
        # x = self.dropout(F.relu(self.fc1(x)))
        # # if is_training:
        # #     x.mul_(math.sqrt(1 - self.dropout_p))
        # feat = x
        
        if self.grl_layer is not None:
            features_adv = self.grl_layer(feat)
            outputs_adv = self.adv_head(features_adv)
        
        x = self.fc2(x)
        # logits = self.fc(feat) 
        # pred_each = self.fc(ori_feat)  
        
        if self.grl_layer:
            return x, feat, conv_feat, ori_conv_feat, outputs_adv
        else:
            return x, feat, conv_feat, ori_conv_feat



class TSEncoder_new(nn.Module):
    def __init__(self, input_dims=7, output_dims=64, hidden_dims=64, depth=10, mask_mode='binomial', n_class=4, reconstruct_dim=2):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode
        
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_size=3
        )
        self.repr_dropout = nn.Dropout(p=0.1)
        self.fc = nn.Sequential(
            nn.Linear(output_dims, 64),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(64, n_class)
        )
                    
        self.fc_con = nn.Sequential(
            nn.Linear(output_dims, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        self.fc_va = nn.Sequential(
            nn.Linear(output_dims, 64),
            nn.ReLU(),
            nn.Linear(64, reconstruct_dim)
        )
        
    def forward(self, x, args=None, mask_early=False, mask_late=False):  
        nan_mask = ~x.isnan().any(axis=-1)

        if mask_early:
            # generate & apply mask
            mask = self.mask_mode
            if mask == 'binomial':
                mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
            elif mask == 'continuous':
                mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
            elif mask == 'all_true':
                mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            elif mask == 'all_false':
                mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
            elif mask == 'mask_last':
                mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
                mask[:, -1] = False
            ori_mask = mask.detach()
            x[~mask] = 0 # masked->False
        else:
            ori_mask=None
        
        x = self.input_fc(x)  # B x T x Ch

        # conv encoder
        x = x.transpose(1, 2)  # B x Ch x T
        x = self.feature_extractor(x)  # B x Co x T
        x = x.transpose(1, 2)  # B x T x Co
        ori_feat = x.clone()
        
        feat = x = F.avg_pool1d(
            x.transpose(1, 2),
            kernel_size = x.size(1),
        ).transpose(1, 2).squeeze(1)
        
        
        if mask_late:
            # generate & apply mask
            feat_list=[]
            for i in range(args.n_mask_late):
                mask = self.mask_mode
                tmp_ori_feat = ori_feat.clone()
                if mask == 'binomial':
                    mask = generate_binomial_mask(tmp_ori_feat.size(0), tmp_ori_feat.size(1)).to(x.device)
                else:
                    raise NotImplementedError                
                tmp_ori_feat[~mask] = 0 # masked->False
                tmp_feat = F.avg_pool1d(
                    tmp_ori_feat.transpose(1, 2),
                    kernel_size = tmp_ori_feat.size(1),
                ).transpose(1, 2).squeeze(1)
                feat_list.append(tmp_feat)
                # del tmp_ori_feat,
            logits=con_logits=va_logits=ori_mask=pred_each=None
            return logits, (feat,feat_list), ori_feat, con_logits, va_logits, ori_mask, pred_each
                            
        else:            
            # logits = self.fc(feat) 
            # pred_each = self.fc(ori_feat)
            # con_logits = None
            # va_logits = self.fc_va(ori_feat)
            logits=con_logits=va_logits=ori_mask=pred_each=None
            feat_list=None
            return logits, feat, ori_feat, con_logits, va_logits, ori_mask, pred_each






class PreNorm(nn.Module):
    def __init__(self, dim, fn, use_auxattn=False):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
        self.use_auxattn =use_auxattn
    def forward(self, x, attn_feat, use_cls_tokens=True, **kwargs):
        if self.use_auxattn:
            return self.fn(x, self.norm(attn_feat), use_cls_tokens, **kwargs)
        else:
            return self.fn(self.norm(x), attn_feat, use_cls_tokens, **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            # nn.Linear(hidden_dim, dim),
            # nn.Dropout(dropout)
        )
    def forward(self, x, attn_feat=None, use_cls_tokens=True):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., use_auxattn=False, double_attn=False):
        super().__init__()
        inner_dim = dim_head * heads 
        project_out = (not (heads == 1 and dim_head == dim)) or double_attn
        # project_out = True

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.use_auxattn = use_auxattn
        self.double_attn = double_attn

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        
        # if use_auxattn:
        #     self.to_qkv = nn.Linear(2, inner_dim * 3, bias = False)
        # else:
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_qkv_aux = nn.Linear(dim, inner_dim * 3, bias = False)

        # self.to_out = nn.Sequential(
        #     nn.Linear(inner_dim, dim),
        #     nn.Dropout(dropout)
        # ) if project_out else nn.Identity()
        1

    def forward(self, x, attn_feat, use_cls_tokens):
        # qkv = self.to_qkv(x).chunk(3, dim = -1)
        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        if self.use_auxattn:
            if self.double_attn:
                qkv_aux = self.to_qkv_aux(attn_feat).chunk(3, dim = -1)
                q_aux, k_aux, v_aux = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv_aux)
                v = x.unsqueeze(1)
                
                qkv_self = self.to_qkv(x).chunk(3, dim = -1)
                q_self, k_self, v_self = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv_self)
                
                # q = torch.cat([q_aux,q_self],dim=1)
                # k = torch.cat([k_aux,k_self],dim=1)
                # v = torch.cat([v_aux,v_self],dim=1)
                
                dots_aux = torch.matmul(q_aux, k_aux.transpose(-1, -2)) * self.scale
                dots_self = torch.matmul(q_self, k_self.transpose(-1, -2)) * self.scale
                # attn_aux = self.attend(dots_aux)
                # attn_self = self.attend(dots_self)
                # attn = self.attend(attn_aux * attn_self)
                attn = self.attend(dots_aux * dots_self)
                
                # pdb.set_trace()
                # attn = F.normalize(attn_aux * attn_self, dim=1)
                attn = self.dropout(attn)
                
            else:
                qkv_aux = self.to_qkv_aux(attn_feat).chunk(3, dim = -1)
                q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv_aux)
                v = x.unsqueeze(1)
                
                dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
                attn = self.attend(dots)
                attn = self.dropout(attn)
                        
        else:
            qkv = self.to_qkv(x).chunk(3, dim = -1)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            attn = self.attend(dots)
            attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        # feat = torch.sum(out,dim=1)
        
        # if use_cls_tokens:
        #     output = self.to_out(out)
        # else:
        #     output = self.to_out(feat)
        attn = rearrange(attn, 'b h n d -> b n (h d)')
        return out, out[:,0], attn[:,0]

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., use_auxattn=False, use_cls_tokens=True, double_attn=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.use_cls_tokens=use_cls_tokens

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout, use_auxattn=use_auxattn, double_attn=double_attn), use_auxattn=use_auxattn),
                # PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x, attn_feat):
        for attn, ff in self.layers:
            x,feat,attn = attn(x, attn_feat, self.use_cls_tokens)
            # x = out + x
            # x = attn(x) + x
            x = ff(x, attn_feat) #+ x
        return x, feat, attn

class ViT(nn.Module):
    def __init__(self, *, seq_len=650, patch_size=1, num_classes=4, dim=64, depth=1, heads=1, mlp_dim=64, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., use_cls_tokens=True, use_auxattn=False, double_attn=False, n_aux=2):
        super().__init__()
        assert (seq_len % patch_size) == 0

        num_patches = seq_len // patch_size
        patch_dim = channels * patch_size
        self.use_cls_tokens = use_cls_tokens
        self.double_attn = double_attn
        self.use_auxattn = use_auxattn

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (n p) -> b n (p c)', p = patch_size),
            nn.Linear(patch_dim, dim),
        )
        self.to_patch_embedding_auxfeat = nn.Sequential(
            Rearrange('b c (n p) -> b n (p c)', p = patch_size),
            nn.Linear(n_aux * patch_size, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(dim))
        self.cls_token_aux = nn.Parameter(torch.randn(dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, use_auxattn=self.use_auxattn, use_cls_tokens=self.use_cls_tokens, double_attn=self.double_attn)

        # if args.double_attn:
        #     dim=dim*2
        #     con_dim=con_dim*2
            
        self.mlp_head = nn.Sequential(
            # nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )


    def forward(self, series, attn_feat=None):
        # x = self.to_patch_embedding(series.transpose(1, 2))
        x = series
        b, n, _ = x.shape
        
        attn_feat = self.to_patch_embedding_auxfeat(attn_feat.transpose(1, 2))
        
        if self.use_cls_tokens:
            cls_tokens = repeat(self.cls_token, 'd -> b d', b = b)
            cls_tokens_aux = repeat(self.cls_token_aux, 'd -> b d', b = b)
            x, ps = pack([cls_tokens, x], 'b * d')
            attn_feat, _ = pack([cls_tokens_aux, attn_feat], 'b * d')
            x += self.pos_embedding[:, :(n + 1)]
        else:
            x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        x, feat, attn = self.transformer(x, attn_feat)

        if self.use_cls_tokens:
            cls_tokens, _ = unpack(x, ps, 'b * d')
            pred = self.mlp_head(cls_tokens)
        else:
            # x = torch.sum(x,dim=1)
            pred = self.mlp_head(x)
        
        # feat_con = self.fc_con(feat)

        return pred#, feat, feat_con, attn, cls_tokens
