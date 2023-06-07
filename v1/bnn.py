
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
import numpy as np
import wandb
import sys

# ---------- debug functions --------------------------------
def debug(*mes): print("D> ",mes)
# ---------- Basic Functions ------------------------

def split_to_chunks(arr:torch.Tensor, chunk_size=8):
    "Split the array into chunks"
    chunked =  [arr[i:i + chunk_size] for i in range(0, len(arr), chunk_size)]
    # 0-padding last chunk
    if len(chunked[-1]) < chunk_size:
        padding = torch.tensor([0] * (chunk_size - len(chunked[-1])), dtype=arr.dtype).to( chunked[-1].device)
        chunked[-1] = torch.cat((chunked[-1], padding))
    return chunked
assert tuple(torch.stack(split_to_chunks(torch.randn(15)>0)).shape)==(2,8)

def unpackbits(x, bitorder='big'):
    assert x.dtype == torch.uint8
    if bitorder == 'big':
        bit_indices = torch.arange(7, -1, -1, dtype=torch.int64, device=x.device)
    elif bitorder == 'little':
        bit_indices = torch.arange(0, 8, dtype=torch.int64, device=x.device)
    else:
        raise ValueError("bitorder must be either 'big' or 'little'")
    new_shape = list(x.shape)
    new_shape[-1] *= 8
    return ((x[..., None] >> bit_indices) & 1).type(torch.int8).reshape(new_shape)

assert unpackbits(torch.tensor([7], dtype=torch.uint8)).tolist() == [0, 0, 0, 0, 0, 1, 1, 1]
assert list(unpackbits(torch.randn((32,8)).type(torch.uint8)).shape) == [32, 64]
assert list(unpackbits(torch.randn((32,8,3)).type(torch.uint8)).shape) == [32, 8, 24]

def packbits(x, bitorder='big')-> torch.Tensor:
    "Bit-packing 1D tensors of arbitrary length (with 0-padding)"
    if bitorder == 'big':
        bit_indices = torch.arange(7, -1, -1, dtype=torch.int64, device=x.device)
    elif bitorder == 'little':
        bit_indices = torch.arange(0, 8, dtype=torch.int64, device=x.device)
    else:
        raise ValueError("bitorder must be either 'big' or 'little'")
    new_shape = list(x.shape)
    new_shape[-1] = int(np.ceil(new_shape[-1]/8))
    x_padded = F.pad(x, (0, (8-x.shape[-1]%8)%8), value=0)
    return torch.stack([(xx * (1 << bit_indices)).sum(dim=-1, dtype=torch.uint8) 
        for xx in split_to_chunks(x_padded.flatten())]).reshape(new_shape)
assert packbits(torch.tensor([0, 0, 0, 0, 0, 1, 1, 1], dtype=torch.uint8)).tolist() == [7]
assert tuple(packbits(torch.randn((32,64))>0).shape)==(32,8)
assert tuple(packbits(torch.randn((32,64,3))>0).shape)==(32,64,1)

def a2bstr(arr):  # arr:list,torch.Tensor
    if type(arr) == torch.Tensor: arr = arr.tolist()
    arr = np.array(arr)
    return np.vectorize(lambda x:f"{int(x):08b}")(arr).tolist()

def xnor(v1:torch.Tensor,v2:torch.Tensor):
    "２つのベクトルのxnor演算（ビットが同じなら１）を行う。"
    return torch.bitwise_not(torch.bitwise_xor(v1, v2))
v1 = torch.tensor([0b0001, 0b0001], dtype=torch.uint8)
v2 = torch.tensor([0b0011, 0b0001], dtype=torch.uint8)
assert a2bstr(xnor(v1,v2))==['11111101', '11111111']
assert tuple(xnor(True, torch.randn((2,3,4)).type(torch.uint8)).shape)==(2,3,4), "ブロードキャストしてビット反転"

def popcount(arr:torch.Tensor, dim=-1):
    """popcount（1のビッチの数）を数える。arrはuint8のみ。
    Parameters
    -----------
    arr
    dim:int - 集計対象の軸。None:全軸で集計して0次テンソルを返す
    """
    assert type(arr) == torch.Tensor, "input must be cupy.ndarray"
    assert arr.dtype == torch.uint8, "dtype must be cupy.uint8"
    return torch.count_nonzero(unpackbits(arr), dim=dim)
assert popcount(torch.tensor([0b11111101, 0b11111111], dtype=torch.uint8))==15
assert list(popcount(torch.randn((32,8,3)).type(torch.uint8)).shape) == [32,8]
assert popcount(torch.tensor([[7,7],[7,7]], dtype=torch.uint8), dim=None).tolist() == 12


def sum_votes(flip_in, sum_axis):
    "flipを指定軸で、反転すべきか票数を合計する。"
    vote = unpackbits(flip_in)*2-1   # 1 -> 1, 0 -> -1
    return vote.sum(axis=sum_axis, dtype=torch.int32)

def standardize_bitbalance(tensor:torch.Tensor, n:int, pts_sigma=15)->torch.Tensor:
    """二項分布（binomial distribution）として正規化されたint8テンソルを返す
    Parameters
    ----------
    n:int      dimension of binomial distribution. (usualy, total number of bits)
    pts_sigma:int  pts/sigma after standardization
    """
    return (tensor/(0.5*np.sqrt(n))*pts_sigma).clamp(-127,127).type(torch.int8)
assert tuple(standardize_bitbalance(torch.randn((4,8)).type(torch.int32), 64).shape) == (4,8)

def standardize_popcount(tensor:torch.Tensor, n:int, pts_sigma=15)->torch.Tensor:
    """二項分布（binomial distribution）として正規化されたuint8テンソルを返す
    平均 127, 標準偏差 pts_sigma(15)
    Parameters
    ----------
    n:int      dimension of binomial distribution. (usualy, total number of bits)
    pts_sigma:int  pts/sigma after standardization
    """
    std = tensor.type(torch.float32).std()
    return ((tensor-0.5*n)/std*pts_sigma+127).clamp(0,255).type(torch.uint8)
assert tuple(standardize_popcount(torch.randn((4,8)).type(torch.int32), 64).shape) == (4,8)

def calc_1Ratio(x:torch.Tensor)->float:
    """1のビットの割合を求める
    用途：flipに対して - 反転する割合（Trueの数 / 全要素数）を返す
    ビットバランス - 0.5なら均衡、1なら全部True
    """
    sum_bits = x.numel()*8
    return float(popcount(x, dim=None)/sum_bits)
assert calc_1Ratio(torch.tensor([[7,7],[7,7]], dtype=torch.uint8)) == 0.375

# ------------------- BNN Modules ----------------
class WandbLogger():
    _dict = {}
    def log(self, dict):
        """
        Usage
        ---------
        log float  : logger.log({"BinaryTensor flip_ratio:": floatValue})
        log tensor : logger.log({"x_grad": wandb.Histogram(x_grad.cpu().numpy())})
        """
        self._dict.update(dict)
    def clear(self): self._dict.clear()
    def getLog(self): return self._dict

logger = WandbLogger()

class BinaryTensor(nn.Module):
    "Binary Tensor Module class"
    def __init__(self, shape, weights=None):
        """
        Parameters
        ------------------
        weights : bitpacked uint8 Tensor (or None)
        """
        super().__init__()
        self.shape = shape        
        self.weights = nn.Parameter(torch.randint(low=0, high=256, size=shape, dtype=torch.uint8), requires_grad=False) \
            if weights is None else nn.Parameter(torch.tensor(weights, dtype=torch.uint8), requires_grad=False)
        self.vote_accumulator = None
        self.flip_sigma = 0
        self.half_mask = None  # update... None:100%, 1:50%, 2:25%, ...
        
    def __repr__(self):
        return f"BinaryTensor(weights={self.shape}, dtype=uint8)"    
    def has_accumulator(self):
        return hasattr(self, 'vote_accumulator') and self.vote_accumulator is not None
    def update(self):
        if not self.has_accumulator(): return
        flip = self.vote_accumulator > 127+15*self.flip_sigma
        mask = packbits(flip)
        if self.half_mask is not None:
            times_rand = self.half_mask
            rand_mask = torch.ones(mask.shape, dtype=torch.uint8)*255
            for j in range(times_rand): rand_mask &= torch.randint(0,256, mask.shape, dtype=torch.uint8)
            rand_mask = 255 ^ rand_mask  # when 2: 25% -> 75%
            mask &= rand_mask.to(mask.device)
        self.weights = self.weights.to(mask.device)
        self.weights ^= mask  # update weights (using xor)
        logger.log({"BinaryTensor update_ratio:": calc_1Ratio(mask)})
        del self.vote_accumulator  # Clear memory
        self.vote_accumulator = None

    def forward(self):
        return self.weights

    def backward(self, flip):
        """
        Parameters
        ------------------
        vote:Tensor(uint8)  flip vote (i.e. should flip or not)
        """
        if not self.has_accumulator():
            self.vote_accumulator = unpackbits(self.weights).type(torch.int32).to(flip.device)
            self.vote_accumulator[:] = 0

        logger.log({"w_flip_ratio:": calc_1Ratio(flip)})
        vote = unpackbits(flip).sum([0,1], dtype=torch.int32) # b,d,o,i -> o,i
        vote = standardize_popcount(vote, n=torch.tensor(flip.shape[0:2]).prod())
        logger.log({"w_vote": wandb.Histogram(vote.cpu().numpy())})

        self.vote_accumulator += vote  # o,i
        return None

    def __call__(self): return self.forward()


TH_DEPTH15 = [-1.53, -1.15, -0.89, -0.67, -0.49, -0.32, -0.16, 0.00, 0.16, 0.32, 0.49, 0.67, 0.89, 1.15, 1.53]
TH_DEPTH7 = [-1.15, -0.67, -0.32, 0.00, 0.32, 0.67, 1.15]
TH_DEPTH3 = [-0.67, 0.00, 0.67]
TH_DEPTH1 = [0.00]

class Binarize(nn.Module):
    """
    FP32入力をBinarizeするレイヤ

    期待する入力： FP32 tensor (normalized: mean 0, std 1)
    出力： uint8 tensor. axis:b,d,i(batch, depth, input)
    """
    def __init__(self, depth_ths:list=TH_DEPTH3):
        """
        depth_ths  list of threshold values. ex) [0] -> 1bit, [-1,0,1] -> 2bit
        """
        super().__init__()
        self.depth = torch.tensor(depth_ths).unsqueeze(0).unsqueeze(-1)  # b,d,i

    def forward(self, x):
        h = packbits(x.unsqueeze(1)>self.depth.to(x.device)).clone().detach()  # b,d,i
        return h
    def backward(self, flip, x):
        logger.log({"x_flip_ratio:": calc_1Ratio(flip)})
        sign = (unpackbits(self.forward(x).unsqueeze(2)))*2-1  # b,d,i -> b,d,o,i
        x_grad = (sign * unpackbits(flip)).sum([1,2], dtype=torch.float32) # b,d,o,i -> b,i
        x_grad /= x_grad.std()  # 標準偏差が１になるように標準化。期待平均値は0のまま変わらず。
        # x_grad /= 0.5*torch.tensor(flip.shape[1:3]).prod().sqrt()
        logger.log({"x_grad": wandb.Histogram(x_grad.cpu().numpy())})
        return x_grad
    def __call__(self, x): return self.forward(x)


class BMA(nn.Module):
    """Binary Multiply-Accumulate Layer Module class
    
    期待する入力： packed uint8 tensor
    出力：FP32 (normalized mean0, std1)
    """
    def __init__(self, in_features:int=1, out_features:int=1, depth_ths:list=TH_DEPTH3, weights=None, is_standardize=True, pts_sigma=15):
        """
        Parameters
        ------------------
        in_features:int  - numbers of input bits
        out_features:int - numbers of output bits
        depth_features:int - numbers of bit depth
        weights : uint8 Tensor (or None)
        """
        super().__init__()
        in_features = int(np.ceil(in_features/8))
        self.weights = BinaryTensor(shape=(out_features, in_features), weights=weights)
        self.depth_ths = depth_ths
        self.is_standardize = is_standardize
        self.pts_sigma = pts_sigma
        self.weights_next = None

    def forward(self, x):
        x = x.unsqueeze(-2) # b,d,i -> b,d,o,i
        w = self.weights().unsqueeze(0).unsqueeze(0).to(x.device) # o,i -> b,d,o,i
        h0 = xnor(w, x)
        n1 = popcount(h0).sum(axis=1, dtype=torch.int32)  # b,d,o,i -> b,o
        n_all = (h0.shape[-1]*8)*h0.shape[2]
        assert n1.max() < n_all
        o0 = (2*n1 - n_all).type(torch.float32).requires_grad_().to(x.device)
        logger.log({"BMA_o0": wandb.Histogram(o0.cpu().numpy())})
        o1 = o0/(0.5*torch.tensor(n_all).sqrt())
        logger.log({"BMA_o1": wandb.Histogram(o1.cpu().numpy())})
        return o1

    def backward(self, grad, x):
        grad = grad.unsqueeze(-1).unsqueeze(1)  # b,o -> b,d,o,i
        x = x.unsqueeze(-2)  # b,d,i -> b,d,o,i

        # Update weights
        ww = self.weights().unsqueeze(0).to(x.device)  # d,o,i -> b,d,o,i
        hh =  xnor(x, ww)  # b,d,o,i
        w_flip = torch.bitwise_xor(hh, (grad<0))  # grad:b,d,o -> b,d,o,i
        self.weights.backward(w_flip)
        self.weights.update()
        ww_new = self.weights().unsqueeze(0).to(x.device)  # d,o,i -> b,d,o,i

        # Calculate x flip
        h_new =  xnor(x, ww_new)  # b,d,o,i
        x_flip = torch.bitwise_xor(h_new, (grad<0)) # b,d,o,i
        return x_flip

    def __call__(self, x): return self.forward(x)


