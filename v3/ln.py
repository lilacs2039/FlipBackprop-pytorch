import warnings
from collections import deque
import inspect
from typing import Any, Callable, Dict, Optional, Tuple, Union
import types
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastai.losses import CrossEntropyLossFlat
from fastai.callback.core import Callback, CancelFitException
from fastai.callback.wandb import WandbCallback
from einops import reduce, repeat
from einops import rearrange as orig_rearrange
import pandas as pd
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import wandb
import sys

# ---------------- Constants --------------------------------
# Thresholds which equally split normal distribution
TH_DEPTH15 = [-1.53, -1.15, -0.89, -0.67, -0.49, -0.32, -0.16, 0.00, 0.16, 0.32, 0.49, 0.67, 0.89, 1.15, 1.53]
TH_DEPTH7 = [-1.15, -0.67, -0.32, 0.00, 0.32, 0.67, 1.15]
TH_DEPTH3 = [-0.67, 0.00, 0.67]
TH_DEPTH1 = [0.00]
TH_DICT = {
    1:TH_DEPTH1,
    3:TH_DEPTH3,
    7:TH_DEPTH7,
    15:TH_DEPTH15,
}
DTYPE_TO_BYTES = {
    torch.float16: 2,
    torch.float32: 4,
    torch.float64: 8,
    torch.uint8: 1,
    torch.int8: 1,
    torch.int16: 2,
    torch.int32: 4,
    torch.int64: 8,
    torch.bool: 1,
}

# ----------------- Global Variables --------------------------------
is_update = False  # 2nd Forward（値を更新するForward）ならTrue
update_rate = 0.5  # 重みを更新する割合。0~0.5
disrupt_ratio = 0.5  # 破壊率の、更新率に対する割合。0~0.5。
suppressor_rate = 0.5  # ビット更新(update/disrupt)のランダムな抑制率[%]  .0:update 100%, 0.25:update 75%, ...
                             # log2(1/p) or log2(1/(1-p)) should be integer. (ex. p=0.25, 0.5, 0.75,...)
enable_assert = True # assert文による検証を実行するか
enable_log_global = True    # wandbによるlogのglobal switch.
enable_log_heatmap = False  # heatmapは画像ファイルでサイズ大きいので別途オフにできるように。
enable_call_history = True  # 1st/2nd forward/backward　の呼出順序を記録するか

# ---------- debug functions --------------------------------
warnings.simplefilter('once')
def debug(*mes): print("D> ",mes)
def a2s(arr:Union[list, np.ndarray, torch.Tensor]):
    "Array To String. If bitpacked value (i.e. uint8), shows binary."
    if type(arr)==list: arr = np.array(arr)
    elif type(arr) == torch.Tensor: arr = arr.detach().cpu().numpy()
    return np.vectorize(lambda x:f"{int(x):08b}" if arr.dtype==np.uint8 else f"{x:.3f}")(arr).tolist()
def bits(arr=[0b11111111]):
    return torch.tensor(arr, dtype=torch.uint8)
assert a2s(bits([0b11111111]))==['11111111']
assert a2s(torch.tensor([1]))==['1.000']
def randbits(*shape): 
    """returns random bitpacked uint8 tensor with `shape`
    Usage : randbits(2,2)"""
    return torch.randint(0,256,shape).type(torch.uint8)


def assert_tensor_shape(t:torch.Tensor, shape:str, **kwargs):
    """
    Parameters
    ----------
    t - tensor to check
    shape - tensor shape
    kwargs - each dimension of number to check

    Usage
    ----------
    assert_shape(x_flip, 'b d i', i=calc_packed_features(self.in_features))      # fixed shape
    assert_shape(x_flip, 'b d ... x', x=calc_packed_features(self.in_features))  # variable shape
    """
    if not enable_assert: return    
    symbols_head, symbols_tail = [split_einops_pattern(s) for s in shape.split('...')] if '...' in shape else (split_einops_pattern(shape), [])
    if '...' in shape:
        assert len(symbols_head)+len(symbols_tail) <= len(t.shape), f'Shape not match. actual:{t.shape}, expected:{shape}'
    else:
        assert len(shape.split(' ')) == len(t.shape), f'Shape not match. actual:{t.shape}, expected:{shape}'
    # for s,dim_actual in zip(symbols, t.shape):
    for i in range(len(symbols_head)):
        s = symbols_head[i]
        dim_actual = t.shape[i]
        if s not in kwargs: continue
        dim_expected = kwargs[s]
        assert dim_actual == dim_expected, f'{i+1}th axis not match. actual:[{shape}]={list(t.shape)}, expected:{s}={dim_expected}'
    for i in range(len(symbols_tail)):
        i = -(i+1)
        s = symbols_tail[i]
        dim_actual = t.shape[i]
        if s not in kwargs: continue
        dim_expected = kwargs[s]
        assert dim_actual == dim_expected, f'{i}th axis not match. actual:[{shape}]={list(t.shape)}, expected:{s}={dim_expected}'
        
def add_indent(s:str):
    "複数行の文字列をインデントした文字列を返す。"
    lines = s.split('\n')
    lines = ['  ' + line for line in lines]
    return '\n'.join(lines)
assert add_indent("aaa\nbbb") == '  aaa\n  bbb'

# ---------- Basic Functions ------------------------
def rearrange(tensors:Union[torch.Tensor, list], pattern: Union[str, None], **axes_lengths):
    if pattern is None: return tensors
    if all([t.dtype != torch.uint8 for t in tensors]): return orig_rearrange(tensors, pattern, *axes_lengths)
    before, after = [split_einops_pattern(p) for p in pattern.split('->')]
    if before[-1] == after[-1]: return orig_rearrange(tensors, pattern, *axes_lengths)
    else: return packbits(orig_rearrange(unpackbits(tensors), pattern, *axes_lengths)) # 最下位軸を操作するときはunpackしてから操作する

class Dict2Obj(object):
    def __init__(self, dictionary): self.__dict__ = dictionary
def dict2obj(dict): return Dict2Obj(dict)
assert dict2obj({'a':1}).a==1

def rescale_tensor(tensor:torch.Tensor, old_range=(0,1), new_range=(-2,2)):
    min_val, max_val, new_min, new_max = *old_range, *new_range
    scale = (new_max - new_min) / (max_val - min_val)
    bias = new_min - min_val * scale
    return tensor * scale + bias    
assert float(rescale_tensor(torch.rand(5, 5)).abs().max())<2
assert (rescale_tensor(torch.tensor([0,0.5,1])) == torch.tensor([-2.,0.,2.])).all()

def split_to_chunks(arr:torch.Tensor, chunk_size=8):
    "Split the array into chunks"
    chunked =  [arr[i:i + chunk_size] for i in range(0, len(arr), chunk_size)]
    # 0-padding last chunk
    if len(chunked[-1]) < chunk_size:
        padding = torch.tensor([0] * (chunk_size - len(chunked[-1])), dtype=arr.dtype).to( chunked[-1].device)
        chunked[-1] = torch.cat((chunked[-1], padding))
    return chunked
assert tuple(torch.stack(split_to_chunks(torch.randn(15)>0)).shape)==(2,8)

def unpackbits(x, lowest_features:int=None, bitorder='big'):
    """
    lowest_features:int - 最下位軸の次元数
    """
    assert x.dtype == torch.uint8
    if bitorder == 'big':
        bit_indices = torch.arange(7, -1, -1, dtype=torch.int64, device=x.device)
    elif bitorder == 'little':
        bit_indices = torch.arange(0, 8, dtype=torch.int64, device=x.device)
    else:
        raise ValueError("bitorder must be either 'big' or 'little'")
    new_shape = list(x.shape)
    new_shape[-1] *= 8
    ret =  ((x[..., None] >> bit_indices) & 1).type(torch.int8).reshape(new_shape)
    return ret[...,:lowest_features]
assert unpackbits(torch.tensor([7], dtype=torch.uint8)).tolist() == [0, 0, 0, 0, 0, 1, 1, 1]
assert list(unpackbits(torch.randn((32,8)).type(torch.uint8)).shape) == [32, 64]
assert list(unpackbits(torch.randn((32,8,3)).type(torch.uint8)).shape) == [32, 8, 24]

def packbits(x, bitorder='big')-> torch.Tensor:
    "Bit-packing 1D tensors of arbitrary length (with 0-padding)"
    if x.dtype == torch.bool and x.shape[-1]==1:
        # Repeat BoolTensor before pitpacking. So that returns 0b11111111 (not 0b10000000).
        x = bool2bitpacked(x)
        # warnings.warn("Repeated BoolTensor before pitpacking. So that returns 0b11111111 (not 0b10000000).")
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

def split_einops_pattern(einops_pattern:str): return [s for s in einops_pattern.split(' ') if s != '']
assert len(split_einops_pattern('b d x'))==3

def build_pattern(in_ids, out_ids):
    return f"{' '.join(in_ids)} -> {' '.join(out_ids)}"
assert build_pattern(['b', 'd', 'o', 'i'], ['b', 'o']) == 'b d o i -> b o'

def build_reverse_pattern(reduce_pattern:str):
    ptn_in, ptn_out = reduce_pattern.split('->')
    out_ids = split_einops_pattern(ptn_out)
    ptn_restored = ' '.join([s if s in out_ids else '()' for s in split_einops_pattern(ptn_in)])
    restore_pattern = f"{ptn_out} -> {ptn_restored}"
    return restore_pattern
assert build_reverse_pattern('b d o i -> b o')==' b o -> b () o ()'

def restore_reduced_axis(x_reduced:torch.Tensor, reduce_pattern:str):
    "reduceで削除した軸を挿入して、reduce前の軸の数に回復したテンソルを返す。"
    return rearrange(x_reduced, build_reverse_pattern(reduce_pattern))
assert(list(restore_reduced_axis(torch.rand((2,3)), 'b d o i -> b o').shape)==[2, 1, 3, 1])

def count_votes(flip:torch.Tensor, vote:torch.Tensor, flip_unpacked=False):
    "flipを集計したvoteの集計ビット数を返す。"
    if not flip_unpacked: assert flip.dtype == torch.uint8, "flip(packed) must be torch.uint8 Tensor"
    else: assert flip.dtype == torch.int8, "flip(unpacked) must be torch.int8 Tensor"
    return (flip.numel() * (8 if not flip_unpacked else 1)) // vote.numel()

def calc_packed_features(features:int):
    "ビットパックしたときの要素数を返す"
    return int(np.ceil(features/8))
assert calc_packed_features(8)==1
assert calc_packed_features(9)==2

def calc_packed_features_x8(features:int):
    "ビットをアンパックしたときのビット数（=要素数 × 8) を返す"
    return calc_packed_features(features) * 8

build_randmask_p_warning_history = []
def build_randmask(shape:tuple, p:float):
    """
    shape:tuple - Shape of mask.
    p:float - Probability that each bit is 1. 
              log2(1/p) or log2(1/(1-p)) must be integer. (ex. p=0.25, 0.5, 0.75,...)
    """
    if not (0<p<=1): raise ValueError("p must be 0<p<=1")
    if p>0.5:
        do_inv=True
        p=1-p
    else: do_inv=False
    N_AND = torch.tensor(1/p).log2()
    if not (float(N_AND)).is_integer() and p not in build_randmask_p_warning_history:
        build_randmask_p_warning_history.append(p)
        warnings.warn(f"log2(1/p) or log2(1/(1-p)) must be integer, but p={p:.3f}..., so rounding occurs.")
    N_AND = int(N_AND)
    mask = torch.ones(shape, dtype=torch.uint8)*255  # all bits are 1
    for i in range(N_AND):
        mask &= torch.randint(0,256, shape, dtype=torch.uint8)
    if do_inv: mask = torch.bitwise_not(mask)
    return mask
assert list(build_randmask((2,2), p=0.25).shape) == [2,2]
assert list(build_randmask((2,2), p=0.75).shape) == [2,2]


def vote2flip(votes:torch.Tensor, n_votes:int, upper_percentile=0.5, lower_percentile=None, p_min_upper=None):
    """反転投票を集計・二値化した結果のflipを返す

    Parameters
    -------------
    votes:torch.Tensor -   投票のうち、Trueの数
    n_votes:int -          投票数（True, False含めて）
    update_rate:float - 投票の多い順に、Trueにする割合。0.~0.5を指定する。
    p_min_upper:float -  voteの最小しきい値。（二項分布の平均pNにより、pからしきい値へ変換する。）
                        0.~1.を指定する。Noneなら判定しない。
    """
    if n_votes == 1: 
        if lower_percentile is None: return packbits(votes > 0)
        else:
            flip_upper = packbits(votes > 0)
            flip_lower = bitwise_not(flip_upper)
            return flip_upper,flip_lower
    if lower_percentile is None:
        th_upper = torch.quantile(votes.type(torch.float), upper_percentile)
    else:
        th_upper, th_lower = torch.quantile(votes.type(torch.float), torch.tensor([upper_percentile, lower_percentile], dtype=torch.float).to(votes.device))
    th_upper = max(th_upper-1, 0)  # th=n_votesのとき、votes>thの条件を満たすflipは存在せず、重みの更新ができず学習ができないため。
    if lower_percentile is not None: th_lower = min(th_lower+1, n_votes)
    if p_min_upper is not None:
        # 二項分布のpの最小値保証
        p = th_upper/n_votes
        p = max(p_min_upper, p)
        th_upper = p*n_votes
    # from IPython.core.debugger import Pdb; Pdb().set_trace()
    if lower_percentile is None:
        flip_upper = packbits(votes > int(th_upper))
        return flip_upper
    else:
        flip_upper = packbits(votes > int(th_upper))
        flip_lower = packbits(votes < int(th_lower))
        return flip_upper, flip_lower

def reduce_flip(flip:torch.Tensor, reduce_pattern:str, update_rate:float, vote_p_min:float=None):
    "flipを集計・二値化してreduce処理する。"
    vote  = reduce(unpackbits(flip), reduce_pattern, 'sum').type(torch.int32)
    n_votes = count_votes(flip, vote)
    flip_ret = vote2flip(vote, n_votes, update_rate, p_min_upper=vote_p_min)
    return flip_ret

def flip2grad(bin:torch.Tensor, flip:torch.Tensor):
    """
    bin - Forward output. shape: b d ...
    flip - Backward input. shape: n b d ...
    """
    bin = rearrange(bin, '... -> () ...') # nbd...
    sign = unpackbits(bin)*2-1
    x_grad = reduce(sign*unpackbits(flip), 'n b d ... x -> b ... x', 'sum').type(torch.float32)
    std = x_grad.std()
    if not std.isnan() and std !=0:  # Skip when all values of x_grad are 0 and std is nan
        x_grad /= std  # 標準偏差が１になるように標準化。期待平均値は0のまま変わらず。
    return x_grad

def grad2flip(grad:torch.Tensor, x:torch.Tensor):
    """
    x:uint8 -> o:float
    flip:uint8 <- grad:float

    Parameters
    ----------
    grad - Backward input. shape: b ...
    x - Forward input. shape: b d ...
    """
    assert grad.dtype == torch.float32
    assert x.dtype == torch.uint8
    flip = bitwise_xor(x, grad<0)
    flip = bitwise_and(flip,grad!=0)
    return rearrange(flip, '... -> () ...') # nbd...

def bool2bitpacked(t:torch.Tensor) -> torch.Tensor:
    """最下位軸の次元数が1なら全ビット同じ値となるようにbitpackする。
    最下位軸の次元数が2以上なら、残りのビットを0埋めしてbitpackする。
    """
    assert t.dtype == torch.bool
    return packbits(t if t.shape[-1]!=1 else repeat(t, '... x -> ... (repeat x)', repeat=8))

def prep_bitwise(*tensors:Union[torch.Tensor, bool]) -> list:
    "bitwise演算できるようにテンソルを検証・前処理"
    ret_tensors = []
    for t in tensors:
        if isinstance(t, bool): t = torch.tensor([t])
        if t.dtype == torch.bool: t = bool2bitpacked(t)
        assert t.dtype == torch.uint8
        ret_tensors.append(t)
    if enable_assert:
        expected_shape = ret_tensors[0].shape
        for t in ret_tensors[1:]:
            for i in range(len(expected_shape)):
                if expected_shape[i]==1 or t.shape[i]==1: continue
                assert expected_shape[i]==t.shape[i], f"{i+1}th shape not match(or can't broadcast). actual:{t.shape}, expected:{expected_shape}"
    return ret_tensors

# ---------- Operation functions ------------------------
def bitwise_not(v1:Union[torch.Tensor, bool]):
    [v1] = prep_bitwise(v1)
    return torch.bitwise_not(v1)
def bitwise_xnor(v1:Union[torch.Tensor, bool],v2:Union[torch.Tensor, bool]):
    "２つのベクトルのxnor演算（ビットが同じなら１）を行う。"
    v1, v2 = prep_bitwise(v1, v2)
    return torch.bitwise_not(torch.bitwise_xor(v1, v2))
v1 = torch.tensor([0b0001, 0b0001], dtype=torch.uint8)
v2 = torch.tensor([0b0011, 0b0001], dtype=torch.uint8)
assert a2s(bitwise_xnor(v1,v2))==['11111101', '11111111']
assert tuple(bitwise_xnor(True, torch.randn((2,3,4)).type(torch.uint8)).shape)==(2,3,4), "ブロードキャストしてビット反転"

def bitwise_xor(v1:Union[torch.Tensor, bool],v2:Union[torch.Tensor, bool]):
    v1, v2 = prep_bitwise(v1, v2)
    return torch.bitwise_xor(v1, v2)
def bitwise_and(v1:Union[torch.Tensor, bool],v2:Union[torch.Tensor, bool]):
    v1, v2 = prep_bitwise(v1, v2)
    return torch.bitwise_and(v1, v2)
def bitwise_or(v1:Union[torch.Tensor, bool],v2:Union[torch.Tensor, bool]):
    v1, v2 = prep_bitwise(v1, v2)
    return torch.bitwise_or(v1, v2)
# def bitwise_mix(v1:Union[torch.Tensor, bool],v2:Union[torch.Tensor, bool]):
#     v1, v2 = prep_bitwise(v1, v2)
#     return

def backward_xnor(flip): return flip
def backward_updated_xnor(flip, other_old, other_new):
    """XNOR演算の他方の入力が更新されたときの、更新したflipを求める。
    Usage : flip_b = backward_xnor(flip_out, a_new, a_old)
    """
    other_flipped = rearrange(bitwise_xor(other_new, other_old), '... -> () ...') # n...
    flip_x = bitwise_xor(flip, other_flipped)
    return flip_x
def backward_and(flip, x:torch.Tensor, other:torch.Tensor):
    """
    Parameters
    -------------------
    x:torch.Tensor     - 逆伝搬を計算したい入力
    other:torch.Tensor - もう片方の入力

    Usage
    -------------------
    flip_A = backward_and(flip,A,B)
    """
    case_false = bitwise_and(x, bitwise_not(other))  # 1 when x==1, other==0
    case_false = rearrange(case_false, '... -> () ...') # n...
    return bitwise_and(flip, bitwise_not(case_false))  # 1 when flip==1, case_false==0
assert unpackbits(backward_and(False, randbits(2,2), randbits(2,2))).any()==False, "0 when no flip"
assert unpackbits(backward_and(True, False, randbits(2,2))).all()==True, "1 when input is 0"
assert unpackbits(backward_and(True, True, False)).any()==False, "0 when 'x==1 & other==0'"
assert unpackbits(backward_and(True, True, True)).all()==True, "1 when 'x==1 & other==1'"
def backward_or(flip, x:torch.Tensor, other:torch.Tensor):
    """
    Parameters
    -------------------
    x:torch.Tensor     - 逆伝搬を計算したい入力
    other:torch.Tensor - もう片方の入力

    Usage
    -------------------
    flip_A = backward_or(flip,A,B)
    """
    case_false = bitwise_and(bitwise_not(x), other)  # 1 when x==0, other==1
    case_false = rearrange(case_false, '... -> () ...') # n...
    return bitwise_and(flip, bitwise_not(case_false))  # 1 when flip==1, case_false==0
assert unpackbits(backward_or(False, randbits(2,2), randbits(2,2))).any()==False, "0 when no flip"
assert unpackbits(backward_or(True, False, False)).all()==True, "1 when 'x==0 & other==0'"
assert unpackbits(backward_or(True, False, True)).any()==False, "0 when 'x==0 & other==1'"
assert unpackbits(backward_or(True, True, randbits(2,2))).all()==True, "1 when input is 1"

def bitwise_popcount(x:torch.Tensor):
    "popcount lowest axis by kernighan's algorithm"
    c = torch.zeros_like(x)
    x = x - ((x >> 1) & 0x55)
    x = (x & 0x33) + ((x >> 2) & 0x33)
    x = (x + (x >> 4)) & 0x0F
    return x
assert (bitwise_popcount(torch.tensor([[0, 1, 4], [15, 16, 255]], dtype=torch.uint8)) == torch.tensor([[0, 1, 1],[4, 1, 8]], dtype=torch.uint8)).all()
def popcount(x:torch.Tensor, reduce_pattern:str='... x -> ...', n_features:int=None):
    """popcount（1のビッチの数）を数える。arrはuint8のみ。
    Parameters
    -----------
    x - 集計対象のテンソル
    reduce_pattern:int - 集計する軸をeinopsのパターンで指定。Noneなら全軸を集約。デフォルト：最下位軸だけ集計。
    """
    assert type(x) == torch.Tensor, "input must be torch.Tensor"
    assert x.dtype == torch.uint8, "dtype must be torch.uint8"
    if reduce_pattern is None: return torch.sum(bitwise_popcount(x), dim=None, dtype=torch.int32)
    ptn_in, ptn_out = reduce_pattern.split('->')
    in_ids = split_einops_pattern(ptn_in)
    out_ids = split_einops_pattern(ptn_out)
    if in_ids[-1] not in out_ids:
        x = bitwise_popcount(x)
    else: x = unpackbits(x)[...,:n_features]
    return reduce(x, build_pattern(in_ids, out_ids), 'sum').type(torch.int32)
assert list(popcount(randbits(2,3,4,5), 'b d o i -> b o').shape)==[2, 4]
assert list(popcount(randbits(2,3,4,5), 'b d o i -> b o i').shape) == [2, 4, 40], '最下位軸を集約しない場合、unpackするので要素数8倍になる'
assert popcount(torch.tensor([0b11111101, 0b11111111], dtype=torch.uint8))==15
assert list(popcount(torch.randn((32,8,3)).type(torch.uint8)).shape) == [32,8]
assert popcount(torch.tensor([[7,7],[7,7]], dtype=torch.uint8), reduce_pattern=None).tolist() == 12

def backward_popcount(grad, x):
    flip = grad2flip(grad, x)
    return flip

def bitbalance(arr):
    n1 = popcount(arr)
    n_all = (arr.shape[-1]*8)
    assert n1.max() <= n_all
    return (2*n1 - n_all)
def bitbalance_backward(h, grad):
    flip = bitwise_xor(h, (grad<0))
    return flip


def standardize_bitbalance(tensor:torch.Tensor, n:int, pts_sigma=15)->torch.Tensor:
    """二項分布（binomial distribution）として正規化されたint8テンソルを返す
    Parameters
    ----------
    n:int      dimension of binomial distribution. (usualy, total number of bits)
    pts_sigma:int  pts/sigma after standardization
    """
    return (tensor/(0.5*np.sqrt(n))*pts_sigma).clamp(-127,127).type(torch.int8)
assert tuple(standardize_bitbalance(torch.randn((4,8)).type(torch.int32), 64).shape) == (4,8)

def meanstd_binominal01(n:int, p:float):
    mean = p*n
    std = torch.tensor(n*p*(1-p)).sqrt()
    return mean,std

def normalize_binominal01(tensor:torch.Tensor, n:int, p:float=0.5)->torch.Tensor:
    """False,Trueのときそれぞれ0,1である二項分布（binomial distribution）を、確率pの平均・標準偏差で標準化
    期待平均 0, 期待標準偏差 1
    Parameters
    ----------
    n:int      dimension of binomial distribution. (usualy, total number of bits)
    p:float    
    """
    mean,std = meanstd_binominal01(n,p)
    return ((tensor.type(torch.float32)-mean)/std)
assert tuple(normalize_binominal01(torch.randint(0,32,(4,8)), 32).shape) == (4,8)

def binarize(x:torch.Tensor, depth_ths:torch.Tensor):
    return packbits(x>depth_ths.to(x.device))

def backward_binarize(flip:torch.Tensor, x:torch.Tensor, depth_ths:torch.Tensor):
    x = rearrange(x, 'b ... -> b () ...') #bdx
    bin = binarize(x, depth_ths)
    return flip2grad(bin, flip)

def popcount_normalize(x:torch.Tensor, reduce_pattern:str, n_features:int=None):
    """
    n_out:int - unpack後の出力次元数。bitpack時のpaddingがないときはNoneでよい。
                Why needed? : 最下位軸の次元数はbitpack時に8の倍数に制約されるのでpaddingするが、
                                floatに変換するときにpaddingの分を除去するために次元数の指定が必要。
    """
    # Validate
    assert type(x) == torch.Tensor, "input must be torch.Tensor"
    assert x.dtype == torch.uint8, "dtype must be torch.uint8"
    # popcount
    x = unpackbits(x)[...,:n_features]
    o0 = reduce(x, reduce_pattern, 'sum').type(torch.int32)
    # normalize
    n_all = count_votes(x, o0, flip_unpacked=True)
    o1 = normalize_binominal01(o0, n_all, p=0.5)
    return o1

def backward_popcount_normalize(grad, x):
    return grad2flip(grad,x)

def popcount_normalize_binarize(x:torch.Tensor, depth_ths:torch.Tensor, pcnt_reduce_pattern:str):
    """ popcount, normalize, then binarize
    Shape : b d o i -> b o -> b d o

    Usage
    -----------
    o = popcount_normalize_binarize(h0, self.depth_ths, 'b d o i -> b o')
    """
    n0 = popcount(x, pcnt_reduce_pattern)
    mean,std = meanstd_binominal01(n=count_votes(x, n0), p=0.5)
    # binarize
    n0 = rearrange(n0,'b ... -> b () ...') # bd...
    b0 = binarize(n0, depth_ths*std+mean)
    return b0

def backward_popcount_normalize_binarize(flip, x, depth_ths:torch.Tensor, pcnt_reduce_pattern:str, flip_lowest_features:int):
    """binarize-popcountのbackwardを一緒に実行して無駄な演算を省く
    （標準化しない、float32ではなくintで計算）

    x:uint8 -> bin:uint8
    flip_out:uint8 <- flip:uint8

    Parameters
    ----------
    flip - Backward input. shape: n b d ...
    x -  Forward input. shape: b ...
    flip_lowest_features:int - 最下位軸のpadding前の要素数。paddingないならNoneでよい。
    """
    assert type(depth_ths) == torch.Tensor, "depth_ths must be Tensor."
    # backward binarize
    bin = popcount_normalize_binarize(x,depth_ths,pcnt_reduce_pattern)
    sign = unpackbits(bin, lowest_features=flip_lowest_features)*2-1
    sign = rearrange(sign, '... -> () ...') # nbd...
    intGrad = reduce(sign*unpackbits(flip, lowest_features=flip_lowest_features), 'n b d ... -> b ...', 'sum').type(torch.int32)  # int32 : n*d < 20億 の想定
    intGrad = restore_reduced_axis(intGrad, pcnt_reduce_pattern)  # ex. bdoi
    # backward popcount
    flip_out = bitwise_xor(x, intGrad<0)
    flip_out = bitwise_and(flip_out, intGrad!=0)
    flip_out = rearrange(flip_out, '... -> () ...') # nbd...
    return flip_out

def hamming_distance(y:torch.Tensor, y_target:torch.Tensor, scaled=False)->tuple[torch.Tensor, Union[int,float]]:
    "y, y_target -> flip, Loss"
    assert y.dtype==torch.uint8
    assert y_target.dtype==torch.uint8
    flip = bitwise_xor(y, y_target)  # Lossから求めたflipに相当
    flip = rearrange(flip, '... -> () ...') # n...
    L = popcount(flip).sum()
    if not scaled: return flip, int(L)
    else:
        n = flip.numel()*8
        return flip, float(L)/n
assert hamming_distance(bits([255]), bits([255])) == (torch.tensor([[0]], dtype=torch.uint8), 0)
assert hamming_distance(bits([255]), bits([0]))==(torch.tensor([[255]], dtype=torch.uint8), 8)
assert hamming_distance(bits([15]), bits([255]), scaled=True)==(torch.tensor([[240]], dtype=torch.uint8), 0.5)

# ---------------- Learning ----------------------
def find_ur(learn, ur_list=[.01, .05]+np.arange(.1,.5,.05).tolist()):
    "Find optimal Update Rate"
    # sweep update rate
    cb = URFinderCallback(ur_list=ur_list)
    learn.fit(1000, cbs=[cb])
    # plot results
    x,y = cb.ur_loss_dict.keys(), cb.ur_loss_dict.values()
    y = np.array(list(y))
    plt.plot(x,y)
    plt.title('Search for optimal update rate in BMA')
    plt.xlabel('Update rate')
    plt.ylabel('Loss')

class URFinderCallback(Callback):
    def __init__(self, ur_list):
        self.max_iter = len(ur_list)
        self.cur_iter = 0
        self.ur_list = ur_list
        self.ur_loss_dict = {}
        global update_rate
        update_rate = self.ur_list[self.cur_iter]

    def after_batch(self):
        self.ur_loss_dict[self.ur_list[self.cur_iter]] = self.learn.loss.item()
        self.cur_iter += 1
        if self.cur_iter >= self.max_iter:
            raise CancelFitException()
        global update_rate
        update_rate = self.ur_list[self.cur_iter]

def setup_cbs(learn, ur_range:Union[tuple,float,None] = (0.5, 0.01), ur_end_pct = 0.8, call_history_only_first_call=None): 
    """
    本ライブラリで必要な以下の処理をCallbackを登録する
    - FBF learning
    - update rate scheduling
    - CallHistory
    
    Parameters
    --------------------------------
    call_history_only_first_call - Trueなら最初の呼出だけ記録。Noneなら記録しない。

    Usage
    --------------------------------
    ln.setup_cbs(learn, [model.binary_net], bs_range=(32, 'all'))
    """
    learn.add_cbs([
        FBFCallback(),
        URCallback(ur_range, ur_end_pct),
        ]+([CallHistoryCallback(only_first_call=call_history_only_first_call)] if call_history_only_first_call is not None else [])
        )

class FBFCallback(Callback):
    def __init__(self, **kwargs): 
        """
        FBF learningを行うCallback。
        """
        super().__init__(**kwargs)
    def after_step(self):  # after update of float parameters
        global is_update
        is_update = True
        self.learn.model(*self.learn.xb)
        is_update = False

class CallHistoryCallback(Callback):
    def __init__(self, only_first_call=True, **kwargs): 
        """
        CallHistoryの記録を行うCallback。
        only_first_call:bool - Trueなら最初の呼出だけ記録

        Usage
        --------------------
        learn = Learner(...
               cbs=[
                   ln.CallHistoryCallback(), ...])
        learn.fit(10)
        print('\n'.join(call_history.call_logs))  # show call_history
        """
        super().__init__(**kwargs)
        self.only_first_call = only_first_call
    def before_fit(self): 
        global enable_call_history, call_history
        enable_call_history = True
        call_history.clear()
    def before_epoch(self):
        global enable_call_history, call_history
        # if self.learn.epoch > self.last_epoch: enable_call_history = False
        if enable_call_history: call_history.add_any_message(f"---- epoch {self.epoch} ----")
    def before_train(self):
        global enable_call_history, call_history
        if enable_call_history: call_history.add_any_message(f"** train")
    def before_batch(self):
        global enable_call_history, call_history
        if enable_call_history: call_history.add_any_message(f"**** batch")
    def after_batch(self):
        if self.only_first_call:
            global enable_call_history
            enable_call_history=False
    def before_validate(self):
        global enable_call_history, call_history
        if enable_call_history: call_history.add_any_message(f"** valid")
    def after_fit(self):
        global call_history
        print('\n'.join(['***** Call history *****']+call_history.call_logs))

class URCallback(Callback):
    def __init__(self, ur_range:Union[tuple,float,None] = (0.5, 0.01), ur_end_pct = 0.8, **kwargs): 
        """
        update_rateの動的変更を行うCallback。

        Parameters
        --------------------------------
        ur_range:tuple - update_rateの範囲
                        Noneならスケジューリングしない。
                        floatなら固定
        ur_end_pct:float - update_rateが最小値になるステップの進行度[%]。 0.~1.
        
        Usage
        --------------------------------
        learn = Learner(..., cbs=[
                   ln.URCallback(ur_range=(0.5, 0.01)),
                   ...
                   ])
        """
        super().__init__(**kwargs)
        # self.binary_nets = binary_nets
        if ur_range is None: self.ur_start = self.ur_end = None
        elif type(ur_range) == float: self.ur_start = self.ur_end = ur_range
        else: self.ur_start, self.ur_end = ur_range  # tuple or list
        self.ur_end_pct = ur_end_pct
    def before_fit(self): 
        # prepare ur
        global update_rate
        if self.ur_start is not None: update_rate = self.ur_start
    def after_batch(self):
        # update_rate
        if self.ur_start is not None:
            global update_rate
            if self.pct_train <= self.ur_end_pct:
                update_rate = self.ur_start + (self.ur_end - self.ur_start) * (self.pct_train / self.ur_end_pct)
            else: update_rate = self.ur_end

# ---------------- Evaluation Utils ----------------------
def calc_1ratio(x:torch.Tensor)->float:
    """1のビットの割合を求める
    用途：flipに対して - 反転する割合（Trueの数 / 全要素数）を返す
    ビットバランス - 0.5なら均衡、1なら全部True
    """
    sum_bits = x.numel()*8
    return float(popcount(x, reduce_pattern=None)/sum_bits)
assert calc_1ratio(torch.tensor([[7,7],[7,7]], dtype=torch.uint8)) == 0.375

from sklearn.metrics import auc
from fastai.metrics import accuracy

class TLogicWandbCallback(WandbCallback):
    """
    meric_func:Callable - 学習の良さを測定する指標。

    Usage
    --------------------
    learn = Learner(...
                   cbs=[
                       TLogicWandbCallback(metric_func=accuracy),
                       ...
                   ])
    """
    def __init__(self, metric_func=accuracy, **kwargs):
        super().__init__(**kwargs)
        self._wandb_step = 0
        self.metric_func = metric_func
        self.metric_values = []
    def before_fit(self):
        super().before_fit()
        self.start_time = time.time()
    def after_batch(self):
        super().after_batch()
        # Calculate metric
        if self.metric_func is not None:
            self.metric_values.append(float(self.metric_func(self.pred, self.y)))
        # Log to wandb
        if self.training:
            self._wandb_step += 1
            global update_rate, disrupt_ratio, suppressor_rate
            wandb.log({
                'update_rate': update_rate,
                'disrupt_ratio': disrupt_ratio,
                'suppressor_rate': suppressor_rate,
                **{f"{name}": wandb.Histogram(param.data.cpu().numpy())
                    for name, param in self.learn.model.named_parameters()},
                **logger.getLog()
            }, step=self._wandb_step)
            logger.clear()

    def after_fit(self):
        super().after_fit()
        if self.metric_func is not None:
            wandb.log({f"variance_diff_{self.metric_func.__name__}": np.var(np.diff(self.metric_values))})
            wandb.log({f"auc_{self.metric_func.__name__}": auc(np.arange(0,1,1/len(self.metric_values)), self.metric_values)})
            self.metric_values = [] # Reset for next fit
        wandb.log({
            'total_time': time.time() - self.start_time,
            'parameter_size[Byte]': calc_param_size(self.learn.model),
        })
        global enable_log_global
        if enable_log_global:
            for title,ax in logger.createHeatmaps().items():
                wandb.log({f"{title}": wandb.Image(ax, f"{title}")})
            logger.clear_heatmap()

class WandbLogger():
    """wandb用のログを蓄積するクラス。
    fastai Callbackでwandb処理を行う想定。
    """
    _dict = {}
    _heatmap = {}  # k:title, v:[{values=[], vmax=1,vmin=0, ...]
    def clear(self): self._dict.clear()
    def getLog(self): return self._dict
    # def heatmap(self, title_contents_dict:dict): 
    def getHeatmap(self): return self._heatmap
    def clear_heatmap(self):
        self._heatmap.clear()
        plt.close()
    def createHeatmaps(self):
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        ret = {}
        for title,contents in self._heatmap.items():
            df = pd.DataFrame(contents['values']).T
            fig, ax = plt.subplots(figsize=(20, 10))
            sns.heatmap(df, ax=ax, vmax=contents['vmax'], vmin=contents['vmin'])
            ax.set_title(title)
            ax.set_xlabel("Steps")
            ret[f"{title}"] = ax
        return ret
    def log(self, dict):
        """
        Usage
        ---------
        log float  : logger.log({"BinaryTensor flip_ratio": floatValue})
        log tensor : logger.log({"x_grad": wandb.Histogram(x_grad.cpu().numpy())})
        """
        global enable_log_global
        if enable_log_global: self._dict.update(dict)
    def log_1ratio(self, title, t:torch.Tensor):
        global enable_log_global
        if enable_log_global: self.log({title : calc_1ratio(t)})
    def log_hist(self, title, t:torch.Tensor):
        global enable_log_global
        if enable_log_global: logger.log({title: wandb.Histogram(t.cpu().numpy())})
    def log_heatmap(self, title, values:torch.Tensor, vmax=1, vmin=0): 
        "一ステップ分の１次元テンソルを記録。終了時にheatmapとしてwandbに保存できるように情報を保持。"
        global enable_log_heatmap
        if not enable_log_heatmap: return
        values = values.cpu().numpy()
        if title not in self._heatmap: self._heatmap[title] = {'values':[]}
        self._heatmap[title]['values'].append(values) #, 'vmax':vmax,'vmin':vmin})
        self._heatmap[title]['vmax'] = vmax
        self._heatmap[title]['vmin'] = vmin

logger = WandbLogger()

def _log_fb_tensor(layer, fwdbwd:str, io:str, t:torch.Tensor):
    if t is None: return  # Case: BinaryTensor#backward etc.
    global enable_log_global
    if not enable_log_global: return
    if t.dtype == torch.uint8:
        value_type = 'bits' if fwdbwd == 'forward' else 'flip'
        logger.log_1ratio(f"{layer.path}/{fwdbwd}_{io}/{value_type}-1ratio", t)
        t_dx = reduce(unpackbits(t).type(torch.float32), 'b ... -> ...', 'mean').flatten()
        logger.log_heatmap(f"{layer.path}/{fwdbwd}_{io}/mean_of_bits-heatmap", t_dx)
        logger.log_hist(f"{layer.path}/{fwdbwd}_{io}/mean_of_bits", t_dx)
        if t.ndim>2:
            t_d = reduce(unpackbits(t).type(torch.float32), 'b d ... -> d', 'mean').flatten()
            logger.log_heatmap(f"{layer.path}/{fwdbwd}_{io}/mean_of_bits-d-heatmap", t_d)
            logger.log_hist(f"{layer.path}/{fwdbwd}_{io}/mean_of_bits-d", t_d)
            t_x = reduce(unpackbits(t).type(torch.float32), 'b d ... -> ...', 'mean').flatten()
            logger.log_heatmap(f"{layer.path}/{fwdbwd}_{io}/mean_of_bits-x-heatmap", t_x)
            logger.log_hist(f"{layer.path}/{fwdbwd}_{io}/mean_of_bits-x", t_x)
    elif t.dtype == torch.float32:
        value_type = 'float_value' if fwdbwd == 'forward' else 'grad'
        logger.log_hist(f"{layer.path}/{fwdbwd}_{io}/{value_type}", t)

def log_forward_in(layer, forward_in:torch.Tensor): _log_fb_tensor(layer, 'forward', 'in', forward_in)
def log_forward_out(layer, forward_out:torch.Tensor): _log_fb_tensor(layer, 'forward', 'out', forward_out)
def log_backward_in(layer, backward_in:torch.Tensor): _log_fb_tensor(layer, 'backward', 'in', backward_in)
def log_backward_out(layer, backward_out:torch.Tensor): _log_fb_tensor(layer, 'backward', 'out', backward_out)

def search_parameters(module):
    "Moduleを再帰的に探索してパラメータを取得する。"
    ret = {}
    for module_name, sub_module in module.named_children():
        for param_name, parameter in sub_module.named_parameters(recurse=False):
            ret[(sub_module.path if isinstance(sub_module, Module) else module_name)+'.'+param_name] = parameter
        ret.update(search_parameters(sub_module))  # Recurse
    return ret

def create_parameters_info(model:nn.Module, show_formatted=True)->pd.DataFrame:
    "Moduleを再帰的に探索してパラメータ情報を作成する"
    ret = []
    for param_path, parameter in search_parameters(model).items():
        ret.append({
            'name': param_path,
            'dtype': str(parameter.dtype),
            'param_shape' : list(parameter.shape),
            '1ratio': calc_1ratio(parameter.data) if parameter.dtype==torch.uint8 else None,
            'num_params': torch.numel(parameter) * (1 if parameter.dtype!=torch.uint8 else 8),
            'size[Byte]' : torch.numel(parameter)*DTYPE_TO_BYTES[parameter.dtype],
        })
    df = pd.DataFrame(ret)
    df.num_params = df.num_params.astype(int)
    df.set_index('name', inplace=True)
    df.loc['Total'] = df.select_dtypes(include='int').sum().astype(int)
    if show_formatted:
        df['1ratio'] = df['1ratio'].map(lambda x: '{:.3f}'.format(x) if x is not None else x)
        df['num_params'] = df['num_params'].map('{:,.0f}'.format)
        df['size[Byte]'] = df['size[Byte]'].map('{:,.0f}'.format)
    else:
        df['num_params'] = df['num_params'].astype(int)
        df['size[Byte]'] = df['size[Byte]'].astype(int)
    return df

def calc_param_size(model:nn.Module)->int:
    "modelに含まれるパラメータの合計サイズ[Byte]を求める"
    df = create_parameters_info(model, show_formatted=False)
    param_size_byte = int(df['size[Byte]'][-1])
    return param_size_byte

# ------------------- Utils ----------------
class Path:
    """ Path for module relationships.
    Usage
    -------------------
    path = Path('a.b.c')
    path_d = path+'.d'
    path_parent = path.parents[1]  # 'a.b'
    path_element = path.stem
    """
    def __init__(self, _path): 
        self._path = _path
        self.parents = self.Parents(self)
    @property
    def stem(self)->str: return self._path.split('.')[-1]
    def __add__(self, other:str):
        return Path(f'{self._path}{other}')
    def __str__(self)->str: return self._path
    def __repr__(self)->str: return self._path
    class Parents:
        def __init__(self, path): self.path = path
        def __getitem__(self, index):
            return '.'.join(self.path._path.split('.')[:index+1])

class CallHistory():
    """
    
    Usage
    --------------------

    Result
    --------------------
    1th forward : Seq
        1th forward : Seq.0-Binarize
        1th forward : Seq.1-BMAB
            1th forward : BMAB.0-XnorWeight
            ...
        1th forward : Seq.2-BMA
        ...

    """
    module_stack = deque([])
    call_logs:list[str] = []
    def enter(self, message:str):
        global enable_call_history
        if not enable_call_history: return
        self.call_logs.append(f"{'  '*len(self.module_stack)}{message}")
        self.module_stack.append(message)
    def exit(self):
        global enable_call_history
        if not enable_call_history: return
        self.module_stack.pop()
    def add_any_message(self, message):
        global enable_call_history
        if not enable_call_history: return
        self.call_logs.append(f"{'  '*len(self.module_stack)}{message}")
    def clear(self):
        self.module_stack.clear()
        self.call_logs.clear()
call_history = CallHistory()


# ------------------- Basic Modules ----------------
def num2ordinal(num:int):
    if num ==1: return '1st'
    elif num ==2: return '2nd'
    elif num==3: return '3rd'
    else: return num+'th'
class Module(nn.Module):
    """
    ルール
    - 子モジュールはself.layersに追加する
    - Backwardや2nd Forwardに必要なテンソルは、必要に応じてそれぞれforward_tensor, backward_tensorに保持する。
        - backward_tensorは親Moudleによる書き換えてから2nd forwardの実行がありうる。
    - do_forward,do_backwardを実装すること
    - 外部から呼び出すときはforward/backwardを実行すること（前後のフックが実行されるように）
    - 表現文字列（repr）に表示したいパラメータはself.paramsに保持すること。
      - 表示例： (1): PcntNormBin(depth_ths='tensor([[[0.]]])', pcnt_reduce_pattern='b d ... o i -> b ... o', out_features='None')
    """
    class Params():  # Empty class to keep parameters
        pass

    def __init__(self, layers=[], name:str=None):
        """
        Parameters
        ------------------
        weights : bitpacked uint8 Tensor (or None)
        """
        super().__init__()
        self.layers = nn.ModuleList(layers) if layers != [] else []
        self.parent_path = None
        self._name = name if name is not None else self.__class__.__name__
        self.set_parent_path(None, None)
        self.is_dirty_backward = False
        self.n_backward_runs = 0
        self.params = self.Params()
        # hooks
        self.before_forward_hooks = []
        self.after_forward_hooks = []
        self.before_backward_hooks = []
        self.after_backward_hooks = []
        global enable_call_history
        if enable_call_history:
            global is_update
            self.add_before_forward_hook(lambda x: call_history.enter(f"{'1st' if not is_update else '2nd'} forward : {self._name}"))
            self.add_after_forward_hook(lambda x,o: call_history.exit())
            self.add_before_backward_hook(lambda x: call_history.enter(f"{num2ordinal(self.n_backward_runs)} backward : {self._name}"))
            self.add_after_backward_hook(lambda x,o: call_history.exit())

    def extra_repr(self): return ', '.join([f"{k}='{v}'" for k,v in self.params.__dict__.items()])

    # hooks
    def add_before_forward_hook(self, hook: Callable[[torch.Tensor], None]): 
        assert len(inspect.signature(hook).parameters)==1, "before hook needs 1 parameters"
        self.before_forward_hooks.append(hook)
    def add_after_forward_hook(self, hook: Callable[[torch.Tensor, torch.Tensor], None]):
        assert len(inspect.signature(hook).parameters)==2, "after hook needs 2 parameters"
        self.after_forward_hooks.append(hook)
    def add_before_backward_hook(self, hook: Callable[[torch.Tensor], None]):
        assert len(inspect.signature(hook).parameters)==1, "before hook needs 1 parameters"
        self.before_backward_hooks.append(hook)
    def add_after_backward_hook(self, hook: Callable[[torch.Tensor, torch.Tensor], None]):
        assert len(inspect.signature(hook).parameters)==2, "after hook needs 2 parameters"
        self.after_backward_hooks.append(hook)

    # forward
    def __call__(self, x): return self.forward(x)
    def forward(self, x):
        self.before_forward(x)
        o0 = self.do_forward(x)
        self.after_forward(x, o0)
        return o0
    def before_forward(self, x):  [h(x) for h in self.before_forward_hooks]
    def after_forward(self, x, o0):
        self.n_backward_runs = 0
        [h(x, o0) for h in self.after_forward_hooks]
    def do_forward(self, x): raise NotImplementedError()

    # backward
    def backward(self, backprop_tensor):
        self.before_backward(backprop_tensor)
        o0 = self.do_backward(backprop_tensor)
        self.after_backward(backprop_tensor,o0)
        return o0
    def before_backward(self, backprop_tensor):
        self.n_backward_runs += 1
        [h(backprop_tensor) for h in self.before_backward_hooks]
    def after_backward(self, backprop_tensor, o0):
        self.is_dirty_backward = False
        [h(backprop_tensor, o0) for h in self.after_backward_hooks]
    def do_backward(self, backprop_tensor): raise NotImplementedError()
    
    # path
    @property
    def path(self)->Path:
        # return self._path
        if self.parent_path is None: return Path(self._name)
        else: 
            prefix = f"{self._path_prefix}-" if self._path_prefix != '' else ""
            return self.parent_path+"."+prefix+self._name
    def set_parent_path(self, parent, path_prefix:str):
        if parent is not None: 
            self.parent_path = parent.path
            self._path_prefix = path_prefix
        for i,l in enumerate(self.layers): l.set_parent_path(self, i)

    # temporary tensors (forward/backward)
    @property
    def forward_tensor(self): return self._forward_tensor
    @forward_tensor.setter
    def forward_tensor(self, value): 
        assert (value is None) or (type(value) == torch.Tensor)
        self._forward_tensor = value
    @property
    def backward_tensor(self): return self._backward_tensor
    @backward_tensor.setter
    def backward_tensor(self, value): 
        assert (value is None) or (type(value) == torch.Tensor)
        self._backward_tensor = value
    def set_backward_tensor_for_FBF(self, value):
        """FBF学習のために、Moduleの外部からbackward_tensorを更新するときに使う関数。
        self.layersに子を持つModuleは、is_dirty_backwardの処理を実装する必要があるかも。
        ユースケース : 2nd Forward実行中に逆伝播テンソルが更新するとき、内部の逆伝播テンソルの再計算を予約するときに使う。
        """
        self.backward_tensor = value
        self.is_dirty_backward = True


    def clear_backward_info(self): #🔖不要？
        for l in self.layers: l.clear_backward_info()



class Assert(Module):
    """ 利用側から入出力形状を検証するためのレイヤ
    Usage
    ----------------
    self.binary_net = ln.Sequential([
            ...
            ln.Assert('b d x', b=32, d=3, x=output_dim),
            ...
        ])
    """
    def __init__(self, shape:str, **kwargs):
        super().__init__()
        self.params.shape = shape
        self.each_shape_dict = kwargs
        # self.path = Path(self.__class__.__name__)
    def do_forward(self, x):
        assert_tensor_shape(x, self.params.shape, **self.each_shape_dict)
        self.forward_tensor = x
        return x
    def do_backward(self, flip): 
        x = self.forward_tensor
        assert_tensor_shape(flip, f'n {self.params.shape}', **self.each_shape_dict)
        return flip


# ------------- Container --------------------------------

class Parallel(Module):
    """
    経路の分岐(split)と合流(merge)のあるモジュールのコンテナ。
    2nd forwardでは、まずin_mainの重みを更新し、mergeが適切になるようにin_subの重みを更新する。

    Usage
    ----------------------    
    binary_net = ln.Sequential([
        ...,
        ln.Parallel(
            in_main = None,
            in_sub = ln.BMAB(hidden_features, out_features)
        ),
    ])
    
    """
    # alias for in_main/in_sub
    def __init__(self, in_main, in_sub, 
                 main_rearrange_fw:str=None, main_rearrange_bw:str=None, 
                 sub_rearrange_fw:str=None, sub_rearrange_bw:str=None, 
                 name='Parallel', merge_operation='xnor', enable_log=True):
        """
        merge_operation:str - xnor,and,or
        """
        super().__init__(
            layers=[in_main if in_main is not None else Identity(),
                    in_sub if in_sub is not None else Identity()],  # Don't use alias in __init__. (to avoid error)
            name=name,
        )
        self.main_rearrange_fw = main_rearrange_fw
        self.main_rearrange_bw = main_rearrange_bw
        self.sub_rearrange_fw = sub_rearrange_fw
        self.sub_rearrange_bw = sub_rearrange_bw
        self.merge_operation = merge_operation
        self.enable_log = enable_log

    @property
    def in_main(self): return self.layers[0]
    @in_main.setter
    def in_main(self, value:Module): self.layers[0] = value
    @property
    def in_sub(self): return self.layers[1]
    @in_sub.setter
    def in_sub(self, value:Module): self.layers[1] = value
    def extra_repr(self):
        return f"merge_operation={self.merge_operation}"


    def do_forward(self, x):
        global is_update
        if is_update:
            assert self.backward_tensor is not None
            assert self.h0_old is not None
            assert self.h1_old is not None
            flip = self.backward_tensor
        # split ---------------------------------------------------------------
        # main
        if is_update and self.is_dirty_backward:
            # update backward_tensor for main
            if self.merge_operation=='xnor': flip_h0 = backward_xnor(flip)
            elif self.merge_operation=='and': flip_h0 = backward_and(flip, self.h0_old, self.h1_old)
            elif self.merge_operation=='or': flip_h0 = backward_or(flip, self.h0_old, self.h1_old)
            else: raise NotImplementedError()
            self.in_main.set_backward_tensor_for_FBF(rearrange(flip_h0, self.main_rearrange_bw))
            self.is_dirty_backward = False
        h0_new = self.in_main(x)  # 2nd Forward
        if (not is_update) and self.enable_log: log_forward_out(self.in_main, h0_new)
        h0_new = rearrange(h0_new, self.main_rearrange_fw)
        # sub
        if is_update:
            if self.merge_operation=='xnor':
                flip_h1 = backward_updated_xnor(flip, h0_new, self.h0_old)
            elif self.merge_operation in ['and', 'or']: 
                flip_h1 = backward_and(flip, self.h1_old, h0_new)
            else: raise NotImplementedError()
            self.in_sub.set_backward_tensor_for_FBF(rearrange(flip_h1, self.sub_rearrange_bw))
            self.h0_old, self.h1_old = None, None
        h1_new = self.in_sub(x).to(x.device)  # 2nd Forward.  to(x.device):case when BinaryTensor
        if (not is_update) and self.enable_log: log_forward_out(self.in_sub, h1_new)
        h1_new = rearrange(h1_new, self.sub_rearrange_fw)
        # merge ---------------------------------------------------------------
        if self.merge_operation=='xnor': o0 = bitwise_xnor(h0_new, h1_new)
        elif self.merge_operation=='and': o0 = bitwise_and(h0_new, h1_new)
        elif self.merge_operation=='or': o0 = bitwise_or(h0_new, h1_new)
        else: raise NotImplementedError()
        self.h0_old = h0_new
        self.h1_old = h1_new
        return o0

    def do_backward(self, flip):
        h0 = self.h0_old
        h1 = self.h1_old
        self.backward_tensor = flip
        # merge
        if self.merge_operation=='xnor': flip_h0, flip_h1 = backward_xnor(flip), backward_xnor(flip)
        elif self.merge_operation=='and': flip_h0, flip_h1 = backward_and(flip, h0, h1), backward_and(flip, h1, h0)
        elif self.merge_operation=='or': flip_h0, flip_h1 = backward_or(flip, h0, h1), backward_or(flip, h1, h0)
        else: raise NotImplementedError()
        # rearrange
        flip_h0 = rearrange(flip_h0, self.main_rearrange_bw)
        flip_h1 = rearrange(flip_h1, self.sub_rearrange_bw)
        # main
        if self.enable_log: log_backward_in(self.in_main, flip_h0)
        flip_x_h0 = self.in_main.backward(flip_h0)
        if self.enable_log: log_backward_out(self.in_main, flip_x_h0)
        # sub
        if self.enable_log: log_backward_in(self.in_sub, flip_h1)
        flip_x_h1 = self.in_sub.backward(flip_h1)
        # if self.merge_operation=='xnor': self.in_sub.clear_backward_info()
        if self.enable_log: log_backward_out(self.in_sub, flip_x_h1)
        # split
        if flip_x_h1 is not None:
            global update_rate
            flip_x_h0 = reduce_flip(flip_x_h0, 'n ... -> ...', update_rate) if flip_x_h0.shape[0] != 1 else rearrange(flip_x_h0, 'n ... -> ...')
            flip_x_h1 = reduce_flip(flip_x_h1, 'n ... -> ...', update_rate) if flip_x_h1.shape[0] != 1 else rearrange(flip_x_h1, 'n ... -> ...')
            flip_x = bitwise_or(flip_x_h0, flip_x_h1)
            return rearrange(flip_x, '... -> () ...') # n...
        else:    # Case where in_sub is BinaryTensor etc...
            return flip_x_h0

class Serial(Module):
    """
    直列なモジュールのコンテナ。
    2nd forwardでは、まず内部の各モジュールの逆伝播テンソルを更新し、その後forwardを実行する。

    Usage
    ----------------------    
    binary_net = ln.Sequential([
        ...,
        ln.Serial(
            [
                ln.BMAB(hidden_features, out_features),
                ...
            ]
        ),
    ])
    
    """
    def __init__(self, layers, name='Serial', enable_log=True):
        super().__init__(layers, name=name)
        self.enable_log = enable_log

    def do_forward(self, x):
        global is_update
        if is_update:
            assert self.backward_tensor is not None
            flip = self.backward_tensor
            if self.is_dirty_backward:
                # 2nd Backward - update each layer's backward_tensor
                for l in reversed(self.layers[1:]):
                    flip = l.backward(flip)
                self.layers[0].set_backward_tensor_for_FBF(flip)
            # 2nd Forward
            o0 = self._forward(x)
        else: o0 = self._forward(x)
        return o0
    
    def _forward(self, x0):
        if self.enable_log: log_forward_in(self.layers[0], x0)
        for layer in self.layers:
            x0 = layer(x0)
            if self.enable_log: log_forward_out(layer, x0)
        return x0


    def do_backward(self, backprop_tensor):
        self.backward_tensor = backprop_tensor
        # if self.enable_log: log_backward_in(self.layers[-1], backprop_tensor)
        for layer in reversed(self.layers):
            if self.enable_log: log_backward_in(layer, backprop_tensor)
            backprop_tensor = layer.backward(backprop_tensor)
            if self.enable_log: log_backward_out(layer, backprop_tensor)
        return backprop_tensor

# ------------------ Modules ----------------------------------------------------------------



class Rearrange(Module):
    "rearrange Module"
    def __init__(self, ptn_forward, ptn_backward):
        """
        Usage
        --------------------
        ln.Serial([
            ...,
            ln.Rearrange(ptn_forward = 'b d ... i -> b d ... () i', 
                          ptn_backward = 'n b d ... o i -> (n o) b d ... i')
        ])
        """
        super().__init__()
        self.params.ptn_forward = ptn_forward
        self.params.ptn_backward = ptn_backward
    def do_forward(self, x): return rearrange(x, self.params.ptn_forward)
    def do_backward(self, flip): return rearrange(flip, self.params.ptn_backward)
    def extra_repr(self):
        return f"ptn_forward='{self.params.ptn_forward}', ptn_backward='{self.params.ptn_backward}'"


class PcntNormBin(Module):
    "popcount_normalize_binarize Module"
    def __init__(self, depth_ths:torch.Tensor, pcnt_reduce_pattern:str, out_features:int=None):
        """
        out_features:int - ビットパッキング前のo軸の次元。8の倍数ならNoneでよい（padding発生しないので）
                            ※'b d o i -> b o'のようなpcnt_reduce_patternが想定されている。
        """
        super().__init__()
        assert type(depth_ths) == torch.Tensor, "depth_ths must be Tensor"
        self.params.depth_ths = depth_ths
        self.params.pcnt_reduce_pattern = pcnt_reduce_pattern
        self.params.out_features = out_features
        # self.path = Path(self.__class__.__name__)
    def do_forward(self, x):
        assert_tensor_shape(x, 'b d ... x')
        o0 = popcount_normalize_binarize(x, self.params.depth_ths, self.params.pcnt_reduce_pattern)
        assert_tensor_shape(o0, 'b d ... x', d=self.params.depth_ths.numel())
        self.forward_tensor = x
        return o0
    def do_backward(self, flip):
        x = self.forward_tensor
        # validate input
        assert_tensor_shape(flip, 'n b d ... x', d=self.params.depth_ths.numel())  
        # backward
        flip_out = backward_popcount_normalize_binarize(flip, x, self.params.depth_ths, 
                                                        self.params.pcnt_reduce_pattern, 
                                                        flip_lowest_features=self.params.out_features)
        # validate output
        assert_tensor_shape(flip_out, 'n b d ... x')
        assert x.shape == flip_out.shape[1:], f'x & flip shape not match. expected:{x.shape}, actual:{flip_out.shape[1:]}'
        return flip_out

class PcntNorm(Module):
    "popcount_normalize Module"
    def __init__(self, pcnt_reduce_pattern:str, out_features:int=None):
        super().__init__()
        self.pcnt_reduce_pattern = pcnt_reduce_pattern
        self.params.out_features = out_features
        # self.path = Path(self.__class__.__name__)
    def do_forward(self, x): 
        assert_tensor_shape(x, 'b ... x')  # validate input
        o0 = popcount_normalize(x, self.pcnt_reduce_pattern, self.params.out_features)
        # validate output
        if self.params.out_features is not None: assert_tensor_shape(o0, 'b ... x', x=self.params.out_features)
        else : assert_tensor_shape(o0, 'b ... x')
        self.forward_tensor = x
        return o0
    def do_backward(self, grad): 
        x = self.forward_tensor
        # validate input
        if self.params.out_features is not None: assert_tensor_shape(grad, 'b ... x', x=self.params.out_features)
        else: assert_tensor_shape(grad, 'b ... x')
        flip = backward_popcount_normalize(restore_reduced_axis(grad, self.pcnt_reduce_pattern), x)
        # validate output
        assert x.shape == flip.shape[1:], f'x & flip shape not match. expected:{x.shape}, actual:{flip.shape[1:]}'
        assert_tensor_shape(flip, 'n b d ... x')
        return flip


class BinaryTensor(Module):
    "Binary Tensor Module class"
    def __init__(self, shape, weights=None, vote_p_min=None):
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
        self.n_votes = 0
        self.vote_p_min = vote_p_min
        # self.path = Path(self.__class__.__name__)
        
    def extra_repr(self):
        return f"weights={self.shape}, dtype=uint8"
    def has_accumulator(self):
        return hasattr(self, 'vote_accumulator') and self.vote_accumulator is not None

    def update(self):
        if not self.has_accumulator() or not self.training: return
        # calculate update mask
        logger.log_hist(f"{self.parent_path}/{self.path.stem}-vote_accumulator", self.vote_accumulator)
        global update_rate
        update_percentile = 1-update_rate
        global disrupt_ratio
        disrupt_percentile = update_rate*disrupt_ratio
        update_mask, disrupt_mask = vote2flip(self.vote_accumulator, self.n_votes, update_percentile, disrupt_percentile, p_min_upper=self.vote_p_min)
        # calculate suppressor_mask
        global suppressor_rate
        if suppressor_rate != 0:
            permission_mask_update = build_randmask(update_mask.shape, 1-suppressor_rate)
            update_mask &= permission_mask_update.to(update_mask.device)
            if disrupt_ratio !=0:
                permission_mask_disrupt = build_randmask(disrupt_mask.shape, 1-suppressor_rate)
                disrupt_mask &= permission_mask_disrupt.to(disrupt_mask.device)
        # from IPython.core.debugger import Pdb; Pdb().set_trace()
        # update weights
        self.weights.data = self.weights.to(update_mask.device)
        self.weights.data = bitwise_xor(self.weights, update_mask)  # update weights
        if disrupt_ratio!=0:
            self.weights.data = bitwise_xor(self.weights, disrupt_mask)  # disrupt weights
        logger.log_1ratio(f"{self.parent_path}/{self.path.stem}-update_ratio", update_mask)
        logger.log_heatmap(f"{self.parent_path}/{self.path.stem}-weights", unpackbits(self.weights.flatten()))
        # if 'BMAB' in str(self.parent_path): from IPython.core.debugger import Pdb; Pdb().set_trace()
        # Clear memory
        self.clear_backward_info()
    
    def clear_backward_info(self):
        self.vote_accumulator = None
        self.n_votes = 0
        return super().clear_backward_info()

    def do_forward(self, x=None):
        global is_update
        if is_update: 
            assert self.backward_tensor is not None
            flip = self.backward_tensor
            self.backward_tensor = None
            self.clear_backward_info()
            self.backward(flip)
            self.update()
        return self.weights.data

    def do_backward(self, flip):
        """
        Parameters
        ------------------
        flip:Tensor(packed uint8)
        """
        self.backward_tensor = flip
        global is_update
        if not is_update: return None  # Do nothing when 1st Backward
        # Validate
        assert_tensor_shape(flip, 'n ...')  # ex. ndoi
        # assert  flip.shape[1:] == self.weights.data.shape, f'BinaryTensor backward input axis not match. actual:{flip.shape[1:]}, expected:{self.weights.data.shape}'
        # backward
        if not self.has_accumulator():
            self.vote_accumulator = unpackbits(self.weights.data).type(torch.int32).to(flip.device)
            self.vote_accumulator[:] = 0
            self.n_votes = 0
        if self.weights.data.shape[0]==1: # when d weights shared
            flip = rearrange(flip, 'n d ... -> (n d) () ...') # d軸の要素をn軸へ移動し、d=1とする
        self.vote_accumulator += reduce(unpackbits(flip), 'n ... -> ...', 'sum') #ex. doi
        self.n_votes += flip.shape[0]
        return None

class Binarize(Module):
    """
    FP32入力をBinarizeするレイヤ

    期待する入力： FP32 tensor (normalized: mean 0, std 1)
    出力： uint8 tensor. axis:b,d,x(batch, depth, (inputs))
    """
    def __init__(self, depth_ths:Union[list,torch.Tensor]=TH_DEPTH3):
        """
        depth_ths  list of threshold values. ex) [0] -> 1bit, [-1,0,1] -> 2bit
        """
        super().__init__()
        if isinstance(depth_ths, list): depth_ths = rearrange(torch.tensor(depth_ths), 'd -> () d ()')  # bdx
        self.params.depth_ths = depth_ths
        self.n_in = None  # initialize when first input

    def do_forward(self, x):
        if self.n_in is None: self.n_in = x.shape[-1]
        assert_tensor_shape(x, 'b x', x=self.n_in)
        # forward
        x0 = rearrange(x, 'b x -> b () x') 
        h0 = binarize(x0, self.params.depth_ths)
        self.forward_tensor = x
        assert_tensor_shape(h0, 'b d x', d=self.params.depth_ths.numel(), x=calc_packed_features(self.n_in))
        return h0
    def do_backward(self, flip):
        x = self.forward_tensor
        assert_tensor_shape(flip, 'n b d x', d=self.params.depth_ths.numel(), x=calc_packed_features(self.n_in))
        x_grad = backward_binarize(flip, x, self.params.depth_ths)
        return x_grad[..., :self.n_in]  # Drop the padded values when bitpacking.

class ApplyWeight(Parallel):
    def __init__(self, merge_operation, shape, weights=None, name=None):
        """
        Parameters
        ------------------
        shape : tuple - Shape of Weights. Must be added o-axis, removed b-axis from input.
                ex. input:'b d i' -> weight:'d o i'
        weights : uint8 Tensor (or None)
        """
        super().__init__(in_main=None, 
                         in_sub=BinaryTensor(shape=shape, weights=weights),
                         main_rearrange_fw='... i -> ... () i',   # add o axis
                         main_rearrange_bw='n ... o i -> (n o) ... i',  # o to n
                         sub_rearrange_fw='... -> () ...',  # add b axis
                         sub_rearrange_bw='n b ... -> (n b) ...', # b to n
                         name=f"{self.__class__.__name__}-{merge_operation}" if name is None else name,
                         merge_operation=merge_operation,
                         enable_log=False)
        self.shape = shape
        # self.enable_log = enable_log
    def do_forward(self,x):
        x = super().do_forward(x)
        # if self.enable_log: log_forward_out(self, x)
        return x
    def do_backward(self, flip):
        flip = super().do_backward(flip)
        # if self.enable_log: log_backward_out(self, flip)
        return flip

class XnorWeight(ApplyWeight):
    def __init__(self, shape, weights=None):
        """
        Parameters
        ------------------
        shape : tuple - each axis numbers of bits
        weights : uint8 Tensor (or None)
        """
        super().__init__(shape=shape, 
                         weights=weights,
                         name=self.__class__.__name__, 
                         merge_operation='xnor')

class BMA(Serial):
    """Binary Multiply-Accumulate Layer Module class
    
    期待する入力： packed uint8 tensor
    出力：FP32 (normalized mean0, std1)
    """
    def __init__(self, in_features:int, out_features:int, depth_in_features:int=1, weights=None):
        """
        Parameters
        ------------------
        in_features:int  - numbers of input bits, and weight i axis features
        out_features:int - numbers of output bits, and weight o axis features
        depth_features:int - weight d axis features. if 1, broadcast to len(depth_ths)
        depth_ths:list - thresholds of depth
        weights : uint8 Tensor (or None)
        """
        super().__init__(layers=[
                XnorWeight(shape=(depth_in_features, out_features, calc_packed_features(in_features)), weights=weights),
                PcntNorm('b d ... o i -> b ... o'),
            ], name=self.__class__.__name__,
            enable_log=False)        
        self.params.in_features = in_features
        self.params.out_features = out_features
        self.params.depth_in_features = depth_in_features

    def do_forward(self, x):
        assert_tensor_shape(x, 'b d ... i', i=calc_packed_features(self.params.in_features))
        o1 = super().do_forward(x)
        assert_tensor_shape(o1, 'b ... o', o=self.params.out_features)
        return o1

    def do_backward(self, grad):
        assert_tensor_shape(grad, 'b ... o', o=self.params.out_features)
        x_flip = super().do_backward(grad)
        assert_tensor_shape(x_flip, 'n b d ... i', i=calc_packed_features(self.params.in_features))
        return x_flip

    @property
    def weights(self): return self.layers[0].layers[1]

class BMAB(Serial):
    """Binary Matrix-Accumulate Binarize Layer Module class
    期待する入力: packed uint8 tensor
    出力: packed uint8 tensor
    """
    def __init__(self, in_features:int, out_features:int, depth_in_features:int=1, depth_out_ths:list=TH_DEPTH1, weights=None):
        """
        Parameters
        ------------------
        in_features:int  - numbers of input bits, and weight i axis features
        out_features:int - numbers of output bits, and weight o axis features
        depth_features:int - weight d axis features. if 1, broadcast to len(depth_ths)
        depth_ths:list - thresholds of depth
        weights : uint8 Tensor (or None)
        """
        if out_features%8 !=0: raise ValueError(f'out_features must be a multiple of 8 to bitpack. actual:{out_features}')
        # if in_features%8 !=0:
        #     in_features_old = in_features
        #     in_features = calc_packed_features_x8(in_features)
        #     warnings.warn(f'{self.path}: in_features is changed to fit bitpack. {in_features_old} -> {in_features}')
        depth_out_ths = rearrange(torch.tensor(depth_out_ths), 'd -> () d ()')  # bdo
        super().__init__(layers=[
            XnorWeight(shape=( depth_in_features, out_features, calc_packed_features(in_features)), weights=weights),
            PcntNormBin(depth_out_ths, 'b d ... o i -> b ... o'),
            ],
            name=self.__class__.__name__, enable_log=False)
        self.params.depth_out_ths = depth_out_ths
        self.params.in_features = in_features
        self.params.out_features = out_features
        self.params.depth_in_features = depth_in_features

    def do_forward(self, x):
        assert_tensor_shape(x, 'b d ... i', i=calc_packed_features(self.params.in_features))
        o0 = super().do_forward(x)
        assert_tensor_shape(o0, 'b d ... o', d=self.params.depth_out_ths.numel(), o=calc_packed_features(self.params.out_features))
        return o0

    def do_backward(self, flip):
        assert_tensor_shape(flip, 'n b d ... o', d=self.params.depth_out_ths.numel(), o=calc_packed_features(self.params.out_features))
        x_flip = super().do_backward(flip)
        assert_tensor_shape(x_flip, 'n b d ... i', i=calc_packed_features(self.params.in_features))
        return x_flip

    @property
    def weights(self): return self.layers[0].layers[1]

class Identity(Module):
    "恒等変換"
    def __init__(self):
        super().__init__()
    def do_forward(self, x): return x
    def do_backward(self, flip, tensors=None): return flip

class Dropout(Module):
    "Dropout"
    def __init__(self, p=0.25):
        """
        p:float - the probability of flipping
        """
        super().__init__()
        self.p = p
    def do_forward(self, x): return bitwise_xor(x, build_randmask(x.shape, self.p).to(x.device)) if self.p!=0 else x
    def do_backward(self, flip, tensors=None): return flip

# ---------- Sequential ---------------------
class Sequential(Serial):
    """

    Usage
    ----------------------    
    binary_net = ln.Sequential([
        ln.Binarize(depth_ths=[-1,0,1]),
        ln.BMA(hidden_features, out_features),]
    )
    
    """
    def __init__(self, layers:list, name='Seq', enable_log=True):
        super().__init__(layers, name=name, enable_log=enable_log)
        self.fn = self.SequentialFunction.apply
        self.enable_log = enable_log

    def forward(self, x): return self.fn(x, self)
    def super_forward(self, x): return super().forward(x)
    def super_backward(self, grad): return super().backward(grad)

    class SequentialFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x0, model):
            ctx.model = model
            return model.super_forward(x0)

        @staticmethod
        def backward(ctx, backprop_tensor:torch.Tensor):
            """
            backprop_tensor : flip or grad
            """
            model = ctx.model
            return model.super_backward(backprop_tensor), None


