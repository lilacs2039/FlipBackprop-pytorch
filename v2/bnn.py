
from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
# from fastai.callback.core import Callback

from functools import partial
import numpy as np
import wandb
import sys

# ---------------- Constants --------------------------------
# Thresholds which equally split normal distribution
TH_DEPTH15 = [-1.53, -1.15, -0.89, -0.67, -0.49, -0.32, -0.16, 0.00, 0.16, 0.32, 0.49, 0.67, 0.89, 1.15, 1.53]
TH_DEPTH7 = [-1.15, -0.67, -0.32, 0.00, 0.32, 0.67, 1.15]
TH_DEPTH3 = [-0.67, 0.00, 0.67]
TH_DEPTH1 = [0.00]

# ----------------- Global Variables --------------------------------
is_update = False
# grad2flip_eps=1e-3  # grad2flipで、勾配=0を判定するしきい値
update_permission_coef = 0  # ビット更新の許可度  0:allow all(100%), 1:50%, 2:25%, ...

# ---------- debug functions --------------------------------
def debug(*mes): print("D> ",mes)
def a2bstr(arr):  # arr:list,torch.Tensor
    if type(arr) == torch.Tensor: arr = arr.tolist()
    arr = np.array(arr)
    return np.vectorize(lambda x:f"{int(x):08b}")(arr).tolist()

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

def build_reverse_pattern(reduce_pattern:str):
    ptn_in, ptn_out = reduce_pattern.split('->')
    out_ids = [s for s in ptn_out.split(' ') if s != '']
    ptn_restored = ' '.join([s if s in out_ids else '()' for s in ptn_in.split(' ') if s != ''])
    restore_pattern = f"{ptn_out} -> {ptn_restored}"
    return restore_pattern
assert build_reverse_pattern('b d o i -> b o')==' b o -> b () o ()'

def restore_reduced_axis(x_reduced:torch.Tensor, reduce_pattern:str):
    "reduceで削除した軸を挿入して、reduce前の軸の数に回復したテンソルを返す。"
    return rearrange(x_reduced, build_reverse_pattern(reduce_pattern))
assert(list(restore_reduced_axis(torch.rand((2,3)), 'b d o i -> b o').shape)==[2, 1, 3, 1])

def count_votes(flip:torch.Tensor, vote:torch.Tensor):
    "flipを集計したvoteの集計ビット数を返す。"
    assert flip.dtype == torch.uint8, "flip must be torch.uint8 Tensor"
    return (flip.numel()*8) // vote.numel()

def calc_packed_features(features:int):
    "ビットパックしたときの要素数を返す"
    return int(np.ceil(features/8))
assert calc_packed_features(8)==1
assert calc_packed_features(9)==2

def vote2flip(votes:torch.Tensor, n_votes:int, vote_p_max:float):
    "反転投票を集計・二値化した結果のflipを返す"
    mean = votes.type(torch.float32).mean()
    p = mean/n_votes
    p = max(vote_p_max, p)
    flip = packbits(votes > p*n_votes)  # flip if larger than mean (of binomial distribution with probability p)
    return flip

def reduce_flip(flip:torch.Tensor, reduce_pattern:str, vote_p_max:float):
    "flipを集計・二値化してreduce処理する。"
    vote  = reduce(unpackbits(flip), reduce_pattern, 'sum').type(torch.int32)
    n_votes = count_votes(flip, vote)
    flip_ret = vote2flip(vote, n_votes, vote_p_max=vote_p_max)
    return flip_ret

def flip2grad(bin:torch.Tensor, flip:torch.Tensor):
    sign = unpackbits(bin)*2-1
    x_grad = reduce(sign*unpackbits(flip), 'b d x -> b x', 'sum').type(torch.float32)
    std = x_grad.std()
    if not std.isnan() and std !=0:  # Skip when all values of x_grad are 0 and std is nan
        x_grad /= std  # 標準偏差が１になるように標準化。期待平均値は0のまま変わらず。
    return x_grad

def grad2flip(grad, x):
    assert grad.dtype == torch.float32
    assert x.dtype == torch.uint8
    flip = bitwise_xor(x, 
        packbits(repeat(grad<0, 'b d o i -> b d o (repeat i)', repeat=8)))
    flip = bitwise_and(flip,
        packbits(repeat(grad!=0, 'b d o i -> b d o (repeat i)', repeat=8)))
    return flip

def bool2bitpacked(t:torch.Tensor) -> torch.Tensor:
    assert t.dtype == torch.bool
    return packbits(repeat(t, '... x -> ... (repeat x)', repeat=8))

def prep_bitwise(*tensors:Union[torch.Tensor, bool]) -> list:
    "bitwise演算できるようにテンソルを前処理"
    ret = []
    for t in tensors:
        if isinstance(t, bool): t = torch.tensor([t])
        if t.dtype == torch.bool: t = bool2bitpacked(t)
        assert t.dtype == torch.uint8
        ret.append(t)
    return ret

# ---------- Operation functions ------------------------
def bitwise_xnor(v1:Union[torch.Tensor, bool],v2:Union[torch.Tensor, bool]):
    "２つのベクトルのxnor演算（ビットが同じなら１）を行う。"
    v1, v2 = prep_bitwise(v1, v2)
    return torch.bitwise_not(torch.bitwise_xor(v1, v2))
v1 = torch.tensor([0b0001, 0b0001], dtype=torch.uint8)
v2 = torch.tensor([0b0011, 0b0001], dtype=torch.uint8)
assert a2bstr(bitwise_xnor(v1,v2))==['11111101', '11111111']
assert tuple(bitwise_xnor(True, torch.randn((2,3,4)).type(torch.uint8)).shape)==(2,3,4), "ブロードキャストしてビット反転"

def bitwise_xor(v1:Union[torch.Tensor, bool],v2:Union[torch.Tensor, bool]):
    v1, v2 = prep_bitwise(v1, v2)
    return torch.bitwise_xor(v1, v2)
def bitwise_and(v1:Union[torch.Tensor, bool],v2:Union[torch.Tensor, bool]):
    v1, v2 = prep_bitwise(v1, v2)
    return torch.bitwise_and(v1, v2)


def popcount(arr:torch.Tensor, dim=-1):
    """popcount（1のビッチの数）を数える。arrはuint8のみ。
    Parameters
    -----------
    arr
    dim:int - 集計対象の軸。None:全軸で集計して0次テンソルを返す
    """
    assert type(arr) == torch.Tensor, "input must be torch.Tensor"
    assert arr.dtype == torch.uint8, "dtype must be torch.uint8"
    return torch.count_nonzero(unpackbits(arr), dim=dim)
assert popcount(torch.tensor([0b11111101, 0b11111111], dtype=torch.uint8))==15
assert list(popcount(torch.randn((32,8,3)).type(torch.uint8)).shape) == [32,8]
assert popcount(torch.tensor([[7,7],[7,7]], dtype=torch.uint8), dim=None).tolist() == 12

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

def normalize_popcount(tensor:torch.Tensor, n:int, p:float=0.5)->torch.Tensor:
    """二項分布（binomial distribution）を、確率pの平均・標準偏差で標準化
    期待平均 0, 期待標準偏差 1
    Parameters
    ----------
    n:int      dimension of binomial distribution. (usualy, total number of bits)
    p:float    
    """
    mean = p*n
    std = torch.tensor(n*p*(1-p)).sqrt()
    return ((tensor.type(torch.float32)-mean)/std)
assert tuple(normalize_popcount(torch.randint(0,32,(4,8)), 32).shape) == (4,8)

def binarize(x:torch.Tensor, depth_ths:torch.Tensor):
    return packbits(x>depth_ths.to(x.device)).clone().detach()

def backward_binarize(flip:torch.Tensor, x:torch.Tensor, depth_ths:torch.Tensor):
    x = rearrange(x, 'b ... -> b () ...') #bdx
    bin = binarize(x, depth_ths)
    return flip2grad(bin, flip)

def popcount_normalize(x:torch.Tensor, reduce_pattern:str):
    # Validate
    assert type(x) == torch.Tensor, "input must be torch.Tensor"
    assert x.dtype == torch.uint8, "dtype must be torch.uint8"
    # popcount
    o0 = reduce(unpackbits(x), reduce_pattern, 'sum').type(torch.int32)
    # normalize
    n_all = count_votes(x, o0)
    o1 = normalize_popcount(o0, n_all, p=0.5)
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
    n0 = popcount_normalize(x, pcnt_reduce_pattern)
    # binarize
    n0 = rearrange(n0,'b ... -> b () ...') # bd...
    b0 = binarize(n0, depth_ths)
    return b0

def backward_popcount_normalize_binarize(flip, x, depth_ths:torch.Tensor, pcnt_reduce_pattern:str):
    """binarize-popcountのbackwardを一緒に実行して無駄な演算を省く
    （標準化しない、float32ではなくint8で計算）
    """
    # backward binarize
    bin = popcount_normalize_binarize(x,depth_ths,pcnt_reduce_pattern)
    sign = unpackbits(bin)*2-1
    grad = reduce(sign*unpackbits(flip), 'b d ... -> b ...', 'sum').type(torch.int8)  # int8:depthは127より小さいので
    grad = restore_reduced_axis(grad, pcnt_reduce_pattern)  # bdoi
    # backward popcount
    flip_out = bitwise_xor(x, grad<0)
    flip_out = bitwise_and(flip_out, grad!=0)
    return flip_out

# ---------------- Evaluation functions ----------------------
def calc_1ratio(x:torch.Tensor)->float:
    """1のビットの割合を求める
    用途：flipに対して - 反転する割合（Trueの数 / 全要素数）を返す
    ビットバランス - 0.5なら均衡、1なら全部True
    """
    sum_bits = x.numel()*8
    return float(popcount(x, dim=None)/sum_bits)
assert calc_1ratio(torch.tensor([[7,7],[7,7]], dtype=torch.uint8)) == 0.375


# ------------------- Utils ----------------
class WandbLogger():
    """wandb用のログを蓄積するクラス。
    fastai Callbackでwandb処理を行う想定。
    """
    _dict = {}
    _heatmap = {}  # k:title, v:[{values=[], vmax=1,vmin=0, ...]
    def log(self, dict):
        """
        Usage
        ---------
        log float  : logger.log({"BinaryTensor flip_ratio": floatValue})
        log tensor : logger.log({"x_grad": wandb.Histogram(x_grad.cpu().numpy())})
        """
        self._dict.update(dict)
    def clear(self): self._dict.clear()
    def getLog(self): return self._dict
    # def heatmap(self, title_contents_dict:dict): 
    def heatmap(self, title, values, vmax=1, vmin=0): 
        "一ステップ分の１次元テンソルを記録。終了時にheatmapとしてwandbに保存できるように情報を保持。"
        if title not in self._heatmap: self._heatmap[title] = {'values':[]}
        self._heatmap[title]['values'].append(values) #, 'vmax':vmax,'vmin':vmin})
        self._heatmap[title]['vmax'] = vmax
        self._heatmap[title]['vmin'] = vmin
    def getHeatmap(self): return self._heatmap
    def clear_heatmap(self): self._heatmap.clear()
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


logger = WandbLogger()

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

# ------------------- BNN Modules ----------------
class Module(nn.Module):
    def __init__(self):
        """
        Parameters
        ------------------
        weights : bitpacked uint8 Tensor (or None)
        """
        super().__init__()
        self.backward_tensor = None
        self.x_old = None
    @property
    def path(self): return self._path      # getter
    @path.setter
    def path(self, value): 
        self._path = value     # setter

class PNB(Module):
    "popcount_normalize_binarize Module"
    def __init__(self, depth_ths:torch.Tensor, pcnt_reduce_pattern:str):
        self.depth_ths = depth_ths
        self.pcnt_reduce_pattern = pcnt_reduce_pattern
        self.path = Path(self.__class__.__name__)

    def forward(self, x):
        o0 = popcount_normalize_binarize(x, self.depth_ths, self.pcnt_reduce_pattern)
        return o0

    def backward(self, flip, x):
        x_flip = backward_popcount_normalize_binarize(flip, x, self.depth_ths, self.pcnt_reduce_pattern)
        return x_flip

    def __call__(self, x): return self.forward(x)

class BinaryTensor(Module):
    "Binary Tensor Module class"
    def __init__(self, shape, weights=None, vote_p_max=0.02):
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
        self.vote_p_max = vote_p_max
        self.path = Path(self.__class__.__name__)
        
    def __repr__(self):
        return f"BinaryTensor(weights={self.shape}, dtype=uint8)"    
    def has_accumulator(self):
        return hasattr(self, 'vote_accumulator') and self.vote_accumulator is not None

    def update(self):
        if not self.has_accumulator(): return
        # calculate update mask
        logger.log({f"{self.path.parents[1]}/{self.path.stem}-vote_accumulator": wandb.Histogram(self.vote_accumulator.cpu().numpy())})
        update_mask = vote2flip(self.vote_accumulator, self.n_votes, self.vote_p_max)
        # calculate suppressor_mask
        global update_permission_coef
        if update_permission_coef > 0:
            times_rand = update_permission_coef
            suppressor_mask = torch.ones(update_mask.shape, dtype=torch.uint8)*255  # if 1 suppress update
            for j in range(times_rand): suppressor_mask &= torch.randint(0,256, update_mask.shape, dtype=torch.uint8)
            permission_mask = 255 ^ suppressor_mask  # if 1 allow update. when 2: 25% -> 75%
            update_mask &= permission_mask.to(update_mask.device)
        # update weights
        self.weights.data = self.weights.to(update_mask.device)
        self.weights.data = bitwise_xnor(self.weights, update_mask)  # update weights
        logger.log({f"{self.path.parents[1]}/{self.path.stem}-update_ratio": calc_1ratio(update_mask)})
        logger.heatmap(f"{self.path.parents[1]}/{self.path.stem}-weights", unpackbits(self.weights.flatten()).tolist())
        # Clear memory
        del self.vote_accumulator  
        self.vote_accumulator = None
        self.n_votes = 0

    def forward(self):
        return self.weights

    def backward(self, flip):
        """
        Parameters
        ------------------
        flip:Tensor(packed uint8)
        """
        # Validate
        assert len(flip.shape) == len(self.weights.data.shape), f'BinaryTensor backward input axis not match. actual:{len(flip.shape)}, expected:{len(self.weights.data.shape)}'
        # backward
        if not self.has_accumulator():
            self.vote_accumulator = unpackbits(flip).type(torch.int32).to(flip.device)
            self.vote_accumulator[:] = 0
            self.n_votes = 0
        self.vote_accumulator += unpackbits(flip)  # o,i
        self.n_votes += 1
        return None        

    def __call__(self): return self.forward()


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
        self.depth_ths = depth_ths
        self.n_in = None
        self.path = Path(self.__class__.__name__)

    def forward(self, x):
        if self.n_in is None: self.n_in = x.shape[-1]
        x = rearrange(x, 'b x -> b () x') 
        h0 = binarize(x, self.depth_ths)
        return h0
    def backward(self, flip, x):
        x_grad = backward_binarize(flip, x, self.depth_ths)
        return x_grad[..., :self.n_in]  # Drop the padded values when bitpacking.
    def __call__(self, x): return self.forward(x)

class XnorWeight(Module):
    """
    """
    def __init__(self, in_features:int=1, out_features:int=1, depth_features:int=1, weights=None):
        """
        Parameters
        ------------------
        in_features:int  - numbers of input bits
        out_features:int - numbers of output bits
        depth_features:int - numbers of bit depth
        weights : uint8 Tensor (or None)
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.depth_features = depth_features
        in_features = calc_packed_features(in_features)
        self.weights = BinaryTensor(shape=(depth_features, out_features, in_features), weights=weights)
        self.path = Path(self.__class__.__name__)

    def forward(self, x):
        global is_update
        if is_update and self.backward_tensor is not None:
            h_flip = self.backward_tensor
            # Calculate w flip
            h_old = self._forward(self.x_old)
            h_new = self._forward(x)
            h_flipped = bitwise_xor(h_old, h_new)
            w_flip = bitwise_xor(h_flip, h_flipped)
            w_flip = reduce_flip(h_flip, 'b d o i -> d o i', vote_p_max=0.5)
            # Update weights
            self.weights.backward(w_flip)
            self.weights.update()
            self.backward_tensor, self.x_old = None, None

        h0 = self._forward(x)  # bdoi
        return h0

    def _forward(self, x):
        # Validate
        b,d,i = x.shape
        assert i==calc_packed_features(self.in_features),f'output in axis not match. actual:{i}, expected:{calc_packed_features(self.in_features)}'
        # forward
        x = rearrange(x, 'b d i -> b d () i')
        w = rearrange(self.weights(), 'd o i -> () d o i').to(x.device)
        h0 = bitwise_xnor(w, x)  # bdoi
        # Validate
        b,d,o,i = h0.shape
        assert o==self.out_features,f'output out axis not match. actual:{o}, expected:{self.out_features}'
        assert i==calc_packed_features(self.in_features),f'output in axis not match. actual:{i}, expected:{calc_packed_features(self.in_features)}'
        return h0
    
    def backward(self, h_flip, x):
        self.backward_tensor, self.x_old = h_flip, x
        # Validate
        b,d,o,i = h_flip.shape
        assert o==self.out_features,f'output out axis not match. actual:{o}, expected:{self.out_features}'
        assert i==calc_packed_features(self.in_features),f'output in axis not match. actual:{i}, expected:{calc_packed_features(self.in_features)}'
        # Update weights
        x_flip = reduce_flip(h_flip, 'b d o i -> b d i', vote_p_max=0.5)
        # Validate
        b,d,i = x_flip.shape
        assert i==calc_packed_features(self.in_features),f'output in axis not match. actual:{i}, expected:{calc_packed_features(self.in_features)}'
        return x_flip
    
    @property
    def path(self): return super().path      # getter    
    @path.setter
    def path(self, value:Path):   #override
        self._path = value     # setter
        self.weights.path = self._path +'.'+ self.weights.path.stem

class BMA(Module):
    """Binary Multiply-Accumulate Layer Module class
    
    期待する入力： packed uint8 tensor
    出力：FP32 (normalized mean0, std1)
    """
    def __init__(self, in_features:int, out_features:int, depth_features:int=1, weights=None):
        """
        Parameters
        ------------------
        in_features:int  - numbers of input bits, and weight i axis features
        out_features:int - numbers of output bits, and weight o axis features
        depth_features:int - weight d axis features. if 1, broadcast to len(depth_ths)
        depth_ths:list - thresholds of depth
        weights : uint8 Tensor (or None)
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.depth_features = depth_features
        self.xnor_weight = XnorWeight(in_features, out_features, depth_features, weights=weights)
        self.path = Path(self.__class__.__name__)

    def forward(self, x):
        # Validate
        b,d,i = x.shape
        assert i==calc_packed_features(self.in_features),f'output in axis not match. actual:{i}, expected:{calc_packed_features(self.in_features)}'
        # forward
        h0 = self.xnor_weight(x)
        o1 = popcount_normalize(h0, 'b d o i -> b o')
        # Validate
        b,o = o1.shape
        assert o==self.out_features,f'output out axis not match. actual:{o}, expected:{self.out_features}'
        return o1

    def backward(self, grad, x):
        # Validate
        b,o = grad.shape
        assert o==self.out_features,f'output out axis not match. actual:{o}, expected:{self.out_features}'
        # backward
        grad = rearrange(grad, 'b o -> b () o ()')
        h_old = self.xnor_weight(x) # bdi ->bdoi
        h_flip = backward_popcount(grad, h_old)
        x_flip = self.xnor_weight.backward(h_flip, x)
        # from IPython.core.debugger import Pdb; Pdb().set_trace()
        # Validate
        b,d,i = x_flip.shape
        assert i==calc_packed_features(self.in_features),f'output in axis not match. actual:{i}, expected:{calc_packed_features(self.in_features)}'
        return x_flip

    def __call__(self, x): return self.forward(x)

    @property
    def path(self): return super().path      # getter    
    @path.setter
    def path(self, value:Path):   #override
        self._path = value     # setter
        self.xnor_weight.path = self._path +'.'+ self.xnor_weight.path.stem
        
class BMAB(BMA):
    "Binary Matrix-Accumulate Binarize Layer Module class"
    def __init__(self, in_features:int, out_features:int, depth_features:int=1, out_depth_ths:list=TH_DEPTH3, weights=None, is_standardize=True, pts_sigma=15):
        super().__init__(in_features, out_features, depth_features, weights)
        out_depth_ths = rearrange(torch.tensor(out_depth_ths), 'd -> () d ()')  # bdo
        self.out_depth_ths = out_depth_ths
        self.pnb = PNB(out_depth_ths, 'b d o i -> b o')

    def forward(self, x):
        """
        Parameters
        ------------------
        x : uint8 Tensor
        
        Returns
        --------------
        Tensor (int8)
        """
        # Validate
        b,d,i = x.shape
        assert i==calc_packed_features(self.in_features),f'input in axis not match. actual:{i}, expected:{calc_packed_features(self.in_features)}'
        # forward
        h0 = self.xnor_weight(x)
        o0 = self.pnb(h0)
        # Validate
        b,d,o = o0.shape
        assert d==self.out_depth_ths.numel(),f'output depth axis not match. d:{d}'
        assert o*8==self.out_features,f'output out axis not match. actual:{o*8}, expected:{self.out_features}'
        return o0
    
    def backward(self, flip, x):
        # Validate
        b,d,o = flip.shape
        assert d==self.out_depth_ths.numel(),f'input depth axis not match. d:{d}'
        assert o*8==self.out_features,f'input out axis not match. actual:{o*8}, expected:{self.out_features}'
        # backward
        h_old = self.xnor_weight(x) # bdi ->bdoi
        h_flip = self.pnb.backward(flip, h_old)
        x_flip = self.xnor_weight.backward(h_flip, x)
        # Validate
        b,d,i = x_flip.shape
        assert i==calc_packed_features(self.in_features),f'output in axis not match. actual:{i}, expected:{calc_packed_features(self.in_features)}'
        # from IPython.core.debugger import Pdb; Pdb().set_trace()
        return x_flip


# ---------- Sequential ---------------------
class Sequential(Module):
    """

    Usage
    ----------------------    
    binary_net = bnn.Sequential([
        bnn.Binarize(depth_ths=[-1,0,1]),
        bnn.BMA(hidden_features, out_features),]
    )
    
    """
    def __init__(self, layers:list, name='Seq', log_bin_output=True):
        super().__init__()
        self.path = Path(name)
        for i,l in enumerate(layers):
            l.path = self.path+'.'+f'{i:03d}-{l.path}'
        self.layers = layers
        self.fn = self.BinaryLayerFunction.apply
        self.log_bin_output = log_bin_output

    def forward(self, x):
        x = self.fn(x, self)
        return x 
    def update(self, x):
        self.fn(x, self)

    class BinaryLayerFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x0, model):
            h = x0
            inputs = []
            for layer in model.layers:
                inputs.append(h)
                h = layer(h)
                if model.log_bin_output:
                    if h.dtype == torch.uint8:
                        o_dx = reduce(unpackbits(h).type(torch.float32), 'b ... -> ...', 'mean').flatten().cpu().numpy()
                        logger.heatmap(f"{layer.path}/forward_out-mean_of_bits-heatmap", o_dx)
                        logger.log({f"{layer.path}/forward_out-mean_of_bits": wandb.Histogram(o_dx)})
                        if h.ndim>2:
                            o_d = reduce(unpackbits(h).type(torch.float32), 'b d ... -> d', 'mean').flatten().cpu().numpy()
                            logger.heatmap(f"{layer.path}/forward_out-mean_of_bits-d-heatmap", o_d)
                            logger.log({f"{layer.path}/forward_out-mean_of_bits-d": wandb.Histogram(o_d)})
                            o_x = reduce(unpackbits(h).type(torch.float32), 'b d ... -> ...', 'mean').flatten().cpu().numpy()
                            logger.heatmap(f"{layer.path}/forward_out-mean_of_bits-x-heatmap", o_x)
                            logger.log({f"{layer.path}/forward_out-mean_of_bits-x": wandb.Histogram(o_x)})
                    elif h.dtype == torch.float32:
                        logger.log({f"{layer.path}/forward_out-fp32": wandb.Histogram(h.cpu().numpy())})
            # ctx.save_for_backward(inputs)
            ctx.inputs = inputs
            ctx.model = model
            return h

        @staticmethod
        def backward(ctx, flip):
            inputs = ctx.inputs
            model = ctx.model
            backprop_tensor = flip
            for layer,x in zip(reversed(model.layers), reversed(inputs)):
                if model.log_bin_output:
                    if backprop_tensor.dtype == torch.uint8:
                        logger.log({f"{layer.path}/backward_in - flip_ratio": calc_1ratio(backprop_tensor)})
                    elif backprop_tensor.dtype == torch.float32:
                        logger.log({f"{layer.path}/backward_in - grad": wandb.Histogram(backprop_tensor.cpu().numpy())})
                backprop_tensor = layer.backward(backprop_tensor, x)
                if model.log_bin_output:
                    if backprop_tensor.dtype == torch.uint8:
                        logger.log({f"{layer.path}/backward_out - flip_ratio": calc_1ratio(backprop_tensor)})
                    elif backprop_tensor.dtype == torch.float32:
                        logger.log({f"{layer.path}/backward_out - grad": wandb.Histogram(backprop_tensor.cpu().numpy())})
            grad = backprop_tensor
            return grad, None

