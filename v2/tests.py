# Usage
# ----------------------------------------------------------------
# pytest tests.py

import bnn as bnn
# from bnn import *

# Setup
from fastai.tabular.all import TabularDataLoaders,Categorify, FillMissing, Normalize, CategoryBlock, \
      Learner, CrossEntropyLossFlat, accuracy, ShowGraphCallback  #*
import torch
import torch.nn as nn
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from fastai.callback.wandb import WandbCallback
import seaborn as sns

## Prepare Dataset
# Iris load & make DataFrame
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)

train_df.reset_index(drop=True, inplace=True)
valid_df.reset_index(drop=True, inplace=True)

cat_names = []
cont_names = iris.feature_names
dep_var = 'target'

## DataLoader
dls = TabularDataLoaders.from_df(train_df, 
                                 path='.', 
                                 procs = [Categorify, FillMissing, Normalize],
                                 cat_names=cat_names,
                                 cont_names=cont_names, 
                                 y_names=dep_var, 
                                 y_block=CategoryBlock, 
                                 valid_idx=list(valid_df.index),
                                 shuffle=True,
                                 bs=64,
                                )

# CallBacks
from fastai.callback.core import Callback
class BinaryLearnCallback(Callback):
    def __init__(self, binary_nets:list, **kwargs): 
        super().__init__(**kwargs)
        self.binary_nets = binary_nets
    def after_step(self):  # after update of parameter
        # for net in self.binary_nets: net.is_update = True
        bnn.is_update = True
        self.learn.model(*self.learn.xb)
        bnn.is_update = False


# Model
th_depth = bnn.TH_DEPTH3
class Model02(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, depth_features):
        super(Model02, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.norm = torch.nn.BatchNorm1d(input_dim)
        self.binary_net = bnn.Sequential([
            bnn.Binarize(depth_ths=th_depth),
            # bnn.BMAB(input_dim, hidden_dim),
            bnn.BMAB(input_dim, hidden_dim, depth_features),
            bnn.BMA(hidden_dim, output_dim, depth_features),]
        )

    def forward(self, x_cat, x_cont):
        x = self.norm(x_cont)
        x = self.binary_net(x)
        return x

input_dim = len(cont_names)
hidden_dim = 64
output_dim = len(iris.target_names)
depth_features = len(th_depth)  # 1

# model = Model02(input_dim, hidden_dim, output_dim, depth_features)

# # Learn
# learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(), metrics=accuracy,
#                cbs=[
#                    # CustomWandbCallback(), 
#                    BinaryLearnCallback([model.binary_net]),
#                    ShowGraphCallback()])


# learn.dls.bs=len(dls.train_ds)
# learn.fit_one_cycle(5, lr_max=1e-2)

def test_depth3():
    model = Model02(input_dim, hidden_dim, output_dim, depth_features)

    # Learn
    learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(), metrics=accuracy,
                cbs=[
                    # CustomWandbCallback(), 
                    BinaryLearnCallback([model.binary_net]),
                    ShowGraphCallback()])
    learn.dls.bs=len(dls.train_ds)
    learn.fit_one_cycle(5, lr_max=1e-2)
    return True

def test_depth1():
    model = Model02(input_dim, hidden_dim, output_dim, depth_features=1)
    # Learn
    learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(), metrics=accuracy,
                cbs=[
                    # CustomWandbCallback(), 
                    BinaryLearnCallback([model.binary_net]),
                    ShowGraphCallback()])
    learn.dls.bs=len(dls.train_ds)
    learn.fit_one_cycle(5, lr_max=1e-2)
    return True
