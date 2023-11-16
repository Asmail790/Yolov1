from torch import nn
from typing import Literal
from typing import NamedTuple, Final
import torch

class ConvParms(NamedTuple):
    s:int # stride
    p:int # padding
    k:int # kernel size
    o:int # filter out


Params = list[Literal["M"]|ConvParms]
darknet:Params =[]


darknet =[]
darknet+=[
    ConvParms(o=64,k=7,s=2,p=3),
    "M",
    ConvParms(o=192,k=3,s=1,p=1),
    "M",
    ConvParms(o=128,k=1,s=1,p=0),
    ConvParms(o=256,k=3,s=1,p=1),
    ConvParms(o=256,k=1,s=1,p=0),
    ConvParms(o=512,k=3,s=1,p=1),
    "M"
   ]

darknet+=[
    ConvParms(o=256,k=1,s=1,p=0),
    ConvParms(o=512,k=3,s=1,p=1)
    ]*4

darknet+=[
    ConvParms(o=512,k=1,s=1,p=0),
    ConvParms(o=1024,k=3,s=1,p=1),
    "M"   
]


darknet+=[
    ConvParms(o=512,k=1,s=1,p=0),
    ConvParms(o=1024,k=3,s=1,p=1)
    ]*2

darknet+=[
    ConvParms(o=1024,k=3,s=1,p=1),
    ConvParms(o=1024,k=3,s=2,p=1)
]


def make_layers(config:Params=darknet, in_channels:int=3):
    layers=[]
    
    for layer in config:
        if isinstance(layer,ConvParms):
            layers+=[
                nn.Conv2d(in_channels=in_channels,out_channels=layer.o,kernel_size=layer.k,stride=layer.s,padding=layer.p),
                nn.BatchNorm2d(layer.o),
                nn.LeakyReLU(0.1)
            ]
            in_channels = layer.o

        if isinstance(layer,str):
            layers+=[nn.MaxPool2d(kernel_size=2,stride=2)]

    return nn.Sequential(*layers)
