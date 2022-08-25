import torch
import torch.nn as nn
from blocks import *

# 
class ResUnetPlus_2L(nn.Module):
    def __init__(self, paras=[1,32,16,8,4,1], p=0):
        super(ResUnetPlus_2L, self).__init__()
        self.Input = InputBlock(paras[0:2])  # 16 /1
        self.downsample_input = nn.MaxPool2d(kernel_size=2, stride=2) # 16 /2
        
        self.Encoder1 = nn.Sequential(
            SqueezeExciteBlock(paras[1]),   
            ResCoderBlock(paras[1],up=False),
            nn.Dropout(p),
        )   # 32 /2
        self.downsample1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 32 /4
        
        self.Encoder2 = nn.Sequential(
            SqueezeExciteBlock(paras[1]*2),
            ResCoderBlock(paras[1]*2,up=False),
            nn.Dropout(p),
        )   # 64 /4
        
        self.Bridge = ASPP(paras[1]*4)   # 64 /4
        
        
        self.Attention1 = AttentionBlock([paras[1]*4,paras[2]])   # 128 /4
        self.Decoder1 = nn.Sequential(
            ResCoderBlock(paras[1]*8,up=True),
            nn.Dropout(p),
        )   # 64 /4
        self.upsample1 = nn.ConvTranspose2d(paras[1]*4, paras[1]*2, \
                         kernel_size=2, stride=2, output_padding=(1,0))   # 32 /2
        
        self.Attention2 = AttentionBlock([paras[1]*2,paras[3]])   # 64 /2
        self.Decoder2 = nn.Sequential(
            ResCoderBlock(paras[1]*4,up=True),
            nn.Dropout(p),
        )   # 32 /2
        self.upsample2 = nn.ConvTranspose2d(paras[1]*2, paras[1]*1, \
                         kernel_size=2, stride=2)   # 16 /1
        
        self.Attention3 = AttentionBlock([paras[1]*1,paras[4]])   # 32 /1
        self.Decoder3 = nn.Sequential(
            ResCoderBlock(paras[1]*2,up=True),
            nn.Dropout(p),
        )   # 16 /1
        
        self.Output = OutputBlock([paras[1],paras[5]])   # 1 /1

    def forward(self, x):
        cache1 = self.Input(x)  # 16 /1
        out = self.downsample_input(cache1)  # 16 /2
#        print(out.size(2),out.size(3))
        cache2 = self.Encoder1(out)  # 32 /2
        out = self.downsample1(cache2)  # 32 /4
#        print(out.size(2),out.size(3))
        cache3 = self.Encoder2(out)  # 64 /4
        out = self.Bridge(cache3)   # 64 /4
        out = self.Attention1(cache3,out)  # 128 /4
        out = self.Decoder1(out)  # 64 /4
        out = self.upsample1(out)  # 32 /2
#        print(out.size(2),out.size(3))
        out = self.Attention2(cache2,out)  # 64 /2
        out = self.Decoder2(out)  # 32 /2
        out = self.upsample2(out)  # 16 /1
#        print(out.size(2),out.size(3))
        out = self.Attention3(cache1,out)  # 32 /1
        out = self.Decoder3(out)  # 16 /1
        out = self.Output(out)  # 1 /1
        return out.squeeze()



class ResUnetPlus_3L(nn.Module):
    def __init__(self, paras=[1,32,32,16,8,4,1], p=0):
        super(ResUnetPlus_3L, self).__init__()
        self.Input = InputBlock(paras[0:2])  # 16 /1
        self.downsample_input = nn.MaxPool2d(kernel_size=2, stride=2) # 16 /2
        
        self.Encoder1 = nn.Sequential(
            SqueezeExciteBlock(paras[1]),   
            ResCoderBlock(paras[1],up=False),
            nn.Dropout(p),
        )   # 32 /2
        self.downsample1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 32 /4
        
        self.Encoder2 = nn.Sequential(
            SqueezeExciteBlock(paras[1]*2),
            ResCoderBlock(paras[1]*2,up=False),
            nn.Dropout(p),
        )   # 64 /4
        self.downsample2 = nn.MaxPool2d(kernel_size=2, stride=2)   # 64 /8
        
        self.Encoder3 = nn.Sequential(
            SqueezeExciteBlock(paras[1]*4),
            ResCoderBlock(paras[1]*4,up=False),
            nn.Dropout(p),
        )   # 128 /8
        
        self.Bridge = ASPP(paras[1]*8)   # 128 /8
        
        self.Attention1 = AttentionBlock([paras[1]*8,paras[2]])  # 256 /8
        self.Decoder1 = nn.Sequential(
            ResCoderBlock(paras[1]*16,up=True),
            nn.Dropout(p),
        )   # 128 /8
        self.upsample1 = nn.ConvTranspose2d(paras[1]*8, paras[1]*4, \
                       kernel_size=2, stride=2, output_padding=(0,1))   # 64 /4
        
        self.Attention2 = AttentionBlock([paras[1]*4,paras[3]])   # 128 /4
        self.Decoder2 = nn.Sequential(
            ResCoderBlock(paras[1]*8,up=True),
            nn.Dropout(p),
        )   # 64 /4
        self.upsample2 = nn.ConvTranspose2d(paras[1]*4, paras[1]*2, \
                         kernel_size=2, stride=2, output_padding=(1,0))   # 32 /2
        
        self.Attention3 = AttentionBlock([paras[1]*2,paras[4]])   # 64 /2
        self.Decoder3 = nn.Sequential(
            ResCoderBlock(paras[1]*4,up=True),
            nn.Dropout(p),
        )   # 32 /2
        self.upsample3 = nn.ConvTranspose2d(paras[1]*2, paras[1]*1, \
                         kernel_size=2, stride=2)   # 16 /1
        
        self.Attention4 = AttentionBlock([paras[1]*1,paras[5]])   # 32 /1
        self.Decoder4 = nn.Sequential(
            ResCoderBlock(paras[1]*2,up=True),
            nn.Dropout(p),
        )   # 16 /1
        
        self.Output = OutputBlock([paras[1],paras[6]])   # 16 /1

    def forward(self, x):
        cache1 = self.Input(x)  # 16 /1
        out = self.downsample_input(cache1)  # 16 /2
#        print(out.size(2),out.size(3))
        cache2 = self.Encoder1(out)  # 32 /2
        out = self.downsample1(cache2)  # 32 /4
#        print(out.size(2),out.size(3))
        cache3 = self.Encoder2(out)  # 64 /4
        out = self.downsample2(cache3)  # 64 /8
#        print(out.size(2),out.size(3))
        cache4 = self.Encoder3(out)  # 128 /8
        out = self.Bridge(cache4)   # 128 /8
        out = self.Attention1(cache4,out)  # 256 /8
        out = self.Decoder1(out)  # 128 /8
        out = self.upsample1(out)  # 64 /4
#        print(out.size(2),out.size(3))
        out = self.Attention2(cache3,out)  # 128 /4
        out = self.Decoder2(out)  # 64 /4
        out = self.upsample2(out)  # 32 /2
#        print(out.size(2),out.size(3))
        out = self.Attention3(cache2,out)  # 64 /2
        out = self.Decoder3(out)  # 32 /2
        out = self.upsample3(out)  # 16 /1
#        print(out.size(2),out.size(3))
        out = self.Attention4(cache1,out)  # 32 /1
        out = self.Decoder4(out)  # 16 /1
        out = self.Output(out)  # 16 /1
        return out.squeeze()
