import torch
import torch.nn as nn


class InputBlock(nn.Module):
    def __init__(self, channels):
        super(InputBlock, self).__init__()
        self.input_block = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(),
            nn.Conv2d(channels[1], channels[1], kernel_size=3, stride=1, padding=1),
        )
        self.skip = nn.Sequential(
#            nn.BatchNorm2d(channels[0]),
#            nn.ReLU(),
            nn.Conv2d(
                channels[0], channels[1], kernel_size=1, stride=1
            ),
        )
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.input_block(x) + self.skip(x))
    
class ResCoderBlock(nn.Module):
    def __init__(self, channel,up=False):
        super(ResCoderBlock, self).__init__()
        if up == False:
            self.coder_block = nn.Sequential(
#                nn.BatchNorm2d(channel),
#                nn.ReLU(),
                nn.Conv2d(
                    channel, channel*2, kernel_size=3, stride=1, padding=1
                ),
                nn.BatchNorm2d(channel*2),
                nn.ReLU(),
                nn.Conv2d(channel*2, channel*2, kernel_size=3, stride=1, padding=1),
            )
            self.skip = nn.Sequential(
#                nn.BatchNorm2d(channel),
#                nn.ReLU(),
                nn.Conv2d(
                    channel, channel*2, kernel_size=1, stride=1
                ),
            )
        else:
            self.coder_block = nn.Sequential(
#                nn.BatchNorm2d(channel),
#                nn.ReLU(),
                nn.Conv2d(
                    channel, channel//2, kernel_size=3, stride=1, padding=1
                ),
                nn.BatchNorm2d(channel//2),
                nn.ReLU(),
                nn.Conv2d(channel//2, channel//2, kernel_size=3, stride=1, padding=1),
            )
            self.skip = nn.Sequential(
#                nn.BatchNorm2d(channel),
#                nn.ReLU(),
                nn.Conv2d(
                    channel, channel//2, kernel_size=1, stride=1
                ),
            )
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.coder_block(x) + self.skip(x))
    
class SqueezeExciteBlock(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SqueezeExciteBlock, self).__init__()
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Linear(channel, channel//reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.adaptive_pool(x).squeeze()
        y = self.excite(y).view(x.size(0),x.size(1),1,1)
        return x * y.expand_as(x)
    
class ASPP(nn.Module):
    def __init__(self, channel, rate=[1, 6, 12, 18]):
        super(ASPP, self).__init__()
        self.aspp_block1 = nn.Sequential(
#            nn.BatchNorm2d(channel),
#            nn.ReLU(inplace=True),
            nn.Conv2d(
                channel, channel, 3, stride=1, padding=rate[0], dilation=rate[0]
            ),
        )
        self.aspp_block2 = nn.Sequential(
#            nn.BatchNorm2d(channel),
#            nn.ReLU(inplace=True),
            nn.Conv2d(
                channel, channel, 3, stride=1, padding=rate[1], dilation=rate[1]
            ),
        )
        self.aspp_block3 = nn.Sequential(
#            nn.BatchNorm2d(channel),
#            nn.ReLU(inplace=True),
            nn.Conv2d(
                channel, channel, 3, stride=1, padding=rate[2], dilation=rate[2]
            ),
        )
        self.aspp_block4 = nn.Sequential(
#            nn.BatchNorm2d(channel),
#            nn.ReLU(inplace=True),
            nn.Conv2d(
                channel, channel, 3, stride=1, padding=rate[3], dilation=rate[3]
            ),
        )


        self.output = nn.Sequential(
            nn.BatchNorm2d(len(rate)*channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(len(rate)*channel, channel, kernel_size=1),
        )
        self._init_weights()

    def forward(self, x):
        x1 = self.aspp_block1(x)
        x2 = self.aspp_block2(x)
        x3 = self.aspp_block3(x)
        x4 = self.aspp_block3(x)
        out = torch.cat([x1, x2, x3, x4], dim=1)
        return self.output(out)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super(AttentionBlock, self).__init__()

        self.attention_encoder = nn.Sequential(
#            nn.BatchNorm2d(channels[0]),
#            nn.ReLU(),
            nn.Conv2d(channels[0], channels[1], kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.attention_decoder = nn.Sequential(
#            nn.BatchNorm2d(channels[0]),
#            nn.ReLU(),
            nn.Conv2d(channels[0], channels[1], kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.attention = nn.Sequential(
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(),
            nn.Conv2d(channels[1], 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU()
        
    def forward(self, x1, x2):
        out = self.attention_encoder(x1) + self.attention_decoder(x2)
        out = self.attention(self.relu(out))
        out = torch.cat((out*x2, x1), dim=1)
        return out

class OutputBlock(nn.Module):
    def __init__(self, channels):
        super(OutputBlock, self).__init__()
        self.output_block = nn.Sequential(
             ASPP(channels[0]),
             nn.Conv2d(channels[0], channels[1], kernel_size=1, stride=1),
             nn.Sigmoid(),
        )
    def forward(self, x):
        return self.output_block(x)
    
    
    
class Discriminator(nn.Module):
    def __init__(self):   
        super(Discriminator, self).__init__()     
        self.decision = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=5, stride=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=5, stride=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(3*3*256, 3*3*64, bias=True),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(3*3*64, 1, bias=True),
            nn.Sigmoid()
        )
        
        self._initialize_weights()
        
    def forward(self, x, y):
        y = (y>0.5)*1
        x = x*y
        out = self.decision(x)
        return out
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data)
        
        
        
    
    