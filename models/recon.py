import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pado.fourier import fft, ifft
from utils.utils import *

class Arch(nn.Module):
    def __init__(self, args, nf = 12):
        super().__init__()
        self.nf = nf

        self.down0_0 = ConvBlock(6, nf, 7, 1, 3)
        self.down0_1 = ConvBlock(nf,nf, 3,1,1)

        self.down1_0 = ConvBlock(nf, 2*nf, 3,2,1)
        self.down1_1 = ConvBlock(2*nf, 2*nf, 3,1,1)
        self.down1_2 = ConvBlock(2*nf, 2*nf, 3,1,1)

        self.down2_0 = ConvBlock(2*nf, 4*nf, 3,2,1)
        self.down2_1 = ConvBlock(4*nf, 4*nf, 3,1,1)
        self.down2_2 = ConvBlock(4*nf, 4*nf, 3,1,1)

        self.down3_0 = ConvBlock(4*nf, 8*nf, 3,2,1)
        self.down3_1 = ConvBlock(8*nf, 8*nf, 3,1,1)
        self.down3_2 = ConvBlock(8*nf, 8*nf, 3,1,1)
    
        self.down4_0 = ConvBlock(8*nf, 12*nf, 3,2,1)
        self.down4_1 = ConvBlock(12*nf, 12*nf, 3,1,1)
        self.down4_2 = ConvBlock(12*nf, 12*nf, 3,1,1)

        self.bottleneck_0 = ConvBlock(24*nf, 24*nf, 3,1,1)
        self.bottleneck_1 = ConvBlock(24*nf, 12*nf, 3,1,1)

        self.up4_0 = ConvTraspBlock(12*nf, 8*nf, 2,2,0)
        self.up4_1 = ConvBlock(24*nf, 8*nf, 3,1,1)

        self.up3_0 = ConvTraspBlock(8*nf, 4*nf, 2,2,0)
        self.up3_1 = ConvBlock(12*nf, 4*nf, 3,1,1)
        
        self.up2_0 = ConvTraspBlock(4*nf, 2*nf, 2,2,0)
        self.up2_1 = ConvBlock(6*nf, 2*nf, 3,1,1)
        
        self.up1_0 = ConvTraspBlock(2*nf, nf, 2,2,0)
        self.up1_1 = ConvBlock(3*nf, nf, 3,1,1)
        
        self.up0_0 = ConvBlock(nf, 3, 3,1,1)
        self.up0_1 = nn.Sequential(
                nn.ReflectionPad2d(2),
                nn.Conv2d(3, 3, kernel_size=5, stride=1),
                nn.Tanh()
                )

        self.res0 = BasicBlock(3 * 2, 3 * 2, 5, 1, 2).to(args.device)
        self.res1 = BasicBlock(3 * 2, 3 * 2, 5, 1, 2).to(args.device)
        self.res2 = BasicBlock(3 * 2, 3 * 2, 5, 1, 2).to(args.device)

        self.out0 = ConvBlock(3 * 2, 3, 5,1,2)
        self.out1 = nn.Sequential(
                nn.ReflectionPad2d(2),
                nn.Conv2d(3, 3, kernel_size=5, stride=1),
                nn.Tanh()
                )

    def forward(self, image, psf_near, psf_far):
        psf0 = torch.tile(psf_far, (1,int(self.nf/3),1,1))
        psf1 = torch.tile(sample_psf(psf_far, 2), (1,int(2*self.nf/3),1,1))
        psf2 = torch.tile(sample_psf(psf_far, 4), (1,int(4*self.nf/3),1,1))
        psf3 = torch.tile(sample_psf(psf_far, 8), (1,int(8*self.nf/3),1,1))
        psf4 = torch.tile(sample_psf(psf_far, 16), (1,int(12*self.nf/3),1,1))

        images = [image, Wiener_deconv(image, psf_near)]
        images = torch.cat(images, dim = 1)

        down0 = self.down0_0(images)
        down0 = self.down0_1(down0)
        deconv0 = Wiener_deconv(down0, psf0)

        down1 = self.down1_0(down0)
        down1 = self.down1_1(down1)
        down1 = self.down1_2(down1)
        deconv1 = Wiener_deconv(down1, psf1)
        
        down2 = self.down2_0(down1)
        down2 = self.down2_1(down2)
        down2 = self.down2_2(down2)
        deconv2 = Wiener_deconv(down2, psf2)
        
        down3 = self.down3_0(down2)
        down3 = self.down3_1(down3)
        down3 = self.down3_2(down3)
        deconv3 = Wiener_deconv(down3, psf3)
    
        down4 = self.down4_0(down3)
        down4 = self.down4_1(down4)
        down4 = self.down4_2(down4)
        deconv4 = Wiener_deconv(down4, psf4)
        
        bottleneck = self.bottleneck_0(torch.cat([deconv4,down4], 1))
        bottleneck = self.bottleneck_1(bottleneck)

        up4 = self.up4_0(bottleneck)
        up4 = self.up4_1(torch.cat([up4,down3, deconv3], 1))

        up3 = self.up3_0(up4)
        up3 = self.up3_1(torch.cat([up3,down2, deconv2], 1))
        
        up2 = self.up2_0(up3)
        up2 = self.up2_1(torch.cat([up2,down1, deconv1], 1))
        
        up1 = self.up1_0(up2)
        up1 = self.up1_1(torch.cat([up1,down0, deconv0], 1))
        
        up0 = self.up0_0(up1)
        up0 = self.up0_1(up0)
        up0 = up0 + image

        res = self.res0(torch.cat([up0,image], 1))
        res = self.res1(res)
        res = self.res2(res)

        out = self.out0(res)
        out = self.out1(out)
        out = (out + 1) / 2

        return out
    
def ConvBlock(in_channels, out_channels, kernel_size, stride, padding):
    block = nn.Sequential(
            nn.ReflectionPad2d(padding),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride), #padding=padding),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU()
            )
    return block

def ConvTraspBlock(in_channels, out_channels, kernel_size, stride, padding):
    block = nn.Sequential(
            nn.ReflectionPad2d(padding),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride), #padding=padding),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU()
            )
    return block

def Wiener_deconv(image, psf):
    otf = fft(real2complex(psf))
    wiener = otf.conj() / real2complex(otf.get_mag() **2 + 1e-6) 
    image_deconv = ifft(wiener * fft(real2complex(image))).get_mag()
    return torch.clamp(image_deconv.type_as(image), min = 0, max = 1)

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(padding),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride), #padding=padding),
            nn.InstanceNorm2d(out_channels),
            )
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)
        out += identity
        out = self.leakyrelu(out)

        return out