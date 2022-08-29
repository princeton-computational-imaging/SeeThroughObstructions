import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from importlib.machinery import SourceFileLoader
import numpy as np
import os
import argparse
from tqdm import trange
import lpips

from utils.save import *
from utils.utils import *

from models.forwards import *
from models.ROLE import compute_raindrop
from models.Defence import compute_fence
from models.Dirt import compute_dirt

def compute_dirt_raindrop(image_far, depth, args):
    if np.random.uniform() < 0.5:
        return compute_dirt(image_far, depth, args)
    else:
        return compute_raindrop(image_far, depth, args)

def log(DOE_phase, G, batch_data, step,args):
    image_far, _ = batch_data 
    image_far = image_far.to(args.device)
    image_near, mask, image_DOE, image_near_DOE, image_far_DOE, psf_near, psf_far, mask_doe, height_map = image_formation(image_far,DOE_phase, args.compute_obstruction, args)
    image = image_near * mask + image_far * (1 - mask)
    
    image_recon = torch.clamp(G(image_DOE, psf_near, psf_far), min=0, max=1)
    G_l1_loss = args.l1_loss_weight * args.l1_criterion(image_recon, image_far)
    G_perc_loss = torch.mean(args.perceptual_loss_weight * args.perceptual_criterion(2 * image_recon - 1, 2 * image_far - 1))
    G_masked_loss = args.masked_loss_weight * args.l1_criterion(image_recon*mask, image_far*mask)
    loss = G_l1_loss + G_perc_loss + G_masked_loss

    psfs, log_psfs = plot_depth_based_psf(DOE_phase, args, args.param.plot_depth)
    DOE_phase_wrapped = DOE_phase % (2*np.pi)
    DOE_phase_wrapped =  DOE_phase_wrapped - torch.min(DOE_phase_wrapped)
    DOE_phase_wrapped = DOE_phase_wrapped / torch.max(DOE_phase_wrapped)
    DOE_phase = DOE_phase - torch.min(DOE_phase)
    DOE_phase = DOE_phase / torch.max(DOE_phase)

    args.writer.add_scalar('val_loss/L1_loss',G_l1_loss, step)  
    args.writer.add_scalar('val_loss/perc_loss',G_perc_loss, step)
    args.writer.add_scalar('val_loss/masked_loss',G_masked_loss, step)
    args.writer.add_scalar('val_loss/loss',loss, step)
    args.writer.add_image('result', torch.cat([torch.cat([image[0],image_far[0]],1), torch.cat([image_DOE[0],image_recon[0]],1)], -1), step)
    args.writer.add_image('image_sensor_component', torch.cat([image_far_DOE[0], image_near_DOE[0], mask_doe[0]],-1), step)
    args.writer.add_image('RGB_psf', psfs[0], step)
    args.writer.add_image('RGB_logpsf', log_psfs[0], step)
    args.writer.add_image('height_map',(height_map/torch.max(height_map))[0], step)
    args.writer.add_image('phase_map', DOE_phase[0], step)
    args.writer.add_image('phase_map_wrapped', DOE_phase_wrapped[0], step)


def train_step(batch_data, DOE_phase, optics_optimizer, G, G_optimizer, step, args):
    image_far, _ = batch_data
    image_far = image_far.to(args.device)
    image_near, mask, image_DOE, image_near_DOE, image_far_DOE, psf_near, psf_far, mask_doe, height_map = image_formation(image_far,DOE_phase, args.compute_obstruction, args)
    
    image_recon = G(image_DOE, psf_near, psf_far)
    G_l1_loss = args.l1_loss_weight * args.l1_criterion(image_recon, image_far)
    G_perc_loss = torch.mean(args.perceptual_loss_weight * args.perceptual_criterion(2 * image_recon - 1, 2 * image_far - 1))
    G_masked_loss = args.masked_loss_weight * args.l1_criterion(image_recon*mask, image_far*mask)
    loss = G_l1_loss + G_perc_loss + G_masked_loss
    loss.backward()

    G_optimizer.step()
    G_optimizer.zero_grad() 
    if args.train_optics:
        optics_optimizer.step()
        optics_optimizer.zero_grad()
    return loss.detach()

def train(args):
    
    if args.debug:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.device == 'cuda':
            torch.cuda.manual_seed(args.seed)
        cudnn.benchmark = True
        cudnn.enabled=True
    param = args.param

    transform_train = transforms.Compose([
            transforms.RandomCrop(param.data_resolution,pad_if_needed=True), # Places365 image size varies
            transforms.RandomCrop([param.equiv_crop_size, param.equiv_crop_size],pad_if_needed=True),
            transforms.Resize([param.img_res, param.img_res]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

    transform_test = transforms.Compose([
            transforms.RandomCrop(param.data_resolution,pad_if_needed=True), # Places365 image size varies
            transforms.CenterCrop(param.equiv_crop_size),
            transforms.Resize([param.img_res, param.img_res]),
            transforms.ToTensor(),
        ])
    if args.obstruction == 'fence':
        trainset = torchvision.datasets.Places365(
            root=param.dataset_dir, split="train-standard", transform=transform_train)
        testset = torchvision.datasets.Places365(
            root=param.dataset_dir, split="val", transform=transform_test)   
        args.compute_obstruction = compute_fence     
    elif args.obstruction == 'raindrop':
        trainset = torchvision.datasets.ImageFolder(param.training_dir, transform=transform_train)
        testset = torchvision.datasets.ImageFolder(param.val_dir, transform=transform_test)
        args.compute_obstruction = compute_raindrop
    elif args.obstruction == 'dirt':
        trainset = torchvision.datasets.ImageFolder(param.training_dir, transform=transform_train)
        testset = torchvision.datasets.ImageFolder(param.val_dir, transform=transform_test)
        args.compute_obstruction = compute_dirt
    elif args.obstruction == 'dirt_raindrop':
        trainset = torchvision.datasets.ImageFolder(param.training_dir, transform=transform_train)
        testset = torchvision.datasets.ImageFolder(param.val_dir, transform=transform_test)
        args.compute_obstruction = compute_dirt_raindrop

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

    if args.debug: 
        trainloader = testloader

    # build model and loss
    args.perceptual_criterion = lpips.LPIPS(net='vgg').to(args.device)
    args.l1_criterion = nn.L1Loss().to(args.device)
    args.writer = SummaryWriter(args.result_path)

    param = args.param
    if args.train_optics:
        DOE_phase = Variable(param.DOE_phase_init.to(args.device), requires_grad=True)
        optics_optimizer = optim.Adam([DOE_phase], lr=args.optics_lr)
    else:
        DOE_phase = Variable(param.DOE_phase_init.to(args.device), requires_grad=False)
        optics_optimizer = None

    from models.recon import Arch
    G = Arch(args).to(args.device)
    if args.pretrained_G is not None:
        G.load_state_dict(torch.load(args.G_init_ckpt, map_location='cpu'))
        G.to(args.device)
    G_optimizer = optim.Adam(params=G.parameters(), lr=args.G_lr)

    for _, batch_data in enumerate(testloader):
        test_data = batch_data
        break

    total_step = 0
    log(DOE_phase, G, test_data, total_step, args)
    for epoch_cnt in trange(args.n_epochs, desc="Epoch"):
        train_loss = 0
        for _, batch_data in enumerate(trainloader):
            step_loss = train_step(batch_data, DOE_phase, optics_optimizer, G, G_optimizer, total_step, args)
            total_step += 1
            train_loss += step_loss
            if total_step % args.log_freq == 0:
                log(DOE_phase, G, test_data, total_step, args)
                args.writer.add_scalar('train_loss/loss',train_loss/args.log_freq, total_step)
                train_loss = 0
            if total_step % args.save_freq == 0:
                torch.save(G.state_dict(), os.path.join(args.result_path,'G_%03d.pt' % (total_step//args.save_freq)))
                if args.train_optics:
                    torch.save(DOE_phase, os.path.join(args.result_path,'DOE_phase_%03d.pt' % (total_step//args.save_freq)))
                    

def main():
    parser = argparse.ArgumentParser(
        description='Obstruction-Free DOE',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    def str2bool(v):
        assert(v == 'True' or v == 'False')
        return v.lower() in ('true')

    def none_or_str(value):
        if value.lower() == 'none':
            return None
        return value

    parser.add_argument('--debug', action="store_true", help='debug mode, train on validation data to speed up the process')
    parser.add_argument('--train_optics', action="store_true", help='optimize optical element design')
    parser.add_argument('--pretrained_DOE', default = None, type =none_or_str, help = 'use a pretrained DOE')
    parser.add_argument('--pretrained_G', default = None, type =none_or_str, help = 'use a pretrained G')
    parser.add_argument('--result_path', default = './ckpt/opt', type=str, help='dir to save models and checkpoints')
    parser.add_argument('--param_file', default= 'config/param_MV_1600.py', type=str, help='path to param file')

    parser.add_argument('--obstruction', default = 'dirt_raindrop', type = str, help = 'obsturction type')
    parser.add_argument('--sensor_noise', default = 0.008, type=float, help='sensor noise level')
    parser.add_argument('--n_epochs', default = 100, type = int, help = 'max num of training epoch')
    parser.add_argument('--optics_lr', default=0.1, type=float, help='optical element learning rate')
    parser.add_argument('--G_lr', default=1e-4, type=float, help='network learning rate')

    parser.add_argument('--l1_loss_weight', default = 1, type = float, help = 'weight for L1 loss')
    parser.add_argument('--masked_loss_weight', default = 1, type = float, help = 'weight for masked loss (focus on obstructed scene)')
    parser.add_argument('--perceptual_loss_weight', default = 1, type = float, help = 'weight for perceptual loss')

    parser.add_argument('--log_freq', default=400, type=int, help = 'frequency (num_steps) of logging')
    parser.add_argument('--save_freq', default=2000, type=int, help = 'frequency (num_steps) of saving checkpoint and visual performance')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    
    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    param = SourceFileLoader("param", args.param_file).load_module()
    param = convert_resolution(param,args)

    if args.pretrained_DOE is not None:
        args.DOE_phase_init_ckpt = last_save(args.pretrained_DOE, 'DOE_phase_*')
        param.DOE_phase_init = torch.load(args.DOE_phase_init_ckpt, map_location='cpu').detach()

    if args.pretrained_G is not None:
        args.G_init_ckpt = last_save(args.pretrained_DOE, 'G_*')
            
    save_settings(args, param)
    train(args)

if __name__ == '__main__':
    
    main()

