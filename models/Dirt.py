import numpy as np
import torch
import torchvision.transforms as transforms
from models.PerlinBlob import *

def circle_grad(img_res):
    center_x, center_y = img_res // 2, img_res // 2
    circle_grad = np.zeros([img_res,img_res])

    for y in range(img_res):
        for x in range(img_res):
            distx = abs(x - center_x)
            disty = abs(y - center_y)
            dist = np.sqrt(distx*distx + disty*disty)
            circle_grad[y][x] = dist
    max_grad = np.max(circle_grad)
    circle_grad = circle_grad / max_grad
    circle_grad -= 0.5
    circle_grad *= 2.0
    circle_grad = -circle_grad

    circle_grad -= np.min(circle_grad)
    max_grad = np.max(circle_grad)
    circle_grad = circle_grad / max_grad
    return circle_grad

def compute_dirt(image_far, depth, args):
    param = args.param
    brown = (np.array([30, 20, 10]) * np.random.uniform(1,2.5) + np.random.uniform(-5,5,3) )* np.ones([param.img_res, param.img_res, 3]) / 255
    perlin_noise = generate_fractal_noise_2d([param.img_res, param.img_res], [param.perlin_res,param.perlin_res], tileable=(True,True), interpolant=interpolant)
    depth_adj = depth / param.depth_near_max
    T = transforms.Compose([transforms.ToTensor(),
                            transforms.RandomCrop(int(param.img_res * depth_adj)), 
                            transforms.Resize([param.img_res, param.img_res])])
    perlin_noise = T(perlin_noise).squeeze().numpy()  
    alpha_map = perlin_noise * (perlin_noise > param.perlin_cutoff) * circle_grad(param.img_res)
    alpha_map /= np.max(alpha_map)
    image_near = torch.tensor(brown* alpha_map[...,None]).permute(2,0,1)[None,...]
    mask = torch.tile(torch.tensor(1.0*(alpha_map[...,None] > 0.3)).permute(2,0,1)[None,...],(1,3,1,1)) 
    image_near = image_near.to(args.device) * mask.to(args.device) + image_far * (1-mask.to(args.device))
    return image_near.to(args.device), mask.to(args.device)

