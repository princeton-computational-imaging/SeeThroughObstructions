import cv2
import math
import numpy as np
import torch

from utils.utils import *

class raindrop():
    def __init__(self, centerxy, radius):	
        self.ifcol = False	
        self.col_with = []
        self.center = centerxy
        self.radius = radius
        self.alphamap = np.zeros((self.radius * 5, self.radius*4,1))
        self.texture = None
        self._createDefaultDrop()

    def updateTexture(self, fg):		
        
        # add fish eye effect to simulate the background
        K = np.array([[30*self.radius, 0, 2*self.radius],
                [0., 20*self.radius, 3*self.radius],
                [0., 0., 1]])
        D = np.array([0.0, 0.0, 0.0, 0.0])
        Knew = K.copy()
        Knew[(0,1), (0,1)] = math.pow(self.radius, 1/3) * 1.2 * Knew[(0,1), (0,1)]
        fisheye = cv2.fisheye.undistortImage(fg, K, D=D, Knew=Knew)
        tmp = np.array(fisheye)

        self.texture = np.clip((0.6 + 1.5*(tmp-0.5)),0,1)

    def _createDefaultDrop(self):
        cv2.circle(self.alphamap, (self.radius * 2, self.radius * 3), self.radius, 128, -1)
        cv2.ellipse(self.alphamap, (self.radius * 2, self.radius * 3), (self.radius, int(np.random.uniform(0.8,1.5) * self.radius)), 0, 180, 360, 128, -1)		
        # set alpha map for png 
        # self.alphamap = cv2.GaussianBlur(self.alphamap,(1,1),0)
        self.alphamap = np.asarray(self.alphamap)
        self.alphamap = (self.alphamap/np.max(self.alphamap))

def compute_raindrop(image_far, depth, args):
    '''generate raindrop obstruction'''
    param = args.param
    image = image_far[0].permute(1,2,0).cpu().numpy()

    drop_num = int(np.random.randint(param.drop_Nmin, param.drop_Nmax) * depth / param.depth_near_max)
    
    alpha_map = np.zeros_like(image)[:,:,0:1]
    imgh, imgw, _ = image.shape
    edge_gap = metric2pixel(param.drop_Rmin,depth, args)
    ran_pos = [(edge_gap + int(np.random.rand() * (imgw - 2*edge_gap)), 2*edge_gap + int(np.random.rand() * (imgh - 3*edge_gap))) for _ in range(drop_num)]
    listRainDrops = []
    #########################
    # Create Raindrop
    #########################
    # create raindrop by default

    for pos in ran_pos:
        radius = np.minimum(metric2pixel(np.random.uniform(param.drop_Rmin, param.drop_Rmax), depth, args), int(np.floor(param.img_res/20)))
        drop = raindrop(pos, radius)
        listRainDrops.append(drop)
    # add texture
    for drop in listRainDrops:
        (ix, iy) = drop.center
        radius = drop.radius
        ROI_WL = 2*radius
        ROI_WR = 2*radius
        ROI_HU = 3*radius
        ROI_HD = 2*radius
        if (iy-3*radius) <0 :
            ROI_HU = iy	
        if (iy+2*radius)>imgh:
            ROI_HD = imgh - iy
        if (ix-2*radius)<0:
            ROI_WL = ix
        if  (ix+2*radius) > imgw:
            ROI_WR = imgw - ix

        drop_alpha = drop.alphamap

        alpha_map[iy - ROI_HU:iy + ROI_HD, ix - ROI_WL: ix+ROI_WR,:] += drop_alpha[3*radius - ROI_HU:3*radius + ROI_HD, 2*radius - ROI_WL: 2*radius+ROI_WR,:]
        drop.image_coor = np.array([iy - ROI_HU, iy + ROI_HD, ix - ROI_WL, ix+ROI_WR])
        drop.alpha_coor = np.array([3*radius - ROI_HU, 3*radius + ROI_HD, 2*radius - ROI_WL, 2*radius+ROI_WR])

        upshift = int(0.1 * (iy - ROI_HU))
        drop.updateTexture(image[iy - ROI_HU - upshift: iy + ROI_HD - upshift, ix - ROI_WL: ix+ROI_WR])
        
    image_near = np.asarray(cv2.GaussianBlur(image,(5,5),0))
    for drop in listRainDrops:
        img_hl,img_hr, img_wl, img_wr = drop.image_coor
        drop_hl, drop_hr, drop_wl, drop_wr = drop.alpha_coor 
        texture_blend = drop.texture*(drop.alphamap[drop_hl:drop_hr, drop_wl:drop_wr])
        update_alpha = drop.alphamap[drop_hl:drop_hr, drop_wl:drop_wr] > 0
        image_near[img_hl:img_hr, img_wl: img_wr] = texture_blend * update_alpha + image_near[img_hl:img_hr, img_wl: img_wr] * (1-update_alpha)
    
    image_near = torch.tensor(image_near *  (alpha_map > 0)).permute(2,0,1)[None,...]
    # image_near = torch.tensor(image_near).permute(2,0,1)[None,...]
    mask = torch.tile(torch.tensor(1.0*(alpha_map > 0)).permute(2,0,1)[None,...],(1,3,1,1))
    return image_near.to(args.device), mask.to(args.device)