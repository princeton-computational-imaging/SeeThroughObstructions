import numpy as np
import os
from matplotlib import pyplot   
import cv2
import shutil
import json
from glob import glob

from models.forwards import *

def save_settings(args, param):
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    args_dict = vars(args)
    with open(os.path.join(args.result_path,'args.json'), "w") as f:
        json.dump(args_dict, f, indent=4, sort_keys=False)
    shutil.copy(args.param_file, args.result_path)
    if args.pretrained_DOE is not None:
        shutil.copy(args.DOE_phase_init_ckpt, os.path.join(args.result_path, 'init'))
    if args.pretrained_G is not None:
        shutil.copy(args.G_init_ckpt, os.path.join(args.result_path, 'init'))
    args.param = param

def plot_depth_based_psf(DOE_phase, args, depths = [0.05, 0.1, 0.2, 0.4, 0.8, 1, 3, 5], wvls = 'RGB', normalize = True, merge_channel = False):
    param = args.param  
    doe = DOE(param.R, param.C, param.DOE_pitch, param.material, param.DOE_wvl,args.device, phase = DOE_phase)

    psfs = []
    if wvls == 'RGB':
        for i in range(len(param.wvls)):
            wvl = param.wvls[i]
            psf_depth = []
            for z in depths:
                psf = compute_psf_Fraunhofer(wvl, torch.tensor(z), doe, args)
                if normalize:
                    psf_depth.append(psf/torch.max(psf))
                else:
                    psf_depth.append(psf)
            psfs.append(torch.cat(psf_depth, -1))
        if merge_channel:
            psfs = torch.cat(psfs, 1)
        else:
            psfs = torch.cat(psfs, -2)
    elif wvls == 'design':
        wvl = param.DOE_wvl
        psf_depth = []
        for z in depths:
            psf = compute_psf_Fraunhofer(wvl, torch.tensor(z), doe, args)
            if normalize:
                psf_depth.append(psf/torch.max(psf))
            else:
                psf_depth.append(psf)
        psfs.append(torch.cat(psf_depth, -1))
        psfs = torch.cat(psfs, -2)
    else:
        assert False, "%s not defined" %wvls

    log_psfs = torch.log(psfs + 1e-9)
    log_psfs -= torch.min(log_psfs)
    log_psfs /= torch.max(log_psfs)
    return psfs, log_psfs
    
def last_save(ckpt_path, file_format):
    return sorted(glob(os.path.join(ckpt_path, file_format)))[-1]

def trapez(y,y0,w):
    return np.clip(np.minimum(y+1+w/2-y0, -y+1+w/2+y0),0,1)

def weighted_line(r0, c0, r1, c1, w, rmin=0, rmax=np.inf):
    # The algorithm below works fine if c1 >= c0 and c1-c0 >= abs(r1-r0).
    # If either of these cases are violated, do some switches.
    if abs(c1-c0) < abs(r1-r0):
        # Switch x and y, and switch again when returning.
        xx, yy, val = weighted_line(c0, r0, c1, r1, w, rmin=rmin, rmax=rmax)
        return (yy, xx, val)

    # At this point we know that the distance in columns (x) is greater
    # than that in rows (y). Possibly one more switch if c0 > c1.
    if c0 > c1:
        return weighted_line(r1, c1, r0, c0, w, rmin=rmin, rmax=rmax)

    # The following is now always < 1 in abs
    slope = (r1-r0) / (c1-c0)

    # Adjust weight by the slope
    w *= np.sqrt(1+np.abs(slope)) / 2

    # We write y as a function of x, because the slope is always <= 1
    # (in absolute value)
    x = np.arange(c0, c1+1, dtype=float)
    y = x * slope + (c1*r0-c0*r1) / (c1-c0)

    # Now instead of 2 values for y, we have 2*np.ceil(w/2).
    # All values are 1 except the upmost and bottommost.
    thickness = np.ceil(w/2)
    yy = (np.floor(y).reshape(-1,1) + np.arange(-thickness-1,thickness+2).reshape(1,-1))
    xx = np.repeat(x, yy.shape[1])
    vals = trapez(yy, y.reshape(-1,1), w).flatten()

    yy = yy.flatten()

    # Exclude useless parts and those outside of the interval
    # to avoid parts outside of the picture
    mask = np.logical_and.reduce((yy >= rmin, yy < rmax, vals > 0))

    return (yy[mask].astype(int), xx[mask].astype(int), vals[mask])

def plot_line(x1,x2,y1,y2,psf):
    rr, cc, val = weighted_line(x1,x2, y1, y2,2)
    psf[rr,cc] = val[:,None]    
    return psf

def viz_psf(psf, param, center_R = None, g = 2.2, weight = 4):
    psf = (psf / np.max(psf))**(1/g)
    if center_R is not None:
        size = 2*center_R
        w = h = int(param.img_res/2-center_R)
        psf_center = psf[h:h+size,w:w+size]
        psf_center = cv2.resize(psf_center, (center_R * 15,center_R * 15),interpolation = cv2.INTER_NEAREST)
        psf[:center_R * 15, :center_R * 15] = psf_center
        
        if len(psf.shape) == 2:
            psf = pyplot.cm.hot(psf)[...,:-1]

        # psf[:int(param.img_res/3),0:weight,:] = 1
        psf[:center_R * 15+ weight,center_R * 15: center_R * 15 + weight,:] = 1
        # psf[0:weight,:int(param.img_res/3),:] = 1
        psf[center_R * 15:center_R * 15 + weight,:center_R * 15+ weight,:] = 1
        psf[h:h+size+ weight,w-weight:w,:] = 1
        psf[h:h+size+ weight,w + size : w+size+weight,:] = 1
        psf[h-weight:h,w-weight:w+size+ weight,:] = 1
        psf[h+size:h+size+weight,w:w+size+ weight,:] = 1
        rr, cc, val = weighted_line(center_R * 15,0, h+size + weight, w,weight)
        psf[rr,cc] = val[:,None]
        rr, cc,  val = weighted_line(0, center_R * 15,h, w+size+ weight,weight)
        psf[rr,cc] = val[:,None]
    else:
        if len(psf.shape) == 2:
            psf = pyplot.cm.hot(psf)[...,:-1]
    return psf

def plot_psf_array(psfs, param, center_R = 10, gap = 10, g = 1.5):
    # if psfs.shape[1] == 1:
    #     psfs = torch.tile(psfs,(1,3,1,1))
    psfs_singles = torch.split(psfs, param.img_res, dim=-1)
    cnt = len(psfs_singles)
    canvas = np.ones([param.img_res, param.img_res * cnt + gap * (cnt-1), 3])
    for i in range(cnt):
        if psfs.shape[1] == 1:
            psf = psfs_singles[i][0,0].cpu().numpy()
        else:
            psf = psfs_singles[i][0].permute(1,2,0).cpu().numpy()
        if i < 3:
            canvas[:, i * (param.img_res+gap):i * (param.img_res+gap) + param.img_res, :] = viz_psf(psf, param, center_R = center_R, g = g)

        else:
            canvas[:, i * (param.img_res+gap):i * (param.img_res+gap) + param.img_res, :] = viz_psf(psf, param, center_R = None, g = g)
    pyplot.figure(figsize = (30,10))
    pyplot.imshow(canvas)    
    return canvas



