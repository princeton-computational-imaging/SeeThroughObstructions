import numpy as np
import os
from matplotlib import pyplot   
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



