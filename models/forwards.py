from pado.light import * 
from pado.optical_element import *
from pado.propagator import *

from utils.utils import *

def compute_psf(wvl, depth, doe, args):
    '''simulate depth based psf'''
    param = args.param
    prop = Propagator('Fresnel')
    
    light = Light(param.R, param.C, param.DOE_pitch, wvl, args.device,B=1)
    light.set_spherical_light(depth .numpy())

    lens = RefractiveLens(param.R, param.C, param.DOE_pitch, param.focal_length, wvl,args.device)
    light = lens.forward(light)

    doe.change_wvl(wvl)
    light = doe.forward(light)
    
    aperture = Aperture(param.R, param.C, param.DOE_pitch, param.aperture_diamter, param.aperture_shape, wvl, args.device)
    light = aperture.forward(light)

    light_prop = prop.forward(light, param.sensor_dist)
    psf = light_prop.get_intensity()
    psf = sample_psf(psf, param.DOE_sample_ratio)
    psf /= torch.sum(psf)

    psf_size = psf.shape
    if psf_size[-2]*psf_size[-1] < param.img_res**2:
        wl, wr = compute_pad_size(psf_size[-1], param.img_res)
        hl, hr = compute_pad_size(psf_size[-2], param.img_res)
        psf = F.pad(psf, (wl, wr, hl, hr), "constant", 0)
    elif psf_size[-2]*psf_size[-1] > param.img_res**2:
        wl, wr = compute_pad_size(param.img_res, psf_size[-1])
        hl, hr = compute_pad_size(param.img_res, psf_size[-2])  
        psf = psf[:,:,hl:-hr, wl:-wr]    
    if depth <= param.depth_near_max:
        cutoff = np.tan(np.sinh(wvl/(2*param.DOE_pitch)))*param.focal_length / param.equiv_camera_pitch 
        DOE_mask = edge_mask(int(param.img_res / 2),cutoff, args.device)
        psf *= DOE_mask  
    psf /= torch.sum(psf)
    return psf

def compute_psf_Fraunhofer(wvl, depth, doe, args):
    '''simulate depth based psf'''
    param = args.param
    prop = Propagator('Fraunhofer')
    
    light = Light(param.R, param.C, param.DOE_pitch, wvl, args.device,B=1)
    light.set_spherical_light(depth.numpy())

    doe.change_wvl(wvl)
    light = doe.forward(light)

    aperture = Aperture(param.R, param.C, param.DOE_pitch, param.aperture_diamter, param.aperture_shape, wvl, args.device)
    light = aperture.forward(light)

    light_prop = prop.forward(light, param.sensor_dist)
    psf = light_prop.get_intensity()
    
    # resize 
    psf = F.interpolate(psf, int(param.R * light_prop.pitch / param.DOE_pitch))
     
    psf = sample_psf(psf, param.DOE_sample_ratio)

    psf_size = psf.shape
    if psf_size[-2]*psf_size[-1] < param.img_res**2:
        wl, wr = compute_pad_size(psf_size[-1], param.img_res)
        hl, hr = compute_pad_size(psf_size[-2], param.img_res)
        psf = F.pad(psf, (wl, wr, hl, hr), "constant", 0)
    elif psf_size[-2]*psf_size[-1] > param.img_res**2:
        wl, wr = compute_pad_size(param.img_res, psf_size[-1])
        hl, hr = compute_pad_size(param.img_res, psf_size[-2])  
        psf = psf[:,:,hl:-hr, wl:-wr]  
    if depth <= param.depth_near_max:
        cutoff = np.tan(np.sinh(wvl/(2*param.DOE_pitch)))*param.focal_length / param.equiv_camera_pitch 
        DOE_mask = edge_mask(int(param.img_res / 2),cutoff, args.device)
        psf *= DOE_mask     
    psf /= torch.sum(psf)
    return psf

def image_formation(image_far, DOE_phase, compute_obstruction, args, z_near = None):
    param = args.param
    doe = DOE(param.R, param.C, param.DOE_pitch, param.material, param.DOE_wvl,args.device, phase = DOE_phase)
    height_map = doe.get_height() * edge_mask(int(param.R/ 2),int(param.R/ 2), args.device)

    if z_near is None:
        z_near = randuni(param.depth_near_min, param.depth_near_max, 1)[0] # randomly sample the near-point depth from a range

    z_far = randuni(param.depth_far_min, param.depth_far_max, 1)[0] # randomly sample the far-point depth from a range
    image_near, mask = compute_obstruction(image_far, z_near, args)

    img_doe = []
    img_near_doe = []
    img_far_doe = []
    psf_far_doe = []
    psf_near_doe = []
    mask_doe = []

    for i in range(len(param.wvls)):

        wvl = param.wvls[i]

        psf_near = compute_psf_Fraunhofer(wvl, z_near, doe, args)
        psf_far = compute_psf_Fraunhofer(wvl, z_far, doe, args)

        img_near_conv = conv_fft(real2complex(image_near[:,i,:,:]), real2complex(psf_near), (int(param.R/2),int(param.R/2),int(param.R/2),int(param.R/2))).get_mag() # 
        img_far_conv = conv_fft(real2complex(image_far[:,i,:,:]), real2complex(psf_far)).get_mag()
        mask_conv = conv_fft(real2complex(mask[:,i,:,:]), real2complex(psf_near), (int(param.R/2),int(param.R/2),int(param.R/2),int(param.R/2))).get_mag() 
        mask_conv = torch.clamp(1.5*mask_conv,0,1)
        img_conv = img_near_conv * mask_conv + img_far_conv * (1 - mask_conv)
    
        img_doe.append(img_conv)
        img_near_doe.append(img_near_conv)
        img_far_doe.append(img_far_conv)
        psf_near_doe.append(psf_near)
        psf_far_doe.append(psf_far)
        mask_doe.append(mask_conv)
    img_doe = torch.cat(img_doe, dim = 1)
    img_near_doe = torch.cat(img_near_doe, dim = 1)
    img_far_doe = torch.cat(img_far_doe, dim = 1)
    psf_near_doe = torch.cat(psf_near_doe, dim = 1)
    psf_far_doe = torch.cat(psf_far_doe, dim = 1)
    mask_doe = torch.cat(mask_doe, dim = 1)

    if args.sensor_noise > 0:
        noise = torch.rand(img_doe.shape) * 2 * args.sensor_noise - args.sensor_noise
        img_doe = torch.clamp(img_doe + noise.type_as(img_doe), 0, 1)

    return image_near.type_as(image_far), mask.type_as(image_far).type_as(image_far), \
        img_doe.type_as(image_far), img_near_doe.type_as(image_far), img_far_doe.type_as(image_far), \
            psf_near_doe.type_as(image_far), psf_far_doe.type_as(image_far), mask_doe.type_as(image_far), height_map.type_as(image_far)