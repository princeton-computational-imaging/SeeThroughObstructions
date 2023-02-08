import torch
from pado.material import *

# https://www.usa.canon.com/internet/portal/us/home/products/details/cameras/eos-dslr-and-mirrorless-cameras/dslr/eos-rebel-t5-ef-s-18-55-is-ii-kit/eos-rebel-t5-18-55-is-ii-kit
camera_resolution = [5196,3464] # 18e6
camera_pitch = 4.3e-6
background_pitch = camera_pitch * 4
sensor_dist = focal_length = 50e-3
aperture_shape='circle'
wvls = [656e-9, 589e-9, 486e-9] # camera RGB wavelength
DOE_wvl = 550e-9 # wavelength used to set DOE

# DOE specs
R = C = 2800 # Resolution of the simulated wavefront
DOE_material = 'FUSED_SILICA'
material = Material(DOE_material)
DOE_pitch = camera_pitch * 1.5
aperture_diamter = DOE_pitch * R 
DOE_sample_ratio = 2
image_sample_ratio = 3
equiv_camera_pitch = camera_pitch * image_sample_ratio
img_res = 512
assert DOE_pitch * DOE_sample_ratio == equiv_camera_pitch
DOE_max_height = 1.2e-6
DOE_height_noise_scale = 4*10e-9
DOE_phase_noise_scale = 0.05

DOE_phase_init = torch.zeros((1,1, R, C))

# depth
depth_near_min = 0.4
depth_near_max = 0.8
depth_far_min = 5
depth_far_max = 10
plot_depth = [10,8,5,0.8,0.6,0.4]

# fence
fence_min = 5e-3
fence_max = 15e-3
