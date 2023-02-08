import numpy as np
from pado.material import *
from utils.utils import *

# https://www.edmundoptics.com/p/8mm-UC-Series-Fixed-Focal-Length-Lens/41864?gclid=CjwKCAjwh5qLBhALEiwAioods3Z90aJenLK11rrj2R5E7SpKF0gvF8a9vZrsd0H5aY72nIgcYq42QRoC4hAQAvD_BwE
# https://www.edmundoptics.com/p/bfs-u3-120s4c-cs-usb3-blackflyreg-s-color-camera/40172/

camera_resolution = [4000,3000]
camera_pitch = 1.85e-6
background_pitch = camera_pitch * 2
sensor_dist = focal_length = 8e-3
aperture_shape='circle'
wvls = [656e-9, 589e-9, 486e-9] # camera RGB wavelength
DOE_wvl = 550e-9 # wavelength used to set DOE

# DOE specs
R = C = 1600 # Resolution of the simulated wavefront
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
DOE_phase_noise_scale = 0.1

DOE_phase_init = torch.zeros((1,1, R, C))

# depth
depth_near_min = 0.05
depth_near_max = 0.12
depth_far_min = 5
depth_far_max = 10
plot_depth = [5,3,1,0.12,0.08,0.05]

# raindrop
drop_Nmin = 5
drop_Nmax = 8
drop_Rmin = 1e-3
drop_Rmax = 3e-3

# dirt
perlin_res = 8
perlin_cutoff = 0.55