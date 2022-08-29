import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import os

from utils.utils import *

class DefencingDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_list = [f for f in os.listdir(root_dir) if not f.startswith('.')]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.image_list[idx])
        image = Image.open(img_name).convert('RGB')
        pixel_width = int(self.image_list[idx].split('-')[-1].split('.')[0])
        if self.transform:
            image = self.transform(image)
        sample = {'image': image , 'pixel_width': pixel_width}
        return sample

def _largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.
    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow
    Converted to Python by Aaron Snoswell
    Source: http://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
    """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )

transform_fence = transforms.Compose([
        transforms.ColorJitter(contrast=0.2),
        transforms.RandomRotation(90),
        transforms.CenterCrop(1300),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

fenceset = DefencingDataset('/projects/FHEIDE/obstruction_free_doe/De-fencing/dataset_parsed/', transform=transform_fence)

def compute_fence(image_far, depth, args):
    '''generate fence obstruction'''
    param = args.param
    fence_width = metric2pixel(randuni(param.fence_min, param.fence_max, 1)[0] , depth, args)

    fence = fenceset[np.random.randint(len(fenceset))]

    fence_image = fence['image']
    T = transforms.Compose([
        transforms.Resize(size=(int(fence_width/fence['pixel_width']*fence_image.shape[-2]),int(fence_width/fence['pixel_width']*fence_image.shape[-1]))),
        transforms.CenterCrop([param.img_res, param.img_res])
        ])

    image_near = T(fence_image)[None, ...]
    mask = (image_near > torch.median(image_near))*1.0
    image_near = image_near.to(args.device) * mask.to(args.device) + image_far * (1-mask.to(args.device))
    # if torch.mean(mask) > 0.3:
    #     print(torch.mean(mask))
    #     print(torch.median(image_near))
    return image_near.to(args.device), mask.to(args.device)