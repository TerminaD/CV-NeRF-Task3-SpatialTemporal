import os
import json

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from utils.pixel2ray import init_p2c_directions, get_p2w_ray_directions


class BlenderDataset(Dataset):
    """
    A dataset class for 360-degrees inward-facing (synthetic) images.
    
    This class is ported from kwea123/nerf_pl 
    (https://github.com/kwea123/nerf_pl/tree/master)
    under the MIT license.
    """
    def __init__(self, root_dir, split='train', img_wh=(200, 200)):
        self.root_dir = root_dir
        self.split = split
        assert img_wh[0] == img_wh[1], 'image width must equal image height!'
        self.img_wh = img_wh
        self.transform = T.ToTensor()
        
        self.read_meta()
        self.white_back = True

    def read_meta(self):
        with open(os.path.join(self.root_dir,
                               f"transforms_{self.split}.json"), 'r') as f:
            self.meta = json.load(f)

        w, h = self.img_wh
        self.focal = 0.5*800/np.tan(0.5*self.meta['camera_angle_x']) # original focal length
                                                                     # when W=800

        self.focal *= self.img_wh[0]/800 # modify focal length to match size self.img_wh

        # bounds, common for all scenes
        self.near = 2.0
        self.far = 6.0
        self.bounds = np.array([self.near, self.far])
        
        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = \
            init_p2c_directions(h, w, self.focal) # (h, w, 3)
            
        if self.split == 'train': # create buffer of all rays and rgb data
            self.image_paths = []
            self.poses = []
            self.all_rays = []
            self.all_rgbs = []
            self.all_times = []
            for frame in self.meta['frames']:
                pose = np.array(frame['transform_matrix'])[:3, :4]
                self.poses += [pose]
                c2w = torch.FloatTensor(pose)

                image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
                self.image_paths += [image_path]
                img = Image.open(image_path)
                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img) # (4, h, w)
                img = img.view(4, -1).permute(1, 0) # (h*w, 4) RGBA
                img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB
                self.all_rgbs += [img]
                
                rays_o, rays_d = get_p2w_ray_directions(self.directions, c2w) # both (h*w, 3)

                self.all_rays += [torch.cat([rays_o, rays_d, 
                                             self.near*torch.ones_like(rays_o[:, :1]),
                                             self.far*torch.ones_like(rays_o[:, :1])],
                                             1)] # (h*w, 8)
                
                self.all_times.append(frame['time'])

            self.all_rays = torch.cat(self.all_rays, 0) # (len(self.meta['frames])*h*w, 8)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # (len(self.meta['frames])*h*w, 3)
            
            self.all_times = torch.repeat_interleave(torch.tensor(self.all_times), w*h, 0)
            assert self.all_times.shape == torch.Size([len(self.meta['frames'])*h*w])
            

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        if self.split == 'val':
            return 8 # only validate 8 images (to support <=8 gpus)
        return len(self.meta['frames'])

    def __getitem__(self, idx):
        if self.split == 'train': # use data in the buffers
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx],
                      'times': self.all_times[idx]}

        else: # create data for each image separately
            frame = self.meta['frames'][idx]
            c2w = torch.FloatTensor(frame['transform_matrix'])[:3, :4]

            img = Image.open(os.path.join(self.root_dir, f"{frame['file_path']}.png"))
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img) # (4, H, W)
            img = img.view(4, -1).permute(1, 0) # (H*W, 4) RGBA
            img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB

            rays_o, rays_d = get_p2w_ray_directions(self.directions, c2w)

            rays = torch.cat([rays_o, rays_d, 
                              self.near*torch.ones_like(rays_o[:, :1]),
                              self.far*torch.ones_like(rays_o[:, :1])],
                              1) # (H*W, 8)

            sample = {'rays': rays,
                      'rgbs': img,
                      'time': frame['time']}

        return sample