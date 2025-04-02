import os
import pdb

import torch
import torch.utils.data as data

from .datahelpers import default_loader, imresize
import torchvision.transforms as T

class ImagesFromListVG(data.Dataset):
    """A generic data loader that loads images from a list 
        (Based on ImageFolder from pytorch)

    Args:
        root (string): Root directory path.
        images (list): Relative image paths as strings.
        imsize (int, Default: None): Defines the maximum size of longer image side
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        images_fn (list): List of full image filename
    """

    def __init__(self, root, images, imsize=[480, 640], transform=None, loader=default_loader, central_crop=None):

        images_fn = [os.path.join(root,images[i]) for i in range(len(images))]

        if len(images_fn) == 0:
            raise(RuntimeError("Dataset contains 0 images!"))

        self.root = root
        self.images = images
        self.imsize = imsize
        self.images_fn = images_fn
        self.transform = transform
        self.loader = loader
        self.central_crop = central_crop

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            image (PIL): Loaded image
        """
        path = self.images_fn[index]
        img = self.loader(path)
        img = self.transform(img)

        if self.central_crop is not None:
            # Perform squared central crop with size==shorter_side of the original image
            # Then resize to self.resize
            img = T.functional.center_crop(img, 1012) # hard-coded for endomapper
            img = T.functional.resize(img, self.imsize, antialias=None)
        else:
            img = T.functional.resize(img, self.imsize, antialias=None)
       

        return img

    def __len__(self):
        return len(self.images_fn)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of images: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
