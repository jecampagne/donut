import numpy as np
from astropy.io.fits import getdata
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


###############
#hdu = fits.open("WLconv_z0.50_2000r.fits")
#img=hdu[0].data
#plt.imshow(img)
#plt.show()
#image de 512x512 pixels

#data, hdr = getdata("WLconv_z0.50_0002r.fits", 0, header=True)


class MinMaxScaler(object):
    """
    Scale to [0, 1] range and get back
    """
    def __init__(self):
        self.min_ = 0.0
        self.max_ = 1.0
        self.inv_scale_ = 1.0
    def __call__(self, X):
        self.max_ = X.max()
        self.min_ = X.min()
##        print(self.min_,self.max_)
        dist = self.max_ - self.min_
        if dist ==0.:  # pathology
            print("MinMaxScaler error min=max")
            dist = 1.
        scale = 1.0 / dist
        self.inv_scale_ = dist
##        print(scale,self.inv_scale_)
        return (X-self.min_)*scale
    def inverse_transform(self, X):
##        print(self.min_, self.inv_scale_)
        return X*self.inv_scale_+self.min_

class DatasetKapa(Dataset):
    def __init__(self, dir_path="/sps/lsst/data/campagne/convergence/Maps05/",
                 file_tag=None, file_range=None):

        #file list
        list_files = []
        if file_tag == None and file_range == None:
            #for debug
            list_files = [
                "WLconv_z0.50_0001r.fits",
                "WLconv_z0.50_0002r.fits",
                "WLconv_z0.50_0003r.fits",
                "WLconv_z0.50_0004r.fits", 
                "WLconv_z0.50_0005r.fits",
                "WLconv_z0.50_0006r.fits"
                ]
        elif file_tag != None and file_range != None:
            for i in range(*file_range):
                new_file = file_tag + str(i).zfill(4) + 'r.fits'
                list_files.append(new_file)
        else:
            print("DatasetKapa: ERROR list of file")
            return
        
        self.imgs = []
        for i in range(len(list_files)):
            fname = dir_path + list_files[i]
            data = getdata(fname, 0, header=False) # read fits -32 BITPIX
            data = data.astype(np.float32)         # hack because of bit ordering ?
            data  = data[:,:,np.newaxis]           # HW => HWC with C=1 
            self.imgs.append(data)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        image = self.imgs[index]

        return image

class ToTensorKapa(object):
    def __call__(self, pic):
        img = torch.from_numpy(pic.transpose((2, 0, 1)).copy()) # numpy HWC -> torch CHW
        return img
    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomApplyKapa(transforms.RandomApply):
    """Apply randomly a list of transformations with a given probability

    Args:
        transforms (list or tuple): list of transformations
        p (float): list of probabilities
    """

    def __init__(self, transforms):
        super(RandomApplyKapa, self).__init__(transforms)

    def __call__(self, img):
        # for each list of transforms
        # apply random sample to apply or not the transform
        for itset in range(len(self.transforms)):
            transf = self.transforms[itset]
            t = random.choice(transf)
            #### print('t:=',t)
            img = t(img)

        return img

#    def __repr__(self):
#        format_string = self.__class__.__name__ + '('
#        for itset in range(len(self.transforms)):
#            format_string += '[ '
#            transf = self.transforms[itset]
#            for t in transf:
#                format_string += '{0} '.format(t.__name__)
#            format_string += ']'
#        format_string += ')'
#        return format_string


def flipH(a):
    """
    Parameters
    ----------
    a: an image

    Returns
    -------
    an image flipped wrt the Horizontal axe
    """
    return np.flip(a,0)

def flipV(a):
    """
    Parameters
    ----------
    a: an image

    Returns
    -------
    an image flipped wrt the Vertical axe
    """
    return np.flip(a,1)

def rot90(a):
    """
    Parameters
    ----------
    a: an image

    Returns
    -------
    an image rotated 90deg anti-clockwise
    """
    return np.rot90(a,1)

def rot180(a):
    """
    Parameters
    ----------
    a: an image

    Returns
    -------
    an image rotated 180deg anti-clockwise
    """
    return np.rot90(a,2)

def rot270(a):
    """
    Parameters
    ----------
    a: an image

    Returns
    -------
    an image rotated 270deg anti-clockwise
    """
    return np.rot90(a,3)

def identity(a):
    """
    Parameters
    ----------
    a: an image

    Returns
    -------
    the same image
    """
    return a

class RandomCrop(object):
    def __init__(self,size=128):
        self.size=size

    def __call__(self,a):
        h,w,c = a.shape
        th = tw = self.size
        i = random.randint(0, h - th) 
        j = random.randint(0, w - tw)
        return a[i:i+th,j:j+tw,:]

    def __repr__(self):
        return self.__class__.__name__ + '('+str(self.size)+')'
   

class Rescaling(object):
    def __init__(self,scale=1.0):
        self.scale=scale

    def __call__(self,a):
        return a*self.scale

    def __repr__(self):
        return self.__class__.__name__ + '('+str(self.scale)+')'
