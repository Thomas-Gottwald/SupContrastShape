import torch
import torch.nn.functional as nnf
from torchvision import transforms, datasets


class SameTwoRandomResizedCrop:
    """Creates the same random resize crop of two images"""
    def __init__(self, size, scale=(0.08,1), ratio=(3/4,4/3)):
        if hasattr(size, "__getitem__") and hasattr(size, "__len__"):
            assert len(size) in [1,2]
            if len(size) == 1:
                self.size = size[0],size[0]
            else:
                self.size == size
        else:
            assert type(size) is int
            self.size = size,size
        self.scale = scale
        self.ratio = ratio

    def __call__(self, x):
        x_orig, x_diff = x
        i,j,h,w = transforms.RandomResizedCrop.get_params(x_orig, scale=self.scale, ratio=self.ratio)

        x_orig = transforms.functional.resized_crop(x_orig, i, j, h, w, size=self.size)
        x_diff = transforms.functional.resized_crop(x_diff, i, j, h, w, size=self.size)

        return [x_orig, x_diff]
    

class SameTwoColorJitter:
    """Applies the same random color jitter to two images"""
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = (max(0,1-brightness), 1+brightness)
        self.contrast = (max(0,1-contrast), 1+contrast)
        self.saturation = (max(0,1-saturation), 1+saturation)
        self.hue = (-hue, hue)

    def adjust_color_component(img, factor, component):
        if component == 0:
            return transforms.functional.adjust_brightness(img, factor)
        elif component == 1:
            return transforms.functional.adjust_contrast(img, factor)
        elif component == 2:
            return transforms.functional.adjust_saturation(img, factor)
        else:
            return transforms.functional.adjust_hue(img, factor)

    def __call__(self, x):
        x_orig, x_diff = x

        order,b,c,s,h = transforms.ColorJitter.get_params(self.brightness, self.contrast, self.saturation, self.hue)
        factors = b,c,s,h

        for c in order:
            x_orig = SameTwoColorJitter.adjust_color_component(x_orig, factors[c], c)
            x_diff = SameTwoColorJitter.adjust_color_component(x_diff, factors[c], c)

        return [x_orig, x_diff]


class SameTwoApply:
    """Applies the same transform to two images"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        x_orig, x_diff = x
        return [self.transform(x_orig), self.transform(x_diff)]
    

class DiffTransform:
    """Applies to two images the same same_transform and then transform individually"""
    def __init__(self, transform, same_transform=None):
        self.transform = transform
        self.same_transform = same_transform

    def __call__(self, x):
        if self.same_transform:
            x_orig, x_diff = self.same_transform(x)
        else:
            x_orig, x_diff = x
        return [self.transform(x_orig), self.transform(x_diff)]
    

class SelectTransform:
    """Selects random one of the coupled images"""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x):
        x_orig, x_diff = x

        if torch.rand(1).item() >= self.p:
            return x_orig
        else:
            return x_diff
    

class DiffLoader:
    """Loads two images one from path the other from the same path where path_orig is replaced by path_diff"""
    def __init__(self, path_orig:str, path_diff:str, loader=datasets.folder.default_loader):
        self.path_orig = path_orig
        self.path_diff = path_diff
        self.loader = loader

    def __call__(self, path:str):
        return [self.loader(path), self.loader(path.replace(self.path_orig, self.path_diff))]
    

class ShufflePatches:
    # inspired from https://stackoverflow.com/questions/66962837/shuffle-patches-in-image-batch
    def __init__(self, patch_size):
        self.patch_size = patch_size

    def __call__(self, x):
        # unfold the tensor image
        u = nnf.unfold(x, kernel_size=self.patch_size , stride=self.patch_size , padding=0)
        # shuffle the patches in unfolded form
        pu = u[:,torch.randperm(u.shape[-1])]
        # fold the tensor back in its original form
        f = nnf.fold(pu, x.shape[-2:], kernel_size=self.patch_size, stride=self.patch_size, padding=0)

        return f
    
    def apply_to_batch(self, x):
        # unfold the tensor image
        u = nnf.unfold(x, kernel_size=self.patch_size , stride=self.patch_size , padding=0)
        # shuffle the patches in unfolded form
        pu = torch.stack([b_[:,torch.randperm(b_.shape[-1])] for b_ in u], dim=0)
        # fold the tensor back in its original form
        f = nnf.fold(pu, x.shape[-2:], kernel_size=self.patch_size, stride=self.patch_size, padding=0)

        return f
    

class ShuffleInnerPatches:
    # inspired from https://stackoverflow.com/questions/66962837/shuffle-patches-in-image-batch
    def __init__(self, patch_size, tol=0.2):
        self.patch_size = patch_size
        self.tol = tol

    def __call__(self, x):
        # detect zero padding
        height, width = x.shape[1:]
        
        top = 0
        while(torch.allclose(x[:,top,:], torch.tensor([0.0]), atol=self.tol)):
            top += 1
            if top > height//2:
                break
        top = ((top-1) // self.patch_size + 1) * self.patch_size

        bottom = height-1
        while(torch.allclose(x[:,bottom,:], torch.tensor([0.0]), atol=self.tol)):
            bottom -= 1
            if bottom < height//2:
                break
        bottom = ((bottom+1) // self.patch_size) * self.patch_size

        left = 0
        while(torch.allclose(x[:,:,left], torch.tensor([0.0]), atol=self.tol)):
            left += 1
            if left > width//2:
                break
        left = ((left-1) // self.patch_size + 1) * self.patch_size

        right = width-1
        while(torch.allclose(x[:,:,right], torch.tensor([0.0]), atol=self.tol)):
            right -= 1
            if right < width//2:
                break
        right = ((right+1) // self.patch_size) * self.patch_size

        if top < bottom and left < right:
            x_inner = x[:,top:bottom,left:right]
            
            # unfold the tensor image
            u_inner = nnf.unfold(x_inner, kernel_size=self.patch_size , stride=self.patch_size , padding=0)
            # shuffle the patches in unfolded form
            pu_inner = u_inner[:,torch.randperm(u_inner.shape[-1])]
            # fold the tensor back in its original form
            f_inner = nnf.fold(pu_inner, x_inner.shape[-2:], kernel_size=self.patch_size, stride=self.patch_size, padding=0)

            x[:,top:bottom,left:right] = f_inner

        return x
    

class TwoDifferentApply:

    def __init__(self, transform_orig=None, transform_diff=None):
        self.transform_orig = transform_orig
        self.transform_diff = transform_diff

    def __call__(self, x):
        x_orig, x_diff = x

        if self.transform_orig:
            x_orig = self.transform_orig(x_orig)
        if self.transform_diff:
            x_diff = self.transform_diff(x_diff)

        return [x_orig, x_diff]