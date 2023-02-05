import random
import math
import numbers
import collections
import numpy as np
import torch
import copy
from torch import Tensor
from typing import Optional, Tuple, List, Any

from PIL import Image, ImageOps
try:
    import accimage
except ImportError:
    accimage = None

def _equalize_single_image(img: Tensor) -> Tensor:
    return torch.stack([_scale_channel(img[c]) for c in range(img.size(0))])
def _is_tensor_a_torch_image(x: Tensor) -> bool:
    return x.ndim >= 2
def _assert_image_tensor(img):
    if not _is_tensor_a_torch_image(img):
        raise TypeError("Tensor is not a torch image.")
def _assert_channels(img: Tensor, permitted: List[int]) -> None:
    c = _get_image_num_channels(img)
    if c not in permitted:
        raise TypeError("Input image tensor permitted channel values are {}, but found {}".format(permitted, c))
def _get_image_num_channels(img: Tensor) -> int:
    if img.ndim == 2:
        return 1
    elif img.ndim > 2:
        return img.shape[-3]
def _scale_channel(img_chan):
    # TODO: we should expect bincount to always be faster than histc, but this
    # isn't always the case. Once
    # https://github.com/pytorch/pytorch/issues/53194 is fixed, remove the if
    # block and only use bincount.
    if img_chan.is_cuda:
        hist = torch.histc(img_chan.to(torch.float32), bins=256, min=0, max=255)
    else:
        hist = torch.bincount(img_chan.view(-1), minlength=256)

    nonzero_hist = hist[hist != 0]
    step = torch.div(nonzero_hist[:-1].sum(), 255, rounding_mode='floor')
    if step == 0:
        return img_chan

    lut = torch.div(
        torch.cumsum(hist, 0) + torch.div(step, 2, rounding_mode='floor'),
        step, rounding_mode='floor')
    lut = torch.nn.functional.pad(lut, [1, 0])[:-1].clamp(0, 255)

    return lut[img_chan.to(torch.int64)].to(torch.uint8)
def tensor_equalize(img: Tensor) -> Tensor:

    _assert_image_tensor(img)

    if not (3 <= img.ndim <= 4):
        raise TypeError("Input image tensor should have 3 or 4 dimensions, but found {}".format(img.ndim))
    if img.dtype != torch.uint8:
        raise TypeError("Only torch.uint8 image tensors are supported, but found {}".format(img.dtype))

    _assert_channels(img, [1, 3])

    if img.ndim == 3:
        return _equalize_single_image(img)

    return torch.stack([_equalize_single_image(x) for x in img])
def _is_pil_image(img: Any) -> bool:
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)
def pil_equalize(img):
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
    return ImageOps.equalize(img)

def equalize(img: Tensor) -> Tensor:
    """Equalize the histogram of an image by applying
    a non-linear mapping to the input in order to create a uniform
    distribution of grayscale values in the output.
    Args:
        img (PIL Image or Tensor): Image on which equalize is applied.
            If img is torch Tensor, it is expected to be in [..., 1 or 3, H, W] format,
            where ... means it can have an arbitrary number of leading dimensions.
            The tensor dtype must be ``torch.uint8`` and values are expected to be in ``[0, 255]``.
            If img is PIL Image, it is expected to be in mode "P", "L" or "RGB".
    Returns:
        PIL Image or Tensor: An image that was equalized.
    """
    if not isinstance(img, torch.Tensor):
        return pil_equalize(img)

    return tensor_equalize(img)

class RandomEqualize(torch.nn.Module):
    """Equalize the histogram of the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "P", "L" or "RGB".

    Args:
        p (float): probability of the image being equalized. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be equalized.

        Returns:
            PIL Image or Tensor: Randomly equalized image.
        """
        if torch.rand(1).item() < self.p:
            return equalize(img)
        return img


    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

    def randomize_parameters(self, c_size=0):
        pass


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def randomize_parameters(self, c_size=0):
        for t in self.transforms:
            t.randomize_parameters(c_size)


class ToTensor(object):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __init__(self, norm_value=255):
        self.norm_value = norm_value

    def __call__(self, pic):
        """
        Args:
            pic (PIL.Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """

        if isinstance(pic, np.ndarray):
            # handle numpy array
            #print('nparray')
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            # backward compatibility
            return img.float().div(self.norm_value)

        if accimage is not None and isinstance(pic, accimage.Image):
            #print('accimage')
            nppic = np.zeros(
                [pic.channels, pic.height, pic.width], dtype=np.float32)
            pic.copyto(nppic)
            return torch.from_numpy(nppic)

        # handle PIL Image
        if pic.mode == 'I':
            #print('pil I')
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            #print('pil I16')
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            #print('pil else')
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        #print(nchannel)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(self.norm_value)
        else:
            return img

    def randomize_parameters(self, c_size=0):
        pass


class Normalize(object):
    """Normalize an tensor image with mean and standard deviation.
    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respecitvely.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        # TODO: make efficient
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor

    def randomize_parameters(self, c_size=0):
        pass


class Scale(object):
    """Rescale the input PIL.Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size,
                          int) or (isinstance(size, collections.Iterable) and
                                   len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be scaled.
        Returns:
            PIL.Image: Rescaled image.
        """
        if isinstance(self.size, int):
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return img.resize((ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return img.resize((ow, oh), self.interpolation)
        else:
            return img.resize(self.size, self.interpolation)

    def randomize_parameters(self, c_size=0):
        pass


class CenterCrop(object):
    """Crops the given PIL.Image at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th))

    def randomize_parameters(self, c_size=0):
        pass



class CenterCropScaled(object):
    """Crops the given PIL.Image at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        crop_size = min(img.size[0], img.size[1])
        w, h = img.size
        x1 = int(round((w - crop_size) / 2.))
        y1 = int(round((h - crop_size) / 2.))

        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))

        return img.resize(self.size, self.interpolation)

    def randomize_parameters(self, c_size=0):
        pass

'''
class CenterCropScaledMultiple(object):
    """Crops the given PIL.Image at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size, crops, interpolation=Image.BILINEAR):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.interpolation = interpolation
        self.crops = crops

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        crop_size = min(img.size[0], img.size[1])
        w, h = img.size
        x1 = int(round((w - crop_size) / float(self.crops+1)))
        y1 = int(round((h - crop_size) / float(self.crops+1)))

        imgs=[]
        print(img.size, x1, y1)
        for i in range(self.crops):
            st_x = min(i*x1, w-crop_size)
            st_y = min(i*y1, h-crop_size)
            #print(img.size, st_x, st_y, st_x + crop_size, st_y + crop_size)
            img_p = img.copy()
            img_p = img_p.crop((st_x, st_y, st_x + crop_size, st_y + crop_size))
            print(img.size, img_p.size)
            imgs.append(img_p.resize(self.size, self.interpolation))
        return np.stack(imgs, axis=0)

    def randomize_parameters(self, c_size=0):
        pass
'''

class CornerCrop(object):

    def __init__(self, size, crop_position=None):
        self.size = size
        if crop_position is None:
            self.randomize = True
        else:
            self.randomize = False
        self.crop_position = crop_position
        self.crop_positions = ['c', 'tl', 'tr', 'bl', 'br']

    def __call__(self, img):
        image_width = img.size[0]
        image_height = img.size[1]

        if self.crop_position == 'c':
            th, tw = (self.size, self.size)
            x1 = int(round((image_width - tw) / 2.))
            y1 = int(round((image_height - th) / 2.))
            x2 = x1 + tw
            y2 = y1 + th
        elif self.crop_position == 'tl':
            x1 = 0
            y1 = 0
            x2 = self.size
            y2 = self.size
        elif self.crop_position == 'tr':
            x1 = image_width - self.size
            y1 = 0
            x2 = image_width
            y2 = self.size
        elif self.crop_position == 'bl':
            x1 = 0
            y1 = image_height - self.size
            x2 = self.size
            y2 = image_height
        elif self.crop_position == 'br':
            x1 = image_width - self.size
            y1 = image_height - self.size
            x2 = image_width
            y2 = image_height

        img = img.crop((x1, y1, x2, y2))

        return img

    def randomize_parameters(self, c_size=0):
        if self.randomize:
            self.crop_position = self.crop_positions[random.randint(
                0,
                len(self.crop_positions) - 1)]


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        if self.p < 0.5:
            if isinstance(img, np.ndarray):
                return np.fliplr(img).copy()
            else:
                return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

    def randomize_parameters(self, c_size=0):
        self.p = random.random()


class RandomVerticalFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        if self.p < 0.5:
            if isinstance(img, np.ndarray):
                return np.flipud(img).copy()
            else:
                return img.transpose(Image.FLIP_TOP_BOTTOM)
        return img

    def randomize_parameters(self, c_size=0):
        self.p = random.random()


class MultiScaleCornerCrop(object):
    """Crop the given PIL.Image to randomly selected size.
    A crop of size is selected from scales of the original size.
    A position of cropping is randomly selected from 4 corners and 1 center.
    This crop is finally resized to given size.
    Args:
        scales: cropping scales of the original size
        size: size of the smaller edge
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self,
                 scales,
                 size,
                 interpolation=Image.BILINEAR,
                 crop_positions=['c', 'tl', 'tr', 'bl', 'br']):
        self.scales = scales
        self.size = size
        self.interpolation = interpolation

        self.crop_positions = crop_positions

    def __call__(self, img):
        min_length = min(img.size[0], img.size[1])
        crop_size = int(min_length * self.scale)

        image_width = img.size[0]
        image_height = img.size[1]

        if self.crop_position == 'c':
            center_x = image_width // 2
            center_y = image_height // 2
            box_half = crop_size // 2
            x1 = center_x - box_half
            y1 = center_y - box_half
            x2 = center_x + box_half
            y2 = center_y + box_half
        elif self.crop_position == 'tl':
            x1 = 0
            y1 = 0
            x2 = crop_size
            y2 = crop_size
        elif self.crop_position == 'tr':
            x1 = image_width - crop_size
            y1 = 0
            x2 = image_width
            y2 = crop_size
        elif self.crop_position == 'bl':
            x1 = 0
            y1 = image_height - crop_size
            x2 = crop_size
            y2 = image_height
        elif self.crop_position == 'br':
            x1 = image_width - crop_size
            y1 = image_height - crop_size
            x2 = image_width
            y2 = image_height

        img = img.crop((x1, y1, x2, y2))

        return img.resize((self.size, self.size), self.interpolation)

    def randomize_parameters(self, c_size=0):
        self.scale = self.scales[random.randint(0, len(self.scales) - 1)]
        self.crop_position = self.crop_positions[random.randint(
            0,
            len(self.scales) - 1)]


class MultiScaleRandomCrop(object):

    def __init__(self, scales, size, interpolation=Image.BILINEAR):
        self.scales = scales
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):

        min_length = min(img.size[0], img.size[1])
        crop_size = int(min_length * self.scale)

        image_width = img.size[0]
        image_height = img.size[1]

        x1 = self.tl_x * (image_width - crop_size)
        y1 = self.tl_y * (image_height - crop_size)
        x2 = x1 + crop_size
        y2 = y1 + crop_size

        img = img.crop((x1, y1, x2, y2))

        return img.resize((self.size, self.size), self.interpolation)

    def randomize_parameters(self, c_size=0):
        self.scale = self.scales[random.randint(0, len(self.scales) - 1)]
        self.tl_x = random.random()
        self.tl_y = random.random()


class MultiScaleRandomCropMultigrid(object):

    def __init__(self, scales, size, interpolation=Image.BILINEAR):
        self.scales = scales
        self.init_size = size
        self.size = self.init_size
        self.interpolation = interpolation

    def __call__(self, img):

        min_length = min(img.size[0], img.size[1])
        crop_size = int(min_length * self.scale)

        image_width = img.size[0]
        image_height = img.size[1]

        x1 = int(self.tl_x * (image_width - crop_size))
        y1 = int(self.tl_y * (image_height - crop_size))
        x2 = x1 + crop_size
        y2 = y1 + crop_size

        img = img.crop((x1, y1, x2, y2))

        return img.resize((self.size, self.size), self.interpolation)

    def randomize_parameters(self, c_size):
        self.size = c_size
        self.scale = self.scales[random.randint(0, len(self.scales) - 1)]
        self.tl_x = random.random()
        self.tl_y = random.random()
