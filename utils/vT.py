import random
import numpy as np
import torch
import torchvision.transforms.transforms as T
import torchvision.transforms.functional as F
from torchvision import transforms
import PIL
from PIL import Image


class ToTensor(T.ToTensor):
    def __call__(self, pics):
        return [F.to_tensor(pic) for pic in pics]


class Normalize(T.Normalize):
    def __call__(self, pics):
        return [F.normalize(pic, self.mean, self.std, self.inplace) for pic in pics]


class Resize(T.Resize):
    def __call__(self, pics):
        return [F.resize(pic, self.size, self.interpolation) for pic in pics]


class CenterCrop(T.CenterCrop):
    def __call__(self, imgs):
        return [F.center_crop(img, self.size) for img in imgs]


class RandomCrop(T.RandomCrop):
    def __call__(self, imgs):
        i, j, h, w = self.get_params(imgs[0], self.size)
        return [F.crop(img, i, j, h, w) for img in imgs]


class ToClip(object):
    def __call__(self, pics):
        return torch.stack(pics, dim=0)


class Transpose(object):
    def __call__(self, tensor):
        return tensor.transpose(0, 1)


class RandomHorizontalFlip(T.RandomHorizontalFlip):
    """
    Default self.p=0.5, can be modified while init
    Args:
        imgs (PIL Image): Image to be flipped.

    Returns:
        PIL Image: Randomly flipped image.
    """
    def __call__(self, imgs):
        if random.random() < self.p:
            return [F.hflip(img) for img in imgs]
        return imgs


class RandomResizedCrop(T.RandomResizedCrop):
    """
    Args:
        imgs (PIL Image): Image to be cropped and resized.

    Returns:
        PIL Image: Randomly cropped and resized image.
    """
    def __call__(self, imgs):
        i, j, h, w = self.get_params(imgs[0], self.scale, self.ratio)
        return [F.resized_crop(img, i, j, h, w, self.size, self.interpolation) for img in imgs]


class RandomErasing(T.RandomErasing):
    def random_erase(self, img):
        if isinstance(self.value, (int, float)):
            value = [float(self.value)]
        elif isinstance(self.value, str):
            value = None
        elif isinstance(self.value, (list, tuple)):
            value = [float(v) for v in self.value]
        else:
            value = self.value

        if value is not None and not (len(value) in (1, img.shape[-3])):
            raise ValueError(
                "If value is a sequence, it should have either a single value or "
                f"{img.shape[-3]} (number of input channels)"
            )

        x, y, h, w, v = self.get_params(img, scale=self.scale, ratio=self.ratio, value=value)
        return F.erase(img, x, y, h, w, v, self.inplace)

    def __call__(self, pics):
        if torch.rand(1) < self.p:
            return [self.random_erase(pic) for pic in pics]
        return pics


class RandomGrayscale(T.RandomGrayscale):
    def __call__(self, imgs):
        num_output_channels = 1 if imgs[0].mode == 'L' else 3
        if random.random() < self.p:
            return [F.to_grayscale(img, num_output_channels=num_output_channels) for img in imgs]
        return imgs


class RandomTranslateWithReflect:
    """
    Translate image randomly

    Translate vertically and horizontally by n pixels where
    n is integer drawn uniformly independently for each axis
    from [-max_translation, max_translation].

    Fill the uncovered blank area with reflect padding.
    """

    def __init__(self, max_translation):
        self.max_translation = max_translation

    def prosess(self, xtranslation, ytranslation, old_image):
        xpad, ypad = abs(xtranslation), abs(ytranslation)
        xsize, ysize = old_image.size

        flipped_lr = old_image.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_tb = old_image.transpose(Image.FLIP_TOP_BOTTOM)
        flipped_both = old_image.transpose(Image.ROTATE_180)

        new_image = Image.new('RGB', (xsize + 2 * xpad, ysize + 2 * ypad))
        new_image.paste(old_image, (xpad, ypad))

        new_image.paste(flipped_lr, (xpad + xsize - 1, ypad))
        new_image.paste(flipped_lr, (xpad - xsize + 1, ypad))

        new_image.paste(flipped_tb, (xpad, ypad + ysize - 1))
        new_image.paste(flipped_tb, (xpad, ypad - ysize + 1))

        new_image.paste(flipped_both, (xpad - xsize + 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad - xsize + 1, ypad + ysize - 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad + ysize - 1))

        new_image = new_image.crop((xpad - xtranslation, ypad - ytranslation, xpad + xsize - xtranslation, ypad + ysize - ytranslation))

        return new_image

    def __call__(self, imgs):
        xtranslation, ytranslation = np.random.randint(-self.max_translation, self.max_translation + 1, size=2)
        return [self.prosess(xtranslation, ytranslation, img) for img in imgs]


class ColorJitter(T.ColorJitter):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, per_img=True):
        super().__init__(brightness, contrast, saturation, hue)
        self.per_img = per_img  # decide whethor do the same augment for the whole clip or not

    def __call__(self, imgs):
        if isinstance(imgs[0], np.ndarray):
            raise TypeError('Color jitter not yet implemented for numpy arrays')
        elif isinstance(imgs[0], PIL.Image.Image):
            if self.per_img:
                _, brightness, contrast, saturation, hue = self.get_params(self.brightness, self.contrast, self.saturation, self.hue)

                img_transforms = [
                    lambda x: F.adjust_brightness(x, brightness),
                    lambda x: F.adjust_saturation(x, saturation),
                    lambda x: F.adjust_hue(x, hue),
                    lambda x: F.adjust_contrast(x, contrast)
                ]
                random.shuffle(img_transforms)

                jittered_clip = []
                for img in imgs:
                    jittered_img = img
                    for func in img_transforms:
                        jittered_img = func(jittered_img)
                    jittered_clip.append(jittered_img)
                return jittered_clip
            else:
                trans = T.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)
                return [trans(img) for img in imgs]
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image but got list of {0}'.format(type(imgs[0])))


def aug3_transform(video_size):
    return transforms.Compose([
        RandomResizedCrop(video_size),
        RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([RandomTranslateWithReflect(4)], p=0.8),
        transforms.RandomApply([ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8),
        RandomGrayscale(p=0.25),
        ToTensor(),
        Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        RandomErasing(),
        ToClip()
    ])


def aug2_transform(video_size):
    return transforms.Compose([
        Resize(int((96 / 84) * video_size)),
        RandomHorizontalFlip(p=0.5),
        RandomCrop(video_size),
        ToTensor(),
        Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        RandomErasing(),
        ToClip()
    ])


def aug1_transform(video_size):
    return transforms.Compose([
        Resize(int((96 / 84) * video_size)),
        RandomHorizontalFlip(p=0.5),
        RandomCrop(video_size),
        ToTensor(),
        Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ToClip()
    ])


def aug0_transform(video_size):
    return transforms.Compose([
        Resize(int((96 / 84) * video_size)),
        CenterCrop(video_size),
        ToTensor(),
        Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ToClip()
    ])


def select_transform(aug_name='aug0', video_size=224):
    if aug_name == 'aug0':
        return aug0_transform(video_size)
    elif aug_name == 'aug1':
        return aug1_transform(video_size)
    elif aug_name == 'aug2':
        return aug2_transform(video_size)
    elif aug_name == 'aug3':
        return aug3_transform(video_size)
    else:
        raise ValueError('transform error')
