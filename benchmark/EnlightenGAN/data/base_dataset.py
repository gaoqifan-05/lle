import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import random

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

def get_transform(opt):
    transform_list = []
    if opt.resize_or_crop == 'resize_and_crop':
        zoom = 1 + 0.1*radom.randint(0,4)
        osize = [int(400*zoom), int(600*zoom)]
        transform_list.append(transforms.Scale(osize, Image.BICUBIC))
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'crop':
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'scale_width':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.fineSize)))
    elif opt.resize_or_crop == 'scale_width_and_crop':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.loadSize)))
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'resize':
        # 等比降采样：长边对齐 fineSize，不裁切
        transform_list.append(transforms.Lambda(
            lambda img: __resize_long_edge(img, opt.fineSize)))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.RandomHorizontalFlip())

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def __scale_width(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), Image.BICUBIC)


def __resize_long_edge(img, target_size):
    """等比降采样：长边对齐 target_size，保持宽高比，不裁切"""
    ow, oh = img.size
    if max(ow, oh) <= target_size:
        # 即便不缩放也取整到 16 倍数
        w, h = ow // 16 * 16, oh // 16 * 16
        return img.resize((w, h), Image.BICUBIC)
    scale = target_size / max(ow, oh)
    w, h = int(ow * scale) // 16 * 16, int(oh * scale) // 16 * 16
    return img.resize((w, h), Image.BICUBIC)
