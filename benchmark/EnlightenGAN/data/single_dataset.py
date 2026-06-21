import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image


class SingleDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot)

        self.A_paths = make_dataset(self.dir_A)

        self.A_paths = sorted(self.A_paths)

        self.transform = get_transform(opt)

    def __getitem__(self, index):
        A_path = self.A_paths[index]

        A_img = Image.open(A_path).convert('RGB')
        # 等比降采样：长边对齐 fineSize，保持宽高比，不裁切
        w, h = A_img.size
        max_size = self.opt.fineSize
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            w, h = int(w * scale), int(h * scale)
        # 确保尺寸是 16 的倍数（网络下采样需要）
        w, h = w // 16 * 16, h // 16 * 16
        A_img = A_img.resize((w, h), Image.BICUBIC)

        A_img = self.transform(A_img)

        return {'A': A_img, 'A_paths': A_path}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'SingleImageDataset'
