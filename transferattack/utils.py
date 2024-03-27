import torch
import torchvision.transforms as transforms

from PIL import Image
import numpy as np
import pandas as pd
import sys
import os

sys.path.append('../../pytorchcv')
from pytorchcv.model_provider import get_model as ptcv_get_model

img_height, img_width = 32, 32 
img_max, img_min = 1., 0

test_model = \
[
    "nin_cifar100", 
    "resnet1202_cifar100", 
    "preresnet1001_cifar100",
    "resnext272_2x32d_cifar100", 
    "seresnet542bn_cifar100",
    "sepreresnet542bn_cifar100", 
    "pyramidnet272_a200_bn_cifar100",
    "densenet250_k24_bc_cifar100",
    "xdensenet40_2_k36_bc_cifar100",
    "wrn40_8_cifar100",
    "ror3_164_cifar100",
    "rir_cifar100",
    "shakeshakeresnet26_2x32d_cifar100",
    "diaresnet164bn_cifar100",
    "diapreresnet164bn_cifar100"
]


def load_pretrained_model():

    for model_name in test_model:
        yield model_name, ptcv_get_model(model_name, pretrained=True)


def wrap_model(model):
    """
    Add normalization layer with mean and std in training configuration
    """
    mean = [0.5014, 0.4793, 0.4339]
    std = [0.1998, 0.1963, 0.2025]
    normalize = transforms.Normalize(mean, std)
    return torch.nn.Sequential(normalize, model)


def save_images(output_dir, adversaries, filenames):
    adversaries = (adversaries.detach().permute((0,2,3,1)).cpu().numpy() * 255).astype(np.uint8)
    for i, filename in enumerate(filenames):
        Image.fromarray(adversaries[i]).save(os.path.join(output_dir, filename))

def clamp(x, x_min, x_max):
    return torch.min(torch.max(x, x_min), x_max)


class EnsembleModel(torch.nn.Module):
    def __init__(self, models, mode='mean'):
        super(EnsembleModel, self).__init__()
        self.device = next(models[0].parameters()).device
        for model in models:
            model.to(self.device)
        self.models = models
        self.softmax = torch.nn.Softmax(dim=1)
        self.type_name = 'ensemble'
        self.num_models = len(models)
        self.mode = mode

    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        outputs = torch.stack(outputs, dim=0)
        if self.mode == 'mean':
            outputs = torch.mean(outputs, dim=0)
            return outputs
        elif self.mode == 'ind':
            return outputs
        else:
            raise NotImplementedError


class AdvDataset(torch.utils.data.Dataset):
    def __init__(self, input_dir=None, output_dir=None, targeted=False, eval=False):
        self.targeted = targeted
        self.data_dir = input_dir
        self.f2l = self.load_labels(os.path.join(self.data_dir, "labels.csv"))

        if eval:
            self.data_dir = output_dir
            # load images from output_dir, labels from input_dir/labels.csv
            print('=> Eval mode: evaluating on {}'.format(self.data_dir))
        else:
            self.data_dir = os.path.join(self.data_dir, "images")
            print('=> Train mode: training on {}'.format(self.data_dir))
            print('Save images to {}'.format(output_dir))

    def __len__(self):
        return len(self.f2l.keys())

    def __getitem__(self, idx):
        filename = list(self.f2l.keys())[idx]

        assert isinstance(filename, str)

        filepath = os.path.join(self.data_dir, filename)
        image = Image.open(filepath)
        image = image.resize((img_height, img_width)).convert('RGB')
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        image = np.array(image).astype(np.float32)/255
        image = torch.from_numpy(image).permute(2, 0, 1)
        label = self.f2l[filename]

        return image, label, filename

    def load_labels(self, file_name):
        dev = pd.read_csv(file_name)
        if self.targeted:
            f2l = {dev.iloc[i]['filename']: [dev.iloc[i]['label'],
                                             dev.iloc[i]['targeted_label']] for i in range(len(dev))}
        else:
            f2l = {dev.iloc[i]['filename']: dev.iloc[i]['label']
                   for i in range(len(dev))}
        return f2l


if __name__ == '__main__':
    dataset = AdvDataset(input_dir='./data_targeted',
                         targeted=True, eval=False)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=False, num_workers=0)

    for i, (images, labels, filenames) in enumerate(dataloader):
        print(images.shape)
        print(labels)
        print(filenames)
        break
