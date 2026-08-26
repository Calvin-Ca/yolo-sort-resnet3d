import json
import torch
import torch.nn.functional as F

from spatial_transforms import Normalize
import random
import numpy as np
from model import generate_model
from torchvision import transforms
import argparse
from munch import DefaultMunch


def resume_model(resume_path, arch, model):
    print('loading checkpoint {} model'.format(resume_path))
    checkpoint = torch.load(resume_path, map_location='cpu')
    assert arch == checkpoint['arch']

    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['state_dict'])

    return model


def get_normalize_method(mean, std, no_mean_norm, no_std_norm):
    if no_mean_norm:
        if no_std_norm:
            return Normalize([0, 0, 0], [1, 1, 1])
        else:
            return Normalize([0, 0, 0], std)
    else:
        if no_std_norm:
            return Normalize(mean, [1, 1, 1])
        else:
            return Normalize(mean, std)

class TemporalSubsampling(torch.nn.Module):
    def __init__(self, stride):
        super(TemporalSubsampling, self).__init__()
        self.stride = stride

    def forward(self, inputs):
        return inputs[:, :, ::self.stride, :, :]

class VideoPreProcess(torch.nn.Module):
    def __init__(self, opt):
        super(VideoPreProcess, self).__init__()
        self.norm = get_normalize_method(opt.mean, opt.std, opt.no_mean_norm, opt.no_std_norm)
        self.resize = transforms.Resize([opt.sample_size], antialias=False)
        self.center_crop = transforms.CenterCrop(opt.sample_size)
        #opt.sample_t_stride = 2
        #self.downsample = TemporalSubsampling(opt.sample_t_stride)

    def forward(self, inputs):
        # transform from (N, C, T, H, W) to (N*T, C, H, W)
        N, C, T, H, W = inputs.size()
        inputs = inputs.transpose(1, 2).contiguous() # (N, T, C, H, W)
        inputs = inputs.view(-1, C, H, W) # (N*T, C, H, W)
        inputs = self.resize(inputs)
        inputs = self.center_crop(inputs)
        inputs = inputs / 255.0 # ToTensor() default is 255.0
        _, _, H, W = inputs.size()
        # opt.value_scale default is 1.0, so we don't need to scale here
        inputs = self.norm(inputs)
        inputs = inputs.view(N, T, C, H, W)
        inputs = inputs.transpose(1, 2).contiguous()
        #inputs = self.downsample(inputs)
        return inputs




class End2End3DResNet(torch.nn.Module):
    def __init__(self, model, preprocess):
        super(End2End3DResNet, self).__init__()
        self.model = model
        self.preprocess = preprocess

    def forward(self, inputs, do_preprocess: bool = True):
        if do_preprocess:
            inputs = self.preprocess(inputs)
        outputs = self.model(inputs)
        outputs = F.softmax(outputs, dim=1).cpu()
        return outputs




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt_path', required=True, type=str, help='Path of opt file.')
    parser.add_argument('--resume_path', required=True, type=str, help='Path of resume file.')
    parser.add_argument('--save_path', required=True, type=str, help='Path of save file.')

    args = parser.parse_args()

    opt_path = args.opt_path
    with open(opt_path, 'r') as f:
        opt = json.load(f)
    opt = DefaultMunch.fromDict(opt)
    opt.resume_path = args.resume_path

    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)

    opt.device = torch.device('cuda:0')

    model = generate_model(opt)
    if opt.batchnorm_sync:
        assert opt.distributed, 'SyncBatchNorm only supports DistributedDataParallel.'
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = resume_model(opt.resume_path, opt.arch, model)
    model.to(opt.device)
    print(model)
    print(opt)

    model.eval()

    with torch.no_grad():
        raw_inputs = torch.randint(0, 256, (2, 3, 16, 140, 140)).to(opt.device)
        print(raw_inputs[:, :, :, :, 0])

        preprocess = VideoPreProcess(opt)
        inputs = preprocess(raw_inputs)
        outputs = model(inputs)
        outputs = F.softmax(outputs, dim=1).cpu()
        print(outputs)

        #script_model = torch.jit.script(model)
        #script_preprocess = torch.jit.script(preprocess)
        end2end = End2End3DResNet(model, preprocess)
        end2end = torch.jit.script(end2end)
        outputs = end2end(raw_inputs)
        print(outputs.shape)
        print(outputs)
        torch.jit.save(end2end, f"{args.save_path}.end2end.pt")


