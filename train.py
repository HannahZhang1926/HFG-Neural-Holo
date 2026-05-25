import os
import argparse

parser = argparse.ArgumentParser(description='physic-transformer')

# parallel
parser.add_argument('--gpu_list', default='0')
parser.add_argument('--num_workers', default=0)

# Dir
parser.add_argument('--img_dir', type=str, default='./dataset2/train/')
parser.add_argument('--model_dir', type=str, default='./checkpoints')
parser.add_argument('--log_dir', type=str, default='./log')
parser.add_argument('--save_dir', type=str, default='./results_images/waveletTV4/', help='Directory to save input and output images')

# Optimization Setting
parser.add_argument('--batch_size', default=1)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--warmup_lr',default=5e-7)
parser.add_argument('--min_lr',default=5e-6)


parser.add_argument('--tv_weight', type=float, default=1e-4, help='weight for Total Variation Loss')
parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
parser.add_argument('--end_epoch', type=int, default=40, help='epoch number of end training')
parser.add_argument('--seed', type=int, default=1111, help='random seed')

# Network Setting
parser.add_argument("--dataset_average", action="store_true", help="Dataset_average")
parser.add_argument("--layer_num", type=int, default=30, help="Number of layers")
parser.add_argument('--img_res', default=(1072,1920), help='resolution of input image')
parser.add_argument("--distance_range", type=float, default=0.02, help="Distance range")
parser.add_argument("--config", type=str, default="./configs/train/train_stage2.yaml", help="path to config file")

# Optical Setting
parser.add_argument('--img_distance', default=35*1e-2, help='distance from SLM plane to target plane')
parser.add_argument('--wavelength', type=float, nargs='+', default=[632e-9, 520e-9, 450e-9], help='wavelengths for RGB reconstruction (m)')
parser.add_argument('--feature_size', default=8*1e-6, help='SLM pitch')

args, unknown = parser.parse_known_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_list

import torch
torch.backends.cudnn.enabled = False
from torch.utils.data import Dataset, DataLoader
from utils2 import myDataset
from tqdm import tqdm
import scipy.io as scio
import matplotlib.pyplot as plt
import cv2
import torch.nn as nn
from model import *
from FDGNet import FDGNet
from model import ControlLDM, Diffusion
from omegaconf import OmegaConf
from utils.common import instantiate_from_config
import numpy as np
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance
from skimage.metrics import peak_signal_noise_ratio as psnr
import bitsandbytes as bnb

args = parser.parse_args()

class TVLoss(nn.Module):
    def __init__(self, weight=1.0):
        super(TVLoss, self).__init__()
        self.weight = weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = x.size()[1] * (h_x - 1) * w_x
        count_w = x.size()[1] * h_x * (w_x - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

experiment_name = "wavelet_tv4"
model_dir = "%s/%s" % (args.model_dir, experiment_name)
log_file_name = "%s/%s.txt" % (args.log_dir, experiment_name)

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

if __name__ == '__main__':

    train_img = myDataset(args)
    n_gpu = len(args.gpu_list.split(','))
    train_loader = DataLoader(train_img, batch_size=args.batch_size*n_gpu, shuffle=False, num_workers=args.num_workers)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # ==========================================================
    cfg = OmegaConf.load(args.config) 
    cldm: ControlLDM = instantiate_from_config(cfg.model.cldm)
    diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion)
 
    sd = torch.load(cfg.train.sd_path, map_location="cpu", weights_only=False)["state_dict"]
    cldm.load_pretrained_sd(sd)
    del sd
    
    if getattr(cfg.train, "resume", None):
        cldm.load_controlnet_from_ckpt(torch.load(cfg.train.resume, map_location="cpu"))

    if getattr(cfg.train, "vae_resume", None):
        cldm.load_vae_from_ckpt(torch.load(cfg.train.vae_resume, map_location="cpu"))
   
    cldm.eval().to(device)
    diffusion.to(device)
    # ==========================================================
    
    model = FDGNet(
        cldm=cldm,
        size=args.img_res,
        feature_size=args.feature_size,
        distance_range=args.distance_range,
        img_distance=args.img_distance,
        layers_num=args.layer_num
    )
    # model = torch.nn.DataParallel(model).cuda()
    model = model.to(device)
    
    if args.start_epoch != 0:
        model.load_state_dict(torch.load('./checkpoints/{}/net_params_{}.pkl'.format(experiment_name, args.start_epoch)))
        print('Loading model parameters from ', 'net_params_{}.pkl'.format(args.start_epoch))
        
    

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    mse_loss = torch.nn.MSELoss() 
    tv_loss = TVLoss(weight=args.tv_weight).to(device)

    # # ==========================================================
    # calc_lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device)
    # calc_fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    # # ==========================================================

    warmup_epochs =  0 # 

    for epoch_i in range(args.start_epoch + 1, args.end_epoch + 1):
        epoch_save_dir = os.path.join(args.save_dir)
        if not os.path.exists(epoch_save_dir):
            os.makedirs(epoch_save_dir)

        num_iter = len(train_loader)
        pbar = tqdm(range(num_iter))
        

        psnr_list = []
        losses = []
        for iter, batch in enumerate(train_loader):
            img, ikk = batch
            img = img.cuda()
            image = img.clone()

            # ==========================================================
            x_output = model(image, ikk)
            # ==========================================================
            
            loss_MSE = mse_loss(x_output, image)
            loss_TV = tv_loss(x_output)
            
            loss_all = loss_MSE + args.tv_weight*loss_TV
        

            # # ==========================================================
            # with torch.no_grad():
            #     pred_clamped = x_output.clamp(0.0, 1.0)
            #     target_clamped = image.clamp(0.0, 1.0)
                
            #     calc_lpips.update(pred_clamped, target_clamped)
            #     pred_uint8 = (pred_clamped * 255).to(torch.uint8)
            #     target_uint8 = (target_clamped * 255).to(torch.uint8)
                
            #     calc_fid.update(target_uint8, real=True)
            #     calc_fid.update(pred_uint8, real=False)
            # # ==========================================================
            
            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()

            losses.append(loss_all.item())
            pbar.update()

        pbar.close()

        # ==========================================================
        # epoch_lpips = calc_lpips.compute().item()
        # epoch_fid = calc_fid.compute().item()
        
        output_data = "Epoch: %d, Avg Loss: %.8f,psnr: %.4f\n" % (epoch_i, sum(losses)/len(losses),sum(psnr_list) / len(psnr_list))
        output_file = open(log_file_name, 'a')
        output_file.write(output_data)
        output_file.close()
        # ==========================================================

        if epoch_i % 5 == 0:
            print("Save Model to: %s/net_params_%d.pkl" % (model_dir, epoch_i))
            torch.save(model.state_dict(), "%s/net_params_%d.pkl" % (model_dir, epoch_i))
