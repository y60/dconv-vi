# -*- coding: utf-8 -*-

from __future__ import division
import torch
from torch.utils import data
import torch.nn as nn
from PIL import Image
import numpy as np
import tqdm
import os
import argparse
import sys
import pytorch_ssim
ssim_loss = pytorch_ssim.SSIM(window_size = 11)
### My libs
sys.path.append('models/')
from DAVIS_dataset import DAVIS_Test
from train_manager import TrainManager
 
def get_arguments():
    parser = argparse.ArgumentParser(description="CPNet")
    parser.add_argument("-data_root", type=str) # dataset path
    parser.add_argument("-baseline", action='store_true')
    parser.add_argument("-name", type=str, default="train_1")
    parser.add_argument("-pcd_align", action='store_true')
    parser.add_argument("-resume", type=str, default="recent")
    parser.add_argument("-gs_max", type=int, default=100)
    parser.add_argument("-nf", type=int, default=128)
    parser.add_argument("-stride", type=int, default=2, help='every stride-th frame will be used as a ref frame.')
    parser.add_argument("-imset", type=str, default="2016/train.txt")
    parser.add_argument("-frame_ext", type=str, default="jpg")
    parser.add_argument("-postfix", type=str, default="")
    parser.add_argument("-resolution", type=str, default="480p")
    parser.add_argument("-dataset", type=str, default='DAVIS')
    return parser.parse_args()

args = get_arguments()
assert torch.cuda.is_available()
assert args.dataset in {'DAVIS', 'VOS'}


# model
from models.CPNet_model import CPNet
model = nn.DataParallel(CPNet(args, mode="Infer"))
model.cuda()
model.eval()
num_length = 60

# dataset
Pset = DAVIS_Test(root=args.data_root, imset=args.imset, size='half', frameext=args.frame_ext, resolution=args.resolution)
Trainloader = data.DataLoader(Pset, batch_size=1, shuffle=False, num_workers=0, drop_last=True)

# set up
manager = TrainManager(args, model, {})
manager.load_checkpoint(args.resume, model, {})

# out dir
out_dir = os.path.join(args.name,"test")
out_dir = out_dir + "_gsm_" +str(args.gs_max)
out_dir = out_dir + "_%d" %  args.stride 
out_dir = out_dir + "_res_" + args.resolution
out_dir = out_dir + args.postfix


psnr_all = 0
ssim_all = 0


for i, V in enumerate(Trainloader):
    if args.dataset == 'VOS':
        frames, GTs, masks, info = V # b,3,t,h,w / b,1,t,h,w
        frames  = (1-masks)*GTs + masks*torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1,1)
    elif args.dataset == 'DAVIS':
        frames, masks, GTs, info = V  # b,3,t,h,w / b,1,t,h,w
    else:
        exit()
    frames__= frames.clone()
    frames = frames.cuda()
    GTs = GTs.cuda()
    masks = masks.cuda()
    
    seq_name = info['name'][0]
    num_frames = frames.size()[2]
    print(seq_name, frames.size())

    with torch.no_grad():
        rfeats = model(frames, masks)
    frames_ = frames.clone()
    masks_ = masks.clone() 
    index = [f for f in reversed(range(num_frames))]

    vis = torch.sum(masks, (4, 3, 1))*3  #b, t 
    psnr = 0
    ssim = 0

    for t in range(2): # forward : 0, backward : 1
        if t == 1:
            comp0 = frames.clone()
            frames = frames_
            masks = masks_
            index.reverse()

        for f in tqdm.tqdm(index):
            ridx = []
            
            start = f - num_length
            end = f + num_length

            if start < 0:
                end = num_length * 2
            elif end > num_frames:
                start = num_frames - num_length * 2

            start = max(start, 0)
            end = min(end, num_frames)

            # interval: 2
            # ridx: ref ids
            target_index = 0
            for i in range(start, end, args.stride):
                if i != f:
                    ridx.append(i)
            target_index = (f-start+1) // args.stride
            with torch.no_grad():
                comp, aligned_r_, gs_ = model(rfeats[:,:,ridx], frames[:,:,ridx], masks[:,:,ridx], frames[:,:,f], masks[:,:,f], GTs[:,:,f], target_num=target_index)
                
                c_s = comp.shape
                Fs = torch.empty((c_s[0], c_s[1], 1, c_s[2], c_s[3])).float().cuda()
                Hs = torch.zeros((c_s[0], 1, 1, c_s[2], c_s[3])).float().cuda()
                Fs[:,:,0] = comp.detach()
                frames[:,:,f] = Fs[:,:,0]
                masks[:,:,f] = Hs[:,:,0]             
                rfeats[:,:,f] = model(Fs, Hs)[:,:,0]
            if args.baseline:
                save_path = os.path.join(out_dir, seq_name)
            else:
                save_path = os.path.join(out_dir, "%s_%s_%s" %(seq_name, args.name, manager.current_epoch))
            if t == 1:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
            
                est = comp0[:,:, f] * (len(index) - f) / len(index) + comp * f / len(index)
                # psnr
                L2 = torch.pow(GTs[:,:, f] - est, 2) 
                L2 = torch.sum(L2, (3, 2, 1))/3
                L2 /= vis[:, f]
                psnr_ = - 10 * torch.log10(L2)
                psnr += torch.mean(psnr_)
                ssim_ = ssim_loss(GTs[:,:, f], est)
                ssim += ssim_
                est = est.cpu()
                canvas = (est[0].permute(1,2,0).numpy() * 255.).astype(np.uint8)
                
                if canvas.shape[1] % 2 != 0:
                    canvas = np.pad(canvas, [[0,0],[0,1],[0,0]], mode='constant')

                canvas = Image.fromarray(canvas)
                canvas.save(os.path.join(save_path, 'f%03d.png' % (f)))

    psnr /= len(index)
    ssim /= len(index)
    print('PSNR', psnr.item())
    print('SSIM', ssim.item())
    psnr_all += psnr
    ssim_all += ssim

    vid_path = os.path.join(out_dir, "%s_%s_%s.gif" %(seq_name, args.name, manager.current_epoch))
    frame_path = os.path.join(save_path, 'f%03d.png')
    ffmpeg_cmd = f"ffmpeg -framerate 10 -i {frame_path} -filter_complex \"scale=424:-1,split [a][b];[a] palettegen [p];[b][p] paletteuse\" -t 00:00:15.000 {vid_path}   -nostats -loglevel 0 -y"
    print(ffmpeg_cmd)
    os.system(ffmpeg_cmd)
    print('----------------------------------------------------------')

print('mean PSNR', psnr_all.item() / len(Trainloader))
print('mean SSIM', ssim_all.item() / len(Trainloader))
