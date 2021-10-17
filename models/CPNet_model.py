from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F

from vgg import Vgg16, gram_matrix
from .model_module import *
from itertools import chain

import EDVR_arch

sys.path.insert(0, '.')
# from .common import *
sys.path.insert(0, '../utils/')


# Alignment Encoder
class A_Encoder(nn.Module):
    def __init__(self):
        super(A_Encoder, self).__init__()
        self.conv12 = Conv2d(4, 64, kernel_size=5, stride=2, padding=2, activation=nn.ReLU()) # 2
        self.conv2 = Conv2d(64, 64, kernel_size=3, stride=1, padding=1, activation=nn.ReLU()) # 2
        self.conv23 = Conv2d(64, 128, kernel_size=3, stride=2, padding=1, activation=nn.ReLU()) # 4
        self.conv3 = Conv2d(128, 128, kernel_size=3, stride=1, padding=1, activation=nn.ReLU()) # 4
        self.conv34 = Conv2d(128, 256, kernel_size=3, stride=2, padding=1, activation=nn.ReLU()) # 8
        self.conv4a = Conv2d(256, 256, kernel_size=3, stride=1, padding=1, activation=nn.ReLU()) # 8
        self.conv4b = Conv2d(256, 256, kernel_size=3, stride=1, padding=1, activation=nn.ReLU()) # 8
        init_He(self)
        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        
    def forward(self, in_f, in_v):
        f = (in_f - self.mean) / self.std
        x = torch.cat([f, in_v], dim=1)
        x = F.upsample(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = self.conv12(x)
        x = self.conv2(x)
        x = self.conv23(x)
        x = self.conv3(x)
        x = self.conv34(x)
        x = self.conv4a(x)
        x = self.conv4b(x)
        return x

# Alignment Regressor
class A_Regressor(nn.Module):
    def __init__(self):
        super(A_Regressor, self).__init__()
        self.conv45 = Conv2d(512, 512, kernel_size=3, stride=2, padding=1, activation=nn.ReLU()) # 16
        self.conv5a = Conv2d(512, 512, kernel_size=3, stride=1, padding=1, activation=nn.ReLU()) # 16
        self.conv5b = Conv2d(512, 512, kernel_size=3, stride=1, padding=1, activation=nn.ReLU()) # 16
        self.conv56 = Conv2d(512, 512, kernel_size=3, stride=2, padding=1, activation=nn.ReLU()) # 32
        self.conv6a = Conv2d(512, 512, kernel_size=3, stride=1, padding=1, activation=nn.ReLU()) # 32
        self.conv6b = Conv2d(512, 512, kernel_size=3, stride=1, padding=1, activation=nn.ReLU()) # 32
        init_He(self)
        
        self.fc = nn.Linear(512, 6)
        self.fc.weight.data.zero_()
        self.fc.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32))

    def forward(self, feat1, feat2):
        x = torch.cat([feat1, feat2], dim=1)
        x = self.conv45(x)
        x = self.conv5a(x)
        x = self.conv5b(x)
        x = self.conv56(x)
        x = self.conv5a(x)
        x = self.conv5b(x)

        x = F.avg_pool2d(x, x.shape[2])
        x = x.view(-1, x.shape[1])

        theta = self.fc(x)
        theta = theta.view(-1, 2, 3)

        return theta

# Encoder (Copy network)
class Encoder(nn.Module):
    def __init__(self, nf=128):
        super(Encoder, self).__init__()
        self.conv12 = Conv2d(4, 64, kernel_size=5, stride=2, padding=2, activation=nn.ReLU()) # 2
        self.conv2 = Conv2d(64, 64, kernel_size=3, stride=1, padding=1, activation=nn.ReLU()) # 2
        self.conv23 = Conv2d(64, nf, kernel_size=3, stride=2, padding=1, activation=nn.ReLU()) # 4
        self.conv3 = Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, activation=nn.ReLU()) # 4
        self.value3 = Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, activation=None) # 4
        init_He(self)
        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        
    def forward(self, in_f, in_v):
        f = (in_f - self.mean) / self.std
        x = torch.cat([f, in_v], dim=1)
        x = self.conv12(x)
        x = self.conv2(x)
        x = self.conv23(x)
        x = self.conv3(x)
        v = self.value3(x)
        return v

# Decoder (Paste network)
class Decoder(nn.Module):
    def __init__(self, nf=128):
        super(Decoder, self).__init__()
        nf2 = 2*nf +1
        self.conv4 = Conv2d(nf2, nf2, kernel_size=3, stride=1, padding=1, activation=nn.ReLU())
        self.conv5_1 = Conv2d(nf2, nf2, kernel_size=3, stride=1, padding=1, activation=nn.ReLU())
        self.conv5_2 = Conv2d(nf2, nf2, kernel_size=3, stride=1, padding=1, activation=nn.ReLU())
        
        # dilated convolution blocks
        self.convA4_1 = Conv2d(nf2, nf2, kernel_size=3, stride=1, padding=2, D=2, activation=nn.ReLU())
        self.convA4_2 = Conv2d(nf2, nf2, kernel_size=3, stride=1, padding=4, D=4, activation=nn.ReLU())
        self.convA4_3 = Conv2d(nf2, nf2, kernel_size=3, stride=1, padding=8, D=8, activation=nn.ReLU())
        self.convA4_4 = Conv2d(nf2, nf2, kernel_size=3, stride=1, padding=16, D=16,activation=nn.ReLU())

        self.conv3c = Conv2d(nf2, nf2, kernel_size=3, stride=1, padding=1, activation=nn.ReLU()) # 4
        self.conv3b = Conv2d(nf2, nf, kernel_size=3, stride=1, padding=1, activation=nn.ReLU()) # 4
        self.conv3a = Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, activation=nn.ReLU()) # 4
        self.conv32 = Conv2d(nf, 64, kernel_size=3, stride=1, padding=1, activation=nn.ReLU()) # 2
        self.conv2 = Conv2d(64, 64, kernel_size=3, stride=1, padding=1, activation=nn.ReLU()) # 2
        init_He(self)
        self.conv21 = Conv2d(64, 3, kernel_size=5, stride=1, padding=2, activation=None) # 1

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))
    
    def forward(self, x):
        x = self.conv4(x)
        x = self.conv5_1(x)
        x = self.conv5_2(x)

        x = self.convA4_1(x)
        x = self.convA4_2(x)
        x = self.convA4_3(x)
        x = self.convA4_4(x)

        x = self.conv3c(x)
        x = self.conv3b(x)
        x = self.conv3a(x)
        x = F.upsample(x, scale_factor=2, mode='nearest') # 2
        x = self.conv32(x)
        x = self.conv2(x)
        x = F.upsample(x, scale_factor=2, mode='nearest') # 2
        x = self.conv21(x)

        p = (x *self.std) + self.mean
        return p

class PCD_Align(nn.Module):
    def __init__(self, nf=64):
        super(PCD_Align, self).__init__()
        self.nf = nf
        #### extract features (for each frame)
        self.fea_L2_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L2_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.fea_L3_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L3_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        init_He(self,nonlinearity="leaky_relu",a=0.1)

        self.pcd_align = EDVR_arch.PCD_Align(nf=nf)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        B, N, C, H, W = x.size()  # N video frames
        #### extract LR features
        # L1
        L1_fea = x.view(-1, C, H, W)
        # L2
        L2_fea = self.lrelu(self.fea_L2_conv1(L1_fea))
        L2_fea = self.lrelu(self.fea_L2_conv2(L2_fea))
        # L3
        L3_fea = self.lrelu(self.fea_L3_conv1(L2_fea))
        L3_fea = self.lrelu(self.fea_L3_conv2(L3_fea))

        L1_fea = L1_fea.view(B, N, -1, H, W)
        L2_fea = L2_fea.view(B, N, -1, H // 2, W // 2)
        L3_fea = L3_fea.view(B, N, -1, H // 4, W // 4)

        #### pcd align
        # ref feature list
        ref_fea_l = [
            L1_fea[:, 0, :, :, :].clone(), L2_fea[:, 0, :, :, :].clone(),
            L3_fea[:, 0, :, :, :].clone()
        ]
        aligned_fea = [] # N-1 *( B, C, H, W)
        for i in range(1,N):
            nbr_fea_l = [
                L1_fea[:, i, :, :, :].clone(), L2_fea[:, i, :, :, :].clone(),
                L3_fea[:, i, :, :, :].clone()
            ]
            aligned_fea.append(self.pcd_align(nbr_fea_l, ref_fea_l))
        # aligned_fea = torch.stack(aligned_fea, dim=1)  # [B, N-1, C, H, W]

        return aligned_fea

# Context Matching Module
class CM_Module(nn.Module):
    def __init__(self,args,nf=128):
        super(CM_Module, self).__init__()
        self.args = args
        if self.args.pcd_align:
            self.pcd_align = PCD_Align(nf=nf)
    def masked_softmax(self, vec, mask, dim):
        masked_vec = vec * mask # b,1,r,1,1 
        max_vec = torch.max(masked_vec, dim=dim, keepdim=True)[0]
        exps = torch.exp(masked_vec-max_vec)
        masked_exps = exps * mask
        masked_sums = masked_exps.sum(dim, keepdim=True)
        zeros = (masked_sums <1e-4)        
        masked_sums += zeros.float()
        return masked_exps/masked_sums

    def forward(self, values, tvmap, rvmaps, target_idx = -1):

        B, C, H, W = values[0].size()
        # t_feat: target feature
        t_feat = values[0] # b,c,h,w
        # r_feats: refetence features
        r_feats = values[1:] # b,c,r,h,w c=128
        
        B, Cv, H, W = r_feats[0].size()
        T = len(r_feats)
        # vmap: visibility map
        # tvmap: target visibility map
        # rvmap: reference visibility map
        # gs: cosine similarity
        # c_m: c_match
        gs_,vmap_ = [], []
        tvmap_t = (F.upsample(tvmap, size=(H, W), mode='bilinear', align_corners=False)>0.5).float() # b,c,h,w
        for r in range(T):
            rvmap_t = (F.upsample(rvmaps[r], size=(H, W), mode='bilinear', align_corners=False)>0.5).float() # b,c,h,w
            # vmap: visibility map
            vmap = tvmap_t * rvmap_t                                            # b,c,h,w
            gs = (vmap * t_feat * r_feats[r]).sum((-1,-2,-3))       # b         Global similarity θ

            # valid sum
            v_sum = vmap[:,0].sum((-1,-2)) # b,c,h,w →　b,h,w　→ b    number of pixels visible in both target and ref
            zeros = (v_sum <1e-4)
            gs[zeros] = 0                  # gs = 0 when v_sum == 0
            v_sum += zeros.float()         # avoid 0 div
            gs = gs / v_sum / C            # mean in visible area
            gs = gs.view(B,1,1,1)
            gs_.append(gs)
            vmap_.append(rvmap_t)

        gss = torch.stack(gs_, dim=2)           # b,1,1,1 →　 b,1,r,1,1
        if self.args.gs_max is not 0:
            gss /= (torch.clamp(torch.max(gss,dim=2,keepdim=True)[0], min=0.1) / self.args.gs_max)
        vmaps = torch.stack(vmap_, dim=2)       # b,c,h,w →　 b,c,r,h,w


        if self.args.pcd_align:
            still_inv = 1 - tvmap_t
            align_target = t_feat * tvmap_t
            
            neighbors = []
            for i in range(max(target_idx, T - target_idx)):
                if target_idx - i -1 >=0:
                    neighbors.append(target_idx - i - 1)
                if target_idx + i < T:
                    neighbors.append(target_idx + i)
            for i in neighbors:
                new_vis = still_inv * vmap_[i]
                align_target = align_target + new_vis * r_feats[i]
                still_inv = still_inv - new_vis
            r_feats = torch.stack([align_target.detach()]+r_feats,dim=1) # b,c,h,w → b*t, r, c, h, w /b*t, r+1, 128, 64, 64
            # align_target_fea
            r_feats = self.pcd_align(r_feats) # r *( B, C, H, W)


        r_feats = torch.stack(r_feats, dim=2) # r *( B, C, H, W) →　( B, C, r, H, W)
        #weighted pixelwise masked softmax
        c_match = self.masked_softmax(gss, vmaps, dim=2).detach()
        c_out = torch.sum(r_feats * c_match, dim=2)

        # c_mask
        c_mask = (c_match * vmaps)
        c_mask = torch.sum(c_mask,2)
        c_mask = 1. - (torch.mean(c_mask, 1, keepdim=True))
        c_mask = c_mask.detach()
        return torch.cat([t_feat, c_out, c_mask], dim=1), c_mask, gs_


class CPNet(nn.Module):
    def __init__(self, args, mode='Infer'):
        super(CPNet, self).__init__()
        self.args = args
        nf = args.nf
        self.A_Encoder = A_Encoder()  # Align 
        self.A_Regressor = A_Regressor() # output: alignment network

        self.Encoder = Encoder(nf=nf)  # Merge

        self.CM_Module = CM_Module(args,nf=nf)

        self.Decoder = Decoder(nf=nf) 
        
        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('mean3d', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1,1))
        self.mode = mode
        params=0
        for p in self.Encoder.parameters():
            params += p.numel()
        print(params)
        params=0
        for p in self.CM_Module.parameters():
            params += p.numel()
        print(params)
        params=0
        for p in self.Decoder.parameters():
            params += p.numel()
        print(params)
        if self.mode == "Train":
            self.vgg1 = Vgg16(requires_grad=True)
            self.vgg2 = Vgg16()
    def calc_loss(self, frames, GTs, masks):
        batch_size, _, num_frames, height, width = frames.size()
        feats = self.encoding(frames, masks, compute_loss=True)
        L1_ = 0
        L_align_ = 0
        w_variance_ = 0
        h_variance_ = 0
        pred_=[]
        comp_=[]
        c_mask_=[]
        aligned_rs_=[]
        comp_features_ = [[],[],[],[]]
        gt_features_ = [[],[],[],[]]
        comp_gm_ = [[],[],[],[]]
        gt_gm_ = [[],[],[],[]]
        
        for target in range(num_frames):
            out = self.inpainting(feats, frames, masks, frames[:,:,target], masks[:,:,target], GTs[:,:,target], target_num=target, compute_loss=True)
            pred, comp, aligned_rs, rvmaps, c_mask, comp_features, gt_features, comp_gm, gt_gm, w_variance, h_variance, L1, L_align = out
            pred_.append(pred)
            comp_.append(comp)
            c_mask_.append(c_mask)
            aligned_rs_.append(aligned_rs)
            for i, (comp_f, gt_f) in enumerate(zip(comp_features, gt_features)):
                comp_features_[i].append(comp_f)
                gt_features_[i].append(gt_f)
            for i, (comp_g, gt_g) in enumerate(zip(comp_gm, gt_gm)):
                comp_gm_[i].append(comp_g)
                gt_gm_[i].append(gt_g)
            w_variance_ += w_variance
            h_variance_ += h_variance
            L1_ += L1 
            L_align_ += L_align
        
        for i in range(len(comp_features_)):
            comp_features_[i] = torch.stack(comp_features_[i], dim=1)
            gt_features_[i] = torch.stack(gt_features_[i], dim=1)
            comp_gm_[i] = torch.stack(comp_gm_[i], dim=1)
            gt_gm_[i] = torch.stack(gt_gm_[i], dim=1)

        comp_features = comp_features_
        gt_features = gt_features_
        comp_gm = comp_gm_
        gt_gm = gt_gm_
        L1_ /= num_frames
        L_align_ /= num_frames
        w_variance_ /= num_frames
        h_variance_ /= num_frames

        GTs_arr = GTs # b,3,t,h,w 
        return pred_, comp_, L1_, c_mask_, L_align_, comp_features, gt_features, comp_gm, gt_gm,  w_variance_, h_variance_, GTs_arr, aligned_rs_

    def encoding(self, frames, holes, compute_loss=False):
        batch_size, _, num_frames, height, width = frames.size()
        # padding
        (frames, holes), pad = pad_divide_by([frames, holes], 16, (frames.size()[3], frames.size()[4]))
        
        feat_ = []
        for t in range(num_frames):
            feat = self.A_Encoder(frames[:,:,t], holes[:,:,t])
            feat_.append(feat)
        if compute_loss:
            return feat_
        else:
            feats = torch.stack(feat_, dim=2)
            return feats

    def inpainting(self, rfeats, rframes, rholes, frame, hole, gt, target_num = -1, compute_loss=False): 
                # b,c,r,h,w, b,3,r,h,w, b,1,r,h,w, b,3,h,w, b,1,h,w, b,3,h,w
        batch_size, _, height, width = frame.size() # B C H W
        (rframes, rholes, frame, hole, gt), pad = pad_divide_by([rframes, rholes, frame, hole, gt], 16, (height, width))
        tvmap = 1- hole
        if compute_loss: # rfeatsにtfeatsも含まれる
            tfeat = rfeats[target_num]
            ridx = list(range(0, target_num)) + list(range(target_num+1, len(rfeats)))
            c_feat_ = [self.Encoder(frame, hole)]

            aligned_r_ = []
            rvmap_ = []
            trvmap_ = []
            for r in ridx:
                theta_rt = self.A_Regressor(tfeat, rfeats[r])
                grid_rt = F.affine_grid(theta_rt, frame.size())

                # aligned_r: aligned reference frame
                # reference frame affine transformation
                aligned_r = F.grid_sample(rframes[:,:,r], grid_rt)
                
                # aligned_v: aligned reference visiblity map
                # reference mask affine transformation
                aligned_v = F.grid_sample(1-rholes[:,:,r], grid_rt)
                aligned_v = (aligned_v>0.5).float()

                aligned_r_.append(aligned_r)

                #intersection of target and reference valid map
                trvmap_.append(tvmap * aligned_v)
                # compare the aligned frame - target frame 
                
                c_feat_.append(self.Encoder(aligned_r, aligned_v))
                
                rvmap_.append(aligned_v)

        else:
            num_r = rfeats.size()[2] # # of reference frames
            # Target embedding
            tfeat = self.A_Encoder(frame, hole)

            
            # c_feat: Encoder(Copy Network) features
            c_feat_ = [self.Encoder(frame, hole)]
            
            # aligned_r: aligned reference frames
            aligned_r_ = []

            # rvmap: aligned reference frames valid maps
            rvmap_ = []
            
            for r in range(num_r):
                theta_rt = self.A_Regressor(tfeat, rfeats[:,:,r])
                grid_rt = F.affine_grid(theta_rt, frame.size())

                # aligned_r: aligned reference frame
                # reference frame affine transformation
                aligned_r = F.grid_sample(rframes[:,:,r], grid_rt)
                
                # aligned_v: aligned reference visiblity map
                # reference mask affine transformation
                aligned_v = F.grid_sample(1-rholes[:,:,r], grid_rt)
                aligned_v = (aligned_v>0.5).float()

                aligned_r_.append(aligned_r)

                # intersection of target and reference valid map
                # compare the aligned frame - target frame 
                c_feat_.append(self.Encoder(aligned_r, aligned_v))
                
                rvmap_.append(aligned_v)


        aligned_rs = torch.stack(aligned_r_, 2) # b*t, c, r, h, w

        # p_in: paste network input(target features + c_out + c_mask)
        p_in, c_mask, gs_ = self.CM_Module(c_feat_, tvmap, rvmap_, target_idx = int(target_num))
        
        pred = self.Decoder(p_in)
        
        _, _, _, H, W = aligned_rs.shape
        c_mask = (F.upsample(c_mask, size=(H, W), mode='bilinear', align_corners=False))

        comp = pred * (hole) + gt * tvmap


        if pad[2]+pad[3] > 0:
            comp = comp[:,:,pad[2]:-pad[3],:]

        if pad[0]+pad[1] > 0:
            comp = comp[:,:,:,pad[0]:-pad[1]]
            
        comp = torch.clamp(comp, 0, 1)

        if self.mode == "Train":
            comp_features = self.vgg1(comp)
            gt_features = self.vgg2(gt)
            comp_gm = []
            gt_gm = []
            for comp_f, gt_f in zip(comp_features, gt_features):
                comp_gm.append(gram_matrix(comp_f))
                gt_gm.append(gram_matrix(gt_f))

            w_variance = torch.pow(pred[:,:,:,:-1] - pred[:,:,:,1:], 2)
            h_variance = torch.pow(pred[:,:,:-1,:] - pred[:,:,1:,:], 2)

            # Reconstruction loss
            L1 = torch.abs(pred - gt) # |Yt^ - Yt|  b,3,h,w
            # 10MC + 20M(1 - C) + 6(1 - M)  = M(10C + 20 - 20C -6) + 6  = M(14 - 10C) + 6
            L1_weight = hole * (14. - c_mask * 10) + 6.
            L1 = L1 * L1_weight
            L1 = L1.mean(1).mean(1).mean(1)
            w_variance = w_variance.mean(1).mean(1).mean(1)
            h_variance = h_variance.mean(1).mean(1).mean(1)
            # L_align
            trvmaps = torch.stack(trvmap_, dim=2)

            L_align = (trvmaps * torch.abs(aligned_rs - gt.unsqueeze(2))).mean(1).mean(1).mean(1).mean(1)

            # Yt^, Yt^comp, Xr→t, Vt*Vr→t, Cmask 
            return pred, comp, aligned_rs, trvmaps, c_mask, comp_features, gt_features, comp_gm, gt_gm, w_variance, h_variance, L1, L_align
        elif self.mode == "Infer":
            return comp, aligned_r_, gs_
    def parameters_to_train(self):
        return chain(*[child.parameters() for child in self.children() if not isinstance(child,Vgg16)])
         
    def forward(self, *args, compute_loss=False, **kwargs):
        if compute_loss:
            return self.calc_loss(*args)
        else:
            if len(args) == 2:
                return self.encoding(*args)
            else:
                return self.inpainting(*args, **kwargs)