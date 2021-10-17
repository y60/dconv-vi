# Video Inpainting by Frame Alignment with Deformable Convolution

## Dataset
- DAVIS https://davischallenge.org/davis2017/code.html (Semi-supervised, 480p)

## Environment
A Volta or Pascal NVIDIA GPU is required. Our code does not work on Ampere GPUs.

We provide a Dockerfile. You can build and run it as follows.
```sh
cd dconv-vi
docker build . --tag dconv-vi
docker run -it --shm-size 64g --gpus all -v /home/{dir to be mounted}:/mnt dconv-vi bash
```
See here https://docs.docker.com/storage/bind-mounts/ to check how to mount your host storage into the container.

## Test
The snapshot of a pretrained model is here: https://drive.google.com/file/d/1Fvc4m2a6aQMhYmCPOBja2n5scVpykpAq/view

Run DCPNet_test.py to test our model on DAVIS. 
```
cd dconv-vi
mkdir -p experiment_0/weight
mv 1580616176_700epoch.pt experiment_0/weight
python DCPNet_test.py -name experiment_0 -pcd_align -gs_max 100 -resume recent -data_root /path/to/DAVIS
```

## Reference
Our code is based on Copy-and-Paste-Networks-for-Deep-Video-Inpainting (ICCV 2019) https://github.com/shleecs/Copy-and-Paste-Networks-for-Deep-Video-Inpainting .

We borrowed some code from the repositories below.
- pytorch_ssim: https://github.com/Po-Hsun-Su/pytorch-ssim (pytorch_ssim/)
- EDVR: https://github.com/xinntao/EDVR/tree/old_version/ (EDVR_arch.py, arch_util.py)
- DCNv2 https://github.com/CharlesShang/DCNv2 (dcn/)
