# Video Inpainting by Frame Alignment with Deformable Convolution

## Dataset
- DAVIS https://davischallenge.org/davis2017/code.html (Semi-supervised, 480p)

## Environment

We provide a Dockerfile. You can build and run it as follows.

```sh
cd dconv-vi
docker build . --tag dconv-vi
docker run -it --shm-size 64g --gpus all -v /home/{dir to be mounted}:/mnt dconv-vi bash
```
See here https://docs.docker.com/storage/bind-mounts/ to check how to mount your host storage into the container.

## Reference
Our code is based on Copy-and-Paste-Networks-for-Deep-Video-Inpainting (ICCV 2019) https://github.com/shleecs/Copy-and-Paste-Networks-for-Deep-Video-Inpainting .

We borrowed some code from the repositories below.
- pytorch_ssim: https://github.com/Po-Hsun-Su/pytorch-ssim (pytorch_ssim/)
- EDVR: https://github.com/xinntao/EDVR/tree/old_version/ (EDVR_arch.py, arch_util.py)
- DCNv2 https://github.com/CharlesShang/DCNv2 (dcn/)
