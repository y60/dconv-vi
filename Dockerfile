FROM nvcr.io/nvidia/pytorch:20.03-py3
RUN apt update
RUN apt install -y libgl1-mesa-dev ffmpeg
