ldd --version保证版本高于2.34

conda create -n env_isaaclab python=3.11 -y

conda activate env_isaaclab

非得装CUDA12.8，以适配对面的显卡。A6000是最新的显卡只适配12.8以上
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_570.86.10_linux.run

/data/why/cuda-12.8设置成你自己的CUDA地址
conda env config vars set PATH=/data/why/cuda-12.8/bin:$PATH -n env_isaaclab
conda env config vars set LD_LIBRARY_PATH=/data/why/cuda-12.8/lib64:$LD_LIBRARY_PATH -n env_isaaclab
conda env config vars set CUDA_HOME=/data/why/cuda-12.8 -n env_isaaclab

conda deactivate

conda activate env_isaaclab

<!-- conda env config vars unset PATH -n env_isaaclab -->

pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

pip install "isaacsim[all,extscache]==5.0.0" --extra-index-url https://pypi.nvidia.com

pip install 'setuptools<81'
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab/
git checkout v2.3.0
./isaaclab.sh --install rl_games
cd ..

git clone https://github.com/sharpa-robotics/sharpa-rl-lab.git
cd sharpa-rl-lab
pip install -e .

# IsaacLab v2.3.0 兼容修改

如果运行 `gen_grasp.py` 时遇到下面这些报错，需要先修改 sharpa-rl-lab 里的源码：

1. `FileNotFoundError: [Errno 2] No such file or directory: 'outputs/'`

在sharpa-rl-lab下创建outputs文件夹



python rl_isaaclab/scripts/gen_grasp.py --task Isaac-Inhand-Rotate-Grasp-Sharpa-Wave-v0 --device cuda:2 --num_envs 512 --headless

CUDA_DEVICE_ORDER=PCI_BUS_ID python rl_isaaclab/scripts/train.py \
  --task Isaac-Inhand-Rotate-Sharpa-Wave-v0 \
  --headless \
  --video \
  --video_length 300 \
  --video_interval 400 \
  --num_envs 16 \
  --device cuda:0

当前命令已经可以跑出视频了
