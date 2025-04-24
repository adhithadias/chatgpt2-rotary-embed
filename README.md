## Install custom implementation of rotary embedding locally


```bash
python3 -m venv .venv
source .venv/bin/activate

# to deactivate
deactivate
```

```bash
nvcc --version

# install the correct version of pytorch that supports nvcc version of the system. I have V12.2.140 in my system and installing torch with cu121 seems to work.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install packaging ninja wheel nvtx einops tiktoken

# bypass ImportError: libc10.so: cannot open shared object file: No such file or directory. Needs for importing shared libraries such as libc10.so, libtorch.so, libtorch_cpu.so, etc
# IMPORTANT: Set the path properly
export LD_LIBRARY_PATH="/home/min/a/$USER/scratch-space/workspace/chatgpt2-rotary-embed/.venv/lib/python3.10/site-packages/torch/lib/:$LD_LIBRARY_PATH"

cd rotary
pip uninstall rotary_emb
pip install --no-cache-dir -e .
```

## Execute the scripts

```bash
python train_gpt2.py
python test_llama_rotary_embedding.py
python test_tiny_llama_rotary_embedding.py
```


```bash
nsys profile -f true -o profile_reports/default_rope python test_llama_rotary_embedding.py

nsys profile -f true -o profile_reports/custom_rope python test_tiny_llama_rotary_embedding.py

nsys profile -f true -o profile_reports/rope python train_gpt2.py

nsys stats --force-export=true profile_reports/rope.nsys-rep

# to view aggregate range summary
nsys stats --force-export=true profile_reports/custom_rope.nsys-rep

scp triton:/home/min/a/kadhitha/scratch-space/workspace/chatgpt2-rotary-embed/profile_reports/default_rope.nsys-rep ~/Downloads/ece695aih/project/

scp triton:/home/min/a/kadhitha/scratch-space/workspace/chatgpt2-rotary-embed/profile_reports/custom_rope.nsys-rep ~/Downloads/ece695aih/project/

```


```bash

cd layer_norm && pip install --no-cache-dir -e . && cd ..

# dependencies for tiny llama
pip install xformers lightning flash_attn lightning_utilities

```



```
V100 GPUs does not support sm_80. So I had to force everything to use float32


```
