## Install custom implementation of rotary embedding locally

```bash
nvcc --version

# install the correct version of pytorch that supports nvcc version of the system. I have V12.2.140 in my system and installing torch with cu121 seems to work.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install packaging
pip install ninja
pip install wheel

# bypass ImportError: libc10.so: cannot open shared object file: No such file or directory. Needs for importing shared libraries such as libc10.so, libtorch.so, libtorch_cpu.so, etc
export LD_LIBRARY_PATH="/home/min/a/$USER/scratch-space/workspace/build-nanogpt/.venv/lib/python3.10/site-packages/torch/lib/:$LD_LIBRARY_PATH"

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


