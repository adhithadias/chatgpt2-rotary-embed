

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install .

find $(python -c "import torch; print(torch.__path__[0])") -name "libc10.so"
export LD_LIBRARY_PATH=/local/scratch/a/kadhitha/workspace/build-nanogpt/.venv/lib/python3.10/site-packages/torch/lib/libc10.so:$LD_LIBRARY_PATH
```