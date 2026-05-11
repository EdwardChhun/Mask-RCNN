# CUDA Setup — Native Windows

Step-by-step to bring this repo up on a Windows PC with an NVIDIA GPU. Expect a ~12-hour CPU run on the Mac to drop to **15–30 minutes** on a mid-range CUDA GPU.

## 0. Verify the GPU

Open PowerShell:

```powershell
nvidia-smi
```

You should see your GPU listed with a driver version. If not, install the latest [NVIDIA Game Ready / Studio Driver](https://www.nvidia.com/download/index.aspx) first. **You do not need the full CUDA Toolkit just to run PyTorch** — the toolkit is only needed if you build Detectron2 from source (see step 4).

Note the **CUDA Version** shown in `nvidia-smi` (top-right). That's the *maximum* CUDA your driver supports — pick a PyTorch wheel ≤ that.

## 1. Install Python 3.10

Detectron2 works best on **Python 3.10**. Download from [python.org](https://www.python.org/downloads/release/python-31011/) — check "Add python.exe to PATH" during install.

```powershell
python --version    # should print 3.10.x
```

## 2. Install Visual Studio Build Tools

Detectron2's CUDA ops compile from C++ on Windows. You need MSVC.

1. Download [Build Tools for Visual Studio 2022](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
2. In the installer, select **"Desktop development with C++"**. Default components are fine.
3. After install, open the **"x64 Native Tools Command Prompt for VS 2022"** (Start menu) — you'll run all subsequent commands in this shell so `cl.exe` is on PATH.

## 3. Clone the repo and create a venv

```powershell
cd C:\dev    # or wherever you keep code
git clone <your-repo-url> Mask-RCNN
cd Mask-RCNN

python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
```

## 4. Install PyTorch with CUDA

Pick the CUDA version matching your driver (12.1 is a safe default in 2026):

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Verify:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

You should see `True` and your GPU name. If `False`, your driver is too old for cu121 — drop to cu118 (`...whl/cu118`).

## 5. Install Detectron2

There are no official Windows wheels, so build from source. The MSVC shell from step 2 is required here.

```powershell
pip install ninja
pip install "git+https://github.com/facebookresearch/detectron2.git"
```

This compiles for ~5–10 minutes. If it fails:
- Confirm you're in the **x64 Native Tools** prompt (not regular PowerShell).
- Confirm `torch.cuda.is_available()` is True before installing.
- Set `set DISTUTILS_USE_SDK=1` and retry.

## 6. Install remaining deps

```powershell
pip install opencv-python pycocotools tqdm matplotlib scikit-learn
```

## 7. Download the TACO dataset

The dataset is not in this repo. Use TACO's official downloader:

```powershell
cd C:\dev
git clone https://github.com/pedropro/TACO.git TACO_repo
cd TACO_repo
pip install -r requirements.txt
python download.py
```

This pulls ~2.5 GB of images into `TACO_repo\data\batch_*`. Takes 15–60 min depending on bandwidth.

## 8. Point the training code at the dataset

The repo reads `TACO_DATA_DIR` from the environment (with a Mac fallback). Set it for this shell:

```powershell
$env:TACO_DATA_DIR = "C:\dev\TACO_repo\data"
```

Or set it permanently:

```powershell
[System.Environment]::SetEnvironmentVariable("TACO_DATA_DIR", "C:\dev\TACO_repo\data", "User")
```

(Reopen the terminal after a permanent set.)

## 9. (Optional) Bump batch size for GPU

`training.py` has `cfg.SOLVER.IMS_PER_BATCH = 4`, sized for CPU. With GPU you can push higher:

| VRAM | Suggested `IMS_PER_BATCH` |
|---|---|
| 6 GB (e.g. GTX 1660) | 4 (keep as-is) |
| 8 GB (e.g. RTX 3060 Ti / 4060) | 8 |
| 12 GB+ (e.g. RTX 3060 12GB / 4070+) | 16 |

If you bump it, scale `BASE_LR` linearly (8 → 0.002, 16 → 0.004) and you can drop `MAX_ITER` proportionally.

## 10. Run training

```powershell
cd C:\dev\Mask-RCNN
venv\Scripts\activate
python training.py
```

You should see `torch.cuda.is_available() = True` in the first lines of output, and per-iter time around 0.3–0.5s instead of ~14s.

Checkpoints land in `output/` every 500 iters. The final eval pass runs automatically and writes `output/evaluation_summary.json`.

## Troubleshooting

- **`torch.cuda.is_available()` is False after install** — driver too old, or you installed CPU-only torch. Reinstall with `pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu121`.
- **Detectron2 build fails with `cl.exe not found`** — you're not in the x64 Native Tools prompt. Open it from the Start menu.
- **Out of memory during training** — lower `IMS_PER_BATCH` or `ROI_HEADS.BATCH_SIZE_PER_IMAGE`.
- **Image files not found** — confirm `TACO_DATA_DIR` is set in *this* shell (`echo $env:TACO_DATA_DIR`). The training script filters to images present on disk, so a wrong path silently yields zero training images.
