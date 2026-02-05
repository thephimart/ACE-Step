# ACE-Step XPU Installation Guide

## **IMPORTANT: Separate Installation Path for XPU Users**

XPU users have a **completely different installation path** from CUDA/CPU/MPS users.

---

## XPU Installation (Complete - from XPU Index)

### Step 1: Install All Available Packages from XPU Index

```bash
# Install from XPU index with --pre flag to get pre-release versions
pip install --pre \
  torch torchvision torchaudio \
  transformers tokenizers \
  datasets accelerate \
  numpy matplotlib tqdm \
  fastapi uvicorn click \
  soundfile \
  --index-url https://download.pytorch.org/whl/nightly/xpu
```

**This installs these XPU-optimized packages:**
- torch (XPU-enabled)
- torchvision (XPU-enabled)
- torchaudio (XPU-enabled)
- transformers
- tokenizers
- datasets
- plus all XPU-specific dependencies automatically (triton-xpu, intel-sycl-rt, onemkl-*, etc.)

### Step 2: Install Missing Packages from PyPI

These packages are **NOT available** in the XPU index:

```bash
pip install \
  "diffusers>=0.33.0" \
  gradio \
  "librosa==0.11.0" \
  "loguru==0.7.3" \
  "pypinyin==0.53.0" \
  "pytorch_lightning==2.5.1" \
  "py3langid==0.3.0" \
  "hangul-romanize==0.1.0" \
  "num2words==0.5.14" \
  "spacy==3.8.4" \
  cutlet \
  "fugashi[unidic-lite]" \
  peft \
  tensorboard \
  tensorboardX
```

### Step 3: Install FFmpeg6 (Required for torchcodec)

**IMPORTANT:** torchcodec requires FFmpeg with matching libraries. If pre-built FFmpeg packages don't work, you may need to build from source:

```bash
# Option 1: Try pre-built package first
sudo apt-get update && sudo apt-get install -y ffmpeg6

# Option 2: Build FFmpeg6 from source (if package doesn't work)
sudo apt-get install -y build-essential nasm yasm
cd /tmp
wget https://ffmpeg.org/releases/ffmpeg-6.0.tar.bz2
tar xjf ffmpeg-6.0.tar.bz2
cd ffmpeg-6.0
./configure --enable-shared --disable-static --prefix=/usr/local
make -j$(nproc)
sudo make install
sudo ldconfig
```

### Step 4: Install torchcodec (Compatible with PyTorch XPU)

**IMPORTANT:** torchcodec must be compatible with your PyTorch version. If pre-built packages fail, build from source:

```bash
# Option 1: Try pre-built package first
pip install torchcodec

# Option 2: Build torchcodec from source (if package doesn't work)
git clone https://github.com/pytorch/torchcodec.git
cd torchcodec
python setup.py develop
```

**Note:** PyTorch XPU nightly builds may require building torchcodec from source due to compatibility issues.

### Step 5: Set Environment Variable for Large Memory Allocations

**IMPORTANT:** To enable XPU memory allocations over 4GB, you must set this environment variable:

```bash
# Add to ~/.bashrc for persistent configuration
echo "export SYCL_UR_USE_LEVEL_ZERO_V2=1" >> ~/.bashrc

# Source bashrc to apply changes in current session
source ~/.bashrc
```

This is required for large model inference and training with ACE-Step on Intel Arc GPUs.

### Step 6: Install ACE-Step

```bash
pip install -e ".[xpu]"
```

### Step 7: Verify XPU Installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'XPU available: {torch.xpu.is_available() if hasattr(torch, \"xpu\") else False}')"
```

---

## CUDA/CPU/MPS Installation (Original - from PyPI)

### Standard Installation

```bash
# Install all packages from PyPI (original requirements.txt)
pip install -r requirements.txt

# Install ACE-Step
pip install -e ".[train]"
```

---

## Complete Installation Script for XPU

Here's a complete script for XPU users:

```bash
#!/bin/bash
set -e

echo "=========================================="
echo "ACE-Step XPU Installation"
echo "=========================================="

# Step 1: Create environment
conda create -n acestep-xpu python=3.10 -y
conda activate acestep-xpu

# Step 2: Install from XPU index (with --pre)
echo ""
echo "Step 1: Installing XPU-optimized packages..."
pip install --pre \
  torch torchvision torchaudio \
  transformers tokenizers \
  datasets accelerate \
  numpy matplotlib tqdm \
  fastapi uvicorn click \
  soundfile \
  --index-url https://download.pytorch.org/whl/nightly/xpu

# Step 3: Install FFmpeg6
echo ""
echo "Step 2: Installing FFmpeg6..."
echo "Note: May need to build from source if package installation fails"
sudo apt-get update && sudo apt-get install -y ffmpeg6

# Step 4: Install torchcodec
echo ""
echo "Step 3: Installing torchcodec..."
echo "Note: May need to build from source if package installation fails"
pip install torchcodec

# Step 5: Install missing packages from PyPI
echo ""
echo "Step 4: Installing packages not available in XPU index..."
pip install \
  "diffusers>=0.33.0" \
  gradio \
  "librosa==0.11.0" \
  "loguru==0.7.3" \
  "pypinyin==0.53.0" \
  "pytorch_lightning==2.5.1" \
  "py3langid==0.3.0" \
  "hangul-romanize==0.1.0" \
  "num2words==0.5.14" \
  "spacy==3.8.4" \
  cutlet \
  "fugashi[unidic-lite]" \
  peft \
  tensorboard \
  tensorboardX

# Step 6: Install ACE-Step
echo ""
echo "Step 5: Installing ACE-Step..."
pip install -e ".[xpu]"

# Step 7: Set environment variable for large memory allocations
echo ""
echo "Step 6: Setting environment variable for XPU large memory allocations..."
echo "export SYCL_UR_USE_LEVEL_ZERO_V2=1" >> ~/.bashrc
echo "✓ Added SYCL_UR_USE_LEVEL_ZERO_V2=1 to ~/.bashrc"
echo "⚠ Please run: source ~/.bashrc (or open a new terminal) to apply changes"

# Step 8: Verify
echo ""
echo "Step 7: Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'XPU available: {torch.xpu.is_available() if hasattr(torch, \"xpu\") else False}')"

echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="
```

Save this as `install_xpu.sh` and run with:
```bash
bash install_xpu.sh
```

**Note:** If torchaudio.save() fails during inference, you may need to:
1. Build FFmpeg6 from source (see Step 3 above)
2. Build torchcodec from source (see Step 4 above)

This is common with PyTorch XPU nightly builds due to version compatibility issues.

---

## Package Source Matrix

| Package | XPU Index (--pre) | PyPI | XPU Users | CUDA/CPU/MPS Users |
|----------|-------------------|--------|------------|-------------------|
| torch | ✅ | ✅ | XPU index | PyPI |
| torchvision | ✅ | ✅ | XPU index | PyPI |
| torchaudio | ✅ | ✅ | XPU index | PyPI |
| transformers | ✅ | ✅ | XPU index | PyPI |
| tokenizers | ✅ | ✅ | XPU index | PyPI |
| datasets | ✅ | ✅ | XPU index | PyPI |
| accelerate | ✅ | ✅ | XPU index | PyPI |
| numpy | ✅ | ✅ | XPU index | PyPI |
| matplotlib | ✅ | ✅ | XPU index | PyPI |
| tqdm | ✅ | ✅ | XPU index | PyPI |
| fastapi | ✅ | ✅ | XPU index | PyPI |
| uvicorn | ✅ | ✅ | XPU index | PyPI |
| click | ✅ | ✅ | XPU index | PyPI |
| soundfile | ✅ | ✅ | XPU index | PyPI |
| torchcodec | ❌ | ❌ | **Build from source** | N/A |
| **diffusers** | ❌ | ✅ | **PyPI** | PyPI |
| **gradio** | ❌ | ✅ | **PyPI** | PyPI |
| **librosa** | ❌ | ✅ | **PyPI** | PyPI |
| **loguru** | ❌ | ✅ | **PyPI** | PyPI |
| **pypinyin** | ❌ | ✅ | **PyPI** | PyPI |
| **pytorch_lightning** | ❌ | ✅ | **PyPI** | PyPI |
| **py3langid** | ❌ | ✅ | **PyPI** | PyPI |
| **hangul-romanize** | ❌ | ✅ | **PyPI** | PyPI |
| **num2words** | ❌ | ✅ | **PyPI** | PyPI |
| **spacy** | ❌ | ✅ | **PyPI** | PyPI |
| **cutlet** | ❌ | ✅ | **PyPI** | PyPI |
| **fugashi** | ❌ | ✅ | **PyPI** | PyPI |
| **peft** | ❌ | ✅ | **PyPI** | PyPI |
| **tensorboard** | ❌ | ✅ | **PyPI** | PyPI |
| **tensorboardX** | ❌ | ✅ | **PyPI** | PyPI |
| **ffmpeg6** | ❌ | ❌ | **System package** | System |

---

## Why --pre Flag?

The `--pre` flag allows installation of pre-release versions from the XPU index. This is **required** because:

1. **XPU support is cutting-edge** - XPU builds are often pre-release
2. **Latest optimizations** - XPU-specific optimizations are in pre-release
3. **Hardware support** - New Intel GPUs may only have pre-release support

**This is standard practice for XPU installations.**

---

## Key Differences: XPU vs CUDA/CPU/MPS

### XPU Installation
- Source: PyTorch XPU nightly index (with --pre)
- Packages: 14 from XPU index + 13 from PyPI
- PyTorch: XPU-enabled nightly builds
- Key feature: XPU-specific optimizations

### CUDA/CPU/MPS Installation
- Source: PyPI
- Packages: All 26 from requirements.txt
- PyTorch: Stable or CUDA-specific builds
- Key feature: Standard PyTorch behavior

---

## Troubleshooting

### Issue 1: Package not found in XPU index

**Symptom:** `ERROR: Could not find a version that satisfies the requirement <package>`

**Solution:** That package is not available in the XPU index. Install it from PyPI:
```bash
pip install <package>
```

### Issue 2: Version conflicts

**Symptom:** `ERROR: Conflicting dependencies`

**Solution:** Use `--pre` flag consistently and install packages in the correct order:
1. First: XPU index packages (with --pre)
2. Second: PyPI packages (missing from XPU index)

### Issue 3: XPU not available

**Symptom:** `torch.xpu.is_available()` returns `False`

**Solution:**
1. Check Intel GPU driver installation
2. Verify oneAPI environment variables are set
3. Run `clinfo` to check GPU availability

### Issue 4: Out of Memory or Allocation Errors (>4GB)

**Symptom:** Memory allocation errors when loading large models or training

**Solution:** Set the SYCL environment variable to enable large memory allocations:
```bash
# Add to ~/.bashrc
echo "export SYCL_UR_USE_LEVEL_ZERO_V2=1" >> ~/.bashrc
source ~/.bashrc
```

This is **required** for ACE-Step which requires large memory allocations for model loading and inference.

### Issue 5: torchcodec ImportError - Missing FFmpeg Libraries

**Symptom:** `ModuleNotFoundError: No module named 'torchcodec'` or `OSError: Could not load libtorchcodec`

**Solution:** Install FFmpeg6 and build torchcodec compatible with your PyTorch version:
```bash
# Install FFmpeg6 (may need to build from source)
sudo apt-get install -y ffmpeg6
# OR build from source (see Step 3 above)

# Install torchcodec (may need to build from source)
pip install torchcodec
# OR build from source (see Step 4 above)
```

**Note:** PyTorch XPU nightly builds often require building both FFmpeg6 and torchcodec from source due to compatibility issues.

---

## Verification Commands

After installation, run these commands to verify:

```bash
# Check PyTorch version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Check XPU availability
python -c "import torch; print(f'XPU available: {torch.xpu.is_available() if hasattr(torch, \"xpu\") else False}')"

# Check package versions
pip list | grep -E "torch|transformers|diffusers|datasets"
```

---

## Summary

**XPU users require a separate installation path:**

1. **6 packages** from XPU index (with --pre flag)
2. **19 packages** from PyPI (not available in XPU index)
3. **Total:** 25 packages (plus 50+ XPU-specific dependencies auto-installed)
4. **Environment variable:** `export SYCL_UR_USE_LEVEL_ZERO_V2=1` (required for >4GB allocations)

**CUDA/CPU/MPS users:**

1. **26 packages** from PyPI (requirements.txt)
2. **Total:** 26 packages

The difference is significant and requires **separate documentation and installation instructions**.
