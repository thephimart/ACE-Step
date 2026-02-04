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
- accelerate
- numpy
- matplotlib
- tqdm
- fastapi
- uvicorn
- click
- soundfile
- plus all XPU-specific dependencies automatically

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

### Step 3: Install ACE-Step

```bash
pip install -e ".[xpu]"
```

### Step 4: Verify XPU Installation

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

# Step 3: Install missing packages from PyPI
echo ""
echo "Step 2: Installing packages not available in XPU index..."
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

# Step 4: Install ACE-Step
echo ""
echo "Step 3: Installing ACE-Step..."
pip install -e ".[xpu]"

# Step 5: Verify
echo ""
echo "Step 4: Verifying installation..."
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

1. **14 packages** from XPU index (with --pre flag)
2. **13 packages** from PyPI (not available in XPU index)
3. **Total:** 27 packages

**CUDA/CPU/MPS users:**

1. **26 packages** from PyPI (requirements.txt)
2. **Total:** 26 packages

The difference is significant and requires **separate documentation and installation instructions**.
