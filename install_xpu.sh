#!/bin/bash
# ACE-Step XPU Installation Script
#
# Complete installation script for Intel XPU users
#
# This script installs ACE-Step with Intel XPU support
# using PyTorch XPU nightly builds.

set -e

echo "=========================================="
echo "ACE-Step XPU Installation"
echo "=========================================="
echo ""

# Step 1: Create conda environment
echo "Step 1: Creating conda environment..."
conda create -n acestep-xpu python=3.10 -y
conda activate acestep-xpu
echo "✓ Environment created and activated"
echo ""

# Step 2: Install from XPU index (with --pre)
echo "Step 2: Installing XPU-optimized packages..."
echo "This may take several minutes..."
pip install --pre \
  -r requirements-xpu-part1.txt \
  --index-url https://download.pytorch.org/whl/nightly/xpu
echo "✓ XPU packages installed"
echo ""

# Step 3: Install missing packages from PyPI
echo "Step 3: Installing packages not available in XPU index..."
pip install -r requirements-xpu-part2.txt
echo "✓ Additional packages installed"
echo ""

# Step 4: Install ACE-Step
echo "Step 4: Installing ACE-Step..."
pip install -e ".[xpu]"
echo "✓ ACE-Step installed"
echo ""

# Step 5: Verify installation
echo "Step 5: Verifying installation..."
echo ""

# Check PyTorch version
echo "PyTorch version:"
python -c "import torch; print(f'  {torch.__version__}')"
echo ""

# Check XPU availability
echo "XPU availability:"
python -c "import torch; print(f'  Available: {torch.xpu.is_available() if hasattr(torch, \"xpu\") else False}')"
if python -c "import torch; exit(0 if hasattr(torch, 'xpu') and torch.xpu.is_available() else 1)"; then
    echo "  XPU device info:"
    python -c "import torch; print(f'    Device count: {torch.xpu.device_count() if hasattr(torch, \"xpu\") else 0}'); print(f'    Device name: {torch.xpu.get_device_name(0) if hasattr(torch, \"xpu\") and torch.xpu.device_count() > 0 else \"N/A\"}')"
fi
echo ""

# Show package versions
echo "Key package versions:"
pip list | grep -E "^torch |^transformers |^diffusers |^datasets |^accelerate" || true
echo ""

echo "=========================================="
echo "Installation complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Activate the environment: conda activate acestep-xpu"
echo "  2. Run inference: acestep --port 7865"
echo "  3. Or run training: python trainer.py --dataset_path <path>"
echo ""
echo "For more information, see INSTALL_XPU.md"
