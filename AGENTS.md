# ACE-Step Development Guide

**CRITICAL PRINCIPLE: Add XPU support without breaking any existing functionality.**

This guide provides essential information for agentic coding agents working with the ACE-Step music generation foundation model codebase.

## Primary Development Goal

**Add Intel XPU support to ACE-Step while maintaining 100% backward compatibility with existing NVIDIA CUDA, CPU, and MPS workflows.**

### Requirements

1. **DO NOT BREAK ANY EXISTING CODE**
   - All current NVIDIA/CUDA functionality must continue to work exactly as before
   - CPU fallback must continue to work
   - MPS (Apple Silicon) support must continue to work
   - Existing tests and workflows must pass unchanged

2. **XPU Support Must Be Additive Only**
   - XPU support should be added alongside existing device support, not as a replacement
   - Use conditional branching to detect and handle different device types
   - Default behavior must remain unchanged for existing users

3. **Implementation Approach**
   - Add alternative paths within the framework/pipeline for XPU operations where needed
   - Device detection should be automatic and fallback gracefully
   - Use feature detection (`hasattr(torch, 'xpu')`) rather than hard assumptions
   - Maintain clear separation between device-specific and device-agnostic code

4. **Separate Requirements**
   - Use `requirements.txt` for base dependencies (NVIDIA/CPU/MPS)
   - Use `requirements-xpu.txt` for XPU-specific dependencies
   - XPU installation should be opt-in via `pip install -e ".[xpu]"`

## Project Overview

ACE-Step is an open-source foundation model for music generation that integrates diffusion-based generation with DCAE and transformer architectures. It enables fast, high-quality music synthesis with support for multiple languages, vocal techniques, and advanced control mechanisms.

## Build & Development Commands

### Environment Setup
```bash
# Virtual environment already configured
source .venv/bin/activate
# Python: 3.10.12
# which python: /home/phil/opencode/ACE-Step/.venv/bin/python
# which pip: /home/phil/opencode/ACE-Step/.venv/bin/pip
```

### Installation Options

#### Option 1: Base Installation (NVIDIA/CPU/MPS) - DEFAULT
```bash
# This is the default installation path
# Do NOT modify to require XPU dependencies
pip install -e .

# Install with training dependencies
pip install -e ".[train]"
```

#### Option 2: XPU Support - ADDITIVE ONLY
```bash
# XPU support is opt-in - users must explicitly install
# Install PyTorch with XPU support from Intel XPU nightly builds
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/xpu

# Install ACE-Step with XPU support
pip install -e ".[xpu]"
```

#### Option 3: NVIDIA CUDA Support (Alternative)
```bash
# Windows GPU support (install before pip install)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Windows triton support (for torch_compile)
pip install triton-windows

# Then install ACE-Step
pip install -e "."
```

### Running the Application
```bash
# Basic GUI launch (uses default device detection)
acestep --port 7865

# Advanced options
acestep --checkpoint_path /path/to/checkpoint --device_id 0 --bf16 true --torch_compile true --cpu_offload true --share true
```

### Training Commands
```bash
# Convert data to HuggingFace dataset format
python convert2hf_dataset.py --data_dir "./data" --repeat_count 2000 --output_name "zh_lora_dataset"

# Run LoRA training
python trainer.py \
    --dataset_path "./zh_lora_dataset" \
    --exp_name "my_lora_experiment" \
    --learning_rate 1e-4 \
    --max_steps 2000000 \
    --every_n_train_steps 2000 \
    --devices 1 \
    --lora_config_path "config/zh_rap_lora_config.json"

# Run training with API
python trainer-api.py
```

### Inference Commands
```bash
# Command line inference
python infer.py --checkpoint_path "" --output_path "./output" --device_id 0

# API inference server
python infer-api.py
```

## Code Style Guidelines - CRITICAL FOR XPU INTEGRATION

### Device Detection Pattern - MANDATORY

When adding XPU support, ALWAYS use this pattern:

```python
def get_device(device_id: int = 0) -> torch.device:
    """
    Get device with automatic detection and fallback.

    Priority: CUDA > XPU > MPS > CPU

    CRITICAL: This must maintain backward compatibility.
    Do NOT change default behavior for existing users.
    """
    if torch.cuda.is_available():
        return torch.device(f"cuda:{device_id}")
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        return torch.device(f"xpu:{device_id}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
```

### Memory Management Pattern - MANDATORY

```python
def cleanup_memory():
    """
    Clean up accelerator memory - MUST support all device types.

    CRITICAL: Do NOT assume CUDA is the only GPU.
    """
    # Clean up CUDA memory if available
    if hasattr(torch, 'cuda') and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        logger.info(f"CUDA Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    # Clean up XPU memory if available (ADDITIVE, not replacement)
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        torch.xpu.empty_cache()
        torch.xpu.synchronize()
        allocated = torch.xpu.memory_allocated() / (1024 ** 3)
        reserved = torch.xpu.memory_reserved() / (1024 ** 3)
        logger.info(f"XPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    # Collect Python garbage
    import gc
    gc.collect()
```

### Backend Configuration Pattern - MANDATORY

```python
def configure_backend():
    """
    Configure device-specific backend settings.

    CRITICAL: Only configure backends for devices that are actually available.
    Do NOT break CUDA configuration when adding XPU.
    """
    # CUDA-specific settings (EXISTING - DO NOT MODIFY)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = True

    # XPU-specific settings (ADDITIVE ONLY)
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        logger.info("Intel XPU detected and enabled")
```

### AMP/Autocast Pattern - MANDATORY

```python
def get_device_type():
    """
    Get device type string for autocast.

    CRITICAL: Must return correct device type for autocast context.
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        return "xpu"
    else:
        return "cpu"

device_type = get_device_type()
with torch.amp.autocast(device_type=device_type, dtype=dtype):
    # Training/inference code here
    pass
```

### Training Accelerator Pattern - MANDATORY

```python
def get_lightning_accelerator():
    """
    Get PyTorch Lightning accelerator string.

    CRITICAL: Must return correct accelerator for Lightning trainer.
    """
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        return "xpu"
    elif torch.cuda.is_available():
        return "gpu"
    else:
        return "cpu"

accelerator = get_lightning_accelerator()
trainer = Trainer(
    accelerator=accelerator,
    devices=args.devices,
    # ... other parameters
)
```

### GradScaler Pattern - CRITICAL FOR XPU

```python
def create_grad_scaler(device_type: str, use_amp: bool = True):
    """
    Create gradient scaler with device-specific handling.

    CRITICAL: GradScaler requires FP64 support, which is NOT available on Intel Arc GPUs.
    Must disable GradScaler for XPU to prevent crashes.
    """
    if device_type == "xpu":
        # Intel Arc GPUs don't support FP64, so disable GradScaler
        # Use AMP with torch.autocast instead
        return torch.amp.GradScaler(enabled=use_amp and False)
    else:
        # CUDA/MPS can use GradScaler
        return torch.amp.GradScaler(enabled=use_amp)

# Usage
device_type = get_device_type()
scaler = create_grad_scaler(device_type, use_amp=True)

with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=True):
    output = model(data)
    loss = criterion(output, target)

# Only scale if scaler is enabled
if scaler.is_enabled():
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
else:
    loss.backward()
    optimizer.step()
```

### Python Style
- **Python Version**: 3.10+ recommended
- **Formatting**: Standard PEP 8 style with 4-space indentation
- **Line Length**: No strict limit, but keep code readable
- **Docstrings**: Use triple quotes for module and function documentation

### Import Organization
```python
# Standard library imports first
import os
import re
import time
from typing import Dict, List, Optional

# Third-party imports
import torch
import numpy as np
from loguru import logger
from tqdm import tqdm

# Local imports
from acestep.models.ace_step_transformer import ACEStepTransformer2DModel
from acestep.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
```

### Type Hints
- Use type hints for function signatures and complex data structures
- Import from `typing` module: `Dict`, `List`, `Optional`, `Union`, `Callable`
- Example: `def generate_audio(prompt: str, duration: float) -> torch.Tensor:`

### Naming Conventions
- **Variables**: `snake_case` (e.g., `audio_duration`, `guidance_scale`)
- **Functions**: `snake_case` (e.g., `load_model()`, `process_audio()`)
- **Classes**: `PascalCase` (e.g., `ACEStepPipeline`, `FlowMatchScheduler`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_AUDIO_LENGTH`, `DEFAULT_SAMPLE_RATE`)
- **Files**: `snake_case.py` (e.g., `pipeline_ace_step.py`, `lyric_tokenizer.py`)

### Error Handling
- Use `try-except` blocks for file operations and model loading
- Log errors using `loguru.logger.error()` or `logger.exception()`
- Raise appropriate exceptions with descriptive messages
- Example:
```python
try:
    model = ACEStepTransformer2DModel.from_pretrained(checkpoint_path)
except FileNotFoundError:
    logger.error(f"Checkpoint not found at {checkpoint_path}")
    raise
```

### File Structure Patterns
- **Modules**: Organize under `acestep/` with clear subdirectories
- **Models**: Place in `acestep/models/` with descriptive names
- **Utils**: Place utility functions in `acestep/utils/` or appropriate subdirectories
- **Configs**: Store configuration files in `config/` directory

### Configuration Management
- Use JSON files for model and training configurations
- Store LoRA configs in `config/` directory (e.g., `zh_rap_lora_config.json`)
- Load configurations using standard JSON parsing
- Example config structure:
```json
{
    "r": 256,
    "lora_alpha": 32,
    "target_modules": ["linear_q", "linear_k", "linear_v"],
    "use_rslora": true
}
```

### Logging
- Use `loguru` for logging throughout the codebase
- Import: `from loguru import logger`
- Levels: `logger.debug()`, `logger.info()`, `logger.warning()`, `logger.error()`

### CLI Interface
- Use `click` for command-line interfaces
- Import: `import click`
- Use decorators: `@click.command()`, `@click.option()`
- Example:
```python
@click.command()
@click.option("--device_id", type=int, default=0, help="Device ID")
@click.option("--bf16", type=bool, default=True, help="Use bfloat16 precision")
def main(device_id: int, bf16: bool):
    pass
```

### Memory Optimization Patterns
- Use `cpu_offload` parameter for large models on limited GPU memory
- Implement `torch.compile()` optimization when supported
- Use overlapped decoding for faster inference on long sequences
- Monitor memory usage and implement gradient checkpointing for training

### Model Loading Patterns
- Use HuggingFace's `from_pretrained()` pattern for model loading
- Support automatic model downloading from HuggingFace Hub
- Handle checkpoint path resolution with fallbacks
- Example:
```python
if checkpoint_path and os.path.exists(checkpoint_path):
    model_path = checkpoint_path
else:
    model_path = snapshot_download(repo_id="ACE-Step/ACE-Step-v1-3.5B")
```

## Key Architecture Points

### Core Components
- **ACEStepPipeline**: Main inference pipeline
- **ACEStepTransformer2DModel**: Core transformer model
- **MusicDCAE**: Audio compression/decompression
- **Schedulers**: Flow-matching schedulers (Euler, Heun, PingPong)
- **Lyrics Processing**: Multi-language tokenization and processing

### Language Support
- Supports 19+ languages with specialized tokenizers
- Uses `spacy` for language detection and processing
- Implements language-specific text normalization
- Core languages: English, Chinese, Japanese, Korean, Spanish, Russian

### Training Architecture
- Uses PyTorch Lightning for training framework
- Supports LoRA fine-tuning with PEFT integration
- Implements distributed training with multiple GPU support
- TensorBoard logging for training visualization

### Device Support Architecture
**Current (Before XPU):**
- NVIDIA CUDA (primary)
- CPU fallback
- MPS (Apple Silicon) fallback

**Target (After XPU):**
- NVIDIA CUDA (primary) - MUST CONTINUE TO WORK
- Intel XPU (new, additive) - MUST NOT BREAK CUDA
- CPU fallback - MUST CONTINUE TO WORK
- MPS (Apple Silicon) fallback - MUST CONTINUE TO WORK

**Detection Priority:**
1. CUDA (if available)
2. XPU (if available, AND CUDA not available)
3. MPS (if available, AND CUDA/XPU not available)
4. CPU (fallback)

**API Mapping:**
| Operation | CUDA | XPU | MPS | CPU |
|-----------|------|-----|-----|-----|
| Device Check | `torch.cuda.is_available()` | `torch.xpu.is_available()` | `torch.backends.mps.is_available()` | Always |
| Empty Cache | `torch.cuda.empty_cache()` | `torch.xpu.empty_cache()` | N/A | N/A |
| Synchronize | `torch.cuda.synchronize()` | `torch.xpu.synchronize()` | N/A | N/A |
| Memory Allocated | `torch.cuda.memory_allocated()` | `torch.xpu.memory_allocated()` | N/A | N/A |
| Autocast Device | `"cuda"` | `"xpu"` | `"cpu"` | `"cpu"` |
| Lightning Accel | `"gpu"` | `"xpu"` | `"cpu"` | `"cpu"` |

## Testing

### Validation Approach
The project currently uses minimal formal testing. Validation is performed during training through:
- Training loss monitoring (logged via tensorboard)
- Periodic model checkpoint evaluation
- Manual audio quality assessment

**Note**: No formal pytest or unit test framework is currently configured. Test by running inference examples and comparing outputs.

### XPU Validation Requirements
Before merging any XPU changes, verify:
1. Existing CUDA workflows continue to work identically
2. CPU fallback continues to work
3. MPS (if available) continues to work
4. XPU works as expected when available
5. All existing tests pass
6. No performance regression on CUDA

## Requirements Files

### requirements.txt
- Base dependencies for NVIDIA/CPU/MPS support
- DO NOT add XPU-specific dependencies here
- This is the default installation path

### requirements-xpu.txt
- XPU-specific dependencies only
- Installed via `pip install -e ".[xpu]"`
- Only affects users who opt-in to XPU support

### setup.py
```python
extras_require={
    "train": [
        "peft",
        "tensorboard",
        "tensorboardX"
    ],
    "xpu": [],  # No additional packages needed - XPU is built into PyTorch XPU builds
}
```

## Development Checklist for XPU Changes

Before committing any code that adds XPU support:

- [ ] Existing CUDA functionality tested and works identically
- [ ] CPU fallback tested and works
- [ ] MPS (if available) tested and works
- [ ] XPU functionality tested and works
- [ ] All device detection uses `hasattr()` checks
- [ ] No hard assumptions about CUDA being the only GPU
- [ ] GradScaler disabled for XPU (Intel Arc FP64 limitation)
- [ ] Memory management handles all device types
- [ ] Backend configuration is device-specific, not device-replacing
- [ ] Autocast uses correct device_type
- [ ] Lightning trainer uses correct accelerator
- [ ] No new dependencies added to requirements.txt (use requirements-xpu.txt instead)
- [ ] Documentation updated
- [ ] No breaking changes to existing APIs

This guide should help agents understand the codebase structure and contribute effectively while maintaining backward compatibility.
