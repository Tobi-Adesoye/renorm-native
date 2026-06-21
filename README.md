
---

# renorm-native

[![PyPI version](https://img.shields.io/pypi/v/renorm-native.svg)](https://pypi.org/project/renorm-native/)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-supported-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 🚀 Overview

`renorm-native` is a PyTorch-compatible neural network module designed to improve numerical stability in deep learning models.

It provides transformer-ready layers that are robust to:

* Training instability (NaNs / exploding gradients)
* Irregular tensor shapes and sequence lengths
* Mixed CPU/GPU execution environments
* Memory pressure in large-scale workloads

It is designed to be a **drop-in architectural component** for modern deep learning pipelines.

---

## 📦 Installation

Install from PyPI:

```bash
pip install renorm-native
```

Upgrade to latest version:

```bash
pip install --upgrade renorm-native
```

---

## ⚡ Quick Start (30 seconds)

### Transformer Layer Example

```python
import torch
from renorm import RenormTransformerLayer

# Initialize layer
layer = RenormTransformerLayer(dim=512, heads=8)

# Dummy input: (batch, sequence, features)
x = torch.randn(2, 16, 512)

# Forward pass
y = layer(x)

print(y.shape)
```

### Expected Output

```text
torch.Size([2, 16, 512])
```

---

## 🧠 Core API

### 1. RenormTransformerLayer

A lightweight transformer block with built-in normalization stability.

```python
RenormTransformerLayer(
    dim: int,
    heads: int,
    eps: float = 1e-5
)
```

#### Parameters:

* `dim`: Hidden dimension size
* `heads`: Number of attention heads
* `eps`: Numerical stability constant

---

### 2. RenormLinear

A stable replacement for `torch.nn.Linear`.

```python
from renorm.layers import RenormLinear
```

Example:

```python
layer = RenormLinear(256, 128)
y = layer(torch.randn(4, 256))
```

---

## ⚙️ Device Compatibility

Automatically works across:

* CPU (Windows / Linux / Mac)
* CUDA (NVIDIA GPUs)
* Mixed environments (fallback-safe execution)

Example:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"

layer = RenormTransformerLayer(dim=512, heads=8).to(device)
x = torch.randn(2, 16, 512).to(device)

y = layer(x)
```

---

## 🧪 Minimal Validation Test

Run this to verify installation:

```bash
python -c "from renorm import RenormTransformerLayer; print(RenormTransformerLayer(dim=256, heads=4))"
```

Expected behavior: no errors and model prints successfully.

---

## 🏗 Architecture Summary

`renorm-native` uses a dual-path execution design:

* **CUDA Path (GPU):**

  * Optimized tensor execution path
  * High-performance kernel routing (where available)

* **CPU Path (Fallback):**

  * Stable numerical execution engine
  * Strict variance preservation for stability

This ensures consistent behavior across heterogeneous compute environments.

---

## 📊 Stability Design Principles

### 1. Variance Stabilization

Prevents numerical collapse in deep stacks by maintaining bounded activation scaling.

### 2. Memory Safety

Ensures gradient computation remains isolated from unsafe tensor views in dynamic graphs.

### 3. Execution Portability

Same model behavior across CPU and GPU environments.

---

## 📌 Example Use Case

* Transformer models (LLMs)
* Time-series forecasting systems
* Anomaly detection pipelines
* Edge-device inference systems
* Low-memory GPU environments

---

## ⚠️ Notes

* Requires PyTorch ≥ 2.0
* Python ≥ 3.10 recommended
* CUDA optional but supported

---

## 📄 License

MIT License — see `LICENSE` for details.

---

## 🤝 Contributing

Contributions, issues, and improvements are welcome.

---

## 🔗 Project

Maintained by the renorm-native team.

---

# 🧩 Enterprise / Production Add-On Section

---

## 🏢 Enterprise / Production Usage

`renorm-native` can be used in production systems requiring deterministic numerical stability under high load.

Typical deployment environments:

* GPU inference clusters (CUDA-enabled)
* On-prem ML pipelines
* Edge inference systems
* Distributed training environments (PyTorch DDP)

---

## 🔐 Enterprise License Mode (Optional)

Some builds may enable enterprise validation for regulated or production deployments.

### Environment Variable

```bash
export RENORM_ENTERPRISE_KEY="your_token_here"
```

### Format

```
base64_payload.hex_hmac_signature
```

### Programmatic Validation

```python
from renorm.auth import check_enterprise_license

check_enterprise_license()
```

### Failure Modes

| Condition         | Behavior               |
| ----------------- | ---------------------- |
| Missing key       | Raises PermissionError |
| Invalid signature | Raises PermissionError |
| Expired token     | Raises TimeoutError    |

---

## ⚙️ Production Integration Pattern

Recommended structure in production pipelines:

```python
import torch
from renorm import RenormTransformerLayer

def build_model():
    model = RenormTransformerLayer(dim=1024, heads=16)
    return model

def forward_pass(model, x):
    return model(x)
```

---

## 🧪 CI / Validation Test

Run a deterministic sanity check:

```bash
python -c "
import torch
from renorm import RenormTransformerLayer

layer = RenormTransformerLayer(dim=256, heads=4)
x = torch.randn(2, 8, 256)
y = layer(x)

assert y.shape[-1] == 256
print('OK')
"
```

---

## 📊 Performance Notes

`renorm-native` is optimized for:

* Stable forward/backward propagation under long sequence lengths
* Reduced numerical drift in deep stacks
* Consistent execution across heterogeneous compute backends

It is not intended as a raw speed-optimized kernel replacement for PyTorch primitives.

---

## 🔄 Compatibility Matrix

| Environment                | Status          |
| -------------------------- | --------------- |
| CPU (Windows)              | ✅ Supported     |
| CPU (Linux)                | ✅ Supported     |
| CUDA 11+                   | ✅ Supported     |
| MPS (Apple Silicon)        | ⚠️ Experimental |
| Distributed training (DDP) | ✅ Compatible    |

---

## 🧠 Design Philosophy

`renorm-native` prioritizes:

* Numerical correctness over raw speed
* Stability over aggressive optimization
* Cross-device consistency over hardware specialization

It is designed to behave predictably under:

* gradient explosion conditions
* low precision arithmetic
* fragmented tensor memory layouts

---

## 📦 Recommended Deployment (Docker)

```dockerfile
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

WORKDIR /app

RUN pip install renorm-native

COPY . .

CMD ["python", "main.py"]
```

---

## 📈 Benchmark (Example Placeholder)

| Layer              | Stability Score | NaN Rate            |
| ------------------ | --------------- | ------------------- |
| torch.nn.LayerNorm | baseline        | medium under stress |
| renorm-native      | improved        | near-zero           |

*(Replace with your real measured results when ready — do NOT leave as-is in final production release if publishing publicly.)*

---

## 🌐 Roadmap

Planned improvements:

* Distributed kernel optimization (multi-GPU aware routing)
* Expanded attention primitives
* Quantization-aware renormalization mode
* Torch compile integration (torch.compile support)

---

## 📩 Support

For production integration or enterprise deployment:

* GitHub Issues: https://github.com/Tobi-Adesoye/renorm-native
* Contact: Adesoyetobe@gmail.com

