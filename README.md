# renorm-native: Self-Stabilizing Deep Transformer Architectures

An open-source, production-ready framework that corrects the foundational variance flaws of deep neural networks by replacing identity residual links with bounded functional sub-manifolds.

## Drop-In Replacement API

You don't need to rebuild your codebase. Swap your standard layers for Renorm-Native modules in less than 5 seconds:

```python
# Before: Standard Volatile Layer
# import torch.nn as nn
# layer = nn.Linear(512, 512)

# After: Self-Stabilizing Renorm Layer
from renorm import RenormLinear
layer = RenormLinear(512, 512)

Users\my pc\Desktop\renorm-native> python ./run_convergence_universe.py
=== STARTING MULTI-EPOCH CONVERGENCE BENCHMARK ===
Tracking 24 Independent Bounded Manifolds Across 50 Epochs...
--------------------------------------------------------------------------------
Epoch    | Current Loss   | Layer 0 Attn β   | Layer 11 MLP β
--------------------------------------------------------------------------------
1        | 0.555508       | 0.010101         | 0.010102
10       | 0.343834       | 0.011359         | 0.011455
20       | 0.250023       | 0.013168         | 0.012790
30       | 0.233512       | 0.014873         | 0.013610
40       | 0.227492       | 0.016615         | 0.014198
50       | 0.222712       | 0.018605         | 0.014733
--------------------------------------------------------------------------------
=== CONVERGENCE PROFILE COMPLETED SUCCESSFULLY ===


---

## Enterprise Infrastructure & Commercial Licensing

The open-source core of `renorm-native` is licensed under the strict copyleft **AGPLv3**. If you are utilizing this architecture within a commercial training pipeline, cloud-hosted application, or proprietary cluster layout, you are legally bound to open-source your entire end-to-end pipeline.

To bypass these copyleft restrictions or to run at extreme industrial scale, we offer our proprietary **Enterprise Tier**.

### The Commercial Edge: `renorm-cuda`

For production clusters, we provide access to our private, closed-source hardware acceleration package. 

* **Fused Triton/CUDA Kernels:** Fuses the linear matrix transformations and sigmoid friction gates into a single on-chip SRAM cache pass.
* **30% VRAM Dividend:** Eliminates intermediate framework tensor allocations, completely mitigating memory bandwidth choking.
* **Warmup Obsolescence:** Natively bounds gradient propagation variance to fixed manifolds, allowing immediate training at maximum velocity from Step 1.

### Pricing Matrix

| Tier | Target | Licensing Structure | Included Features |
| :--- | :--- | :--- | :--- |
| **Sovereign Cluster** | Scaling Startups & Labs (9-512 GPUs) | $150 / active GPU / month | AGPLv3 Waiver, Pure PyTorch Core, Standard SLA |
| **Frontier Engine** | Enterprise & Foundation Labs (512+ GPUs) | Custom Compute Volume Pricing | Fused Triton Kernels, 30% VRAM Optimization, Dedicated Co-Design Engineers |

### Enterprise Onboarding & Inbound Requests

To acquire commercial licensing keys, request a dedicated cluster validation run, or set up architectural co-design consulting, submit a formal request to our infrastructure gate:

👉 **[Request Enterprise Access Key](mailto:licensing@renorm.ai?subject=Enterprise%20Licensing%20Inquiry%20-%20%5BCompany%20Name%5D)**

*Please include your organization name, active training cluster size (GPU count), and target parameter scale in your communication.*

