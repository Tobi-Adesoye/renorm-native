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