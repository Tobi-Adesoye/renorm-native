import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import torch
import torch.nn as nn
from renorm.transformer import RenormTransformerLayer

def verify_deep_transformer_stack():
    print("=== INITIALIZING DEEP STACK VERIFICATION: RENORM-TRANSFORMER ===")
    
    NUM_BLOCKS = 12
    BATCH_SIZE = 2
    SEQ_LEN = 32
    EMBED_DIM = 256
    HEADS = 8
    
    # Instantiate a deep 12-block encoder stack sequential module
    transformer_stack = nn.Sequential(*[
        RenormTransformerLayer(d_model=EMBED_DIM, n_heads=HEADS) 
        for _ in range(NUM_BLOCKS)
    ])
    
    # Generate random sequence embeddings
    x = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM, requires_grad=True)
    
    # Run the forward block progression
    output = transformer_stack(x)
    
    # Compute backward step to verify full graph compilation
    loss = output.mean()
    loss.backward()
    
    # Extract telemetry metrics from the first and last blocks
    first_block = transformer_stack[0]
    last_block = transformer_stack[-1]
    
    print(f"Stack Depth Evaluated       : {NUM_BLOCKS} Blocks")
    print(f"Input Processing Shape      : {list(x.shape)}")
    print(f"Output Processing Shape     : {list(output.shape)}")
    print("-" * 65)
    print(f"Block 0  - Functional Attention Beta : {torch.sigmoid(first_block.attn.beta_proxy).item():.6f}")
    print(f"Block 0  - Functional MLP Beta       : {torch.sigmoid(first_block.beta_mlp_proxy).item():.6f}")
    print(f"Block 11 - Functional Attention Beta : {torch.sigmoid(last_block.attn.beta_proxy).item():.6f}")
    print(f"Block 11 - Functional MLP Beta       : {torch.sigmoid(last_block.beta_mlp_proxy).item():.6f}")
    print("-" * 65)
    print(f"Source Gradient Status Norm : {x.grad.norm().item():.6f}")
    print("=== STACK GRAPH COMPILED AND VERIFIED SUCCESSFULLY ===")

if __name__ == "__main__":
    verify_deep_transformer_stack()