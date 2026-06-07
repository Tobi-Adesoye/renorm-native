import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import torch
from renorm.attention import RenormAttention

def verify_attention_flow():
    print("=== INITIALIZING TRANSFORMATION TEST: RENORM-ATTENTION ===")
    
    BATCH_SIZE = 2
    SEQ_LEN = 16
    EMBED_DIM = 128
    HEADS = 4
    
    # Instantiate our new module
    attn_layer = RenormAttention(d_model=EMBED_DIM, n_heads=HEADS)
    
    # Create simulated token embedding matrix
    x = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM, requires_grad=True)
    
    # Process through the functional manifold
    output = attn_layer(x)
    
    # Target loss calculation
    loss = output.mean()
    loss.backward()
    
    print(f"Input Shape                 : {list(x.shape)}")
    print(f"Output Shape                : {list(output.shape)}")
    print(f"Functional Attention Beta   : {torch.sigmoid(attn_layer.beta_proxy).item():.6f}")
    print(f"Beta Gradient Extracted     : {attn_layer.beta_proxy.grad.item():.6f}")
    print("==========================================================")

if __name__ == "__main__":
    verify_attention_flow()