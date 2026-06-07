import sys, os
import re
import torch
from renorm.lm import RenormLM

def export_deployment_package():
    print("=== EXPORTING PRODUCTION RE-NORM DEPLOYMENT PACKAGE ===")
    
    # Target Corpus for vocabulary extraction
    raw_text = """
    Deep in the system architecture the gradient vector flows smoothly through the functional manifold 
    Without normalizations the parameters adjust their friction gates dynamically to maintain variance 
    The self stabilizing transformer converges efficiently proving that native scaling coordinates perfectly 
    """
    words = re.findall(r'\b\w+\b', raw_text.lower())
    vocab = sorted(list(set(words)))
    
    # 1. Instantiate deployment-ready model layout
    model = RenormLM(vocab_size=len(vocab), d_model=64, n_heads=4, depth=4, max_seq_len=8)
    
    # 2. Build deployment payload asset package
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
        'hyperparameters': {
            'vocab_size': len(vocab),
            'd_model': 64,
            'n_heads': 4,
            'depth': 4,
            'max_seq_len': 8
        }
    }
    
    # Write structural weights binary to disk
    output_filename = "renorm_production_model.pt"
    torch.save(checkpoint, output_filename)
    
    print(f"Deployment binary state successfully frozen: {output_filename}")
    print(f"Payload Size: {os.path.getsize(output_filename) / 1024:.2f} KB")
    print("=== EXPORT COMPLETE ===")

if __name__ == "__main__":
    export_deployment_package()