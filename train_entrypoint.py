import sys, os
import re
import math
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from renorm.lm import RenormLM

def main():
    print("=== STARTING SCHEDULED COSINE ANNEALING TRAINING PIPELINE ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Expanded Text Corpus
    raw_text = """
    Deep in the system architecture the gradient vector flows smoothly through the functional manifold 
    Without normalizations the parameters adjust their friction gates dynamically to maintain variance 
    The self stabilizing transformer converges efficiently proving that native scaling coordinates perfectly 
    """
    
    words = re.findall(r'\b\w+\b', raw_text.lower())
    vocab = sorted(list(set(words)))
    vocab_size = len(vocab)
    
    wtoi = {word: i for i, word in enumerate(vocab)}
    itow = {i: word for i, word in enumerate(vocab)}
    
    encode = lambda s: [wtoi[w] for w in re.findall(r'\b\w+\b', s.lower()) if w in wtoi]
    decode = lambda l: ' '.join([itow[i] for i in l])
    
    tokenized_data = torch.tensor([wtoi[w] for w in words], dtype=torch.long)
    
    # 2. Architecture Dimensions
    EMBED_DIM = 64
    HEADS = 4
    DEPTH = 4
    BLOCK_SIZE = 8
    TOTAL_STEPS = 250  # Extended training path to leverage annealing mechanics
    
    model = RenormLM(
        vocab_size=vocab_size, 
        d_model=EMBED_DIM, 
        n_heads=HEADS, 
        depth=DEPTH, 
        max_seq_len=BLOCK_SIZE
    ).to(device)
    
    # 3. Optimization + Cosine Scheduler Hook Setup
    optimizer = optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    
    # Decay learning rate down to 10% of its initial value using a cosine curve
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_STEPS, eta_min=3e-4)
    criterion = nn.CrossEntropyLoss()
    
    def get_batch(batch_size=8):
        ix = torch.randint(len(tokenized_data) - BLOCK_SIZE, (batch_size,))
        x = torch.stack([tokenized_data[i:i+BLOCK_SIZE] for i in ix])
        y = torch.stack([tokenized_data[i+1:i+BLOCK_SIZE+1] for i in ix])
        return x.to(device), y.to(device)

    print("--------------------------------------------------------------------------------")
    print(f"{'Step':<8} | {'Cross-Entropy Loss':<20} | {'Active LR':<12} | {'Beta Scale':<12}")
    print("--------------------------------------------------------------------------------")
    
    # 4. Scheduled Optimization Loop
    for step in range(1, TOTAL_STEPS + 1):
        model.train()
        xb, yb = get_batch()
        
        optimizer.zero_grad()
        logits = model(xb)
        B, T, C = logits.shape
        loss = criterion(logits.view(B*T, C), yb.view(B*T))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Step the learning rate decay scheduler forwards
        scheduler.step()
        
        # Log telemetry metrics every 50 steps
        if step == 1 or step % 50 == 0:
            model.eval()
            with torch.no_grad():
                current_lr = optimizer.param_groups[0]['lr']
                current_beta = torch.sigmoid(model.blocks[0].attn.beta_proxy).item()
                print(f"{step:<8} | {loss.item():<20.6f} | {current_lr:<12.6f} | {current_beta:<12.6f}")
                
                # Sample text generation from the seed prompt word
                seed_word = "deep"
                context = torch.tensor([encode(seed_word)], dtype=torch.long, device=device)
                
                for _ in range(12):
                    context_cond = context[:, -BLOCK_SIZE:]
                    out_logits = model(context_cond)
                    next_word_logits = out_logits[:, -1, :] / 0.7
                    probs = torch.softmax(next_word_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    context = torch.cat((context, next_token), dim=1)
                
                generated_sentence = decode(context.squeeze().tolist())
                print(f"  >>> [Step {step} Phrasing]: \"{generated_sentence}\"")
                print("-" * 80)

    print("=== PIPELINE OPTIMIZATION AND ANNEALING COMPLETED ===")

if __name__ == "__main__":
    main()