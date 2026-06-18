import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import torch

from renorm.lm import RenormLM


def verify_autoregressive_generation():
    print("=== INITIALIZING CAUSAL PRODUCTION LANGUAGE MODEL ===")

    VOCAB_SIZE = 1000
    EMBED_DIM = 128
    HEADS = 4
    DEPTH = 6

    # Instantiate the complete language model
    model = RenormLM(vocab_size=VOCAB_SIZE, d_model=EMBED_DIM, n_heads=HEADS, depth=DEPTH)
    model.eval()  # Switch to evaluation mode

    # Create artificial token index history (e.g., Batch=1, Sequence Length=5)
    context_tokens = torch.tensor([[12, 45, 882, 14, 99]], dtype=torch.long)

    print(f"Input Seed Prompt Tokens: {context_tokens.tolist()[0]}")

    # Autoregressive sampling generation loop (Generate 5 new tokens sequentially)
    with torch.no_grad():
        for step in range(5):
            logits = model(context_tokens)

            # Pull only the logits corresponding to the absolute latest step vector
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Append generated token to context history dynamically
            context_tokens = torch.cat((context_tokens, next_token), dim=1)

            print(
                f"Generation Step {step + 1} | Predicted Token: {next_token.item()} | Active Sequence Context: {context_tokens.tolist()[0]}"
            )

    print("-" * 85)
    print(f"Final Model Tensor History Context Shape: {list(context_tokens.shape)}")
    print("=== AUTOREGRESSIVE GENERATION ENGINE ONLINE & FUNCTIONAL ===")


if __name__ == "__main__":
    verify_autoregressive_generation()
