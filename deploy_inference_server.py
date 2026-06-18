import os
import re

import torch

from renorm.lm import RenormLM


def encode_text(s, wtoi):
    return [wtoi[w] for w in re.findall(r"\b\w+\b", s.lower()) if w in wtoi]


def decode_tokens(token_list, itow):
    return " ".join([itow[i] for i in token_list])


def launch_inference_engine():
    print("=== LOADING RE-NORM RUNTIME INFERENCE SERVER ===")
    checkpoint_path = "renorm_production_model.pt"

    if not os.path.exists(checkpoint_path):
        print(
            f"Error: Deployment asset '{checkpoint_path}' not found! Run save_production_checkpoint.py first."
        )
        return

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    hparams = checkpoint["hyperparameters"]
    vocab = checkpoint["vocab"]

    wtoi = {word: i for i, word in enumerate(vocab)}
    itow = {i: word for i, word in enumerate(vocab)}

    model = RenormLM(
        vocab_size=hparams["vocab_size"],
        d_model=hparams["d_model"],
        n_heads=hparams["n_heads"],
        depth=hparams["depth"],
        max_seq_len=hparams["max_seq_len"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(
        f"Model successfully loaded. Active Parameters: {sum(p.numel() for p in model.parameters()):,}"
    )
    print("Inference Engine Status: ONLINE (Type 'exit' to terminate deployment server)")
    print("-" * 75)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    while True:
        try:
            user_prompt = input("Inference Request Seed Word -> ").strip()

            if user_prompt.lower() == "exit":
                break

            tokens = encode_text(user_prompt, wtoi)

            if not tokens:
                print(
                    "Deployment Warning: Input seed word missing from vocabulary. Try words like 'deep', 'transformer', or 'gradient'."
                )
                continue

            context = torch.tensor([tokens], dtype=torch.long, device=device)

            with torch.no_grad():
                for _ in range(10):
                    context_cond = context[:, -hparams["max_seq_len"] :]
                    logits = model(context_cond)

                    probs = torch.softmax(logits[:, -1, :] / 0.7, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

                    context = torch.cat((context, next_token), dim=1)

            print(f'Engine Response: "{decode_tokens(context.squeeze().tolist(), itow)}"')
            print("-" * 75)

        except KeyboardInterrupt:
            break

    print("\n=== INFERENCE SERVER SHUTDOWN CLEANLY ===")


if __name__ == "__main__":
    launch_inference_engine()
