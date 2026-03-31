"""Logit lens: project residual stream through unembedding at each layer."""
import math
import numpy as np
import torch
import torch.nn.functional as F

from common import make_args, load_model, load_validation_tokens

DEVICE = "cuda:0"
SEQ_LEN = 2048
NUM_EVAL_SEQS = 64


def logit_lens_forward(model, input_ids, targets):
    """Run forward pass layer-by-layer, probing predictions at each point.
    Returns (losses, accuracies) — one entry per probe point."""
    n = model.num_layers
    unembed_w = model.tok_emb.weight  # tied embeddings
    softcap = model.logit_softcap
    flat_targets = targets.reshape(-1)

    def probe(h):
        logits = softcap * torch.tanh(F.linear(F.rms_norm(h, (h.size(-1),)), unembed_w) / softcap)
        flat = logits.reshape(-1, logits.size(-1)).float()
        loss = F.cross_entropy(flat, flat_targets, reduction="mean").item()
        acc = (flat.argmax(-1) == flat_targets).float().mean().item()
        return loss, acc

    # Embedding
    x = model.tok_emb(input_ids)
    if model.bigram is not None:
        x = x + model.bigram(input_ids)
    x = F.rms_norm(x, (x.size(-1),))
    x = model.smear(x)
    x0 = x

    losses, accs = [], []
    losses_acc, accs_acc = probe(x)
    losses.append(losses_acc)
    accs.append(accs_acc)

    # Encoder
    v0, skips, ve_cache = None, [], {}
    for i in range(model.num_encoder_layers):
        ve = model._get_ve(i, input_ids, ve_cache)
        x, raw_v = model.blocks[i](x, x0,
            model.qo_bank[i], model.kv_bank[i], model.kv_bank[n + i],
            model.qo_bank[n + i], model.mlp_up_bank[i], model.mlp_down_bank[i],
            v_embed=ve, v0=v0)
        if v0 is None and raw_v is not None:
            v0 = raw_v
        skips.append(x)
        l, a = probe(x)
        losses.append(l)
        accs.append(a)

    # Decoder
    for i in range(model.num_decoder_layers):
        bi = model.num_encoder_layers + i
        if skips:
            x = x + model.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
        ve = model._get_ve(bi, input_ids, ve_cache)
        x, _ = model.blocks[bi](x, x0,
            model.qo_bank[bi], model.kv_bank[bi], model.kv_bank[n + bi],
            model.qo_bank[n + bi], model.mlp_up_bank[bi], model.mlp_down_bank[bi],
            v_embed=ve, v0=v0)
        l, a = probe(x)
        losses.append(l)
        accs.append(a)

    # Final (after final_norm — should match model output)
    x_final = model.final_norm(x)
    logits = softcap * torch.tanh(F.linear(x_final, unembed_w) / softcap)
    flat = logits.reshape(-1, logits.size(-1)).float()
    losses.append(F.cross_entropy(flat, flat_targets, reduction="mean").item())
    accs.append((flat.argmax(-1) == flat_targets).float().mean().item())

    return losses, accs


def main():
    args = make_args()
    device = torch.device(DEVICE)
    model = load_model(args, device)
    val_tokens = load_validation_tokens(args.val_files, SEQ_LEN)

    num_seqs = min(NUM_EVAL_SEQS, (val_tokens.numel() - 1) // SEQ_LEN)
    labels = ["embed"] + [f"layer_{i}" for i in range(args.num_layers)] + ["final"]
    all_losses = [[] for _ in labels]
    all_accs = [[] for _ in labels]

    print(f"Logit lens on {num_seqs} sequences of length {SEQ_LEN}")

    with torch.inference_mode():
        for seq_idx in range(num_seqs):
            start = seq_idx * SEQ_LEN
            input_ids = val_tokens[start : start + SEQ_LEN].unsqueeze(0).to(device).long()
            targets = val_tokens[start + 1 : start + SEQ_LEN + 1].unsqueeze(0).to(device).long()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                losses, accs = logit_lens_forward(model, input_ids, targets)

            for i, (l, a) in enumerate(zip(losses, accs)):
                all_losses[i].append(l)
                all_accs[i].append(a)

            if (seq_idx + 1) % 16 == 0:
                print(f"  {seq_idx + 1}/{num_seqs} done")

    LOG2 = math.log(2)
    print("\n" + "=" * 70)
    print("LOGIT LENS REPORT")
    print("=" * 70)
    print(f"\n{'Layer':<12} {'Loss (nats)':>12} {'Bits/tok':>10} {'Top-1 Acc':>10} {'Delta':>10}")
    print("-" * 56)
    prev = None
    for i, label in enumerate(labels):
        mean_loss = np.mean(all_losses[i])
        mean_acc = np.mean(all_accs[i])
        bpt = mean_loss / LOG2
        delta = f"{bpt - prev:+.4f}" if prev is not None else ""
        print(f"{label:<12} {mean_loss:>12.4f} {bpt:>10.4f} {mean_acc:>9.1%} {delta:>10}")
        prev = bpt

    print("\n" + "=" * 70)
    print("NOTE: 'Bits/tok' is bits per TOKEN (not per byte). To convert to BPB,")
    print("divide by avg bytes/token (~2.7 for sp1024). Final model BPB ~ 1.13.")
    print("")
    print("INTERPRETATION:")
    print("- Large drop at a layer = that layer contributes significant signal")
    print("- Small drop = layer may be underutilized (candidate for narrowing)")
    print("- Increase at encoder layers 3-4 = these layers optimize features for")
    print("  skip connections to decoder, not for direct prediction via unembedding")
    print("- Increase at decoder layer 5+ = skip connection injection disrupts")
    print("  residual stream alignment with output space")
    print("=" * 70)


if __name__ == "__main__":
    main()
