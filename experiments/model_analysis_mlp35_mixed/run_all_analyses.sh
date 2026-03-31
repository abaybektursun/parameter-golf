#!/bin/bash
# Run all 8 analyses and save results to /root/parameter-golf/analysis/results/
set -e
cd /root/parameter-golf/analysis

RESULTS=results
mkdir -p $RESULTS

echo "=== Starting analysis suite ==="
echo "Model: MLP 3.5x, BigramHash 3072x112, 29.95M params"
echo "Results dir: $(pwd)/$RESULTS"
echo ""

# 1. SVD (CPU only — runs fast)
echo "[1/8] SVD analysis..."
python3 analysis_svd.py 2>&1 | tee $RESULTS/svd.txt
echo ""

# 2. Logit lens (1 GPU)
echo "[2/8] Logit lens..."
CUDA_VISIBLE_DEVICES=0 python3 analysis_logit_lens.py 2>&1 | tee $RESULTS/logit_lens.txt
echo ""

# 3. Loss decomposition (1 GPU)
echo "[3/8] Loss decomposition..."
CUDA_VISIBLE_DEVICES=0 python3 analysis_loss_decomposition.py 2>&1 | tee $RESULTS/loss_decomposition.txt
echo ""

# 4. Quant sensitivity — uniform int6 (1 GPU, ~5 min)
echo "[4/8] Quant sensitivity (int6)..."
CUDA_VISIBLE_DEVICES=1 python3 analysis_quant_sensitivity.py 2>&1 | tee $RESULTS/quant_sensitivity.txt
echo ""

# 5. Attention heads (1 GPU, ~2 min, memory-intensive)
echo "[5/8] Attention head classification..."
CUDA_VISIBLE_DEVICES=2 python3 analysis_attention_heads.py 2>&1 | tee $RESULTS/attention_heads.txt
echo ""

# 6. Confidence calibration (1 GPU)
echo "[6/8] Confidence calibration..."
CUDA_VISIBLE_DEVICES=0 python3 analysis_confidence_calibration.py 2>&1 | tee $RESULTS/confidence_calibration.txt
echo ""

# 7. Token interpretability (1 GPU)
echo "[7/8] Token interpretability..."
CUDA_VISIBLE_DEVICES=1 python3 analysis_token_interpretability.py 2>&1 | tee $RESULTS/token_interpretability.txt
echo ""

# 8. Mixed-precision quant (1 GPU, ~10 min — 132 evals)
echo "[8/8] Mixed-precision quant allocation..."
CUDA_VISIBLE_DEVICES=2 python3 analysis_mixed_quant.py 2>&1 | tee $RESULTS/mixed_quant.txt
echo ""

echo "=== All analyses complete ==="
echo "Results saved to: $(pwd)/$RESULTS/"
ls -la $RESULTS/
