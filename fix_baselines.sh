#!/bin/bash
# Run on server to fix both baseline scripts
# cd ~/sharon/mineral_mapping && bash fix_baselines.sh

# =============================================================================
# FIX 1: Random Forest — predict_proba returns 1 column when single class
# =============================================================================
python3 - << 'EOF'
path = "src/baselines/random_forest.py"
with open(path) as f:
    code = f.read()

# Fix predict_proba to handle single-class case
old = "    y_score = rf.predict_proba(X_val)[:, 1]"
new = """    proba = rf.predict_proba(X_val)
    if proba.shape[1] == 1:
        # Only one class in training — RF predicts constant
        print("  WARNING: RF trained on single class, using constant scores")
        y_score = proba[:, 0]
    else:
        y_score = proba[:, 1]"""
code = code.replace(old, new)

# Also fix the map prediction part
old2 = "    y_pred_map = rf.predict_proba(X_pred)[:, 1]"
new2 = """    proba_map = rf.predict_proba(X_pred)
    y_pred_map = proba_map[:, 1] if proba_map.shape[1] > 1 else proba_map[:, 0]"""
code = code.replace(old2, new2)

with open(path, "w") as f:
    f.write(code)
print("RF fixed")
EOF

# =============================================================================
# FIX 2: WoE — fix empty woe_maps stack + lower studentized threshold
# =============================================================================
python3 - << 'EOF'
path = "src/baselines/weights_of_evidence.py"
with open(path) as f:
    code = f.read()

# Lower the threshold so more channels qualify
code = code.replace(
    '"studentized_threshold": 1.5,',
    '"studentized_threshold": 0.5,'
)

# Fix the stack when woe_maps is still empty
old = """    # Sum of WoE weights = posterior logit
    woe_stack   = np.stack(woe_maps, axis=0)"""
new = """    # If still no maps, use all channels regardless
    if not woe_maps:
        print("  WARNING: Still no maps, using all channels without threshold")
        for ch in range(data.shape[0]):
            binary_map, Wp, Wm, C, sC = compute_woe(
                data[ch], labels, mask=train_mask
            )
            if binary_map is not None:
                woe_ch = np.where(
                    np.isnan(binary_map), np.nan,
                    np.where(binary_map == 1, Wp, Wm)
                )
                woe_maps.append(woe_ch)

    if not woe_maps:
        print("ERROR: Could not compute any WoE maps")
        return {"auc_pr":0.0,"auc_roc":0.0,"f1":0.0,"mcc":0.0,"fold":fold}

    # Sum of WoE weights = posterior logit
    woe_stack   = np.stack(woe_maps, axis=0)"""
code = code.replace(old, new)

with open(path, "w") as f:
    f.write(code)
print("WoE fixed")
EOF

echo ""
echo "Both scripts fixed. Relaunching..."

# Kill old processes
pkill -f "random_forest.py" 2>/dev/null
pkill -f "weights_of_evidence.py" 2>/dev/null
sleep 2

mkdir -p results/baselines/random_forest
mkdir -p results/baselines/woe

# Relaunch
nohup python src/baselines/weights_of_evidence.py \
  > results/woe.log 2>&1 &
echo "WoE PID: $!"

nohup python src/baselines/random_forest.py \
  > results/rf.log 2>&1 &
echo "RF PID: $!"

sleep 15
echo ""
echo "=== WoE log ==="
tail -5 results/woe.log

echo ""
echo "=== RF log ==="
tail -5 results/rf.log
