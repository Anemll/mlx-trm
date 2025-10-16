#!/bin/bash
# Test script to verify halt_max_steps fix

echo "Testing halt_max_steps fix with 2 epochs..."
echo "Expected: steps should increase from 1.0 to ~2-3 with exploration"
echo ""

python3 train_arc_multi_opt.py \
  -b 16 \
  -e 2 \
  --dim 128 \
  --halt-max-steps 4 \
  --halt-exploration 0.3 \
  --bf16 \
  --save test_halt_fix \
  --val-freq 2

echo ""
echo "Checking results..."
python3 -c "
import json
with open('test_halt_fix/training_history.json', 'r') as f:
    history = json.load(f)
steps = history.get('train_avg_steps_trace', [])
print(f'Steps progression: {[round(s, 2) for s in steps]}')
if max(steps) > 1.1:
    print('✅ SUCCESS: Model is using multiple steps!')
else:
    print('❌ FAIL: Model still stuck at 1 step')
"
