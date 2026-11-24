#!/usr/bin/env bash
set -euo pipefail

nvidia-smi
ollaya --version
python --version
/opt/kev/.venv/bin/python --version

python - <<'PY'
import numpy as np
import xgboost as xgb

features = np.asarray([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
labels = np.asarray([0, 1, 1, 0])
model = xgb.XGBClassifier(n_estimators=2, max_depth=2, tree_method="hist", device="cuda")
model.fit(features, labels)
print("XGBoost CUDA: ok")
print(xgb.build_info())
PY

/opt/kev/.venv/bin/python -c "import kev, torch; print('Kev: ok'); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
