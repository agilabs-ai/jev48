PYTHON ?= python
RUN := PYTHONPATH=. $(PYTHON)

.PHONY: test smoke data benchmark train-head eval serve package

test:
	$(RUN) -m pytest

data:
	$(RUN) scripts/generate_simulator.py --output data/simulator --seed 17

benchmark:
	$(RUN) scripts/build_smoke_benchmark.py --output data/benchmark/semantic_smoke.jsonl
	$(RUN) scripts/freeze_benchmark.py data/benchmark/semantic_smoke.jsonl

smoke: data benchmark
	$(RUN) scripts/run_cpu_smoke.py --data-dir data/simulator --output results/cpu_smoke.json

train-head:
	$(RUN) scripts/train.py --config configs/train_head.yaml

eval:
	$(RUN) scripts/evaluate.py --checkpoint checkpoints/head-v0 --data data/simulator/test.jsonl

serve:
	$(RUN) scripts/serve.py --checkpoint checkpoints/head-v0 --port 8000

package:
	cd .. && zip -qr open-system-one.zip open-system-one -x 'open-system-one/.venv/*' 'open-system-one/checkpoints/*'
