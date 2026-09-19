PYTHON ?= python
RUN := PYTHONPATH=. $(PYTHON)

.PHONY: test smoke data benchmark public-data combined-data train-head train-head-public train-lora-public eval serve nanojev package

test:
	$(RUN) -m pytest

data:
	$(RUN) scripts/generate_simulator.py --output data/simulator --seed 17

benchmark:
	$(RUN) scripts/build_smoke_benchmark.py --output data/benchmark/semantic_smoke.jsonl
	$(RUN) scripts/freeze_benchmark.py data/benchmark/semantic_smoke.jsonl

public-data:
	$(RUN) scripts/build_public_data.py --output data/public_decisions --benchmark-output data/benchmark/open_decision_bench.jsonl
	$(RUN) scripts/freeze_benchmark.py data/benchmark/open_decision_bench.jsonl

combined-data: data public-data
	$(RUN) scripts/merge_data_dirs.py --inputs data/simulator data/public_decisions --output data/combined
	$(RUN) scripts/validate_dataset.py --data-dir data/combined

smoke: data benchmark
	$(RUN) scripts/run_cpu_smoke.py --data-dir data/simulator --output results/cpu_smoke.json

train-head:
	$(RUN) scripts/train.py --config configs/train_head.yaml

train-head-public: combined-data
	$(RUN) scripts/train.py --config configs/train_head_public_fast.yaml

train-lora-public: combined-data
	$(RUN) scripts/train_lora.py --config configs/train_lora_public_fast.yaml

eval:
	$(RUN) scripts/evaluate.py --checkpoint checkpoints/head-v0 --data data/simulator/test.jsonl

nanojev:
	$(RUN) scripts/benchmark_nanojev.py --data data/benchmark/open_decision_bench.jsonl --nanojev-repo vendor/NanoJev --checkpoint-dir checkpoints/nanojev --output results/nanojev_open_decision.jsonl

serve:
	$(RUN) scripts/serve.py --checkpoint checkpoints/head-v0 --port 8000

package:
	cd .. && zip -qr open-system-one.zip open-system-one -x 'open-system-one/.venv/*' 'open-system-one/checkpoints/*'
