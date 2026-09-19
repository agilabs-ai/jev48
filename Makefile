PYTHON ?= python
RUN := PYTHONPATH=. $(PYTHON)

.PHONY: test package modal modal-jev

test:
	$(RUN) -m pytest

modal:
	modal run modal_app.py

modal-jev:
	@test -n "$(RUN_NAME)" || (echo "RUN_NAME is required" && exit 1)
	modal run modal_jev.py --run-name $(RUN_NAME)

package:
	cd .. && zip -qr JEV48_HANDOFF.zip jev48 -x 'jev48/.venv/*' 'jev48/checkpoints/*' 'jev48/.pytest_cache/*' '**/__pycache__/*'
