.PHONY: setup data train eval serve clean help

# Prefer the project venv if present so an active conda env can't shadow it.
PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python)
CONFIG ?= configs/default.yaml

help:
	@echo "Targets:"
	@echo "  setup  - install package + dependencies (editable)"
	@echo "  data   - download, split, and curate the dataset (CONFIG=$(CONFIG))"
	@echo "  train  - LoRA fine-tune Qwen2.5-0.5B (CONFIG=$(CONFIG))"
	@echo "  eval   - evaluate runs/LATEST on the test split (CONFIG=$(CONFIG))"
	@echo "  serve  - start FastAPI server on :8000 (CONFIG=$(CONFIG))"
	@echo "  clean  - remove caches and build artifacts (keeps data/ and runs/)"

setup:
	$(PYTHON) -m pip install -e .

data:
	$(PYTHON) -m titan.data.load --config $(CONFIG)

train:
	$(PYTHON) -m titan.train --config $(CONFIG)

eval:
	$(PYTHON) -m titan.eval.run --config $(CONFIG) \
	  $(if $(TEST_FROM),--test-from $(TEST_FROM)) \
	  $(if $(RUN_ID),--run-id $(RUN_ID))

serve:
	TITAN_CONFIG=$(CONFIG) $(PYTHON) -m uvicorn titan.serve:app --host 0.0.0.0 --port 8000

clean:
	rm -rf build dist *.egg-info .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
