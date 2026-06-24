PY  ?= ~/miniconda3/envs/personal/bin/python
PIP ?= ~/miniconda3/envs/personal/bin/pip

.PHONY: install seed summary serve dashboard test

install:
	$(PIP) install -e ".[all]"

seed:
	$(PY) -m llmobs.cli seed --reset --n 40

summary:
	$(PY) -m llmobs.cli summary

serve:
	$(PY) -m uvicorn api.main:app --reload --port 8000

dashboard:
	$(PY) -m streamlit run app/dashboard.py

test:
	$(PY) -m pytest -q
