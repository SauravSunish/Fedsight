PYTHON ?= python

.PHONY: run test demo eval-chat
run:
	$(PYTHON) -m streamlit run app.py --server.address 127.0.0.1
test:
	$(PYTHON) -m pytest -q
demo:
	$(PYTHON) -m fedsight demo
eval-chat:
	$(PYTHON) scripts/eval_investigator.py
