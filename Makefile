PYTHON ?= python3

.PHONY: install test check

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

check:
	$(PYTHON) scripts/check_setup.py

test:
	$(PYTHON) -m compileall .
