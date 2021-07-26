PYTHON = .venv/bin/python

clean: clean-build clean-tmp clean-pyc

clean-build:
	rm -rf glove/build
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-tmp:
	rm -rf glove/.tmp

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

compile-glove:
	cd glove ; make

venv:
	. .venv/bin/activate

dist: clean venv
	$(PYTHON) setup.py sdist
	$(PYTHON) setup.py bdist_wheel
	ls -l dist

install: clean venv
	$(PYTHON) setup.py install