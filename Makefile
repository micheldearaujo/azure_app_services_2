install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

install-gcp:
	pip install --upgrade pip &&\
		pip install -r requirements-gcp.txt

install-azure:
	pip install --upgrade pip &&\
		pip install -r requirements-azure.txt
		
install-aws:
	pip install --upgrade pip &&\
		pip install -r requirements-aws.txt

test:
	python -m pytest -vv --cov=src/utils src/test_utils.py

lint:
	pylint --disable=R,C *.py
	
format:
	black *.py

clean:
	rm -rf __pycache__
	rm -f *.log
	rm -f *.log
