.PHONY: install run quick baseline compare report list test clean

install:
	pip install -r requirements.txt
	pip install -e .
	pip install mlx  # optional: Apple MLX backend (Apple Silicon only)

run:
	pyrex run

quick:
	pyrex run --quick --label "ci-$(shell date +%Y%m%d)"

baseline:
	pyrex baseline --name baseline

compare:
	@echo "Usage: make compare RUN=<run_id>"
	@pyrex compare baseline $(RUN)

report:
	@echo "Usage: make report RUN=<run_id>"
	@pyrex report $(RUN)

list:
	pyrex list

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=pyrex --cov-report=term-missing

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
