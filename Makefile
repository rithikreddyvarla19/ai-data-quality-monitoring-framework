.PHONY: install test lint run-api run-dashboard docker-up sample-run

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check .

run-api:
	uvicorn api.main:app --reload --port 8000

run-dashboard:
	streamlit run dashboards/streamlit_app.py

docker-up:
	docker compose up --build

sample-run:
	python scripts/run_monitoring.py --reference data/sample/reference_dataset.csv --current data/sample/current_dataset.csv --predictions data/sample/model_predictions.csv --output outputs/sample_monitoring_report.json

