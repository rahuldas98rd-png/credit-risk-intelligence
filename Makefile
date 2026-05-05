.PHONY: install test lint run-api run-app

install:
    pip install -r requirements.txt

test:
    pytest tests/ -v

lint:
    flake8 src/ api/ app/

run-api:
    uvicorn api.main:app --reload --port 8000

run-app:
    streamlit run app/streamlit_app.py

clean:
    find . -type f -name "*.pyc" -delete
    find . -type d -name "__pycache__" -delete
