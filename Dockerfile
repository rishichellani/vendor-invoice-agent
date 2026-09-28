FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agent.py app.py checks.py ./
COPY services/ ./services/
COPY data/ ./data/
COPY .streamlit/ ./.streamlit/

RUN mkdir -p /app/output
VOLUME ["/app/output"]

# API keys are passed at `docker run` time (--env-file .env) and are never baked into the image.

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
