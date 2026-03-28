FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir uv==0.8.13 \
    && uv pip install --system -r requirements.txt

COPY app.py ./

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
