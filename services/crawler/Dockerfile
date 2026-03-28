# syntax=docker/dockerfile:1.7
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/uv \
    pip install --no-cache-dir uv==0.8.13 \
    && uv pip install --system -r requirements.txt

COPY ./*.py ./

CMD ["python", "main.py"]
