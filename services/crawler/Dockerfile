FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir uv==0.8.13 \
    && uv pip install --system -r requirements.txt

COPY ./*.py ./

CMD ["python", "main.py"]
