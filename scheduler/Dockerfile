FROM python:3.11-slim

WORKDIR /app

# Ensure Python output is not buffered
ENV PYTHONUNBUFFERED=1

# Fix ONNX runtime thread affinity errors and parallelism
ENV OMP_NUM_THREADS=4
ENV ORT_DISABLE_CPU_AFFINITY=1
ENV TOKENIZERS_PARALLELISM=false

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "scheduler.py"]
