FROM python:3.11-slim

WORKDIR /app

# Install curl for healthchecks
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Fix ONNX runtime thread affinity errors and parallelism
ENV OMP_NUM_THREADS=4
ENV ORT_DISABLE_CPU_AFFINITY=1
ENV TOKENIZERS_PARALLELISM=false

COPY . .

EXPOSE 8501

CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8501"]
