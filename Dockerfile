FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libreoffice-writer libreoffice-calc ghostscript fonts-dejavu && apt-get clean && rm -rf /var/lib/apt/lists/*
ENV LIBREOFFICE_PATH=/usr/bin/soffice
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
