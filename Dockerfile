FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV LIBREOFFICE_PATH=/usr/bin/soffice
CMD apt-get update && apt-get install -y --no-install-recommends libreoffice ghostscript fonts-dejavu && apt-get clean && rm -rf /var/lib/apt/lists/* && uvicorn main:app --host 0.0.0.0 --port 10000
