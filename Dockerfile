FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV LIBREOFFICE_PATH=/usr/bin/soffice
CMD apt-get update && apt-get install -y --no-install-recommends libreoffice xvfb ghostscript fonts-dejavu && apt-get clean && rm -rf /var/lib/apt/lists/* && Xvfb :99 -screen 0 1024x768x24 & sleep 2 && DISPLAY=:99 uvicorn main:app --host 0.0.0.0 --port 10000
