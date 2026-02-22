FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libreoffice-writer \
    libreoffice-calc \
    ghostscript \
    fonts-dejavu \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV LIBREOFFICE_PATH=/usr/bin/soffice

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
```

Key changes:
- `libreoffice-writer` + `libreoffice-calc` instead of full `libreoffice` (much smaller)
- `--no-install-recommends` (cuts size significantly)
- `apt-get clean` added
- `LIBREOFFICE_PATH=/usr/bin/soffice` — the binary is actually called **`soffice`**, not `libreoffice` on Debian/Ubuntu slim images

---

## Also Add CORS to Render Environment

You also need to add this env variable in Render → Environment:
```
CORS_ALLOW_ORIGINS=https://pdf-editor-frontend-cyan.vercel.app
