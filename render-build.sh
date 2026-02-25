#!/bin/bash
apt-get update
apt-get install -y libreoffice
pip install -r requirements.txt
```

Then in your Render dashboard:
- Go to your backend service → **Settings**
- Find **Build Command** and change it to:
```
chmod +x render-build.sh && ./render-build.sh
