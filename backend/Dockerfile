# AI Comment Moderation API - backend container (build context: ./backend)
FROM python:3.11-slim

WORKDIR /app
ENV NLTK_DATA=/usr/local/share/nltk_data

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Provision NLTK corpora at build time (preprocess.py never downloads at runtime)
RUN python -c "import nltk; [nltk.download(r, download_dir='/usr/local/share/nltk_data', quiet=True) for r in ('stopwords','wordnet')]"

COPY . .

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/ready')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
