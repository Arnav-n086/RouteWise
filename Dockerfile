FROM python:3.11-slim

WORKDIR /app

# requirements.txt is intentionally lean — transformers/torch (needed only
# for the grey-zone ML classifier) live in the opt-in requirements-ml.txt
# instead. router.py's get_classifier() lazy-imports transformers inside a
# try/except that already falls back to the rule-based score on any failure
# (including ModuleNotFoundError) — see src/router.py. Add requirements-ml.txt
# to this install if you want that path to work inside the container too.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY src/ src/
COPY eval/ eval/

RUN mkdir -p cache_store logs

CMD ["python", "main.py"]
