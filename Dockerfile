FROM python:3.11-slim

WORKDIR /app

# Lean install on purpose: transformers/torch (needed only for the grey-zone
# ML classifier) are excluded here. router.py's get_classifier() lazy-imports
# transformers inside a try/except that already falls back to the rule-based
# score on any failure (including ModuleNotFoundError) — see src/router.py.
# Install the full requirements.txt instead if you want that path to work.
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY main.py .
COPY src/ src/
COPY eval/ eval/

RUN mkdir -p cache_store logs

CMD ["python", "main.py"]
