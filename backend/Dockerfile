FROM python:3.12-slim-bookworm

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/app ./app
COPY models/catboost_model /models/catboost_model

RUN groupadd --gid 1000 app \
    && useradd --uid 1000 --gid app --home-dir /app --no-create-home app \
    && chown -R app:app /app /models

USER app

EXPOSE 8888

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8888"]
