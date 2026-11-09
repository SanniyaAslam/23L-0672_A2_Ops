# VisionGuard - 23L-0672_A2_Ops
![CI](https://github.com/SanniyaAslam/23L-0672_A2_Ops/actions/workflows/ci.yml/badge.svg)

AI-generated vs real image classifier: Keras -> MLflow/DagsHub -> cloud model API -> Flask UI + Postgres -> Docker Compose.

## Architecture (Task 7 pathway B: Vercel)
    Browser -> Flask UI (localhost, Compose: web + db) --HTTP + X-API-Key--> Model API (Vercel, TFLite)
                          |
                       Postgres (history)

## Run order
1. python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
2. Put the Kaggle dataset in data/raw/, then: python src/audit.py data/raw && python src/eda.py && python src/data.py
3. dvc init; dvc add data/raw; dvc remote add -d origin <dagshub dvc url>; dvc push
4. cp .env.example .env  (fill DagsHub values)  ->  python src/run_experiments.sh (or bash src/run_experiments.sh)
5. python src/register_model.py --run-id <A>; python src/register_model.py --run-id <B> --alias champion
6. python src/load_champion.py --image test.jpg      # also writes models/champion.keras
7. Cloud VM: docker build -f model_api/Dockerfile -t vg-api . && docker run -d -p 8000:8000 --env-file .env --restart unless-stopped vg-api
8. Local: set INFERENCE_API_URL in .env, then docker compose up --build
