.PHONY: setup seed api test eval up
setup: ; pip install -r requirements-dev.txt
seed:  ; python -m rag.seed
api:   ; uvicorn rag.api:app --reload
test:  ; pytest -q
eval:  ; python -m rag.evaluate
up:    ; docker compose up --build
