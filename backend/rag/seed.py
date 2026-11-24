"""python -m rag.seed [--if-empty]: create demo users, generate SYNTHETIC sample docs and ingest them."""
import argparse
from . import runtime, config
from .db import Session, Document
from .ingest import ingest
from .sample_data import write_files


def run(if_empty=False):
    runtime.init()
    with Session() as s:
        if if_empty and s.query(Document).count(): return print("already seeded")
    roles = write_files(config.SAMPLE_DIR, config.EVAL_FILE)
    for name, r in roles.items(): ingest(name, (config.SAMPLE_DIR / name).read_bytes(), r, "seed")
    print(f"seeded {len(roles)} synthetic documents and demo users")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--if-empty", action="store_true"); run(ap.parse_args().if_empty)
