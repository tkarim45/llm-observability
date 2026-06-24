"""`llmobs` CLI: seed traffic, print the metrics summary, serve the API, open the dashboard."""
from __future__ import annotations

import argparse
import json
import sys

from . import store
from .demo_app import generate_traffic
from .metrics import summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="llmobs", description="LLM observability + tracing")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_seed = sub.add_parser("seed", help="generate demo traffic into the trace store")
    p_seed.add_argument("--n", type=int, default=40)
    p_seed.add_argument("--reset", action="store_true", help="wipe the store first")

    sub.add_parser("summary", help="print the metrics summary")
    p_serve = sub.add_parser("serve", help="run the FastAPI service")
    p_serve.add_argument("--port", type=int, default=8000)
    sub.add_parser("dashboard", help="open the Streamlit dashboard")

    args = ap.parse_args(argv)

    if args.cmd == "seed":
        if args.reset:
            store.reset()
        n = generate_traffic(args.n)
        print(f"seeded {n} traces")
        print(json.dumps(summary(), indent=2))
    elif args.cmd == "summary":
        print(json.dumps(summary(), indent=2))
    elif args.cmd == "serve":
        import uvicorn
        uvicorn.run("api.main:app", host="0.0.0.0", port=args.port)
    elif args.cmd == "dashboard":
        import subprocess
        return subprocess.call([sys.executable, "-m", "streamlit", "run", "app/dashboard.py"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
