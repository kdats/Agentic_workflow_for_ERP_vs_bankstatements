

## main.py


"""Entry point for executing the agentic reconciliation workflow locally."""
from __future__ import annotations
import argparse, json, sys
from config import init_vertex_ai, set_model_override
from agentic_workflow import run_workflow


def parse_args(argv):
    p = argparse.ArgumentParser(description="Run agentic reconciliation")
    p.add_argument("--erp", required=True, help="Path to ERP Excel")
    p.add_argument("--bank", required=True, help="Path to Bank PDF")
    p.add_argument("--output", required=True, help="Path to output Excel")
    p.add_argument("--model", default=None, help="Override Gemini model name")
    return p.parse_args(argv)

def main(argv):
    args = parse_args(argv)
    # optional: override model for this process
    set_model_override(args.model)
    init_vertex_ai()
    result = run_workflow(args.erp, args.bank, args.output, model_name=args.model)
    print(json.dumps({
        "narrative": result.narrative,
        "reconciled_rows": len(result.reconciled_rows),
        "discrepancies": len(result.discrepancies),
        "trace": result.trace,
    }, indent=2))

if __name__ == "__main__":
    main(sys.argv[1:])