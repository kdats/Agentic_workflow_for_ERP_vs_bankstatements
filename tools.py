
## tools.py


"""I/O tools only (no matching logic). LLM handles all reasoning.

- read_erp_xlsx: load ERP Excel to rows
- read_bank_pdf_text: extract raw text from PDF (no regex)
- write_outputs: persist reconciled dataset + narrative/logs
"""
from __future__ import annotations
from typing import List, Dict, Any
import pandas as pd
from pypdf import PdfReader
from pathlib import Path
import json


def read_erp_xlsx(path: str) -> List[Dict[str, Any]]:
    df = pd.read_excel(path)
    # Normalize expected column names for the Pydantic model
    colmap = {
        "Date": "date",
        "Invoice ID": "invoice_id",
        "Amount": "amount",
        "Status": "status",
    }
    # Be robust to casing/whitespace variants
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={k: v for k, v in colmap.items() if k in df.columns})

    # Types the LLM can consume
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    # Optional: drop rows that failed to parse essential fields
    df = df.dropna(subset=["date", "invoice_id", "amount", "status"])

    return df.to_dict(orient="records")




def read_bank_pdf_text(path: str) -> str:
    # Extract plain text; *no* parsing logic here
    text_parts: List[str] = []
    reader = PdfReader(path)
    for page in reader.pages:
        text = page.extract_text() or ""
        text_parts.append(text)
    return "\n".join(text_parts)


def write_outputs_v1(output_path: str, reconciled_rows: List[Dict[str, Any]],
                  discrepancies: List[Dict[str, Any]],
                  narrative: str, agent_log: Dict[str, Any]) -> None:
    """Persist outputs to disk: an Excel with 2 sheets + a JSON log file."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        pd.DataFrame(reconciled_rows).to_excel(writer, sheet_name="reconciled", index=False)
        pd.DataFrame(discrepancies).to_excel(writer, sheet_name="discrepancies", index=False)

    # Also drop the narrative + agent trace
    log_path = out.with_suffix(".logs.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({"narrative": narrative, "trace": agent_log}, f, ensure_ascii=False, indent=2)
        
def write_outputs(output_path: str,
                  reconciled_rows: List[Dict[str, Any]],
                  discrepancies: List[Dict[str, Any]],
                  narrative: str,
                  agent_log: Dict[str, Any],
                  extra_sheets: Dict[str, List[Dict[str, Any]]] | None = None) -> None:
    """Persist outputs to disk: Excel with multiple sheets + JSON log."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        pd.DataFrame(reconciled_rows).to_excel(writer, sheet_name="reconciled", index=False)
        pd.DataFrame(discrepancies).to_excel(writer, sheet_name="discrepancies", index=False)
        if extra_sheets:
            for name, rows in extra_sheets.items():
                sheet_name = (name or "sheet")[:31]  # Excel sheet name limit
                pd.DataFrame(rows).to_excel(writer, sheet_name=sheet_name, index=False)

    log_path = out.with_suffix(".logs.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({"narrative": narrative, "trace": agent_log}, f, ensure_ascii=False, indent=2)