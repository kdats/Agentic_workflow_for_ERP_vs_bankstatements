"""Centralised configuration for Vertex AI + LangChain."""
from __future__ import annotations
import os
from google.cloud import aiplatform

PROJECT = os.getenv("GCP_PROJECT", "gen-lang-client-0981824891")
LOCATION = os.getenv("GCP_LOCATION", "us-central1")
# default; can be overridden at runtime
MODEL_NAME = os.getenv("GENAI_MODEL", "gemini-2.5-pro")

_aiplatform_inited = False

def init_vertex_ai() -> None:
    global _aiplatform_inited
    if not _aiplatform_inited:
        aiplatform.init(project=PROJECT, location=LOCATION)
        _aiplatform_inited = True

def set_model_override(name: str | None) -> None:
    """Allow main.py to override the model for the current process."""
    global MODEL_NAME
    if name:
        MODEL_NAME = name

# pip install -r requirements.txt && 
# gcloud auth application-default login && 
# gcloud config set project gen-lang-client-0981824891 && 
# python main.py --erp "erp_data.xlsx" --bank "bank_statement.pdf" --output "reconciliation_summary.xlsx" --model "gemini-2.5-pro"