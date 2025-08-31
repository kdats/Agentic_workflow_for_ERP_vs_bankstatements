# Financial Data Reconciliation with Agentic AI

## Problem Statement
Organizations often need to compare internal Enterprise Resource Planning (ERP) records with external bank statements. Manual reconciliation is time‑consuming and error‑prone. The goal is to automatically reconcile transactions between an ERP Excel file and a Bank Statement PDF and surface discrepancies such as missing entries, duplicates, or amount mismatches.

More detailes are provided here in PDF doc :  [Overview](https://github.com/kdats/Agentic_workflow_for_ERP_vs_bankstatements/blob/main/assignment_brief_updated%20(1)%20(1).pdf)

## Objective
Design an end‑to‑end workflow powered by autonomous AI agents that:
- Read ERP data from Excel and bank data from PDF.
- Reconcile transactions using large language models (LLMs) instead of traditional scripts.
- Explain decisions and produce a reconciliation summary file and narrative report.

## High‑Level Solution
1. **Input ingestion** – tools extract structured ERP rows from Excel and raw text from the bank statement PDF.
2. **LLM reasoning** – Google Gemini (via Vertex AI) parses bank text, matches transactions, classifies discrepancies and computes rollups.
3. **Memory** – a `ConversationBufferMemory` retains intermediate context so the agent can reason across multiple steps.
4. **Output generation** – reconciled rows, discrepancies and narrative are written to an Excel workbook plus a JSON log for traceability.

## Installation & Setup
```bash
# install dependencies
pip install -r requirements.txt

# authenticate to Google Cloud (once per environment)
gcloud auth application-default login

# set the project used by Vertex AI
gcloud config set project <your-gcp-project>
```
Environment variables can override defaults in `config.py` (e.g. `GCP_PROJECT`, `GENAI_MODEL`).

## Workflow Components
- **`main.py`** – command-line entry that initialises Vertex AI and triggers the workflow.
- **`workflow_agent.py`** – implemented in [`agentic_workflow.py`](agentic_workflow.py); builds the LangChain agent, adds memory, and delegates reconciliation to Gemini.
- **`tools.py`** – pure I/O utilities for reading ERP Excel, extracting PDF text, and writing outputs; all matching logic is deferred to the LLM.
- **`config.py`** – centralises Vertex AI configuration and allows overriding the Gemini model at runtime.

## Agentic AI Capabilities
- Uses **Google Gemini** models through Vertex AI for parsing, matching and reasoning.
- Reconciliation is entirely performed by the LLM using structured outputs (`Pydantic` schemas).
- Employs conversation memory so later steps can reference earlier decisions.

## Running the Workflow
```bash
python main.py --erp erp_data.xlsx --bank bank_statement.pdf --output reconciliation_summary.xlsx --model gemini-2.5-pro
```
The command produces `reconciliation_summary.xlsx` with sheets for reconciled and discrepant transactions and a companion `reconciliation_summary.logs.json` containing the narrative and trace.

## Interpreting the Output
- **Reconciled** – rows where ERP and bank amounts align (tolerating minor rounding).
- **Discrepancies** – classified issues such as `MissingInERP`, `MissingInBank`, `Duplicate`, `AmountMismatch`, or `RoundingDiff`.
- **Narrative & trace** – the log file captures the agent's reasoning steps for auditability.

## Notes on Memory Experiments
The workflow explores adding memory to Agentic AI. `ConversationBufferMemory` preserves dialogue between steps, enabling the agent to reference prior context when validating or correcting reconciliation results.

