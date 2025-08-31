# Financial Data Reconciliation with Agentic AI

## Problem Statement
Organizations often need to compare internal Enterprise Resource Planning (ERP) records with external bank statements. Manual reconciliation is time‑consuming and error‑prone. The goal is to automatically reconcile transactions between an ERP Excel file and a Bank Statement PDF and surface discrepancies such as missing entries, duplicates, or amount mismatches.

More detailes are provided here in PDF doc :  [Overview](https://github.com/kdats/Agentic_workflow_for_ERP_vs_bankstatements/blob/main/assignment_brief_updated%20(1)%20(1).pdf)

## Objective
Design an end‑to‑end workflow powered by autonomous AI agents that:
- Read ERP data from Excel and bank data from PDF.
- Reconcile transactions using large language models (LLMs) instead of traditional scripts.
- When you reconcile, each transaction (or invoice) should fall into exactly one category.
- Overlapping means the same record is counted in two categories (e.g., a cancelled ERP invoice that also gets flagged as a mismatch). That inflates your discrepancy totals and makes the rollup inconsistent.
- Non-overlapping means categories are mutually exclusive. Each invoice or bank transaction is classified once, no double counting.
- Explain decisions and produce a reconciliation summary file and narrative report.


## High‑Level Solution
1. **Input ingestion** – tools extract structured ERP rows from Excel and raw text from the bank statement PDF.
2. **LLM reasoning** – Google Gemini (via Vertex AI) parses bank text, matches transactions, classifies discrepancies and computes rollups.
3. **Memory** – a `ConversationBufferMemory` retains intermediate context so the agent can reason across multiple steps.
4. **Output generation** – reconciled rows, discrepancies and narrative are written to an Excel workbook plus a JSON log for traceability.

## Output Categories (mutually exclusive) 

- Reconciled
→ ERP invoice has a bank payment that matches amount (or within rounding tolerance).

- MissingInBank
→ ERP invoice expected a payment (status = Paid in ERP) but no bank record found.

- BankOnly
→ Bank transaction exists, but there’s no corresponding ERP invoice (or it’s cancelled).

- AmountMismatch
→ ERP invoice matched to a bank record, but amounts differ by more than tolerance.

- Duplicate
→ Same invoice paid twice in bank statement (multiple bank entries for one invoice).

### Example of **Reconciled**

* ERP row: `2025-02-17 | INV0106 | 1123.26 | Paid`
* Bank row: `2025-02-17 | Payment INV0106 | 1123.26 | Ref …`
  → Perfect match, goes to **Reconciled**.


### Example of **MissingInBank**

* ERP row: `2025-02-15 | INV0107 | 425.54 | Paid`
* Bank statement: No entry with INV0107.
  → Invoice marked Paid in ERP, but not found in bank → **MissingInBank**.


### Example of **BankOnly**

* Bank row: `2025-01-06 | Adjustment -37.76 | Ref 1000`
* No such invoice in ERP (ERP has only positive amounts for invoices).
  → Bank has a transaction ERP never recorded → **BankOnly**.


### Example of **AmountMismatch**

* ERP row: `2025-02-15 | INV0021 | 365.02 | Paid`
* Bank rows:

  * `2025-02-20 | Payment INV0021 | 364.00 | Ref …`
    → Same invoice, but 1.02 INR difference (bigger than 0.05 tolerance) → **AmountMismatch**.


### Example of **Duplicate**

* Bank has two identical rows:

  * `2025-02-10 | Payment INV0017 | 1992.44 | Ref 17`
  * `2025-02-10 | Payment INV0017 | 1992.44 | Ref 17`
* ERP has just one `INV0017` invoice.
  → Bank shows the payment twice → **Duplicate**.

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
- <img width="1318" height="425" alt="image" src="https://github.com/user-attachments/assets/7594b966-e5be-4ac3-8940-1b5e195fd59c" />
- **Discrepancies** – classified issues such as `MissingInERP`, `MissingInBank`, `Duplicate`, `AmountMismatch`, or `RoundingDiff`.
- <img width="2011" height="379" alt="image" src="https://github.com/user-attachments/assets/207222f5-7fb1-4dee-8ea2-41810f98cd3a" />
- **Narrative & trace** – the log file captures the agent's reasoning steps for auditability.

```
{
  "narrative": "Reconciled 165 of 200 ERP invoices (82.5%). Discrepancies (non-overlapping): MissingInBank=20, BankOnly=20, AmountMismatch=7. Duplicate groups=8. All figures derived from output tables.",
  "trace": {
    "model": "gemini-2.5-pro",
    "rate": 82.5
  }
 }
 ```

## Notes on Memory Experiments
The workflow explores adding memory to Agentic AI. `ConversationBufferMemory` preserves dialogue between steps, enabling the agent to reference prior context when validating or correcting reconciliation results.

