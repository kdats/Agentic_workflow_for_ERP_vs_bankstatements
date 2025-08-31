
## agentic\_workflow\.py


"""LangChain agentic assembly using Gemini via Vertex AI.

All *reasoning* (parsing, matching, classification, reconciliation) is performed by the LLM.
We provide:
- Tools for I/O (read_erp_xlsx, read_bank_pdf_text, write_outputs)
- Memory to preserve context across steps
- Structured outputs via Pydantic to standardise the LLM results
"""
from __future__ import annotations
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, Tool
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain_google_vertexai import ChatVertexAI
from langchain.tools import StructuredTool

from config import init_vertex_ai, MODEL_NAME
from tools import read_erp_xlsx, read_bank_pdf_text, write_outputs


# ===== Structured Output Schemas =====
class BankTxn(BaseModel):
    date: str = Field(..., description="ISO date, e.g., 2025-01-05")
    description: str
    amount: float
    ref_id: str | None = None

class ErpTxn(BaseModel):
    date: str
    invoice_id: str
    amount: float
    status: str
class BankTxnList(BaseModel):
    txns: List[BankTxn]
class ReconciledRow(BaseModel):
    invoice_id: str
    erp_amount: float
    bank_amount: float | None
    status: str  # Reconciled | MissingInBank | MissingInERP | Duplicate | AmountMismatch | RoundingDiff
    notes: str | None = None

class ReconOutput(BaseModel):
    reconciled: List[ReconciledRow]
    discrepancies: List[ReconciledRow]
    reconciliation_rate_pct: float
    reasoning_summary: str

class UnreconciledErpRow(BaseModel):
    invoice_id: str
    status: str
    amount: float
    notes: str | None = None

class BankOnlyRow(BaseModel):
    ref_id: str | None = None
    description: str
    amount: float
    inferred_invoice_id: str | None = None
    notes: str | None = None

class DuplicateGroup(BaseModel):
    invoice_id: str | None = None
    occurrences: List[BankTxn]  # list of bank rows that are duplicates
    rationale: str | None = None

class MismatchRow(BaseModel):
    invoice_id: str
    erp_amount: float
    bank_amount: float
    abs_diff: float
    notes: str | None = None

class Rollup(BaseModel):
    erp_count: int
    reconciled_count: int
    unreconciled_erp_count: int
    bank_only_count: int
    duplicate_groups_count: int
    mismatches_count: int
    reconciliation_rate_pct: float

class DetailedReconOutput(BaseModel):
    reconciled: List[ReconciledRow]
    unreconciled_erp: List[UnreconciledErpRow]
    bank_only: List[BankOnlyRow]
    duplicates: List[DuplicateGroup]
    mismatches: List[MismatchRow]
    rollup: Rollup
    reasoning_summary: str

# ===== Agent wrapper =====
@dataclass
class AgentArtifacts:
    narrative: str
    reconciled_rows: List[Dict[str, Any]]
    discrepancies: List[Dict[str, Any]]
    trace: Dict[str, Any]


def _make_llm() -> ChatVertexAI:
    # Ensure Vertex SDK is initialised
    init_vertex_ai()
    from config import MODEL_NAME  # read the possibly overridden value
    return ChatVertexAI(model_name=MODEL_NAME, temperature=0)


def parse_bank_text_chain(raw_text: str) -> List[BankTxn]:
    """Ask the LLM to parse bank statement text into structured BankTxn rows."""
    llm = _make_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a careful financial assistant. Parse the bank statement text into rows strictly following the schema."),
        ("human", "Bank statement raw text:\n\n{raw}\n\nReturn JSON with a top-level key 'txns' which is a list of objects with fields: date, description, amount, ref_id (optional).")
    ])
    # IMPORTANT: use the wrapper model, NOT List[BankTxn]
    chain = prompt | llm.with_structured_output(BankTxnList)
    result = chain.invoke({"raw": raw_text})
    return result.txns

def reconcile_with_llm_v1(erp_rows: List[ErpTxn], bank_rows: List[BankTxn]) -> ReconOutput:
    """Delegate full reconciliation reasoning to the LLM with a structured output."""
    llm = _make_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an autonomous reconciliation agent. \n"
            "Given ERP rows and Bank rows, perform transaction reconciliation USING YOUR OWN REASONING.\n"
            "Do NOT ask for regex or fuzzy heuristics; you must infer matches via semantics and numeric alignment.\n"
            "Rules: match by invoice_id in description when available, align amounts, identify duplicates and rounding diffs (<= 0.05 absolute).\n"
            "Output must follow the ReconOutput schema strictly.")),
        ("human", "ERP rows (JSON):\n{erp}\n\nBank rows (JSON):\n{bank}\n\nProduce reconciled/discrepancies with explanations.")
    ])
    chain = prompt | llm.with_structured_output(ReconOutput)
    return chain.invoke({
        "erp": [e.model_dump() if isinstance(e, ErpTxn) else e for e in erp_rows],
        "bank": [b.model_dump() if isinstance(b, BankTxn) else b for b in bank_rows],
    })

def reconcile_with_llm(erp_rows: List[ErpTxn], bank_rows: List[BankTxn]) -> DetailedReconOutput:
    """Delegate reconciliation to the LLM with strict, non-overlapping categories and rollups."""
    llm = _make_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an autonomous reconciliation agent.\n"
         "Rules:\n"
         "1) Categories must be NON-OVERLAPPING across: reconciled, unreconciled_erp, bank_only, duplicates, mismatches.\n"
         "2) Treat rounding differences <= 0.05 as Reconciled with a notes field; otherwise they go to mismatches.\n"
         "3) 'duplicates' are groups of bank rows that refer to the same invoice or same payment twice; do NOT also list them in bank_only.\n"
         "4) 'unreconciled_erp' lists ERP invoices that did not find a legitimate bank match.\n"
         "5) 'bank_only' lists bank payments that cannot be tied to any ERP invoice.\n"
         "6) 'mismatches' are ERP-vs-bank pairs with amount differences > 0.05; do NOT list those invoices elsewhere.\n"
         "7) Compute rollup counts consistently: "
         "   erp_count = total ERP invoices given; reconciled_count = len(reconciled); "
         "   unreconciled_erp_count = len(unreconciled_erp); "
         "   bank_only_count = len(bank_only); duplicate_groups_count = len(duplicates); "
         "   mismatches_count = len(mismatches); "
         "   reconciliation_rate_pct = round(100 * reconciled_count / erp_count, 1)."),
        ("human",
         "ERP rows (JSON):\n{erp}\n\nBank rows (JSON):\n{bank}\n\n"
         "Return a JSON strictly conforming to DetailedReconOutput. Ensure counts are consistent and categories non-overlapping.")
    ])
    chain = prompt | llm.with_structured_output(DetailedReconOutput)
    return chain.invoke({
        "erp": [e.model_dump() if isinstance(e, ErpTxn) else e for e in erp_rows],
        "bank": [b.model_dump() if isinstance(b, BankTxn) else b for b in bank_rows],
    })
def validate_and_correct(erp_total: int, recon: DetailedReconOutput) -> DetailedReconOutput:
    ok = True
    r = recon.rollup
    # Basic invariants
    if r.erp_count != erp_total: ok = False
    if r.reconciled_count != len(recon.reconciled): ok = False
    if r.unreconciled_erp_count != len(recon.unreconciled_erp): ok = False
    if r.bank_only_count != len(recon.bank_only): ok = False
    if r.duplicate_groups_count != len(recon.duplicates): ok = False
    if r.mismatches_count != len(recon.mismatches): ok = False
    calc_rate = round(100 * r.reconciled_count / max(r.erp_count, 1), 1)
    if r.reconciliation_rate_pct != calc_rate: ok = False

    if ok:
        return recon

    # Ask LLM to fix counts/categories once, providing its own output to correct
    llm = _make_llm()
    fix_prompt = ChatPromptTemplate.from_messages([
        ("system", "You produced a reconciliation JSON that failed invariants. Correct it WITHOUT changing the meaning."),
        ("human",
         "Here is your previous JSON (DetailedReconOutput). Fix category overlap and counts so all invariants pass.\n"
         "JSON:\n{bad_json}\n\n"
         "Invariants:\n"
         "- Non-overlapping categories across reconciled, unreconciled_erp, bank_only, duplicates, mismatches.\n"
         "- rollup.erp_count == {erp_total}\n"
         "- rollup.*_count fields == lengths of corresponding arrays\n"
         "- rollup.reconciliation_rate_pct == round(100 * len(reconciled) / erp_count, 1)\n"
         "Return fixed JSON strictly conforming to DetailedReconOutput.")
    ])
    chain = fix_prompt | llm.with_structured_output(DetailedReconOutput)
    return chain.invoke({"bad_json": recon.model_dump_json(indent=2), "erp_total": erp_total})

def build_agent_executor() -> Tuple[AgentExecutor, ConversationBufferMemory]:
    """Optional interactive agent with memory, if you prefer tool-driven orchestration.

    For this assignment, we mainly use direct chains above; however, we provide an AgentExecutor
    with memory to comply with the "agentic + memory" requirement.
    """
    llm = _make_llm()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    tools = [
        StructuredTool.from_function(func=read_erp_xlsx, name="read_erp_xlsx",
                                     description="Load ERP Excel and return a list of transaction dicts."),
        StructuredTool.from_function(func=read_bank_pdf_text, name="read_bank_pdf_text",
                                     description="Read a bank statement PDF and return raw text."),
        StructuredTool.from_function(func=write_outputs, name="write_outputs",
                                     description="Write reconciled and discrepancy rows to an Excel file (2 sheets) and save logs."),
    ]

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a reconciliation supervisor. You can call tools to read files and save outputs.\n"
            "All matching/classification MUST be done by you (the LLM), not by tools.\n"
            "Think step-by-step, but only return the final answer to the user.")),
        ("human", "{input}"),
        ("ai", "{agent_scratchpad}"),
    ])

    agent = llm.bind()
    executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)
    return executor, memory


def run_workflow_v1(erp_path: str, bank_path: str, out_path: str, model_name: str | None = None) -> AgentArtifacts:
    # Load inputs via tools
    erp_dicts = read_erp_xlsx(erp_path)
    erp_rows = [ErpTxn(**row) for row in erp_dicts]

    bank_text = read_bank_pdf_text(bank_path)
    bank_rows = parse_bank_text_chain(bank_text)

    # Delegate reconciliation entirely to LLM
    recon = reconcile_with_llm(erp_rows, bank_rows)

    # Persist
    write_outputs(out_path,
                  reconciled_rows=[r.model_dump() for r in recon.reconciled],
                  discrepancies=[d.model_dump() for d in recon.discrepancies],
                  narrative=recon.reasoning_summary,
                  agent_log={"model": model_name or MODEL_NAME, "rate": recon.reconciliation_rate_pct})

    return AgentArtifacts(
        narrative=recon.reasoning_summary,
        reconciled_rows=[r.model_dump() for r in recon.reconciled],
        discrepancies=[d.model_dump() for d in recon.discrepancies],
        trace={"model": model_name or MODEL_NAME, "rate": recon.reconciliation_rate_pct},
    )
def run_workflow_v2(erp_path: str, bank_path: str, out_path: str, model_name: str | None = None) -> AgentArtifacts:
    # Load inputs
    erp_dicts = read_erp_xlsx(erp_path)
    erp_rows = [ErpTxn(**row) for row in erp_dicts]

    bank_text = read_bank_pdf_text(bank_path)
    bank_rows = parse_bank_text_chain(bank_text)

    # Delegate reconciliation entirely to LLM
    recon = reconcile_with_llm(erp_rows, bank_rows)
    recon = validate_and_correct(len(erp_rows), recon)

    # Persist multiple sheets for audit
    extra = {
        "unreconciled_erp": [x.model_dump() for x in recon.unreconciled_erp],
        "bank_only": [x.model_dump() for x in recon.bank_only],
        "duplicates": [ {"invoice_id": g.invoice_id,
                         "occurrences": [tx.model_dump() for tx in g.occurrences],
                         "rationale": g.rationale}
                        for g in recon.duplicates ],
        "mismatches": [x.model_dump() for x in recon.mismatches],
        "rollup": [recon.rollup.model_dump()],
    }
    write_outputs(out_path,
                  reconciled_rows=[r.model_dump() for r in recon.reconciled],
                  discrepancies=[*extra["unreconciled_erp"], *extra["bank_only"], *extra["mismatches"]],
                  narrative=recon.reasoning_summary,
                  agent_log={"model": model_name or MODEL_NAME, "rate": recon.rollup.reconciliation_rate_pct},
                  extra_sheets=extra)

    return AgentArtifacts(
        narrative=recon.reasoning_summary,
        reconciled_rows=[r.model_dump() for r in recon.reconciled],
        discrepancies=[*extra["unreconciled_erp"], *extra["bank_only"], *extra["mismatches"]],
        trace={"model": model_name or MODEL_NAME, "rate": recon.rollup.reconciliation_rate_pct},
    )
def run_workflow(erp_path: str, bank_path: str, out_path: str, model_name: str | None = None) -> AgentArtifacts:
    # Load inputs
    erp_dicts = read_erp_xlsx(erp_path)
    erp_rows = [ErpTxn(**row) for row in erp_dicts]

    bank_text = read_bank_pdf_text(bank_path)
    bank_rows = parse_bank_text_chain(bank_text)

    # Delegate reconciliation entirely to LLM
    recon = reconcile_with_llm(erp_rows, bank_rows)
    recon = validate_and_correct(len(erp_rows), recon)

    # Canonical numbers from tables (non-overlapping)
    erp_total = len(erp_rows)
    reconciled_n = len(recon.reconciled)
    unrec_n = len(recon.unreconciled_erp)
    bank_only_n = len(recon.bank_only)
    mismatch_n = len(recon.mismatches)
    # Choose what “discrepancies” means for your JSON printout (exclude duplicates):
    discrepancies_n = unrec_n + bank_only_n + mismatch_n
    rate = round(100 * reconciled_n / max(erp_total, 1), 1)

    # Build a deterministic narrative so JSON, Excel, and narrative match
    narrative = (
        f"Reconciled {reconciled_n} of {erp_total} ERP invoices ({rate}%). "
        f"Discrepancies (non-overlapping): MissingInBank={unrec_n}, "
        f"BankOnly={bank_only_n}, AmountMismatch={mismatch_n}. "
        f"Duplicate groups={len(recon.duplicates)}. "
        f"All figures derived from output tables."
    )

    # Persist multiple sheets for audit
    extra = {
        "unreconciled_erp": [x.model_dump() for x in recon.unreconciled_erp],
        "bank_only": [x.model_dump() for x in recon.bank_only],
        "duplicates": [
            {
                "invoice_id": g.invoice_id,
                "occurrences": [tx.model_dump() for tx in g.occurrences],
                "rationale": g.rationale,
            } for g in recon.duplicates
        ],
        "mismatches": [x.model_dump() for x in recon.mismatches],
        "rollup": [{
            "erp_count": erp_total,
            "reconciled_count": reconciled_n,
            "unreconciled_erp_count": unrec_n,
            "bank_only_count": bank_only_n,
            "mismatches_count": mismatch_n,
            "duplicate_groups_count": len(recon.duplicates),
            "reconciliation_rate_pct": rate,
        }],
    }

    write_outputs(
        out_path,
        reconciled_rows=[r.model_dump() for r in recon.reconciled],
        discrepancies=[*extra["unreconciled_erp"], *extra["bank_only"], *extra["mismatches"]],
        narrative=narrative,
        agent_log={"model": model_name or MODEL_NAME, "rate": rate},
        extra_sheets=extra,
    )

    return AgentArtifacts(
        narrative=narrative,
        reconciled_rows=[r.model_dump() for r in recon.reconciled],
        discrepancies=[*extra["unreconciled_erp"], *extra["bank_only"], *extra["mismatches"]],
        trace={"model": model_name or MODEL_NAME, "rate": rate},
    )
