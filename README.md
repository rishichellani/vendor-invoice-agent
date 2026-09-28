# Vendor Invoice & Compliance Reconciliation Agent

Upload one vendor invoice or a whole batch (PDF or image); the agent extracts each, looks up the
matching purchase order, master contract and vendor-master record, applies the Halvorsen AP policy
(FIN-AP-004), and returns **Approved**, **Flagged**, or **Rejected** with explicit discrepancy reasons.

## Architecture

| Layer    | Where | What |
|----------|-------|------|
| Goal     | `agent.py` | Audit an invoice for full financial and policy compliance. |
| Planner  | `agent.audit_invoice` | extract fields → look up PO + contract + vendor + policy → load invoice history → cross-check → policy review → verdict. Every step is recorded in the report's `trace`. |
| Memory / RAG | `services/reference_store.py` | `ReferenceStore` interface; `LocalReferenceStore` reads the records under `data/`. Past audits in `output/` are the invoice history. |
| Executor | `checks.py` (`run_checks`) | Deterministic policy rules: three-way match, price/quantity/total tolerances, cumulative billing and not-to-exceed, freight and sales-tax rules, contractor T&E limits, milestone billing, duplicate/reprint detection, vendor-master and bank-fraud red flags, contract validity, payment terms, tiered approval authority. Thresholds live in `data/policies/thresholds.json`. Only these rules can produce a *Rejected* verdict. |
| AI review | `agent.review_policy` | An LLM reads the invoice, PO, contract and policy for anything the rules can't see. Its findings are advisory (capped at *warning*); it can add findings but never reject or downgrade. |
| UI       | `app.py` | Streamlit multi-file uploader, batch summary table (CSV export) and per-invoice audit dashboard. |

LLM calls request JSON from Gemini first and fall back to Groq on any Gemini failure
(rate limit, 5xx, unparseable output). The Groq fallback reads PDFs via text extraction, and
images and scanned image-only PDFs via a vision model.

## Reference data (`data/`)

- `purchase_orders/<PO>.json`, `contracts/<ID>.json` -- structured records built from the source PDFs in `reference_pdfs/`. Contracts carry machine-readable `rules` (tax treatment, freight terms, expense limits, milestones) and dated rate cards (e.g. annual escalators).
- `vendors.json` -- the vendor master (legal name, address, email, phone) used for the fraud checks.
- `policies/` -- the AP policy text for the LLM and `thresholds.json` (tolerances, approval tiers) for the rules.
- `sample_invoices/` -- 13 test invoices covering clean, duplicate, fraud, over-billing, T&E and scanned cases. `legacy_mock/` holds the earlier mock data set.

To use different reference documents, drop new JSON records into these folders (mount `data/` in Docker to avoid a rebuild).

## Audit history

Every audit is saved to `output/audit_<file>.json`. Later audits read these to catch **duplicate invoices** and **cumulative over-billing** on a PO. Invoices are audited in filename order in a batch so an original is on record before its reprint. Use *Audit history > Clear audit history* in the sidebar for a fresh test run.

## Local development

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # GEMINI_API_KEY (required), GROQ_API_KEY (fallback)
streamlit run app.py
```

## Docker

```bash
docker build -t invoice-agent .
docker run --rm -p 8501:8501 --env-file .env \
  -v "$(pwd)/output":/app/output -v "$(pwd)/data":/app/data:ro invoice-agent
```

Audit reports are saved as JSON under `output/` (mounted so history survives restarts).

## Environment variables

| Variable           | Required | Purpose |
|--------------------|----------|---------|
| `GEMINI_API_KEY`   | Yes      | Primary engine. |
| `GROQ_API_KEY`     | No       | Fallback engine. |
| `GEMINI_MODEL`     | No       | Override the Gemini model id. |

On Streamlit Community Cloud, set these under **Settings > Secrets**; `app.py` mirrors them into the environment.

## Tests

```bash
python -m unittest discover -s tests -v
```

The rules are tested without any LLM or network: `tests/fixtures/invoices.json` holds the extracted data for the 13 sample invoices, and the tests assert each verdict and its key findings (duplicates, fraud flags, milestones, escalators, T&E limits).
