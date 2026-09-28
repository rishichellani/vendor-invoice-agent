"""Streamlit front-end for the vendor invoice reconciliation agent.

Upload a PDF/image invoice in the sidebar; agent.audit_invoice extracts it,
pulls the PO / contract / policy, runs the cross-checks, and this page renders
the resulting audit dashboard.
"""
import html
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

# On Streamlit Community Cloud, secrets set in the dashboard land in
# st.secrets, not in the environment -- mirror them into os.environ so
# agent.py's os.environ / dotenv-based lookups keep working unchanged.
# Locally there's no secrets.toml at all (we use .env instead), and merely
# touching st.secrets in that case raises StreamlitSecretNotFoundError, so
# this whole block is best-effort.
try:
    for _key in ("GEMINI_API_KEY", "GROQ_API_KEY"):
        if _key in st.secrets and not os.environ.get(_key):
            os.environ[_key] = st.secrets[_key]
except st.errors.StreamlitSecretNotFoundError:
    pass

from agent import OUTPUT_DIR, audit_invoice

st.set_page_config(page_title="Invoice Reconciliation Agent", page_icon="🧾", layout="wide")

# Brand styling to match rishichellani.netlify.app (navy/teal, -apple-system stack).
# Theme colors (dark base, teal primary) come from .streamlit/config.toml; this
# covers the bits Streamlit's theme engine doesn't reach: fonts, buttons, and
# the custom header/footer.
st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    }
    .audit-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(14,165,233,0.12);
        border: 1px solid rgba(14,165,233,0.3);
        border-radius: 100px;
        padding: 0.375rem 1rem;
        font-size: 0.8125rem;
        color: #38bdf8;
        margin-bottom: 1rem;
        letter-spacing: 0.04em;
        font-weight: 500;
    }
    .audit-badge::before { content: '●'; font-size: 0.5rem; }
    .audit-title { font-size: 2.25rem; font-weight: 700; letter-spacing: -0.03em; margin-bottom: 0.25rem; }
    .audit-title span { color: #0ea5e9; }
    .audit-tagline { color: #94a3b8; font-size: 1rem; margin-bottom: 1.5rem; }

    .verdict {
        border-radius: 12px; padding: 1rem 1.25rem; margin-bottom: 1rem;
        border: 1px solid; display: flex; align-items: center; gap: 1rem;
    }
    .verdict .label { font-size: 1.5rem; font-weight: 700; letter-spacing: -0.02em; }
    .verdict .why { color: #cbd5e1; font-size: 0.95rem; }
    .verdict.approved { background: rgba(34,197,94,0.10); border-color: rgba(34,197,94,0.4); }
    .verdict.approved .label { color: #4ade80; }
    .verdict.flagged  { background: rgba(245,158,11,0.10); border-color: rgba(245,158,11,0.4); }
    .verdict.flagged .label { color: #fbbf24; }
    .verdict.rejected { background: rgba(239,68,68,0.10); border-color: rgba(239,68,68,0.4); }
    .verdict.rejected .label { color: #f87171; }

    /* Metric cards: grid reflows 4 -> 2x2 -> 1 column; values wrap, never clip. */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(min(100%, 210px), 1fr));
        gap: 0.75rem;
        margin-bottom: 1.25rem;
    }
    .metric-card {
        background: rgba(30,41,59,0.6);
        border: 1px solid rgba(148,163,184,0.18);
        border-radius: 12px;
        padding: 0.875rem 1rem;
        min-width: 0;
    }
    .metric-card .m-label { color: #94a3b8; font-size: 0.8125rem; line-height: 1.2; margin-bottom: 0.25rem; }
    .metric-card .m-value {
        color: #e2e8f0; font-weight: 600; line-height: 1.25;
        font-size: clamp(1.05rem, 0.75rem + 1.1vw, 1.6rem);
        overflow-wrap: anywhere;
    }
    .metric-card .m-delta { font-size: 0.8125rem; margin-top: 0.25rem; overflow-wrap: anywhere; }
    .metric-card .m-delta.bad { color: #f87171; }
    .metric-card .m-delta.good { color: #4ade80; }
    .metric-card .m-delta.flat { color: #94a3b8; }

    div.stButton > button, div.stDownloadButton > button, div[data-testid="stFormSubmitButton"] > button {
        background: #0ea5e9 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: background 0.2s, transform 0.15s !important;
    }
    div.stButton > button:hover, div.stDownloadButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
        background: #0284c7 !important;
        transform: translateY(-1px);
    }

    .audit-footer {
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid rgba(148,163,184,0.15);
        text-align: center;
        font-size: 0.8125rem;
        color: #64748b;
    }
    .audit-footer a { color: #0ea5e9; text-decoration: none; font-weight: 600; }
    .audit-footer a:hover { text-decoration: underline; }
    </style>

    <div class="audit-badge">Vendor Invoice &amp; Compliance Reconciliation</div>
    <div class="audit-title">Audit every vendor invoice with <span>confidence</span></div>
    <p class="audit-tagline">Upload an invoice — the agent matches it to the PO and master contract, checks policy
    thresholds, and returns Approved, Flagged, or Rejected with explicit reasons.</p>
    """,
    unsafe_allow_html=True,
)

MIME_BY_EXT = {"pdf": "application/pdf", "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}
ENGINE_LABEL = {"gemini": "Gemini", "groq": "Groq (fallback)"}
SEVERITY_ICON = {"critical": "🔴", "warning": "🟠", "info": "🔵"}
STATUS_ICON = {"OK": "✅", "Flagged": "🟠", "Rejected": "🔴"}


def money(value, currency="USD"):
    return f"{currency} {value:,.2f}" if isinstance(value, (int, float)) else "—"


def metric_card(label, value, delta=None):
    """HTML card for the metrics row. `delta` is invoice minus PO: over = bad, under = good."""
    delta_html = ""
    if delta is not None:
        tone = "flat" if abs(delta) < 0.005 else ("bad" if delta > 0 else "good")
        delta_html = f'<div class="m-delta {tone}">Invoice {delta:+,.2f} vs PO</div>'
    return (f'<div class="metric-card"><div class="m-label">{html.escape(str(label))}</div>'
            f'<div class="m-value">{html.escape(str(value))}</div>{delta_html}</div>')


VERDICT_ORDER = ["Approved", "Flagged", "Rejected"]


def render_report(report):
    inv, po, contract = report["invoice"], report["purchase_order"], report["contract"]
    cur = inv.get("currency") or "USD"
    status = report["status"]
    findings = report["findings"]
    issues = [f for f in findings if f["severity"] != "info"]

    st.markdown(
        f'<div class="verdict {status.lower()}"><div class="label">{status}</div>'
        f'<div class="why">{html.escape(str(report["summary"]))}</div></div>',
        unsafe_allow_html=True,
    )

    po_delta = None
    if po and isinstance(inv.get("total"), (int, float)) and po.get("approved_total"):
        po_delta = inv["total"] - po["approved_total"]
    st.markdown(
        '<div class="metric-grid">'
        + metric_card("Invoice total", money(inv.get("total"), cur))
        + metric_card("PO approved total", money(po["approved_total"], cur) if po else "—",
                      delta=po_delta)
        + metric_card("Issues found", len(issues))
        + metric_card("Vendor", inv.get("vendor_name") or "—")
        + "</div>",
        unsafe_allow_html=True,
    )

    tab_find, tab_lines, tab_refs, tab_trace, tab_raw = st.tabs(
        ["Findings", "Line items", "Reference documents", "Agent trace", "Raw JSON"])

    with tab_find:
        if not findings:
            st.success("No discrepancies found.")
        for f in findings:
            where = " · AI review" if f["source"] == "llm" else ""
            st.markdown(f"{SEVERITY_ICON[f['severity']]} **{f['category']}**{where} — {f['message']}")
        if report["recommended_actions"]:
            st.subheader("Recommended actions")
            for action in report["recommended_actions"]:
                st.markdown(f"- {action}")

    with tab_lines:
        rows = report["line_results"]
        if rows:
            df = pd.DataFrame([{
                "": STATUS_ICON[r["status"]],
                "Line": r["line_no"],
                "SKU": r["sku"],
                "Description": r["description"],
                "Inv qty": r["inv_qty"],
                "PO qty": r["po_qty"],
                "Inv price": r["inv_unit_price"],
                "PO price": r["po_unit_price"],
                "UOM": r.get("uom"),
                "Contract price": r["contract_unit_price"],
                "Amount": r["amount"],
                "Notes": "; ".join(r["notes"]),
            } for r in rows])
            st.dataframe(df, hide_index=True, use_container_width=True, column_config={
                "Inv price": st.column_config.NumberColumn(format="%.2f"),
                "PO price": st.column_config.NumberColumn(format="%.2f"),
                "Contract price": st.column_config.NumberColumn(format="%.2f"),
                "Amount": st.column_config.NumberColumn(format="%.2f"),
            })
        else:
            st.warning("No line items could be extracted from this invoice.")
        st.caption(
            f"Invoice {inv.get('invoice_number') or '—'} · dated {inv.get('invoice_date') or '—'} · "
            f"PO {inv.get('po_number') or '—'} · terms {inv.get('payment_terms') or '—'} · "
            f"subtotal {money(inv.get('subtotal'), cur)} · tax {money(inv.get('tax'), cur)} · total {money(inv.get('total'), cur)}")

    with tab_refs:
        st.caption(f"Source: {report['reference_source']}")
        left, right = st.columns(2)
        with left:
            st.subheader("Purchase order")
            st.json(po) if po else st.warning("No matching purchase order.")
        with right:
            st.subheader("Master contract")
            st.json(contract) if contract else st.warning("No matching contract.")

    with tab_trace:
        for s in report["trace"]:
            icon = "✅" if s["status"] == "ok" else "⚠️"
            st.markdown(f"{icon} **{s['step']}** — {s['detail']}")
        engines = ", ".join(f"{k}: {ENGINE_LABEL.get(v, v)}" for k, v in report["engines"].items())
        st.caption(f"Engines — {engines} · audited {report['audited_at']}")

    with tab_raw:
        payload = json.dumps(report, indent=2)
        st.download_button("Download audit report (.json)", key=f"dl_{report['source_file']}", data=payload,
                           file_name=f"audit_{Path(str(report['source_file'])).stem}.json",
                           mime="application/json")
        st.code(payload, language="json")


def run_batch(files):
    """Audit each upload in filename order (so an original invoice is on record before its reprint)."""
    results = []
    bar = st.progress(0.0, text="Starting...")
    for i, f in enumerate(sorted(files, key=lambda f: f.name.lower()), 1):
        bar.progress((i - 1) / len(files), text=f"Auditing {f.name} ({i} of {len(files)})...")
        ext = f.name.rsplit(".", 1)[-1].lower()
        try:
            results.append({"file": f.name, "report": audit_invoice(f.getvalue(), MIME_BY_EXT[ext], filename=f.name), "error": None})
        except Exception as exc:
            results.append({"file": f.name, "report": None, "error": str(exc)})
    bar.empty()
    return results


with st.sidebar:
    st.header("Invoices")
    uploaded = st.file_uploader("Upload vendor invoice(s)", type=list(MIME_BY_EXT), accept_multiple_files=True,
                                help="One invoice or many at once. PDF, PNG or JPG. Scanned images are read by the vision model.")
    n_up = len(uploaded or [])
    run = st.button(f"Run audit ({n_up} invoice{'s' if n_up != 1 else ''})" if n_up else "Run audit",
                    use_container_width=True, disabled=n_up == 0)
    st.caption("Reference data: `data/`")
    with st.expander("Audit history"):
        st.caption("Past audits are kept in `output/` and used to catch duplicate invoices and cumulative "
                   "over-billing. Clear them to start a fresh test run.")
        confirm = st.checkbox("I want to delete all saved audits")
        if st.button("Clear audit history", disabled=not confirm, use_container_width=True):
            removed = 0
            for path in OUTPUT_DIR.glob("audit_*.json"):
                path.unlink()
                removed += 1
            st.session_state.pop("results", None)
            st.success(f"Deleted {removed} saved audit(s).")

if run and uploaded:
    st.session_state["results"] = run_batch(uploaded)

results = st.session_state.get("results")
if not results:
    st.info("Upload one or more invoices in the sidebar and click **Run audit** to begin.")
else:
    if len(results) > 1:
        counts = {v: sum(1 for r in results if r["report"] and r["report"]["status"] == v) for v in VERDICT_ORDER}
        errors = sum(1 for r in results if r["error"])
        st.markdown(
            '<div class="metric-grid">'
            + "".join(metric_card(v, c) for v, c in counts.items())
            + metric_card("Could not audit", errors)
            + "</div>", unsafe_allow_html=True)
        table = pd.DataFrame([{
            "": {"Approved": "✅", "Flagged": "🟠", "Rejected": "🔴"}.get(r["report"]["status"], "⚠️") if r["report"] else "⚠️",
            "File": r["file"],
            "Invoice #": (r["report"]["invoice"].get("invoice_number") if r["report"] else None),
            "Vendor": (r["report"]["invoice"].get("vendor_name") if r["report"] else None),
            "PO": (r["report"]["invoice"].get("po_number") if r["report"] else None),
            "Total": (r["report"]["invoice"].get("total") if r["report"] else None),
            "Verdict": r["report"]["status"] if r["report"] else "Error",
            "Issues": sum(1 for f in r["report"]["findings"] if f["severity"] != "info") if r["report"] else None,
            "Top issue": (next((f["message"] for f in r["report"]["findings"] if f["severity"] != "info"), "None")
                          if r["report"] else r["error"]),
        } for r in results])
        st.dataframe(table, hide_index=True, use_container_width=True,
                     column_config={"Total": st.column_config.NumberColumn(format="%.2f")})
        st.download_button("Download results (.csv)", data=table.drop(columns=[""]).to_csv(index=False),
                           file_name="invoice_audit_results.csv", mime="text/csv")
        st.divider()
        choice = st.selectbox("Inspect invoice", [r["file"] for r in results])
        selected = next(r for r in results if r["file"] == choice)
    else:
        selected = results[0]
    if selected["error"]:
        st.error(f"Audit failed for {selected['file']}: {selected['error']}")
    else:
        if len(results) > 1:
            st.subheader(selected["file"])
        render_report(selected["report"])

st.markdown(
    """
    <div class="audit-footer">
      Built by <a href="https://rishichellani.netlify.app" target="_blank">Rishi Chellani</a>
    </div>
    """,
    unsafe_allow_html=True,
)
