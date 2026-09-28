"""Deterministic invoice checks for Halvorsen policy FIN-AP-004.

`run_checks` cross-references an extracted invoice against its PO, contract,
vendor-master record, numeric thresholds and the history of invoices audited
before it (for duplicates and cumulative billing). It returns
(findings, line_results); the worst finding decides the verdict.
"""
import difflib
import re
from datetime import date, datetime

APPROVED, FLAGGED, REJECTED = "Approved", "Flagged", "Rejected"
INFO, WARNING, CRITICAL = "info", "warning", "critical"
SEVERITY_RANK = {INFO: 0, WARNING: 1, CRITICAL: 2}
STATUS_FOR = {INFO: APPROVED, WARNING: FLAGGED, CRITICAL: REJECTED}

REPRINT_LABEL_RE = re.compile(r"reprint|copy|duplicate|second notice|statement", re.I)
REPRINT_NOTE_RE = re.compile(r"reprint|second notice|duplicate invoice", re.I)
PAST_DUE_RE = re.compile(r"past due|overdue", re.I)
URGENCY_RE = re.compile(r"credit hold|late fee|\bimmediately\b|within \d+ days|due in \d+ days", re.I)
BANK_CHANGE_RE = re.compile(r"new remittance|new bank|changed? (of |in )?(our )?bank|banking partner|update your records|discard (all )?previous|remittance (details|instructions) (have )?changed", re.I)
WIRE_RE = re.compile(r"\bwire\b", re.I)
EMAIL_ONLY_RE = re.compile(r"e-?mail only", re.I)
ACCOUNT_NO_RE = re.compile(r"routing\W{0,3}[\d-]{6,}|\baccount\W{0,3}(no\.?|number|#)?\W{0,3}\d[\d-]{5,}", re.I)
FEE_RE = re.compile(r"\badmin(istrative)?\b|\bhandling\b|restocking|\brush\b|credit[- ]card|late fee|fuel surcharge|\binterest\b", re.I)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def num(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def norm(value):
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def pct(actual, expected):
    return (actual - expected) / expected * 100 if expected else 0.0


def parse_date(value):
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _domain(email):
    return str(email or "").rsplit("@", 1)[-1].strip().lower() if "@" in str(email or "") else ""


def _digits(value):
    return re.sub(r"\D", "", str(value or ""))[-10:]


def _net_days(terms):
    t = (terms or "").lower()
    if "receipt" in t:
        return 0
    m = re.search(r"net\s*(\d+)", t) or re.search(r"(\d+)\s*days", t)
    return int(m.group(1)) if m else None


def _match_po_line(inv_line, po_lines, used):
    """Match by SKU, else by fuzzy description. Returns the PO line or None."""
    sku = norm(inv_line.get("sku"))
    if sku:
        for line in po_lines:
            if id(line) not in used and norm(line.get("sku")) == sku:
                return line
    desc = str(inv_line.get("description") or "").lower()
    best, best_score = None, 0.6
    for line in po_lines:
        if id(line) in used:
            continue
        score = difflib.SequenceMatcher(None, desc, str(line.get("description") or "").lower()).ratio()
        if score > best_score:
            best, best_score = line, score
    return best


def _rate_for(contract, sku, on_date):
    """Contract rate for a SKU on a date (rate cards may carry effective periods)."""
    entries = [r for r in (contract or {}).get("rate_card", []) if norm(r.get("sku")) == norm(sku)]
    if not entries:
        return None
    for r in entries:
        start, end = parse_date(r.get("effective_from")), parse_date(r.get("effective_to"))
        if (not start or not on_date or on_date >= start) and (not end or not on_date or on_date <= end):
            return num(r["unit_price"])
    return None


def _same_period(a, b):
    """Two invoices for different service periods are recurring billing, not duplicates."""
    pa, pb = parse_date(a.get("service_period_start")), parse_date(b.get("service_period_start"))
    return not (pa and pb and pa != pb)


def _line_signature(invoice):
    return sorted((norm(l.get("sku")), num(l.get("quantity"))) for l in invoice.get("line_items") or [])


# --------------------------------------------------------------------------
# main entry point
# --------------------------------------------------------------------------
def run_checks(invoice, po, contract, thresholds, vendor=None, history=None, today=None):
    """`history`: prior audits, each {invoice, status, po_number, is_duplicate, source_file}."""
    findings, rows = [], []
    t = thresholds
    tol = t["math_tolerance_abs"]
    today = today or date.today()
    history = history or []
    rules = (contract or {}).get("rules") or {}

    def add(severity, category, message, line_no=None):
        findings.append({"severity": severity, "category": category, "message": message,
                         "line_no": line_no, "source": "rules"})

    inv_lines = invoice.get("line_items") or []
    inv_date = parse_date(invoice.get("invoice_date"))
    ship_date = parse_date(invoice.get("ship_date"))
    period_start, period_end = parse_date(invoice.get("service_period_start")), parse_date(invoice.get("service_period_end"))
    service_date = period_start or ship_date or inv_date
    freight = num(invoice.get("freight", invoice.get("shipping")))
    tax = num(invoice.get("tax"))
    total = invoice.get("total")
    text = " ".join(str(invoice.get(k) or "") for k in ("remittance_text", "notes", "payment_terms")
                    ) + " " + " ".join(map(str, invoice.get("document_labels") or []))

    # -- history: prior audits of other invoices
    po_number = (po or {}).get("po_number")
    inv_vendor = norm(invoice.get("vendor_name"))
    peers = [h for h in history if not h.get("is_duplicate")]
    prior_on_po = [h for h in peers if po_number and h.get("po_number") == po_number and h.get("status") != REJECTED]
    prior_total = sum(num(h["invoice"].get("total")) for h in prior_on_po)
    prior_qty = {}
    for h in prior_on_po:
        for l in h["invoice"].get("line_items") or []:
            prior_qty[norm(l.get("sku"))] = prior_qty.get(norm(l.get("sku")), 0.0) + num(l.get("quantity"))

    # -- 1. submission requirements (policy s3)
    if not inv_lines:
        add(WARNING, "Extraction", "No line items could be read from the invoice; review it manually.")
    if not invoice.get("invoice_number"):
        add(CRITICAL, "Invoice requirements", "Invoice number is missing; the invoice is returned to the supplier.")
    if not inv_date:
        add(CRITICAL, "Invoice requirements", "Invoice date is missing or unreadable; the invoice is returned to the supplier.")
    if invoice.get("bill_to") and "halvorsen" not in str(invoice["bill_to"]).lower():
        add(CRITICAL, "Invoice requirements", f"Invoice is not addressed to Halvorsen Marine Systems, Inc. (bill-to: '{invoice['bill_to']}').")
    if inv_date and inv_date > today:
        add(CRITICAL, "Invoice requirements", f"Invoice is dated in the future ({inv_date}).")
    if po and inv_date and parse_date(po.get("issued_date")) and inv_date < parse_date(po["issued_date"]):
        add(CRITICAL, "Invoice requirements", f"Invoice dated {inv_date} is before the PO issue date {po['issued_date']}.")
    if rules.get("billing_type") == "goods":
        if not invoice.get("packing_slip"):
            add(WARNING, "Invoice requirements", "Packing slip number is missing (required for goods).")
        if not ship_date:
            add(WARNING, "Invoice requirements", "Ship date is missing (required for goods).")
    elif contract and not (period_start or invoice.get("service_period")):
        add(WARNING, "Invoice requirements", "Service period covered is not stated (required for services).")
    end_of_service = period_end or ship_date
    if inv_date and end_of_service and (inv_date - end_of_service).days > t["late_submission_days"]:
        add(WARNING, "Late submission", f"Invoice was issued {(inv_date - end_of_service).days} days after the ship/service date (limit {t['late_submission_days']}); Controller review required.")

    # -- 2. PO, vendor identity and vendor master
    if po is None:
        add(CRITICAL, "PO match", f"No valid purchase order found for PO number '{invoice.get('po_number')}'. "
            "No PO, no pay: verbal or email approvals are not purchase orders.")
    else:
        if po.get("status", "").lower() not in ("open", "partially received"):
            add(CRITICAL, "PO match", f"PO {po['po_number']} is not open (status: {po.get('status')}).")
        if invoice.get("currency") and po.get("currency") and invoice["currency"] != po["currency"]:
            add(CRITICAL, "Currency", f"Invoice currency {invoice['currency']} differs from PO currency {po['currency']}.")
        if invoice.get("contract_id") and norm(invoice["contract_id"]) != norm(po.get("contract_id")):
            add(WARNING, "PO match", f"Invoice cites contract {invoice['contract_id']} but the PO is under {po.get('contract_id')}.")
    expected_name = (po or {}).get("vendor_name") or (vendor or {}).get("legal_name")
    if expected_name and norm(expected_name) != inv_vendor:
        add(CRITICAL, "Vendor match", f"Invoice vendor '{invoice.get('vendor_name')}' does not match the vendor of record '{expected_name}'.")
    if vendor is None and po is None:
        add(CRITICAL, "Vendor master", f"Vendor '{invoice.get('vendor_name')}' is not in the vendor master; suppliers must be onboarded through Procurement before payment.")
    record = {**(po or {}).get("vendor_contact", {}), **{k: v for k, v in (vendor or {}).items() if k in ("address", "email", "phone")}}
    if record:
        addr = invoice.get("vendor_address")
        if addr and record.get("address") and difflib.SequenceMatcher(None, norm(addr), norm(record["address"])).ratio() < 0.85:
            add(CRITICAL, "Vendor master", f"Supplier address on the invoice ('{addr}') differs from the vendor master ('{record['address']}'). Fraud red flag.")
        if _domain(invoice.get("vendor_email")) and record.get("email") and _domain(invoice["vendor_email"]) != _domain(record["email"]):
            add(CRITICAL, "Vendor master", f"Supplier email domain '{_domain(invoice['vendor_email'])}' differs from the vendor master '{_domain(record['email'])}'. Fraud red flag.")
        if _digits(invoice.get("vendor_phone")) and record.get("phone") and _digits(invoice["vendor_phone"]) != _digits(record["phone"]):
            add(CRITICAL, "Vendor master", f"Supplier phone '{invoice['vendor_phone']}' differs from the vendor master '{record['phone']}'. Fraud red flag.")

    # -- 3. fraud red flags in remittance / urgency language (policy s9)
    if BANK_CHANGE_RE.search(text):
        add(CRITICAL, "Fraud red flag", "Invoice announces new or changed remittance/bank details. Bank changes need a call-back to the contact of record and dual approval; hold and report to the Controller.")
    if WIRE_RE.search(text):
        add(CRITICAL, "Fraud red flag", "Invoice demands payment by wire; remittance is only to the account in the vendor master.")
    if ACCOUNT_NO_RE.search(text):
        add(CRITICAL, "Fraud red flag", "Invoice prints bank routing/account numbers; remittance instructions on an invoice are never used.")
    if EMAIL_ONLY_RE.search(text):
        add(CRITICAL, "Fraud red flag", "Invoice asks for confirmation 'by email only', bypassing the call-back control.")
    if URGENCY_RE.search(text):
        add(CRITICAL, "Fraud red flag", "Invoice applies unusual urgency (short deadline, late-fee or credit-hold threat).")
    due = parse_date(invoice.get("due_date"))
    if PAST_DUE_RE.search(text) and due and due > today:
        add(CRITICAL, "Fraud red flag", f"Invoice claims to be past due, but its due date ({due}) has not been reached.")

    # -- 4. duplicates and reprints (policy s8)
    dup_reasons = []
    for h in peers:
        hi = h["invoice"]
        same_vendor = norm(hi.get("vendor_name")) == inv_vendor
        same_po = bool(po_number) and h.get("po_number") == po_number
        label = f"invoice {hi.get('invoice_number')} ({h.get('source_file')})"
        if same_vendor and norm(hi.get("invoice_number")) == norm(invoice.get("invoice_number")) and invoice.get("invoice_number"):
            dup_reasons.append(f"same invoice number as {label}")
        elif total is not None and _same_period(invoice, hi):
            same_total = abs(num(hi.get("total")) - num(total)) < 0.005
            hd = parse_date(hi.get("invoice_date"))
            if same_po and same_total:
                dup_reasons.append(f"same PO and amount as {label}")
            elif same_vendor and same_total and inv_date and hd and abs((inv_date - hd).days) <= t["duplicate_amount_window_days"]:
                dup_reasons.append(f"same amount within {t['duplicate_amount_window_days']} days of {label}")
            elif same_po and inv_lines and _line_signature(invoice) == _line_signature(hi):
                dup_reasons.append(f"same line items and quantities already billed against the PO on {label}")
    for reason in dup_reasons:
        add(CRITICAL, "Duplicate", f"Suspected duplicate: {reason}.")
    if dup_reasons:  # already rejected as a duplicate: don't also double-count it against the original's billing
        prior_on_po, prior_total, prior_qty = [], 0.0, {}
    labels = " ".join(map(str, invoice.get("document_labels") or []))
    if REPRINT_LABEL_RE.search(labels) or REPRINT_NOTE_RE.search(str(invoice.get("notes") or "")):
        if dup_reasons:
            add(CRITICAL, "Duplicate", "Document is marked reprint/second notice; it is not a payable document and the original is already recorded.")
        else:
            add(WARNING, "Duplicate", "Document is marked reprint/copy/second notice; it is not a payable document. Research against the original before any action.")

    # -- 5. contract validity and payment terms (policy s7)
    if contract is None:
        add(WARNING, "Contract", "No master contract found for this vendor/PO; contract terms could not be verified.")
    else:
        start, end = parse_date(contract.get("effective_date")), parse_date(contract.get("expiry_date"))
        for label, when in (("invoice", inv_date), ("goods/services", service_date)):
            if when and end and when > end:
                add(WARNING, "Contract validity", f"The {label} date {when} is after contract {contract['contract_id']} expired on {end}.")
            if when and start and when < start:
                add(WARNING, "Contract validity", f"The {label} date {when} is before contract {contract['contract_id']} took effect on {start}.")
        if invoice.get("payment_terms") and norm(invoice["payment_terms"]) != norm(contract.get("payment_terms")):
            inv_net, con_net = _net_days(invoice["payment_terms"]), _net_days(contract.get("payment_terms"))
            shorter = " (shorter than the contract; AP pays on contract terms)" if inv_net is not None and con_net is not None and inv_net < con_net else ""
            add(WARNING, "Payment terms", f"Invoice terms '{invoice['payment_terms']}' differ from contract terms '{contract.get('payment_terms')}'{shorter}.")

    # -- 6. freight (policy s6)
    subtotal = invoice.get("subtotal")
    line_sum = sum(num(l.get("amount")) for l in inv_lines)
    base = num(subtotal, line_sum)
    if freight > tol:
        fr, allowance = rules.get("freight"), num((po or {}).get("freight_allowance"))
        limit, reason = None, ""
        if fr:
            if base >= fr["included_at_or_above"]:
                limit, reason = 0.0, f"freight is included in unit prices on orders of {fr['included_at_or_above']:,.2f} or more"
            else:
                limit, reason = fr["max_per_shipment_below"], f"the contract caps freight at {fr['max_per_shipment_below']:,.2f} per shipment"
        if allowance > 0 and (limit is None or allowance < limit):
            limit, reason = allowance, f"the PO freight allowance is {allowance:,.2f}"
        if limit is None:
            limit, reason = 0.0, "neither the contract nor the PO provides for separately billed freight"
        if freight > limit + tol:
            add(WARNING, "Freight", f"Freight of {freight:,.2f} is not payable: {reason}.")

    # -- 7. sales tax (policy s6)
    tax_rule = rules.get("tax")
    if tax_rule == "non_taxable" and tax > tol:
        add(WARNING, "Sales tax", f"Sales tax of {tax:,.2f} was charged, but these services/subscriptions are not taxable under the contract.")
    elif tax_rule == "taxable":
        rate = num((po or {}).get("sales_tax_rate"), t["sales_tax_rate"]) or t["sales_tax_rate"]
        expected_tax = round(base * rate, 2)
        if abs(tax - expected_tax) > tol:
            add(WARNING, "Sales tax", f"Sales tax is {tax:,.2f}; {rate * 100:.2f}% of the {base:,.2f} taxable subtotal is {expected_tax:,.2f}.")

    # -- 8. line-by-line matching (policy s5)
    po_lines = (po or {}).get("line_items", [])
    exp_cfg = rules.get("expenses")
    exp_prefixes = [norm(p) for p in (exp_cfg or {}).get("sku_prefixes", [])]
    milestones = {norm(m["sku"]): m for m in rules.get("milestones", [])}
    used, expense_lines, fees_total = set(), [], 0.0
    for i, line in enumerate(inv_lines, 1):
        n = line.get("line_no") or i
        sku = norm(line.get("sku"))
        qty, price, amount = num(line.get("quantity")), num(line.get("unit_price")), num(line.get("amount"))
        c_price = _rate_for(contract, line.get("sku"), service_date)
        row = {"line_no": n, "sku": line.get("sku"), "description": line.get("description"), "uom": line.get("uom"),
               "inv_qty": qty, "po_qty": None, "inv_unit_price": price, "po_unit_price": None,
               "contract_unit_price": c_price, "amount": amount, "status": "OK", "notes": []}
        rows.append(row)
        row_sev = INFO

        def flag(severity, category, message):
            nonlocal row_sev
            add(severity, category, f"Line {n}: {message}", n)
            row["notes"].append(message)
            if SEVERITY_RANK[severity] > SEVERITY_RANK[row_sev]:
                row_sev = severity

        if abs(qty * price - amount) > tol:
            flag(WARNING, "Arithmetic", f"quantity × unit price = {qty * price:.2f}, but the line amount is {amount:.2f}.")

        is_expense = bool(exp_prefixes) and any(sku.startswith(p) for p in exp_prefixes)
        is_fee = bool(FEE_RE.search(str(line.get("description") or "")))
        if is_fee:
            flag(WARNING, "Non-payable charge", f"'{line.get('description')}' is an administrative/handling-type fee, not payable unless the contract provides for it.")
        if is_expense or is_fee:
            (expense_lines if is_expense else []).append(line)
            row["status"] = {INFO: "OK", WARNING: "Flagged", CRITICAL: "Rejected"}[row_sev]
            continue
        fees_total += amount

        milestone = milestones.get(sku)
        on_milestone = bool(milestone) and any(abs(price - p) < 0.005 for p in milestone["payments"])
        if on_milestone:
            billed = sum(num(l.get("amount")) for h in prior_on_po for l in h["invoice"].get("line_items") or [] if norm(l.get("sku")) == sku)
            if billed + amount > milestone["total"] + tol:
                flag(WARNING, "Milestone billing", f"milestone billing would total {billed + amount:,.2f}, over the {milestone['total']:,.2f} fixed fee.")
            else:
                row["notes"].append("Scheduled milestone payment of the fixed fee (not a price variance).")

        if po is not None:
            po_line = _match_po_line(line, po_lines, used)
            if po_line is None:
                extra = " Work performed without a work order is not billable." if rules.get("work_order_required_for_extras") else ""
                flag(WARNING, "PO match", f"item is not on the purchase order; a PO change order is required before payment.{extra}")
            else:
                used.add(id(po_line))
                po_qty, po_price = num(po_line.get("quantity")), num(po_line.get("unit_price"))
                row["po_qty"], row["po_unit_price"] = po_qty, po_price
                if not on_milestone:
                    p_var = pct(price, po_price)
                    if p_var > t["price_variance_reject_pct"]:
                        flag(CRITICAL, "Price variance", f"unit price {price:.2f} is {p_var:.1f}% above PO price {po_price:.2f}.")
                    elif abs(p_var) > t["price_variance_flag_pct"]:
                        flag(WARNING, "Price variance", f"unit price {price:.2f} differs {p_var:+.1f}% from PO price {po_price:.2f}.")
                cum_qty = qty + (0.0 if milestone else prior_qty.get(sku, 0.0))
                q_var = pct(cum_qty, po_qty)
                cum_note = f" (incl. {cum_qty - qty:g} already billed)" if cum_qty != qty else ""
                if q_var > t["quantity_overage_reject_pct"]:
                    flag(CRITICAL, "Quantity variance", f"billed {cum_qty:g} vs {po_qty:g} ordered ({q_var:+.1f}%){cum_note}.")
                elif q_var > t["quantity_overage_flag_pct"]:
                    flag(WARNING, "Quantity variance", f"billed {cum_qty:g} vs {po_qty:g} ordered ({q_var:+.1f}%){cum_note}.")

        if (c_price is not None and not on_milestone and abs(pct(price, c_price)) > t["price_variance_flag_pct"]
                and not any("PO price" in note for note in row["notes"])):
            flag(WARNING, "Contract rate", f"unit price {price:.2f} differs from contract rate {c_price:.2f}.")

        row["status"] = {INFO: "OK", WARNING: "Flagged", CRITICAL: "Rejected"}[row_sev]

    for line in po_lines:
        if id(line) not in used and inv_lines and not (po or {}).get("blanket"):
            add(INFO, "PO match", f"PO line {line.get('line_no')} ({line.get('description')}) was not invoiced.")

    # -- 9. contractor expenses and time & materials (policy s5, s10)
    if exp_cfg:
        cfg = {**t.get("expenses", {}), **exp_cfg}
        details = invoice.get("expense_details") or []
        items = [{"description": d.get("description"), "amount": num(d.get("amount")), "unit": None} for d in details] or \
                [{"description": l.get("description"), "amount": num(l.get("amount")), "unit": (l.get("uom"), num(l.get("unit_price")))} for l in expense_lines]
        expense_total = sum(i["amount"] for i in items)
        for it in items:
            desc = str(it["description"] or "")
            m = re.search(r"@\s*\$?\s*([\d,]+(?:\.\d+)?)", desc)
            rate = num(m.group(1).replace(",", "")) if m else (it["unit"][1] if it["unit"] and str(it["unit"][0]).upper() in ("NT", "DAY", "NIGHT") else None)
            if re.search(r"first class|business class", desc, re.I):
                add(WARNING, "Expenses", f"Non-compliant airfare ('{desc}'): the contract allows economy class only. Short-pay {it['amount']:,.2f}.")
            if re.search(r"hotel|lodging", desc, re.I) and rate and rate > cfg["lodging_per_night"] + 0.005:
                add(WARNING, "Expenses", f"Lodging at {rate:,.2f}/night exceeds the {cfg['lodging_per_night']:,.2f} limit.")
            if re.search(r"meal|per diem", desc, re.I) and rate and rate > cfg["meals_per_day"] + 0.005:
                add(WARNING, "Expenses", f"Meals per diem of {rate:,.2f}/day exceeds the {cfg['meals_per_day']:,.2f} limit.")
        if items and any(i["amount"] >= cfg["receipt_threshold"] for i in items):
            add(INFO, "Expenses", f"Itemized receipts are required for every expense of {cfg['receipt_threshold']:,.2f} or more; confirm they are attached.")
        if expense_total > 0 and fees_total > 0 and expense_total / fees_total * 100 > cfg["max_pct_of_fees"]:
            add(WARNING, "Expenses", f"Expenses of {expense_total:,.2f} are {expense_total / fees_total * 100:.1f}% of the {fees_total:,.2f} fees billed (limit {cfg['max_pct_of_fees']:g}%); prior Controller approval is required.")
    hours = sum(num(l.get("quantity")) for l in inv_lines if str(l.get("uom") or "").upper() in ("HR", "HOUR", "HOURS"))
    if invoice.get("timesheet_hours") is not None and hours and abs(hours - num(invoice["timesheet_hours"])) > 0.05:
        add(WARNING, "Timesheet", f"Invoice lines bill {hours:g} hours but the timesheet summary states {num(invoice['timesheet_hours']):g}.")

    # -- 10. arithmetic and totals (policy s5)
    if subtotal is not None and abs(line_sum - num(subtotal)) > tol:
        add(WARNING, "Arithmetic", f"Line amounts sum to {line_sum:.2f} but the invoice subtotal is {num(subtotal):.2f}.")
    if total is not None:
        expected = base + freight + tax
        if abs(expected - num(total)) > tol:
            add(WARNING, "Arithmetic", f"Subtotal + freight + tax = {expected:.2f} but the invoice total is {num(total):.2f}.")
    else:
        add(WARNING, "Extraction", "Invoice total could not be read.")

    # -- 11. cumulative billing vs PO total and not-to-exceed
    if total is not None:
        cumulative = prior_total + num(total)
        if po and po.get("approved_total"):
            var = pct(cumulative, num(po["approved_total"]))
            scope = "Cumulative billing against the PO" if prior_total else "Invoice total"
            if var > t["total_variance_reject_pct"]:
                add(CRITICAL, "Total variance", f"{scope} ({cumulative:,.2f}) exceeds the PO approved total ({num(po['approved_total']):,.2f}) by {var:.1f}%.")
            elif var > t["total_variance_flag_pct"]:
                add(WARNING, "Total variance", f"{scope} ({cumulative:,.2f}) exceeds the PO approved total ({num(po['approved_total']):,.2f}) by {var:.1f}%.")
        nte = num((contract or {}).get("not_to_exceed"))
        if nte and cumulative > nte + tol:
            add(CRITICAL, "Not-to-exceed", f"Cumulative billing of {cumulative:,.2f} exceeds the contract not-to-exceed of {nte:,.2f}.")
        stated = invoice.get("cumulative_billed_to_date")
        if stated is not None and abs(num(stated) - cumulative) > tol:
            add(WARNING, "Cumulative billing", f"Invoice states {num(stated):,.2f} billed to date, but audited invoices on this PO total {cumulative:,.2f}.")

    # -- 12. approval authority (policy s11)
    if total is not None:
        for tier in t["approval_tiers"]:
            if tier["below"] is None or num(total) < tier["below"]:
                if tier.get("manual"):
                    add(WARNING, "Approval limit", f"Invoice total {num(total):,.2f} requires manual approval by the {tier['approver']} even if all checks pass.")
                else:
                    add(INFO, "Approval", f"Approval authority: {tier['approver']}.")
                break

    return findings, rows
