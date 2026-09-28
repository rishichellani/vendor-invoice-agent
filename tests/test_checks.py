"""Regression tests for the deterministic policy rules (no LLM, no network).

Fixtures are the extracted-invoice JSON for the 13 sample invoices in sample_invoices/.
They are run in filename order with the same history rules as agent.audit_invoice, so
duplicate detection and cumulative billing are exercised.

    python -m unittest discover -s tests -v
"""
import json
import sys
import unittest
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from checks import STATUS_FOR, SEVERITY_RANK, parse_date, run_checks  # noqa: E402
from services.reference_store import LocalReferenceStore  # noqa: E402

TODAY = date(2026, 9, 28)
EXPECTED = {  # verdict, and categories that must appear among the non-info findings
    "INV-01": ("Approved", set()),
    "INV-02": ("Rejected", {"Duplicate", "Fraud red flag"}),
    "INV-03": ("Flagged", {"Freight", "Price variance", "Payment terms", "Total variance"}),
    "INV-04": ("Rejected", {"Quantity variance", "Total variance", "PO match"}),
    "INV-05": ("Flagged", {"Arithmetic"}),
    "INV-06": ("Rejected", {"Vendor master", "Fraud red flag", "Duplicate"}),
    "INV-07": ("Approved", set()),
    "INV-08": ("Flagged", {"Price variance", "Expenses", "Non-payable charge"}),
    "INV-09": ("Approved", set()),
    "INV-10": ("Flagged", {"Sales tax", "Price variance", "PO match", "Payment terms"}),
    "INV-11": ("Flagged", {"Approval limit"}),
    "INV-12": ("Rejected", {"PO match", "Price variance", "Contract validity"}),
    "INV-13": ("Rejected", {"PO match", "Vendor master", "Fraud red flag"}),
}


class PolicyRules(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        store = LocalReferenceStore()
        thresholds = store.get_policies()["thresholds"]
        fixtures = json.loads((ROOT / "tests/fixtures/invoices.json").read_text())
        cls.results, history = {}, []
        for name in sorted(fixtures):
            inv = fixtures[name]
            po = store.get_purchase_order(inv.get("po_number"))
            contract = (store.get_contract(inv.get("contract_id") or (po or {}).get("contract_id"))
                        or store.find_contract_for_vendor(inv.get("vendor_name")))
            vendor = store.find_vendor(po.get("vendor_no") if po else None, inv.get("vendor_name"))
            d = parse_date(inv.get("invoice_date"))
            peers = [h for h in history if not (d and parse_date(h["invoice"].get("invoice_date")) and parse_date(h["invoice"]["invoice_date"]) > d)]
            findings, _ = run_checks(inv, po, contract, thresholds, vendor=vendor, history=peers, today=TODAY)
            worst = max((f["severity"] for f in findings), key=SEVERITY_RANK.get)
            status = STATUS_FOR[worst]
            cls.results[name[:6]] = (status, findings)
            history.append({"invoice": inv, "status": status, "source_file": name, "po_number": (po or {}).get("po_number"),
                            "is_duplicate": any(f["category"] == "Duplicate" for f in findings)})

    def test_verdicts_and_categories(self):
        for key, (verdict, categories) in EXPECTED.items():
            with self.subTest(invoice=key):
                status, findings = self.results[key]
                self.assertEqual(status, verdict, [f"{f['severity']}:{f['category']}" for f in findings if f["severity"] != "info"])
                got = {f["category"] for f in findings if f["severity"] != "info"}
                self.assertLessEqual(categories, got)

    def test_clean_invoices_have_no_issues(self):
        for key in ("INV-01", "INV-07", "INV-09"):
            with self.subTest(invoice=key):
                self.assertEqual([f for f in self.results[key][1] if f["severity"] != "info"], [])

    def test_duplicate_does_not_inflate_quantities(self):
        cats = {f["category"] for f in self.results["INV-02"][1]}
        self.assertNotIn("Quantity variance", cats)
        self.assertNotIn("Total variance", cats)

    def test_milestone_is_not_a_price_variance(self):
        self.assertNotIn("Price variance", {f["category"] for f in self.results["INV-11"][1]})

    def test_escalator_rate_selected_by_service_date(self):
        # Aug 2026 falls in contract year 3 (8,964.61); the 5% "annual adjustment" is 9,138.68.
        msgs = [f["message"] for f in self.results["INV-10"][1] if f["category"] == "Price variance"]
        self.assertTrue(any("9138.68" in m and "8964.61" in m for m in msgs), msgs)


if __name__ == "__main__":
    unittest.main()
