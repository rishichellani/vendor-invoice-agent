"""Reference-document access for the audit agent (the Memory / RAG layer).

The agent only talks to the `ReferenceStore` interface, so the backing source
can change without touching agent.py:

  * LocalReferenceStore -- reads the reference records under data/.

Another backing source can be added by implementing the same interface.
"""
import json
from pathlib import Path
from typing import Optional, Protocol

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class ReferenceStore(Protocol):
    source: str

    def get_purchase_order(self, po_number: str) -> Optional[dict]: ...
    def get_contract(self, contract_id: str) -> Optional[dict]: ...
    def find_contract_for_vendor(self, vendor_name: str) -> Optional[dict]: ...
    def find_vendor(self, vendor_no: Optional[str], vendor_name: Optional[str]) -> Optional[dict]: ...
    def get_policies(self) -> dict:
        """{"text": <policy markdown>, "thresholds": <dict of numeric limits>}"""
        ...


def _norm(value: str) -> str:
    return "".join(ch for ch in (value or "").lower() if ch.isalnum())


class LocalReferenceStore:
    source = "local (data/)"

    def __init__(self, root: Path = DATA_DIR):
        self.root = root

    def _load(self, folder: str, stem: str) -> Optional[dict]:
        path = self.root / folder / f"{stem}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    def get_purchase_order(self, po_number):
        if not po_number:
            return None
        # Tolerate formatting differences such as "po 10482" vs "PO-10482".
        for path in (self.root / "purchase_orders").glob("*.json"):
            if _norm(path.stem) == _norm(po_number):
                return json.loads(path.read_text(encoding="utf-8"))
        return None

    def get_contract(self, contract_id):
        return self._load("contracts", contract_id) if contract_id else None

    def find_contract_for_vendor(self, vendor_name):
        for path in (self.root / "contracts").glob("*.json"):
            contract = json.loads(path.read_text(encoding="utf-8"))
            if _norm(contract.get("vendor_name")) == _norm(vendor_name):
                return contract
        return None

    def find_vendor(self, vendor_no, vendor_name):
        """Vendor-master record by vendor number, else by exact (normalized) legal name."""
        path = self.root / "vendors.json"
        vendors = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
        for v in vendors:
            if vendor_no and v.get("vendor_no") == vendor_no:
                return v
        for v in vendors:
            if _norm(v.get("legal_name")) == _norm(vendor_name):
                return v
        return None

    def get_policies(self):
        folder = self.root / "policies"
        text = "\n\n".join(p.read_text(encoding="utf-8").strip() for p in sorted(folder.glob("*.md")))
        thresholds = json.loads((folder / "thresholds.json").read_text(encoding="utf-8"))
        return {"text": text, "thresholds": thresholds}


def get_reference_store() -> ReferenceStore:
    return LocalReferenceStore()
