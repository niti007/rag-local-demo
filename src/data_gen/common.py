"""
Shared helpers for the Cascade Bank synthetic data generators.

****************************************************************************
*  ALL DATA PRODUCED BY THIS PACKAGE IS SYNTHETIC / FICTIONAL TEST DATA.   *
*  "Cascade Bank" is a made-up company invented for a local RAG teaching  *
*  demo. Every person, account, transaction, figure, and document below   *
*  is fabricated by Faker / templated text generators and does NOT       *
*  describe any real institution, employee, or customer.                  *
****************************************************************************

This module centralizes:
- seeding (python `random` + Faker) for reproducible corpora
- the standard disclaimer text used on every generated document
- a small "banking flavored" paragraph generator (Faker sentences woven
  into banking-domain templates, so text reads as topical rather than
  pure lorem ipsum)
- a base fpdf2 PDF class with a title page + footer disclaimer
"""

from __future__ import annotations

import random
from pathlib import Path

from faker import Faker
from fpdf import FPDF

# ---------------------------------------------------------------------------
# Disclaimers
# ---------------------------------------------------------------------------

DISCLAIMER = "SYNTHETIC / FICTIONAL DEMO DATA — NOT A REAL BANK"

DISCLAIMER_LONG = (
    "This document was generated entirely by an automated synthetic-data "
    "generator for a local RAG (retrieval-augmented generation) teaching "
    "demo. Cascade Bank is a fictional company; all names, accounts, "
    "figures, dates, and events are fabricated (via the Faker library) and "
    "do not describe any real institution, employee, or customer."
)

FOOTER_DISCLAIMER = "Cascade Bank (fictional) — synthetic demo data, not a real institution"


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

_faker_instance: Faker | None = None


def seed_all(seed: int) -> Faker:
    """Seed python's `random` module and Faker for reproducible output.

    Returns a Faker instance to use for the rest of a generation run.
    """
    global _faker_instance
    random.seed(seed)
    Faker.seed(seed)
    fake = Faker("en_US")
    _faker_instance = fake
    return fake


def get_faker(seed: int | None = None) -> Faker:
    """Return a process-wide Faker instance, seeding it on first use."""
    global _faker_instance
    if _faker_instance is None:
        _faker_instance = seed_all(seed if seed is not None else 42)
    return _faker_instance


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def slugify(text: str) -> str:
    keep = "".join(c if c.isalnum() else " " for c in text.lower())
    return "_".join(keep.split())


_UNICODE_TO_ASCII = {
    "—": "-",    # em dash —
    "–": "-",    # en dash –
    "‘": "'",    # left curly single quote '
    "’": "'",    # right curly single quote '
    "‚": "'",    # single low-9 quote
    "‛": "'",    # single high-reversed-9 quote
    "“": '"',    # left curly double quote "
    "”": '"',    # right curly double quote "
    "„": '"',    # double low-9 quote
    "…": "...",  # ellipsis …
    " ": " ",    # non-breaking space
    "•": "-",    # bullet •
    "‐": "-",    # hyphen
    "‑": "-",    # non-breaking hyphen
    "‒": "-",    # figure dash
    "―": "-",    # horizontal bar
    "×": "x",    # multiplication sign
    "−": "-",    # minus sign
}


def latin1_safe(text) -> str:
    """fpdf2's built-in core fonts (helvetica) only support latin-1.

    Faker (including non-en_US locale artifacts) and hand-written templates
    can produce unicode punctuation (curly quotes, em/en dashes, ellipses,
    bullets, non-breaking spaces, accented names, etc.) that fpdf2's core
    fonts cannot encode, raising ``FPDFUnicodeEncodingException``.

    This first maps common unicode punctuation to visually-equivalent ASCII,
    then strips/replaces anything still outside latin-1 so this function can
    NEVER raise or crash fpdf2, regardless of input.
    """
    if not isinstance(text, str):
        text = str(text)
    for unicode_char, ascii_equivalent in _UNICODE_TO_ASCII.items():
        if unicode_char in text:
            text = text.replace(unicode_char, ascii_equivalent)
    return text.encode("latin-1", "replace").decode("latin-1")


# ---------------------------------------------------------------------------
# Banking-flavored text generation
# ---------------------------------------------------------------------------

BANKING_VOCAB = [
    "overdraft protection", "wire transfer authorization", "KYC verification",
    "AML transaction screening", "ACH settlement", "routing number validation",
    "escrow account management", "loan origination", "credit underwriting",
    "debit card issuance", "mobile check deposit", "two-factor authentication",
    "FDIC insurance coverage", "interest accrual", "minimum balance requirements",
    "line of credit", "collateral valuation", "internal compliance audit",
    "fraud detection", "chargeback dispute", "account reconciliation",
    "regulatory reporting", "customer due diligence", "suspicious activity reporting",
    "cash management services", "treasury services", "merchant services",
    "ATM network operations", "branch teller operations", "digital onboarding",
    "wire fraud prevention", "credit bureau reporting", "loan servicing",
    "identity verification", "customer relationship management", "risk scoring",
]

_TEMPLATES = [
    "Cascade Bank's {topic} procedures require staff to follow documented controls. {fake_sentence}",
    "As part of {topic}, employees must verify customer information before proceeding. {fake_sentence}",
    "{fake_sentence} This directly affects {topic} across all branches.",
    "Customers should be aware that {topic} may involve additional review steps. {fake_sentence}",
    "{fake_sentence} Regulators expect strong internal controls around {topic}.",
    "Under this policy, {topic} is reviewed on a quarterly basis. {fake_sentence}",
    "{fake_sentence} Branch staff receive annual training on {topic}.",
    "Any exception to {topic} must be documented and approved by a supervisor. {fake_sentence}",
    "{fake_sentence} Cascade Bank monitors {topic} continuously for irregularities.",
    "Failure to follow {topic} guidelines may result in disciplinary action. {fake_sentence}",
]


def banking_sentence(fake: Faker) -> str:
    """One sentence combining a Faker sentence with a banking-domain template."""
    topic = random.choice(BANKING_VOCAB)
    template = random.choice(_TEMPLATES)
    return template.format(topic=topic, fake_sentence=fake.sentence())


def banking_paragraph(fake: Faker, min_sentences: int = 3, max_sentences: int = 7) -> str:
    """A paragraph of banking-flavored text: Faker sentences woven into
    banking vocabulary templates so it reads as topical, not pure lorem."""
    n = random.randint(min_sentences, max_sentences)
    return " ".join(banking_sentence(fake) for _ in range(n))


# ---------------------------------------------------------------------------
# Base PDF class
# ---------------------------------------------------------------------------


class CascadePDF(FPDF):
    """fpdf2 subclass with a Cascade Bank title page and disclaimer footer."""

    def __init__(self, company_name: str = "Cascade Bank"):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.company_name = company_name
        self.set_auto_page_break(auto=True, margin=20)
        self.set_margins(20, 20, 20)

    # -- Central sanitization -------------------------------------------
    #
    # Every fpdf2 text-writing primitive routes through here, so callers
    # (in this class or in the generator modules) can never accidentally
    # feed raw unicode straight into the latin-1-only core fonts.

    def cell(self, *args, **kwargs):  # noqa: D102 (fpdf2 hook override)
        if "text" in kwargs:
            kwargs["text"] = latin1_safe(kwargs["text"])
        elif len(args) >= 3:
            args = list(args)
            args[2] = latin1_safe(args[2])
        return super().cell(*args, **kwargs)

    def multi_cell(self, *args, **kwargs):  # noqa: D102 (fpdf2 hook override)
        if "text" in kwargs:
            kwargs["text"] = latin1_safe(kwargs["text"])
        elif len(args) >= 3:
            args = list(args)
            args[2] = latin1_safe(args[2])
        return super().multi_cell(*args, **kwargs)

    def write(self, *args, **kwargs):  # noqa: D102 (fpdf2 hook override)
        if "text" in kwargs:
            kwargs["text"] = latin1_safe(kwargs["text"])
        elif len(args) >= 2:
            args = list(args)
            args[1] = latin1_safe(args[1])
        return super().write(*args, **kwargs)

    def footer(self) -> None:  # noqa: D102 (fpdf2 hook, called automatically)
        self.set_y(-15)
        self.set_font("helvetica", "I", 7)
        self.set_text_color(140, 140, 140)
        self.cell(0, 10, f"{FOOTER_DISCLAIMER} - Page {self.page_no()}", align="C")
        self.set_text_color(0, 0, 0)

    def set_disclaimed_metadata(self, title: str, subject: str = "synthetic demo data") -> None:
        self.set_title(latin1_safe(title))
        self.set_author("Cascade Bank (fictional)")
        self.set_subject(subject)
        self.set_creator("cascade-bank synthetic data generator")
        self.set_keywords("synthetic, fictional, demo, cascade bank")

    def add_title_page(self, doc_title: str, subtitle: str = "") -> None:
        self.add_page()
        self.ln(35)
        self.set_font("helvetica", "B", 24)
        self.multi_cell(0, 12, latin1_safe(self.company_name), align="C")

        self.set_font("helvetica", "", 16)
        self.ln(6)
        self.multi_cell(0, 10, latin1_safe(doc_title), align="C")

        if subtitle:
            self.set_font("helvetica", "I", 12)
            self.ln(2)
            self.multi_cell(0, 8, latin1_safe(subtitle), align="C")

        self.ln(25)
        self.set_font("helvetica", "B", 14)
        self.set_text_color(180, 0, 0)
        self.multi_cell(0, 9, DISCLAIMER, align="C")
        self.set_text_color(0, 0, 0)

        self.ln(4)
        self.set_font("helvetica", "", 10)
        self.multi_cell(0, 6, latin1_safe(DISCLAIMER_LONG), align="C")

    def add_heading(self, heading: str, new_page: bool = True) -> None:
        if new_page:
            self.add_page()
        self.set_font("helvetica", "B", 16)
        self.multi_cell(0, 10, latin1_safe(heading))
        self.ln(2)
        self.set_font("helvetica", "", 11)

    def add_paragraphs(self, paragraphs: list[str]) -> None:
        self.set_font("helvetica", "", 11)
        for p in paragraphs:
            self.multi_cell(0, 6, latin1_safe(p))
            self.ln(3)

    def add_section(self, heading: str, paragraphs: list[str], new_page: bool = True) -> None:
        self.add_heading(heading, new_page=new_page)
        self.add_paragraphs(paragraphs)

    def add_numbered_list(self, items: list[str]) -> None:
        self.set_font("helvetica", "", 11)
        for i, item in enumerate(items, start=1):
            self.multi_cell(0, 6, latin1_safe(f"{i}. {item}"))
            self.ln(1)

    def add_key_value_table(self, rows: list[tuple[str, str]], key_width: float = 55) -> None:
        self.set_font("helvetica", "B", 11)
        for key, value in rows:
            x_start = self.get_x()
            y_start = self.get_y()
            self.set_font("helvetica", "B", 11)
            self.multi_cell(key_width, 6, latin1_safe(key), align="L")
            y_end_key = self.get_y()
            self.set_xy(x_start + key_width, y_start)
            self.set_font("helvetica", "", 11)
            self.multi_cell(0, 6, latin1_safe(value), align="L")
            y_end_val = self.get_y()
            self.set_y(max(y_end_key, y_end_val))
            self.ln(1)

    def add_simple_table(self, header: list[str], rows: list[list[str]], col_widths: list[float]) -> None:
        self.set_font("helvetica", "B", 10)
        self.set_fill_color(230, 230, 230)
        for h, w in zip(header, col_widths):
            self.cell(w, 8, latin1_safe(h), border=1, fill=True)
        self.ln()
        self.set_font("helvetica", "", 9)
        for row in rows:
            for value, w in zip(row, col_widths):
                self.cell(w, 8, latin1_safe(str(value)), border=1)
            self.ln()
