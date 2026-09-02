"""Official-label contract for the KArSL Core-28 subset.

The official KArSL page links the ``KARSL-502_Labels.xlsx`` workbook.  The
workbook is intentionally not vendored in this repository: it is a small
source artifact that the acquisition command downloads outside Git and hashes.
This module contains only the frozen, auditable interpretation of its rows.

The source vocabulary contains 39 letter-like entries in the workbook.  The
first 28 standard Arabic alphabet letters are SignIDs 32--59.  SignIDs 60--70
are retained as extended-letter candidates and are not part of Core-28.
``validate_core28_records`` must be run against the downloaded official
workbook before a production extraction is submitted.
"""

from __future__ import annotations

import csv
import json
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping
from xml.etree import ElementTree

OFFICIAL_SITE_URL = "https://hamzah-luqman.github.io/KArSL/"
OFFICIAL_RGB_PAGE_URL = "https://hamzah-luqman.github.io/KArSL/download_video_502.html"
OFFICIAL_REPOSITORY_URL = "https://github.com/Hamzah-Luqman/KArSL"
ORIGINAL_RESEARCH_URL = "https://dl.acm.org/doi/10.1145/3423420"
OFFICIAL_LABELS_URL = (
    "https://kfupmedusa-my.sharepoint.com/:x:/g/personal/"
    "hluqman_kfupm_edu_sa/EQQz8zKWYWtDl2kt2bSGSlsB6-73UZAo-NHKbDSYwPynBA?e=nJO7LO"
)

# Current official page links.  A URL is kept even when a provider later
# disables it, because changing the source silently would break provenance.
OFFICIAL_RGB_SOURCES: Mapping[tuple[str, str], str] = {
    ("01", "train"): (
        "https://kfupmedusa-my.sharepoint.com/:f:/g/personal/"
        "hluqman_kfupm_edu_sa/Ei2vkIK1I3BJgIzon9_F9PIBffWYnrs4RDxuJPU7exefuw?e=aLH57c"
    ),
    ("01", "test"): (
        "https://kfupmedusa-my.sharepoint.com/:f:/g/personal/"
        "hluqman_kfupm_edu_sa/EonTe3maT6tMgdp44zxDiVkBzbJtFQ-arcipbBmvAyJFiA?e=R9laLg"
    ),
    ("02", "train"): (
        "https://kfupmedusa-my.sharepoint.com/:f:/g/personal/"
        "hluqman_kfupm_edu_sa/EhAfvzLGQ9BPh5IKE9ufclcB5qmB2__daTjsOnrctnyYgw?e=1NNa5L"
    ),
    ("02", "test"): (
        "https://kfupmedusa-my.sharepoint.com/:f:/g/personal/"
        "hluqman_kfupm_edu_sa/Epc8HYVNCWhDmg9GMIHTx4wBf0RrSAlVQP61Sc1QBNtd5Q?e=YdhIE5"
    ),
    ("03", "train"): (
        "https://kfupmedusa-my.sharepoint.com/:f:/g/personal/"
        "hluqman_kfupm_edu_sa/Eg3pV0Nxd1xPjNq_NH-6Qy0ByMzVvxw0zpSirpbV_yH_hg?e=ZUSigD"
    ),
    ("03", "test"): (
        "https://kfupmedusa-my.sharepoint.com/:f:/g/personal/"
        "hluqman_kfupm_edu_sa/Elzlkj5QrLxOrHbMauGl4PgBRxr0YTaEMfFwUydR3DVT-A?e=P8Oj1H"
    ),
}

CORE28_SIGN_IDS: tuple[int, ...] = tuple(range(32, 60))
EXTENDED_LETTER_SIGN_IDS: tuple[int, ...] = tuple(range(60, 71))

# These are the exact visible Arabic cells and English glosses from the
# workbook's 28 standard-letter rows.  Unicode NFC is used only for equality
# checking; the CSV keeps these labels as written (without transliteration).
_CORE28_LABELS: tuple[tuple[str, str], ...] = (
    ("ا", "alif"),
    ("ب", "baa"),
    ("ت", "ta"),
    ("ث", "tha"),
    ("ج", "Jiim"),
    ("ح", "Haa"),
    ("خ", "kha"),
    ("د", "daal"),
    ("ذ", "thal"),
    ("ر", "raa"),
    ("ز", "zay"),
    ("س", "siin"),
    ("ش", "shiin"),
    ("ص", "Saad"),
    ("ض", "Daad"),
    ("ط", "Taa"),
    ("ظ", "Zaa"),
    ("ع", "Ayn"),
    ("غ", "ghayn"),
    ("ف", "faa"),
    ("ق", "qaaf"),
    ("ك", "kaaf"),
    ("ل", "laam"),
    ("م", "miim"),
    ("ن", "noon"),
    ("ه", "haa"),
    ("و", "waaw"),
    ("ي", "yaa"),
)

_EXTENDED_LABELS: tuple[tuple[str, str], ...] = (
    ("ة", "taa marbuuTa"),
    ("أ", "alif with hamza above"),
    ("ؤ", "Waaw with hamza"),
    ("ئ", "Alif maqsoura with hamza"),
    ("ئـ", "hamza on line"),
    ("ء", "hamza"),
    ("إ", "alif with hamza below"),
    ("آ", "ALif with maad"),
    ("ى", "Alif maqsoura"),
    ("لا", "laam Alif"),
    ("ال", "Al"),
)


@dataclass(frozen=True, slots=True)
class LabelRecord:
    """One row from ``KARSL-502_Labels.xlsx`` or its canonical CSV export."""

    sign_id: int
    label_ar: str
    label_en: str
    source_row: int | None = None
    chapter: str = ""
    is_core28: bool = False
    is_extended_letter: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "sign_id": f"{self.sign_id:04d}",
            "label_ar": self.label_ar,
            "label_en_if_available": self.label_en,
            "chapter_category": self.chapter,
            "is_core28": self.is_core28,
            "is_extended_letter": self.is_extended_letter,
            "source_label_row": self.source_row,
        }


def normalize_label(value: str) -> str:
    """Normalize only for comparisons; never rewrite stored source labels."""

    return unicodedata.normalize("NFC", str(value)).strip()


def _cell_value(cell: ElementTree.Element, shared: list[str], ns: dict[str, str]) -> str:
    value = cell.find("a:v", ns)
    if cell.attrib.get("t") == "s" and value is not None:
        return shared[int(value.text or "0")]
    if cell.attrib.get("t") == "inlineStr":
        return "".join(text.text or "" for text in cell.findall(".//a:t", ns))
    return "" if value is None else value.text or ""


def _load_xlsx_rows(path: Path) -> list[list[str]]:
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    try:
        with zipfile.ZipFile(path) as workbook:
            shared: list[str] = []
            if "xl/sharedStrings.xml" in workbook.namelist():
                shared_root = ElementTree.fromstring(workbook.read("xl/sharedStrings.xml"))
                shared = [
                    "".join(text.text or "" for text in item.findall(".//a:t", ns))
                    for item in shared_root.findall("a:si", ns)
                ]
            sheet = ElementTree.fromstring(workbook.read("xl/worksheets/sheet1.xml"))
    except (KeyError, OSError, zipfile.BadZipFile) as error:
        raise ValueError(f"Cannot read KArSL label workbook {path}: {error}") from error
    return [
        [_cell_value(cell, shared, ns) for cell in row.findall("a:c", ns)]
        for row in sheet.findall(".//a:row", ns)
    ]


def _load_csv_rows(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        return [list(row) for row in reader]


def load_label_records(path: str | Path) -> list[LabelRecord]:
    """Load the official workbook or a CSV with its three label columns."""

    source = Path(path)
    rows = _load_xlsx_rows(source) if source.suffix.lower() == ".xlsx" else _load_csv_rows(source)
    if not rows:
        raise ValueError(f"Empty label source: {source}")
    header = [normalize_label(value).lower() for value in rows[0]]
    aliases = {
        "signid": {"signid", "sign_id", "id"},
        "arabic": {"sign-arabic", "label_ar", "label_arabic", "arabic"},
        "english": {"sign-english", "label_en", "label_en_if_available", "label_english", "english"},
    }

    def index_for(names: set[str]) -> int:
        for index, value in enumerate(header):
            if value in names:
                return index
        raise ValueError(f"Label source {source} is missing one of {sorted(names)}")

    id_index = index_for(aliases["signid"])
    ar_index = index_for(aliases["arabic"])
    en_index = index_for(aliases["english"])
    records: list[LabelRecord] = []
    for source_row, row in enumerate(rows[1:], start=2):
        if not row or not any(str(value).strip() for value in row):
            continue
        try:
            sign_id = int(float(row[id_index]))
        except (IndexError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid SignID on source row {source_row}: {row}") from error
        try:
            arabic = str(row[ar_index])
            english = str(row[en_index])
        except IndexError as error:
            raise ValueError(f"Incomplete label row {source_row}: {row}") from error
        records.append(LabelRecord(sign_id, arabic, english, source_row=source_row))
    return records


def validate_core28_records(records: Iterable[LabelRecord]) -> list[LabelRecord]:
    """Validate that source rows exactly match the frozen Core-28 contract."""

    by_id: dict[int, LabelRecord] = {}
    for record in records:
        if record.sign_id in by_id:
            raise ValueError(f"Duplicate SignID in label source: {record.sign_id}")
        by_id[record.sign_id] = record
    missing = sorted(set(CORE28_SIGN_IDS) - set(by_id))
    if missing:
        raise ValueError(f"Official label source is missing Core-28 SignIDs: {missing}")

    validated: list[LabelRecord] = []
    for index, sign_id in enumerate(CORE28_SIGN_IDS):
        source = by_id[sign_id]
        expected_ar, expected_en = _CORE28_LABELS[index]
        if normalize_label(source.label_ar) != normalize_label(expected_ar):
            raise ValueError(
                f"SignID {sign_id:04d} Arabic label mismatch: "
                f"source={source.label_ar!r}, expected={expected_ar!r}"
            )
        if normalize_label(source.label_en) != normalize_label(expected_en):
            raise ValueError(
                f"SignID {sign_id:04d} English label mismatch: "
                f"source={source.label_en!r}, expected={expected_en!r}"
            )
        validated.append(
            LabelRecord(
                sign_id=sign_id,
                label_ar=source.label_ar,
                label_en=source.label_en,
                source_row=source.source_row,
                chapter="Letters",
                is_core28=True,
            )
        )
    return validated


def core28_records() -> list[LabelRecord]:
    """Return the committed mapping used when no source workbook is present."""

    return [
        LabelRecord(
            sign_id=sign_id,
            label_ar=labels[0],
            label_en=labels[1],
            source_row=sign_id + 1,
            chapter="Letters",
            is_core28=True,
        )
        for sign_id, labels in zip(CORE28_SIGN_IDS, _CORE28_LABELS)
    ]


def extended_letter_records() -> list[LabelRecord]:
    """Return letter-like rows excluded from Core-28 for documentation."""

    return [
        LabelRecord(
            sign_id=sign_id,
            label_ar=labels[0],
            label_en=labels[1],
            source_row=sign_id + 1,
            chapter="Letters / extended-letter candidate",
            is_extended_letter=True,
        )
        for sign_id, labels in zip(EXTENDED_LETTER_SIGN_IDS, _EXTENDED_LABELS)
    ]


def mapping_document() -> dict[str, object]:
    """Small JSON-compatible source/mapping declaration for reports and tools."""

    return {
        "mapping_version": "karsl-core28-v1",
        "official_site_url": OFFICIAL_SITE_URL,
        "official_rgb_page_url": OFFICIAL_RGB_PAGE_URL,
        "official_repository_url": OFFICIAL_REPOSITORY_URL,
        "original_research_url": ORIGINAL_RESEARCH_URL,
        "official_label_workbook_url": OFFICIAL_LABELS_URL,
        "label_workbook_name": "KARSL-502_Labels.xlsx",
        "retrieval_note": (
            "The official SharePoint link was recorded on 2026-09-02 but returned "
            "an administrator-disabled page at development time. The committed "
            "rows are the frozen SignID 32-59 mapping; --labels-only must verify "
            "the downloaded official workbook before production extraction."
        ),
        "unicode_comparison": "NFC + strip for validation; stored labels are not rewritten",
        "core28": [record.to_dict() | {"label_index": index} for index, record in enumerate(core28_records())],
        "extended_letter_candidates": [record.to_dict() for record in extended_letter_records()],
    }


def write_mapping_csv(path: str | Path) -> None:
    """Write the compact committed Core-28 mapping."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sign_id",
        "label_ar",
        "label_en_if_available",
        "label_index",
        "chapter_category",
        "is_core28",
        "is_extended_letter",
        "source_label_row",
    ]
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for index, record in enumerate(core28_records()):
            row = record.to_dict()
            row["label_index"] = index
            writer.writerow({field: row.get(field, "") for field in fields})


def write_mapping_json(path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(mapping_document(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
