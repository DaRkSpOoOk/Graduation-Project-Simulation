# TASK-008B — Official KArSL Label Verification

**Authoritative source:** `/home/hatim/datasets/KArSL-502/KARSL-502_Labels.xlsx`
**Mapping version frozen:** `karsl-core28-v2-official`
**Verification date:** 2026-09-03

This workbook shipped with the officially downloaded dataset and is treated as
the single authority for labels. Public mirrors, the repository's candidate CSV
and inferred SignID ranges were all checked *against* it, never preferred over
it. Stored values are preserved exactly; Unicode NFC normalization was used for
equality comparison only and nothing was written back.

---

## Workbook

| Property | Value |
|---|---|
| File name | `KARSL-502_Labels.xlsx` |
| Size | 27,141 bytes |
| SHA-256 | `c13717c549b8cb8cfa465237a3f7dfed73f84149ca5c448b2b253fd96321b14e` |
| Sheets | ['Sheet1'] |
| Sheet used | `Sheet1` |
| Total rows | 503 (1 header + 502 data) |
| Columns | ['SignID', 'Sign-Arabic', 'Sign-English'] |
| SignID range | 1–502 (contiguous) |
| Parse issues | 0 |

The workbook covers the whole KArSL-502 vocabulary (502 signs). Only
SignIDs 1–70 are downloaded locally and therefore in scope here.

### Sign-Arabic cell types

SignIDs 1–31 store their label as a spreadsheet **integer**, not text. They are
numeric magnitudes, and `Sign-Arabic` equals `Sign-English` for every one of
them. This is recorded rather than "corrected": inventing Arabic-Indic numeral
glyphs the workbook does not contain would be a fabrication.

---

## Core-28 — verification result: **PASS**

28 classes, SignIDs 0032–0059, matching the repository's candidate mapping with
**zero** label mismatches on either the Arabic or the English field.

| SignID | Sign-Arabic | Sign-English | class index |
|---|---|---|---|
| 0032 | ا | alif | 0 |
| 0033 | ب | baa | 1 |
| 0034 | ت | ta | 2 |
| 0035 | ث | tha | 3 |
| 0036 | ج | Jiim | 4 |
| 0037 | ح | Haa | 5 |
| 0038 | خ | kha | 6 |
| 0039 | د | daal | 7 |
| 0040 | ذ | thal | 8 |
| 0041 | ر | raa | 9 |
| 0042 | ز | zay | 10 |
| 0043 | س | siin | 11 |
| 0044 | ش | shiin | 12 |
| 0045 | ص | Saad | 13 |
| 0046 | ض | Daad | 14 |
| 0047 | ط | Taa | 15 |
| 0048 | ظ | Zaa | 16 |
| 0049 | ع | Ayn | 17 |
| 0050 | غ | ghayn | 18 |
| 0051 | ف | faa | 19 |
| 0052 | ق | qaaf | 20 |
| 0053 | ك | kaaf | 21 |
| 0054 | ل | laam | 22 |
| 0055 | م | miim | 23 |
| 0056 | ن | noon | 24 |
| 0057 | ه | haa | 25 |
| 0058 | و | waaw | 26 |
| 0059 | ي | yaa | 27 |

---

## Extended letters — verification result: **PASS**

11 classes, SignIDs 0060–0070. The workbook confirms these are **extended
Arabic letter forms** (hamza carriers, taa marbuuta, alif maqsoura, laam-alif
and the definite article), not a different category.

| SignID | Sign-Arabic | Sign-English | classification |
|---|---|---|---|
| 0060 | ة | taa marbuuTa | extended Arabic letter form |
| 0061 | أ | alif with hamza above | extended Arabic letter form |
| 0062 | ؤ | Waaw with hamza | extended Arabic letter form |
| 0063 | ئ | Alif maqsoura with hamza | extended Arabic letter form |
| 0064 | ئـ | hamza on line | extended Arabic letter form |
| 0065 | ء | hamza | extended Arabic letter form |
| 0066 | إ | alif with hamza below | extended Arabic letter form |
| 0067 | آ | ALif with maad | extended Arabic letter form |
| 0068 | ى | Alif maqsoura | extended Arabic letter form |
| 0069 | لا | laam Alif | extended Arabic letter form |
| 0070 | ال | Al | extended Arabic letter form |

**Letters in total: 28 + 11 = 39 classes (SignIDs 0032–0070).**

---

## Numbers / digits — 31 classes, not 30

The KArSL literature describes approximately 30 number classes while SignIDs
0001–0031 span 31 IDs. The workbook settles it: there are **31** number classes.
The count is not forced to match the literature.

| SignID | Sign-Arabic | Sign-English | classification |
|---|---|---|---|
| 0001 | `0` | 0 | number (int cell) |
| 0002 | `1` | 1 | number (int cell) |
| 0003 | `2` | 2 | number (int cell) |
| 0004 | `3` | 3 | number (int cell) |
| 0005 | `4` | 4 | number (int cell) |
| 0006 | `5` | 5 | number (int cell) |
| 0007 | `6` | 6 | number (int cell) |
| 0008 | `7` | 7 | number (int cell) |
| 0009 | `8` | 8 | number (int cell) |
| 0010 | `9` | 9 | number (int cell) |
| 0011 | `10` | 10 | number (int cell) |
| 0012 | `20` | 20 | number (int cell) |
| 0013 | `30` | 30 | number (int cell) |
| 0014 | `40` | 40 | number (int cell) |
| 0015 | `50` | 50 | number (int cell) |
| 0016 | `60` | 60 | number (int cell) |
| 0017 | `70` | 70 | number (int cell) |
| 0018 | `80` | 80 | number (int cell) |
| 0019 | `90` | 90 | number (int cell) |
| 0020 | `100` | 100 | number (int cell) |
| 0021 | `200` | 200 | number (int cell) |
| 0022 | `300` | 300 | number (int cell) |
| 0023 | `400` | 400 | number (int cell) |
| 0024 | `500` | 500 | number (int cell) |
| 0025 | `600` | 600 | number (int cell) |
| 0026 | `700` | 700 | number (int cell) |
| 0027 | `800` | 800 | number (int cell) |
| 0028 | `900` | 900 | number (int cell) |
| 0029 | `1000` | 1000 | number (int cell) |
| 0030 | `1000000` | 1000000 | number (int cell) |
| 0031 | `10000000` | 10000000 | number (int cell) |

The series is 0–9 (10), then the tens 10–90 (9), the hundreds 100–900 (9), then
1000, 1000000 and 10000000 (3) — 31 in total. Note the magnitudes jump from
1000 straight to 1000000, so 10000 and 100000 have no class.

### SignID 0031

**SignID 0031 is the numeric magnitude `10000000` (ten million).** It is a
number, not a letter and not an extended form, and its cell is an integer. This
is the single ID responsible for the 30-vs-31 discrepancy in the literature.

---

## Independent corroboration from the filenames

The official filename convention encodes a chapter code in its leading field,
and it partitions exactly as the workbook does — without being derived from it:

| Chapter field | SignIDs | Classes | Workbook category |
|---|---|---|---|
| `01` | 0001–0031 | 31 | numbers |
| `02` | 0032–0070 | 39 | letters (28 core + 11 extended) |

Two independent sources agreeing on the same 31/39 split is stronger evidence
than either alone.
