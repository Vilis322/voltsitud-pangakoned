# Deduplication report

Input file: 5151 rows.

| Metric | Value |
|---|---|
| Rows before | 5151 |
| Exact duplicates removed | 103 |
| Near-duplicates removed | 48 |
| Distinct near-dup groups | 48 |
| Rows after | 5000 |
| Total removed | 151 (2.93%) |

## Definitions

- **Exact duplicate** — every column is identical to a row that
  appeared earlier in the file. Removed via `df.duplicated(keep='first')`.
- **Near-duplicate** — same `caller_number`, `called_number`, and
  timestamp rounded to the nearest minute, with `duration_sec`
  within ±5 s of an earlier row in the same group. These are the
  rows the generator perturbed by a small numeric offset.

## Removal order

Exact duplicates are removed first; near-duplicate detection is
then run on the surviving rows. This keeps the two counts disjoint
and avoids double-counting a row that is both a strict duplicate
and inside a near-dup cluster.
