# Dataset card — `raw_bank_calls.csv`

## Purpose

A synthetic dataset that emulates phone-call events at an Estonian retail
bank, used for training and evaluating fraud-detection models in the
defense project "Võltsitud Pangakõned" (fake bank calls).

It deliberately mirrors the data model described in the course brief on
fake bank calls and the companion ML methodology document. The intent
is that every column has a justification rooted in those briefs, and
that the in-data patterns reproduce the risk signals named there.

## How it is produced

```
python dataset/generate.py [--rows 5000] [--fraud-rate 0.15] [--seed 42]
```

Output is written to `data/raw_bank_calls.csv` (gitignored).

## Schema

| Column | Type | Description | Source justification |
|---|---|---|---|
| `call_id` | string | Per-call unique identifier `C\d{8}`. | Standard trace field; section 8.1 of methodology (identifiers). |
| `caller_number` | string | E.164 number of the caller (`+372…`). | Section 4 — call-event field. Also used for mass-fraud signal in section 5. |
| `called_number` | string | E.164 number of the customer. | Section 4 — call-event field. |
| `timestamp` | string | Call start time. **Deliberately corrupted**: mixed formats (ISO, slash, dash 2-digit year, ISO+TZ, epoch, free-form Estonian). | Section 4 — time of call. Noise feeds Sprint 1 normalization task. |
| `duration_sec` | float | Call duration in seconds. | Section 4 — kõne kestus. |
| `was_answered` | bool | Whether the customer picked up. | Section 4 — kas kõnele vastati. |
| `channel` | string | Connection type. **Multiple spellings** of `voip` / `mobile` / `landline`. | Section 4 — võimalik kanal. Noise feeds the standardization task. |
| `was_hangup_by_client` | bool | True if the customer ended the call (vs the caller). | Section 4 — kas klient katkestas. |
| `login_after_call` | bool | Customer logged into internet bank within 1 hour after the call. | Section 4 — sisselogimine internetipanka. Core risk signal. |
| `twofa_confirmed_after_call` | bool | A 2FA approval (Mobiil-ID / Smart-ID) was confirmed within 1 hour. | Section 4 — 2FA või autentimistoiming. |
| `transfer_after_call` | bool | Outgoing payment within 1 hour. | Section 4 — kas tehti makse. |
| `new_payee_after_call` | bool | A new payment beneficiary was added within 1 hour. | Section 4 — kas lisati uus makse saaja. |
| `settings_changed_after_call` | bool | Contact or security settings were modified within 1 hour. | Section 4 — turvaseaded. |
| `time_to_next_action_min` | float | Minutes from call end to the next account action; lower means tighter coupling (a strong fraud indicator). | Section 6 — ajatunnused. |
| `complaint_text` | string | Free-text customer complaint in Estonian. Often empty for legitimate calls; populated with keyword-rich text for fraud. | Section 4 — kliendi kirjeldus kõnest. |
| `manual_severity` | int | Manual case rating, 1–5. | Section 4 — juhtumi käsitsi hinnang. |
| `label_5way` | string | One of the 5 label classes. **Multiple casings** are present and must be normalized. | Section 7 — soovitatav märgistus. |
| `confidence_score` | float | 0..1 score for label trustworthiness; higher for explicitly confirmed states. | "Confidence score" section of the ML methodology document. |

## Label classes (canonical form)

| Class | Meaning |
|---|---|
| `confirmed_fraud` | Customer lost money or fraud was internally confirmed. |
| `high_risk_reported` | Reported as fraud and the risk pattern is strong. |
| `community_flagged` | Number flagged by external community sources; internal evidence is partial. |
| `unknown` | Suspicious but evidence is insufficient. |
| `verified_legitimate` | Verified bank-controlled call (known service numbers). |

## Generation parameters (defaults)

| Parameter | Default | Why |
|---|---|---|
| `rows` | 5000 | Provides enough volume for hyperparameter search while keeping CI runs fast. |
| `fraud_rate` | 0.15 | Realistic class imbalance for the problem domain. |
| `seed` | 42 | Deterministic regeneration for reproducibility. |
| `NUM_FRAUD_CAMPAIGNS` | 50 | Mass-fraud signal: 50 distinct caller numbers, each calling 30–100 victims. |
| `LEGITIMATE_SERVICE_NUMBERS` | 5 hard-coded numbers | Section 8 — negative class only from verified service numbers. |

## Patterns intentionally embedded

These should be rediscovered during Sprints 2–4 (EDA + modeling):

1. **Mass-fraud campaigns** — the top non-bank caller numbers each hit 30–100
   victims within a short window; legitimate numbers are spread across
   many days at lower per-day volume.
2. **Tight time-to-next-action for fraud** — `time_to_next_action_min` is
   centered around 8 minutes for `confirmed_fraud` and ~200 for `unknown`
   / verified.
3. **High post-call action rates for fraud** — `login_after_call ≈ 75%`
   for confirmed fraud vs ~18% for verified legitimate; similar gaps for
   `transfer_after_call` and `new_payee_after_call`.
4. **Business-hour clustering for fraud** — fraud calls cluster around
   10:00–14:00 (urgent prompts when customers can act); legitimate calls
   are spread across the working day.
5. **Estonian keyword signal** — fraud complaints contain "pank", "pettus",
   "kahtlane", "turvakonto", "Mobiil-ID", "Smart-ID"; legitimate complaints
   are short or empty.
6. **Confidence-weighted labels** — `confidence_score` is high for
   `confirmed_fraud` and `verified_legitimate`, lower for `community_flagged`
   and `unknown`. This supports confidence-aware training in Sprint 5.

## Quality issues injected (for Sprint 1 cleaning)

| Issue | Approx. rate | Affected columns |
|---|---|---|
| Mixed timestamp formats | 5% of rows | `timestamp` |
| Missing values | 2–8% per column | `channel`, `duration_sec`, `complaint_text`, `manual_severity`, `time_to_next_action_min` |
| Exact duplicates | 2% of rows | all columns |
| Near-duplicates (duration drift ±2s) | 1% of rows | all columns |
| Inconsistent categorical casing | always | `channel`, `label_5way` |

A Sprint 1 task verifies cleaning produces a single canonical form per
column and removes both duplicate classes.

## Known limitations

1. **Synthetic** — no protected real-world data was used. Patterns are
   intentionally clean enough to learn; real-world data would be noisier.
2. **No spoofed-number simulation** — every caller_number is treated as
   the visible identifier. Number-spoofing is the topic of a sister
   project, not this one.
3. **Estonian language only** — complaint text is templated Estonian; no
   English / Russian variants.
4. **No textual variety** — only 7 fraud complaint templates. Sufficient
   for keyword analysis but limits NLP exploration.

## Reproducibility

Same seed + same generator code → bit-identical CSV (modulo CSV row
ordering: a shuffle uses an RNG-derived seed). To regenerate from a
clean state:

```bash
rm -f data/raw_bank_calls.csv
python dataset/generate.py --seed 42
python dataset/qa.py
```
