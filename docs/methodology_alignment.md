# Methodology alignment

This document maps every design choice in `dataset/generate.py` and
`dataset/qa.py` to the exact section of the course briefs that justifies
it. The intent is that during the defense any question of the form
"why is this column here / why this distribution / why this label" can
be answered by pointing at a specific section.

The two source documents referenced below are:

- **Topic brief** — the fake bank calls section of the course AI
  assistant material.
- **ML methodology** — the general "Petukõnede numbritega seotud
  ML-andmestiku koostamise juhend" guidance from the same assistant.

## 1. Data groups covered

The topic brief section 4 (`Andmete kogumine`) splits the required data
into three groups. The dataset covers all three:

| Group from brief | Columns in dataset |
|---|---|
| Kõnesündmuse andmed | `call_id`, `caller_number`, `called_number`, `timestamp`, `duration_sec`, `was_answered`, `channel`, `was_hangup_by_client` |
| Kliendi käitumise andmed pärast kõnet | `login_after_call`, `twofa_confirmed_after_call`, `transfer_after_call`, `new_payee_after_call`, `settings_changed_after_call`, `time_to_next_action_min` |
| Kaebuse ja tekstiandmed | `complaint_text`, `manual_severity` |

## 2. Risk signals embedded

Topic brief section 5 (`Võimalikud riskisignaalid`) lists the patterns
a model should be able to detect. Each is realized in the data:

| Risk signal | How it appears in the data | Verifying QA output |
|---|---|---|
| Sama numbri alt helistatakse paljudele klientidele | 50 campaign caller_numbers each emit 30–100 calls; legitimate numbers are spread thinly. | `qa.py` "top 10 callers by volume" — non-bank numbers dominate by fraud rate. |
| Kõne järel tekib kohe autentimiskatse või makse | `login_after_call`, `twofa_confirmed_after_call`, `transfer_after_call` set with high probability for fraud labels. | `qa.py` "behavior share by label". |
| Kõne ja kahtlane tehing toimuvad lähestikku | `time_to_next_action_min` centered at 8 min for fraud vs 200 min for unknown. | Compare per-label medians of `time_to_next_action_min`. |
| Kliendikirjeldustes korduvad samad pettusmustrid | 7 fixed Estonian fraud templates with shared keywords (`pank`, `pettus`, `kahtlane`, `turvakonto`, `Mobiil-ID`, `Smart-ID`). | Sprint 2 keyword cloud task will surface this. |
| Nähtav number ei sobitu legitiimse teenusnumbriga | Legitimate numbers come from a small whitelist; everything else is generated random or campaign. | `qa.py` "top 10 callers" — the 5 whitelist numbers have 0% fraud rate. |

## 3. Feature types from brief section 6

The topic brief enumerates four feature families. The dataset already
carries the raw fields needed to derive each:

| Family | Derived from | Where it lands in Sprint 3 |
|---|---|---|
| Ajatunnused | `timestamp`, `time_to_next_action_min` | Issue "Temporal feature extraction" — hour, weekday, is_business_hours, is_weekend. |
| Käitumuslikud tunnused | `caller_number`, post-call action booleans | Issue "Additional behavioral features" in Sprint 5 — aggregates over time windows. |
| Tekstilised tunnused | `complaint_text` | Issue "Complaint keyword cloud" (Sprint 2) and downstream NLP features. |
| Järgnev sündmusahel | post-call action booleans + `time_to_next_action_min` | Already represented; ML prep wraps these unchanged. |

## 4. Label schema (brief section 7)

The brief recommends 5 classes; the dataset uses exactly those:

| Class in brief | Used as canonical value |
|---|---|
| `confirmed_fraud` | yes |
| `high_risk_reported` | yes |
| `community_flagged` | yes |
| `unknown` | yes |
| `verified_legitimate` | yes |

Multiple casings (`Confirmed_Fraud`, `CONFIRMED_FRAUD`, etc.) are
injected on purpose so the Sprint 1 task on standardization has work
to do. The canonical form used by all downstream code is lower-case
snake (`confirmed_fraud`).

## 5. Negative-class construction (brief section 8)

> Negatiivsete näidete jaoks sobivad ainult kinnitatud legitiimsed
> pangakõned, näiteks panga teadaolevad teenusnumbrid.

Honored: the only source of `verified_legitimate` rows is a hard-coded
whitelist of 5 Estonian bank service numbers (SEB, Swedbank, LHV, Coop
Pank, Luminor). Random one-off numbers are never labeled legitimate.

## 6. Confidence score (ML methodology, section "Confidence score")

The ML methodology document argues complaint data is unverified and so
each row should carry a confidence-of-label score. Implemented in
`_confidence_for_label`:

| Label | Center | Rationale |
|---|---|---|
| `confirmed_fraud` | 0.95 | Internally verified outcome. |
| `verified_legitimate` | 0.95 | Known service number. |
| `high_risk_reported` | 0.75 | Strong report, no internal confirmation yet. |
| `community_flagged` | 0.55 | External signal only. |
| `unknown` | 0.35 | Suspicious but insufficient evidence. |

Noise ±0.05 (Gaussian) is added so the score is not bimodal.

## 7. Data architecture (ML methodology, "Soovitatav andmestiku arhitektuur")

The methodology document recommends a 4-layer architecture (complaint
sources, community lookup, internal telemetry, regulatory enrichment).
The synthetic dataset covers layers 2 and 3 directly:

| Layer | Coverage |
|---|---|
| Layer 1 — official complaint base (FCC/FTC) | Out of scope (no public Estonian equivalent). Acknowledged limitation. |
| Layer 2 — community signal | Approximated by `community_flagged` class. |
| Layer 3 — internal telemetry | All call-event + post-call behavior columns. |
| Layer 4 — numbering / regulatory enrichment | Caller numbers follow Estonian numbering plan (`+372 5xxxxxxx` mobile, `+372 6xxxxxx` short bank lines); ready for prefix-based features in Sprint 3. |
