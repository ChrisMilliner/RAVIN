# RAVIN Preliminary Baseline v1.3 Response Behaviour Review

## 1. Purpose

This document records the response behaviour classification review of the 30 questions in:

`evaluation/retrieval_baseline.json`

Dataset:

`RAVIN Preliminary Retrieval Development Baseline v1.3`

Status:

`preliminary-not-gold-standard`

The review was completed using the frozen:

`docs/evaluation/response-behavior-classification-rubric.md`

The classifications were determined from:

- question wording
- existing expected evidence
- existing question notes
- the approved response behaviour classification rubric

Retrieval rankings, retrieval experiment results, and candidate performance were not used to determine classifications.

This separation is intended to prevent benchmark labels from being changed merely to improve retrieval scores.

---

## 2. Review Result

| Question | Behaviour | Confidence | Classification reason |
|---|---|---|---|
| RB001 | `DIRECT_ANSWER` | HIGH | Specific governance question with clear answer-bearing evidence. Multiple evidence locations express the same answer rather than materially different interpretations. |
| RB002 | `GROUNDED_OVERVIEW` | MEDIUM | The question broadly asks what happens when academic progress is unsatisfactory. A useful response may need to summarise monitoring, progression stages, interventions and possible consequences. |
| RB003 | `GROUNDED_OVERVIEW` | MEDIUM | The question asks generally about admission requirements. A safe grounded overview can explain general and course-specific requirements without first requiring personal clarification. |
| RB004 | `DIRECT_ANSWER` | HIGH | Specific question asking where an operational requirement is recorded. |
| RB005 | `DIRECT_ANSWER` | HIGH | Specific yes/no question concerning one academic dress provision. |
| RB006 | `DIRECT_ANSWER` | HIGH | Narrow factual question with a specific answer. |
| RB007 | `DIRECT_ANSWER` | HIGH | Specific procedure question concerning a request to change approved academic dress. |
| RB008 | `DIRECT_ANSWER` | HIGH | Specific condition and consequence concerning a student passing every enrolled subject. |
| RB009 | `DIRECT_ANSWER` | HIGH | Specific Academic Progression Stage Three procedural deadline. |
| RB010 | `DIRECT_ANSWER` | HIGH | Specific Academic Progression Stage Two enrolment-load requirement. |
| RB011 | `DIRECT_ANSWER` | HIGH | The scenario is precise: a course transfer while an enrolment sanction remains active. |
| RB012 | `DIRECT_ANSWER` | HIGH | Two closely related subquestions concern fixed-term staff promotion eligibility and the effect of promotion on the appointment. |
| RB013 | `DIRECT_ANSWER` | HIGH | Specific request for the three defined domains of academic work. |
| RB014 | `DIRECT_ANSWER` | HIGH | Specific qualification requirement for academic promotion. |
| RB015 | `DIRECT_ANSWER` | HIGH | Specific Level D referee and external assessor requirements. Multiple evidence locations may contribute to one coherent answer. |
| RB016 | `DIRECT_ANSWER` | HIGH | Specific procedural deadline for reconsideration. |
| RB017 | `DIRECT_ANSWER` | HIGH | Straight policy-scope question. |
| RB018 | `DIRECT_ANSWER` | HIGH | Specific safeguards applying to teaching staff who do not yet meet Professional Equivalence. |
| RB019 | `DIRECT_ANSWER` | HIGH | Specific record-keeping requirement. |
| RB020 | `DIRECT_ANSWER` | HIGH | Specific scholarly-activity expectation. |
| RB021 | `DIRECT_ANSWER` | HIGH | Definition question targeting one defined term. |
| RB022 | `DIRECT_ANSWER` | HIGH | Specific policy-scope inclusion or exclusion question. |
| RB023 | `DIRECT_ANSWER` | HIGH | Specific distinction between satisfying entry requirements and being guaranteed an offer. |
| RB024 | `DIRECT_ANSWER` | HIGH | Specific governance responsibility. |
| RB025 | `GROUNDED_OVERVIEW` | MEDIUM | The question broadly asks what admissions information should be available. The expected evidence represents multiple transparency obligations that can be combined into a useful summary. |
| RB026 | `GROUNDED_OVERVIEW` | HIGH | The question asks how La Trobe supports participation for disadvantaged or underrepresented applicants and naturally requires summarising multiple support mechanisms. |
| RB027 | `DIRECT_ANSWER` | HIGH | Specific eligibility question concerning an international student on a student visa enrolling in a Single Subject. |
| RB028 | `DIRECT_ANSWER` | HIGH | The question has one precise purpose: identify the recognised types of disadvantage relevant to SEAS adjustments. |
| RB029 | `DIRECT_ANSWER` | HIGH | Specific offer-deferment timeframe. |
| RB030 | `DIRECT_ANSWER` | HIGH | Specific course-transfer outcome notification timeframe. |

---

## 3. Classification Distribution

The review produced:

- `DIRECT_ANSWER`: 26 questions
- `GROUNDED_OVERVIEW`: 4 questions
- `CLARIFY`: 0 questions
- `NO_GROUNDED_ANSWER`: 0 questions

The four questions classified as `GROUNDED_OVERVIEW` are:

- RB002
- RB003
- RB025
- RB026

---

## 4. Review Observations

### 4.1 RB002

Question:

> What happens when a student is not making satisfactory academic progress?

The question asks broadly about the response to unsatisfactory academic progress.

A useful grounded answer may need to describe several related concepts, including monitoring, progression stages, interventions, support and possible consequences.

The question therefore fits `GROUNDED_OVERVIEW` rather than requiring one arbitrary evidence chunk to be treated as the sole correct Top-1 result.

### 4.2 RB003

Question:

> What are the admission requirements for university applicants?

The question is broad but does not necessarily require clarification.

RAVIN can safely provide a general grounded overview explaining that applicants must satisfy applicable general and course-specific admission requirements and that particular requirements may vary.

This is different from a personalised question such as:

> What admission requirements apply to me?

The personalised version may require `CLARIFY` because the answer could materially depend on the user's circumstances.

RB003 is therefore classified as `GROUNDED_OVERVIEW`.

### 4.3 RB025

Question:

> What admissions information should be available to prospective students before they apply or accept an offer?

The existing expected evidence already identifies multiple transparency-related policy propositions.

A useful answer should combine the relevant obligations rather than require one fragment to represent the complete response.

RB025 is therefore classified as `GROUNDED_OVERVIEW`.

### 4.4 RB026

Question:

> How does La Trobe support admission for people who have experienced disadvantage or are underrepresented in higher education?

The question asks how support is provided.

A useful answer should explain the overall policy position and the mechanisms used to provide that support.

RB026 is therefore classified as `GROUNDED_OVERVIEW`.

---

## 5. Dataset Coverage Limitation

This preliminary dataset was originally developed for retrieval evaluation.

It currently contains no questions classified as:

- `CLARIFY`
- `NO_GROUNDED_ANSWER`

This does not invalidate the dataset for its preliminary retrieval purpose.

However, the dataset cannot by itself validate RAVIN's future ambiguity-handling or controlled-abstention behaviour.

A later human-validated evaluation set should deliberately include representative cases for all supported response behaviours.

---

## 6. Quality Measurement Implication

The four behaviour categories should not be combined into one undifferentiated Top-1 retrieval metric.

For validated evaluation:

### DIRECT_ANSWER

Primary retrieval quality gate:

**Top-1 retrieval accuracy >= 95%**

Supporting measures may include:

- Hit@K
- Mean Reciprocal Rank
- evidence correctness

### GROUNDED_OVERVIEW

Evaluation should focus on:

- required evidence coverage
- major concept coverage
- grounding
- citation coverage
- absence of unsupported claims

### CLARIFY

Evaluation should focus on:

- ambiguity detection
- appropriateness of clarification
- grounded clarification options
- retrieval after clarified intent

### NO_GROUNDED_ANSWER

Evaluation should focus on:

- correct abstention
- false-answer rate
- absence of unsupported factual claims

---

## 7. Next Dataset Version

The next preliminary dataset version should:

1. be versioned separately from v1.3
2. preserve the v1.3 dataset history
3. explicitly record the reviewed response behaviour for all 30 questions
4. retain existing expected evidence unless independently justified otherwise
5. preserve previous experiment records unchanged
6. establish a new behaviour-aware evaluation baseline before beginning another retrieval candidate experiment

The expected next dataset version is:

`RAVIN Preliminary Retrieval Development Baseline v1.4`

This classification review does not itself establish a new retrieval accuracy result.