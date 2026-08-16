# RAVIN Response Behaviour Classification Rubric

## 1. Purpose

This rubric defines how evaluation questions are classified according to the response behaviour RAVIN should exhibit.

The purpose is to ensure evaluation reflects the behaviour expected from the production application rather than forcing every user question into a single retrieval-ranking test.

The rubric must be applied independently of current retrieval performance. A question must not be reclassified simply because a retrieval experiment succeeds or fails on it.

The supported response behaviours are:

- `DIRECT_ANSWER`
- `GROUNDED_OVERVIEW`
- `CLARIFY`
- `NO_GROUNDED_ANSWER`

---

## 2. Core Principles

### 2.1 Grounding

RAVIN must base substantive policy answers on authoritative policy evidence contained in the approved corpus.

User feedback can indicate what a user intended, but user feedback is not authoritative evidence about what a policy says.

### 2.2 Accuracy

RAVIN should not select one interpretation of an ambiguous question when alternative reasonable interpretations would materially change the answer.

### 2.3 Controlled Abstention

When sufficient authoritative evidence is unavailable, RAVIN should state that it cannot provide a grounded answer rather than generate unsupported information.

### 2.4 Broad Questions Are Not Automatically Ambiguous

A broad question may still have a clear intent and support a useful grounded overview.

A question should be classified as `CLARIFY` only when different reasonable interpretations would materially affect the answer.

### 2.5 Evaluation Independence

Questions must be classified using this rubric before considering whether the current retrieval implementation ranks their expected evidence correctly.

Classification must not be changed merely to improve an experimental score.

---

## 3. Classification Decision Process

Apply the following questions in order.

### Step 1 - Is sufficient authoritative evidence available?

If the approved policy corpus does not contain sufficient evidence to provide a useful grounded response:

`NO_GROUNDED_ANSWER`

If sufficient evidence exists, continue to Step 2.

### Step 2 - Are there multiple reasonable interpretations?

Determine whether two or more plausible interpretations of the user's question would produce materially different:

- requirements
- eligibility conditions
- consequences
- processes
- obligations
- actions
- recommendations about what the user should do next

If no materially different interpretations exist, continue to Step 3.

If materially different interpretations exist, determine whether a common high-level answer can still be provided safely.

If no safe common answer exists:

`CLARIFY`

If a useful common answer can be provided without making an unsafe assumption:

`GROUNDED_OVERVIEW`

RAVIN may then offer more specific follow-up options.

### Step 3 - Does the answer require synthesis?

If the user's intent is clear but a useful answer requires synthesising several related policy propositions or stages:

`GROUNDED_OVERVIEW`

If one clear proposition or coherent evidence target directly answers the question:

`DIRECT_ANSWER`

---

## 4. Behaviour Definitions

### 4.1 DIRECT_ANSWER

Use `DIRECT_ANSWER` when:

- the user's intent is sufficiently specific
- a clear grounded answer exists
- one coherent evidence target, or equivalent evidence locations expressing the same answer, directly supports the response
- clarification is not necessary to avoid materially changing the answer

Example:

> What happens at Academic Progression Stage Three?

Expected behaviour:

RAVIN retrieves the relevant Stage Three evidence and provides a grounded answer with citations.

Evaluation focus:

- Top-1 retrieval accuracy
- Hit@K
- Mean Reciprocal Rank
- evidence correctness

---

### 4.2 GROUNDED_OVERVIEW

Use `GROUNDED_OVERVIEW` when:

- the user's overall intent is clear
- the question is broad
- a useful answer requires several related pieces of evidence
- RAVIN can safely provide a high-level answer without first requiring clarification

Example:

> What is the academic progression process?

Expected behaviour:

RAVIN provides a grounded overview of the progression process using the relevant policy evidence and may offer follow-up options for individual stages.

Evaluation focus:

- required evidence coverage
- completeness of major policy concepts
- grounding
- citation coverage
- absence of unsupported claims

A single arbitrary chunk should not necessarily be required to rank first if several pieces of evidence are legitimately required to construct the overview.

---

### 4.3 CLARIFY

Use `CLARIFY` when:

- multiple reasonable interpretations exist
- the interpretations would materially change the answer
- selecting one interpretation without user input risks providing incorrect or misleading information
- a single safe common answer is insufficient

Example:

> What admission requirements apply to me?

Possible interpretations may depend on:

- course
- applicant category
- international or domestic status
- prior qualifications
- course-specific entry requirements
- Single Subject enrolment
- other relevant circumstances

Expected behaviour:

RAVIN should explain that the question depends on additional information and present grounded clarification options.

After the user chooses an option, retrieval should be rerun using the clarified intent.

Evaluation focus:

- correct ambiguity detection
- appropriateness of clarification
- grounded clarification options
- correct retrieval after clarification

---

### 4.4 NO_GROUNDED_ANSWER

Use `NO_GROUNDED_ANSWER` when:

- the approved corpus contains insufficient evidence to answer the question
- answering would require unsupported assumptions or external knowledge
- RAVIN cannot provide a reliable grounded response

Expected behaviour:

RAVIN should use the controlled no-grounded-answer response and must not fabricate an answer.

Evaluation focus:

- correct abstention
- false-answer rate
- absence of unsupported factual claims

For evaluation data:

`expected_evidence` must be empty.

---

## 5. Material Difference Rule

Two interpretations are materially different when they would cause RAVIN to provide meaningfully different:

- requirements
- eligibility outcomes
- consequences
- policy processes
- obligations
- rights
- actions

Differences in wording alone are not material.

Multiple evidence locations that communicate substantially the same answer do not automatically make a question ambiguous.

---

## 6. Evidence Requirements

### DIRECT_ANSWER

`expected_evidence` is required.

### GROUNDED_OVERVIEW

`expected_evidence` is required.

Multiple evidence locations may be required to represent the major components of the overview.

### CLARIFY

`expected_evidence` is required.

The evidence should establish the legitimate policy interpretations or domains from which clarification options are derived.

A future evaluation schema may explicitly represent expected clarification options or intents.

### NO_GROUNDED_ANSWER

`expected_evidence` must be empty.

---

## 7. Classification Examples

### Example A

Question:

> Who approves changes to academic dress?

Classification:

`DIRECT_ANSWER`

Reason:

The intent is specific and a clear policy proposition directly answers the question.

### Example B

Question:

> What is the academic progression process?

Classification:

`GROUNDED_OVERVIEW`

Reason:

The intent is clear, but a useful response requires synthesising multiple progression stages and related policy information.

### Example C

Question:

> What admission requirements apply to me?

Classification:

`CLARIFY`

Reason:

Requirements may materially differ depending on the user's circumstances. Selecting one interpretation without clarification could produce an incorrect answer.

### Example D

Question:

> What parking fine will I receive for parking outside the library?

Classification:

`NO_GROUNDED_ANSWER` if the approved RAVIN corpus contains no sufficient policy evidence addressing the question.

Reason:

RAVIN must not answer from unsupported knowledge.

---

## 8. User Clarification and Feedback

User clarification selections are evidence of user intent, not evidence of policy truth.

Example:

A user asks:

> What happens if I am failing?

RAVIN may offer:

- overall academic progression process
- Stage One
- Stage Two
- Stage Three and possible exclusion
- review or appeal options

If the user selects:

> Stage Three and possible exclusion

the selection may be recorded as clarification feedback.

The selection does not establish what the policy says. The authoritative answer must still be grounded in the approved policy corpus.

---

## 9. Continuous Improvement

Clarification and feedback interactions may be logged for later review.

Relevant information may include:

- original question
- detected response behaviour
- ambiguity reason
- clarification options presented
- option selected by the user
- clarified query
- retrieved evidence
- answer outcome
- user feedback
- retrieval configuration
- model version
- application version

Application administrators may review these logs.

Approved examples may later be promoted into validated evaluation data.

User feedback must not automatically modify retrieval behaviour, evaluation labels, policy facts, or model behaviour without review and validation.

---

## 10. Quality-Gate Principle

The validated `DIRECT_ANSWER` retrieval set retains the project requirement of at least:

**95% Top-1 retrieval accuracy**

Other response behaviours should be evaluated using behaviour-appropriate metrics rather than being mixed into the Direct Answer Top-1 denominator.

Future behaviour-specific quality measures may include:

- Grounded Overview evidence coverage
- Clarification detection accuracy
- clarification-option quality
- No-Grounded-Answer abstention accuracy
- unsupported-answer rate

---

## 11. Future CI/CD Integration

The evaluation framework should eventually be integrated into the project CI/CD pipeline.

Relevant code changes should trigger automated evaluation.

A future pipeline may perform:

1. unit tests
2. integration tests
3. validated Direct Answer retrieval evaluation
4. behaviour evaluation
5. regression comparison
6. quality-gate enforcement

Changes that fail required validated quality gates should not automatically be accepted for deployment.

Evaluation failures and regressions should be reported for developer and application administrator review.

---

## 12. Dataset Classification Procedure

When reviewing an evaluation dataset:

1. Apply this rubric without reference to current retrieval success or failure.
2. Assign one response behaviour to every question.
3. Record a short classification reason.
4. Review uncertain classifications separately.
5. Version the dataset after classification changes.
6. Preserve all previous experiment evidence unchanged.
7. Re-establish baselines before beginning a new retrieval candidate experiment.

This process prevents benchmark changes from being made solely to improve experimental performance.