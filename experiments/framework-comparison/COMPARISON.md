# COPF-217 - Application/API Framework Comparison

## Evaluation Status

**Status:** Provisional recommendation - team review and architecture decision pending  
**Evaluation date:** 8 August 2026  
**Initial evaluator:** Christopher Milliner  
**Jira issue:** COPF-217  
**Technology category:** Application / API framework

This evaluation compares FastAPI and Flask as candidate application/API frameworks for RAVIN.

The comparison provides technical evidence and a provisional recommendation only. It does not constitute final team approval or a final Architecture Baseline technology decision.

## Architecture Responsibility

The application/API framework is expected to support the RAVIN application boundary, including:

- receiving application requests;
- validating question input;
- returning stable HTTP/JSON responses;
- translating invalid input and technical failures into controlled outcomes;
- supporting modular service boundaries;
- integrating with later retrieval, grounding, citation and logging components;
- supporting repeatable local development and automated testing.

The final framework is not responsible for implementing the complete RAG pipeline by itself.

## Candidates

| Candidate | Version evaluated | Runtime | Local server used in PoC |
|---|---:|---:|---|
| FastAPI | 0.141.1 | Python 3.12.2 | Uvicorn 0.52.1 |
| Flask | 3.1.3 | Python 3.12.2 | Flask development server / Werkzeug 3.1.8 |

## Proof-of-Concept Scope

Both candidates were evaluated using the same deliberately small vertical slice.

### Capability 1 - Health endpoint

`GET /health`

Expected response:

```json
{
  "status": "healthy",
  "service": "ravin"
}
```

Expected status:

`HTTP 200`

### Capability 2 - Question validation

`POST /questions/validate`

Valid request:
```json
{
  "question": "What is the special consideration policy?"
}
```

Expected successful response:
```json
{
  "valid": true,
  "question": "What is the special consideration policy?"
}
```

Invalid empty or whitespace-only questions must be rejected using a controlled validation response.

## PoC Results
| Test                              | FastAPI   | Flask                       |
| --------------------------------- | --------- | --------------------------- |
| Local Python 3.12 execution       | PASS      | PASS                        |
| Dependency installation           | PASS      | PASS                        |
| Application startup               | PASS      | PASS                        |
| `GET /health`                     | PASS      | PASS                        |
| HTTP 200 health response          | PASS      | PASS                        |
| Expected health JSON              | PASS      | PASS                        |
| Valid question request            | PASS      | PASS                        |
| Empty question rejected           | PASS      | PASS                        |
| Whitespace-only question rejected | PASS      | PASS                        |
| Automated tests                   | PASS      | PASS                        |
| External paid service required    | NO        | NO                          |
| Interactive API documentation     | AVAILABLE | NOT PROVIDED BY MINIMAL POC |

The observed pytest execution times are not treated as framework performance benchmarks. The test suites are too small and were not executed under controlled benchmarking conditions.

Setup duration was not formally measured and is therefore not scored as measured timing evidence.

## Implementation Observations
### FastAPI
FastAPI used a Pydantic request model to describe and validate the question contract.

The application declared the input requirement through the request model rather than implementing each validation check directly inside the endpoint.

The declared model was also represented in the automatically generated OpenAPI/API documentation.

FastAPI required a separate ASGI server for local execution. Uvicorn was used for this proof of concept.

The initial testing configuration used HTTPX and produced a Starlette deprecation warning. HTTPX was replaced with HTTPX2 and the complete test was rerun successfully without the warning.

### Flask
Flask produced a compact implementation of the health endpoint and supplied a built-in application test client.

The question-validation endpoint required explicit application logic to:

- obtain the JSON request body;
- confirm that the body was a JSON object;
- obtain the question field;
- validate its type;
- reject empty or whitespace-only values;
- construct the controlled validation response.

This provides greater explicit control but introduces additional application-level validation code for the tested API contract.

The minimal Flask implementation did not automatically expose an interactive OpenAPI documentation interface.

## Mandatory Eligibility Gates
| Gate                                | FastAPI | Flask | Evidence / Reasoning|
| ----------------------------------- | ------- | ----- | ------------------- |
| G-01 Baseline cost                  | PASS    | PASS  | Both operated locally without mandatory recurring cost or hosted infrastructure.|
| G-02 Local operation                | PASS    | PASS  | Both were installed, started and tested locally on the development device.                                                                                                                                    |
| G-03 Core capability fit            | PASS    | PASS  | Both implemented the required API endpoints and controlled validation behaviour.                                                                                                                              |
| G-04 Data and privacy boundary      | PASS    | PASS  | Neither framework requires external transmission of policy or question data for local execution.                                                                                                              |
| G-05 Licensing and use rights       | PASS    | PASS  | FastAPI is MIT licensed. Flask is BSD-3-Clause licensed. Both permit the intended local capstone development use.                                                                                             |
| G-06 Team and timeframe feasibility | PASS    | PASS  | Both candidates were successfully learned and implemented within the time-boxed PoC. Team-wide capability remains to be reviewed.                                                                             |
| G-07 Environment independence       | PASS    | PASS  | Neither candidate required university infrastructure, credentials or a hosted provider.                                                                                                                       |
| G-08 Testability                    | PASS    | PASS  | Repeatable pytest-based tests were executed locally for both candidates.                                                                                                                                      |
| G-09 Failure control                | PASS    | PASS  | Controlled validation errors were demonstrated. Both frameworks provide mechanisms for application-level error/exception handling. Full downstream dependency-failure translation remains to be tested later. |
| G-10 Service continuity / fallback  | PASS    | PASS  | Both operate locally and do not depend on a hosted framework service.                                                                                                                                         |

No mandatory gate currently excludes either candidate.

## Weighted Evaluation

Scoring uses the Architecture Baseline 0-5 scale:
0 - Not met
1 - Poor
2 - Limited
3 - Acceptable
4 - Strong
5 - Excellent

Weighted criterion score:
`(candidate score / 5) × criterion weight`

| ID    | Criterion                                | Weight | FastAPI score | FastAPI weighted | Flask score | Flask weighted | Confidence | Evidence / rationale                                                                                                                                                                                              |
| ----- | ---------------------------------------- | -----: | ------------: | ---------------: | ----------: | -------------: | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| EC-01 | Project scope and complexity             |     12 |             5 |             12.0 |           4 |            9.6 | High       | Both satisfy the API responsibility. FastAPI required less custom request-validation glue in the tested contract.                                                                                                 |
| EC-02 | Delivery timeframe                       |     10 |             4 |              8.0 |           4 |            8.0 | Medium     | Both reached a working, testable vertical slice. Exact setup duration was not formally measured.                                                                                                                  |
| EC-03 | Team capability                          |     10 |             3 |              6.0 |           3 |            6.0 | Low        | One team member completed the PoCs. Team-wide familiarity and learning effort have not yet been evaluated.                                                                                                        |
| EC-04 | Sponsor and product constraints          |     10 |             5 |             10.0 |           5 |           10.0 | High       | Both support the confirmed local-first prototype model without requiring ordinary-user authentication or production infrastructure.                                                                               |
| EC-05 | Local environment and deployment fit     |     12 |             5 |             12.0 |           5 |           12.0 | High       | Both installed and operated successfully in isolated Python 3.12 virtual environments on a team-owned Windows device.                                                                                             |
| EC-06 | Security, privacy and governance         |     14 |             3 |              8.4 |           3 |            8.4 | Medium     | Both can operate locally without mandatory external data transmission. Broader secret management, logging minimisation and security controls remain application-level work and were not fully tested in this PoC. |
| EC-07 | Maintainability and replaceability       |     10 |             4 |              8.0 |           4 |            8.0 | Medium     | Both support modular Python development and can sit behind the technology-neutral application boundary. Larger-scale modularity has not yet been demonstrated.                                                    |
| EC-08 | Testability and development quality      |      8 |             5 |              8.0 |           5 |            8.0 | High       | Both supported repeatable automated pytest tests without a live external service.                                                                                                                                 |
| EC-09 | Integration and data compatibility       |      8 |             5 |              8.0 |           4 |            6.4 | High       | Both exchange normal JSON successfully. FastAPI's typed request models and generated API schema provide a stronger fit for explicit application contracts in the current architecture.                            |
| EC-10 | Cost, licensing and service availability |      6 |             5 |              6.0 |           5 |            6.0 | High       | Both are locally available permissively licensed frameworks with no mandatory hosted service or recurring framework cost.                                                                                         |

### Weighted Totals
| Candidate   | Weighted score |
| ----------- | -------------: |
| **FastAPI** | **86.4 / 100** |
| **Flask**   | **82.4 / 100** |

Difference:

**4.0 weighted points**

Both candidates exceed the normal 70/100 recommendation threshold and neither scores below 3 for EC-01, EC-05 or EC-06.

The candidates are within five weighted points. The Architecture Baseline therefore requires proof-of-concept evidence before selection. The comparative PoCs documented under COPF-217 satisfy that requirement.

## Category-Specific Application/API Checks
| Check                                  | FastAPI                                                 | Flask                                                                         |
| -------------------------------------- | ------------------------------------------------------- | ----------------------------------------------------------------------------- |
| Request validation                     | Strong native fit through typed/Pydantic request models | Supported through explicit application logic or additional validation tooling |
| Controlled error contracts             | Supported                                               | Supported                                                                     |
| Modular service boundaries             | Supported                                               | Supported                                                                     |
| Automated testing                      | Demonstrated                                            | Demonstrated                                                                  |
| Configuration                          | Supported but not materially evaluated in PoC           | Supported but not materially evaluated in PoC                                 |
| Logging                                | Supported but not materially evaluated in PoC           | Supported but not materially evaluated in PoC                                 |
| Local execution                        | Demonstrated                                            | Demonstrated                                                                  |
| API schema / interactive documentation | Automatically available in tested PoC                   | Not supplied by the minimal tested configuration                              |

## Evidence Gaps and Limitations
The following areas were not sufficiently tested to justify high-confidence differentiation:

- team-wide framework familiarity;
- production or sustained-load performance;
- RAG-service integration;
- retrieval-service integration;
- persistent storage;
- application logging implementation;
- secret/configuration management;
- downstream dependency-failure translation;
- production deployment;
- concurrency under realistic chatbot workloads.

These gaps do not currently block a provisional framework recommendation because the capstone baseline is local-first and the framework is intended to remain behind replaceable service boundaries.

They remain review triggers if later implementation exposes material complexity or reliability problems.

## Provisional Recommendation
**Recommended FastAPI as the provisional application/API framework for the RAVIN reference implementation, subject to team review and a formal architecture decision.**

FastAPI and Flask both passed the mandatory gates and both successfully implemented the controlled PoC.

FastAPI is provisionally preferred because the current RAVIN architecture is API-oriented and requires explicit question validation, stable application response contracts, controlled errors and clear boundaries between the web interface and later retrieval/RAG services.

The PoC demonstrated that FastAPI can express part of this contract through typed request models with automatic validation and generated API schema/documentation, reducing custom validation glue for the tested capability.

FastAPI's weighted result is:

**86.4 / 100**

Flask's weighted result is:

**82.4 / 100**

The numerical difference is small and is not the sole basis for the recommendation.

## Alternative Candidate
Flask remains a viable fallback.

It passed every mandatory gate and all implemented automated tests.

Its strengths in this PoC include:

- minimal framework structure;
- straightforward local execution;
- built-in test client;
- explicit application control;
- low conceptual overhead for a small web service.

Flask was not provisionally preferred because the tested RAVIN API contract required more manual request-validation and response-handling logic, while FastAPI's API-oriented model aligned more directly with the responsibilities already defined for the application/API layer.

## Risks and Mitigations
### FastAPI dependency stack
FastAPI introduces supporting dependencies including Starlette and Pydantic and uses an ASGI server such as Uvicorn.

**Mitigation:** Pin tested direct dependencies, retain a resolved dependency snapshot, run automated regression tests and keep the RAVIN application services separated from framework-specific route code.

### Framework lock-in
Request models and route definitions can become tightly coupled to a selected framework.

**Mitigation:** Keep retrieval, generation, grounding, citation and logging logic in framework-independent service modules and treat FastAPI as an application adapter.

### Team capability
Only one team member has completed the comparative PoCs.

**Mitigation:** Have another team member review the implementation and evaluation before final selection.

### Untested operational behaviour
Logging, configuration, downstream failures and realistic workload behaviour have not yet been demonstrated.

**Mitigation:** Add these capabilities incrementally as later application slices and retain the architecture re-evaluation triggers.

## Fallback / Exit Path
If later implementation demonstrates that FastAPI introduces unacceptable complexity, incompatibility or maintainability problems, the application/API boundary can be reimplemented using Flask or another evaluated framework.

The retrieval, generation, grounding, citation and logging services should remain separated from framework-specific route handling so framework replacement does not require redesigning the complete RAG pipeline.

## Re-evaluation Triggers
Re-evaluate the framework decision if:

- Product Owner scope materially changes;
- local performance or reliability becomes unacceptable;
- security, privacy, licensing or governance concerns emerge;
- framework or dependency support materially changes;
- later RAG integration exposes excessive framework coupling;
- automated testing becomes difficult;
- deployment requirements materially change;
- another implementation demonstrates a significant maintainability advantage.

## Evidence
### Local proof-of-concept commits
`06c2330` - COPF-217: add FastAPI framework proof of concept
`3f3b063` - COPF-217: add Flask framework proof of concept
`d69af71` - COPF-217: extend FastAPI PoC with request validation
`5a5dcdb` - COPF-217: extend Flask PoC with request validation

### Project evidence
- Architecture Baseline v0.1 - framework/service evaluation criteria
- COPF-113 - Framework and Service Evaluation Criteria
- COPF-110 - Chatbot Retrieval and Citation Flow
- Sponsor Discovery Pack v1.0
- COPF-217 Jira task and PoC evidence

### External framework evidence
FastAPI official documentation and repository:

- Request body validation and Pydantic models
- OpenAPI/API documentation
- Testing with TestClient
- Error and exception handling
- MIT licence

Flask official documentation and repository:

- Testing utilities and application test client
- Error handlers and controlled API errors
- Configuration handling
- BSD-3-Clause licence

## Decision Status
**No final framework selection has been made by this document.**

The provisional recommendation should be:

1. peer reviewed;
2. discussed with the project team;
3. independently scored by another team member where practical; and
4. recorded as a Jira Architecture Decision if the team adopts FastAPI.

Until that occurs, FastAPI and Flask remain evaluated candidates and RAVIN remains a reference/proof-of-concept repository.