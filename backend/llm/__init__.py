"""
Provide language-model contracts and adapters used by RAVIN generation.

The LLM package isolates model-specific generation technology from the
evidence-first business pipeline. RAVIN uses these providers only after
deterministic components have selected a grounded generation behaviour.

Language models in this package do not control intent classification,
evidence sufficiency, routing, or release validation.
"""
