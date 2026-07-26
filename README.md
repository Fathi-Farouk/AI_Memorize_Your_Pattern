# AI Memorize Your Pattern

> An AI-assisted, deterministic validation system for telecom RF activation and rollback scripts.

**AI Memorize Your Pattern** is a two-phase tool that teaches an AI system to understand telecom RF script formats from confirmed examples, then converts that learned understanding into executable Python validation logic.

The generated parser is used to validate new activation and rollback script pairs, identify dangerous mismatches, and catch configuration errors before scripts reach a live network.

The core design principle is simple:

> **Use AI once to learn the format, then use deterministic Python code for every production validation.**

This provides the flexibility of AI without depending on repeated, expensive, and non-deterministic AI calls during operational checking.

---

## Project Overview

The system operates in two main phases:

### Phase 1 — AI Training

An administrator or RF expert provides:

- One confirmed-correct activation script
- Its matching confirmed-correct rollback script
- One or more independent sample pairs for validation

The multi-agent training workflow then learns:

- Site ID structure
- Cell ID structure
- Command format
- Parameter names and values
- Activation-to-rollback relationships
- Forward-filled site and cell context
- Comments and non-active lines
- Parameter reversal rules
- Vendor-specific formatting conventions

The learned specification must reach the configured confidence threshold before it can be approved.

Current training configuration:

- **Training model:** `qwen-aw-35b`
- **Confidence threshold:** `90%`
- **Maximum refinement rounds:** `10`

> Model name, threshold, and maximum rounds can be configured according to the deployment environment.

---

## Genuine Multi-Agent Architecture

The training process is implemented as a multi-agent workflow rather than a single AI request.

### Trainer Agent

The Trainer studies the confirmed activation and rollback examples and proposes a candidate pattern specification.

It attempts to identify:

- How sites and cells are declared
- How commands are separated
- How parameters are extracted
- Which fields must be compared
- How activation values should relate to rollback values
- Which lines are comments or inactive
- How context continues across multiple commands

### Teacher Agent

The Teacher independently tests the candidate pattern against separate validation samples.

It evaluates whether the learned pattern correctly extracts:

- Site IDs
- Cell IDs
- Command names
- Parameter names
- Activation values
- Rollback values
- Expected reversal relationships

The Teacher returns:

- Approval status
- Confidence score
- Per-sample extraction results
- Identified issues
- Specific feedback for the next refinement round

When the confidence score is below the configured threshold, the pattern and Teacher feedback are sent back to the Trainer.

This loop continues until:

- The confidence threshold is achieved, or
- The maximum number of training rounds is reached

### Code Generator Agent

After Teacher approval, the Code Generator converts the learned specification into executable Python parsing logic.

For example, it generates functions such as:

```python
def parse_line(line: str, context: dict) -> dict | None:
    ...
