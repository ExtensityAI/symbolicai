# Formal Engine (Axiom)

The Axiom engine provides Lean4 formal verification via the cloud-based Axle SDK. It supports proof checking, theorem extraction, proof repair, and more.

## Setup

1. Install the lean extras:
   ```bash
   pip install symbolicai[lean]
   ```

2. Set your API key and engine in `symai.config.json`:
   ```json
   {
     "FORMAL_ENGINE_API_KEY": "your-axle-api-key",
     "FORMAL_ENGINE": "axiom"
   }
   ```

## Basic Usage

```python
from symai import Interface

axiom = Interface('axiom')

# Check if Lean4 code is valid
result = axiom("""
theorem hello (a b : Prop) (ha : a) (hb : b) : a ∧ b := by
  exact ⟨ha, hb⟩
""", tool="check")
print(result.raw['okay'])  # True

# Verify a proof against a formal statement
result = axiom(proof_code, tool="verify_proof", config={
    "formal_statement": "theorem my_thm (n : Nat) : 0 + n = n := by sorry"
})

# Extract theorems from Lean4 code
result = axiom(lean_code, tool="extract_theorems")
print(result.raw['documents'])
```

## Using with `@contract`

Each contract's post-condition can delegate LLM outputs to Axiom's Lean4 engine for formal verification before accepting them. The `@contract` decorator calls the LLM (using the `prompt` property and the data models), then runs `post()` to validate. If `post()` raises, the contract feeds the error back to the LLM and retries.

```python
from pydantic import Field

from symai import Expression, Interface
from symai.models.base import LLMDataModel
from symai.strategy import contract


class InpProveTheorem(LLMDataModel):
    formal_statement: str = Field(
        description=(
            "A complete Lean4 theorem signature with `sorry` as the proof body. "
            "Example: 'theorem my_thm (n : Nat) : 0 + n = n := by sorry'. "
            "The LLM must preserve this exact name and signature."
        ),
    )
    hint: str = Field(
        description=(
            "Optional tactic hint for the LLM, e.g. 'Use omega' or 'Try induction on n'. "
            "Left empty if the LLM should figure out the proof strategy on its own."
        ),
    )


class OutpProveTheorem(LLMDataModel):
    proof: str = Field(
        description=(
            "The complete Lean4 theorem with a valid tactic proof replacing `sorry`. "
            "Must use the EXACT same theorem name and type signature from the formal_statement. "
            "Must NOT include any import statements — the environment already has Mathlib."
        ),
    )


@contract(
    post_remedy=True,
    remedy_retry_params={"tries": 8, "delay": 0.5, "max_delay": 0.0,
                         "backoff": 0.5, "jitter": 0.0, "graceful": False},
)
class ProveTheorem(Expression):
    """LLM writes a Lean4 proof; Axiom verifies it in post()."""

    def __init__(self, formal_statement: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._formal_statement = formal_statement

    @property
    def prompt(self) -> str:
        return (
            "You are a Lean4 proof assistant.\n\n"
            "Given a formal statement (with `sorry` as placeholder), replace `sorry` with a real proof.\n"
            "You MUST use the EXACT same theorem name and signature from the formal_statement.\n"
            "Return ONLY the complete theorem with the proof filled in, no markdown fences.\n"
            "Do NOT import anything — the environment already has Mathlib available.\n"
        )

    def post(self, output: OutpProveTheorem) -> bool:
        axiom = Interface("axiom")
        result = axiom(
            output.proof,
            tool="verify_proof",
            config={"formal_statement": self._formal_statement},
        )
        if not result.raw["okay"]:
            errors = result.raw.get("lean_messages", {}).get("errors", [])
            msg = f"Proof verification failed. Lean errors: {errors}"
            raise ValueError(msg)
        return True

    def forward(self, _input: InpProveTheorem, **_kwargs) -> OutpProveTheorem:
        if not self.contract_successful:
            raise self.contract_exception or ValueError("Contract validation failed.")
        return self.contract_result
```

See [`examples/formal_verification.ipynb`](../../../examples/formal_verification.ipynb) for a full runnable example.

## Tools Reference

| Tool | Description |
|---|---|
| `check` | Check if Lean4 code is valid |
| `verify_proof` | Verify a proof against a formal statement |
| `extract_theorems` | Extract theorems from Lean4 code |
| `rename` | Rename identifiers |
| `theorem2lemma` | Convert theorems to lemmas |
| `theorem2sorry` | Replace theorem proofs with sorry |
| `merge` | Merge multiple Lean4 documents (takes `documents` arg) |
| `simplify_theorems` | Simplify theorem statements |
| `repair_proofs` | Attempt to repair broken proofs |
| `have2lemma` | Convert have expressions to lemmas |
| `have2sorry` | Replace have proofs with sorry |
| `sorry2lemma` | Replace sorry placeholders with proofs |
| `disprove` | Attempt to disprove a theorem |
| `normalize` | Normalize Lean4 code |
