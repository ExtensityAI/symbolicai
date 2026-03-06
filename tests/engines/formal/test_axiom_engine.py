import pytest

from symai import Interface
from symai.backend.settings import SYMAI_CONFIG

pytestmark = pytest.mark.skipif(
    SYMAI_CONFIG.get("FORMAL_ENGINE") != "axiom" or not SYMAI_CONFIG.get("FORMAL_ENGINE_API_KEY"),
    reason="Axiom tests require FORMAL_ENGINE=axiom and a valid FORMAL_ENGINE_API_KEY",
)


LEAN4_VALID = """
theorem hello_world (a b : Prop) (ha : a) (hb : b) : a ∧ b := by
  exact ⟨ha, hb⟩
""".strip()

LEAN4_INVALID = """
theorem broken (a : Prop) : a := by
  sorry_not_a_tactic
""".strip()

LEAN4_THEOREM = """
theorem my_zero_add (n : Nat) : 0 + n = n := by
  omega
""".strip()

LEAN4_FALSE_THEOREM = """
theorem false_claim (n : Nat) : n + 1 = n := by
  omega
""".strip()


@pytest.fixture
def axiom():
    return Interface("axiom")


def test_axiom_check(axiom):
    result = axiom(LEAN4_VALID, tool="check")
    assert result.raw["okay"] is True


def test_axiom_verify_proof(axiom):
    formal_statement = "theorem my_zero_add (n : Nat) : 0 + n = n := by sorry"
    result = axiom(
        LEAN4_THEOREM,
        tool="verify_proof",
        config={"formal_statement": formal_statement},
    )
    assert result.raw["okay"] is True


def test_axiom_extract_theorems(axiom):
    result = axiom(LEAN4_THEOREM, tool="extract_theorems")
    assert "documents" in result.raw


def test_axiom_disprove(axiom):
    result = axiom(LEAN4_FALSE_THEOREM, tool="disprove")
    assert result.raw is not None


def test_axiom_check_invalid(axiom):
    result = axiom(LEAN4_INVALID, tool="check")
    assert result.raw["okay"] is False
