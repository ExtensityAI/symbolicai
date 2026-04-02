import importlib.util
import subprocess

import pytest

from symai import Interface
from symai.backend.settings import SYMAI_CONFIG
from symai.functional import EngineRepository

# Check if requests is available (HTTP client for server)
requests_available = importlib.util.find_spec("requests") is not None


def _docker_image_exists() -> bool:
    """Check if lean4-container-image exists."""
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", "lean4-container-image"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


pytestmark = [
    pytest.mark.skipif(
        not requests_available,
        reason="Lean local tests require requests: pip install requests",
    ),
    pytest.mark.skipif(
        not _docker_image_exists(),
        reason="Lean local tests require Docker image: docker build -t lean4-container-image -f symai/backend/engines/formal/Dockerfile symai/backend/engines/formal/",
    ),
    pytest.mark.skipif(
        SYMAI_CONFIG.get("FORMAL_ENGINE") != "local",
        reason="Lean local tests require FORMAL_ENGINE=local",
    ),
]


# ---------------------------------------------------------------------------
# Basic snippets
# ---------------------------------------------------------------------------

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

LEAN4_SORRY = """
theorem sorry_proof (n : Nat) : n = n := by
  sorry
""".strip()

# ---------------------------------------------------------------------------
# Complex snippets: multi-step proofs, induction, structures, type classes
# ---------------------------------------------------------------------------

LEAN4_REWRITE = """
theorem add_right_comm (a b c : Nat) : a + b + c = a + c + b := by
  rw [Nat.add_assoc, Nat.add_comm b c, ← Nat.add_assoc]
""".strip()

LEAN4_INDUCTION = """
theorem length_reverse (l : List α) : l.reverse.length = l.length := by
  induction l with
  | nil => rfl
  | cons h t ih => simp [ih]
""".strip()

LEAN4_STRUCTURE = """
structure Point where
  x : Float
  y : Float

def Point.dist (p q : Point) : Float :=
  Float.sqrt ((p.x - q.x) ^ 2 + (p.y - q.y) ^ 2)

#check Point.dist
""".strip()

LEAN4_MATCH_EXPR = """
def fib : Nat → Nat
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

theorem fib_zero : fib 0 = 0 := rfl
theorem fib_one  : fib 1 = 1 := rfl
""".strip()

LEAN4_TYPECLASS = """
class Printable (α : Type) where
  repr : α → String

instance : Printable Nat where
  repr n := toString n

def showIt [Printable α] (x : α) : String := Printable.repr x

#eval showIt 42
""".strip()

LEAN4_DO_NOTATION = """
def safeDivMod (a b : Nat) : Option (Nat × Nat) := do
  if b == 0 then none
  else pure (a / b, a % b)

#eval safeDivMod 10 3
""".strip()

LEAN4_TYPE_ERROR = """
def bad : Nat := "hello"
""".strip()

LEAN4_INCOMPLETE_MATCH = """
def head (l : List α) : α :=
  match l with
  | h :: _ => h
""".strip()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def lean():
    iface = Interface("lean4_local")
    # Capture engine reference while modules are fully loaded (not during teardown)
    engine = EngineRepository.get("formal")
    yield iface
    engine.cleanup()


# ---------------------------------------------------------------------------
# Basic tests
# ---------------------------------------------------------------------------

def test_lean_check_valid(lean):
    result = lean(LEAN4_VALID)
    assert result.raw["status"] == "success"


def test_lean_check_invalid(lean):
    result = lean(LEAN4_INVALID)
    assert result.raw["status"] == "failure"
    assert "unknown tactic" in result.raw["output"]


def test_lean_theorem_proving(lean):
    result = lean(LEAN4_THEOREM)
    assert result.raw["status"] == "success"


def test_lean_false_theorem(lean):
    result = lean(LEAN4_FALSE_THEOREM)
    assert result.raw["status"] == "failure"
    assert "omega could not prove the goal" in result.raw["output"]


def test_lean_sorry_proof(lean):
    """sorry compiles (exit 0) but emits a warning."""
    result = lean(LEAN4_SORRY)
    assert result.raw["status"] == "success"
    assert "sorry" in result.raw["output"].lower()


# ---------------------------------------------------------------------------
# Complex tests
# ---------------------------------------------------------------------------

def test_lean_rewrite_chain(lean):
    """Multi-step rewrite using Nat.add_assoc and Nat.add_comm."""
    result = lean(LEAN4_REWRITE)
    assert result.raw["status"] == "success"


def test_lean_induction(lean):
    """Induction proof: reversing a list preserves length."""
    result = lean(LEAN4_INDUCTION)
    assert result.raw["status"] == "success"


def test_lean_structure_and_function(lean):
    """Structure definition with a method and #check."""
    result = lean(LEAN4_STRUCTURE)
    assert result.raw["status"] == "success"


def test_lean_pattern_matching_and_rfl(lean):
    """Recursive function with pattern matching + rfl proofs."""
    result = lean(LEAN4_MATCH_EXPR)
    assert result.raw["status"] == "success"


def test_lean_typeclass(lean):
    """Typeclass definition, instance, and #eval."""
    result = lean(LEAN4_TYPECLASS)
    assert result.raw["status"] == "success"
    assert "42" in result.raw["output"]


def test_lean_do_notation(lean):
    """Monadic do-notation with Option."""
    result = lean(LEAN4_DO_NOTATION)
    assert result.raw["status"] == "success"
    assert "some" in result.raw["output"].lower()


def test_lean_type_error(lean):
    """Type mismatch: assigning String to Nat."""
    result = lean(LEAN4_TYPE_ERROR)
    assert result.raw["status"] == "failure"
    # Lean capitalizes "Type mismatch" — case-insensitive check
    assert "type mismatch" in result.raw["output"].lower()


def test_lean_incomplete_match(lean):
    """Missing match case: non-exhaustive pattern."""
    result = lean(LEAN4_INCOMPLETE_MATCH)
    assert result.raw["status"] == "failure"
    assert "missing cases" in result.raw["output"].lower()
