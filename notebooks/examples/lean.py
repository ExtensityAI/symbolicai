from symai.backend.engines.lean.engine_lean4 import LeanEngine
# Example usage
if __name__ == "__main__":
    # Initialize LeanEngine
    engine = LeanEngine()

    # Sample Lean code
    code = '''
    theorem and_commutative (A B : Prop) : A ∧ B → B ∧ A :=
      fun h : A ∧ B => ⟨h.right, h.left⟩dsdsdds
    '''

    # Execute the Lean code
    print("Running LeanEngine with a sample Lean theorem...")
    results, metadata = engine.forward(code)
    
    # Check if results are valid and print the output
    if results:
        print("Results:", results[0].value['output'])
    print("Metadata:", metadata)
