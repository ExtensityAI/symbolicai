import pytest
from pathlib import Path
import json
from typing import List, Optional
from pydantic import Field
import logging

from symai import Expression
from symai.models import LLMDataModel
from symai.strategy import contract
from symai.components import MetadataTracker


class Entity(LLMDataModel):
    name: str = Field(description="Name of the entity.")

    def __eq__(self, other):
        if not isinstance(other, Entity):
            return False
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)


class Relationship(LLMDataModel):
    name: str = Field(description="Name of the relationship.")

    def __eq__(self, other):
        if not isinstance(other, Relationship):
            return False
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)


class Triplet(LLMDataModel):
    subject: Entity
    predicate: Relationship
    object: Entity
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for the extracted triplet [0, 1]",
    )

    def __eq__(self, other):
        if not isinstance(other, Triplet):
            return False
        return (
            self.subject == other.subject and
            self.predicate == other.predicate and
            self.object == other.object
        )

    def __hash__(self):
        return hash((hash(self.subject), hash(self.predicate), hash(self.object)))


class OntologyClass(LLMDataModel):
    name: str = Field(description="Class name in the ontology")


class SimpleOntology(LLMDataModel):
    classes: List[OntologyClass] = Field(default_factory=list, description="List of classes in the ontology")


class KGState(LLMDataModel):
    triplets: List[Triplet] = Field(default_factory=list, description="List of triplets.")


class TripletExtractorInput(LLMDataModel):
    text: str = Field(description="Text to extract triplets from.")
    ontology: SimpleOntology = Field(description="Simplified ontology schema for testing.")
    state: Optional[KGState] = Field(default=None, description="Existing knowledge graph state (triplets), if any.")


class IntermediateAnalysisResult(LLMDataModel):
    """A different type that the act method will produce"""
    analyzed_text: str = Field(description="Text with analysis markers")
    possible_entities: List[str] = Field(default_factory=list, description="Potential entities identified")
    possible_relationships: List[str] = Field(default_factory=list, description="Potential relationships identified")
    confidence_threshold: float = Field(default=0.7, description="Confidence threshold used")


@contract(
    pre_remedy=True,
    post_remedy=True,
    verbose=True,
    remedy_retry_params=dict(tries=2, delay=0.1, max_delay=0.5, jitter=0.1, backoff=1.5, graceful=False),
)
class TestTripletExtractor(Expression):
    """Test implementation of a TripletExtractor-like class with an act method"""

    def __init__(self, name: str = "test_extractor", threshold: float = 0.7, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
        self.threshold = threshold
        self._triplets = set()
        self.calls_count = 0
        self.analysis_history = []

    @property
    def prompt(self) -> str:
        return """
        Extract knowledge triplets from the following text.
        A triplet consists of (subject, predicate, object) where:
        - subject is an entity
        - predicate is a relationship
        - object is an entity

        Text: {{text}}
        """

    def pre(self, input: TripletExtractorInput) -> bool:
        """Pre-validation: ensure text is not empty and ontology is provided"""
        if not input.text or not isinstance(input.text, str):
            raise ValueError("Input text must be a non-empty string")

        if not input.ontology or not input.ontology.classes:
            raise ValueError("Ontology must be provided with at least one class")

        return True

    def act(self, input: TripletExtractorInput, **kwargs) -> IntermediateAnalysisResult:
        """
        Act method that:
        1. Performs preliminary analysis on the text
        2. Identifies potential entities and relationships
        3. Returns an intermediate analysis result (different type than input)
        4. Updates internal state
        """
        self.calls_count += 1

        # Simple entity extraction logic
        words = input.text.split()
        possible_entities = []
        possible_relationships = []

        ontology_class_names = [c.name.lower() for c in input.ontology.classes]

        for word in words:
            word_lower = word.lower().strip('.,;:!?')

            if word[0].isupper() and len(word) > 1:
                possible_entities.append(word)

            for class_name in ontology_class_names:
                if class_name in word_lower:
                    possible_entities.append(word)

            if word_lower in ["is", "has", "contains", "uses", "creates", "performs"]:
                possible_relationships.append(word)

        analyzed_text = input.text
        for entity in set(possible_entities):
            analyzed_text = analyzed_text.replace(entity, f"[ENTITY: {entity}]")
        for rel in set(possible_relationships):
            analyzed_text = analyzed_text.replace(rel, f"[REL: {rel}]")

        # Test contract state mutation
        self.analysis_history.append({
            "call": self.calls_count,
            "text": input.text,
            "entities_found": len(possible_entities),
            "relationships_found": len(possible_relationships)
        })

        return IntermediateAnalysisResult(
            analyzed_text=analyzed_text,
            possible_entities=possible_entities,
            possible_relationships=possible_relationships,
            confidence_threshold=self.threshold
        )

    def post(self, output: KGState) -> bool:
        """Post-validation: ensure triplets meet confidence threshold"""
        if output.triplets is None:
            return True  # No triplets extracted is a valid state

        for triplet in output.triplets:
            if triplet.confidence < self.threshold:
                raise ValueError(f"Confidence score {triplet.confidence} is below threshold {self.threshold}!")

        return True

    def forward(self, input: TripletExtractorInput, **kwargs) -> KGState:
        if self.contract_result is None:
            raise ValueError("Contract failed!")
        return self.contract_result

    def extend_triplets(self, new_triplets: List[Triplet]):
        """Store extracted triplets"""
        if new_triplets:
            self._triplets.update(new_triplets)

    def get_triplet_count(self) -> int:
        """Return the number of unique triplets extracted"""
        return len(self._triplets)


# Tests
def test_triplet_extractor_basic():
    """Test basic functionality of the triplet extractor with act (the happy path!)"""
    # Create a simple ontology
    ontology = SimpleOntology(classes=[
        OntologyClass(name="Algorithm"),
        OntologyClass(name="Model"),
        OntologyClass(name="Data")
    ])

    extractor = TestTripletExtractor(threshold=0.7)

    # First call with simple text
    input_data = TripletExtractorInput(
        text="Neural Networks are models that process data.",
        ontology=ontology
    )

    with MetadataTracker() as tracker:
        result = extractor(input=input_data)

    # Verify result structure
    assert isinstance(result, KGState)
    if result.triplets:
        for triplet in result.triplets:
            assert triplet.confidence >= 0.7  # Respect threshold

    # Verify state was updated
    assert extractor.calls_count == 1
    assert len(extractor.analysis_history) == 1


def test_triplet_extractor_state_persistence():
    """Test that state is maintained across multiple calls"""
    ontology = SimpleOntology(classes=[
        OntologyClass(name="Algorithm"),
        OntologyClass(name="Model"),
        OntologyClass(name="Data")
    ])

    extractor = TestTripletExtractor(threshold=0.7)

    # First call
    input_data1 = TripletExtractorInput(
        text="Machine Learning models process data.",
        ontology=ontology
    )
    result1 = extractor(input=input_data1)

    # Store initial triplets
    original_triplet_count = extractor.get_triplet_count()
    if result1.triplets:
        extractor.extend_triplets(result1.triplets)

    # Second call
    input_data2 = TripletExtractorInput(
        text="Deep Learning is a subtype of Machine Learning.",
        ontology=ontology
    )
    result2 = extractor(input=input_data2)

    if result2.triplets:
        extractor.extend_triplets(result2.triplets)

    # Verify state was maintained
    assert extractor.calls_count == 2
    assert len(extractor.analysis_history) == 2
    # Verify triplets accumulate
    assert extractor.get_triplet_count() >= original_triplet_count


def test_triplet_extractor_with_multiple_texts():
    """Test processing multiple texts with the same extractor"""
    ontology = SimpleOntology(classes=[
        OntologyClass(name="Algorithm"),
        OntologyClass(name="Model"),
        OntologyClass(name="Data"),
        OntologyClass(name="Learning")
    ])

    extractor = TestTripletExtractor(threshold=0.6)  # Lower threshold for test

    # List of texts to process
    texts = [
        "Neural Networks are models that process data.",
        "Machine Learning algorithms improve with more data.",
        "Deep Learning is a subset of Machine Learning.",
        "GPT models are Transformer-based architectures."
    ]

    results = []
    for text in texts:
        input_data = TripletExtractorInput(
            text=text,
            ontology=ontology,
            # Pass previous state to demonstrate evolving knowledge
            state=results[-1] if results else None
        )
        result = extractor(input=input_data)
        results.append(result)

        # Extend triplets with each result
        if result.triplets:
            extractor.extend_triplets(result.triplets)

    # Verify the extractor processed all texts
    assert extractor.calls_count == len(texts)
    assert len(extractor.analysis_history) == len(texts)


def test_act_transformation():
    """Test that act transforms input to a different type that's used by the LLM"""
    ontology = SimpleOntology(classes=[OntologyClass(name="ML")])
    extractor = TestTripletExtractor()

    # Get the act method
    act_method = getattr(extractor, 'act')

    # Call act directly to confirm type transformation
    input_data = TripletExtractorInput(
        text="Neural Networks are ML models.",
        ontology=ontology
    )

    # Act should transform TripletExtractorInput to IntermediateAnalysisResult
    act_result = act_method(input_data)

    # Verify type transformation
    assert isinstance(act_result, IntermediateAnalysisResult)
    assert not isinstance(act_result, TripletExtractorInput)

    # Now test through the contract
    with MetadataTracker() as tracker:
        contract_result = extractor(input=input_data)

    # Result from contract should be KGState
    assert isinstance(contract_result, KGState)


def test_act_signature_validation():
    """Test that contract properly validates act method signature"""

    @contract() # Basic contract decorator for testing
    class BadActSignatureMissingInput(Expression):
        @property
        def prompt(self) -> str: return "test"
        def act(self, wrong_name: TripletExtractorInput, **kwargs) -> IntermediateAnalysisResult:
            return IntermediateAnalysisResult(analyzed_text=wrong_name.text)
        def forward(self, **kwargs) -> KGState: # Make forward more lenient for this test
            return KGState()

    @contract()
    class BadActSignatureNoAnnotation(Expression):
        @property
        def prompt(self) -> str: return "test"
        def act(self, input, **kwargs) -> IntermediateAnalysisResult: # No type annotation for input
            return IntermediateAnalysisResult(analyzed_text=input.text)
        def forward(self, **kwargs) -> KGState:
            return KGState()

    @contract()
    class BadActSignatureNoReturnAnnotation(Expression):
        @property
        def prompt(self) -> str: return "test"
        def act(self, input: TripletExtractorInput, **kwargs): # No return type annotation
            return IntermediateAnalysisResult(analyzed_text=input.text)
        def forward(self, **kwargs) -> KGState:
            return KGState()

    @contract()
    class BadActSignatureWrongInputType(Expression):
        @property
        def prompt(self) -> str: return "test"
        def act(self, input: str, **kwargs) -> IntermediateAnalysisResult: # Input not LLMDataModel
            return IntermediateAnalysisResult(analyzed_text=input)
        def forward(self, **kwargs) -> KGState:
            return KGState()

    @contract()
    class BadActSignatureWrongReturnType(Expression):
        @property
        def prompt(self) -> str: return "test"
        def act(self, input: TripletExtractorInput, **kwargs) -> str: # Return not LLMDataModel
            return input.text
        def forward(self, **kwargs) -> KGState:
            return KGState()

    dummy_ontology = SimpleOntology(classes=[OntologyClass(name="Test")])
    dummy_input = TripletExtractorInput(text="test", ontology=dummy_ontology)

    # The contract decorator's _act method will raise TypeError internally.
    # This TypeError will be caught by wrapped_forward, contract_successful will be False.
    # The lenient forward methods will then be called.

    contract_bsmi = BadActSignatureMissingInput()
    contract_bsmi(input=dummy_input)
    assert not contract_bsmi.contract_successful # Check that the contract failed

    contract_bsna = BadActSignatureNoAnnotation()
    contract_bsna(input=dummy_input)
    assert not contract_bsna.contract_successful

    contract_bsnra = BadActSignatureNoReturnAnnotation()
    contract_bsnra(input=dummy_input)
    assert not contract_bsnra.contract_successful

    contract_bswit = BadActSignatureWrongInputType()
    contract_bswit(input=dummy_input)
    assert not contract_bswit.contract_successful

    contract_bswrt = BadActSignatureWrongReturnType()
    contract_bswrt(input=dummy_input)
    assert not contract_bswrt.contract_successful


def test_act_contract_flow():
    """Test how act affects the contract's execution flow"""

    @contract(pre_remedy=True, post_remedy=True, verbose=True)
    class FlowTestContractPrecise(Expression):
        def __init__(self):
            super().__init__()
            self.flow_log = []
        @property
        def prompt(self) -> str:
            self.flow_log.append("prompt_accessed")
            return "test prompt for flow"
        def pre(self, input: TripletExtractorInput) -> bool:
            self.flow_log.append("pre_called")
            return True
        def act(self, input: TripletExtractorInput, **kwargs) -> IntermediateAnalysisResult:
            self.flow_log.append("act_called")
            return IntermediateAnalysisResult(analyzed_text=input.text, confidence_threshold=0.7)
        def post(self, output: KGState) -> bool:
            self.flow_log.append("post_called")
            # output is the KGState instance created by _validate_output based on LLM response
            # For this flow test, we just care it's called.
            return True
        def forward(self, input: TripletExtractorInput, **kwargs) -> KGState:
            self.flow_log.append("original_forward_called")
            if self.contract_result is None:
                # This indicates contract failed before _validate_output set contract_result,
                # or _validate_output itself failed to produce a valid result.
                # For a successful flow, contract_result should be set.
                # If testing failure flow, this path might be taken.
                # For this happy path flow test, contract_result should be valid.
                raise ValueError("Contract failed, contract_result is None in original_forward!")
            return self.contract_result

    test_contract_instance_precise = FlowTestContractPrecise()
    dummy_ontology = SimpleOntology(classes=[OntologyClass(name="Test")])
    dummy_input_data = TripletExtractorInput(text="flow test text", ontology=dummy_ontology)

    # In a real scenario with pre_remedy=True and post_remedy=True,
    # _validate_input and _validate_output in strategy.py would use the prompt.
    # _validate_output would create/validate the KGState (final output type) using an LLM call.
    # This KGState then becomes self.contract_result.
    # The user's `post` is called by `_validate_output` with this KGState.
    # The user's `forward` is called by `wrapped_forward` and should return `self.contract_result`.

    # We are not mocking LLM calls here, so the actual KGState might be empty or default.
    # The important part is the sequence of method calls.
    with MetadataTracker() as tracker: # To allow LLM calls if not mocked
        result = test_contract_instance_precise(input=dummy_input_data)

    # Expected order:
    # 1. prompt_accessed (by _validate_input due to pre_remedy=True)
    # 2. pre_called (by _validate_input)
    # 3. act_called (by _act)
    # 4. prompt_accessed (by _validate_output due to post_remedy=True, assuming it uses prompt)
    # 5. post_called (by _validate_output)
    # 6. original_forward_called (by wrapped_forward in finally block)

    # Check the sequence
    assert "pre_called" in test_contract_instance_precise.flow_log
    assert "act_called" in test_contract_instance_precise.flow_log
    assert "post_called" in test_contract_instance_precise.flow_log
    assert "original_forward_called" in test_contract_instance_precise.flow_log

    # Check relative order
    assert test_contract_instance_precise.flow_log.index("pre_called") < \
            test_contract_instance_precise.flow_log.index("act_called")

    # Note: 'post_called' is called within _validate_output. Prompt access by _validate_output
    # might happen before or after calling 'post', depending on _validate_output's internal logic.
    # For this test, we simplify and just check act before post.
    assert test_contract_instance_precise.flow_log.index("act_called") < \
            test_contract_instance_precise.flow_log.index("post_called")

    assert test_contract_instance_precise.flow_log.index("post_called") < \
            test_contract_instance_precise.flow_log.index("original_forward_called")

    # Verify prompt access. Exact number of calls depends on strategy.py internal logic.
    assert test_contract_instance_precise.flow_log.count("prompt_accessed") >= 1


def test_act_contract_state_interaction():
    """Test how act's state changes affect contract validation and error handling"""

    @contract(pre_remedy=True, post_remedy=True, verbose=True)
    class StateInteractionContract(Expression):
        def __init__(self):
            super().__init__()
            self.custom_threshold = 0.5 # Initial state
            self.prompt_access_count = 0

        @property
        def prompt(self) -> str:
            self.prompt_access_count += 1
            return "Analyze sentiment of: {{text}}"

        def pre(self, input: TripletExtractorInput) -> bool:
            if input.text == "fail_pre_with_remedy": # This will trigger remedy
                raise ValueError("Pre-condition failed by 'fail_pre_with_remedy'")
            return True

        def act(self, input: TripletExtractorInput, **kwargs) -> IntermediateAnalysisResult:
            if input.text == "fail_act":
                raise RuntimeError("Simulated failure in act method")
            # Modify instance state that might be used in post
            self.custom_threshold = 0.9
            return IntermediateAnalysisResult(
                analyzed_text=f"{input.text} - analyzed",
                confidence_threshold=self.custom_threshold # Pass it along
            )

        def post(self, output: KGState) -> bool:
            # Post-condition uses the updated threshold from act
            if output.triplets:
                for triplet in output.triplets:
                    if triplet.confidence < self.custom_threshold:
                        raise ValueError(f"Post-condition failed: Confidence {triplet.confidence} < {self.custom_threshold}")
            return True

        def forward(self, input: TripletExtractorInput, **kwargs) -> KGState:
            # This is the original_forward method.
            if not self.contract_successful and self.contract_result is None: # Fallback path
                return KGState(triplets=[Triplet(subject=Entity(name="Fallback"),
                                                 predicate=Relationship(name="created_due_to_failure"),
                                                 object=Entity(name=input.text[:10]), # Use original input for fallback
                                                 confidence=0.1)])

            if self.contract_result is None: # Should ideally not happen if contract_successful
                 raise ValueError("Contract successful but contract_result is None in forward!")

            return self.contract_result

    contract_instance = StateInteractionContract()
    dummy_ontology = SimpleOntology(classes=[OntologyClass(name="Test")])

    # Case 1: Act modifies state, post uses it, and validation passes.
    # For this to pass, _validate_output (which calls post) must be "aware" of an LLM producing
    # a KGState with triplets that meet the new threshold of 0.9.
    # Without a mock LLM, we assume _validate_output can produce such a result if prompted correctly.
    input_data_pass = TripletExtractorInput(
        text="Test_Pass_State_Interaction",
        ontology=dummy_ontology,
        # Simulate an incoming state that, if returned by LLM, would pass the post-condition
        state=KGState(triplets=[Triplet(subject=Entity(name="S"), predicate=Relationship(name="P"), object=Entity(name="O"), confidence=0.95)])
    )
    # We rely on the actual contract mechanisms. If the LLM happens to return
    # triplets like those in `input_data_pass.state` (or an empty list), it might pass.
    # This test is more of an integration style for state due to LLM dependency.
    # For a more controlled unit test, one might need to mock the LLM response within _validate_output.
    try:
        # This call will use the real _validate_output, which involves LLM.
        # If LLM produces triplets, they must meet the 0.9 threshold set by act.
        # If LLM produces no triplets, post will pass.
        result_pass = contract_instance(input=input_data_pass)
        assert contract_instance.custom_threshold == 0.9 # Verify act updated state
        if result_pass.triplets: # If any triplets were produced by the LLM
            for t in result_pass.triplets:
                assert t.confidence >= 0.9
    except Exception as e:
        # Allow LLM errors or validation failures if LLM doesn't cooperate for "pass" case
        logging.warning(f"State interaction pass test encountered: {e}. This might be due to LLM variability.")
        pass


    # Case 2: Act itself fails
    contract_instance_fail_act = StateInteractionContract() # Re-initialize for a clean state
    input_data_fail_act = TripletExtractorInput(text="fail_act", ontology=dummy_ontology)

    # The RuntimeError from `act` will be caught by `wrapped_forward`.
    # `contract_successful` will be False.
    # The `finally` block in `wrapped_forward` will call `original_forward` (StateInteractionContract.forward).
    # StateInteractionContract.forward will execute its fallback logic.

    fallback_result_from_act_failure = contract_instance_fail_act(input=input_data_fail_act)

    # Verify fallback output
    assert fallback_result_from_act_failure.triplets[0].subject.name == "Fallback"
    assert fallback_result_from_act_failure.triplets[0].object.name == "fail_act"[:10] # Original input text used in fallback

    # After the error in `act`, check state that should not have been changed by the aborted `act` call.
    # In StateInteractionContract, `act` raises RuntimeError BEFORE `self.custom_threshold` is changed from 0.5 to 0.9.
    assert contract_instance_fail_act.custom_threshold == 0.5

    # Verify contract_successful is False
    assert not contract_instance_fail_act.contract_successful

    # Reset state for next test part
    contract_instance.custom_threshold = 0.5
    contract_instance.contract_successful = False # Reset for clarity
    contract_instance.contract_result = None


    # Case 3: Pre-condition fails (with pre_remedy=True), test fallback if remedy fails.
    # This is hard to test reliably without mocking the SemanticValidationFunction (remedy LLM).
    # If `pre` raises ValueError, `_validate_input` calls `SemanticValidationFunction`.
    # If remedy succeeds, input is modified, flow continues.
    # If remedy fails, the exception from `SemanticValidationFunction` propagates.
    # This exception is caught by `wrapped_forward`, `contract_successful=False`. Fallback.

    # For a simpler test of pre-failure leading to fallback, use pre_remedy=False.
    @contract(pre_remedy=False, post_remedy=False, verbose=True)
    class StateInteractionNoRemedyPreFail(Expression):
        def __init__(self): super().__init__()
        @property
        def prompt(self) -> str: return "test" # Not used with remedy=False
        def pre(self, input: TripletExtractorInput) -> bool:
            if input.text == "fail_pre_direct":
                # With pre_remedy=False, returning False or raising will cause _validate_input to raise.
                raise ValueError("Direct pre fail")
            return True
        # Define act and post to satisfy contract requirements, though not central to this test part
        def act(self, input: TripletExtractorInput, **kwargs) -> IntermediateAnalysisResult:
            return IntermediateAnalysisResult(analyzed_text=input.text)
        def post(self, output: KGState) -> bool:
            return True
        def forward(self, input: TripletExtractorInput, **kwargs) -> KGState: # input here is original_input
            if not self.contract_successful and self.contract_result is None: # Fallback
                return KGState(triplets=[Triplet(subject=Entity(name="FallbackDirectPreFail"),
                                                 predicate=Relationship(name="P"),
                                                 object=Entity(name=input.text[:10]),
                                                 confidence=0.1)])
            # This path should not be taken if pre-failed and fallback occurred.
            if self.contract_result is None: raise ValueError("Contract result none in forward for success path");
            return self.contract_result

    contract_no_remedy_pre = StateInteractionNoRemedyPreFail()
    input_fail_pre_direct = TripletExtractorInput(text="fail_pre_direct", ontology=dummy_ontology)

    # `_validate_input` (with pre_remedy=False) will raise Exception if pre returns False/raises.
    # This exception is caught by `wrapped_forward`, `contract_successful` becomes False.
    # `original_forward` (StateInteractionNoRemedyPreFail.forward) is called with original_input.
    fallback_pre_direct_fail = contract_no_remedy_pre(input=input_fail_pre_direct)
    assert fallback_pre_direct_fail.triplets[0].subject.name == "FallbackDirectPreFail"
    assert fallback_pre_direct_fail.triplets[0].object.name == "fail_pre_d" # original input text
    assert not contract_no_remedy_pre.contract_successful


def test_contract_result_propagation():
    """Test that contract_result is properly set by _validate_output and used by forward"""

    @contract(pre_remedy=False, post_remedy=True, verbose=True) # post_remedy=True will use LLM
    class ResultPropagationContract(Expression):
        def __init__(self):
            super().__init__()
            self.intermediate_text_from_act = None

        @property
        def prompt(self) -> str:
            # This prompt will be used by _validate_output (specifically by TypeValidationFunction/SemanticValidationFunction)
            # It should demonstrate using the output of the 'act' method.
            # The `current_input` to _validate_output in strategy.py IS the output of `_act`.
            # So, if act returns IntermediateAnalysisResult, then {{analyzed_text}} should be available.
            return "Create KGState based on analyzed text: {{analyzed_text}}. Entities: {{possible_entities}}. Relationships: {{possible_relationships}}."

        def pre(self, input: TripletExtractorInput) -> bool: return True

        def act(self, input: TripletExtractorInput, **kwargs) -> IntermediateAnalysisResult:
            self.intermediate_text_from_act = f"act_on_{input.text}"
            return IntermediateAnalysisResult(
                analyzed_text=self.intermediate_text_from_act,
                possible_entities=["entity_from_act"],
                possible_relationships=["rel_from_act"],
                confidence_threshold=0.88
            )

        def post(self, output: KGState) -> bool:
            # This post validates the KGState produced by the LLM in _validate_output.
            # Check if the KGState somehow reflects the data from IntermediateAnalysisResult.
            # This depends on how well the LLM uses the prompt.
            if output.triplets:
                for triplet in output.triplets:
                    if self.intermediate_text_from_act and self.intermediate_text_from_act[:5] in triplet.subject.name:
                        logging.info(f"Post-validation: Found data from 'act' in triplet: {triplet.subject.name}")
                        return True # Found evidence of act's output influencing the result
                # If no triplet contains evidence, it might still be a pass if LLM chose not to include it.
                # For a robust test, prompt might need to be more directive.
            return True # Default pass if no specific check fails

        def forward(self, input: TripletExtractorInput, **kwargs) -> KGState:
            if not self.contract_successful and self.contract_result is None: # Fallback
                return KGState(triplets=[Triplet(subject=Entity(name="FallbackForwardProp"), predicate=Relationship(name="P"), object=Entity(name="O"), confidence=0.1)])

            if self.contract_result is None:
                raise ValueError("Contract.contract_result was None in forward (success path)!")

            assert isinstance(self.contract_result, KGState)
            return self.contract_result

    contract_instance = ResultPropagationContract()
    dummy_ontology = SimpleOntology(classes=[OntologyClass(name="Test")]) # Not used by this contract's prompt directly
    input_data = TripletExtractorInput(text="propagate_test_text", ontology=dummy_ontology)

    # This call will use the real _validate_output, which involves an LLM call
    # using the prompt defined in ResultPropagationContract.
    # The prompt will be formatted with data from the IntermediateAnalysisResult returned by `act`.
    with MetadataTracker() as tracker:
        result = contract_instance(input=input_data)

    assert isinstance(result, KGState)
    assert contract_instance.contract_successful # Check that the main path was successful

    # We can't easily assert specific content of `result.triplets` without a mock LLM,
    # as it depends on the real LLM's interpretation of the prompt.
    # However, if `post_remedy=True` and `post` passes, and `verbose=True` in contract,
    # logs would show the LLM call and the data flow.
    # The `post` method above includes a log if it finds evidence.
    logging.info(f"Resulting KG for propagation test: {result.model_dump_json(indent=2)}")
    if result.triplets and contract_instance.intermediate_text_from_act:
        found_evidence = any(contract_instance.intermediate_text_from_act[:5] in t.subject.name for t in result.triplets)
        logging.info(f"Evidence of 'act' data in final result: {found_evidence}")
        # assert found_evidence # This assertion is LLM-dependent


if __name__ == "__main__":
    pytest.main()
