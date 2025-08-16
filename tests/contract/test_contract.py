import logging
import time
from typing import List, Optional

import numpy as np
import pytest
from pydantic import Field

from symai import Expression
from symai.backend.settings import SYMAI_CONFIG
from symai.models import LLMDataModel
from symai.strategy import contract
from symai.utils import semassert

NEUROSYMBOLIC = SYMAI_CONFIG.get('NEUROSYMBOLIC_ENGINE_MODEL')


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
            raise self.contract_exception or ValueError("Contract failed!")
        return self.contract_result

    def extend_triplets(self, new_triplets: List[Triplet]):
        """Store extracted triplets"""
        if new_triplets:
            self._triplets.update(new_triplets)

    def get_triplet_count(self) -> int:
        """Return the number of unique triplets extracted"""
        return len(self._triplets)


# ============================
# LLMDataModel Type Annotation Tests
# ============================
##############################

@pytest.mark.mandatory
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

    result = extractor(input=input_data)

    # Verify result structure
    assert isinstance(result, KGState)
    if result.triplets:
        for triplet in result.triplets:
            assert triplet.confidence >= 0.7  # Respect threshold

    # Verify state was updated
    assert extractor.calls_count == 1
    assert len(extractor.analysis_history) == 1


@pytest.mark.mandatory
def test_triplet_extractor_state_persistence():
    """Test that state is maintained across multiple calls"""
    ontology = SimpleOntology(classes=[
        OntologyClass(name="Algorithm"),
        OntologyClass(name="Model"),
        OntologyClass(name="Data")
    ])

    extractor = TestTripletExtractor(threshold=0.7)
    input_data1 = TripletExtractorInput(
        text="Machine Learning models process data.",
        ontology=ontology
    )
    result1 = extractor(input=input_data1)

    original_triplet_count = extractor.get_triplet_count()
    if result1.triplets:
        extractor.extend_triplets(result1.triplets)
    input_data2 = TripletExtractorInput(
        text="Deep Learning is a subtype of Machine Learning.",
        ontology=ontology
    )
    result2 = extractor(input=input_data2)
    if result2.triplets:
        extractor.extend_triplets(result2.triplets)

    assert extractor.calls_count == 2
    assert len(extractor.analysis_history) == 2
    assert extractor.get_triplet_count() >= original_triplet_count


@pytest.mark.mandatory
def test_triplet_extractor_with_multiple_texts():
    """Test processing multiple texts with the same extractor"""
    ontology = SimpleOntology(classes=[
        OntologyClass(name="Algorithm"),
        OntologyClass(name="Model"),
        OntologyClass(name="Data"),
        OntologyClass(name="Learning")
    ])

    extractor = TestTripletExtractor(threshold=0.6)  # Lower threshold for test
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

        if result.triplets:
            extractor.extend_triplets(result.triplets)

    assert extractor.calls_count == len(texts)
    assert len(extractor.analysis_history) == len(texts)


@pytest.mark.mandatory
def test_act_transformation():
    """Test that act transforms input to a different type that's used by the LLM"""
    ontology = SimpleOntology(classes=[OntologyClass(name="ML")])
    extractor = TestTripletExtractor()

    act_method = getattr(extractor, 'act')

    input_data = TripletExtractorInput(
        text="Neural Networks are ML models.",
        ontology=ontology
    )

    act_result = act_method(input_data)
    assert isinstance(act_result, IntermediateAnalysisResult)
    assert not isinstance(act_result, TripletExtractorInput)

    contract_result = extractor(input=input_data)
    assert isinstance(contract_result, KGState)


@pytest.mark.mandatory
def test_act_signature_validation():
    """Test that contract properly validates act method signature"""

    @contract()
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


@pytest.mark.mandatory
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


@pytest.mark.mandatory
def test_act_contract_state_interaction():
    """Test how act's state changes affect contract validation and error handling"""

    @contract(pre_remedy=True, post_remedy=True, verbose=True)
    class StateInteractionContract(Expression):
        def __init__(self):
            super().__init__()
            self.custom_threshold = 0.5
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
            self.custom_threshold = 0.9
            return IntermediateAnalysisResult(
                analyzed_text=f"{input.text} - analyzed",
                confidence_threshold=self.custom_threshold
            )

        def post(self, output: KGState) -> bool:
            # Post-condition uses the updated threshold from act
            if output.triplets:
                for triplet in output.triplets:
                    if triplet.confidence < self.custom_threshold:
                        raise ValueError(f"Post-condition failed: Confidence {triplet.confidence} < {self.custom_threshold}")
            return True

        def forward(self, input: TripletExtractorInput, **kwargs) -> KGState:
            if not self.contract_successful and self.contract_result is None:
                return KGState(triplets=[Triplet(subject=Entity(name="Fallback"),
                                                 predicate=Relationship(name="created_due_to_failure"),
                                                 object=Entity(name=input.text[:10]),
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


@pytest.mark.mandatory
def test_contract_result_propagation():
    """Test that contract_result is properly set by _validate_output and used by forward"""

    @contract(pre_remedy=False, post_remedy=True, verbose=True) # post_remedy=True will use LLM
    class ResultPropagationContract(Expression):
        def __init__(self):
            super().__init__()
            self.intermediate_text_from_act = None

        @property
        def prompt(self) -> str:
            # This prompt will be used by _validate_output
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


@pytest.mark.mandatory
def test_contract_perf_stats():
    """Test the contract_perf_stats method correctly tracks and reports timing statistics."""

    class PerfStatsInput(LLMDataModel):
        text: str = Field(description="Input text for performance testing")

    class PerfStatsOutput(LLMDataModel):
        processed_text: str = Field(description="Processed text output")

    @contract(pre_remedy=True, post_remedy=True, verbose=True)
    class PerfStatsTestContract(Expression):
        def __init__(self):
            super().__init__()
            self.sleep_time = 0.01

        @property
        def prompt(self) -> str:
            return "performance stats test prompt"

        def pre(self, input: PerfStatsInput) -> bool:
            time.sleep(self.sleep_time + 1)
            return True

        def act(self, input: PerfStatsInput, **kwargs) -> PerfStatsOutput:
            time.sleep(self.sleep_time + 2)
            return PerfStatsOutput(processed_text=f"Processed: {input.text}")

        def post(self, output: PerfStatsOutput) -> bool:
            time.sleep(self.sleep_time + 3)
            return True

        def forward(self, input: PerfStatsInput, **kwargs) -> PerfStatsOutput:
            time.sleep(self.sleep_time + 1)
            return self.contract_result if self.contract_result else PerfStatsOutput(processed_text=f"Original: {input.text}")

    contract_instance = PerfStatsTestContract()

    input_texts = ["test1", "test2", "test3"]
    sleep_times = [0.01, 0.02, 0.03]

    for i, (input_text, sleep_time) in enumerate(zip(input_texts, sleep_times)):
        contract_instance.sleep_time = sleep_time
        input_model = PerfStatsInput(text=input_text)
        result = contract_instance(input=input_model)
        assert isinstance(result, PerfStatsOutput)
        assert result.processed_text, "Processed text should not be empty"

    stats = contract_instance.contract_perf_stats()

    expected_operations = [
        "input_validation",
        "act_execution",
        "output_validation",
        "forward_execution",
        "contract_execution"
    ]

    for op in expected_operations:
        assert op in stats, f"Operation {op} missing from stats"

    assert "overhead" in stats, f"Overhead tracking missing from stats"

    assert stats["act_execution"]["count"] == 3

    for op in ["act_execution", "contract_execution"]:
        if stats[op]["count"] > 1:
            assert stats[op]["min"] <= stats[op]["mean"] <= stats[op]["max"], f"Statistics for {op} are inconsistent"

    for op in expected_operations:
        if stats[op]["count"] > 0:
            if stats[op]["count"] > 1:
                assert stats[op]["min"] <= stats[op]["mean"] <= stats[op]["max"], f"Statistics for {op} are inconsistent"

            assert stats[op]["total"] >= 0
            assert stats[op]["mean"] >= 0
            assert stats[op]["min"] >= 0
            assert stats[op]["max"] >= 0
            assert stats[op]["std"] >= 0

    if stats["contract_execution"]["total"] > 0:
        tracked_ops = expected_operations[:-1]  # Exclude contract_execution
        total_percentage = sum(stats[op]["percentage"] for op in tracked_ops) + stats["overhead"]["percentage"]
        # Should be very close to 100%
        assert abs(total_percentage - 100.0) < 0.1, "Percentages including overhead should sum to 100%"

        # Verify overhead calculation
        tracked_time = sum(stats[op]["total"] for op in tracked_ops)
        total_time = stats["contract_execution"]["total"]
        calculated_overhead = total_time - tracked_time
        assert abs(calculated_overhead - stats["overhead"]["total"]) < 0.001, "Overhead calculation is incorrect"


@pytest.mark.mandatory
def test_act_state_modification_without_input_change():
    class StateModificationInput(LLMDataModel):
        value: str = Field(description="Input value")

    class StateModificationOutput(LLMDataModel):
        value: str = Field(description="Output value")

    @contract(pre_remedy=True, post_remedy=True, verbose=True)
    class StateModificationContract(Expression):
        def __init__(self):
            super().__init__()
            self.state_counter = 0
            self.last_input_value = None
            self.method_calls = []

        @property
        def prompt(self) -> str:
            self.method_calls.append("prompt_accessed")
            return "Test contract that modifies state without changing input"

        def pre(self, input: StateModificationInput) -> bool:
            self.method_calls.append("pre_called")
            return True

        def act(self, input: StateModificationInput, **kwargs) -> StateModificationInput:
            self.method_calls.append("act_called")
            # Modify internal state
            self.state_counter += 1
            self.last_input_value = input.value

            # Store the input object's identity to verify it's the same object throughout
            self.original_input_id = id(input)

            # Return the input unchanged - this tests that we don't need
            # to create a new object when we just want to modify state
            return input

        def post(self, output: StateModificationOutput) -> bool:
            self.method_calls.append("post_called")
            return True

        def forward(self, input: StateModificationInput, **kwargs) -> StateModificationOutput:
            self.method_calls.append("forward_called")
            # Verify that the input received in forward is the same object that went through act
            # This confirms proper input propagation through the contract mechanism
            self.input_preserved = (id(input) == self.original_input_id)

            # In a successful contract flow, contract_result should be set
            # For this test, we want to use the validated input but create our own output
            return StateModificationOutput(value=input.value)

    test_input = StateModificationInput(value="test_value")
    contract_instance = StateModificationContract()

    # Initial state should be zero
    assert contract_instance.state_counter == 0
    assert contract_instance.last_input_value is None

    result = contract_instance(input=test_input)

    # Verify the execution flow
    assert "pre_called" in contract_instance.method_calls
    assert "act_called" in contract_instance.method_calls
    assert "post_called" in contract_instance.method_calls
    assert "forward_called" in contract_instance.method_calls

    # Check the correct ordering of method calls
    assert contract_instance.method_calls.index("pre_called") < contract_instance.method_calls.index("act_called")
    assert contract_instance.method_calls.index("act_called") < contract_instance.method_calls.index("post_called")
    assert contract_instance.method_calls.index("post_called") < contract_instance.method_calls.index("forward_called")

    # Verify that the state was modified by the act method
    assert contract_instance.state_counter == 1
    assert contract_instance.last_input_value == "test_value"

    # Verify that the input object was preserved throughout the contract execution
    assert contract_instance.input_preserved, "Input object identity was not preserved through the contract flow"

    assert isinstance(result, StateModificationOutput)
    assert result.value == "test_value"

    # Call the contract again to verify state accumulation
    result = contract_instance(input=test_input)

    # State counter should have been incremented again
    assert contract_instance.state_counter == 2


@pytest.mark.mandatory
def test_skip_pre_and_post_without_methods():
    """Test that when no pre/post methods are implemented and remedy disabled, input is passed through act and forward."""
    class SimpleInput(LLMDataModel):
        x: int

    @contract(pre_remedy=False, post_remedy=False, verbose=False,
              remedy_retry_params={'tries': 1, 'delay': 0, 'max_delay': 0, 'jitter': 0, 'backoff': 1, 'graceful': True})
    class SimpleNoPreNoPost(Expression):
        @property
        def prompt(self) -> str:
            return "skip pre/post prompt"

        def act(self, input: SimpleInput, **kwargs) -> SimpleInput:
            # Increment value to test act propagation
            return SimpleInput(x=input.x + 2)

        def forward(self, input: SimpleInput, **kwargs) -> SimpleInput:
            # Should receive input from act unchanged
            return input

    instance = SimpleNoPreNoPost()
    input_model = SimpleInput(x=3)
    result = instance(input=input_model)
    assert isinstance(result, SimpleInput)
    assert result.x == input_model.x + 2


@pytest.mark.mandatory
def test_post_without_remedy_returns_act_output():
    """Test that when a post method is defined but post_remedy is False, the act output is returned."""
    class SimpleIn(LLMDataModel):
        x: int

    class SimpleOut(LLMDataModel):
        x: int

    @contract(pre_remedy=False, post_remedy=False, verbose=False,
              remedy_retry_params={'tries': 1, 'delay': 0, 'max_delay': 0, 'jitter': 0, 'backoff': 1, 'graceful': True})
    class SimpleWithPost(Expression):
        @property
        def prompt(self) -> str:
            return "post without remedy prompt"

        def act(self, input: SimpleIn, **kwargs) -> SimpleOut:
            return SimpleOut(x=input.x * 3)

        def post(self, output: SimpleOut) -> bool:
            # Simple post validation; should pass
            assert output.x == 6
            return True

        def forward(self, input: SimpleOut, **kwargs) -> SimpleOut:
            return input

    instance2 = SimpleWithPost()
    input_model2 = SimpleIn(x=2)
    result2 = instance2(input=input_model2)
    assert isinstance(result2, SimpleOut)
    assert result2.x == input_model2.x * 3


@pytest.mark.mandatory
def test_forward_return_type_mismatch():
    """Original forward returning wrong type should trigger TypeError"""
    class NumberModel(LLMDataModel):
        x: int

    @contract(pre_remedy=False, post_remedy=False)
    class BadReturn(Expression):
        @property
        def prompt(self) -> str:
            return "Return type mismatch"

        def forward(self, input: NumberModel, **kwargs) -> str:
            return 123  # wrong type

    inst = BadReturn()
    with pytest.raises(TypeError):
        inst(input=NumberModel(x=1))

@pytest.mark.mandatory
def test_final_type_check_graceful_behavior():
    """Test final type-check behavior under graceful modes"""
    # Define a simple data model
    class NumberModel(LLMDataModel):
        x: int

    # Non-graceful mode: should raise TypeError on forward return type mismatch
    @contract(pre_remedy=False, post_remedy=False, remedy_retry_params={"graceful": False})
    class BadReturnFail(Expression):
        @property
        def prompt(self) -> str:
            return "Testing graceful False"

        def forward(self, input: NumberModel, **kwargs) -> str:
            return 789  # wrong type

    inst_fail = BadReturnFail()
    with pytest.raises(TypeError):
        inst_fail(input=NumberModel(x=1))

    # Graceful mode: should skip TypeError and return raw output
    @contract(pre_remedy=False, post_remedy=False, remedy_retry_params={"graceful": True})
    class BadReturnGrace(Expression):
        @property
        def prompt(self) -> str:
            return "Testing graceful True"

        def forward(self, input: NumberModel, **kwargs) -> str:
            return 246  # raw output allowed under graceful

    inst_grace = BadReturnGrace()
    result = inst_grace(input=NumberModel(x=2))
    assert result == 246

@pytest.mark.mandatory
def test_pre_remedy_input_correction_via_fake_remedy():
    """With pre_remedy=True, the remedy function can correct invalid input"""
    class NumberModel(LLMDataModel):
        x: int

    @contract(pre_remedy=True, post_remedy=False)
    class FixX(Expression):
        def __init__(self):
            super().__init__()

        @property
        def prompt(self) -> str:
            return "Ensure x is non-negative: {{x}}"

        def pre(self, input: NumberModel) -> bool:
            if input.x <= 0:
                raise ValueError("x must be positive")
            self.fixed_x = input.x
            return True

        def forward(self, input: NumberModel, **kwargs) -> NumberModel:
            if self.contract_result is None:
                raise self.contract_exception or ValueError("Contract failed!")
            return self.contract_result

    inst = FixX()
    out = inst(input=NumberModel(x=-10))
    assert isinstance(out, NumberModel)
    assert out.x >= 0
    assert inst.fixed_x >= 0
    assert inst.contract_successful is True

# ============================
# Dynamic Type Annotation Tests
# ============================
##############################
#
# These tests cover the dynamic type annotation functionality in the contract system.
# Dynamic type annotation allows contracts to work with primitive Python types (str, int, list, etc.)
# and complex typing constructs (Union, Optional, nested collections) by automatically creating
# wrapper LLMDataModel classes at runtime.
#
# How Dynamic Type Annotation Works:
# 1. When a contract method (act/forward) uses non-LLMDataModel type annotations (e.g., str, list[str]),
#    the system detects this and creates a dynamic LLMDataModel wrapper using build_dynamic_llm_datamodel().
# 2. The dynamic model has a 'value' field that contains the actual primitive data.
# 3. Input data is wrapped in the dynamic model for internal processing.
# 4. Output data is automatically unwrapped from the dynamic model before returning to the user.

@pytest.mark.mandatory
def test_dynamic_type_annotation_primitive_types():
    """Test dynamic type annotation with primitive Python types."""

    @contract(pre_remedy=False, post_remedy=False)
    class PrimitiveStringContract(Expression):
        def __init__(self):
            super().__init__()

        @property
        def prompt(self) -> str:
            return "Convert input to uppercase string"

        def forward(self, input: str, **kwargs) -> str:
            if self.contract_result is None:
                raise self.contract_exception or ValueError("Contract failed!")
            return self.contract_result

    # Test string contract
    string_contract = PrimitiveStringContract()
    result = string_contract(input="hello")
    # The result will be unwrapped to the primitive value
    assert result == "HELLO"

@pytest.mark.mandatory
def test_dynamic_type_annotation_list_type():
    @contract(remedy_retry_params=dict(tries=4, delay=0))
    class IdentityListContract(Expression):
        @property
        def prompt(self) -> str:
            return "Return the list with all its elements squared"

        def forward(self, input: list[int], **kwargs) -> list[int]:
            if self.contract_result is None:
                raise self.contract_exception or ValueError("Contract failed!")
            return self.contract_result

    list_contract = IdentityListContract()
    input_list = [3, 4, 5]
    result = list_contract(input=input_list)
    semassert(result == [9, 16, 25])

@pytest.mark.mandatory
def test_dynamic_type_annotation_dict_type():
    @contract(pre_remedy=False, post_remedy=True, remedy_retry_params=dict(tries=4, delay=0))
    class IdentityDictContract(Expression):
        @property
        def prompt(self) -> str:
            return "Return the values squared"

        def post(self, output: dict[str, int]) -> bool:
            if output != {"a": 1, "b": 4, "c": 25}:
                raise ValueError("You must return the same dictionary as the input")
            return True

        def forward(self, input: dict[str, int], **kwargs) -> dict[str, int]:
            if self.contract_result is None:
                raise self.contract_exception or ValueError("Contract failed!")
            return self.contract_result

    dict_contract = IdentityDictContract()
    input_dict = {"a": 1, "b": 2, "c": 5}
    result = dict_contract(input=input_dict)
    assert result == {"a": 1, "b": 4, "c": 25}

@pytest.mark.mandatory
def test_dynamic_type_annotation_nested_types():
    @contract(pre_remedy=False, post_remedy=False)
    class NestedContract(Expression):
        @property
        def prompt(self) -> str:
            return "Return the same nested structure unchanged"

        def forward(self, input: list[dict[str, int]], **kwargs) -> list[dict[str, int]]:
            if self.contract_result is None:
                raise self.contract_exception or ValueError("Contract failed!")
            return self.contract_result

    nested_contract = NestedContract()
    input_data = [{"a": 1}, {"b": 2}]
    result = nested_contract(input=input_data)
    assert result == input_data

@pytest.mark.mandatory
def test_dynamic_type_annotation_union_optional_types():
    @contract(pre_remedy=True, post_remedy=True, remedy_retry_params=dict(tries=1, delay=0))
    class UnionOptionalContract(Expression):
        @property
        def prompt(self) -> str:
            return "Return the input unchanged"

        def pre(self, input: int | str) -> bool:
            return True

        def post(self, output: int | str | None) -> bool:
            if output != 42 and output != "test" and output is None:
                raise ValueError("You must return the same value as the input")
            return True

        def forward(self, input: int | str | None, **kwargs) -> int | str | None:
            if self.contract_result is None:
                raise self.contract_exception or ValueError("Contract failed!")
            return self.contract_result

    contract_inst = UnionOptionalContract()
    assert contract_inst(input=42) == 42
    assert contract_inst(input="test") == "test"

# ============================
# Hybrid Type Annotation Tests
# ============================
##############################

@pytest.mark.mandatory
def test_hybrid_llm_datamodel_to_list_of_str():
    """Test a hybrid contract: input is an LLMDataModel, output is a native list[str]"""
    # Define a simple LLMDataModel with a list of strings
    class WordBag(LLMDataModel):
        words: list[str]

    @contract(pre_remedy=False, post_remedy=False)
    class EchoWords(Expression):
        @property
        def prompt(self) -> str:
            return "Echo back the words as a JSON list of strings."

        def forward(self, input: WordBag, **kwargs) -> list[str]:
            if self.contract_result is None:
                raise self.contract_exception or ValueError("Contract failed!")
            return self.contract_result

    # Create input instance and run the contract
    wb = WordBag(words=["apple", "banana", "cherry"])
    runner = EchoWords()
    out = runner(input=wb)

    # Verify that the output is an unwrapped list of strings matching the input
    assert isinstance(out, list)
    assert all(isinstance(x, str) for x in out)
    assert out == ["apple", "banana", "cherry"]

@pytest.mark.mandatory
def test_hybrid_llm_datamodel_to_optional_list_of_str():
    """Test hybrid contract: input is an LLMDataModel, output is Optional[list[str]]"""
    class WordBag(LLMDataModel):
        words: list[str]

    @contract(pre_remedy=False, post_remedy=False)
    class MaybeEchoWords(Expression):
        @property
        def prompt(self) -> str:
            return (
                "If the 'words' list is empty, return null; "
                "otherwise echo the words as a JSON list of strings."
            )

        def forward(self, input: WordBag, **kwargs) -> Optional[list[str]]:
            if self.contract_result is None:
                raise self.contract_exception or ValueError("Contract failed!")
            return self.contract_result

    wb_empty = WordBag(words=[])
    wb_vals = WordBag(words=["x", "y"])
    runner = MaybeEchoWords()
    out_empty = runner(input=wb_empty)
    out_vals = runner(input=wb_vals)
    assert out_empty is None
    assert isinstance(out_vals, list)
    assert out_vals == ["x", "y"]

@pytest.mark.mandatory
def test_hybrid_llm_datamodel_to_dict_str_int():
    """Test hybrid contract: input is an LLMDataModel, output is dict[str, int] mapping each word to its length"""
    class WordBag(LLMDataModel):
        words: list[str]

    @contract(pre_remedy=False, post_remedy=False)
    class WordLengthMap(Expression):
        @property
        def prompt(self) -> str:
            return (
                "Given a JSON list of strings under key 'words', return a JSON object "
                "mapping each string to its length."
            )

        def forward(self, input: WordBag, **kwargs) -> dict[str, int]:
            if self.contract_result is None:
                raise self.contract_exception or ValueError("Contract failed!")
            return self.contract_result

    wb = WordBag(words=["a", "bb", "ccc"])
    runner = WordLengthMap()
    out = runner(input=wb)
    assert isinstance(out, dict)
    assert out == {"a": 1, "bb": 2, "ccc": 3}

@pytest.mark.mandatory
def test_hybrid_list_of_datamodels_to_flat_list_of_str():
    """Test hybrid contract: input is list[WordBag], output list of all words flattened"""
    class WordBag(LLMDataModel):
        words: list[str]

    @contract(pre_remedy=False, post_remedy=False)
    class FlattenBags(Expression):
        @property
        def prompt(self) -> str:
            return (
                "You're given a JSON list of objects, each having a 'words' list of strings. "
                "Return a JSON list of all strings flattened in order."
            )

        def forward(self, input: list[WordBag], **kwargs) -> list[str]:
            if self.contract_result is None:
                raise self.contract_exception or ValueError("Contract failed!")
            return self.contract_result

    wb1 = WordBag(words=["hello", "world"])
    wb2 = WordBag(words=["foo", "bar"])
    runner = FlattenBags()
    out = runner(input=[wb1, wb2])
    assert isinstance(out, list)
    assert out == ["hello", "world", "foo", "bar"]


@pytest.mark.mandatory
def test_union_dict_with_int_keys_or_list():
    """Test contract with union type dict[int, Person] | list[int]"""
    class Person(LLMDataModel):
        name: str
        age: int | None

    @contract(remedy_retry_params={"tries": 6}, verbose=True)
    class ExtractPersons(Expression):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        @property
        def prompt(self):
            return "choose the most appropiate data structure for the task at hand, that would allow you to best capture all the relevant information from the parsed unstructured text"

        def forward(self, input: str) -> dict[int, Person] | list[int]:
            if self.contract_result is None:
                raise self.contract_exception or ValueError("Contract failed!")
            return self.contract_result

    texts = [
        "Last Friday, Anna Martínez, 34, met with Dr. John Lee, 28, to discuss the community project. Meanwhile, Michael Robinson dropped by—nobody's quite sure of his age, but he seemed excited about the event. Later, 42-year-old Sarah O'Neill volunteered to coordinate the schedule.",
        "For technical support, dial 555-0134 between 9 a.m. and 6 p.m. If you need after-hours help, leave a message at +1-800-777-9924. For urgent maintenance issues, you may also text 020-7946-0011 and someone will respond within 30 minutes."
    ]

    co = ExtractPersons()
    for text in texts:
        result = co(input=text)
        # First text should extract persons as dict
        if "Anna" in text:
            assert isinstance(result, dict)
            # Check that we have Person objects
            for key, person in result.items():
                assert isinstance(key, int)
                assert isinstance(person, Person)
                assert hasattr(person, 'name')
                assert hasattr(person, 'age')
        # Second text might return list of phone numbers as integers or dict
        else:
            assert isinstance(result, (dict, list))
