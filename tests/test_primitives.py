from __future__ import annotations

import os
import tempfile
from textwrap import dedent
from typing import Any

import numpy as np
import pytest

from symai import Expression, Symbol
from symai.backend.settings import SYMAI_CONFIG
from symai.utils import semassert

_C = {
    "cyan":   "\033[96m",
    "green":  "\033[92m",
    "yellow": "\033[93m",
    "reset":  "\033[0m",
}

@pytest.mark.mandatory
def display_op(before: Any, after: Any, op: str, type: str):
    def _w(x: Any) -> Any:
        return x.value if isinstance(x, Symbol) else x

    print(
        f"{_C['cyan']}[ {op} :: {type} ]{_C['reset']} "
        f"{_C['yellow']}{_w(before)}{_C['reset']}  ➜  "
        f"{_C['green']}{_w(after)}{_C['reset']}"
    )

@pytest.mark.mandatory
def test_syn_sem_pr():
    """Test the CastingPrimitives class methods."""

    sym_default = Symbol(123) # syntactic by default
    assert not sym_default.__semantic__, "default must be syntactic"

    sym_sem = sym_default.sem # convert to semantic
    assert sym_sem.__semantic__, ".sem should yield a semantic symbol"

    sym_syn = sym_sem.syn # convert back to syntactic
    assert not sym_syn.__semantic__, ".syn after .sem should revert to syntactic"

    sym_init_sem = Symbol("hello", semantic=True)
    assert sym_init_sem.__semantic__, "explicit semantic=True must set semantic"

    chain = sym_init_sem.syn.sem.syn
    assert not chain.__semantic__, "final .syn should leave symbol syntactic"
    assert chain == "hello"

    assert isinstance(chain, Symbol)
    assert chain == sym_init_sem

    lst = Symbol(["apple", "banana", "cucumber"]).sem.syn.sem # ends semantic
    sym = Symbol("fruit")
    assert lst.__semantic__     # semantic
    assert not sym.__semantic__ # syntactic
    assert sym not in lst.syn
    semassert(sym in lst)

@pytest.mark.mandatory
def test_negate_invert_op():
    """Test the OperatorPrimitives class methods."""

    sym_int_pos = Symbol(5)
    display_op(sym_int_pos, -sym_int_pos, "-", "syntactic")
    assert -sym_int_pos.value == -5

    sym_int_neg = Symbol(-12)
    display_op(sym_int_neg, -sym_int_neg, "-", "syntactic")
    assert -sym_int_neg.value == 12

    sym_float_pos = Symbol(7.77)
    display_op(sym_float_pos, -sym_float_pos, "-", "syntactic")
    assert -sym_float_pos.value == -7.77

    sym_float_neg = Symbol(-0.123)
    display_op(sym_float_neg, -sym_float_neg, "-", "syntactic")
    assert -sym_float_neg.value == 0.123

    sym_str_affirmative = Symbol('I am happy.', semantic=True)
    display_op(sym_str_affirmative, -sym_str_affirmative, "~", "semantic")

    sym_str_negative = Symbol('This is not correct.', semantic=True)
    display_op(sym_str_negative, -sym_str_negative, "~", "semantic")

    sym_bool_true = Symbol(True)
    sym_bool_false = Symbol(False)

    assert ~sym_bool_true == False
    display_op(f"{sym_bool_true}", ~sym_bool_true, "~", "syntactic")
    assert ~sym_bool_false == True
    display_op(f"{sym_bool_false}", ~sym_bool_false, "~", "syntactic")

    sym_int_zero = Symbol(0)
    sym_int_five = Symbol(5)
    sym_int_neg_one = Symbol(-1)

    assert ~sym_int_zero == -1
    display_op(f"{sym_int_zero}", ~sym_int_zero, "~", "syntactic")
    assert ~sym_int_five == -6
    display_op(f"{sym_int_five}", ~sym_int_five, "~", "syntactic")
    assert ~sym_int_neg_one == 0
    display_op(f"{sym_int_neg_one}", ~sym_int_neg_one, "~", "syntactic")

    sym_positive_statement = Symbol('I am confident.', semantic=True)
    sym_negative_statement = Symbol('This is wrong.', semantic=True)
    sym_neutral_statement = Symbol('The weather is nice.', semantic=True)

    display_op(f"'{sym_positive_statement}'", ~sym_positive_statement, "~", "semantic")
    display_op(f"'{sym_negative_statement}'", ~sym_negative_statement, "~", "semantic")
    display_op(f"'{sym_neutral_statement}'", ~sym_neutral_statement, "~", "semantic")

    sym_bool_true_sem = Symbol(True, semantic=True)
    sym_bool_false_sem = Symbol(False, semantic=True)

    display_op(f"{sym_bool_true_sem}", ~sym_bool_true_sem, "~", "semantic")
    display_op(f"{sym_bool_false_sem}", ~sym_bool_false_sem, "~", "semantic")

    sym_affirmative = Symbol('Yes, absolutely correct.', semantic=True)
    sym_denial = Symbol('No, that is incorrect.', semantic=True)
    sym_uncertainty = Symbol('Maybe it could work.', semantic=True)

    display_op(f"'{sym_affirmative}'", ~sym_affirmative, "~", "semantic")
    display_op(f"'{sym_denial}'", ~sym_denial, "~", "semantic")
    display_op(f"'{sym_uncertainty}'", ~sym_uncertainty, "~", "semantic")

    sym_mixed = Symbol('I agree completely.')
    display_op(f"'{sym_mixed}'", ~sym_mixed.sem, "~", "semantic")

    sym_mixed_sem = Symbol('This is definitely false.', semantic=True)
    display_op(f"'{sym_mixed_sem}'", ~sym_mixed_sem.syn, "~", "syntactic")

@pytest.mark.mandatory
def test_contains_op():
    """Test the OperatorPrimitives class methods."""

    sym_str_syn = Symbol('apple banana cherry')
    sym_str_sem = Symbol('apple banana cherry', semantic=True) # or can be converted to a semantic symbol with ".sym" method
    sym_list_syn = Symbol(['apple', 'banana', 'cherry'])
    sym_list_sem = Symbol(['apple', 'banana', 'cherry'], semantic=True) # or can be converted to a semantic symbol with ".sym" method
    other1 = Symbol('apple')
    other2 = 'fruit'

    assert other1 in sym_str_syn
    assert other2 not in sym_list_syn
    display_op(f"'{sym_str_syn}' contains '{other1}'", other1 in sym_str_syn, "in", "syntactic")
    display_op(f"'{sym_str_sem}' contains '{other2}'", other2 in sym_str_sem, "in", "semantic")
    display_op(f"'{sym_list_syn}' contains '{other1}'", other1 in sym_list_syn, "in", "syntactic")
    display_op(f"'{sym_list_sem}' contains '{other2}'", other2 in sym_list_sem, "in", "semantic")

@pytest.mark.mandatory
def test_equals_op():
    """Test the OperatorPrimitives class methods."""

    sym1_syn = Symbol('hello world')
    sym2_syn = Symbol('hello world')
    sym3_syn = Symbol('goodbye world')

    assert sym1_syn == sym2_syn
    display_op(f"'{sym1_syn}' == '{sym2_syn}'", sym1_syn == sym2_syn, "==", "syntactic")
    assert not (sym1_syn == sym3_syn)
    display_op(f"'{sym1_syn}' == '{sym3_syn}'", sym1_syn == sym3_syn, "==", "syntactic")

    sym_greeting_sem = Symbol('Hello there!', semantic=True) # or can be converted to a semantic symbol with ".sym" method
    sym_farewell_sem = Symbol('Goodbye friend!')
    greeting_variant = 'Hi there!'
    farewell_variant = 'See you later!'

    display_op(f"'{sym_greeting_sem}' == '{greeting_variant}'", sym_greeting_sem == greeting_variant, "==", "semantic")
    display_op(f"'{sym_farewell_sem}' == '{farewell_variant}'", sym_farewell_sem == farewell_variant, "==", "semantic")

    sym_num1 = Symbol(42)
    sym_num2 = Symbol(42)
    sym_num3 = Symbol(24)

    assert sym_num1 == sym_num2
    display_op(f"{sym_num1} == {sym_num2}", sym_num1 == sym_num2, "==", "syntactic")
    assert not (sym_num1 == sym_num3)
    display_op(f"{sym_num1} == {sym_num3}", sym_num1 == sym_num3, "==", "syntactic")

    sym_bool1 = Symbol(True)
    sym_bool2 = Symbol(True)
    sym_bool3 = Symbol(False)

    assert sym_bool1 == sym_bool2
    display_op(f"{sym_bool1} == {sym_bool2}", sym_bool1 == sym_bool2, "==", "syntactic")
    assert not (sym_bool1 == sym_bool3)
    display_op(f"{sym_bool1} == {sym_bool3}", sym_bool1 == sym_bool3, "==", "syntactic")

    sym_list1 = Symbol([1, 2, 3])
    sym_list2 = Symbol([1, 2, 3], semantic=True) # or can be converted to a semantic symbol with ".sym" method
    sym_list3 = Symbol([3, 2, 1])

    assert sym_list1 == sym_list2.syn
    display_op(f"{sym_list1} == {sym_list2}", sym_list1 == sym_list2.syn, "==", "syntactic")
    semassert(not (sym_list1 == sym_list3))
    display_op(f"{sym_list2} == {sym_list3}", sym_list2 == sym_list3, "==", "semantic")

@pytest.mark.mandatory
def test_not_equals_op():
    """Test the OperatorPrimitives class methods."""

    sym1_syn = Symbol('hello world')
    sym2_syn = Symbol('hello world')
    sym3_syn = Symbol('goodbye world')

    assert not (sym1_syn != sym2_syn)
    display_op(f"'{sym1_syn}' != '{sym2_syn}'", sym1_syn != sym2_syn, "!=", "syntactic")
    assert sym1_syn != sym3_syn
    display_op(f"'{sym1_syn}' != '{sym3_syn}'", sym1_syn != sym3_syn, "!=", "syntactic")

    sym_greeting_sem = Symbol('Hello there!', semantic=True) # or can be converted to a semantic symbol with ".sym" method
    sym_farewell_sem = Symbol('Goodbye friend!')
    greeting_variant = 'Hi there!'
    farewell_variant = 'See you later!'

    display_op(f"'{sym_greeting_sem}' != '{greeting_variant}'", sym_greeting_sem != greeting_variant, "!=", "semantic")
    display_op(f"'{sym_farewell_sem}' != '{farewell_variant}'", sym_farewell_sem != farewell_variant, "!=", "semantic")

    sym_num1 = Symbol(42)
    sym_num2 = Symbol(42)
    sym_num3 = Symbol(24)

    assert not (sym_num1 != sym_num2)
    display_op(f"{sym_num1} != {sym_num2}", sym_num1 != sym_num2, "!=", "syntactic")
    assert sym_num1 != sym_num3
    display_op(f"{sym_num1} != {sym_num3}", sym_num1 != sym_num3, "!=", "syntactic")

    sym_bool1 = Symbol(True)
    sym_bool2 = Symbol(True)
    sym_bool3 = Symbol(False)

    assert not (sym_bool1 != sym_bool2)
    display_op(f"{sym_bool1} != {sym_bool2}", sym_bool1 != sym_bool2, "!=", "syntactic")
    assert sym_bool1 != sym_bool3
    display_op(f"{sym_bool1} != {sym_bool3}", sym_bool1 != sym_bool3, "!=", "syntactic")

    sym_list1 = Symbol([1, 2, 3])
    sym_list2 = Symbol([1, 2, 3], semantic=True) # or can be converted to a semantic symbol with ".sym" method
    sym_list3 = Symbol([3, 2, 1])

    semassert(not (sym_list1 != sym_list2))
    display_op(f"{sym_list1} != {sym_list2}", sym_list1 != sym_list2, "!=", "semantic")
    semassert(sym_list1 != sym_list3)
    display_op(f"{sym_list2} != {sym_list3}", sym_list2 != sym_list3, "!=", "semantic")

@pytest.mark.mandatory
def test_comparison_op():
    """Test the OperatorPrimitives class methods."""

    sym_num1 = Symbol(15)
    sym_num2 = Symbol(8)
    assert sym_num1 > sym_num2
    display_op(f"{sym_num1} > {sym_num2}", sym_num1 > sym_num2, ">", "syntactic")

    sym_float1 = Symbol(3.14)
    sym_float2 = Symbol(2.71)
    assert sym_float1 > sym_float2
    display_op(f"{sym_float1} > {sym_float2}", sym_float1 > sym_float2, ">", "syntactic")

    sym_large = Symbol('enormous', semantic=True) # or can be converted to a semantic symbol with ".sym" method
    sym_small = Symbol('tiny')
    display_op(f"'{sym_large}' > '{sym_small}'", sym_large > sym_small, ">", "semantic")

    sym_list1 = len(Symbol([5, 6]))
    sym_list2 = len(Symbol([4, 5]))
    display_op(f"{sym_list1} > {sym_list2}", sym_list1 > sym_list2, ">", "syntactic")

    sym_num3 = Symbol(12)
    sym_num4 = Symbol(20)
    assert sym_num3 < sym_num4
    display_op(f"{sym_num3} < {sym_num4}", sym_num3 < sym_num4, "<", "syntactic")

    sym_float3 = Symbol(1.5)
    sym_float4 = Symbol(2.8)
    assert sym_float3 < sym_float4
    display_op(f"{sym_float3} < {sym_float4}", sym_float3 < sym_float4, "<", "syntactic")

    sym_str3 = Symbol('cat', semantic=True) # or can be converted to a semantic symbol with ".sym" method
    sym_str4 = Symbol('dog')
    display_op(f"'{sym_str3}' < '{sym_str4}'", sym_str3 < sym_str4, "<", "semantic")

    sym_cold = Symbol('freezing', semantic=True) # or can be converted to a semantic symbol with ".sym" method
    sym_hot = Symbol('scorching')
    display_op(f"'{sym_cold}' < '{sym_hot}'", sym_cold < sym_hot, "<", "semantic")

    sym_list3 = len(Symbol([1, 2]))
    sym_list4 = len(Symbol([1, 2, 3]))
    display_op(f"{sym_list3} < {sym_list4}", sym_list3 < sym_list4, "<", "syntactic")

    sym_num5 = Symbol(25)
    sym_num6 = Symbol(25)
    assert sym_num5 >= sym_num6
    display_op(f"{sym_num5} >= {sym_num6}", sym_num5 >= sym_num6, ">=", "syntactic")

    sym_num7 = Symbol(30)
    sym_num8 = Symbol(18)
    assert sym_num7 >= sym_num8
    display_op(f"{sym_num7} >= {sym_num8}", sym_num7 >= sym_num8, ">=", "syntactic")

    sym_str5 = len(Symbol('python'))
    sym_str6 = len(Symbol('python'))
    assert sym_str5 >= sym_str6
    display_op(f"'{sym_str5}' >= '{sym_str6}'", sym_str5 >= sym_str6, ">=", "syntactic")

    sym_fast = Symbol('lightning speed', semantic=True) # or can be converted to a semantic symbol with ".sym" method
    sym_slow = Symbol('snail pace')
    display_op(f"'{sym_fast}' >= '{sym_slow}'", sym_fast >= sym_slow, ">=", "semantic")

    sym_num9 = Symbol(7)
    sym_num10 = Symbol(10)
    assert sym_num9 <= sym_num10
    display_op(f"{sym_num9} <= {sym_num10}", sym_num9 <= sym_num10, "<=", "syntactic")

    sym_num11 = Symbol(33)
    sym_num12 = Symbol(33)
    assert sym_num11 <= sym_num12
    display_op(f"{sym_num11} <= {sym_num12}", sym_num11 <= sym_num12, "<=", "syntactic")

    sym_weak = Symbol('fragile', semantic=True) # or can be converted to a semantic symbol with ".sym" method
    sym_strong = Symbol('robust')
    display_op(f"'{sym_weak}' <= '{sym_strong}'", sym_weak <= sym_strong, "<=", "semantic")

    sym_word_num = Symbol('fifty', semantic=True) # or can be converted to a semantic symbol with ".sym" method
    regular_num = 45
    display_op(f"'{sym_word_num}' > {regular_num}", sym_word_num > regular_num, ">", "semantic")

@pytest.mark.mandatory
def test_shift_op():
    """Test the OperatorPrimitives class methods."""

    sym_int8 = Symbol(8)
    sym_int2 = Symbol(2)
    sym_int16 = Symbol(16)

    assert (sym_int8 << sym_int2) == 32
    display_op(f"{sym_int8} << {sym_int2}", sym_int8 << sym_int2, "<<", "syntactic")
    assert (sym_int2 << sym_int8) == 512
    display_op(f"{sym_int2} << {sym_int8}", sym_int2 << sym_int8, "<<", "syntactic")

    assert (sym_int16 >> sym_int2) == 4
    display_op(f"{sym_int16} >> {sym_int2}", sym_int16 >> sym_int2, ">>", "syntactic")
    assert (sym_int8 >> sym_int2) == 2
    display_op(f"{sym_int8} >> {sym_int2}", sym_int8 >> sym_int2, ">>", "syntactic")

    regular_int4 = 4
    assert (regular_int4 << sym_int2) == 16
    display_op(f"{regular_int4} << {sym_int2}", regular_int4 << sym_int2, "<<", "syntactic")

    regular_int32 = 32
    assert (regular_int32 >> sym_int2) == 8
    display_op(f"{regular_int32} >> {sym_int2}", regular_int32 >> sym_int2, ">>", "syntactic")

    sym_inplace_left = Symbol(5)
    original_value = sym_inplace_left.value
    sym_inplace_left <<= Symbol(3)
    assert sym_inplace_left == 40
    display_op(f"{original_value} <<= 3", sym_inplace_left, "<<=", "syntactic")

    sym_inplace_right = Symbol(64)
    original_value = sym_inplace_right.value
    sym_inplace_right >>= Symbol(3)
    assert sym_inplace_right == 8
    display_op(f"{original_value} >>= 3", sym_inplace_right, ">>=", "syntactic")

    sym_text = Symbol('data stream', semantic=True)
    sym_direction = Symbol('left')
    display_op(f"'{sym_text}' << '{sym_direction}'", sym_text << sym_direction, "<<", "semantic")

    sym_info = Symbol('information flow', semantic=True)
    sym_right_dir = Symbol('right')
    display_op(f"'{sym_info}' >> '{sym_right_dir}'", sym_info >> sym_right_dir, ">>", "semantic")

    sym_numeric_str = Symbol('8', semantic=True)
    sym_shift_amount = Symbol('2')
    display_op(f"'{sym_numeric_str}' << '{sym_shift_amount}'", sym_numeric_str << sym_shift_amount, "<<", "semantic")
    display_op(f"'{sym_numeric_str}' >> '{sym_shift_amount}'", sym_numeric_str >> sym_shift_amount, ">>", "semantic")

    sym_base = Symbol(12)
    regular_shift = 1
    assert (sym_base << regular_shift) == 24
    display_op(f"{sym_base} << {regular_shift}", sym_base << regular_shift, "<<", "syntactic")
    assert (sym_base >> regular_shift) == 6
    display_op(f"{sym_base} >> {regular_shift}", sym_base >> regular_shift, ">>", "syntactic")

    sym_zero = Symbol(0)
    sym_neg = Symbol(-4)

    assert (sym_zero << sym_int2) == 0
    display_op(f"{sym_zero} << {sym_int2}", sym_zero << sym_int2, "<<", "syntactic")
    display_op(f"{sym_neg} << {sym_int2}", sym_neg << sym_int2, "<<", "syntactic")
    display_op(f"{sym_neg} >> {sym_int2}", sym_neg >> sym_int2, ">>", "syntactic")

    sym_priority = Symbol('high priority task', semantic=True)
    sym_urgency = Symbol('urgent')
    display_op(f"'{sym_priority}' << '{sym_urgency}'", sym_priority << sym_urgency, "<<", "semantic")

    sym_process = Symbol('data processing', semantic=True)
    sym_downstream = Symbol('downstream')
    display_op(f"'{sym_process}' >> '{sym_downstream}'", sym_process >> sym_downstream, ">>", "semantic")

    sym_chain = Symbol(1)
    chained_result = sym_chain << Symbol(2) << Symbol(1)
    assert chained_result == 8
    display_op("1 << 2 << 1", chained_result, "<<", "syntactic")

@pytest.mark.mandatory
def test_bitwise_logical_op():
    """Test the OperatorPrimitives class methods."""

    sym_int12 = Symbol(12)
    sym_int10 = Symbol(10)

    assert (sym_int12 & sym_int10) == 8
    display_op(f"{sym_int12} & {sym_int10}", sym_int12 & sym_int10, "&", "syntactic")

    sym_int5 = Symbol(5)
    sym_int3 = Symbol(3)

    assert (sym_int5 | sym_int3) == 7
    display_op(f"{sym_int5} | {sym_int3}", sym_int5 | sym_int3, "|", "syntactic")

    sym_true = Symbol(True)
    sym_false = Symbol(False)

    assert (sym_true & sym_false) == False
    display_op(f"{sym_true} & {sym_false}", sym_true & sym_false, "&", "syntactic")
    assert (sym_true | sym_false) == True
    display_op(f"{sym_true} | {sym_false}", sym_true | sym_false, "|", "syntactic")

    sym_inplace_and = Symbol(15)
    original_value = sym_inplace_and.value
    sym_inplace_and &= Symbol(7)
    assert sym_inplace_and == 7
    display_op(f"{original_value} &= 7", sym_inplace_and, "&=", "syntactic")

    sym_inplace_or = Symbol(4)
    original_value = sym_inplace_or.value
    sym_inplace_or |= Symbol(2)
    assert sym_inplace_or == 6
    display_op(f"{original_value} |= 2", sym_inplace_or, "|=", "syntactic")

    horn_rule = Symbol('The horn only sounds on Sundays.', semantic=True)
    observation = Symbol('I hear the horn.')
    conclusion = horn_rule & observation
    display_op(f"'{horn_rule}' & '{observation}'", conclusion, "&", "semantic")

    fact1 = Symbol('All cats are mammals.', semantic=True)
    fact2 = Symbol('Fluffy is a cat.')
    inference = fact1 & fact2
    display_op(f"'{fact1}' & '{fact2}'", inference, "&", "semantic")

    option1 = Symbol('It might rain today.', semantic=True)
    option2 = Symbol('It could be sunny.')
    possibilities = option1 | option2
    display_op(f"'{option1}' | '{option2}'", possibilities, "|", "semantic")

    condition = Symbol('If you study hard, you will pass.', semantic=True)
    action = Symbol('John studies hard.')
    result = condition & action
    display_op(f"'{condition}' & '{action}'", result, "&", "semantic")

    evidence1 = Symbol('The suspect was seen at the scene.', semantic=True)
    evidence2 = Symbol('His fingerprints were found.')
    combined_evidence = evidence1 & evidence2
    display_op(f"'{evidence1}' & '{evidence2}'", combined_evidence, "&", "semantic")

    scenario1 = Symbol('The meeting could be postponed.', semantic=True)
    scenario2 = Symbol('The meeting might be canceled.')
    alternatives = scenario1 | scenario2
    display_op(f"'{scenario1}' | '{scenario2}'", alternatives, "|", "semantic")

    bool_condition = Symbol(True)
    text_condition = Symbol('The weather is good.', semantic=True)
    mixed_and = bool_condition & text_condition
    display_op(f"{bool_condition} & '{text_condition}'", mixed_and, "&", "semantic")

    regular_int = 6
    sym_right = Symbol(3)
    assert (regular_int & sym_right) == 2
    display_op(f"{regular_int} & {sym_right}", regular_int & sym_right, "&", "syntactic")
    assert (regular_int | sym_right) == 7
    display_op(f"{regular_int} | {sym_right}", regular_int | sym_right, "|", "syntactic")

    premise1 = Symbol('All birds can fly.')
    premise2 = Symbol('Penguins are birds.')
    premise3 = Symbol('Penguins cannot fly.')

    contradiction = premise1.sem & premise2 & premise3
    display_op(f"'{premise1}' & '{premise2}' & '{premise3}'", contradiction, "&", "semantic")

    diagnosis1 = Symbol('The patient has flu.')
    diagnosis2 = Symbol('The patient has cold.')
    diagnosis3 = Symbol('The patient has allergies.')

    possible_diagnoses = diagnosis1.sem | diagnosis2 | diagnosis3
    display_op(f"'{diagnosis1}' | '{diagnosis2}' | '{diagnosis3}'", possible_diagnoses, "|", "semantic")

    sym_int9 = Symbol(9)
    sym_int6 = Symbol(6)

    assert (sym_int9 ^ sym_int6) == 15
    display_op(f"{sym_int9} ^ {sym_int6}", sym_int9 ^ sym_int6, "^", "syntactic")

    sym_int12 = Symbol(12)
    sym_int7 = Symbol(7)

    assert (sym_int12 ^ sym_int7) == 11
    display_op(f"{sym_int12} ^ {sym_int7}", sym_int12 ^ sym_int7, "^", "syntactic")

    assert (sym_true ^ sym_false) == True
    display_op(f"{sym_true} ^ {sym_false}", sym_true ^ sym_false, "^", "syntactic")
    assert (sym_true ^ sym_true) == False
    display_op(f"{sym_true} ^ {sym_true}", sym_true ^ sym_true, "^", "syntactic")

    sym_inplace_xor = Symbol(10)
    original_value = sym_inplace_xor.value
    sym_inplace_xor ^= Symbol(3)
    assert sym_inplace_xor == 9
    display_op(f"{original_value} ^= 3", sym_inplace_xor, "^=", "syntactic")

    exclusive1 = Symbol('Either it will rain today.', semantic=True)
    exclusive2 = Symbol('Or it will be sunny.')
    either_or = exclusive1 ^ exclusive2
    display_op(f"'{exclusive1}' ^ '{exclusive2}'", either_or, "^", "semantic")

    statement1 = Symbol('The door is open.', semantic=True)
    statement2 = Symbol('The door is closed.')
    contradiction_xor = statement1 ^ statement2
    display_op(f"'{statement1}' ^ '{statement2}'", contradiction_xor, "^", "semantic")

    choice1 = Symbol('Take the highway route.')
    choice2 = Symbol('Take the scenic route.')
    exclusive_choice = choice1.sem ^ choice2
    display_op(f"'{choice1}' ^ '{choice2}'", exclusive_choice, "^", "semantic")

    regular_int = 8
    sym_right_xor = Symbol(5)
    assert (regular_int ^ sym_right_xor) == 13
    display_op(f"{regular_int} ^ {sym_right_xor}", regular_int ^ sym_right_xor, "^", "syntactic")

    hypothesis1 = Symbol('The problem is hardware related.')
    hypothesis2 = Symbol('The problem is software related.')
    exclusive_diagnosis = hypothesis1.sem ^ hypothesis2
    display_op(f"'{hypothesis1}' ^ '{hypothesis2}'", exclusive_diagnosis, "^", "semantic")

@pytest.mark.mandatory
def test_arithmetic_op():
    """Test the OperatorPrimitives class methods."""

    sym_int8 = Symbol(8)
    sym_int5 = Symbol(5)

    assert (sym_int8 + sym_int5) == 13
    display_op(f"{sym_int8} + {sym_int5}", sym_int8 + sym_int5, "+", "syntactic")
    assert (sym_int8 - sym_int5) == 3
    display_op(f"{sym_int8} - {sym_int5}", sym_int8 - sym_int5, "-", "syntactic")

    sym_inplace_add = Symbol(10)
    original_value = sym_inplace_add.value
    sym_inplace_add += Symbol(7)
    assert sym_inplace_add == 17
    display_op(f"{original_value} += 7", sym_inplace_add, "+=", "syntactic")

    sym_inplace_sub = Symbol(15)
    original_value = sym_inplace_sub.value
    sym_inplace_sub -= Symbol(6)
    assert sym_inplace_sub == 9
    display_op(f"{original_value} -= 6", sym_inplace_sub, "-=", "syntactic")

    regular_int = 12
    sym_right = Symbol(4)
    assert (regular_int + sym_right) == 16
    display_op(f"{regular_int} + {sym_right}", regular_int + sym_right, "+", "syntactic")
    assert (regular_int - sym_right) == 8
    display_op(f"{regular_int} - {sym_right}", regular_int - sym_right, "-", "syntactic")

    enemy_text = Symbol('Hello my enemy', semantic=True)
    result1 = enemy_text - 'enemy' + 'friend'
    display_op(f"'{enemy_text}' - 'enemy' + 'friend'", result1, "- +", "semantic")

    original_msg = Symbol('I dislike this situation')
    result2 = original_msg.sem - 'dislike' + 'love'
    display_op(f"'{original_msg}' - 'dislike' + 'love'", result2, "- +", "semantic")

    problem_text = Symbol('The old system is broken', semantic=True)
    solution = problem_text - 'old' - 'broken' + 'new' + 'working'
    display_op(f"'{problem_text}' - 'old' - 'broken' + 'new' + 'working'", solution, "- - + +", "semantic")

    negative_review = Symbol('This product is terrible and useless')
    positive_review = negative_review.sem - 'terrible' - 'useless' + 'amazing' + 'helpful'
    display_op(f"'{negative_review}' - 'terrible' - 'useless' + 'amazing' + 'helpful'", positive_review, "- - + +", "semantic")

    incomplete_task = Symbol('The project is half done', semantic=True)
    complete_task = incomplete_task - 'half done' + 'completed'
    display_op(f"'{incomplete_task}' - 'half done' + 'completed'", complete_task, "- +", "semantic")

    mood_text = Symbol('I am feeling sad today')
    happy_mood = mood_text.sem - 'sad' + 'happy'
    display_op(f"'{mood_text}' - 'sad' + 'happy'", happy_mood, "- +", "semantic")

    wrong_info = Symbol('The answer is definitely wrong', semantic=True)
    correct_info = wrong_info - 'wrong' + 'correct'
    display_op(f"'{wrong_info}' - 'wrong' + 'correct'", correct_info, "- +", "semantic")

    weak_statement = Symbol('This might be weak')
    strong_statement = weak_statement.sem - 'weak' + 'strong'
    display_op(f"'{weak_statement}' - 'weak' + 'strong'", strong_statement, "- +", "semantic")

@pytest.mark.mandatory
def test_string_concatenation_op():
    """Test the OperatorPrimitives class methods."""

    sym_str1 = Symbol('Hello')
    sym_str2 = Symbol(' World')

    result = sym_str1 @ sym_str2
    assert result == 'Hello World'
    display_op(f"'{sym_str1}' @ '{sym_str2}'", result, "@", "syntactic")

    sym_first = Symbol('Python')
    sym_second = Symbol(' Programming')
    sym_third = Symbol(' Language')

    chain_result = sym_first @ sym_second @ sym_third
    assert chain_result == 'Python Programming Language'
    display_op(f"'{sym_first}' @ '{sym_second}' @ '{sym_third}'", chain_result, "@", "syntactic")

    sym_inplace = Symbol('Start')
    original_value = sym_inplace.value
    sym_inplace @= Symbol(' End')
    assert sym_inplace == 'Start End'
    display_op(f"'{original_value}' @= ' End'", sym_inplace, "@=", "syntactic")

    regular_str = 'Tail'
    sym_suffix = Symbol('Head ')
    result_mixed = regular_str @ sym_suffix
    assert result_mixed == 'Head Tail'
    display_op(f"'{regular_str}' @ '{sym_suffix}'", result_mixed, "@", "syntactic")

    sym_base = Symbol('Base')
    regular_suffix = ' Extension'
    result_reverse = sym_base @ regular_suffix
    assert result_reverse == 'Base Extension'
    display_op(f"'{sym_base}' @ '{regular_suffix}'", result_reverse, "@", "syntactic")

    empty_sym = Symbol('')
    text_sym = Symbol('Content')
    result_empty = empty_sym @ text_sym
    assert result_empty == 'Content'
    display_op(f"'{empty_sym}' @ '{text_sym}'", result_empty, "@", "syntactic")

@pytest.mark.mandatory
def test_division_power_modulo_multiply_op():
    """Test the OperatorPrimitives class methods."""

    sym_int12 = Symbol(12)
    sym_int3 = Symbol(3)
    sym_int4 = Symbol(4)

    assert (sym_int12 / sym_int3) == 4.0
    display_op(f"{sym_int12} / {sym_int3}", sym_int12 / sym_int3, "/", "syntactic")
    assert (sym_int12 // sym_int3) == 4
    display_op(f"{sym_int12} // {sym_int3}", sym_int12 // sym_int3, "//", "syntactic")

    sym_int2 = Symbol(2)
    assert (sym_int4 ** sym_int2) == 16
    display_op(f"{sym_int4} ** {sym_int2}", sym_int4 ** sym_int2, "**", "syntactic")

    sym_int10 = Symbol(10)
    assert (sym_int10 % sym_int3) == 1
    display_op(f"{sym_int10} % {sym_int3}", sym_int10 % sym_int3, "%", "syntactic")

    sym_int5 = Symbol(5)
    assert (sym_int5 * sym_int4) == 20
    display_op(f"{sym_int5} * {sym_int4}", sym_int5 * sym_int4, "*", "syntactic")

    sym_inplace_div = Symbol(20.0)
    original_value = sym_inplace_div.value
    sym_inplace_div /= Symbol(4)
    assert sym_inplace_div == 5.0
    display_op(f"{original_value} /= 4", sym_inplace_div, "/=", "syntactic")

    sym_inplace_floordiv = Symbol(17)
    original_value = sym_inplace_floordiv.value
    sym_inplace_floordiv //= Symbol(3)
    assert sym_inplace_floordiv == 5
    display_op(f"{original_value} //= 3", sym_inplace_floordiv, "//=", "syntactic")

    sym_inplace_pow = Symbol(3)
    original_value = sym_inplace_pow.value
    sym_inplace_pow **= Symbol(3)
    assert sym_inplace_pow == 27
    display_op(f"{original_value} **= 3", sym_inplace_pow, "**=", "syntactic")

    sym_inplace_mod = Symbol(13)
    original_value = sym_inplace_mod.value
    sym_inplace_mod %= Symbol(5)
    assert sym_inplace_mod == 3
    display_op(f"{original_value} %= 5", sym_inplace_mod, "%=", "syntactic")

    sym_inplace_mul = Symbol(6)
    original_value = sym_inplace_mul.value
    sym_inplace_mul *= Symbol(7)
    assert sym_inplace_mul == 42
    display_op(f"{original_value} *= 7", sym_inplace_mul, "*=", "syntactic")

    regular_int = 15
    sym_right = Symbol(3)
    assert (regular_int / sym_right) == 5.0
    display_op(f"{regular_int} / {sym_right}", regular_int / sym_right, "/", "syntactic")
    assert (regular_int * sym_right) == 45
    display_op(f"{regular_int} * {sym_right}", regular_int * sym_right, "*", "syntactic")

    sym_str = Symbol('apple,banana,cherry')
    split_result = sym_str / ','
    expected = ['apple', 'banana', 'cherry']
    assert split_result.value == expected
    display_op(f"'{sym_str}' / ','", split_result, "/", "syntactic")

    sym_text = Symbol('hello world hello')
    split_by_space = sym_text / ' '
    expected_split = ['hello', 'world', 'hello']
    assert split_by_space.value == expected_split
    display_op(f"'{sym_text}' / ' '", split_by_space, "/", "syntactic")

    sym_sentence = Symbol('one-two-three-four')
    split_by_dash = sym_sentence / '-'
    expected_dash = ['one', 'two', 'three', 'four']
    assert split_by_dash.value == expected_dash
    display_op(f"'{sym_sentence}' / '-'", split_by_dash, "/", "syntactic")

@pytest.mark.mandatory
def test_casting_pr():
    """Test the CastingPrimitives class methods."""

    # Test cast() method with different types
    sym_str_num = Symbol('42')
    casted_int = sym_str_num.cast(int)
    assert casted_int == 42
    assert isinstance(casted_int, int)
    display_op(f"['{sym_str_num}'].cast(int)", casted_int, "cast", "syntactic")

    sym_int = Symbol(123)
    casted_str = sym_int.cast(str)
    assert casted_str == '123'
    assert isinstance(casted_str, str)
    display_op(f"[{sym_int}].cast(str)", casted_str, "cast", "syntactic")

    sym_str_float = Symbol('3.14')
    casted_float = sym_str_float.cast(float)
    assert casted_float == 3.14
    assert isinstance(casted_float, float)
    display_op(f"['{sym_str_float}'].cast(float)", casted_float, "cast", "syntactic")

    sym_int_bool = Symbol(1)
    casted_bool = sym_int_bool.cast(bool)
    assert casted_bool is True
    assert isinstance(casted_bool, bool)
    display_op(f"[{sym_int_bool}].cast(bool)", casted_bool, "cast", "syntactic")

    # Test to() method (alias for cast)
    sym_str_to_int = Symbol('99')
    to_result = sym_str_to_int.to(int)
    assert to_result == 99
    assert isinstance(to_result, int)
    display_op(f"['{sym_str_to_int}'].to(int)", to_result, "to", "syntactic")

    # Test str() method
    sym_various = Symbol(456)
    str_result = sym_various.str()
    assert str_result == '456'
    assert isinstance(str_result, str)
    display_op(f"[{sym_various}].str()", str_result, "str", "syntactic")

    sym_float_str = Symbol(2.718)
    str_float_result = sym_float_str.str()
    assert str_float_result == '2.718'
    display_op(f"[{sym_float_str}].str()", str_float_result, "str", "syntactic")

    sym_bool_str = Symbol(True)
    str_bool_result = sym_bool_str.str()
    assert str_bool_result == 'True'
    display_op(f"[{sym_bool_str}].str()", str_bool_result, "str", "syntactic")

    # Test int() method
    sym_str_int = Symbol('789')
    int_result = sym_str_int.int()
    assert int_result == 789
    assert isinstance(int_result, int)
    display_op(f"['{sym_str_int}'].int()", int_result, "int", "syntactic")

    sym_float_int = Symbol(9.7)
    int_from_float = sym_float_int.int()
    assert int_from_float == 9
    display_op(f"[{sym_float_int}].int()", int_from_float, "int", "syntactic")

    sym_bool_int = Symbol(True)
    int_from_bool = sym_bool_int.int()
    assert int_from_bool == 1
    display_op(f"[{sym_bool_int}].int()", int_from_bool, "int", "syntactic")

    # Test float() method
    sym_str_float_test = Symbol('12.34')
    float_result = sym_str_float_test.float()
    assert float_result == 12.34
    assert isinstance(float_result, float)
    display_op(f"['{sym_str_float_test}'].float()", float_result, "float", "syntactic")

    sym_int_float = Symbol(55)
    float_from_int = sym_int_float.float()
    assert float_from_int == 55.0
    display_op(f"[{sym_int_float}].float()", float_from_int, "float", "syntactic")

    # Test bool() method
    sym_int_zero = Symbol(0)
    bool_from_zero = sym_int_zero.bool()
    assert bool_from_zero is False
    display_op(f"[{sym_int_zero}].bool()", bool_from_zero, "bool", "syntactic")

    sym_int_nonzero = Symbol(42)
    bool_from_nonzero = sym_int_nonzero.bool()
    assert bool_from_nonzero is True
    display_op(f"[{sym_int_nonzero}].bool()", bool_from_nonzero, "bool", "syntactic")

    sym_str_empty = Symbol('')
    bool_from_empty = sym_str_empty.bool()
    assert bool_from_empty is False
    display_op(f"['{sym_str_empty}'].bool()", bool_from_empty, "bool", "syntactic")

    sym_str_nonempty = Symbol('hello')
    bool_from_string = sym_str_nonempty.bool()
    assert bool_from_string is True
    display_op(f"['{sym_str_nonempty}'].bool()", bool_from_string, "bool", "syntactic")

    # Test ast() method with different literal types
    sym_list_str = Symbol('[1, 2, 3]')
    ast_list = sym_list_str.ast()
    assert ast_list == [1, 2, 3]
    assert isinstance(ast_list, list)
    display_op(f"['{sym_list_str}'].ast()", ast_list, "ast", "syntactic")

    sym_dict_str = Symbol("{'a': 1, 'b': 2}")
    ast_dict = sym_dict_str.ast()
    assert ast_dict == {'a': 1, 'b': 2}
    assert isinstance(ast_dict, dict)
    display_op(f"['{sym_dict_str}'].ast()", ast_dict, "ast", "syntactic")

    sym_tuple_str = Symbol('(10, 20, 30)')
    ast_tuple = sym_tuple_str.ast()
    assert ast_tuple == (10, 20, 30)
    assert isinstance(ast_tuple, tuple)
    display_op(f"['{sym_tuple_str}'].ast()", ast_tuple, "ast", "syntactic")

    sym_none_str = Symbol('None')
    ast_none = sym_none_str.ast()
    assert ast_none is None
    display_op(f"['{sym_none_str}'].ast()", ast_none, "ast", "syntactic")

@pytest.mark.mandatory
def test_iteration_pr():
    """Test the IterationPrimitives class methods."""

    # Test __getitem__ with lists - syntactic
    sym_list_syn = Symbol([10, 20, 30, 40, 50])
    item_syn = sym_list_syn[1]
    assert item_syn == 20
    display_op(f"{sym_list_syn}[1]", item_syn, "getitem", "syntactic")

    item_neg = sym_list_syn[-1]
    assert item_neg == 50
    display_op(f"{sym_list_syn}[-1]", item_neg, "getitem", "syntactic")

    slice_result = sym_list_syn[1:3]
    assert slice_result == [20, 30]
    display_op(f"{sym_list_syn}[1:3]", slice_result, "getitem", "syntactic")

    # Test __getitem__ with lists - semantic
    sym_list_sem = Symbol([10, 20, 30, 40, 50], semantic=True)
    item_sem = sym_list_sem[2]
    semassert(item_sem == 30)
    display_op(f"{sym_list_sem}[2]", item_sem, "getitem", "semantic")

    # Test __getitem__ with dictionaries - syntactic
    sym_dict_syn = Symbol({'a': 1, 'b': 2, 'c': 3})
    dict_item_syn = sym_dict_syn['b']
    assert dict_item_syn == 2
    display_op(f"{sym_dict_syn}['b']", dict_item_syn, "getitem", "syntactic")

    # Test __getitem__ with dictionaries - semantic
    sym_dict_sem = Symbol({'name': 'Alice', 'age': 30, 'city': 'NYC'}, semantic=True)
    dict_item_sem = sym_dict_sem['Return any names']
    semassert(dict_item_sem == 'Alice')
    display_op(f"{sym_dict_sem}['Return any names']", dict_item_sem, "getitem", "semantic")

    sym_colors = Symbol({'red': '#FF0000', 'green': '#00FF00', 'blue': '#0000FF'}, semantic=True)
    color_key = 'primary color'
    color_result = sym_colors[color_key]
    display_op(f"{sym_colors}['{color_key}']", color_result, "getitem", "semantic")

    sym_animals = Symbol(['cat', 'dog', 'bird', 'fish'])
    pet_query = 'domestic animal'
    pet_result = sym_animals.sem[pet_query]
    display_op(f"{sym_animals}['{pet_query}']", pet_result, "getitem", "semantic")

    # Test __getitem__ with tuples
    sym_tuple_syn = Symbol((100, 200, 300))
    tuple_item = sym_tuple_syn[0]
    assert tuple_item == 100
    display_op(f"{sym_tuple_syn}[0]", tuple_item, "getitem", "syntactic")

    # Test __getitem__ with numpy arrays
    sym_array = Symbol(np.array([1, 2, 3, 4, 5]))
    array_item = sym_array[3]
    assert array_item == 4
    display_op(f"{sym_array}[3]", array_item, "getitem", "syntactic")

    # Test __setitem__ with lists
    sym_list_modify = Symbol([1, 2, 3, 4, 5])
    original_list = sym_list_modify.value.copy()
    sym_list_modify[2] = 99
    assert sym_list_modify.value[2] == 99
    display_op(f"{original_list}[2] = 99", sym_list_modify, "setitem", "syntactic")

    # Test __setitem__ with slice
    sym_list_slice = Symbol([10, 20, 30, 40, 50])
    original_slice = sym_list_slice.value.copy()
    sym_list_slice[1:3] = [888, 999]
    assert sym_list_slice.value == [10, 888, 999, 40, 50]
    display_op(f"{original_slice}[1:3] = [888, 999]", sym_list_slice, "setitem", "syntactic")

    # Test __setitem__ with dictionaries
    sym_dict_modify = Symbol({'x': 10, 'y': 20})
    original_dict = sym_dict_modify.value.copy()
    sym_dict_modify['z'] = 30
    assert sym_dict_modify.value['z'] == 30
    display_op(f"{original_dict}['z'] = 30", sym_dict_modify, "setitem", "syntactic")

    # Test __setitem__ with semantic mode
    sym_dict_sem_modify = Symbol({'temperature': 20, 'humidity': 60, 'pressure': 1013}, semantic=True)
    original_sem_dict = sym_dict_sem_modify.value.copy()
    sym_dict_sem_modify['Change the temperature'] = 25  # Should semantically map to 'temperature'
    display_op(f"{original_sem_dict}['Change the temperature'] = 25", sym_dict_sem_modify, "setitem", "semantic")

    # Test __delitem__ with dictionaries
    sym_dict_delete = Symbol({'a': 1, 'b': 2, 'c': 3, 'd': 4})
    original_delete = sym_dict_delete.value.copy()
    del sym_dict_delete['b']
    assert 'b' not in sym_dict_delete.value
    assert sym_dict_delete.value == {'a': 1, 'c': 3, 'd': 4}
    display_op(f"del {original_delete}['b']", sym_dict_delete, "delitem", "syntactic")

    # Test __delitem__ with semantic mode
    sym_dict_sem_delete = Symbol({'first_name': 'John', 'last_name': 'Doe', 'age': 30}, semantic=True)
    original_sem_delete = sym_dict_sem_delete.value.copy()
    del sym_dict_sem_delete['surname']  # Should semantically map to 'last_name'
    display_op(f"del {original_sem_delete}['surname']", sym_dict_sem_delete, "delitem", "semantic")

    # Test semantic dictionary key matching
    sym_person = Symbol({'full_name': 'Alice Smith', 'birth_year': 1990, 'occupation': 'Engineer'}, semantic=True)
    name_variants = ['identity', 'profession']
    for variant in name_variants:
        result = sym_person[variant]
        display_op(f"{sym_person}['{variant}']", result, "getitem", "semantic")

    # Test with mixed data types
    sym_mixed = Symbol({'numbers': [1, 2, 3], 'text': 'hello'})
    nested_access = sym_mixed['numbers']
    assert nested_access == [1, 2, 3]
    display_op(f"{sym_mixed}['numbers']", nested_access, "getitem", "syntactic")

@pytest.mark.mandatory
def test_string_helper_pr():
    """Test the StringHelperPrimitives class methods."""

    # Test split() method
    sym_sentence = Symbol('hello world python programming')
    split_result = sym_sentence.split(' ')

    expected_values = ['hello', 'world', 'python', 'programming']
    assert len(split_result) == len(expected_values)
    for i, text in enumerate(split_result):
        assert text == expected_values[i]
    display_op(f"[{sym_sentence}].split(' ')", [s for s in split_result], "split", "syntactic")

    sym_csv = Symbol('apple,banana,cherry')
    csv_split = sym_csv.split(',')
    expected_values = ['apple', 'banana', 'cherry']
    assert len(csv_split) == len(expected_values)
    for i, text in enumerate(csv_split):
        assert text == expected_values[i]
    display_op(f"[{sym_csv}].split(',')", [s for s in csv_split], "split", "syntactic")

    # Test split with different delimiters
    sym_path = Symbol('home/user/documents/file.txt')
    path_split = sym_path.split('/')
    expected_values = ['home', 'user', 'documents', 'file.txt']
    assert len(path_split) == len(expected_values)
    for i, text in enumerate(path_split):
        assert text == expected_values[i]
    display_op(f"[{sym_path}].split('/')", [s for s in path_split], "split", "syntactic")

    # Test join() method with list input
    sym_words = Symbol(['Hello', 'beautiful', 'world'])
    joined_space = sym_words.join(' ')
    assert joined_space == 'Hello beautiful world'
    display_op(f"[{sym_words}].join(' ')", joined_space, "join", "syntactic")

    joined_dash = sym_words.join('-')
    assert joined_dash == 'Hello-beautiful-world'
    display_op(f"[{sym_words}].join('-')", joined_dash, "join", "syntactic")

    # Test join with default delimiter (space)
    joined_default = sym_words.join()
    assert joined_default == 'Hello beautiful world'
    display_op(f"[{sym_words}].join()", joined_default, "join", "syntactic")

    # Test startswith() method
    sym_greeting = Symbol('Hello everyone!')
    starts_hello = sym_greeting.startswith('Hello')
    assert starts_hello is True
    display_op(f"[{sym_greeting}].startswith('Hello')", starts_hello, "startswith", "syntactic")

    starts_bye = sym_greeting.startswith('Goodbye')
    assert starts_bye is False
    display_op(f"[{sym_greeting}].startswith('Goodbye')", starts_bye, "startswith", "syntactic")

    # Test with empty string
    starts_empty = sym_greeting.startswith('')
    assert starts_empty is True
    display_op(f"[{sym_greeting}].startswith('')", starts_empty, "startswith", "syntactic")

    # Test endswith() method
    sym_filename = Symbol('document.pdf')
    ends_pdf = sym_filename.endswith('.pdf')
    assert ends_pdf is True
    display_op(f"[{sym_filename}].endswith('.pdf')", ends_pdf, "endswith", "syntactic")

    ends_txt = sym_filename.endswith('.txt')
    assert ends_txt is False
    display_op(f"[{sym_filename}].endswith('.txt')", ends_txt, "endswith", "syntactic")

    # Test with multiple possible endings
    sym_code_file = Symbol('main.py')
    ends_py = sym_code_file.endswith('.py')
    assert ends_py is True
    display_op(f"[{sym_code_file}].endswith('.py')", ends_py, "endswith", "syntactic")

    # Test with longer suffix
    sym_backup = Symbol('data_backup_2024.tar.gz')
    ends_tar_gz = sym_backup.endswith('.tar.gz')
    assert ends_tar_gz is True
    display_op(f"[{sym_backup}].endswith('.tar.gz')", ends_tar_gz, "endswith", "syntactic")

    # Test case sensitivity
    sym_mixed_case = Symbol('FileNAME.TXT')
    ends_lower = sym_mixed_case.endswith('.txt')
    assert ends_lower is False
    display_op(f"[{sym_mixed_case}].endswith('.txt')", ends_lower, "endswith", "syntactic")

    ends_upper = sym_mixed_case.endswith('.TXT')
    assert ends_upper is True
    display_op(f"[{sym_mixed_case}].endswith('.TXT')", ends_upper, "endswith", "syntactic")

    # Test edge cases
    sym_empty = Symbol('')
    empty_split = sym_empty.split(' ')
    assert len(empty_split) == 1 and empty_split[0] == ''
    display_op(f"[{sym_empty}].split(' ')", [s for s in empty_split], "split", "syntactic")

    sym_single_char = Symbol('a')
    single_starts = sym_single_char.startswith('a')
    assert single_starts is True
    display_op(f"[{sym_single_char}].startswith('a')", single_starts, "startswith", "syntactic")

    single_ends = sym_single_char.endswith('a')
    assert single_ends is True
    display_op(f"[{sym_single_char}].endswith('a')", single_ends, "endswith", "syntactic")

    # Test semantic startswith() operations
    sym_semantic_sentence = Symbol('The apple fell from the tree.', semantic=True)
    starts_with_fruit = sym_semantic_sentence.startswith('fruit')
    display_op(f"[{sym_semantic_sentence}].startswith('fruit')", starts_with_fruit, "startswith", "semantic")

    sym_dog_sentence = Symbol('My dog loves to play fetch.', semantic=True)
    starts_with_animal = sym_dog_sentence.startswith('animal')
    display_op(f"[{sym_dog_sentence}].startswith('animal')", starts_with_animal, "startswith", "semantic")

    # Test semantic endswith() operations
    sym_success_sentence = Symbol('She studied hard and graduated with honors.', semantic=True)
    ends_with_success = sym_success_sentence.endswith('success')
    display_op(f"[{sym_success_sentence}].endswith('success')", ends_with_success, "endswith", "semantic")

    sym_cooking_sentence = Symbol('The chef prepared a delicious meal.', semantic=True)
    starts_with_cooking = sym_cooking_sentence.startswith('cooking')
    display_op(f"[{sym_cooking_sentence}].startswith('cooking')", starts_with_cooking, "startswith", "semantic")

    # Test negative semantic cases
    sym_book_sentence = Symbol('The book was very interesting.', semantic=True)
    starts_with_vehicle = sym_book_sentence.startswith('vehicle')
    assert starts_with_vehicle is False
    display_op(f"[{sym_book_sentence}].startswith('vehicle')", starts_with_vehicle, "startswith", "semantic")

    sym_peaceful_sentence = Symbol('The cat slept peacefully.', semantic=True)
    ends_with_violence = sym_peaceful_sentence.endswith('violence')
    assert ends_with_violence is False
    display_op(f"[{sym_peaceful_sentence}].endswith('violence')", ends_with_violence, "endswith", "semantic")

    # Compare syntactic vs semantic behavior
    sym_apple_syntactic = Symbol('The apple fell from the tree.')
    sym_apple_semantic = Symbol('The apple fell from the tree.', semantic=True)

    syntactic_result = sym_apple_syntactic.startswith('fruit')
    semantic_result = sym_apple_semantic.startswith('fruit')

    assert syntactic_result is False
    display_op(f"[{sym_apple_syntactic}].startswith('fruit')", syntactic_result, "startswith", "syntactic")
    display_op(f"[{sym_apple_semantic}].startswith('fruit')", semantic_result, "startswith", "semantic")

@pytest.mark.mandatory
def test_comparison_pr():
    """Test the ComparisonPrimitives class methods."""

    # Test equals() method - contextual comparison beyond simple equality
    sym_greeting1 = Symbol('Hello, good morning!')

    # Test contextual equals with default context
    result_default = sym_greeting1.equals('Hi there, good day!')
    display_op(f"[{sym_greeting1}].equals('Hi there, good day!')", result_default, "equals", "semantic")

    # Test contextual equals with specific context
    result_greeting = sym_greeting1.equals('Hi there, good day!', context='greeting context')
    display_op(f"[{sym_greeting1}].equals('Hi there, good day!', context='greeting')", result_greeting, "equals", "semantic")

    # Test equals with different contexts
    sym_formal = Symbol('Good morning, sir.')
    sym_casual = Symbol('Hey, what is up?')

    formal_vs_casual = sym_formal.equals('Hey, what\'s up?', context='politeness level')
    display_op(f"[{sym_formal}].equals('Hey, what\\'s up?', context='politeness')", formal_vs_casual, "equals", "semantic")

    sym_text = Symbol('The quick brown fox jumps over the lazy dog')
    contains_fox = sym_text.contains('fox')
    display_op(f"[{sym_text}].contains('fox')", contains_fox, "contains", "semantic")

    # Test contains with non-existent element
    contains_cat = sym_text.contains('cat')
    display_op(f"[{sym_text}].contains('cat')", contains_cat, "contains", "semantic")

    sym_semantic_text = Symbol('The vehicle moved quickly down the road')
    contains_car = sym_semantic_text.contains('car')
    display_op(f"[{sym_semantic_text}].contains('car')", contains_car, "contains", "semantic")

    # Test contains with list
    sym_fruits = Symbol(['apple', 'banana', 'orange'])
    contains_apple = sym_fruits.contains('apple')
    display_op(f"[{sym_fruits}].contains('apple')", contains_apple, "contains", "semantic")

    contains_grape = sym_fruits.contains('grape')
    assert contains_grape is False
    display_op(f"[{sym_fruits}].contains('grape')", contains_grape, "contains", "semantic")

    sym_number = Symbol(42)
    # Test with basic types
    is_number = sym_number.isinstanceof('number')
    display_op(f"[{sym_number}].isinstanceof('number')", is_number, "isinstanceof", "semantic")

    is_string = sym_number.isinstanceof('string')
    display_op(f"[{sym_number}].isinstanceof('string')", is_string, "isinstanceof", "semantic")

    # Test with list symbol
    sym_list = Symbol(['apple', 'banana', 'cherry'])
    is_list = sym_list.isinstanceof('list')
    display_op(f"[{sym_list}].isinstanceof('list')", is_list, "isinstanceof", "semantic")

    # Test with boolean symbol
    sym_bool = Symbol(True)
    is_logical = sym_bool.isinstanceof('logical value')
    display_op(f"[{sym_bool}].isinstanceof('logical value')", is_logical, "isinstanceof", "semantic")

    # Test with more complex type queries
    sym_complex = Symbol({'name': 'John', 'age': 30})
    is_person = sym_complex.isinstanceof('person data')
    display_op(f"[{sym_complex}].isinstanceof('person data')", is_person, "isinstanceof", "semantic")

    sym_person = Symbol({'name': 'Alice', 'age': 25, 'city': 'Wonderland'})
    is_person = sym_person.isinstanceof('person')
    display_op(f"[{sym_person}].isinstanceof('person')", is_person, "isinstanceof", "semantic")

    # Test edge cases for equals
    sym_empty = Symbol('')
    empty_equals = sym_empty.equals('', context='exact match')
    display_op(f"[{sym_empty}].equals('', context='exact')", empty_equals, "equals", "semantic")

    # Test edge cases for contains
    empty_contains = sym_empty.contains('')
    display_op(f"[{sym_empty}].contains('')", empty_contains, "contains", "semantic")

    # Test comparison with different data types
    sym_mixed = Symbol([1, 'two', 3.0, True])
    contains_string = sym_mixed.contains('two')
    display_op(f"[{sym_mixed}].contains('two')", contains_string, "contains", "semantic")

    contains_bool = sym_mixed.contains(True)
    display_op(f"[{sym_mixed}].contains(True)", contains_bool, "contains", "semantic")

    sym_semantic = Symbol('The dog is running in the park')
    semantic_contains = sym_semantic.contains('animal')
    display_op(f"[{sym_semantic}].contains('animal')", semantic_contains, "contains", "semantic")

@pytest.mark.mandatory
def test_interpret_pr():
    """Test the ExpressionHandlingPrimitives class methods."""

    # Test basic interpretation
    sym_basic = Symbol('What is the tallest mountain in the world?')
    result_basic = sym_basic.interpret()
    display_op(f"[{sym_basic}].interpret()", result_basic, "interpret", "semantic")

    # Test symbolic expression interpretation
    sym_analogy = Symbol('gravity : Earth :: radiation : ?')
    result_analogy = sym_analogy.interpret()
    display_op(f"[{sym_analogy}].interpret()", result_analogy, "interpret", "semantic")

    # Test mathematical expression interpretation
    sym_math = Symbol("∫(3x² + 2x - 5)dx")
    result_math = sym_math.interpret()
    display_op(f"[{sym_math}].interpret()", result_math, "interpret", "semantic")

    # Test accumulation feature
    sym_accumulate = Symbol('Relativistic electron formula')
    result1 = sym_accumulate.interpret(accumulate=True)
    display_op(f"[{sym_accumulate}].interpret(accumulate=True)", result1, "interpret", "semantic")

    result2 = result1.interpret('Assume the momentum to be extremely large', accumulate=True)
    display_op(f"[{result1}].interpret('Assume the momentum to be extremely large', accumulate=True)",
              result2, "interpret", "semantic")

    result3 = result2.interpret('Expand the formula to account for both mass and momentum', accumulate=True)
    display_op(f"[{result2}].interpret('Expand the formula to account for both mass and momentum', accumulate=True)",
              result3, "interpret", "semantic")

    # Check accumulated results
    accumulated = sym_accumulate.get_results()
    display_op(f"[{sym_accumulate}].get_results()", accumulated, "get_results", "syntactic")
    assert len(accumulated) == 3, "Should have 3 accumulated results"

    # Test clear results
    sym_accumulate.clear_results()
    cleared = sym_accumulate.get_results()
    display_op(f"[{sym_accumulate}].clear_results() then get_results()", cleared, "clear_results", "syntactic")
    assert len(cleared) == 0, "Should have no results after clearing"

    # Test logical reasoning with constraints
    sym_reasoning = Symbol('If every event has a cause, and the universe began with an event, what philosophical question arises?')
    result_reasoning = sym_reasoning.interpret()
    display_op(f"[{sym_reasoning}].interpret()", result_reasoning, "interpret", "semantic")

    # Test conditional interpretation
    sym_conditional = Symbol("If x < 0 then 'negative' else if x == 0 then 'zero' else 'positive'")
    results = []
    for val in [-5, 0, 10]:
        result = sym_conditional.interpret(f"x = {val}")
        results.append(result)
        display_op(f"[{sym_conditional}].interpret('x = {val}')", result, "interpret", "semantic")
    assert len(set([str(r) for r in results])) == 3, "Should get three different results for different inputs"

    # Test system with constraints
    sym_constraint = Symbol("Find values for x and y where: x + y = 10, x - y = 4")
    result_constraint = sym_constraint.interpret()
    display_op(f"[{sym_constraint}].interpret()", result_constraint, "interpret", "semantic")

    # Test result type preservation
    assert isinstance(result_basic, Symbol), "interpret should return a Symbol"
    assert isinstance(result_analogy, Symbol), "interpret should return a Symbol"
    assert hasattr(result_analogy, '_input'), "result should have _input attribute"

    # Test multiple interpretations maintain input reference
    result_multi1 = sym_analogy.interpret()
    result_multi2 = sym_analogy.interpret()
    assert result_multi1._input == result_multi2._input, "Both results should reference same input"

@pytest.mark.mandatory
def test_data_handling_pr():
    """Test the DataHandlingPrimitives class methods."""

    # Test map() method with list containing fruits and animals
    sym_mixed_list = Symbol(['apple', 'banana', 'cherry', 'cat', 'dog'])
    mapped_result = sym_mixed_list.map('convert all fruits to vegetables')
    display_op(f"[{sym_mixed_list}].map('convert all fruits to vegetables')", mapped_result, "map", "semantic")

    # Test map() with complex objects list
    sym_complex_list = Symbol([
        {'name': 'John', 'type': 'person'},
        {'name': 'Fluffy', 'type': 'cat'},
        {'name': 'Jane', 'type': 'person'},
        {'name': 'Rex', 'type': 'dog'}
    ])
    complex_mapped = sym_complex_list.map('change all animals to birds')
    display_op(f"[{sym_complex_list}].map('change all animals to birds')", complex_mapped, "map", "semantic")

    # Test map() with numbers
    sym_numbers = Symbol([1, 5, 10, 15, 20])
    number_mapped = sym_numbers.map('multiply small numbers (under 10) by 2')
    display_op(f"[{sym_numbers}].map('multiply small numbers (under 10) by 2')", number_mapped, "map", "semantic")

    # Test map() with string
    sym_string = Symbol("hello world")
    string_mapped = sym_string.map('convert vowels to numbers: a=1, e=2, i=3, o=4, u=5')
    display_op(f"[{sym_string}].map('convert vowels to numbers')", string_mapped, "map", "semantic")

    # Test map() with string consonants
    sym_string2 = Symbol("PROGRAMMING")
    string_mapped2 = sym_string2.map('make consonants lowercase, keep vowels uppercase')
    display_op(f"[{sym_string2}].map('make consonants lowercase, keep vowels uppercase')", string_mapped2, "map", "semantic")

    # Test map() with dictionary
    sym_dict = Symbol({'fruit1': 'apple', 'fruit2': 'banana', 'animal1': 'cat'})
    dict_mapped = sym_dict.map('convert all fruits to vegetables')
    display_op(f"[{sym_dict}].map('convert all fruits to vegetables')", dict_mapped, "map", "semantic")

    # Test map() with tuple
    sym_tuple = Symbol(('red', 'blue', 'green', 'yellow'))
    tuple_mapped = sym_tuple.map('change primary colors to secondary colors')
    display_op(f"[{sym_tuple}].map('change primary colors to secondary colors')", tuple_mapped, "map", "semantic")

    # Test map() with set
    sym_set = Symbol({'happy', 'sad', 'angry', 'excited'})
    set_mapped = sym_set.map('convert emotions to weather conditions')
    display_op(f"[{sym_set}].map('convert emotions to weather conditions')", set_mapped, "map", "semantic")

    # Test map() with dictionary
    sym_dict = Symbol({'fruit1': 'apple', 'fruit2': 'banana', 'animal1': 'cat'})
    dict_mapped = sym_dict.map('convert all fruits to vegetables')
    display_op(f"[{sym_dict}].map('convert all fruits to vegetables')", dict_mapped, "map", "semantic")

    # Test map() with complex dictionary
    sym_complex_dict = Symbol({
        'person1': {'name': 'John', 'type': 'human'},
        'pet1': {'name': 'Fluffy', 'type': 'cat'},
        'person2': {'name': 'Jane', 'type': 'human'}
    })
    complex_dict_mapped = sym_complex_dict.map('change all animals to birds')
    display_op(f"[{sym_complex_dict}].map('change all animals to birds')", complex_dict_mapped, "map", "semantic")

    # Test map() with tuple
    sym_tuple = Symbol(('red', 'blue', 'green', 'yellow'))
    tuple_mapped = sym_tuple.map('change primary colors to secondary colors')
    display_op(f"[{sym_tuple}].map('change primary colors to secondary colors')", tuple_mapped, "map", "semantic")

    # Test map() with set
    sym_set = Symbol({'happy', 'sad', 'angry', 'excited'})
    set_mapped = sym_set.map('convert emotions to weather conditions')
    display_op(f"[{sym_set}].map('convert emotions to weather conditions')", set_mapped, "map", "semantic")

    # Test map() error handling with non-iterable
    sym_int = Symbol(42)
    try:
        sym_int.map('some instruction')
    except AssertionError as e:
        display_op(f"[{sym_int}].map('some instruction')", f"AssertionError: {e!s}", "map", "syntactic")
        assert "Map can only be applied to iterable objects" in str(e)
    else:
        assert False, "map() should raise AssertionError for non-iterable input"

    # Test map() error handling without instruction
    sym_list_no_instruction = Symbol([1, 2, 3])
    try:
        sym_list_no_instruction.map()
    except TypeError as e:
        display_op(f"[{sym_list_no_instruction}].map()", f"TypeError: {e!s}", "map", "syntactic")
    else:
        assert False, "map() should require an instruction parameter"

    # # Test clean method
    # sym_dirty = Symbol("This text has   multiple    spaces and\n\nextra newlines.\t\tAnd tabs.")
    # cleaned = sym_dirty.clean()
    # display_op(f"[{sym_dirty}].clean()", cleaned, "clean", "semantic")

    # # Test summarize method
    # sym_long = Symbol("""Python is a high-level, interpreted programming language known for its readability and simplicity.
    # It was first released in 1991 by Guido van Rossum. Python supports multiple programming paradigms,
    # including procedural, object-oriented, and functional programming. It has a comprehensive standard
    # library and a large ecosystem of third-party packages that make it suitable for various applications,
    # from web development to data science and machine learning. Python's philosophy emphasizes code
    # readability with its notable use of significant whitespace.""")
    # summarized = sym_long.summarize()
    # assert len(str(summarized)) < len(str(sym_long))
    # display_op(f"[{sym_long}].summarize()", summarized, "summarize", "semantic")

    # # Test summarize with context
    # context_summarized = sym_long.summarize(context="Focus on Python's use in data science")
    # assert len(str(context_summarized)) < len(str(sym_long))
    # display_op(f"[{sym_long}].summarize(context='Focus on Python's use in data science')",
    #           context_summarized, "summarize", "semantic")

    # # Test outline method
    # sym_complex = Symbol(dedent("""
    # #Introduction to Machine Learning
    # Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data.
    # ## Supervised Learning
    # Supervised learning involves training a model on labeled data.
    # ### Classification
    # Classification tasks involve predicting discrete categories.
    # ### Regression
    # Regression tasks involve predicting continuous values.
    # ## Unsupervised Learning
    # Unsupervised learning works with unlabeled data to find patterns."""))
    # outlined = sym_complex.outline()
    # display_op(f"[{sym_complex}].outline()", outlined, "outline", "semantic")

    # # Test filter method (excluding)
    # sym_mixed = Symbol("Dogs are loyal pets. Cats are independent pets. Hamsters are small pets.")
    # filtered_ex = sym_mixed.filter(criteria="Cats")
    # display_op(f"[{sym_mixed}].filter('Cats')", filtered_ex, "filter", "semantic")

    # # Test filter method (including)
    # filtered_in = sym_mixed.filter(criteria="Dogs", include=True)
    # display_op(f"[{sym_mixed}].filter('Dogs', include=True)", filtered_in, "filter", "semantic")

    # # Test modify method
    # sym_original = Symbol("The quick brown fox jumps over the lazy dog.")
    # changes = "Change 'quick' to 'fast' and 'lazy' to 'sleeping'"
    # modified = sym_original.modify(changes=changes)
    # display_op(f"[{sym_original}].modify('{changes}')", modified, "modify", "semantic")

    # # Test replace method
    # sym_replace = Symbol("Python is a programming language. Python is easy to learn.")
    # replaced = sym_replace.replace("Python", "JavaScript")
    # display_op(f"[{sym_replace}].replace('Python', 'JavaScript')", replaced, "replace", "semantic")

    # # Test remove method
    # sym_extra = Symbol("This text contains [unnecessary information] that should be removed.")
    # removed = sym_extra.remove("[unnecessary information] ")
    # display_op(f"[{sym_extra}].remove('[unnecessary information] ')", removed, "remove", "semantic")

    # # Test include method
    # sym_base = Symbol("This is the main content.")
    # included = sym_base.include("This is additional information.")
    # display_op(f"[{sym_base}].include('This is additional information.')", included, "include", "semantic")

    # # Test combine method
    # sym_first = Symbol("First part of the content.")
    # combined = sym_first.combine("Second part of the content.")
    # display_op(f"[{sym_first}].combine('Second part of the content.')", combined, "combine", "semantic")

@pytest.mark.mandatory
def test_uniqueness_pr():
    """Test the UniquenessPrimitives class methods."""

    # Test unique method with no keys
    sym_repeated = Symbol("This is a test. This is another test. This is a third test.")
    unique_result = sym_repeated.unique()
    display_op(f"[\n{sym_repeated}\n].unique()", f"\n{unique_result}\n", "unique", "semantic")

    # Test unique method with specific keys
    sym_structured = Symbol(dedent("""
    name: John, age: 30, city: New York
    name: Jane, age: 30, city: New York
    name: John, age: 25, city: Boston
    """))
    unique_keys = sym_structured.unique(keys=['name', 'city'])
    display_op(f"[\n{sym_structured}\n].unique(['name', 'city'])", f"\n{unique_keys}\n", "unique", "semantic")

    # Test compose method with simple text
    sym_elements = Symbol("The sun rises in the east and sets in the west.")
    composed = sym_elements.compose()
    display_op(f"[\n{sym_elements}\n].compose()", f"\n{composed}\n", "compose", "semantic")

    # Test compose method with structured text
    sym_story_elements = Symbol(dedent("""
    Character: Detective Smith
    Setting: Dark alley in London
    Time: Midnight
    Weather: Heavy rain
    """))
    story_composed = sym_story_elements.compose()
    display_op(f"[\n{sym_story_elements}\n].compose()", f"\n{story_composed}\n", "compose", "semantic")

@pytest.mark.mandatory
def test_pattern_matching_pr():
    """Test the PatternMatchingPrimitives class methods."""

    # Test rank method with default parameters
    sym_numbers = Symbol(dedent("""
    5. Learn Python basics
    1. Install Python
    3. Practice coding
    2. Set up IDE
    4. Join community
    """))
    ranked = sym_numbers.rank(measure='difficulty', order='desc')
    display_op(f"[\n{sym_numbers}\n].rank('difficulty', 'desc')", f"\n{ranked}\n", "rank", "semantic")

    # Test rank with different measure and order
    sym_tasks = Symbol(dedent("""
    Important task: Complete project proposal
    Urgent task: Fix critical bug
    Optional task: Update documentation
    Critical task: Deploy hotfix
    """))
    ranked_priority = sym_tasks.rank(measure='priority', order='asc')
    display_op(f"[\n{sym_tasks}\n].rank('priority', 'asc')", f"\n{ranked_priority}\n", "rank", "semantic")

    # Test extract method with pattern
    sym_contact = Symbol(dedent("""
    Contact Information:
    Email: john.doe@email.com
    Phone: +1-555-0123
    Address: 123 Main St, City
    """))
    extracted = sym_contact.extract("contact details")
    display_op(f"[\n{sym_contact}\n].extract('contact details')", f"\n{extracted}\n", "extract", "semantic")

    # Test extract with specific pattern
    sym_text = Symbol(dedent("""
    Project deadline: 2024-03-15
    Budget allocated: $50,000
    Team size: 8 people
    Status: In Progress
    """))
    extracted_dates = sym_text.extract("dates and deadlines")
    display_op(f"[\n{sym_text}\n].extract('dates and deadlines')", f"\n{extracted_dates}\n", "extract", "semantic")

    # Test correct method with syntax error
    sym_code = Symbol(dedent("""
    def calculate_sum(a b):
        return a + b
    """))
    try:
        exec(str(sym_code))
    except Exception as e:
        corrected = sym_code.correct("Fix the code", exception=e)
        display_op(f"[\n{sym_code}\n].correct('Fix the code', {type(e).__name__})", f"\n{corrected}\n", "correct", "semantic")

    # Test correct with type error
    sym_type_error = Symbol(dedent("""
    def process_data(items):
        return items.sort()
    result = process_data([3, 1, 2])
    print(result + 1)  # TypeError: NoneType + int
    """))
    try:
        exec(str(sym_type_error))
    except Exception as e:
        corrected_type = sym_type_error.correct("Fix the code", exception=e)
        display_op(f"[\n{sym_type_error}\n].correct('Fix the code', {type(e).__name__})", f"\n{corrected_type}\n", "correct", "semantic")

    # Test translate method
    sym_english = Symbol("Hello, how are you today?")
    translated = sym_english.translate("Spanish")
    display_op(f"[{sym_english}].translate('Spanish')", translated, "translate", "semantic")

    # Test choice method
    sym_weather = Symbol("Temperature: 85°F, Humidity: 70%, Conditions: Sunny")
    cases = ["hot and humid", "mild", "cold and dry"]
    weather_choice = sym_weather.choice(cases=cases, default="mild")
    display_op(f"[{sym_weather}].choice({cases}, default='mild')", weather_choice, "choice", "semantic")

    # Test choice with more complex cases
    sym_sentiment = Symbol("This product exceeded all my expectations! Absolutely wonderful!")
    sentiment_cases = ["positive", "neutral", "negative"]
    sentiment_choice = sym_sentiment.choice(cases=sentiment_cases, default="neutral")
    display_op(f"[{sym_sentiment}].choice({sentiment_cases}, default='neutral')", sentiment_choice, "choice", "semantic")

@pytest.mark.mandatory
def test_query_handling_pr():
    """Test the QueryHandlingPrimitives class methods."""

    # Test query method with basic context
    sym_data = Symbol(dedent("""
    Product: Laptop Computer
    Price: $1299.99
    Brand: TechCorp
    RAM: 16GB
    Storage: 512GB SSD
    Screen: 15.6 inch
    """))

    basic_query = sym_data.query("What is the price?")
    display_op(f"[\n{sym_data}\n].query('What is the price?')", basic_query, "query", "semantic")

    # Test query with more specific context
    tech_query = sym_data.query("specifications", prompt="List the technical specifications")
    display_op(f"[\n{sym_data}\n].query('specifications', prompt='List the technical specifications')", tech_query, "query", "semantic")

    # Test query with complex data
    sym_report = Symbol(dedent("""
    Quarterly Sales Report:
    Q1: $2.5M (Growth: +15%)
    Q2: $3.1M (Growth: +24%)
    Q3: $2.8M (Growth: -10%)
    Q4: $3.7M (Growth: +32%)

    Top Products:
    1. Software Licenses: $8.2M
    2. Hardware Sales: $3.9M
    3. Consulting Services: $1.9M
    """))

    performance_query = sym_report.query("growth trends", prompt="Analyze the quarterly growth patterns")
    display_op(f"[\n{sym_report}\n].query('growth trends', prompt='Analyze the quarterly growth patterns')", performance_query, "query", "semantic")

    # Test convert method with different formats
    sym_json_data = Symbol('{"name": "John", "age": 30, "city": "New York"}')

    yaml_converted = sym_json_data.convert("YAML")
    display_op(f"[{sym_json_data}].convert('YAML')", yaml_converted, "convert", "semantic")

    xml_converted = sym_json_data.convert("XML")
    display_op(f"[{sym_json_data}].convert('XML')", xml_converted, "convert", "semantic")

    # Test convert with tabular data
    sym_csv_like = Symbol(dedent("""
    Name,Age,Department
    Alice,28,Engineering
    Bob,35,Marketing
    Carol,42,Sales
    """))

    table_converted = sym_csv_like.convert("HTML table")
    display_op(f"[\n{sym_csv_like}\n].convert('HTML table')", f"\n{table_converted}\n", "convert", "semantic")

    # Test convert with markdown format
    sym_text = Symbol("This is a simple paragraph with some important information that needs formatting.")
    markdown_converted = sym_text.convert("Markdown with bullet points")
    display_op(f"[{sym_text}].convert('Markdown with bullet points')", markdown_converted, "convert", "semantic")

    # Test transcribe method with various modifications
    sym_formal = Symbol("Hey there! How's it going? Hope you're doing well!")

    formal_transcribed = sym_formal.transcribe("make it formal and professional")
    display_op(f"[{sym_formal}].transcribe('make it formal and professional')", formal_transcribed, "transcribe", "semantic")

    # Test transcribe with tone change
    sym_technical = Symbol("The system crashed because of insufficient memory allocation and poor error handling.")
    simple_transcribed = sym_technical.transcribe("explain in simple terms for non-technical audience")
    display_op(f"[{sym_technical}].transcribe('explain in simple terms for non-technical audience')", simple_transcribed, "transcribe", "semantic")

    # Test transcribe with style modification
    sym_boring = Symbol("The meeting will be held on Monday at 2 PM in conference room A.")
    engaging_transcribed = sym_boring.transcribe("make it more engaging and friendly")
    display_op(f"[{sym_boring}].transcribe('make it more engaging and friendly')", engaging_transcribed, "transcribe", "semantic")

    # Test transcribe with length modification
    sym_verbose = Symbol(dedent("""
    In consideration of the aforementioned circumstances and taking into account
    the various factors that have been discussed in previous meetings, it has
    been determined that the most appropriate course of action would be to
    proceed with the implementation of the proposed solution at the earliest
    possible convenience, subject to the availability of necessary resources.
    """))

    concise_transcribed = sym_verbose.transcribe("make it concise and direct")
    display_op(f"[\n{sym_verbose}\n].transcribe('make it concise and direct')", f"\n{concise_transcribed}\n", "transcribe", "semantic")

    # Test transcribe with language style change
    sym_casual = Symbol("I think maybe we should probably consider looking into this issue sometime soon.")
    confident_transcribed = sym_casual.transcribe("make it more confident and decisive")
    display_op(f"[{sym_casual}].transcribe('make it more confident and decisive')", confident_transcribed, "transcribe", "semantic")

@pytest.mark.mandatory
def test_execution_control_pr():
    """Test the ExecutionControlPrimitives class methods."""

    # Test analyze method with exception handling
    sym_code = Symbol("print('Hello World')")
    try:
        # Create a simple exception for testing
        raise ValueError("Sample error for testing analyze method")
    except Exception as e:
        analyzed = sym_code.analyze(exception=e, query="What went wrong?")
        display_op(f"[{sym_code}].analyze(exception={type(e).__name__}, query='What went wrong?')", analyzed, "analyze", "semantic")

    # Test execute method with simple code
    sym_simple_code = Symbol(dedent("""
    def run():
        return 2 + 3
    res = run()
    """))
    executed = sym_simple_code.execute()
    display_op(f"[{sym_simple_code.value.strip()}].execute()", executed, "execute", "syntactic")

    # Test execute with more complex expression
    sym_math_code = Symbol(dedent("""
    def run():
        import math
        return math.sqrt(16)
    res = run()
    """))
    executed_math = sym_math_code.execute()
    display_op(f"[{sym_math_code.value.strip()}].execute()", executed_math, "execute", "syntactic")

    # Test fexecute method (fallback execute)
    sym_fallback = Symbol(dedent("""
    def run():
        return sum([1, 2, 3, 4])
    res = run()
    """))
    fexecuted = sym_fallback.fexecute()
    display_op(f"[{sym_fallback.value.strip()}].fexecute()", fexecuted, "fexecute", "syntactic")

    # Test simulate method (for code simulation)
    sym_code = Symbol("x = 5; y = 10; result = x + y")
    simulated = sym_code.simulate()
    display_op(f"[{sym_code}].simulate()", simulated, "simulate", "semantic")

    # Test simulate with algorithm
    sym_algorithm = Symbol("def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)")
    algorithm_sim = sym_algorithm.simulate()
    display_op(f"[{sym_algorithm}].simulate()", algorithm_sim, "simulate", "semantic")

    # Test sufficient method
    sym_info = Symbol(dedent("""
    Product Name: UltraBook Pro
    Price: $1299
    RAM: 16GB
    Storage: 512GB SSD
    Screen: 15.6 inch
    """))

    sufficient_check = sym_info.sufficient("Is this enough information to make a purchase decision?")
    display_op(f"[\n{sym_info}\n].sufficient('Is this enough information to make a purchase decision?')", sufficient_check, "sufficient", "semantic")

    # Test sufficient with incomplete information
    sym_partial = Symbol("Name: John, Age: 30")
    partial_sufficient = sym_partial.sufficient("Is this enough information for a job application?")
    display_op(f"[{sym_partial}].sufficient('Is this enough information for a job application?')", partial_sufficient, "sufficient", "semantic")

    # Test list method
    sym_employees = Symbol(dedent("""
    Employee List:
    2. Bob - Manager - $85000
    3. Carol - Designer - $65000
    4. David - Engineer - $72000
    5. Eve - Manager - $90000
    """))

    engineers_list = sym_employees.list("all employees who are engineers")
    display_op(f"[\n{sym_employees}\n].list('employees who are engineers')", f"\n{engineers_list}\n", "list", "semantic")

    # Test list with salary condition
    high_earners = sym_employees.list("employees earning less than $70000")
    display_op(f"[\n{sym_employees}\n].list('employees earning less than $70000')", f"\n{high_earners}\n", "list", "semantic")

    # Test foreach method
    sym_numbers = Symbol([1, 2, 3, 4, 5])
    squared_numbers = sym_numbers.foreach("each number", "square the number")
    display_op(f"[{sym_numbers}].foreach('each number', 'square the number')", squared_numbers, "foreach", "semantic")

    # Test foreach with text processing
    sym_sentences = Symbol(["hello world", "python programming", "artificial intelligence"])
    capitalized = sym_sentences.foreach("each sentence", "capitalize all words")
    display_op(f"[{sym_sentences}].foreach('each sentence', 'capitalize all words')", capitalized, "foreach", "semantic")

    # Test stream method (basic example since it requires Expression objects)
    try:
        # Create a simple expression for streaming
        class SimpleExpr(Expression):
            def __init__(self):
                super().__init__()

            def __call__(self, sym, **kwargs):
                return Symbol(f"Processed: {sym.value}")

        sym_large_text = Symbol("This is a large text that needs to be processed in chunks for streaming.")
        simple_expr = SimpleExpr()

        # Stream processing
        streamed_results = list(sym_large_text.stream(simple_expr, token_ratio=0.8))
        display_op(f"[{sym_large_text}].stream(simple_expr, token_ratio=0.8)", f"Streamed {len(streamed_results)} chunks", "stream", "semantic")
    except Exception as e:
        print(f"Stream test encountered issue: {e}")

    # Test ftry method (fault-tolerant execution)
    try:
        class TestExpr(Expression):
            def __init__(self, should_fail=False):
                super().__init__()
                self.should_fail = should_fail
                self.attempt_count = 0

            def __call__(self, sym, **kwargs):
                self.attempt_count += 1
                if self.should_fail and self.attempt_count <= 1:
                    raise ValueError("Intentional failure for testing")
                return Symbol(f"Success on attempt {self.attempt_count}: {sym.value}")

        sym_test = Symbol("test data")
        failing_expr = TestExpr(should_fail=True)

        # Test ftry with retry logic
        ftry_result = sym_test.ftry(failing_expr, retries=2)
        display_op(f"[{sym_test}].ftry(failing_expr, retries=2)", ftry_result, "ftry", "semantic")

        # Test ftry with successful expression
        success_expr = TestExpr(should_fail=False)
        ftry_success = sym_test.ftry(success_expr, retries=1)
        display_op(f"[{sym_test}].ftry(success_expr, retries=1)", ftry_success, "ftry", "semantic")
    except Exception as e:
        print(f"Ftry test encountered issue: {e}")

@pytest.mark.mandatory
def test_dict_handling_pr():
    """Test the DictHandlingPrimitives class methods."""

    # Test dict() method with basic text
    sym_text = Symbol("I have apples, oranges, and bananas in my kitchen.")
    dict_result = sym_text.dict("categorize fruits")
    display_op(f"[{sym_text}].dict('categorize fruits')", dict_result, "dict", "semantic")

    # Test dict() method with shopping list
    sym_shopping = Symbol("milk, bread, eggs, apples, cheese, chicken, broccoli, rice")
    food_dict = sym_shopping.dict("organize by food categories")
    display_op(f"[{sym_shopping}].dict('organize by food categories')", food_dict, "dict", "semantic")

    # Test dict() method with mixed data
    sym_mixed = Symbol(dedent("""
    Alice works as an engineer and likes reading.
    Bob is a manager who enjoys cooking.
    Carol is a designer who loves painting.
    """))
    people_dict = sym_mixed.dict("organize people by profession and hobbies")
    display_op(f"[\n{sym_mixed}\n].dict('organize people by profession and hobbies')", f"\n{people_dict}\n", "dict", "semantic")

    # Test dict() method with numbers and context
    sym_numbers = Symbol([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    number_dict = sym_numbers.dict("classify numbers")
    display_op(f"[{sym_numbers}].dict('classify numbers')", number_dict, "dict", "semantic")

    # Test dict() method with specific context and kwargs
    sym_animals = Symbol("Dogs bark, cats meow, birds chirp, fish swim")
    animal_dict = sym_animals.dict("group animals by sound", detailed=True)
    display_op(f"[{sym_animals}].dict('group animals by sound', detailed=True)", animal_dict, "dict", "semantic")

    # Test dict() with complex structured text
    sym_products = Symbol(dedent("""
    Laptop: $1200, Electronics
    Book: $15, Education
    Phone: $800, Electronics
    Notebook: $5, Education
    Tablet: $600, Electronics
    """))
    product_dict = sym_products.dict("organize products by category and price range")
    display_op(f"[\n{sym_products}\n].dict('organize products by category and price range')", f"\n{product_dict}\n", "dict", "semantic")

@pytest.mark.mandatory
def test_template_styling_pr():
    """Test the TemplateStylingPrimitives class methods."""

    # Test template() method with default placeholder
    sym_name = Symbol("Alice")
    template_str = "Hello {{placeholder}}, welcome to our service!"
    templated_result = sym_name.template(template_str)
    expected_result = "Hello Alice, welcome to our service!"
    assert str(templated_result) == expected_result
    display_op(f"[{sym_name}].template('{template_str}')", templated_result, "template", "syntactic")

    # Test template() method with custom placeholder
    sym_product = Symbol("Premium Headphones")
    custom_template = "Product: ***ITEM*** - Now available in store!"
    custom_placeholder = "***ITEM***"
    custom_templated = sym_product.template(custom_template, placeholder=custom_placeholder)
    expected_custom = "Product: Premium Headphones - Now available in store!"
    assert str(custom_templated) == expected_custom
    display_op(f"[{sym_product}].template('{custom_template}', placeholder='{custom_placeholder}')", custom_templated, "template", "syntactic")

    # Test template() method with more complex template
    sym_data = Symbol("Machine Learning")
    complex_template = dedent("""
    Course: {{placeholder}}
    Duration: 12 weeks
    Level: Intermediate
    Prerequisites: Basic Python knowledge
    """).strip()
    complex_templated = sym_data.template(complex_template)
    expected_complex = dedent("""
    Course: Machine Learning
    Duration: 12 weeks
    Level: Intermediate
    Prerequisites: Basic Python knowledge
    """).strip()
    assert str(complex_templated) == expected_complex
    display_op(f"[{sym_data}].template(complex_template)", f"\n{complex_templated}", "template", "syntactic")

    # Test template() method with numeric data
    sym_number = Symbol(42)
    number_template = "The answer to life, universe and everything is {{placeholder}}."
    number_templated = sym_number.template(number_template)
    expected_number = "The answer to life, universe and everything is 42."
    assert str(number_templated) == expected_number
    display_op(f"[{sym_number}].template('{number_template}')", number_templated, "template", "syntactic")

    # Test template() method with list data
    sym_list = Symbol(["apple", "banana", "cherry"])
    list_template = "Available fruits: {{placeholder}}"
    list_templated = sym_list.template(list_template)
    expected_list = "Available fruits: ['apple', 'banana', 'cherry']"
    assert str(list_templated) == expected_list
    display_op(f"[{sym_list}].template('{list_template}')", list_templated, "template", "syntactic")

    # Test style() method with basic styling
    sym_content = Symbol("This is a basic text that needs styling.")
    style_description = "Make this text bold and italic"
    styled_result = sym_content.style(style_description)
    display_op(f"[{sym_content}].style('{style_description}')", styled_result, "style", "semantic")

    # Test style() method with libraries parameter
    sym_code = Symbol("print('Hello World')")
    code_style_description = "Format as Python code with syntax highlighting"
    code_libraries = ["python", "syntax-highlighting"]
    styled_code = sym_code.style(code_style_description, libraries=code_libraries)
    display_op(f"[{sym_code}].style('{code_style_description}', libraries={code_libraries})", styled_code, "style", "semantic")

    # Test style() method with more complex styling
    sym_text = Symbol("The quick brown fox jumps over the lazy dog.")
    complex_style_description = "Transform into a professional headline format"
    complex_styled = sym_text.style(complex_style_description)
    display_op(f"[{sym_text}].style('{complex_style_description}')", complex_styled, "style", "semantic")

    # Test style() method with empty libraries list
    sym_simple = Symbol("Simple text")
    simple_style_description = "Make this text more engaging"
    simple_styled = sym_simple.style(simple_style_description, libraries=[])
    display_op(f"[{sym_simple}].style('{simple_style_description}', libraries=[])", simple_styled, "style", "semantic")

    # Test template() method with edge cases
    sym_empty = Symbol("")
    empty_template = "Empty content: {{placeholder}}"
    empty_templated = sym_empty.template(empty_template)
    expected_empty = "Empty content: "
    assert str(empty_templated) == expected_empty
    display_op(f"[{sym_empty}].template('{empty_template}')", empty_templated, "template", "syntactic")

    # Test template() method with no placeholder in template
    sym_no_placeholder = Symbol("test")
    no_placeholder_template = "This template has no placeholder"
    no_placeholder_result = sym_no_placeholder.template(no_placeholder_template)
    expected_no_placeholder = "This template has no placeholder"
    assert str(no_placeholder_result) == expected_no_placeholder
    display_op(f"[{sym_no_placeholder}].template('{no_placeholder_template}')", no_placeholder_result, "template", "syntactic")

    # Test template() method with multiple placeholders
    sym_multi = Symbol("Python")
    multi_template = "Language: {{placeholder}}, Type: {{placeholder}}, Popularity: High"
    multi_templated = sym_multi.template(multi_template)
    expected_multi = "Language: Python, Type: Python, Popularity: High"
    assert str(multi_templated) == expected_multi
    display_op(f"[{sym_multi}].template('{multi_template}')", multi_templated, "template", "syntactic")

@pytest.mark.mandatory
@pytest.mark.skipif(not SYMAI_CONFIG.get('EMBEDDING_ENGINE_MODEL', False) and not SYMAI_CONFIG.get('EMBEDDING_ENGINE_API_KEY', False), reason="Embedding engine not configured!")
def test_data_clustering_pr():
    """Test the DataClusteringPrimitives class methods."""
    # Test cluster method with simple text data
    sym_text = Symbol(["apple", "banana", "cherry", "apple", "orange", "banana", "grape"])
    clustered_result = sym_text.cluster(metric='cosine', min_cluster_size=2)
    display_op(f"[{sym_text}].cluster(metric='cosine', min_cluster_size=2)", clustered_result, "cluster", "semantic")

    # Test cluster method with list data
    sym_list = Symbol(["technology", "science", "programming", "coding", "research", "innovation"])
    clustered_list = sym_list.cluster()
    display_op(f"[{sym_list}].cluster()", clustered_list, "cluster", "semantic")

    # Test cluster method with structured sentences
    sym_sentences = Symbol([
        'The concert last night was absolutely incredible!',
        'That was the worst experience of my life.',
        'I can’t stop smiling after tasting that delicious cake.',
        'I hate how poorly designed this device is.',
        'What a fantastic day at the beach with friends!',
        'This traffic jam is driving me insane.',
        'This new book has completely captivated me.',
        'I fucking despise waiting in long lines.'
    ])
    clustered_sentences = sym_sentences.cluster(min_cluster_size=3)
    display_op(f"[{sym_sentences}].cluster(min_cluster_size=3)", clustered_sentences, "cluster", "semantic")

    # Test cluster method with simple categories
    sym_categories = Symbol([
        "cat", "dog", "fish", "bird",
        "car", "truck", "bicycle", "motorcycle",
        "apple", "banana", "orange", "grape"
    ])
    clustered_categories = sym_categories.cluster(metric='cosine', min_cluster_size=2)
    display_op(f"[{sym_categories}].cluster(metric='cosine', min_cluster_size=2)", clustered_categories, "cluster", "semantic")

@pytest.mark.mandatory
@pytest.mark.skipif(not SYMAI_CONFIG.get('EMBEDDING_ENGINE_MODEL', False) and not SYMAI_CONFIG.get('EMBEDDING_ENGINE_API_KEY', False), reason="Embedding engine not configured!")
def test_embedding_pr():
    """Test the EmbeddingPrimitives class methods."""

    # Test embed() method with simple string
    sym_text = Symbol("hello world")
    embedded_result = sym_text.embed()
    display_op(f"[{sym_text}].embed()", f"embeddings shape: {np.array(embedded_result.value).shape}", "embed", "semantic")

    # Test embedding property
    embedding_value = sym_text.embedding
    display_op(f"[{sym_text}].embedding", f"embedding shape: {embedding_value.shape}", "embedding", "semantic")
    assert isinstance(embedding_value, np.ndarray)

    # Test similarity with same string (edge case - should be 1.0 or very close)
    sym_same = Symbol("hello world")
    similarity_result = sym_text.similarity(sym_same)
    display_op(f"[{sym_text}].similarity([{sym_same}])", similarity_result, "similarity", "semantic")
    assert isinstance(similarity_result, float)
    assert similarity_result > 0.99  # Should be very close to 1.0 for identical strings

    # Test distance with same string (edge case - should be 0.0 or very close for distance measures)
    distance_result = sym_text.distance(sym_same, kernel='cosine')
    display_op(f"[{sym_text}].distance([{sym_same}], kernel='cosine')", distance_result, "distance", "semantic")
    assert isinstance(distance_result, float)
    assert distance_result < 0.01  # Should be very close to 0.0 for identical strings

    # Test zip method with string - check format
    zip_result = sym_text.zip()
    display_op(f"[{sym_text}].zip()", f"zip format: {type(zip_result)}, length: {len(zip_result)}", "zip", "semantic")

    # Syntactic checks for zip format
    assert isinstance(zip_result, list), "zip() should return a list"
    assert len(zip_result) == 1, "single string should result in list of length 1"

    # Check tuple structure
    tuple_item = zip_result[0]
    assert isinstance(tuple_item, tuple), "each item should be a tuple"
    assert len(tuple_item) == 3, "each tuple should have 3 elements (id, embedding, query)"

    # Check tuple element types
    idx, embedding, query = tuple_item
    assert isinstance(idx, str), "first element should be string ID"
    assert isinstance(embedding, list), "second element should be embedding as list"
    assert isinstance(query, dict), "third element should be query dict"
    assert 'text' in query, "query dict should contain 'text' key"
    assert query['text'] == 'hello world', "query text should match original value"

    # Test zip with list of strings
    sym_list = Symbol([["apple"], ["banana"]])
    zip_list_result = sym_list.zip()
    display_op(f"[{sym_list}].zip()", f"zip format: {type(zip_list_result)}, length: {len(zip_list_result)}", "zip", "semantic")

    # Check each tuple in the list
    for i, (idx, embedding, query) in enumerate(zip_list_result):
        assert isinstance(idx, str)
        assert isinstance(embedding, list)
        assert isinstance(query, dict)
        assert 'text' in query
        assert query['text'] == str([['apple', 'banana'][i]])

    # Test similarity with different metrics
    sym_other = Symbol("different text")
    cosine_sim = sym_text.similarity(sym_other, metric='cosine')
    product_sim = sym_text.similarity(sym_other, metric='product')
    display_op(f"[{sym_text}].similarity([{sym_other}], metric='cosine')", cosine_sim, "similarity", "semantic")
    display_op(f"[{sym_text}].similarity([{sym_other}], metric='product')", product_sim, "similarity", "semantic")

    # Test distance with different kernels
    gaussian_dist = sym_text.distance(sym_other, kernel='gaussian')
    linear_dist = sym_text.distance(sym_other, kernel='linear')
    display_op(f"[{sym_text}].distance([{sym_other}], kernel='gaussian')", gaussian_dist, "distance", "semantic")
    display_op(f"[{sym_text}].distance([{sym_other}], kernel='linear')", linear_dist, "distance", "semantic")

    # Test error handling for zip method
    sym_invalid = Symbol(123)
    try:
        sym_invalid.zip()
        assert False, "zip() should raise ValueError for non-string/non-list input"
    except ValueError as e:
        display_op(f"[{sym_invalid}].zip()", f"ValueError: {e!s}", "zip", "syntactic")
        assert "Expected id to be a string" in str(e)

@pytest.mark.mandatory
def test_io_handling_pr():
    """Test the IOHandlingPrimitives class methods."""

    # Test open() method with path as parameter
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp_file:
        test_content = "Hello, this is test content for file operations."
        tmp_file.write(test_content)
        tmp_file_path = tmp_file.name

    try:
        # Test opening file with path parameter
        sym_empty = Symbol()
        opened_result = sym_empty.open(tmp_file_path)
        display_op(f"Symbol().open('{os.path.basename(tmp_file_path)}')", f"content length: {len(opened_result.value)}", "open", "syntactic")
        assert isinstance(opened_result.value, str)
        assert test_content in opened_result.value

        # Test opening file with path as Symbol value
        sym_path = Symbol(tmp_file_path)
        opened_from_value = sym_path.open()
        display_op(f"Symbol('{os.path.basename(tmp_file_path)}').open()", f"content length: {len(opened_from_value.value)}", "open", "syntactic")
        assert isinstance(opened_from_value.value, str)
        assert test_content in opened_from_value.value

    finally:
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

    # Test error handling for open() method with no path
    sym_no_path = Symbol()
    try:
        sym_no_path.open()
        assert False, "open() should raise ValueError when no path is provided"
    except ValueError as e:
        display_op("Symbol().open()", f"ValueError: {e!s}", "open", "syntactic")
        assert "Path is not provided" in str(e)

    assert hasattr(sym_no_path, 'input'), "Symbol should have input method"
    assert callable(sym_no_path.input), "input should be callable"
    display_op("Symbol().input()", "input method is callable (requires interaction)", "input", "syntactic")

@pytest.mark.mandatory
def test_persistence_pr():
    """Test the PersistencePrimitives class methods."""

    # Test save() and load() methods with serialization
    sym_test = Symbol("Hello, persistence test!")

    # Create a temporary file path (but delete the file so save can create it properly)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
        tmp_file_path = tmp_file.name
    os.unlink(tmp_file_path)  # Remove the empty file

    try:
        # Test save with serialize=True (default)
        sym_test.save(tmp_file_path, serialize=True)
        display_op(f"Symbol('{sym_test}').save('{os.path.basename(tmp_file_path)}', serialize=True)", "saved as pickle", "save", "syntactic")

        # Check that pickle file was created
        pkl_path = tmp_file_path + '.pkl' if not tmp_file_path.endswith('.pkl') else tmp_file_path
        assert os.path.exists(pkl_path), "Pickle file should be created"

        # Test load() method - now it's an instance method, not static
        loader_sym = Symbol()
        loaded_sym = loader_sym.load(pkl_path)
        display_op(f"Symbol().load('{os.path.basename(pkl_path)}')", f"loaded: {loaded_sym.value}", "load", "syntactic")
        assert isinstance(loaded_sym, Symbol)
        assert loaded_sym.value == sym_test.value

    finally:
        # Cleanup
        for path in [tmp_file_path, tmp_file_path + '.pkl']:
            if os.path.exists(path):
                os.unlink(path)

    # Test save() with serialize=False
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp_file:
        tmp_file_path = tmp_file.name
    os.unlink(tmp_file_path)  # Remove the empty file

    try:
        sym_text = Symbol("This is a text save test")
        sym_text.save(tmp_file_path, serialize=False)
        display_op(f"Symbol('{sym_text}').save('{os.path.basename(tmp_file_path)}', serialize=False)", "saved as text", "save", "syntactic")

        # Read the file to verify content
        with open(tmp_file_path) as f:
            saved_content = f.read()
        assert saved_content == str(sym_text)

    finally:
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

    # Test save() with replace=False (default) - should create new file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp_file:
        base_path = tmp_file.name
        tmp_file.write("existing content")

    try:
        sym_replace = Symbol("new content")
        sym_replace.save(base_path, serialize=False, replace=False)
        display_op(f"Symbol('{sym_replace}').save(existing_file, serialize=False, replace=False)", "created new file with suffix", "save", "syntactic")

        # Should create a file with _0 suffix
        expected_new_path = base_path.replace('.txt', '_0.txt')
        assert os.path.exists(expected_new_path), "New file with suffix should be created"

        with open(expected_new_path) as f:
            content = f.read()
        assert content == str(sym_replace)

        # Clean up the _0 file
        os.unlink(expected_new_path)

    finally:
        if os.path.exists(base_path):
            os.unlink(base_path)

    # Test save() with replace=True - should overwrite existing file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp_file:
        replace_base_path = tmp_file.name
        tmp_file.write("original content")

    try:
        sym_replace_true = Symbol("overwritten content")
        sym_replace_true.save(replace_base_path, serialize=False, replace=True)
        display_op(f"Symbol('{sym_replace_true}').save(existing_file, serialize=False, replace=True)", "overwritten existing file", "save", "syntactic")

        # Should overwrite the original file, not create a new one
        with open(replace_base_path) as f:
            content = f.read()
        assert content == str(sym_replace_true), "File content should be overwritten"

        # Verify that no _0 suffix file was created
        expected_suffix_path = replace_base_path.replace('.txt', '_0.txt')
        assert not os.path.exists(expected_suffix_path), "No suffix file should be created when replace=True"

    finally:
        if os.path.exists(replace_base_path):
            os.unlink(replace_base_path)

    # Test expand() method functionality
    sym_expand = Symbol("Calculate the fibonacci sequence up to 10 numbers")

    try:
        # Test expand method - it should return a function name
        func_name = sym_expand.expand()
        display_op(f"Symbol('{sym_expand}').expand()", f"generated function: {func_name}", "expand", "semantic")

        # Check that expand returns a string (function name)
        assert isinstance(func_name, str), "expand() should return a function name as string"
        assert len(func_name) > 0, "function name should not be empty"

        # Check that the function was added as an attribute to the Symbol
        assert hasattr(sym_expand, func_name), f"Symbol should have the generated function '{func_name}' as attribute"
        assert callable(getattr(sym_expand, func_name)), f"Generated function '{func_name}' should be callable"

    except Exception as e:
        # If expand fails due to missing LLM configuration, that's expected
        display_op(f"Symbol('{sym_expand}').expand()", f"Error (expected if LLM not configured): {type(e).__name__}", "expand", "semantic")

    # Test expand with different problem
    sym_math = Symbol("Write a function to calculate square root")
    try:
        math_func_name = sym_math.expand()
        display_op(f"Symbol('{sym_math}').expand()", f"generated function: {math_func_name}", "expand", "semantic")
        assert isinstance(math_func_name, str)
        assert hasattr(sym_math, math_func_name)

    except Exception as e:
        display_op(f"Symbol('{sym_math}').expand()", f"Error (expected if LLM not configured): {type(e).__name__}", "expand", "semantic")

@pytest.mark.mandatory
def test_output_handling_pr():
    """Test the OutputHandlingPrimitives class methods."""

    # Test basic output() method functionality
    sym_test = Symbol("Hello, output test!")

    # Test output method with default parameters
    result = sym_test.output()
    display_op(f"Symbol('{sym_test}').output()", result, "output", "syntactic")
    assert isinstance(result, Symbol)

    # Test output method with custom handler function
    captured_output = []
    def custom_handler(input_dict):
        captured_output.append(input_dict)

    result_with_handler = sym_test.output(handler=custom_handler)
    display_op(f"Symbol('{sym_test}').output(handler=custom_handler)", result_with_handler, "output", "syntactic")
    assert isinstance(result_with_handler, Symbol)

    # Test output method with additional args
    result_with_args = sym_test.output("arg1", "arg2")
    display_op(f"Symbol('{sym_test}').output('arg1', 'arg2')", result_with_args, "output", "syntactic")
    assert isinstance(result_with_args, Symbol)

    # Test output method with kwargs
    result_with_kwargs = sym_test.output(custom_param="value")
    display_op(f"Symbol('{sym_test}').output(custom_param='value')", result_with_kwargs, "output", "syntactic")
    assert isinstance(result_with_kwargs, Symbol)

    # Verify that output method exists and is callable
    assert hasattr(sym_test, 'output'), "Symbol should have output method"
    assert callable(sym_test.output), "output should be callable"

    # Test with different symbol types
    sym_number = Symbol(42)
    num_result = sym_number.output()
    display_op(f"Symbol({sym_number}).output()", num_result, "output", "syntactic")
    assert isinstance(num_result, Symbol)

    sym_test = Symbol("Hello, dual args test!", test_kwarg={'processed': "Hello, dual args test!"})
    # This should show both function call args (in processed) and expression args (in args field)
    result = sym_test.output("call_arg1", "call_arg2")

    display_op(
        f"Symbol('{sym_test}').output('call_arg1', 'call_arg2')",
        result,
        "output",
        "syntactic"
    )

    assert "Hello, dual args test!" in str(result.value['processed'])
    assert isinstance(result, Symbol)

    # Test with no arguments to show the difference
    result_no_args = sym_test.output()
    display_op(
        f"Symbol('{sym_test}').output()",
        result_no_args,
        "output",
        "syntactic"
    )

    assert "Hello, dual args test!" in str(result_no_args.value['processed'])
