import pytest

from runtime.lotf import LOTFMin
from runtime.lotf.lotf_min import ImplicationNode, SymbolNode


def test_lotf_parser_builds_implication_ast():
    lotf = LOTFMin()
    ast = lotf.parse("TEMP_HIGH -> ACTIVATE_COOLING")
    assert isinstance(ast, ImplicationNode)
    assert isinstance(ast.antecedent, SymbolNode)
    assert ast.antecedent.name == "TEMP_HIGH"


def test_lotf_checker_accepts_boolean_formula():
    lotf = LOTFMin()
    ast = lotf.parse("NOT TEMP_HIGH OR ACTIVATE_COOLING")
    lotf.check(ast, {"TEMP_HIGH": "bool", "ACTIVATE_COOLING": "bool"})


def test_lotf_checker_rejects_untyped_symbol():
    lotf = LOTFMin()
    ast = lotf.parse("TEMP_HIGH -> ACTIVATE_COOLING")
    with pytest.raises(TypeError):
        lotf.check(ast, {"TEMP_HIGH": "bool"})
