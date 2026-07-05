import inspect

import diffmjstep
from diffmjstep import mj_linearize, mj_rollout, mj_step
from diffmjstep.core import _auto_nthread


def test_package_exports_only_supported_api():
    assert diffmjstep.__all__ == [
        "mj_linearize",
        "mj_rollout",
        "mj_step",
    ]
    assert all(callable(fn) for fn in (mj_linearize, mj_rollout, mj_step))
    assert _auto_nthread(1, None) == 1


def _signature_contract(function):
    return [
        (parameter.name, parameter.kind, parameter.default)
        for parameter in inspect.signature(function).parameters.values()
    ]


POSITIONAL = inspect.Parameter.POSITIONAL_OR_KEYWORD
KEYWORD_ONLY = inspect.Parameter.KEYWORD_ONLY
REQUIRED = inspect.Parameter.empty


def test_mj_rollout_signature_is_lean_and_exact():
    assert _signature_contract(mj_rollout) == [
        ("model", POSITIONAL, REQUIRED),
        ("x0", POSITIONAL, REQUIRED),
        ("U", POSITIONAL, REQUIRED),
        ("backend", KEYWORD_ONLY, "auto"),
        ("return_all", KEYWORD_ONLY, True),
        ("return_sensors", KEYWORD_ONLY, False),
        ("eps", KEYWORD_ONLY, 1e-8),
        ("centered", KEYWORD_ONLY, True),
        ("nthread", KEYWORD_ONLY, None),
    ]


def test_mj_step_signature_is_lean_and_exact():
    assert _signature_contract(mj_step) == [
        ("model", POSITIONAL, REQUIRED),
        ("x0", POSITIONAL, REQUIRED),
        ("u", POSITIONAL, REQUIRED),
        ("nstep", KEYWORD_ONLY, 1),
        ("backend", KEYWORD_ONLY, "auto"),
        ("return_sensors", KEYWORD_ONLY, False),
        ("eps", KEYWORD_ONLY, 1e-8),
        ("centered", KEYWORD_ONLY, True),
        ("nthread", KEYWORD_ONLY, None),
    ]


def test_mj_linearize_signature_is_lean_and_exact():
    assert _signature_contract(mj_linearize) == [
        ("model", POSITIONAL, REQUIRED),
        ("x", POSITIONAL, REQUIRED),
        ("u", POSITIONAL, REQUIRED),
        ("sensors", KEYWORD_ONLY, False),
        ("eps", KEYWORD_ONLY, 1e-8),
        ("centered", KEYWORD_ONLY, True),
    ]
