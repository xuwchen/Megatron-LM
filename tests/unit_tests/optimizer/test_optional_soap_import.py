# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Optional SOAP must not gate Muon; installed Muon dependencies are required."""

import subprocess
import sys
from pathlib import Path

import pytest

_IMPORT_PROBE = """
import builtins
import sys

missing_module = sys.argv[1]
original_import = builtins.__import__


def import_without_soap(name, *args, **kwargs):
    if name == "emerging_optimizers.soap":
        raise ModuleNotFoundError(
            f"No module named '{missing_module}'", name=missing_module
        )
    return original_import(name, *args, **kwargs)


builtins.__import__ = import_without_soap
try:
    from megatron.core.optimizer import emerging_optimizers as module
except ModuleNotFoundError as error:
    assert missing_module != "emerging_optimizers.soap", repr(error)
    assert error.name == missing_module, repr(error)
else:
    assert missing_module == "emerging_optimizers.soap", "SOAP dependency error was swallowed"
    assert module.HAVE_EMERGING_OPTIMIZERS, "Missing SOAP disabled installed Muon"
    assert module._EMERGING_OPTIMIZERS["muon"].optimizer_cls is module.TensorParallelMuon
    assert "muon" in module.registry.get_optimizer_name_list()
finally:
    builtins.__import__ = original_import
"""


@pytest.mark.parametrize(
    "missing_module",
    ["emerging_optimizers.soap", "soap_missing_transitive_dependency"],
    ids=["soap-absent", "soap-dependency-broken"],
)
def test_optional_soap_import(missing_module):
    # A fresh interpreter avoids cached imports and global optimizer-registry mutation.
    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_PROBE, missing_module],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
