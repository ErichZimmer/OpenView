"""Three-layer default agreement for the masked-predictor-fill and
instantaneous predictor-BC config keys.

CLAUDE.md gotcha #1: the CLI template (``create_default_config``), the
``Config`` property, and the frontend fallback must all ship the same default.
These tests pin ``masking.predictor_fill`` (default ``harmonic``) and
``instantaneous_piv.predictor_boundary_conditions`` (default ``[]``) across all
three layers.
"""

from pathlib import Path

import pytest
import yaml

from pivtools_cli.cli import create_default_config
from pivtools_core.config import Config

# gui/ sits beside PyPIVTools/ in the fullstack checkout.
_MASKING_HOOK = (
    Path(__file__).resolve().parents[2]
    / "gui"
    / "src"
    / "hooks"
    / "useMaskingConfig.ts"
)


@pytest.fixture
def template_path(tmp_path: Path) -> Path:
    """Write the CLI default config template and return its path."""
    path = tmp_path / "config.yaml"
    create_default_config(str(path))
    return path


@pytest.fixture
def empty_config(tmp_path: Path) -> Config:
    """A Config built from an empty YAML file (every key falls to its default)."""
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    return Config(str(path))


def test_template_predictor_fill_default(template_path: Path) -> None:
    raw = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    assert raw["masking"]["predictor_fill"] == "harmonic"
    assert Config(str(template_path)).masking_predictor_fill == "harmonic"


def test_template_instantaneous_bc_default(template_path: Path) -> None:
    raw = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    assert raw["instantaneous_piv"]["predictor_boundary_conditions"] == []
    assert Config(str(template_path)).instantaneous_predictor_boundary_conditions == []


def test_property_defaults_on_empty_config(empty_config: Config) -> None:
    assert empty_config.masking_predictor_fill == "harmonic"
    assert empty_config.instantaneous_predictor_boundary_conditions == []


def test_invalid_predictor_fill_raises(empty_config: Config) -> None:
    empty_config.data["masking"] = {"predictor_fill": "biharmonic"}
    with pytest.raises(ValueError, match="predictor_fill"):
        _ = empty_config.masking_predictor_fill


def test_frontend_predictor_fill_fallback() -> None:
    if not _MASKING_HOOK.is_file():
        pytest.skip(f"frontend source not present at {_MASKING_HOOK}")
    source = _MASKING_HOOK.read_text(encoding="utf-8")
    assert "initialConfig?.predictor_fill ?? 'harmonic'" in source
