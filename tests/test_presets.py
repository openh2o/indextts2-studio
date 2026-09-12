"""Unit tests for preset rename / duplicate / overwrite protection.

Run: uv run --extra test pytest tests/test_presets.py -v
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from indextts.utils.presets import (
    delete_preset,
    duplicate_preset,
    get_presets_dir,
    list_presets,
    load_preset,
    preset_exists,
    rename_preset,
    safe_preset_name,
    save_preset,
)


@pytest.fixture()
def presets_dir(tmp_path, monkeypatch):
    """Point the presets root at a tmp dir (it is normally project-relative)."""
    monkeypatch.setattr("indextts.utils.presets.get_presets_dir", lambda: tmp_path)
    monkeypatch.setattr("indextts.utils.presets._preset_dir", lambda n: tmp_path / safe_preset_name(n))
    return tmp_path


def _make(presets_dir, name, temp=0.8):
    save_preset(name, {"emo_control_method": 0, "advanced_params": {"temperature": temp}})
    return name


def test_rename_moves_directory(presets_dir):
    _make(presets_dir, "alpha")
    final = rename_preset("alpha", "beta")
    assert final == "beta"
    assert list_presets() == ["beta"]
    assert (presets_dir / "beta" / "preset.json").is_file()
    # rename keeps contents
    assert load_preset("beta")["advanced_params"]["temperature"] == 0.8


def test_rename_missing_source_raises(presets_dir):
    with pytest.raises(ValueError):
        rename_preset("nope", "other")


def test_rename_onto_existing_raises(presets_dir):
    _make(presets_dir, "alpha")
    _make(presets_dir, "beta")
    with pytest.raises(ValueError):
        rename_preset("alpha", "beta")


def test_rename_to_same_normalized_name_is_noop(presets_dir):
    _make(presets_dir, "alpha")
    # "alpha" -> "ALPHA" sanitizes differently; "alpha" -> " alpha " sanitizes the same
    assert rename_preset("alpha", " alpha ") == "alpha"


def test_duplicate_copies_everything(presets_dir):
    _make(presets_dir, "alpha")
    final = duplicate_preset("alpha", "copy1")
    assert final == "copy1"
    assert sorted(list_presets()) == ["alpha", "copy1"]
    assert load_preset("copy1")["advanced_params"]["temperature"] == 0.8
    # source is untouched
    assert preset_exists("alpha")


def test_duplicate_onto_existing_raises(presets_dir):
    _make(presets_dir, "alpha")
    _make(presets_dir, "beta")
    with pytest.raises(ValueError):
        duplicate_preset("alpha", "beta")


def test_duplicate_missing_source_raises(presets_dir):
    with pytest.raises(ValueError):
        duplicate_preset("nope", "other")


def test_delete_then_exists(presets_dir):
    _make(presets_dir, "alpha")
    assert preset_exists("alpha")
    assert delete_preset("alpha")
    assert not preset_exists("alpha")
    assert not delete_preset("alpha")  # second delete: already gone
