"""Unit tests for configuration loading and hashing."""

from driving_wm.utils.config import compute_config_hash, load_yaml


def test_load_yaml_configs() -> None:
    smoke_cfg = load_yaml("configs/smoke.yaml")
    assert "seed" in smoke_cfg
    assert "data" in smoke_cfg
    assert "model" in smoke_cfg
    assert "mpc" in smoke_cfg

    maps_cfg = load_yaml("configs/maps.yaml")
    assert "train" in maps_cfg
    assert "val" in maps_cfg
    assert "test_unseen_seeds" in maps_cfg
    assert "test_unseen_layouts" in maps_cfg
    assert "test_traffic" in maps_cfg


def test_config_hash_deterministic() -> None:
    cfg1 = {"lr": 0.001, "batch_size": 128, "hidden": [256, 256]}
    cfg2 = {"batch_size": 128, "hidden": [256, 256], "lr": 0.001}
    # Key order should not change hash
    h1 = compute_config_hash(cfg1)
    h2 = compute_config_hash(cfg2)
    assert h1 == h2
    assert len(h1) == 12
