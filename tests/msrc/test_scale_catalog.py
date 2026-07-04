from runtime.control.msrc import ScaleCatalog


def test_scale_catalog_default_declares_required_scales():
    catalog = ScaleCatalog.default()
    assert catalog.list_ids() == ["1x1", "2x2", "3x3", "5x5", "10x10", "30x30"]


def test_scale_catalog_executable_subset_is_1x1_and_5x5():
    catalog = ScaleCatalog.default()
    executable = [item.scale_id for item in catalog.executable_scales()]
    assert executable == ["1x1", "5x5"]


def test_scale_catalog_nearest_executable_for_10x10_is_5x5():
    catalog = ScaleCatalog.default()
    nearest = catalog.nearest_executable("10x10")
    assert nearest.scale_id == "5x5"


def test_scenario_params_for_1x1_removes_topology():
    catalog = ScaleCatalog.default()
    params = catalog.scenario_params_for("1x1", topology="hotspot_center", base_params={"initial_temperature": 0.8})
    assert params["grid_size"] == 1
    assert "topology" not in params


def test_scenario_params_for_5x5_keeps_topology():
    catalog = ScaleCatalog.default()
    params = catalog.scenario_params_for("5x5", topology="checkerboard", base_params={"initial_temperature": 0.8})
    assert params["grid_size"] == 5
    assert params["topology"] == "checkerboard"
