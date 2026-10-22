from aierosion.layers.rules import LayerRules, parse_import_linter

FILES = ["app/__init__.py", "app/api/__init__.py", "app/api/routes.py", "app/services/__init__.py",
         "app/services/orders.py", "app/models/__init__.py", "app/models/order.py", "app/utils.py"]


def test_heuristic_layers_and_violations():
    r = LayerRules(FILES, {})
    assert r.source == "directory-heuristic" and r.eligible
    assert r.layer("app/api/routes.py")[0] == "presentation"
    assert r.layer("app/models/order.py")[0] == "data"
    assert r.layer("app/utils.py") is None
    assert r.is_violation("app/models/order.py", "app/api/routes.py")  # data -> presentation
    assert r.is_violation("app/services/orders.py", "app/api/routes.py")  # domain -> presentation
    assert not r.is_violation("app/api/routes.py", "app/models/order.py")  # downward is allowed
    assert not r.is_violation("app/models/order.py", "app/services/orders.py")  # data -> domain not flagged
    assert not r.is_violation("app/utils.py", "app/api/routes.py")  # unassigned module


def test_top_level_package_and_file_names_are_not_layers():
    # WorkflowAI case: project folder and top-level package both named "api".
    files = ["api/api/__init__.py", "api/api/services/__init__.py", "api/api/services/runs.py",
             "api/core/__init__.py", "api/core/utils/__init__.py", "api/core/utils/fields.py"]
    r = LayerRules(files, {})
    assert r.layer("api/core/utils/fields.py") is None
    assert r.layer("api/api/services/runs.py")[0] == "domain"
    assert not r.is_violation("api/api/services/runs.py", "api/core/utils/fields.py")
    # browser-use case: a file named views.py is not the presentation layer.
    r2 = LayerRules(["browser_use/__init__.py", "browser_use/dom/__init__.py", "browser_use/dom/views.py",
                     "browser_use/dom/service.py"], {})
    assert r2.layer("browser_use/dom/views.py") is None


def test_single_layer_repo_not_eligible():
    assert not LayerRules(["pkg/__init__.py", "pkg/models/a.py", "pkg/x.py"], {}).eligible


def test_import_linter_ini_and_toml():
    ini = """
[importlinter]
root_package = shop

[importlinter:contract:layers]
name = Layers
type = layers
layers =
    shop.web
    shop.domain | shop.events
    shop.store
"""
    toml = """
[[tool.importlinter.contracts]]
name = "layers"
type = "layers"
containers = ["shop"]
layers = ["web", "domain", "store"]
"""
    for cfg in ({".importlinter": ini}, {"pyproject.toml": toml}):
        contracts = parse_import_linter(cfg)
        assert len(contracts) == 1
        r = LayerRules(["shop/web/views.py", "shop/domain/x.py", "shop/store/db.py"], cfg)
        assert r.source == "import-linter" and r.eligible
        assert r.is_violation("shop/store/db.py", "shop/web/views.py")
        assert r.is_violation("shop/store/db.py", "shop/domain/x.py")  # contracts keep strict order
        assert not r.is_violation("shop/web/views.py", "shop/store/db.py")
    assert parse_import_linter({".importlinter": ini})[0][1] == ["shop.domain", "shop.events"]
