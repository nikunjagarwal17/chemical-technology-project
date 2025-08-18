"""I/O helpers for SimpleCantera MVP: YAML spec loader."""
import yaml


def load_spec_from_yaml(path: str) -> dict:
    """Load a simulation spec from a YAML file and return a dict compatible with run_simulation."""
    with open(path, 'r') as f:
        doc = yaml.safe_load(f)
    return doc
