"""I/O helpers for Pyroxa MVP: YAML/CTI spec loader."""
import yaml
import os


def load_spec_from_yaml(path: str) -> dict:
    """Load a simulation spec from a YAML file and return a dict compatible with run_simulation.

    If the extension is `.cti` a minimal CTI parser placeholder converts it to an internal dict.
    """
    _, ext = os.path.splitext(path)
    with open(path, 'r') as f:
        txt = f.read()
    if ext.lower() == '.cti':
        # Minimal placeholder: CTI is complex; users should use Cantera for full CTI parsing.
        # We implement very small parsing for simple A <=> B in format:
        # species A B
        # reaction('A <=> B', [kf, kr])
        spec = {'species': [], 'reactions': []}
        for line in txt.splitlines():
            line = line.strip()
            if line.startswith('species'):
                parts = line.split()
                spec['species'] = parts[1:]
            if line.startswith('reaction'):
                # crude parse
                inside = line[line.find('(')+1:line.rfind(')')]
                if ',' in inside:
                    rxnexpr, params = inside.split(',', 1)
                    rxnexpr = rxnexpr.strip().strip("'\"")
                    params = params.strip().strip('[]')
                    vals = [float(x) for x in params.split() if x.replace('.','',1).isdigit()]
                    # support only A <=> B
                    if '<=>' in rxnexpr:
                        a, b = [s.strip() for s in rxnexpr.split('<=>')]
                        spec['reactions'].append({'kf': vals[0] if vals else 1.0, 'kr': vals[1] if len(vals)>1 else 0.0, 'reactants': {a:1}, 'products': {b:1}})
        return spec
    else:
        return yaml.safe_load(txt)


def parse_mechanism(path: str) -> dict:
    """Compatibility wrapper: alias for load_spec_from_yaml.

    Some users expect a function named `parse_mechanism` (from CTI-style tools).
    Provide a thin alias that returns the same spec dict as `load_spec_from_yaml`.
    """
    return load_spec_from_yaml(path)
