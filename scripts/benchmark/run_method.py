# %%
"""Unified dispatch interface for the benchmark causality-estimation methods.

Each method script (PDIF.py, DDC.py, STE.py, GLMCC.py, CCM.py) exposes a
core_function(key, val, shuffle_id, noise_level, T) with almost the same
signature, but can't be `import`ed directly (dash in the filename) and some
import optional third-party deps (smite, glmcc) at module scope. This module
loads each script lazily by file path -- only the requested method's script
(and its deps) gets imported -- and gives them one common call signature.
"""
import importlib.util
import sys
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_PATH = SCRIPT_DIR.parents[1]

# so the dynamically loaded scripts' own `import utils` / `import crossmap_indices`
# resolve even if this module is imported from outside scripts/benchmark/
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

_METHOD_FILES = {
    'PDIF': 'PDIF.py',
    'DDC':    'DDC.py',
    'STE':    'STE.py',
    'GLMCC':  'GLMCC.py',
    'CCM':    'CCM.py',
    'FDCCM':  'CCM.py',
    'SCCM':   'CCM.py',
}

_module_cache = {}


def _load_module(method: str):
    fname = _METHOD_FILES[method]
    if fname not in _module_cache:
        mod_name = fname[:-3].replace('-', '_')
        spec = importlib.util.spec_from_file_location(mod_name, SCRIPT_DIR / fname)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _module_cache[fname] = module
    return _module_cache[fname]


def load_config(cfg_file: str = 'benchmark_causal.yml') -> dict:
    """Load a --cfg-file YAML and resolve each entry's `path` against the repo root."""
    with open(SCRIPT_DIR / cfg_file, 'r') as yamlfile:
        pm_causal_set = yaml.load(yamlfile, Loader=yaml.FullLoader)
    for key in pm_causal_set:
        pm_causal_set[key]['path'] = ROOT_PATH / pm_causal_set[key]['path']
    return pm_causal_set


def run_method(method: str, key: str, val: dict, shuffle_id: int = None,
               noise_level: float = None, T: float = None):
    """Run one causality-estimation method with the shared benchmark signature.

    Args:
        method: one of 'PDIF', 'DDC', 'STE', 'GLMCC', 'CCM', 'FDCCM', 'SCCM'.
        key: dataset key (e.g. 'HHEE'), matching a key in the loaded config.
        val: that key's config dict (as returned by load_config()[key]).
        shuffle_id: subnetwork shuffle index, or None for the full network.
        noise_level: measurement-noise level, or None/0.0 for noise-free (both are equivalent).
        T: data duration/length to use for estimation, or None for the full recording.
    """
    if method not in _METHOD_FILES:
        raise ValueError(f"Unknown method {method!r}; choose from {sorted(set(_METHOD_FILES))}")
    module = _load_module(method)
    if method in ('CCM', 'FDCCM', 'SCCM'):
        return module.core_function(key, val, shuffle_id=shuffle_id, ccm_type=method,
                                     noise_level=noise_level, T=T)
    return module.core_function(key, val, shuffle_id=shuffle_id,
                                 noise_level=noise_level, T=T)


# %%
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Unified dispatcher for the benchmark causality methods')
    parser.add_argument('--method', type=str, required=True, choices=sorted(_METHOD_FILES))
    parser.add_argument('--key', type=str, required=True)
    parser.add_argument('--idx', type=int, default=None)
    parser.add_argument('--noise_level', type=float, default=None)
    parser.add_argument('--T', type=float, default=None,
        help='Duration/length of data to use for estimation (unit is method-specific, '
             'same as that method\'s own --T). Defaults to the full recorded duration.')
    parser.add_argument('--cfg-file', dest='cfg_file', type=str, default='benchmark_causal.yml')
    args = parser.parse_args()

    pm_causal_set = load_config(args.cfg_file)
    run_method(args.method, args.key, pm_causal_set[args.key], args.idx, args.noise_level, args.T)
# %%
