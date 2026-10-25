"""Self-contained source, asset and checkpoint archives for comparison runs."""
from pathlib import Path
import hashlib
import importlib.metadata
import inspect
import json
import platform
import shutil
import sys
from datetime import datetime, timezone


def archive_comparison_files(env, comparison_dir, checkpoint_file=None, checkpoint_dir=None):
    import brax
    import mpx
    import mujoco_playground
    import mujoco

    destination = Path(comparison_dir) / 'snapshot'
    destination.mkdir(parents=True, exist_ok=False)
    files = []

    def copy(source, relative):
        source = Path(source).resolve()
        target = destination / relative
        if target.exists():
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        files.append({'source': str(source), 'path': str(relative),
                      'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                      'bytes': target.stat().st_size})

    def tree(source, prefix, python_only=False):
        source = Path(source)
        for path in sorted(source.rglob('*.py' if python_only else '*')):
            if path.is_file() and not any(part in ('__pycache__', '.git') for part in path.relative_to(source).parts):
                copy(path, Path(prefix) / path.relative_to(source))

    entrypoint = Path(sys.argv[0])
    if entrypoint.is_file():
        copy(entrypoint, Path('entrypoint') / entrypoint.name)
    mpx_root = Path(mpx.__file__).resolve().parent
    for name in ('config', 'utils', 'jax_ocp_solvers'):
        tree(mpx_root / name, Path('mpx') / name)
    copy(mpx_root / '__init__.py', Path('mpx/__init__.py'))
    # Top-level comparison/training helpers, excluding checkpoint/result trees.
    for path in sorted((mpx_root / 'examples').glob('*.py')):
        copy(path, Path('mpx/examples') / path.name)
    pg_root = Path(mujoco_playground.__file__).resolve().parent
    tree(pg_root, 'mujoco_playground', python_only=True)
    env_root = Path(inspect.getfile(type(env))).resolve().parent
    tree(env_root, Path('mujoco_playground') / env_root.relative_to(pg_root))
    # Preserve locally modified PPO/network/normalizer and MJX Python sources.
    tree(Path(brax.__file__).resolve().parent, 'brax', python_only=True)
    tree(Path(mujoco.__file__).resolve().parent, 'mujoco', python_only=True)
    config = getattr(inspect.getmodule(type(env)), 'config', None)
    model_path = getattr(config, 'model_path', None)
    if model_path and Path(model_path).is_file():
        model_root = Path(model_path).resolve().parent
        if not model_root.is_relative_to(env_root):
            tree(model_root, 'mpc_model_assets')
    if checkpoint_file:
        copy(checkpoint_file, Path('checkpoint') / Path(checkpoint_file).name)
    if checkpoint_dir:
        run = Path(checkpoint_dir)
        for name in ('env_overrides.json', 'mpc_overrides.json', 'training_state.json', 'training_info.txt'):
            if (run / name).is_file():
                copy(run / name, Path('checkpoint') / name)
        if (run / 'source_snapshot').is_dir():
            tree(run / 'source_snapshot', 'checkpoint/training_source_snapshot')
    # Assets supplied virtually to MuJoCo may come from outside the env folder.
    asset_index = {}
    for index, (name, value) in enumerate(sorted(getattr(env, '_model_assets', {}).items())):
        relative = Path('runtime_assets') / f'{index:04d}_{Path(name).name}'
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(value)
        asset_index[name] = str(relative)
        files.append({'source': 'env._model_assets:' + name, 'path': str(relative),
                      'sha256': hashlib.sha256(value).hexdigest(), 'bytes': len(value)})
    packages = sorted({f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions() if d.metadata['Name']})
    (destination / 'requirements_versions.txt').write_text('\n'.join(packages) + '\n')
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(),
                'python': sys.version, 'executable': sys.executable,
                'platform': platform.platform(), 'files': files,
                'runtime_asset_names': asset_index,
                'scope': 'MPC sources, Playground sources, active robot assets, Brax/MJX Python sources, exact checkpoint and training provenance. Compiled libraries/drivers are represented by versions, not copied.'}
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (destination / 'README.txt').write_text(
        'Snapshot taken before rollout. manifest.json maps original paths to archived files and SHA256.\n'
        'Keep original extensions: XML/mesh/image assets are binary-safe copies.\n'
        'comparison_setup.txt in the parent records runtime overrides, environment and checkpoint.\n'
        'The archive is evidence of the setup, not an automatic environment installer.\n'
        '--redo preserves the original snapshot rather than recording the current machine.\n')
    print(f'[snapshot] {len(files)} files, {sum(f["bytes"] for f in files)/1e6:.1f} MB -> {destination}')
    return manifest
