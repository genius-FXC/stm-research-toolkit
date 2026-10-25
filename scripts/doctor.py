"""Read-only dependency diagnostics. Optional proprietary tools are not installed."""
from pathlib import Path
import importlib
import json
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    errors = []
    print('Python:', sys.version.split()[0], '|', platform.system(), platform.machine())
    if sys.version_info < (3, 11):
        errors.append('Python 3.11+ is required')
    for name in ('numpy', 'scipy', 'matplotlib', 'PIL', 'tkinter', 'PySide6.QtWidgets',
                 'PySide6.QtWebEngineWidgets', 'pyqtgraph', 'OpenGL.GL'):
        try:
            importlib.import_module(name)
            print('[OK]', name)
        except Exception as exc:
            errors.append(name + ': ' + str(exc))
            print('[FAIL]', errors[-1])
    if sys.platform != 'win32':
        print('[NOTE] Desktop window management and Origin integration target Windows.')
    path = ROOT / 'registry/machine.json'
    try:
        config = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    except (OSError, ValueError) as exc:
        errors.append('Invalid machine.json: ' + str(exc)); config = {}
    for key in ('matlab', 'scilab'):
        executable = config.get(key) or ''
        found = bool(executable) and Path(executable).is_file()
        print('[OK]' if found else '[OPTIONAL]', key, executable if found else 'not configured in registry/machine.json')
    try:
        importlib.import_module('originpro')
        print('[OK] originpro import; actual Origin/COM requires a separate licensed installation.')
    except Exception:
        print('[OPTIONAL] Origin export: install requirements-origin.txt and licensed Origin.')
    db = json.loads((ROOT / 'registry/tools.json').read_text(encoding='utf-8'))
    for tool in db['tools']:
        if not (ROOT / tool['folder'] / tool['entry']).is_file():
            errors.append('Missing tool entry: ' + tool['id'])
    print('Core environment:', 'FAILED' if errors else 'READY', '| registered tools:', len(db['tools']))
    return bool(errors)

if __name__ == '__main__':
    sys.exit(main())
