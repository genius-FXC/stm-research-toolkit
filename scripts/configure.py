"""Initialize user-owned configuration, without overwriting existing settings."""
from pathlib import Path
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]

def configure():
    for name in ('jobs', 'logs', 'sessions'):
        (ROOT / name).mkdir(exist_ok=True)
    path = ROOT / 'registry/machine.json'
    if not path.exists():
        config = json.loads((ROOT / 'registry/machine.example.json').read_text(encoding='utf-8'))
        config['matlab'] = shutil.which('matlab') or ''
        config['scilab'] = shutil.which('Scilex') or shutil.which('scilab') or ''
        path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    folder = ROOT / 'modules/stm_line_spectrum_tool'
    if not (folder / 'config.json').exists():
        shutil.copyfile(folder / 'config.example.json', folder / 'config.json')

if __name__ == '__main__':
    configure()
    print('Local configuration is ready. Existing settings preserved.')
