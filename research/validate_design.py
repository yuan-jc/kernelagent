"""Static QA for the authored design, without executing example GPU code."""
import ast
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
doc = next((ROOT / 'docs').glob('*.md'))
text = doc.read_text(encoding='utf-8')
blocks = re.findall(r'```(\w+)\n(.*?)```', text, re.S)
for language, code in blocks:
    if language == 'python':
        ast.parse(code)
    elif language == 'json':
        json.loads(code)
used = set(re.findall(r'\[S\d+\]', text))
defined = set(re.findall(r'^- \*\*(\[S\d+\])', text, re.M))
assert used == defined, (used - defined, defined - used)
assert text.count('```') % 2 == 0
for p in (doc, ROOT / 'README.md', ROOT / 'research' / 'SOURCES.md'):
    for target in re.findall(r'\]\(([^)]+)\)', p.read_text(encoding='utf-8')):
        if target.startswith(('https://', 'http://', '#')):
            continue
        assert (p.parent / target.split('#')[0]).exists(), (p, target)
manifest = json.loads((ROOT / 'research' / 'snapshot_manifest.json').read_text(encoding='utf-8'))
files = [f for r in manifest['repositories'] for f in r['files']]
files += [{'path': f['file'], **f} for f in manifest['official_docs']]
for f in files:
    assert hashlib.sha256((ROOT / 'research' / f['path']).read_bytes()).hexdigest() == f['sha256'], f['path']
print(json.dumps({
    'design_characters': len(text), 'design_lines': len(text.splitlines()),
    'python_blocks_parsed': sum(lang == 'python' for lang, _ in blocks),
    'json_blocks_parsed': sum(lang == 'json' for lang, _ in blocks),
    'source_references': len(defined), 'snapshot_hashes_verified': len(files),
    'local_links': 'passed', 'gpu_execution': 'not performed',
}, indent=2))
