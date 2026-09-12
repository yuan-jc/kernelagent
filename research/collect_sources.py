"""Download public primary sources for the design review; never execute them."""
import concurrent.futures
import datetime
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent
REPOS = [
    'ScalingIntelligence/KernelBench', 'meta-pytorch/tritonbench',
    'thunlp/TritonBench', 'meta-pytorch/KernelAgent',
    'SakanaAI/AI-CUDA-Engineer', 'NVIDIA/cutlass', 'triton-lang/triton',
    'tile-ai/tilelang', 'KernelTuner/kernel_tuner', 'NVIDIA/nvbench',
    'NVIDIA/MatX', 'flashinfer-ai/flashinfer',
]

def get(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'kernelagent-design-research'})
    return urllib.request.urlopen(req, timeout=40).read()

def collect(repo):
    dest = ROOT / 'sources' / repo.replace('/', '__')
    dest.mkdir(parents=True, exist_ok=True)
    try:
        meta = json.loads(get(f'https://api.github.com/repos/{repo}'))
        commit = json.loads(get(f'https://api.github.com/repos/{repo}/commits/{meta["default_branch"]}'))
        sha = commit['sha']
        tree = json.loads(get(f'https://api.github.com/repos/{repo}/git/trees/{sha}?recursive=1'))
        (dest / 'tree.json').write_text(json.dumps(tree, indent=2), encoding='utf-8')
        paths = [x['path'] for x in tree['tree'] if x['type'] == 'blob']
        selected = [p for p in paths if '/' not in p and (p.lower().startswith('readme') or p.lower().startswith('license'))]
        files = []
        for path in selected:
            url = f'https://raw.githubusercontent.com/{repo}/{sha}/{path}'
            raw = get(url)
            (dest / path).write_bytes(raw)
            files.append({'path': path, 'url': url, 'sha256': hashlib.sha256(raw).hexdigest()})
        record = {'repository': repo, 'url': meta['html_url'], 'commit': sha,
                  'commit_date': commit['commit']['committer']['date'],
                  'retrieved_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'license': meta.get('license'), 'archived': meta['archived'], 'files': files}
        (dest / 'metadata.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
        return record
    except Exception as exc:
        return {'repository': repo, 'error': str(exc)}

if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        records = list(pool.map(collect, REPOS))
    (ROOT / 'source_manifest.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    for r in records:
        print(r['repository'], r.get('commit', r.get('error')))
