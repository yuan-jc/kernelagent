"""Build an offline inventory of the exact research snapshots already collected."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
records = []
lines = [
    '# 调研来源索引', '',
    '调研日期：2026-09-12。以下是设计调研快照，不是已经验证可兼容的运行时依赖锁。', '',
    'README/源码属于上游资料，未作为本项目代码执行。各项目保留原有许可证；未完整审计每个仓库。', '',
    '设计文档区分已核对源码、README 描述和未实测能力。本索引中的文件被保存不等于每行都经人工审计。', '',
    '| 项目 | 固定 commit | 本地材料 |',
    '|---|---|---|',
]
for p in sorted((ROOT / 'sources').glob('*/metadata.json')):
    meta = json.loads(p.read_text(encoding='utf-8'))
    repo, sha = meta['repository'], meta['commit']
    files = []
    for f in sorted(p.parent.rglob('*')):
        if not f.is_file() or f.name in ('metadata.json', 'tree.json'):
            continue
        rel = f.relative_to(p.parent).as_posix()
        files.append({
            'path': f.relative_to(ROOT).as_posix(),
            'url': f'https://raw.githubusercontent.com/{repo}/{sha}/{rel}',
            'sha256': hashlib.sha256(f.read_bytes()).hexdigest(),
        })
    meta['files'] = files
    records.append(meta)
    local = p.parent.relative_to(ROOT).as_posix()
    lines.append(f'| [{repo}](https://github.com/{repo}/tree/{sha}) | `{sha}` | 本地缓存 `{local}`（不随公开仓库分发） |')

details = json.loads((ROOT / 'detail_manifest.json').read_text(encoding='utf-8'))
web = [r for r in details if r.get('file', '').startswith('sources/nvidia_docs/')]
lines += ['', '## NVIDIA 官方文档', '', '| 文档 | 本地快照 |', '|---|---|']
for r in web:
    lines.append(f'| [{Path(r["file"]).stem}]({r["url"]}) | 本地缓存 `{r["file"]}`（不随公开仓库分发） |')
lines += [
    '', '## 复核与获取记录', '',
    '- `snapshot_manifest.json` 是最终快照清单：仓库 commit、文件 URL 和 SHA-256。',
    '- `source_manifest.json`、`additional_manifest.json`、`extension_manifest.json`、`detail_manifest.json` 是分批获取记录。',
    '- extension_manifest 中三个项目最初遇到匿名 GitHub API 限流；之后通过已连接 GitHub 工具成功获取固定 commit 与材料，最终结果以 snapshot_manifest 为准。',
    '- SakanaAI/AI-CUDA-Engineer 精确 URL 返回 404，未纳入成功快照，也不据此断言其研究不存在。',
    '- `collect_sources.py`、`collect_details.py` 可重新获取公开资料；重新运行可能解析到更新的主分支，不能将新结果冒充本次快照。',
    '- `build_source_index.py` 只重建本地清单，不联网、不更新源文件。',
    '- `.txt` 是官方 HTML 的派生纯文本，原始 HTML hash 才是获取清单中记录的内容身份。',
    '- 本次未安装研究项目依赖、未编译 kernel、未做 GPU benchmark，所有性能建议均待实施验证。',
]
(ROOT / 'snapshot_manifest.json').write_text(json.dumps({'retrieved_date': '2026-09-12', 'repositories': records, 'official_docs': web}, ensure_ascii=False, indent=2), encoding='utf-8')
(ROOT / 'SOURCES.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(f'Indexed {len(records)} repositories, {sum(len(r["files"]) for r in records)} source files, {len(web)} official documents')
