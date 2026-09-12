import concurrent.futures
import hashlib
import json
from pathlib import Path
from collect_sources import ROOT, get

FILES = {
 'ScalingIntelligence/KernelBench': ['EVAL.md','src/kernelbench/eval.py','src/kernelbench/timing.py','src/kernelbench/dataset.py','scripts/run_and_check.py','pyproject.toml'],
 'meta-pytorch/tritonbench': ['tritonbench/utils/triton_op.py','tritonbench/operators/gemm/operator.py'],
 'meta-pytorch/KernelAgent': ['triton_kernel_agent/opt_worker_component/profiling/kernel_profiler.py','triton_kernel_agent/opt_worker_component/orchestrator/optimization_orchestrator.py','kernel_perf_agent/kernel_opt/roofline/ncu_roofline.py'],
 'NVlabs/kda': ['docs/agent-flow.md'],
 'triton-lang/triton': ['python/triton/runtime/autotuner.py','python/triton/testing.py'],
 'NVIDIA/cutlass': ['media/docs/cpp/profiler.md'],
}
DOCS = {
 'ncu_profiling_guide': 'https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html',
 'ncu_cli': 'https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html',
 'compute_sanitizer': 'https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html',
 'nsight_systems': 'https://docs.nvidia.com/nsight-systems/UserGuide/index.html',
 'cuda_binary_utilities': 'https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html',
}

def fetch_one(job):
    target, url = job
    try:
        raw = get(url)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        return {'file': target.relative_to(ROOT).as_posix(), 'url': url, 'sha256': hashlib.sha256(raw).hexdigest()}
    except Exception as exc:
        return {'url': url, 'error': str(exc)}

if __name__ == '__main__':
    jobs=[]
    for repo, paths in FILES.items():
        dest=ROOT/'sources'/repo.replace('/','__')
        sha=json.loads((dest/'metadata.json').read_text())['commit']
        jobs.extend((dest/path, f'https://raw.githubusercontent.com/{repo}/{sha}/{path}') for path in paths)
    jobs.extend((ROOT/'sources'/'nvidia_docs'/f'{name}.html',url) for name,url in DOCS.items())
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        results=list(pool.map(fetch_one,jobs))
    (ROOT/'detail_manifest.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    for r in results:
        print(r.get('file',r['url']),r.get('error','OK'))
