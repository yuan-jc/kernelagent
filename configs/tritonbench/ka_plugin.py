"""T18 plugin: registers a project-authored variant into the pinned
tritonbench vector_add operator AT RUNTIME (via the upstream ``--plugin``
hook) without modifying any upstream file. The variant differs from the
operator's own Triton kernel only in meta-parameters (BLOCK_SIZE=4096),
so 'direct upstream comparison' is meaningful."""


def register_ka_plugins():
    import torch
    import triton
    from tritonbench.operators.vector_add.kernels import triton_add_kernel
    from tritonbench.utils.triton_op import register_benchmark

    @register_benchmark(operator_name="vector_add", func_name="ka_vector_add_sm89")
    def ka_vector_add_sm89(x: torch.Tensor, y: torch.Tensor):
        output = torch.empty_like(x)
        n_elements = output.numel()

        def grid(meta):
            return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        def _inner():
            triton_add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=4096)
            return output

        return _inner
