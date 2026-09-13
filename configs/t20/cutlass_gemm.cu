#include <cstdio>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"

// T20 CUTLASS template path: one pinned device-wide GEMM template
// instantiation (Simt, forward-compatible with sm_89), C = A*B with
// constant inputs so the result is exactly verifiable: every element of
// C must equal K * (1.0f * 2.0f).

int main() {
  using ElementInputA = float;
  using ElementInputB = float;
  using ElementOutput = float;
  using ElementAccumulator = float;
  using ElementCompute = float;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;

  using Gemm = cutlass::gemm::device::Gemm<
      ElementInputA, LayoutA,
      ElementInputB, LayoutB,
      ElementOutput, LayoutC,
      ElementAccumulator,
      cutlass::arch::OpClassSimt,
      cutlass::arch::Sm50>;

  int const M = 128;
  int const N = 128;
  int const K = 128;

  cutlass::HostTensor<ElementInputA, LayoutA> A({M, K});
  cutlass::HostTensor<ElementInputB, LayoutB> B({K, N});
  cutlass::HostTensor<ElementOutput, LayoutC> C({M, N});
  cutlass::HostTensor<ElementOutput, LayoutC> Cref({M, N});

  for (int i = 0; i < M * K; ++i) A.host_data()[i] = 1.0f;
  for (int i = 0; i < K * N; ++i) B.host_data()[i] = 2.0f;
  for (int i = 0; i < M * N; ++i) {
    C.host_data()[i] = 0.0f;
    Cref.host_data()[i] = static_cast<float>(K) * 2.0f;
  }

  A.sync_device();
  B.sync_device();
  C.sync_device();

  cutlass::gemm::GemmCoord problem_size(M, N, K);
  typename Gemm::Arguments arguments(
      problem_size,
      {A.device_ref(), B.device_ref()},
      {C.device_ref(), C.device_ref()},
      {ElementCompute(1.0f)});

  cutlass::Status status = Gemm::invoke(arguments);
  if (status != cutlass::Status::kSuccess) {
    std::printf("cutlass-run-failed: %d\n", static_cast<int>(status));
    return 1;
  }
  C.sync_host();

  for (int i = 0; i < M * N; ++i) {
    if (C.host_data()[i] != Cref.host_data()[i]) {
      std::printf("cutlass-mismatch at %d: %f != %f\n", i, C.host_data()[i],
                  Cref.host_data()[i]);
      return 1;
    }
  }
  std::printf("cutlass-ok\n");
  return 0;
}
