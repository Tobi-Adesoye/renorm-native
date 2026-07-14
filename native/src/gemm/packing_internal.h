#ifndef RENORM_NATIVE_SRC_GEMM_PACKING_INTERNAL_H
#define RENORM_NATIVE_SRC_GEMM_PACKING_INTERNAL_H

namespace renorm
{

// Register block configuration definitions for AVX2 microkernel structures
constexpr int MR = 8;
constexpr int NR = 8;

/**
 * @brief Packs a row-major slice of Matrix A into contiguous MR x KC panels.
 * * @param src Pointer to the source submatrix of A.
 * @param dst Pointer to the packed destination workspace buffer.
 * @param rows The logical row count (mc) of the submatrix block.
 * @param cols The logical column count (kc) of the submatrix block.
 * @param lda The leading dimension (stride) of the original Matrix A.
 */
void pack_a_panel(
    const float* src,
    float* dst,
    int rows,
    int cols,
    int lda);

/**
 * @brief Packs a row-major slice of Matrix B into contiguous KC x NR panels.
 * * @param src Pointer to the source submatrix of B.
 * @param dst Pointer to the packed destination workspace buffer.
 * @param rows The logical row count (kc) of the submatrix block.
 * @param cols The logical column count (nc) of the submatrix block.
 * @param ldb The leading dimension (stride) of the original Matrix B.
 */
void pack_b_panel(
    const float* src,
    float* dst,
    int rows,
    int cols,
    int ldb);

} // namespace renorm

#endif // RENORM_NATIVE_SRC_GEMM_PACKING_INTERNAL_H