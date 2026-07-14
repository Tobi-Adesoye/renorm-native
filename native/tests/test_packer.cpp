#include <iostream>
#include <vector>
#include <cassert>
#include "../src/packing_internal.h"

void run_layout_dimension_test(int test_rows, int test_cols)
{
    int lda = test_cols;
    std::vector<float> src(test_rows * test_cols);
    
    // Fill with identifiable floating-point values
    for (int i = 0; i < test_rows; ++i)
    {
        for (int j = 0; j < test_cols; ++j)
        {
            src[i * lda + j] = static_cast<float>((i + 1) * 100 + j);
        }
    }

    // Allocate sufficient memory for full panels including padding space
    int padded_rows = ((test_rows + 7) / 8) * 8;
    std::vector<float> dst(padded_rows * test_cols, -999.0f);

    renorm::pack_a_panel(src.data(), dst.data(), test_rows, test_cols, lda);

    // Verify structural configuration properties
    const float* ptr = dst.data();
    for (int panel_idx = 0; panel_idx < padded_rows; panel_idx += 8)
    {
        int global_row_offset = panel_idx;
        for (int k = 0; k < test_cols; ++k)
        {
            for (int r = 0; r < 8; ++r)
            {
                int current_row = global_row_offset + r;
                float expected_val = 0.0f;
                
                if (current_row < test_rows)
                {
                    expected_val = src[current_row * lda + k];
                }

                if (*ptr != expected_val)
                {
                    std::cerr << "FAIL: Mismatch at Matrix (" << test_rows << "x" << test_cols 
                              << ") Panel Row Loop: " << current_row << ", Col: " << k 
                              << ". Got " << *ptr << ", Expected " << expected_val << "\n";
                    std::exit(1);
                }
                ptr++;
            }
        }
    }
}

int main()
{
    std::cout << "Starting Stage 4.4.1 Panel Packer Isolated Verification...\n";

    // Rigorous multidimensional bounds sweep matching required test vectors
    run_layout_dimension_test(8, 4);   // Exact single-panel block
    run_layout_dimension_test(5, 4);   // Single panel edge cleanup check
    run_layout_dimension_test(13, 6);  // Multi-panel block with edge wrap
    run_layout_dimension_test(17, 8);  // Multi-panel deep with fractional overhang

    std::cout << "SUCCESS: Stage 4.4.1 Packer passed all structural test vectors.\n";
    return 0;
}