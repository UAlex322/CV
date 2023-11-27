#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <utility>
#include <iomanip>
#include <chrono>
#include <random>
#include <omp.h>
using namespace std;

pair<vector<float>,vector<float>>
im2col(size_t k, size_t n, const vector<vector<float>>& kernels,
       size_t c, size_t h, size_t w, const vector<vector<float>>& images) {
    vector<float> mtx_ker(n * k * k * c), mtx_img((h - k + 1) * (w - k + 1) * k * k * c);
    float* mtx_ker_ptr = mtx_ker.data();
    float* mtx_img_ptr = mtx_img.data();

    // filling kernel matrix
// #pragma omp parallel for // collapse(2)
    for (size_t ker_i = 0; ker_i < n; ++ker_i) {
        for (size_t ch_i = 0; ch_i < c; ++ch_i) {
            const float* kernel = kernels[ker_i].data();
            float* mtx_row_ptr = mtx_ker_ptr + (ker_i * c + ch_i) * k * k;
            for (size_t i = 0; i < k * k; ++i) {
                mtx_row_ptr[i] = kernel[i];
            }
        }
    }

    // filling image matrix
    // block algorithm is efficient due to matrix's structure
    size_t mtx_img_w = (h - k + 1) * (w - k + 1);
    size_t block_w = w - k + 1;
    for (size_t ch_i = 0; ch_i < c; ++ch_i) {
        const float* image = images[ch_i].data();
// #pragma omp parallel for // collapse(2)
        for (size_t block_j = 0; block_j < h - k + 1; ++block_j) {
            for (size_t block_i = 0; block_i < k; ++block_i) {
                const float* img_row_ptr = image + (block_i + block_j) * w;
                float* mtx_row_ptr = mtx_img_ptr +
                       ch_i * (mtx_img_w * k * k) +   // channel
                       block_i * (mtx_img_w * k) +   // row block
                       block_j * block_w;            // column block
                for (size_t i = 0; i < k; ++i) {
                    for (size_t j = 0; j < block_w; ++j)
                        mtx_row_ptr[i * mtx_img_w + j] = img_row_ptr[i+j];
                    // memcpy(mtx_row_ptr + i * mtx_img_w, img_row_ptr + i, block_w * sizeof(float));
                }
            }
        }
    }

    return { mtx_ker, mtx_img };
}

pair<vector<float>, vector<float>>
im2col_transpose(size_t k, size_t n, const vector<vector<float>>& kernels,
    size_t c, size_t h, size_t w, const vector<vector<float>>& images) {
    vector<float> mtx_ker(n * k * k * c), mtx_img((h - k + 1) * (w - k + 1) * k * k * c);
    float* mtx_ker_ptr = mtx_ker.data();
    float* mtx_img_ptr = mtx_img.data();

    // filling kernel matrix
//#pragma omp parallel for collapse(2)
    for (size_t ker_i = 0; ker_i < n; ++ker_i) {
        for (size_t ch_i = 0; ch_i < c; ++ch_i) {
            const float* kernel = kernels[ker_i].data();
            float* mtx_row_ptr = mtx_ker_ptr + (ker_i * c + ch_i) * k * k;
            for (size_t i = 0; i < k * k; ++i) {
                mtx_row_ptr[i] = kernel[i];
            }
        }
    }

    // filling image matrix
    // block algorithm is efficient due to matrix's structure

    // size_t mtx_img_w = (h - k + 1) * (w - k + 1);
    // size_t block_w = w - k + 1;
    size_t mtx_img_w = k * k * c;
    size_t block_w = k;
//#pragma omp parallel for
    for (size_t ch_i = 0; ch_i < c; ++ch_i) {
        const float* image = images[ch_i].data();
        float* mtx_img_ch_ptr = mtx_img_ptr + ch_i * k * k;
        for (size_t block_i = 0; block_i < h - k + 1; ++block_i) {
            for (size_t block_j = 0; block_j < k; ++block_j) {
                const float* img_row_ptr = image + (block_i + block_j) * w;
                float* mtx_row_ptr = mtx_img_ch_ptr +
                    block_i * (mtx_img_w * (w - k + 1)) +
                    block_j * k;
                for (size_t i = 0; i < w - k + 1; ++i) {
                    memcpy(mtx_row_ptr + i * mtx_img_w, img_row_ptr + i, block_w * sizeof(float));
                }
            }
        }
    }

    return { mtx_ker, mtx_img };
}

vector<float> conv_naive(size_t k, size_t n, const vector<vector<float>>& kernels,
                         size_t c, size_t h, size_t w, const vector<vector<float>>& images) {
    size_t out_sz = (h-k+1)*(w-k+1);
    size_t out_h = h-k+1;
    size_t out_w = w-k+1;
    vector<float> out(n*out_sz);
#pragma omp parallel for
    for (size_t ker_idx = 0; ker_idx < n; ++ker_idx) {
        const float* ker_ptr = kernels[ker_idx].data();
        for (size_t ch_idx = 0; ch_idx < c; ++ch_idx) {
            const float* img_ptr = images[ch_idx].data();
            for (size_t i = 0; i < out_h; ++i)
                for (size_t j = 0; j < out_w; ++j)
                    for (size_t k_i = 0; k_i < k; ++k_i)
                        for (size_t k_j = 0; k_j < k; ++k_j)
                            out[ker_idx*out_sz + i*out_w + j] += ker_ptr[k_i*k + k_j] * img_ptr[(i+k_i)*w + (j+k_j)];
        }
    }
    return out;
}

vector<float> mtx_mult(size_t M, size_t N, size_t K,
                       const vector<float> &A_vec, const vector<float> &B_vec,
                       size_t block_h, size_t block_w, size_t block_w2) {
    vector<float> C_vec(M*K);

    const float *A = A_vec.data();
    const float *B = B_vec.data();
          float *C = C_vec.data();

    size_t blocks_i = (M + block_h - 1)/block_h;
    size_t blocks_k = (N + block_w - 1)/block_w;
    size_t blocks_j = (K + block_w2 - 1)/block_w2;

#pragma omp parallel for collapse(2)
    for (size_t bi = 0; bi < blocks_i; ++bi) {
        for (size_t bj = 0; bj < blocks_j; ++bj) {
            size_t i_len = min(block_h, M - bi * block_h);
            size_t j_len = min(block_w2, K - bj * block_w2);
            float *c_ptr = C + bi*block_h*K + bj*block_w2;

            for (size_t bk = 0; bk < blocks_k; ++bk) {
                size_t k_len = min(block_w, N - bk * block_w);
                const float *a_ptr = A + bi*block_h*N + bk*block_w;
                const float *b_ptr = B + bk*block_w*K + bj*block_w2;

                for (size_t i = 0; i < i_len; ++i) {
                    for (size_t k = 0; k < k_len; ++k) {
                    #pragma omp simd
                        for (size_t j = 0; j < j_len; ++j)
                            c_ptr[i*K + j] += a_ptr[i*N + k] * b_ptr[k*K + j];
                    }
                }
            }
        }
    }
    return C_vec;
}

void test(size_t k, size_t n, size_t c, size_t h, size_t w) {
    mt19937 rng{random_device{}()};
    uniform_real_distribution<float> dist(-1.0,1.0);

    // initialize kernels
    vector<vector<float>> kernels(n, vector<float>(k*k));
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < k*k; ++j)
            kernels[i][j] = dist(rng);

    // initialize image channels
    vector<vector<float>> images(c, vector<float>(h*w));
    for (size_t i = 0; i < c; ++i)
        for (size_t j = 0; j < h*w; ++j)
            images[i][j] = dist(rng);

    // run IM2COL
    auto begin = chrono::high_resolution_clock::now();
    auto [mtx_ker, mtx_img] = im2col(k, n, kernels, c, h, w, images);
    auto end = chrono::high_resolution_clock::now();
    cout << "Elapsed time on im2col: " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << " milliseconds\n";

    // run GEMM
    // vector<float> mtx_product(n*(h-k+1)*(w-k+1));
    begin = chrono::high_resolution_clock::now();
    vector<float> mtx_product = mtx_mult(n, k*k*c, (h-k+1)*(w-k+1), mtx_ker, mtx_img, 64, 64, 256);
    // cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, (h-k+1)*(w-k+1), k*k*c,
    //     1.0f, mtx_ker.data(), k*k*c, mtx_img.data(), (h-k+1)*(w-k+1), 0.0f, mtx_product.data(), (h-k+1)*(w-k+1));
    end = chrono::high_resolution_clock::now();
    cout << "Elapsed time on GEMM: " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << " milliseconds\n";

    // run naive convolution
    begin = chrono::high_resolution_clock::now();
    vector<float> conv_naive_res = conv_naive(k, n, kernels, c, h, w, images);
    end = chrono::high_resolution_clock::now();
    cout << "Elapsed time on naive convolution: " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << " milliseconds\n";

    // calculate abosolute error
    float max_error = 0.0f;
    for (size_t i = 0; i < n*(h-k+1)*(w-k+1); ++i)
        max_error = max(max_error, abs(conv_naive_res[i]-mtx_product[i]));
    cout << "Error: " << max_error;
}

// void test1() {
//     size_t n = 2, k = 3, c = 2, h = 5, w = 6;
//     vector<vector<float>> kernels{
//         {1, 2, 3, 4, 5, 6, 7, 8, 9},
//         {9, 8, 7, 6, 5, 4, 3, 2, 1}
//     };
//     vector<vector<float>> images{vector<float>(h*w),vector<float>(h*w)};
//     iota(images[0].begin(), images[0].end(), 0);
//     iota(images[1].begin(), images[1].end(), 0);
//     reverse(images[1].begin(), images[1].end());
//     auto [mtx_ker, mtx_img] = im2col(k, n, kernels, c, h, w, images);
//     for (size_t i = 0; i < n; ++i) {
//         for (size_t j = 0; j < k*k*c; ++j)
//             cout << setw(3) << int(mtx_ker[i*k*k*c + j]);
//         cout << '\n';
//     }
//     cout << "\n\n";
//     for (size_t i = 0; i < k*k*c; ++i) {
//         for (size_t j = 0; j < (h-k+1)*(w-k+1); ++j)
//             cout << setw(3) << int(mtx_img[i*(h-k+1)*(w-k+1) + j]);
//         cout << '\n';
//     }
// }
// 
// void test1_transpose() {
//     size_t n = 2, k = 3, c = 2, h = 5, w = 6;
//     vector<vector<float>> kernels{
//         {1, 2, 3, 4, 5, 6, 7, 8, 9},
//         {9, 8, 7, 6, 5, 4, 3, 2, 1}
//     };
//     vector<vector<float>> images{ vector<float>(h * w),vector<float>(h * w) };
//     iota(images[0].begin(), images[0].end(), 0);
//     iota(images[1].begin(), images[1].end(), 0);
//     reverse(images[1].begin(), images[1].end());
//     auto [mtx_ker, mtx_img] = im2col_transpose(k, n, kernels, c, h, w, images);
//     for (size_t i = 0; i < n; ++i) {
//         for (size_t j = 0; j < k * k * c; ++j)
//             cout << setw(3) << int(mtx_ker[i * k * k * c + j]);
//         cout << '\n';
//     }
//     cout << "\n\n";
//     for (size_t j = 0; j < (h - k + 1) * (w - k + 1); ++j) {
//         for (size_t i = 0; i < k * k * c; ++i)
//             cout << setw(3) << int(mtx_img[j*k*k*c + i]);
//         cout << '\n';
//     }
// }

struct test_parameters {
    size_t k, n, c, h, w;
};

int main(int argc, char *argv[]) {
    vector<test_parameters> tests{
        {3,5,10,32,32},
        {5,20,10,256,256},
        {3,20,3,480,640},
        {3,20,10,720,1280},
        {3,20,10,1280,1920}
    };
    for (int i = 0; i < 5; ++i) {
        cout << "TEST " << i << ": image " << tests[i].w << "x" << tests[i].h
             << ", " << tests[i].n << " kernels " << tests[i].k << "x" << tests[i].k
             << ", " << tests[i].c << "channels\n";
        test(tests[i].k, tests[i].n, tests[i].c, tests[i].h, tests[i].w);
        cout << "\n\n";
    }
}