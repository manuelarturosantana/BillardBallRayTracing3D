#pragma once

#include <functional>
#include <cmath>
#include <array>
#include <vector>
#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <complex>
#include <sys/time.h>
#include <sys/resource.h>
#include <fstream>
#include <unistd.h>

#include <mkl.h>
#include <omp.h>
#include <mpi.h>

namespace {
    double EQUAL_TOL = 1.0e-12;
}


inline std::vector<double> barycentric_weights(const std::vector<double>& nodes)
{

    int N = nodes.size();

    std::vector<double> weights(N, 1.0);
    
    for (int j = 0; j < N; j++) {

        double w = 1.0;

        for (int i = 0; i < N; i++) {

            if (i != j) {

                w *= (nodes[j] - nodes[i]);

            }

        }

        weights[j] = 1.0 / w;

    }

    return weights;

}

inline void lagrange_interpolation_2D(const std::vector<double>& xNodes, const std::vector<double>& yNodes,
                               const std::vector<double>& xWeights, const std::vector<double>& yWeights,
                               const std::vector<double>& fNodes, 
                               const std::vector<double>& x, const std::vector<double>& y,
                               double * solution)
{

    const int N = xNodes.size();
    const int M = yNodes.size();

    const int P = x.size();
    const int Q = y.size();

    std::vector<double> lx(P, 1.0);
    std::vector<double> ly(Q, 1.0);

    std::vector<int> idx_equal_x(P, -1);
    std::vector<int> idx_equal_y(Q, -1);

    std::vector<double> vec_x(P*N);
    std::vector<double> vec_y(M*Q);

    for (int i = 0; i < P; ++i) {

        double prod = 1.0;
        int eq_idx = -1;

        for (int j = 0; j < N; ++j) {

            double diff = x[i] - xNodes[j];

            if (std::abs(diff) < EQUAL_TOL) {

                eq_idx = j;
                break;

            }

            vec_x[i * N + j] = xWeights[j] / diff;

            prod *= diff;

        }

        if (eq_idx != -1) {

            std::fill(vec_x.begin() + i*N, vec_x.begin() + (i+1)*N, 0.0);
            vec_x[i*N + eq_idx] = 1.0;
            lx[i] = 1.0;

        } else {

            lx[i] = prod;

        }

    }

    for (int i = 0; i < Q; ++i) {

        double prod = 1.0;
        int eq_idx = -1;

        for (int j = 0; j < M; ++j) {

            double diff = y[i] - yNodes[j];

            if (std::abs(diff) < EQUAL_TOL) {

                eq_idx = j;
                break;

            }

            vec_y[i*M + j] = yWeights[j] / diff;

            prod *= diff;
        }

        if (eq_idx != -1) {

            std::fill(vec_y.begin() + i*M, vec_y.begin() + (i+1)*M, 0.0);
            vec_y[i*M + eq_idx] = 1.0;
            ly[i] = 1.0;

        } else {

            ly[i] = prod;

        }

    }

    std::vector<double> mat_prod(P*M);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, P, M, N, 1.0, &vec_x[0], N, &fNodes[0], M, 0.0, &mat_prod[0], M);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, P, Q, M, 1.0, &mat_prod[0], M, &vec_y[0], M, 0.0, solution, Q);

    for (int i = 0; i < P; i++) {
        for (int j = 0; j < Q; j++) {

            solution[i * Q + j] *= lx[i]*ly[j];

        }
    }

}
