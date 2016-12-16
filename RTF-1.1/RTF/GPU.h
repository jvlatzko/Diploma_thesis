/* This file is part of the "Regression Tree Fields" (RTF) source code distribution,
 * obtained from http://research.microsoft.com/downloads.
 * It is provided to you under the terms of the Microsoft Research License Agreement
 * (MSR-LA). Please see License.txt for details.
 *
 *
 * File: GPU.cu
 * Interface definition for RTF inference on a GPU. Requires CUDA SDK and CUSP.
 *
 */

#ifndef H_RTF_GPU_H
#define H_RTF_GPU_H

#include <vector>
#include <memory>

namespace GPU
{
    class ConjugateGradientSolver;

    std::shared_ptr<ConjugateGradientSolver> GetSolver(const int numRows, const std::vector<int>& rowIndices,
            const int numCols, const std::vector<int>& colIndices,
            const std::vector<float>& values);

    void SolveViaConjugateGradient(std::shared_ptr<ConjugateGradientSolver>& solver, const std::vector<float>& rhs, std::vector<float>& sol, size_t maxNumIt, float residualTol);
}

#endif // H_RTF_GPU_H
