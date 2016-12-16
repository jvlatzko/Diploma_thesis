/* This file is part of the "Regression Tree Fields" (RTF) source code distribution,
 * obtained from http://research.microsoft.com/downloads.
 * It is provided to you under the terms of the Microsoft Research License Agreement
 * (MSR-LA). Please see License.txt for details.
 *
 *
 * File: Priors.h
 * Implements priors over the vector and matrix coefficients of Weights instances.
 *
 */

#ifndef H_RTF_PRIORS_H
#define H_RTF_PRIORS_H

#include "Types.h"
#include "Utility.h"

namespace Priors
{
    /* Penalizes based on the difference between the largest eigenvalue and the
     * smallest eigenvalue. This has the effect of "drawing" the negative
     * precision parameters towards being diagonal.
     * However, note that all diagonal matrices with uniform entries incur a
     * penalty of precisely zero, irrespective of the magnitude of the entries.
     * As such, this prior does not necessarily protect against data with
     * zero variance (or, equivalently, infinite precision). However, it does
     * act to enforce a suitable condition number of our parameter matrices.
     * See http://www.maths.unsw.edu.au/sites/default/files/amr01_6_0.pdf
     * for an explanation of how the gradient arises.
     */
    template <typename TValue, size_t VarDim, size_t BasisDim>
    class SpreadOfEigenvaluesPrior
    {
    public:
        typedef Eigen::Matrix<TValue, VarDim, VarDim> TMatrix;
        typedef Eigen::Matrix<TValue, VarDim, 1>      TVector;

        static TValue Eval(TValue C, const TMatrix& Wq, TMatrix& NablaWq)
        {
            const TValue eps = static_cast<TValue>(1e-0);
            Eigen::SelfAdjointEigenSolver<TMatrix> solver(Wq);
            auto Q          = solver.eigenvectors();
            auto lambda     = solver.eigenvalues();
            TVector mu_max;
            auto lambda_max = eps * Utility::LogSumExp(TVector(lambda / eps), mu_max);
            TVector mu_min;
            auto lambda_min = -eps * Utility::LogSumExp(TVector(- lambda / eps), mu_min);
            NablaWq = C * (Q * (mu_max - mu_min).asDiagonal() * Q.transpose());
            return C * (lambda_max - lambda_min);
        }

        static TValue ComputeObjectiveAddGradient(TValue linearConstant, TValue quadraticConstant, Compute::Weights<TValue, VarDim, BasisDim>& w)
        {
            TMatrix NablaWq;
            auto objectiveContribution = Eval(quadraticConstant, w.Wq, NablaWq);
            w.GetGq() += NablaWq;
            return objectiveContribution;
        }

        static TValue DefaultLinearConstant()
        {
            return static_cast<TValue>(1e-8);
        }

        static TValue DefaultQuadraticConstant()
        {
            return static_cast<TValue>(1e-8);
        }
    };

    /* Penalizes based on the largest absolute eigenvalue.
     * In our case, since the eigenvalues are bounded from above by zero,
     * this is a large _negative_ number, i.e. the smallest eigenvalue.
     * This prior protects effectively against ill-conditioned problems with
     * zero variance (or, equivalently, infinite precision) at the leaves of
     * the regression trees.
     * See http://www.maths.unsw.edu.au/sites/default/files/amr01_6_0.pdf
     * for an explanation of how the gradient arises.
     */
    template <typename TValue, size_t VarDim, size_t BasisDim>
    class LargestEigenvaluePrior
    {
    public:
        typedef Eigen::Matrix<TValue, VarDim, VarDim> TMatrix;
        typedef Eigen::Matrix<TValue, VarDim, 1>      TVector;

        static TValue Eval(TValue C, const TMatrix& Wq, TMatrix& NablaWq)
        {
            const TValue eps = static_cast<TValue>(1e-0);
            Eigen::SelfAdjointEigenSolver<TMatrix> solver(Wq);
            auto Q          = solver.eigenvectors();
            auto lambda     = solver.eigenvalues();
            TVector mu_min;
            auto lambda_min = -eps * Utility::LogSumExp(TVector(- lambda / eps), mu_min);
            NablaWq = -C * (Q * mu_min.asDiagonal() * Q.transpose());
            return  -C * lambda_min;
        }

        static TValue ComputeObjectiveAddGradient(TValue linearConstant, TValue quadraticConstant, Compute::Weights<TValue, VarDim, BasisDim>& w)
        {
            TMatrix NablaWq;
            auto objectiveContribution = Eval(quadraticConstant, w.Wq, NablaWq);
            w.GetGq() += NablaWq;
            return objectiveContribution;
        }

        static TValue DefaultLinearConstant()
        {
            return static_cast<TValue>(1e-8);
        }

        static TValue DefaultQuadraticConstant()
        {
            return static_cast<TValue>(1e-8);
        }
    };

    /* Penalizes based on the sum of eigenvalues of the negative precision
     * parameter (i.e., its trace tr(W)).
     * The gradient of tr(W) is simply the identity matrix I.
     */
    template <typename TValue, size_t VarDim, size_t BasisDim>
    class SumOfEigenvaluesPrior
    {
    public:
        typedef Eigen::Matrix<TValue, VarDim, VarDim> TMatrix;

        static TValue Eval(TValue C, const TMatrix& Wq, TMatrix& NablaWq)
        {
            NablaWq = - C * TMatrix::Identity();
            return - C * Wq.trace();
        }

        static TValue ComputeObjectiveAddGradient(TValue linearConstant, TValue quadraticConstant, Compute::Weights<TValue, VarDim, BasisDim>& w)
        {
            TMatrix NablaWq;
            auto objectiveContribution = Eval(quadraticConstant, w.Wq, NablaWq);
            w.GetGq() += NablaWq;
            return objectiveContribution;
        }

        static TValue DefaultLinearConstant()
        {
            return TValue(1e-8);
        }

        static TValue DefaultQuadraticConstant()
        {
            return TValue(1e-8);
        }
    };

    /* Regularizes through the squares of the parameter components.
     * This has no meaningful interpretation but enforces strict convexity of
     * the overall objective, which is a desirable property from the optimization
     * point of view. Use small regularization constants!
     */
    template <typename TValue, size_t VarDim, size_t BasisDim>
    class FrobeniusPrior
    {
    public:

        static TValue ComputeObjectiveAddGradient(TValue linearC, TValue quadraticC, Compute::Weights<TValue, VarDim, BasisDim>& w)
        {
            w.GetGl() += linearC    * w.Wl;
            w.GetGq() += quadraticC * w.Wq;

            const auto f = TValue(0.5) * (linearC * w.Wl.squaredNorm() + quadraticC * w.Wq.squaredNorm());
            return f;
        }

        static TValue DefaultLinearConstant()
        {
            return TValue(1e-8);
        }

        static TValue DefaultQuadraticConstant()
        {
            return TValue(1e-8);
        }
    };

    /* The conjugate prior of a Gaussian consists of a Gaussian over the mean
     * vector multiplied by a Wishart over the Precision matrix.
     * We use an uninformative conjugate prior here.
     * The parameters of the Wishart are chosen as V = I and n = VarDim+1, such
     * that the Wishart distribution has VarDim+1 degrees of freedom and the
     * identity matrix as its mode.
     * See http://en.wikipedia.org/wiki/Wishart_distribution.
     * Modulo a constant factor, the log of a Wishart then reduces to - Tr(P),
     * where P is the precision matrix.
     * The offset vector (which is closely related to the mean) is regularized
     * using the squared L2 norm, which corresponds to a zero-mean Gaussian.
     * The constant factors of the distributions must be chosen as hyper-parameters.
     */
    template <typename TValue, size_t VarDim, size_t BasisDim>
    class ConjugatePrior
    {
    public:
        typedef Eigen::Matrix<TValue, VarDim, VarDim> TMatrix;

        static TValue ComputeObjectiveAddGradient(TValue linearC, TValue quadraticC, Compute::Weights<TValue, VarDim, BasisDim>& w)
        {
            w.GetGl() +=   linearC    * w.Wl;
            w.GetGq() += - quadraticC * TMatrix::Identity();
            return (TValue(0.5) * linearC * w.Wl.squaredNorm()) - (quadraticC * w.Wq.trace());
        }

        static TValue DefaultLinearConstant()
        {
            return TValue(1e-8);
        }

        static TValue DefaultQuadraticConstant()
        {
            return TValue(1e-8);
        }
    };

    /* The default prior, which does not penalize at all.
     */
    template <typename TValue, size_t VarDim, size_t BasisDim>
    class NullPrior
    {
    public:

        static TValue ComputeObjectiveAddGradient(TValue linearConstant, TValue quadraticConstant, Compute::Weights<TValue, VarDim, BasisDim>& w)
        {
            return TValue(0);
        }

        static TValue DefaultLinearConstant()
        {
            return TValue(0);
        }

        static TValue DefaultQuadraticConstant()
        {
            return TValue(0);
        }
    };
}

#endif // H_RTF_PRIORS_H
