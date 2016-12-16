/* This file is part of the "Regression Tree Fields" (RTF) source code distribution,
 * obtained from http://research.microsoft.com/downloads.
 * It is provided to you under the terms of the Microsoft Research License Agreement
 * (MSR-LA). Please see License.txt for details.
 *
 *
 * File: Unary.h
 * Implements all functionality related to unary factors in a regression tree field.
 *
 */

#ifndef H_RTF_UNARY_H
#define H_RTF_UNARY_H

#include <vector>
#include <cstring>
#include <cassert>

#include "Types.h"
#include "Priors.h"
#include "Training.h"
#include "Utility.h"

namespace Unary
{
    // The unary factor type class. Augments the FactorTypeBase class by functionality
    // that is specific to *unary* factors.
    template<typename TFeature, typename TLabel, typename TPrior, typename TBasis>
    class FactorType : public Compute::FactorTypeBase<TFeature, TLabel, TPrior, TBasis>
    {
    private:
        int quadraticBasisIndex;

    public:
        typedef Compute::FactorTypeBase<TFeature, TLabel, TPrior, TBasis> Base;
        typedef typename TLabel::ValueType                  TValue;
        static const size_t                                 VarDim   = TLabel::Size;
        static const size_t                 BasisDim = TBasis::Size;
        typedef Eigen::Matrix<TValue, VarDim, 1>            TVector;
        typedef Eigen::Matrix<TValue, Eigen::Dynamic, 1>    TSolution;
        typedef Eigen::Matrix<TValue, VarDim, VarDim>       TMatrix;
        typedef Eigen::Matrix<TValue, BasisDim, 1>          TBasisVector;
        typedef Compute::Weights<TValue, VarDim, BasisDim>  TWeights;
        typedef Compute::SystemVectorRef<TValue, VarDim>    TSystemVectorRef;
        typedef Compute::BlockDiagonalRef<TValue, VarDim>   TBlockDiagonalRef;
        typedef Compute::SystemVectorCRef<TValue, VarDim>   TSystemVectorCRef;
        typedef Compute::SystemMatrixRow<TValue, VarDim>    TSystemMatrixRow;

        class Factor
        {
        private:
            TWeights* const          u;
            Eigen::Map<TBasisVector> b;
            TValue                   bq;
            const Vector2D<int>&     i;

        public:
            Factor(TWeights* u_, TValue* b_, TValue bq_, const Vector2D<int>& i_) : u(u_), b(b_), bq(bq_), i(i_)
            {
            }

            void AccumulateGradient(const TSystemVectorCRef& muPrediction, const TSystemVectorCRef& muLossGradient, TValue normC) const
            {
                AccumulateGradient(muPrediction, muLossGradient, normC, u);
            }

            void AccumulateGradient(const TSystemVectorCRef& yRef, const std::vector<TSolution>& muPrediction, const std::vector<TSolution>& muLossGradient, TValue normC) const
            {
                AccumulateGradient(yRef, muPrediction, muLossGradient, normC, u);
            }

            void AccumulateGradient(const TSystemVectorCRef& muPrediction, const TSystemVectorCRef& muLossGradient,
                                    TValue normC, TWeights* w) const
            {
                const auto scale = 1.0/normC;
                w->GetGl() += (scale * bq * muLossGradient(i)) * b.transpose();
                w->GetGq() += (scale * bq * muLossGradient(i)) * muPrediction(i).transpose();
            }

            void AccumulateGradient(const TSystemVectorCRef& yRef, const std::vector<TSolution>& muPrediction, const std::vector<TSolution>& muLossGradient,
                                    TValue normC, TWeights* w) const
            {
                const auto N = muPrediction.size()-1;
                const auto cx = yRef.Width(), cy = yRef.Height();
                const auto scale = 1.0/normC;

                for( int k = N-1; k >=0; --k )
                {
                    const TSystemVectorCRef muPredictionRef(cx, cy, muPrediction[k+1]);
                    const TSystemVectorCRef muLossGradientRef(cx, cy, muLossGradient[k+1]);

                    w->GetGl() += (scale * bq) * (muLossGradientRef(i) * b.transpose());
                    w->GetGq() += (scale * bq) * (muLossGradientRef(i) * muPredictionRef(i).transpose());
                }
            }


            void AccumulateEnergyBasedGradient(const TSystemVectorCRef& yhat, const TSystemVectorCRef& ystar, TValue normC) const
            {
                AccumulateEnergyBasedGradient(yhat, ystar, normC, u);
            }

            void AccumulateEnergyBasedGradient(const TSystemVectorCRef& yhat, const TSystemVectorCRef& ystar,
                                               TValue normC, TWeights* w) const
            {
                const auto scale = 1.0/normC;

                w->GetGl() += (yhat(i) - ystar(i)) * (scale * bq * b.transpose());
                w->GetGq() += yhat(i) * (yhat(i).transpose() * scale * bq * 0.5) - ystar(i) * (ystar(i).transpose() * scale * bq * 0.5);
            }

            void AccumulateMeanFieldGradient(const TSystemVectorCRef& yhat, const TSystemVectorCRef& ystar, const TSystemVectorCRef& invDiag, TValue normC) const
            {
                AccumulateMeanFieldGradient(yhat, ystar, invDiag, normC, u);
            }

            void AccumulateMeanFieldGradient(const TSystemVectorCRef& yhat, const TSystemVectorCRef& ystar, const TSystemVectorCRef& invDiag,
                                             TValue normC, TWeights* w) const
            {
                const auto scale = 1.0/normC;

                w->GetGl() += (yhat(i) - ystar(i)) * (scale * bq * b.transpose());
                w->GetGq() += (TMatrix(yhat(i) * yhat(i).transpose()) + TMatrix(invDiag(i).asDiagonal()) - ystar(i) * ystar(i).transpose()) * (scale * bq * 0.5);
            }

            int PosX() const
            {
                return i.x;
            }

            int PosY() const
            {
                return i.y;
            }
        };

        class ConditionedFactor : public Compute::ConditionedFactor<TValue, VarDim>
        {
        private:
            TWeights* const          u;
            Eigen::Map<TBasisVector> b;
            TValue                   bq;
            const TVector            y_j;

        public:
            ConditionedFactor(TWeights* u_, TValue* b_, TValue bq_, const TVector& y_j_) : u(u_), b(b_), bq(bq_), y_j(y_j_) {}

            // See the technical report for details on how the gradient arises from the Pseudolikelihood objective.
            void AccumulateGradient(const TVector& mu_j, const TMatrix& Sigma_j, size_t numSubgraphs) const
            {
                u->GetGl() += bq * ((mu_j - y_j) / static_cast<TValue>(numSubgraphs)) * b.transpose();
                u->GetGq() += 0.5 * bq * (Sigma_j - y_j * y_j.transpose()) / static_cast<TValue>(numSubgraphs);
            }

            TVector LinearCoefficients() const
            {
                return bq * (u->Wl * b);
            }

            TMatrix QuadraticCoefficients() const
            {
                return bq * (u->Wq);
            }
        };

        // See the ConnectedFactor interface in Compute.h for a detailed description of each of the methods
        // that are implemented here.
        class ConnectedFactor : public Compute::ConnectedFactor<TValue, VarDim>
        {
        private:
            TWeights* const          u;
            Eigen::Map<TBasisVector> b;
            TValue                   bq;
            const Vector2D<int>&     j;

        public:
            ConnectedFactor(TWeights* u_, TValue* b_, TValue bq_, const Vector2D<int>& j_) : u(u_), b(b_), bq(bq_), j(j_)
            {
            }

            void AddInSiteLinearCoefficients(const TSystemVectorRef& rhs) const
            {
                rhs(j) += bq * (u->Wl * b);
            }

            void AddInSiteQuadraticCoefficientsMultipliedBy(const TSystemVectorRef& Ax, const TSystemVectorCRef& x) const
            {
                Ax(j) += bq * (u->Wq * x(j));
            }

            void AddPrecisionBlocks(TSystemMatrixRow& row) const
            {
                row.Add(j, bq * u->Wq);
            }

            TMatrix GetPrecisionBlock() const
            {
                return bq * u->Wq;
            }

            TVector GetLinearCoefficients() const
            {
                return bq * (u->Wl * b);
            }

            void AddInDiagonal(const TSystemVectorRef& diag) const
            {
                diag(j) += (bq * u->Wq).diagonal();
            }

            void AddInDiagonal(const TBlockDiagonalRef& diag) const
            {
                diag(j) += (bq * u->Wq);
            }
        };

        // See the FactorTypeBase class in Compute.h for details on the parameters.
        FactorType(const typename Base::TModelTreeRef& tree_,
                   const VecCRef<Vector2D<int>>& offsets_,
                   TValue smallestEigenvalue_       = 1e-2,
                   TValue largestEigenvalue_        = 1e2,
                   TValue linearRegularizationC_    = TPrior::DefaultLinearConstant(),
                   TValue quadraticRegularizationC_ = TPrior::DefaultQuadraticConstant(),
                   int quadraticBasisIndex_         = -1)
            : Base::FactorTypeBase(tree_, offsets_, smallestEigenvalue_, largestEigenvalue_,
                                   linearRegularizationC_, quadraticRegularizationC_), quadraticBasisIndex(quadraticBasisIndex_)
        {
        }

        // Computes a 'weights image' for this factor type. Each entry of the resulting image
        // is a pointer to the 'active' weights node of the underlying regression tree. This can
        // be used to avoid re-sorting points into the leaves of the tree over and over again.
        ImageRef<TWeights*> WeightsImage(const typename TFeature::PreProcessType& prep, const int width, const int height) const
        {
            ImageRef<TWeights*> ret(width, height);

            #pragma omp parallel for
            for(int y = 0; y < height; ++y)
                for(int x = 0; x < width; ++x)
                    ret(x, y) = &Base::tree.goto_leaf(x, y, prep, Base::offsets)->data;

            return ret;
        }

        ImageRef<TValue, BasisDim> BasisImage(const typename TFeature::PreProcessType& prep, const int width, const int height) const
        {
            ImageRef<TValue, BasisDim> ret(width, height);

            #pragma omp parallel for
            for(int y = 0; y < height; ++y)
                for(int x = 0; x < width; ++x)
                    TBasis::Compute(x, y, prep, Base::offsets, ret.Ptr(x, y));
            return ret;
        }

        template<typename TOp>
        void ForEachConnectedInstance(const Vector2D<int>& j, const typename TFeature::PreProcessType& prep, const ImageRef<TWeights*>& w, const ImageRef<TValue, BasisDim>& b,
                                      const TOp& op) const
        {
            op(ConnectedFactor(w(j.x, j.y), b.Ptr(j.x, j.y), TBasis::ComputeQuadratic(prep, j, quadraticBasisIndex), j));
        }

        template<typename TOp>
        void ForEachConnectedInstance(const Vector2D<int>& j, const typename TFeature::PreProcessType& prep,
                                      const TOp& op) const
        {
            TValue b[BasisDim];
            op(ConnectedFactor(&Base::tree.goto_leaf(j.x, j.y, prep, Base::offsets)->data,
                               TBasis::Compute(j.x, j.y, prep, Base::offsets, b), TBasis::ComputeQuadratic(prep, j, quadraticBasisIndex), j));
        }

        void ForEachConditionedInstance(const Vector2D<int>& j, const typename TFeature::PreProcessType& prep, const ImageRef<TWeights*>& w, const ImageRef<TValue, BasisDim>& b, const ImageRefC<TLabel>& y,
                                        const std::function<void (const Compute::ConditionedFactor<TValue, VarDim>&)>& op) const
        {
            op(ConditionedFactor(w(j.x, j.y), b.Ptr(j.x, j.y), TBasis::ComputeQuadratic(prep, j, quadraticBasisIndex), Utility::LabelToVector(y(j.x, j.y))));
        }

        void ForEachConditionedInstance(const Vector2D<int>& j, const typename TFeature::PreProcessType& prep, const ImageRefC<TLabel>& y,
                                        const std::function<void (const Compute::ConditionedFactor<TValue, VarDim>&)>& op) const
        {
            ForEachConditionedInstance(j, prep, y, &Base::tree.goto_leaf(j.x, j.y, prep, Base::offsets)->data, op);
        }

        void ForEachConditionedInstance(const Vector2D<int>& j, const typename TFeature::PreProcessType& prep, const ImageRefC<TLabel>& y, TWeights* w,
                                        const std::function<void (const Compute::ConditionedFactor<TValue, VarDim>&)>& op) const
        {
            TValue b[BasisDim];
            op(ConditionedFactor(w, TBasis::Compute(j.x, j.y, prep, Base::offsets, b), TBasis::ComputeQuadratic(prep, j, quadraticBasisIndex), Utility::LabelToVector(y(j.x, j.y))));
        }

        template<typename TOp>
        void ForEachInstance(const typename TFeature::PreProcessType& prep, const ImageRefC<TWeights*>& w, const ImageRef<TValue, BasisDim>& b, TOp op) const
        {
            const auto width = w.Width(), height = w.Height();

            #pragma omp parallel for
            for(int y = 0; y < height; ++y)
            {
                for(int x = 0; x < width; ++x)
                {
                    const Vector2D<int> i(x, y);
                    Factor factor(w(x, y), b.Ptr(x, y), TBasis::ComputeQuadratic(prep, i, quadraticBasisIndex), i);
                    op(factor);
                }
            }
        }

        void ForEachInstance(const typename TFeature::PreProcessType& prep, const ImageRefC<TLabel>& ground,
                             const std::function<void (Factor&)>& op) const
        {
            const auto width = ground.Width(), height = ground.Height();

            #pragma omp parallel for
            for(int y = 0; y < height; ++y)
            {
                for(int x = 0; x < width; ++x)
                {
                    const Vector2D<int> i(x, y);
                    TValue b[BasisDim];
                    Factor factor(&Base::tree.goto_leaf(x, y, prep, Base::offsets)->data,
                                  TBasis::Compute(i.x, i.y, prep, Base::offsets, b), TBasis::ComputeQuadratic(prep, i, quadraticBasisIndex), i);

                    op(factor);
                }
            }
        }

        template<typename TOp>
        void ForEachInstance(const typename TFeature::PreProcessType& prep, const ImageRefC<TLabel>& ground,
                             const VecCRef<Vector2D<int>>& subsample, TOp op) const
        {
            const auto width = ground.Width(), height = ground.Height();
            const int size   = static_cast<int>(subsample.size());

            #pragma omp parallel for
            for(int s = 0; s < size; ++s)
            {
                const Vector2D<int> i(subsample[s].x, subsample[s].y);
                TValue b[BasisDim];
                Factor factor(&Base::tree.goto_leaf(i.x, i.y, prep, Base::offsets)->data,
                              TBasis::Compute(i.x, i.y, prep, Base::offsets, b), TBasis::ComputeQuadratic(prep, i, quadraticBasisIndex), i);
                op(factor);
            }
        }

        size_t NumFactors(const int width, const int height) const
        {
            return width * height;
        }

        size_t NumFactors(const int width, const int height, const VecCRef<Vector2D<int>>& subsample) const
        {
            return subsample.size();
        }

        size_t NumConnected(const Vector2D<int>& j, const int width, const int height) const
        {
            return 1;
        }

        void SetQuadraticBasisIndex(int idx)
        {
            quadraticBasisIndex = idx;
        }

        int GetQuadraticBasisIndex() const
        {
            return quadraticBasisIndex;
        }
    };
}

#endif // H_RTF_UNARY_H
