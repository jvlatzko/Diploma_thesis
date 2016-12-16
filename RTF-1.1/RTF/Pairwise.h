/* This file is part of the "Regression Tree Fields" (RTF) source code distribution,
 * obtained from http://research.microsoft.com/downloads.
 * It is provided to you under the terms of the Microsoft Research License Agreement
 * (MSR-LA). Please see License.txt for details.
 *
 *
 * File: Pairwise.h
 * Implements all functionality related to pairwise factors in a regression tree field.
 *
 */

#ifndef H_RTF_PAIRWISE_H
#define H_RTF_PAIRWISE_H

#include <vector>
#include <cstring>
#include <cassert>
#include <utility>

#include "Types.h"
#include "Priors.h"
#include "Training.h"

namespace Pairwise
{
    // The pairwise factor type class. Augments the FactorTypeBase class by functionality
    // that is specific to *pairwise* factors.
    template<typename TFeature, typename TLabel, typename TPrior, typename TBasis>
    class FactorType : public Compute::FactorTypeBase<TFeature, TLabel, TPrior, TBasis>
    {
    private:
        int quadraticBasisIndex;

    public:
        typedef Compute::FactorTypeBase<TFeature, TLabel, TPrior, TBasis> Base;
        static const size_t                                   VarDim   = TLabel::Size;
        static const size_t                                   BasisDim = TBasis::Size;
        typedef typename TLabel::ValueType                    TValue;
        typedef Eigen::Matrix < TValue, VarDim / 2, 1 >            TCondVector;
        typedef Eigen::Matrix < TValue, VarDim / 2, VarDim / 2 >     TCondMatrix;
        typedef Eigen::Matrix<TValue, BasisDim, 1>            TBasisVector;
        typedef Eigen::Matrix<TValue, Eigen::Dynamic, 1>      TSolution;
        typedef Compute::Weights<TValue, VarDim, BasisDim>    TWeights;
        typedef Compute::SystemVectorRef < TValue, VarDim / 2 >    TSystemVectorRef;
        typedef Compute::BlockDiagonalRef < TValue, VarDim / 2 >   TBlockDiagonalRef;
        typedef Compute::SystemVectorCRef < TValue, VarDim / 2 >   TSystemVectorCRef;
        typedef Compute::SystemMatrixRow < TValue, VarDim / 2 >    TSystemMatrixRow;

        class Factor
        {
        private:
            TWeights* const          p;
            Eigen::Map<TBasisVector> b;
            TValue                   bq;
            const Vector2D<int>&     i;
            const Vector2D<int>&     j;

        public:
            Factor(TWeights* p_, TValue* b_, TValue bq_, const Vector2D<int>& i_, const Vector2D<int>& j_)
                : p(p_), b(b_), bq(bq_), i(i_), j(j_)
            {
            }

            void AccumulateGradient(const TSystemVectorCRef& muPrediction, const TSystemVectorCRef& muLossGradient, TValue normC) const
            {
                AccumulateGradient(muPrediction, muLossGradient, normC, p);
            }

            void AccumulateGradient(const TSystemVectorCRef& yRef, const std::vector<TSolution>& muPrediction, const std::vector<TSolution>& muLossGradient, TValue normC) const
            {
                AccumulateGradient(yRef, muPrediction, muLossGradient, normC, p);
            }

            void AccumulateGradient(const TSystemVectorCRef& muPrediction, const TSystemVectorCRef& muLossGradient,
                                    TValue normC, TWeights* w) const
            {
                const auto scale = 1.0/normC;

                w->GetGl().template topRows < VarDim / 2 > ()                       += muLossGradient(i) * (b.transpose() * scale);
                w->GetGl().template bottomRows < VarDim / 2 > ()                    += muLossGradient(j) * (b.transpose() * scale);
                w->GetGq().template topLeftCorner < VarDim / 2, VarDim / 2 > ()     += muLossGradient(i) * (muPrediction(i).transpose() * scale * bq);
                w->GetGq().template topRightCorner < VarDim / 2, VarDim / 2 > ()    += muLossGradient(i) * (muPrediction(j).transpose() * scale * bq);
                w->GetGq().template bottomLeftCorner < VarDim / 2, VarDim / 2 > ()  += muLossGradient(j) * (muPrediction(i).transpose() * scale * bq);
                w->GetGq().template bottomRightCorner < VarDim / 2, VarDim / 2 > () += muLossGradient(j) * (muPrediction(j).transpose() * scale * bq);
            }

            void AccumulateGradient(const TSystemVectorCRef& yRef, const std::vector<TSolution>& muPrediction, const std::vector<TSolution>& muLossGradient,
                                    TValue normC, TWeights* w) const
            {
                const auto N = muPrediction.size()-1;
                const auto cx = yRef.Width(), cy = yRef.Height();
                const auto scale = 1.0/normC;

                for( int k = N-1; k >=0; --k )
                {
                    const TSystemVectorCRef muPrevPredictionRef(cx, cy, muPrediction[k]);
                    const TSystemVectorCRef muPredictionRef(cx, cy, muPrediction[k+1]);
                    const TSystemVectorCRef muLossGradientRef(cx, cy, muLossGradient[k+1]);

                    w->GetGl().template topRows < VarDim / 2 > ()               += scale * (muLossGradientRef(i) * b.transpose());
                    w->GetGl().template bottomRows < VarDim / 2 > ()            += scale * (muLossGradientRef(j) * b.transpose());

                    w->GetGq().template topLeftCorner<VarDim/2, VarDim/2>()     += (scale*bq) * (muLossGradientRef(i) * muPredictionRef(i).transpose());
                    w->GetGq().template topRightCorner<VarDim/2, VarDim/2>()    += (scale*bq) * (muLossGradientRef(i) * muPrevPredictionRef(j).transpose());
                    w->GetGq().template bottomLeftCorner<VarDim/2, VarDim/2>()  += (scale*bq) * (muLossGradientRef(j) * muPrevPredictionRef(i).transpose());
                    w->GetGq().template bottomRightCorner<VarDim/2, VarDim/2>() += (scale*bq) * (muLossGradientRef(j) * muPredictionRef(j).transpose());
                }
            }

            void AccumulateEnergyBasedGradient(const TSystemVectorCRef& yhat, const TSystemVectorCRef& ystar, TValue normC) const
            {
                AccumulateEnergyBasedGradient(yhat, ystar, normC, p);
            }

            void AccumulateEnergyBasedGradient(const TSystemVectorCRef& yhat, const TSystemVectorCRef& ystar, TValue normC, TWeights* w) const
            {
                const auto scale = 1.0/normC;

                w->GetGl().template topRows < VarDim / 2 > ()                       += (yhat(i) - ystar(i)) * (b.transpose() * scale);
                w->GetGl().template bottomRows < VarDim / 2 > ()                    += (yhat(j) - ystar(j)) * (b.transpose() * scale);
                w->GetGq().template topLeftCorner < VarDim / 2, VarDim / 2 > ()     += yhat(i) * (yhat(i).transpose() * scale * bq * 0.5) - ystar(i) * (ystar(i).transpose() * scale * bq * 0.5);
                w->GetGq().template topRightCorner < VarDim / 2, VarDim / 2 > ()    += yhat(i) * (yhat(j).transpose() * scale * bq * 0.5) - ystar(i) * (ystar(j).transpose() * scale * bq * 0.5);
                w->GetGq().template bottomLeftCorner < VarDim / 2, VarDim / 2 > ()  += yhat(j) * (yhat(i).transpose() * scale * bq * 0.5) - ystar(j) * (ystar(i).transpose() * scale * bq * 0.5);
                w->GetGq().template bottomRightCorner < VarDim / 2, VarDim / 2 > () += yhat(j) * (yhat(j).transpose() * scale * bq * 0.5) - ystar(j) * (ystar(j).transpose() * scale * bq * 0.5);
            }

            void AccumulateMeanFieldGradient(const TSystemVectorCRef& yhat, const TSystemVectorCRef& ystar, const TSystemVectorCRef& invDiag, TValue normC) const
            {
                AccumulateMeanFieldGradient(yhat, ystar, invDiag, normC, p);
            }

            void AccumulateMeanFieldGradient(const TSystemVectorCRef& yhat, const TSystemVectorCRef& ystar, const TSystemVectorCRef& invDiag, TValue normC, TWeights* w) const
            {
                const auto scale = 1.0/normC;

                w->GetGl().template topRows < VarDim / 2 > ()                       += (yhat(i) - ystar(i)) * (b.transpose() * scale);
                w->GetGl().template bottomRows < VarDim / 2 > ()                    += (yhat(j) - ystar(j)) * (b.transpose() * scale);

                w->GetGq().template topLeftCorner < VarDim / 2, VarDim / 2 > ()     += (TCondMatrix(yhat(i) * yhat(i).transpose()) + TCondMatrix(invDiag(i).asDiagonal()) - ystar(i) * ystar(i).transpose()) * (scale * bq * 0.5);
                w->GetGq().template topRightCorner < VarDim / 2, VarDim / 2 > ()    += (yhat(i) * yhat(j).transpose() - ystar(i) * ystar(j).transpose()) * (scale * bq * 0.5);
                w->GetGq().template bottomLeftCorner < VarDim / 2, VarDim / 2 > ()  += (yhat(j) * yhat(i).transpose() - ystar(j) * ystar(i).transpose()) * (scale * bq * 0.5);
                w->GetGq().template bottomRightCorner < VarDim / 2, VarDim / 2 > () += (TCondMatrix(yhat(j) * yhat(j).transpose()) + TCondMatrix(invDiag(j).asDiagonal()) - ystar(j) * ystar(j).transpose()) * (scale * bq * 0.5);
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

        // The "center" variable of our conditioned subgraph, j, is involved in two instantiations
        // (p1, p2) of each pairwise factor of type P, see below:
        //
        //  clamped                 center                clamped
        //    ___        ____        ___        ____        ___
        //   / i \ ____ | p1 | ____ / j \ ____ | p2 | ____ / k \
        //   \___/      |____|      \___/      |____|      \___/
        //
        // For p1, we need to condition on its first variable, i. On the other hand,
        // p2 is conditioned on its second variable, k. This reflects the parameter tying in
        // our model. Hence, the weights that are added in to the canonical parameters of
        // the conditioned subgraph, as well as the gradient with respect to the weights,
        // depend on whether the first or the second variable of the factor is clamped.
        class FactorConditionedOnFirst : public Compute::ConditionedFactor < TValue, VarDim / 2 >
        {
        private:
            TWeights* const          p;
            Eigen::Map<TBasisVector> b;
            TValue                   bq;
            const TCondVector        y_i;
            const TCondVector        y_j;

        public:
            FactorConditionedOnFirst(TWeights* p_, TValue* b_, TValue bq_, const TCondVector& y_i_, const TCondVector& y_j_)
                : p(p_), b(b_), bq(bq_), y_i(y_i_), y_j(y_j_) {}

            // See the technical report for details on how the gradient arises from the Pseudolikelihood objective.
            void AccumulateGradient(const TCondVector& mu_j, const TCondMatrix& Sigma_j, size_t numSubgraphs) const
            {
                p->Locked([&]()
                {
                    const auto scale = 1.0/static_cast<TValue>(numSubgraphs);
                    //          p->GetGl().template topRows<VarDim/2>()                 += (scale * (y_i  - y_i)) * b.transpose();
                    p->GetGl().template bottomRows < VarDim / 2 > ()                    += (scale * (mu_j - y_j)) * b.transpose();
                    //          p->GetGq().template topLeftCorner<VarDim/2,VarDim/2>()  += scale * 0.5 * (y_i  * y_i.transpose()  - y_i * y_i.transpose());
                    p->GetGq().template topRightCorner < VarDim / 2, VarDim / 2 > ()    += (scale * 0.5 * bq) * (y_i  * mu_j.transpose() - y_i * y_j.transpose());
                    p->GetGq().template bottomLeftCorner < VarDim / 2, VarDim / 2 > ()  += (scale * 0.5 * bq) * (mu_j * y_i.transpose()  - y_j * y_i.transpose());
                    p->GetGq().template bottomRightCorner < VarDim / 2, VarDim / 2 > () += (scale * 0.5 * bq) * (Sigma_j                 - y_j * y_j.transpose());
                });
            }

            TCondVector LinearCoefficients() const
            {
                return (p->Wl * b).template tail < VarDim / 2 > ()
                       + (0.5 * bq) * (p->Wq.template topRightCorner < VarDim / 2, VarDim / 2 > ().transpose() + p->Wq.template bottomLeftCorner < VarDim / 2, VarDim / 2 > ()) * y_i;
            }

            // Compute the contribution to the canonical parameters \Theta_j occurring in the product <<y_j*y_j^T, \Theta_j>>
            TCondMatrix QuadraticCoefficients() const
            {
                return bq * p->Wq.template bottomRightCorner < VarDim / 2, VarDim / 2 > ();
            }
        };

        class FactorConditionedOnSecond : public Compute::ConditionedFactor < TValue, VarDim / 2 >
        {
        private:
            TWeights* const          p;
            Eigen::Map<TBasisVector> b;
            TValue                   bq;
            const TCondVector        y_j;
            const TCondVector        y_k;

        public:
            FactorConditionedOnSecond(TWeights* p_, TValue* b_, TValue bq_, const TCondVector& y_j_, const TCondVector& y_k_)
                : p(p_), b(b_), bq(bq_), y_j(y_j_), y_k(y_k_) {}

            void AccumulateGradient(const TCondVector& mu_j, const TCondMatrix& Sigma_j, size_t numSubgraphs) const
            {
                p->Locked([&]()
                {
                    const auto scale = 1.0 / static_cast<TValue>(numSubgraphs);
                    p->GetGl().template topRows < VarDim / 2 > ()                           += (scale * (mu_j - y_j)) * b.transpose();
                    //          p->GetGl().template bottomRows<VarDim/2>()                  += (scale * (y_k  - y_k)) * b.transpose();
                    p->GetGq().template topLeftCorner < VarDim / 2, VarDim / 2 > ()         += (scale * bq * 0.5) * (Sigma_j                 - y_j * y_j.transpose());
                    p->GetGq().template topRightCorner < VarDim / 2, VarDim / 2 > ()        += (scale * bq * 0.5) * (mu_j * y_k.transpose()  - y_j * y_k.transpose());
                    p->GetGq().template bottomLeftCorner < VarDim / 2, VarDim / 2 > ()      += (scale * bq * 0.5) * (y_k  * mu_j.transpose() - y_k * y_j.transpose());
                    //          p->GetGq().template bottomRightCorner<VarDim/2,VarDim/2>()  += (scale * bq * 0.5) * (y_k  * y_k.transpose()  - y_k * y_k.transpose());
                });
            }

            // Compute the contribution to the canonical parameters \theta_j occurring in the dot product y_j^T \theta_j
            TCondVector LinearCoefficients() const
            {
                return (p->Wl * b).template head < VarDim / 2 > ()
                       + (0.5 * bq) * (p->Wq.template topRightCorner < VarDim / 2, VarDim / 2 > () + p->Wq.template bottomLeftCorner < VarDim / 2, VarDim / 2 > ().transpose()) * y_k;
            }

            // Compute the contribution to the canonical parameters \Theta_j occurring in the product <<y_j*y_j^T, \Theta_j>>
            TCondMatrix QuadraticCoefficients() const
            {
                return bq * p->Wq.template topLeftCorner < VarDim / 2, VarDim / 2 > ();
            }
        };

        // See the ConnectedFactor interface for a detailed description of each of the methods that are
        // implemented here.
        class FactorConnectedViaFirst : public Compute::ConnectedFactor < TValue, VarDim / 2 >
        {
        private:
            TWeights* const          p;
            Eigen::Map<TBasisVector> b;
            TValue                   bq;
            const Vector2D<int>&     j;
            const Vector2D<int>&     k;

        public:
            FactorConnectedViaFirst(TWeights* p_, TValue* b_, TValue bq_, const Vector2D<int>& j_, const Vector2D<int>& k_)
                : p(p_), b(b_), bq(bq_), j(j_), k(k_)
            {
            }

            void AddInSiteLinearCoefficients(const TSystemVectorRef& rhs) const
            {
                rhs(j) += (p->Wl * b).template head < VarDim / 2 > ();
            }

            void AddInSiteQuadraticCoefficientsMultipliedBy(const TSystemVectorRef& Ax, const TSystemVectorCRef& x) const
            {
                Ax(j) += p->Wq.template topLeftCorner < VarDim / 2, VarDim / 2 > () * (bq * x(j)) + p->Wq.template topRightCorner < VarDim / 2, VarDim / 2 > () * (bq * x(k));
            }

            void AddPrecisionBlocks(TSystemMatrixRow& row) const
            {
                row.Add(j, bq * p->Wq.template topLeftCorner < VarDim / 2, VarDim / 2 > ());
                row.Add(k, bq * p->Wq.template topRightCorner < VarDim / 2, VarDim / 2 > ());
            }

            void AddInDiagonal(const TSystemVectorRef& diag) const
            {
                diag(j) += (bq * p->Wq.template topLeftCorner < VarDim / 2, VarDim / 2 > ()).diagonal();
            }

            void AddInDiagonal(const TBlockDiagonalRef& diag) const
            {
                diag(j) += (bq * p->Wq.template topLeftCorner < VarDim / 2, VarDim / 2 > ());
            }
        };

        class FactorConnectedViaSecond : public Compute::ConnectedFactor < TValue, VarDim / 2 >
        {
        private:
            TWeights* const          p;
            Eigen::Map<TBasisVector> b;
            TValue                   bq;
            const Vector2D<int>&     i;
            const Vector2D<int>&     j;

        public:
            FactorConnectedViaSecond(TWeights* p_, TValue* b_, TValue bq_, const Vector2D<int>& i_, const Vector2D<int>& j_)
                : p(p_), b(b_), bq(bq_), i(i_), j(j_)
            {
            }

            void AddInSiteLinearCoefficients(const TSystemVectorRef& rhs) const
            {
                rhs(j) += (p->Wl * b).template tail < VarDim / 2 > ();
            }

            void AddInSiteQuadraticCoefficientsMultipliedBy(const TSystemVectorRef& Ax, const TSystemVectorCRef& x) const
            {
                Ax(j) += p->Wq.template bottomRightCorner < VarDim / 2, VarDim / 2 > () * (bq *  x(j)) + p->Wq.template bottomLeftCorner < VarDim / 2, VarDim / 2 > () * (bq * x(i));
            }

            void AddPrecisionBlocks(TSystemMatrixRow& row) const
            {
                row.Add(j, bq * p->Wq.template bottomRightCorner < VarDim / 2, VarDim / 2 > ());
                row.Add(i, bq * p->Wq.template bottomLeftCorner < VarDim / 2, VarDim / 2 > ());
            }

            void AddInDiagonal(const TSystemVectorRef& diag) const
            {
                diag(j) += (bq * p->Wq.template bottomRightCorner < VarDim / 2, VarDim / 2 > ()).diagonal();
            }

            void AddInDiagonal(const TBlockDiagonalRef& diag) const
            {
                diag(j) += (bq * p->Wq.template bottomRightCorner < VarDim / 2, VarDim / 2 > ());
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
            const auto processRect = Utility::ComputeProcessRect(Utility::ComputeDeflateRect(Base::offsets), width, height);
            ImageRef<TWeights*> ret(width, height);

            for(int y = 0; y < height; ++y)   // initialize all entries to zero to detect possible access outside of process rect
                for(int x = 0; x < width; ++x)
                    ret(x, y) = NULL;

            #pragma omp parallel for
            for(int y = processRect.top; y < processRect.bottom; ++y)
                for(int x = processRect.left; x < processRect.right; ++x)
                    ret(x, y) = &Base::tree.goto_leaf(x, y, prep, Base::offsets)->data;

            return ret;
        }

        ImageRef<TValue, BasisDim> BasisImage(const typename TFeature::PreProcessType& prep, const int width, const int height) const
        {
            const auto processRect = Utility::ComputeProcessRect(Utility::ComputeDeflateRect(Base::offsets), width, height);
            ImageRef<TValue, BasisDim> ret(width, height);

            #pragma omp parallel for
            for(int y = processRect.top; y < processRect.bottom; ++y)
                for(int x = processRect.left; x < processRect.right; ++x)
                    TBasis::Compute(x, y, prep, Base::offsets, ret.Ptr(x, y));

            return ret;
        }

        void ForEachConnectedInstance(const Vector2D<int>& j, const typename TFeature::PreProcessType& prep, const ImageRefC<TWeights*>& w, const ImageRef<TValue, BasisDim>& b,
                                      const std::function < void (const Compute::ConnectedFactor < TValue, VarDim / 2 > &) > & op) const
        {
            const int width = w.Width(), height = w.Height();
            const auto i = Vector2D<int>(j.x - Base::offsets[1].x, j.y - Base::offsets[1].y);
            const auto k = Vector2D<int>(j.x + Base::offsets[1].x, j.y + Base::offsets[1].y);

            if(i.x >= 0 && i.x < width && i.y >= 0 && i.y < height)
                op(FactorConnectedViaSecond(w(i.x, i.y), b.Ptr(i.x, i.y), TBasis::ComputeQuadratic(prep, i, j, quadraticBasisIndex), i, j));

            if(k.x >= 0 && k.x < width && k.y >= 0 && k.y < height)
                op(FactorConnectedViaFirst(w(j.x, j.y), b.Ptr(j.x, j.y), TBasis::ComputeQuadratic(prep, j, k, quadraticBasisIndex), j, k));
        }

        void ForEachConnectedInstance(const Vector2D<int>& j, const typename TFeature::PreProcessType& prep, const int width, const int height,
                                      const std::function < void (const Compute::ConnectedFactor < TValue, VarDim / 2 > &) > & op) const
        {
            TValue b[BasisDim];
            const auto i = Vector2D<int>(j.x - Base::offsets[1].x, j.y - Base::offsets[1].y);
            const auto k = Vector2D<int>(j.x + Base::offsets[1].x, j.y + Base::offsets[1].y);

            if(i.x >= 0 && i.x < width && i.y >= 0 && i.y < height)
                op(FactorConnectedViaSecond(&Base::tree.goto_leaf(i.x, i.y, prep, Base::offsets)->data,
                                            TBasis::Compute(i.x, i.y, prep, Base::offsets, b), TBasis::ComputeQuadratic(prep, i, j, quadraticBasisIndex), i, j));

            if(k.x >= 0 && k.x < width && k.y >= 0 && k.y < height)
                op(FactorConnectedViaFirst(&Base::tree.goto_leaf(j.x, j.y, prep, Base::offsets)->data,
                                           TBasis::Compute(j.x, j.y, prep, Base::offsets, b), TBasis::ComputeQuadratic(prep, j, k, quadraticBasisIndex), j, k));
        }

        template <typename TUnaryLabel>
        void ForEachConditionedInstance(const Vector2D<int>& j, const typename TFeature::PreProcessType& prep, const ImageRefC<TWeights*>& w, const ImageRefC<TValue, BasisDim>& b, const ImageRefC<TUnaryLabel>& y,
                                        const std::function < void (const Compute::ConditionedFactor < TValue, VarDim / 2 > &) > & op) const
        {
            const int width = w.Width(), height = w.Height();
            const auto i = Vector2D<int>(j.x - Base::offsets[1].x, j.y - Base::offsets[1].y);
            const auto k = Vector2D<int>(j.x + Base::offsets[1].x, j.y + Base::offsets[1].y);

            if(i.x >= 0 && i.x < width && i.y >= 0 && i.y < height)
                op(FactorConditionedOnFirst(w(i.x, i.y), b.Ptr(i.x, i.y),
                                            TBasis::ComputeQuadratic(prep, i, j, quadraticBasisIndex),
                                            Utility::LabelToVector(y(i.x, i.y)),
                                            Utility::LabelToVector(y(j.x, j.y))));

            if(k.x >= 0 && k.x < width && k.y >= 0 && k.y < height)
                op(FactorConditionedOnSecond(w(j.x, j.y), b.Ptr(j.x, j.y),
                                             TBasis::ComputeQuadratic(prep, j, k, quadraticBasisIndex),
                                             Utility::LabelToVector(y(j.x, j.y)),
                                             Utility::LabelToVector(y(k.x, k.y))));
        }

        template <typename TUnaryLabel>
        void ForEachConditionedInstance(const Vector2D<int>& j, const typename TFeature::PreProcessType& prep, const ImageRefC<TUnaryLabel>& y,
                                        const std::function < void (const Compute::ConditionedFactor < TValue, VarDim / 2 > &) > & op) const
        {
            TValue b[BasisDim];
            const int width = y.Width(), height = y.Height();
            const auto i = Vector2D<int>(j.x - Base::offsets[1].x, j.y - Base::offsets[1].y);
            const auto k = Vector2D<int>(j.x + Base::offsets[1].x, j.y + Base::offsets[1].y);

            if(i.x >= 0 && i.x < width && i.y >= 0 && i.y < height)
                op(FactorConditionedOnFirst(&Base::tree.goto_leaf(i.x, i.y, prep, Base::offsets)->data,
                                            TBasis::Compute(i.x, i.y, prep, Base::offsets, b),
                                            TBasis::ComputeQuadratic(prep, i, j, quadraticBasisIndex),
                                            Utility::LabelToVector(y(i.x, i.y)),
                                            Utility::LabelToVector(y(j.x, j.y))));

            if(k.x >= 0 && k.x < width && k.y >= 0 && k.y < height)
                op(FactorConditionedOnSecond(&Base::tree.goto_leaf(j.x, j.y, prep, Base::offsets)->data,
                                             TBasis::Compute(j.x, j.y, prep, Base::offsets, b),
                                             TBasis::ComputeQuadratic(prep, j, k, quadraticBasisIndex),
                                             Utility::LabelToVector(y(j.x, j.y)),
                                             Utility::LabelToVector(y(k.x, k.y))));
        }

        template <typename TUnaryLabel>
        void ForEachConditionedInstance(const Vector2D<int>& j, const typename TFeature::PreProcessType& prep, const ImageRefC<TUnaryLabel>& y, TWeights* w,
                                        const std::function < void (const Compute::ConditionedFactor < TValue, VarDim / 2 > &) > & op) const
        {
            TValue b[BasisDim];
            const int width = y.Width(), height = y.Height();
            const auto i = Vector2D<int>(j.x - Base::offsets[1].x, j.y - Base::offsets[1].y);
            const auto k = Vector2D<int>(j.x + Base::offsets[1].x, j.y + Base::offsets[1].y);

            if(i.x >= 0 && i.x < width && i.y >= 0 && i.y < height)
                op(FactorConditionedOnFirst(w,
                                            TBasis::Compute(i.x, i.y, prep, Base::offsets, b),
                                            TBasis::ComputeQuadratic(prep, i, j, quadraticBasisIndex),
                                            Utility::LabelToVector(y(i.x, i.y)),
                                            Utility::LabelToVector(y(j.x, j.y))));

            if(k.x >= 0 && k.x < width && k.y >= 0 && k.y < height)
                op(FactorConditionedOnSecond(w,
                                             TBasis::Compute(j.x, j.y, prep, Base::offsets, b),
                                             TBasis::ComputeQuadratic(prep, j, k, quadraticBasisIndex),
                                             Utility::LabelToVector(y(j.x, j.y)),
                                             Utility::LabelToVector(y(k.x, k.y))));
        }

        template <typename TOp>
        void ForEachInstance(const typename TFeature::PreProcessType& prep, const ImageRefC<TWeights*>& w, const ImageRef<TValue, BasisDim>& b, TOp op) const
        {
            const auto processRect = Utility::ComputeProcessRect(Utility::ComputeDeflateRect(Base::offsets), w.Width(), w.Height());
            const auto offXFirst   = Base::offsets[0].x, offYFirst = Base::offsets[0].y, offXSecond = Base::offsets[1].x, offYSecond = Base::offsets[1].y;

            #pragma omp parallel for
            for(int y = processRect.top; y < processRect.bottom; ++y)
            {
                for(int x = processRect.left; x < processRect.right; ++x)
                {
                    const Vector2D<int> i(x + offXFirst,  y + offYFirst);
                    const Vector2D<int> j(x + offXSecond, y + offYSecond);
                    Factor factor(w(x, y), b.Ptr(x, y), TBasis::ComputeQuadratic(prep, i, j, quadraticBasisIndex), i, j);
                    op(factor);
                }
            }
        }

        template <typename TUnaryLabel>
        void ForEachInstance(const typename TFeature::PreProcessType& prep, const ImageRefC<TUnaryLabel>& ground,
                             const std::function<void (Factor&)>& op) const
        {
            const auto processRect = Utility::ComputeProcessRect(Utility::ComputeDeflateRect(Base::offsets), ground.Width(), ground.Height());
            const auto offXFirst   = Base::offsets[0].x, offYFirst = Base::offsets[0].y, offXSecond = Base::offsets[1].x, offYSecond = Base::offsets[1].y;

            #pragma omp parallel for
            for(int y = processRect.top; y < processRect.bottom; ++y)
            {
                for(int x = processRect.left; x < processRect.right; ++x)
                {
                    const Vector2D<int> i(x + offXFirst,  y + offYFirst);
                    const Vector2D<int> j(x + offXSecond, y + offYSecond);
                    TValue b[BasisDim];
                    Factor factor(&Base::tree.goto_leaf(i.x, i.y, prep, Base::offsets)->data,
                                  TBasis::Compute(i.x, i.y, prep, Base::offsets, b), TBasis::ComputeQuadratic(prep, i, j, quadraticBasisIndex), i, j);
                    op(factor);
                }
            }
        }

        template <typename TUnaryLabel, typename TOp>
        void ForEachInstance(const typename TFeature::PreProcessType& prep, const ImageRefC<TUnaryLabel>& ground,
                             const VecCRef<Vector2D<int>>& subsample, TOp op) const
        {
            const auto processRect = Utility::ComputeProcessRect(Utility::ComputeDeflateRect(Base::offsets), ground.Width(), ground.Height());
            const auto offXFirst   = Base::offsets[0].x, offYFirst = Base::offsets[0].y, offXSecond = Base::offsets[1].x, offYSecond = Base::offsets[1].y;
            const int  size        = static_cast<int>(subsample.size());

            #pragma omp parallel for
            for(int s = 0; s < size; ++s)
            {
                const auto p = subsample[s];

                if(processRect.PtInRect(p))
                {
                    const Vector2D<int> i(p.x + offXFirst,  p.y + offYFirst);
                    const Vector2D<int> j(p.x + offXSecond, p.y + offYSecond);
                    TValue b[BasisDim];
                    Factor factor(&Base::tree.goto_leaf(i.x, i.y, prep, Base::offsets)->data,
                                  TBasis::Compute(i.x, i.y, prep, Base::offsets, b), TBasis::ComputeQuadratic(prep, i, j, quadraticBasisIndex), i, j);
                    op(factor);
                }
            }
        }

        size_t NumFactors(const int width, const int height) const
        {
            size_t ret = 0;
            const auto processRect = Utility::ComputeProcessRect(Utility::ComputeDeflateRect(Base::offsets), width, height);

            for(int y = processRect.top; y < processRect.bottom; ++y)
                for(int x = processRect.left; x < processRect.right; ++x)
                    ++ret;

            return ret;
        }

        size_t NumFactors(const int width, const int height, const VecCRef<Vector2D<int>>& subsample) const
        {
            const auto processRect = Utility::ComputeProcessRect(Utility::ComputeDeflateRect(Base::offsets), width, height);
            const int  size        = static_cast<int>(subsample.size());
            size_t ret             = 0;

            for(int s = 0; s < size; ++s)
            {
                const auto p = subsample[s];

                if(processRect.PtInRect(p))
                    ++ret;
            }

            return ret;
        }

        size_t NumConnected(const Vector2D<int>& j, const int width, const int height) const
        {
            const auto i = Vector2D<int>(j.x - Base::offsets[1].x, j.y - Base::offsets[1].y);
            const auto k = Vector2D<int>(j.x + Base::offsets[1].x, j.y + Base::offsets[1].y);
            size_t n = 0;

            if(i.x >= 0 && i.x < width && i.y >= 0 && i.y < height)
                n++;

            if(k.x >= 0 && k.x < width && k.y >= 0 && k.y < height)
                n++;

            return n;
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

#endif // H_RTF_PAIRWISE_H
