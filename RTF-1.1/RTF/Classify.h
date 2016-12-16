/* This file is part of the "Regression Tree Fields" (RTF) source code distribution,
 * obtained from http://research.microsoft.com/downloads.
 * It is provided to you under the terms of the Microsoft Research License Agreement
 * (MSR-LA). Please see License.txt for details.
 *
 *
 * File: Classify.h
 * Implements routines for labeling of instances based on a specified RTF model.
 *
 */

#ifndef H_RTF_CLASSIFY_H
#define H_RTF_CLASSIFY_H

#include <stdarg.h>
#include <random>
#include <algorithm>

#include <Eigen/Dense>
#include <Eigen/StdVector>

#ifdef USE_GPU
#include "GPU.h"
#endif // USE_GPU
#include "Types.h"
#include "Trees.h"
#include "Image.h"
#include "Array.h"
#include "Utility.h"
#include "Monitor.h"
#include "Compute.h"
#include "Training.h"
#include "Minimization.h"

namespace Classify
{

    template <typename TTraits>
    ImageRefC<typename TTraits::UnaryGroundLabel> Regress(const typename TTraits::UnaryTreeCRef& tree,
            const typename TTraits::PreProcessType& prep, int cx, int cy)
    {
        ImageRef<typename TTraits::UnaryGroundLabel> ground(cx, cy);
        VecRef<Vector2D<int>> offsets(1);

        #pragma omp parallel for
        for(int y = 0; y < cy; ++y)
        {
            for(int x = 0; x < cx; ++x)
            {
                auto data = tree.GetLeafData(x, y, prep, offsets);
                ground(x, y) = data.average;
            }
        }

        return ground;
    }

    // Predict using the means stored at the leaves of a single unary regression tree. This is a simple baseline
    // to compare more involved methods against.
    template <typename TTraits>
    ImageRefC<typename TTraits::UnaryGroundLabel> Regress(const typename TTraits::UnaryTreeCRef& tree,
            const ImageRefC<typename TTraits::InputLabel>& image)
    {
        auto prep = TTraits::Feature::PreProcess(image);
        return Regress<TTraits>(tree, prep, image.Width(), image.Height());
    }

    // Ordinary random forest prediction
    template <typename TTraits>
    ImageRefC<typename TTraits::UnaryGroundLabel> RegressForest(typename TTraits::UnaryTreeRefVector& forest,
            const ImageRefC<typename TTraits::InputLabel>& image)
    {
        auto prep = TTraits::Feature::PreProcess(image);
        const int cx = image.Width(), cy = image.Height();
        ImageRef<typename TTraits::UnaryGroundLabel> pred(cx, cy);
        double pred_factor = 1.0 / static_cast<double>(forest.size());  // uniform average

        for(size_t ti = 0; ti < forest.size(); ++ti)
        {
            // Get prediction of current tree
            auto cur_pred = Classify::Regress<TTraits>(forest[ti], prep, cx, cy);

            // Per-pixel average
            #pragma omp parallel for
            for(int y = 0; y < cy; ++y)
            {
                for(int x = 0; x < cx; ++x)
                {
                    typename TTraits::UnaryGroundLabel pix_pred = cur_pred(x, y);

                    for(size_t c = 0; c < pix_pred.size(); ++c)
                    {
                        pred(x, y)[c] = (ti == 0 ? 0.0 : pred(x, y)[c]) + pred_factor * pix_pred[c];
                    }
                }
            }
        }

        return (pred);
    }

    // Predict using Gibbs sampling. We keep iterating over the conditioned subgraphs and draw samples from the induced
    // distribution that are in turn used to update the state vector.
    template <typename TTraits>
    ImageRefC<typename TTraits::UnaryGroundLabel> RegressViaSampling(const typename TTraits::UnaryFactorTypeVector& Us,
            const typename TTraits::PairwiseFactorTypeVector& Ps,
            const ImageRefC<typename TTraits::InputLabel> image,
            const size_t numSamples = 1000,
            const size_t numBurnIn  = 100,
            bool verbose = false)
    {
        const size_t                             Dim = TTraits::UnaryGroundLabel::Size;
        typedef typename TTraits::ValueType      TValue;
        const int cx    = image.Width(), cy = image.Height();
        ImageRef<typename TTraits::UnaryGroundLabel> state(cx, cy), posterior(cx, cy); // initialized to all zeros
        std::mt19937 engine;
        std::normal_distribution<TValue> normal; // used to draw samples from the conditioned subgraphs
        TTraits::Monitor::Report("Gibbs: Drawing %u samples out of %u x %u grid\n", numSamples, cx, cy);
        const auto prep = TTraits::Feature::PreProcess(image);
        const auto Ws   = Compute::ComputeWeightsImages<TTraits>(prep, cx, cy, Us, Ps);
        const auto Bs   = Compute::ComputeBasisImages<TTraits>(prep, cx, cy, Us, Ps);
        const size_t numTotal = numBurnIn + numSamples;

        for(unsigned sweep = 0; sweep < numTotal; ++sweep)
        {
            if(verbose)
                TTraits::Monitor::Report("Gibbs: Sweep %5u\n", sweep);

            Compute::for_each_precomputed_conditioned_subgraph<TTraits>(prep, state, Us, Ps, Ws.first, Ws.second, Bs.first, Bs.second,
                    [&](const Compute::PrecomputedConditionedSubgraph<TTraits>& G_j)
            {
                state(G_j.PosX(), G_j.PosY()) = G_j.DrawSample(engine, normal);
            });

            if(sweep >= numBurnIn)
                posterior += state;
        }

        TTraits::Monitor::Report("Gibbs: Done.\n");
        posterior *= TValue(1.0 / numSamples);
        return posterior;
    }

    template <typename TTraits, typename TErrorTerm>
    struct LinearSystem;

    namespace Detail
    {
        // Base class describing the linear system induced by a regression tree field model.
        // See the derived classes below for further details.
        template <typename TTraits, typename TErrorTerm>
        class LinearSystemBase : public Minimization::LinearSystem<typename TTraits::ValueType>
        {
        public:
            typedef Minimization::LinearSystem<typename TTraits::ValueType> Base;
            static  const size_t                 Dim = TTraits::UnaryGroundLabel::Size;
            typedef typename TTraits::ValueType TValue;
            typedef Eigen::Matrix<TValue, Eigen::Dynamic, Dim> BlockDiagonalType;

        public:
            const int cx;
            const int cy;

            const typename TTraits::PreProcessType prep;
            const ImageRefC<typename TTraits::UnaryGroundLabel> ground;

            const typename TTraits::UnaryFactorTypeVector& Us;
            const typename TTraits::PairwiseFactorTypeVector& Ps;
            const typename TTraits::LinearOperatorVector& Ls;

            std::pair < typename TTraits::UnaryWeightsImageVector,
                typename TTraits::PairwiseWeightsImageVector > Ws;

            std::pair < typename TTraits::UnaryBasisImageVector,
                typename TTraits::PairwiseBasisImageVector > Bs;


            // Computes the righthand side of the sparse linear system Ax = b; i.e., the b vector.
            void ComputeRightHandSide(typename Base::VectorType& b_) const
            {
                Compute::SystemVectorRef<TValue, Dim> b(cx, cy, b_);
                Compute::for_each_precomputed_subgraph<TTraits, true>(prep, cx, cy, Us, Ps, Ws.first, Ws.second, Bs.first, Bs.second,
                        [&](const Compute::PrecomputedSubgraph<TTraits>& G_j)
                {
                    G_j.AddInSiteLinearCoefficients(b);
                });
                std::for_each(Ls.begin(), Ls.end(), [&](const typename TTraits::LinearOperatorRef& op)
                {
                    op.AddInLinearContribution(prep, b);
                });
                TErrorTerm::AddInLinearContribution(ground, b);
            }

        public:
            LinearSystemBase(const typename TTraits::PreProcessType& prep_,
                             const int cx_, const int cy_,
                             const typename TTraits::UnaryFactorTypeVector& Us_,
                             const typename TTraits::PairwiseFactorTypeVector& Ps_,
                             const typename TTraits::LinearOperatorVector& Ls_,
                             ImageRefC<typename TTraits::UnaryGroundLabel> ground_ = ImageRefC<typename TTraits::UnaryGroundLabel>())
                : prep(prep_), cx(cx_), cy(cy_), Us(Us_), Ps(Ps_), Ls(Ls_),
                  Ws(Compute::ComputeWeightsImages<TTraits>(prep_, cx_, cy_, Us_, Ps_)),
                  Bs(Compute::ComputeBasisImages<TTraits>(prep_, cx_, cy_, Us_, Ps_)),
                  ground(ground_)
            {
            }

            const ImageRefC<typename TTraits::UnaryGroundLabel> GroundTruth() const
            {
                return ground;
            }

            std::pair < typename TTraits::UnaryWeightsImageVector,
                typename TTraits::PairwiseWeightsImageVector > WeightsImages() const
            {
                return Ws;
            }

            std::pair < typename TTraits::UnaryBasisImageVector,
            typename TTraits::PairwiseBasisImageVector > BasisImages() const
            {
                return Bs;
            }

            // Accessors to the precomputed weights images
            typename TTraits::UnaryWeightsImageRef
            UnaryWeightsImage(size_t index)
            {
                return Ws.first[index];
            }

            const typename TTraits::UnaryWeightsImageRef
            UnaryWeightsImage(size_t index) const
            {
                return Ws.first[index];
            }

            typename TTraits::PairwiseWeightsImageRef
            PairwiseWeightsImage(size_t index)
            {
                return Ws.second[index];
            }

            const typename TTraits::PairwiseWeightsImageRef
            PairwiseWeightsImage(size_t index) const
            {
                return Ws.second[index];
            }

            // Accessors to the pre-computed basis images
            typename TTraits::UnaryBasisImageRef
            UnaryBasisImage(size_t index)
            {
                return Bs.first[index];
            }

            const typename TTraits::UnaryBasisImageRef
            UnaryBasisImage(size_t index) const
            {
                return Bs.first[index];
            }

            typename TTraits::PairwiseBasisImageRef
            PairwiseBasisImage(size_t index)
            {
                return Bs.second[index];
            }

            const typename TTraits::PairwiseBasisImageRef
            PairwiseBasisImage(size_t index) const
            {
                return Bs.second[index];
            }

            const typename TTraits::PreProcessType& Prep() const
            {
                return prep;
            }

            // Instantiates the righthand side of the sparse linear system Ax = b; i.e., the b vector.
            virtual void ProvideRightHandSide(typename Base::VectorType& b_) const
            {
                return ComputeRightHandSide(b_);
            }

            void ProvideInverseDiagonal(typename Base::VectorType& invDiag_) const
            {
                Compute::SystemVectorRef<TValue, Dim> invDiag(cx, cy, invDiag_);
                Compute::for_each_precomputed_subgraph<TTraits, true>(prep, cx, cy, Us, Ps, Ws.first, Ws.second, Bs.first, Bs.second,
                        [&](const Compute::PrecomputedSubgraph<TTraits>& G_j)
                {
                    G_j.AddInDiagonal(invDiag);
                });
                invDiag_ *= TValue(-1);
                for( int L = 0; L < Ls.size(); ++L )
                    Ls[L].AddInDiagonal(prep, invDiag);
                invDiag_ = invDiag_.cwiseInverse();
            }

            void ProvideInverseBlockDiagonal(BlockDiagonalType& invDiag_) const
            {
                Compute::BlockDiagonalRef<TValue, Dim> invDiag(cx, cy, invDiag_);
                Compute::for_each_precomputed_subgraph<TTraits, true>(prep, cx, cy, Us, Ps, Ws.first, Ws.second, Bs.first, Bs.second,
                        [&](const Compute::PrecomputedSubgraph<TTraits>& G_j)
                {
                    G_j.AddInDiagonal(invDiag);
                });
                invDiag_ *= TValue(-1);
                for( int L = 0; L < Ls.size(); ++L )
                    Ls[L].AddInDiagonal(prep, invDiag);

                #pragma omp parallel for
                for( int i = 0; i < invDiag.NumPixels(); ++i )
                    invDiag[i] = invDiag[i].inverse();
            }

            // Forms the product of the system matrix with vector 'x' and stores the
            // result in 'Ax'.
            virtual void MultiplySystemMatrixBy(typename Base::VectorType& Ax, const typename Base::VectorType& x) const = 0;

            // The number of rows of the righthand side.
            unsigned Dimensions() const
            {
                return cx * cy * Dim;
            }

            size_t NumPixels() const
            {
                return cx * cy;
            }

            size_t Width() const
            {
                return cx;
            }

            size_t Height() const
            {
                return cy;
            }

            void Report(const char* fmt, ...) const
            {
                va_list args;
                va_start(args, fmt);
                TTraits::Monitor::ReportVA(fmt, args);
                va_end(args);
            }
        };

        // Concrete implementation of an RTF sparse linear system that avoids instantiation
        // of the actual system matrix. Instead, all matrix coefficients are computed on
        // the fly.  This is faster than explicitly instantiating the matrix if a small
        // number of CG iterations are performed.
        // The factor potentials, etc. are also computed on the fly at each iteration,
        // rather than pre-computed.
        template <typename TTraits, typename TErrorTerm>
        class OnTheFlySystem : public Minimization::LinearSystem<typename TTraits::ValueType>
        {
        public:
            static const size_t Dim = TTraits::UnaryGroundLabel::Size;
            typedef typename Minimization::LinearSystem<typename TTraits::ValueType>::VectorType VectorType;
            typedef typename TTraits::ValueType TValue;
            typedef Eigen::Matrix<TValue, Eigen::Dynamic, Dim> BlockDiagonalType;

        private:
            const int cx;
            const int cy;

            const typename TTraits::PreProcessType prep;
            const ImageRefC<typename TTraits::UnaryGroundLabel> ground;

            const typename TTraits::UnaryFactorTypeVector& Us;
            const typename TTraits::PairwiseFactorTypeVector& Ps;
            const typename TTraits::LinearOperatorVector& Ls;

            VectorType b;

            // Computes the righthand side of the sparse linear system Ax = b; i.e., the b vector.
            void ComputeRightHandSide(VectorType& b_) const
            {
                Compute::SystemVectorRef<TValue, Dim> b(cx, cy, b_);
                Compute::for_each_subgraph<TTraits, true>(prep, cx, cy, Us, Ps,
                        [&](const Compute::Subgraph<TTraits>& G_j)
                {
                    G_j.AddInSiteLinearCoefficients(b);
                });
                std::for_each(Ls.begin(), Ls.end(), [&](const typename TTraits::LinearOperatorRef& op)
                {
                    op.AddInLinearContribution(prep, b);
                });
                TErrorTerm::AddInLinearContribution(ground, b);
            }

        public:
            OnTheFlySystem(const ImageRefC<typename TTraits::InputLabel> img_,
                           const typename TTraits::UnaryFactorTypeVector& Us_,
                           const typename TTraits::PairwiseFactorTypeVector& Ps_,
                           const typename TTraits::LinearOperatorVector& Ls_ = TTraits::LinearOperatorVector(),
                           ImageRefC<typename TTraits::UnaryGroundLabel> ground_ = ImageRefC<typename TTraits::UnaryGroundLabel>())
                : cx(img_.Width()), cy(img_.Height()),
                  prep(TTraits::Feature::PreProcess(img_)), ground(ground_),
                  Us(Us_), Ps(Ps_), Ls(Ls_),
                  b(Dimensions())
            {
                ComputeRightHandSide(b); // Pre-compute the righthand side of the linear system
            }

            OnTheFlySystem(const typename TTraits::PreProcessType& prep_, const int cx_, const int cy_,
                           const typename TTraits::UnaryFactorTypeVector& Us_,
                           const typename TTraits::PairwiseFactorTypeVector& Ps_,
                           const typename TTraits::LinearOperatorVector& Ls_ = TTraits::LinearOperatorVector(),
                           ImageRefC<typename TTraits::UnaryGroundLabel> ground_ = ImageRefC<typename TTraits::UnaryGroundLabel>())
                : cx(cx_), cy(cy_),
                  prep(prep_), ground(ground_),
                  Us(Us_), Ps(Ps_), Ls(Ls_),
                  b(Dimensions())
            {
                ComputeRightHandSide(b); // Pre-compute the righthand side of the linear system
            }

            void MultiplySystemMatrixBy(VectorType& Ax_, const VectorType& x_) const
            {
                Compute::SystemVectorRef<TValue, Dim> Ax(cx, cy, Ax_);
                Compute::SystemVectorCRef<TValue, Dim> x(cx, cy, x_);

                if( Us.size() > 0 || Ps.size() > 0 )
                {
                    Compute::for_each_subgraph<TTraits, true>(prep, cx, cy, Us, Ps,
                            [&](const Compute::Subgraph<TTraits>& G_j)
                    {
                        G_j.AddInSiteQuadraticCoefficientsMultipliedBy(Ax, x);
                    });
                    Ax_ *= TValue(-1); // negate
                }

                // Custom linear operators
                for( int L = 0; L < Ls.size(); ++L )
                    Ls[L].AddInImplicitMatrixMultipliedBy(prep, Ax, x);

                // Add in contribution of error term to loss-augmented inference
                TErrorTerm::AddInImplicitMatrixMultipliedBy(ground, Ax, x);
            }

            // Use a different righthand side while re-using the instantiated sparse system matrix
            void SetRightHandSide(const ImageRefC<typename TTraits::UnaryGroundLabel>& rhs)
            {
                b = Utility::SolutionFromLabeling<TTraits>(rhs);
            }

            // Instantiates the righthand side of the sparse linear system Ax = b; i.e., the b vector.
            void ProvideRightHandSide(VectorType& b_) const
            {
                b_ = b; // Use pre-computed value
            }

            void ProvideInverseDiagonal(VectorType& invDiag_) const
            {
                Compute::SystemVectorRef<TValue, Dim> invDiag(cx, cy, invDiag_);
                Compute::for_each_subgraph<TTraits, true>(prep, cx, cy, Us, Ps,
                        [&](const Compute::Subgraph<TTraits>& G_j)
                {
                    G_j.AddInDiagonal(invDiag);
                });
                invDiag_ *= TValue(-1);
                for( int L = 0; L < Ls.size(); ++L )
                    Ls[L].AddInDiagonal(prep, invDiag);
                invDiag_ = invDiag_.cwiseInverse();
            }

            void ProvideInverseBlockDiagonal(BlockDiagonalType& invDiag_) const
            {
                Compute::BlockDiagonalRef<TValue, Dim> invDiag(cx, cy, invDiag_);
                Compute::for_each_subgraph<TTraits, true>(prep, cx, cy, Us, Ps,
                        [&](const Compute::Subgraph<TTraits>& G_j)
                {
                    G_j.AddInDiagonal(invDiag);
                });
                invDiag_ *= TValue(-1);
                for( int L = 0; L < Ls.size(); ++L )
                    Ls[L].AddInDiagonal(prep, invDiag);

                #pragma omp parallel for
                for( int i = 0; i < invDiag.NumPixels(); ++i )
                {
                    invDiag[i] = invDiag[i].inverse();
                }
            }

            const ImageRefC<typename TTraits::UnaryGroundLabel> GroundTruth() const
            {
                return ground;
            }

            // The number of rows of the righthand side.
            unsigned Dimensions() const
            {
                return cx * cy * Dim;
            }

            size_t NumPixels() const
            {
                return cx * cy;
            }

            size_t Width() const
            {
                return cx;
            }

            size_t Height() const
            {
                return cy;
            }

            void Report(const char* fmt, ...) const
            {
                va_list args;
                va_start(args, fmt);
                TTraits::Monitor::ReportVA(fmt, args);
                va_end(args);
            }
        };

        // Concrete implementation of the LinearSystem class that performs the matrix-vector
        // multiplication without actually instantiating the system matrix.
        // The factor potentials are pre-computed upon instantiation.
        template <typename TTraits, typename TErrorTerm, bool instantiate = false>
        class LinearSystem : public LinearSystemBase<TTraits, TErrorTerm>
        {
        public:
            typedef LinearSystemBase<TTraits, TErrorTerm> Base;
            static const size_t                 Dim = TTraits::UnaryGroundLabel::Size;
            typedef typename TTraits::ValueType TValue;

        private:
            typename Base::VectorType b;

        public:
            LinearSystem(const ImageRefC<typename TTraits::InputLabel> img_,
                         const typename TTraits::UnaryFactorTypeVector& Us_,
                         const typename TTraits::PairwiseFactorTypeVector& Ps_,
                         const typename TTraits::LinearOperatorVector& Ls_ = TTraits::LinearOperatorVector(),
                         ImageRefC<typename TTraits::UnaryGroundLabel> ground_ = ImageRefC<typename TTraits::UnaryGroundLabel>())
                : Base::LinearSystemBase(TTraits::Feature::PreProcess(img_), img_.Width(), img_.Height(), Us_, Ps_, Ls_, ground_),
                  b(Base::Dimensions())
            {
                Base::ComputeRightHandSide(b); // Pre-compute the righthand side of the linear system
            }

            LinearSystem(const typename TTraits::PreProcessType& prep_, const int cx, const int cy,
                         const typename TTraits::UnaryFactorTypeVector& Us_,
                         const typename TTraits::PairwiseFactorTypeVector& Ps_,
                         const typename TTraits::LinearOperatorVector& Ls_ = TTraits::LinearOperatorVector(),
                         ImageRefC<typename TTraits::UnaryGroundLabel> ground_ = ImageRefC<typename TTraits::UnaryGroundLabel>())
                : Base::LinearSystemBase(prep_, cx, cy, Us_, Ps_, Ls_, ground_),
                  b(Base::Dimensions())
            {
                Base::ComputeRightHandSide(b); // Pre-compute the righthand side of the linear system
            }

            void MultiplySystemMatrixBy(typename Base::VectorType& Ax_, const typename Base::VectorType& x_) const
            {
                Compute::SystemVectorRef<TValue, Dim> Ax(Base::cx, Base::cy, Ax_);
                Compute::SystemVectorCRef<TValue, Dim> x(Base::cx, Base::cy, x_);

                if( Base::Us.size() > 0 || Base::Ps.size() > 0 )
                {
                    Compute::for_each_precomputed_subgraph<TTraits, true>(Base::prep, Base::cx, Base::cy, Base::Us, Base::Ps, Base::Ws.first, Base::Ws.second, Base::Bs.first, Base::Bs.second,
                            [&](const Compute::PrecomputedSubgraph<TTraits>& G_j)
                    {
                        G_j.AddInSiteQuadraticCoefficientsMultipliedBy(Ax, x);
                    });
                    Ax_ *= TValue(-1); // negate
                }

                for( int L = 0; L < Base::Ls.size(); ++L )
                    Base::Ls[L].AddInImplicitMatrixMultipliedBy(Base::prep, Ax, x);

                // Add in contribution of error term to loss-augmented inference
                TErrorTerm::AddInImplicitMatrixMultipliedBy(Base::ground, Ax, x);
            }

            // Use a different righthand side while re-using the instantiated sparse system matrix
            void SetRightHandSide(const ImageRefC<typename TTraits::UnaryGroundLabel>& rhs)
            {
                b = Utility::SolutionFromLabeling<TTraits>(rhs);
            }

            // Instantiates the righthand side of the sparse linear system Ax = b; i.e., the b vector.
            virtual void ProvideRightHandSide(typename Base::VectorType& b_) const
            {
                b_ = b; // Use pre-computed value
            }
        };

        // Concrete implementation of the LinearSystem class that instantiates the sparse system
        // matrix initially and performs subsequent matrix-vector multiplications using the
        // instantiated sparse representation. This has typically much better data locality then
        // the approach above, so it starts paying off after a rather small number of CG iterations
        // (say, 30). On the downside, it requires more memory than the above approach.
        template <typename TTraits, typename TErrorTerm>
        class LinearSystem<TTraits, TErrorTerm, true> : public LinearSystemBase<TTraits, TErrorTerm>
        {
        public:
            typedef LinearSystemBase<TTraits, TErrorTerm>     Base;
            static const size_t                 Dim = TTraits::UnaryGroundLabel::Size;
            typedef typename TTraits::ValueType TValue;

        private:
            typename Base::VectorType        b;
            Compute::SystemMatrix<TTraits>   A;
            mutable std::vector< typename Base::VectorType > intermediateResults;

        public:
            LinearSystem(const ImageRefC<typename TTraits::InputLabel> img_,
                         const typename TTraits::UnaryFactorTypeVector& Us_,
                         const typename TTraits::PairwiseFactorTypeVector& Ps_,
                         const typename TTraits::LinearOperatorVector& Ls_ = TTraits::LinearOperatorVector(),
                         ImageRefC<typename TTraits::UnaryGroundLabel> ground_ = ImageRefC<typename TTraits::UnaryGroundLabel>())
                : Base::LinearSystemBase(TTraits::Feature::PreProcess(img_), img_.Width(), img_.Height(), Us_, Ps_, Ls_, ground_),
                  b(GetRightHandSide()),
                  A(Base::prep, img_.Width(), img_.Height(), Us_, Ps_, Base::Ws.first, Base::Ws.second, Base::Bs.first, Base::Bs.second, Ls_),
                  intermediateResults(Ls_.size())
            {
            }

            LinearSystem(const typename TTraits::PreProcessType& prep_, const int cx, const int cy,
                         const typename TTraits::UnaryFactorTypeVector& Us_,
                         const typename TTraits::PairwiseFactorTypeVector& Ps_,
                         const typename TTraits::LinearOperatorVector& Ls_ = TTraits::LinearOperatorVector(),
                         ImageRefC<typename TTraits::UnaryGroundLabel> ground_ = ImageRefC<typename TTraits::UnaryGroundLabel>())
                : Base::LinearSystemBase(prep_, cx, cy, Us_, Ps_, Ls_, ground_),
                  b(GetRightHandSide()),
                  A(prep_, cx, cy, Us_, Ps_, Base::Ws.first, Base::Ws.second, Base::Bs.first, Base::Bs.second, Ls_),
                  intermediateResults(Ls_.size())
            {
            }

            typename Base::VectorType GetRightHandSide()
            {
                typename Base::VectorType b_(Base::Dimensions());
                Base::ComputeRightHandSide(b_);
                return b_;
            }

            // Use a different righthand side while re-using the instantiated sparse system matrix
            void SetRightHandSide(const ImageRefC<typename TTraits::UnaryGroundLabel>& rhs)
            {
                b = Utility::SolutionFromLabeling<TTraits>(rhs);
            }

            // Instantiates the righthand side of the sparse linear system Ax = b; i.e., the b vector.
            virtual void ProvideRightHandSide(typename Base::VectorType& b_) const
            {
                b_ = b; // Use pre-computed value
            }

            void MultiplySystemMatrixBy(typename Base::VectorType& Ax_, const typename Base::VectorType& x_) const
            {
                Compute::SystemVectorRef<TValue, Dim> Ax(Base::cx, Base::cy, Ax_);
                Compute::SystemVectorCRef<TValue, Dim> x(Base::cx, Base::cy, x_);
                A.MultiplyBy(Ax, x);

                // Add in contribution of error term to loss-augmented inference
                TErrorTerm::AddInImplicitMatrixMultipliedBy(Base::ground, Ax, x);
            }

            // Obtain a low-level coordinate storage representation of the sparse system
            // matrix (and a dense representation of the righthand side). This can be used
            // to interface with other sparse linear algebra libraries.
            template <typename TExportValue>
            void Export(int& numRows, std::vector<int>& rowIndices,
                        int& numCols, std::vector<int>& colIndices,
                        std::vector<TExportValue>& values,
                        std::vector<TExportValue>& rhs) const
            {
                //const auto t0 = GetTickCount64();
                A.Export(numRows, rowIndices, numCols, colIndices, values);
                typename Base::VectorType b(Base::Dimensions());
                ProvideRightHandSide(b);
                rhs = Utility::vector_cast<TExportValue>(b);
                //std::cerr << "Export takes " << (GetTickCount64()-t0) << " ms." << std::endl;
            }

            template <typename TExportValue>
            void Export(Eigen::SparseMatrix<TExportValue>& Q,
                        Eigen::Matrix<TExportValue, -1, 1>& b) const
            {
                A.Export(Q);
                ProvideRightHandSide(b);
            }

#ifdef USE_GPU
            std::shared_ptr< GPU::ConjugateGradientSolver > GetGPUSolver()
            {
                int numRows, numCols;
                std::vector<int> rowIndices, colIndices;
                std::vector<float> values;
                A.Export<float>(numRows, rowIndices, numCols, colIndices, values);
                return GPU::GetSolver(numRows, rowIndices, numCols, colIndices, values);
            }

            template<typename TExportValue>
            Eigen::Matrix<TExportValue, -1, 1>
            SolveOnGPU(std::shared_ptr< GPU::ConjugateGradientSolver > solver, const size_t maxNumIt, TExportValue residualTol)
            {
                std::vector<float> sol;
                const auto b_ = Utility::vector_cast<float>(b);
                GPU::SolveViaConjugateGradient(solver, b_, sol, maxNumIt, residualTol);
                return Utility::vector_cast<TExportValue>(sol);
            }
#endif // USE_GPU
        };

        template <typename TTraits, typename TErrorTerm, int CachingMode = WEIGHTS_AND_BASIS_PRECOMPUTED>
        struct LinearSystemDispatcher
        {
            typedef LinearSystem<TTraits, TErrorTerm, false> Type;
        };

        template <typename TTraits, typename TErrorTerm>
        struct LinearSystemDispatcher<TTraits, TErrorTerm, WEIGHTS_AND_BASIS_AND_MATRIX_PRECOMPUTED>
        {
            typedef LinearSystem<TTraits, TErrorTerm, true> Type;
        };

        template <typename TTraits, typename TErrorTerm>
        struct LinearSystemDispatcher<TTraits, TErrorTerm, ON_THE_FLY>
        {
            typedef OnTheFlySystem<TTraits, TErrorTerm> Type;
        };

        // Minimize 1/2 x^T A x - x^Tb instead of solving the linear system Ax = b.
        // If A is positive-definite (which it is, in our case) this has the same solution.
        template<typename TTraits, typename TErrorTerm=Loss::NoErrorTerm<TTraits> >
        class UnconstrainedQuadratic : public Minimization::UnconstrainedProblem<typename TTraits::ValueType>
        {
        public:
            typedef typename TTraits::ValueType TValue;
            typedef typename Classify::LinearSystem<TTraits, TErrorTerm>::Type TLinearSystem;
            typedef typename TLinearSystem::VectorType TVector;

        private:
            const TLinearSystem& system;
            TVector b;

        public:
            UnconstrainedQuadratic(const TLinearSystem& system_) : system(system_), b(system_.Dimensions())
            {
                system.ProvideRightHandSide(b);
            }

            // Note: The gradient with respect to x is given by Ax - b.
            TValue Eval(const TVector& x, TVector& g)
            {
                TVector Ax(Dimensions());
                system.MultiplySystemMatrixBy(Ax, x);
                g = Ax - b;
                return 0.5 * x.dot(Ax) - x.dot(b) + TErrorTerm::ConstantContribution(system.GroundTruth());
            }

            unsigned int Dimensions() const
            {
                return system.Dimensions();
            }

            void ProvideStartingPoint(TVector& x0) const
            {
                x0 = TVector::Zero(Dimensions());
            }


            void Report(const char* fmt, ...) const
            {
                va_list args;
                va_start(args, fmt);
                TTraits::Monitor::ReportVA(fmt, args);
                va_end(args);
            }

            double Eval(const std::vector<double>& x_, std::vector<double>& g)
            {
                TVector Ax(Dimensions());
                const TVector x = Utility::vector_cast<TValue>(x_);
                system.MultiplySystemMatrixBy(Ax, x);
                g = Utility::vector_cast<double>((Ax - b).eval());
                return 0.5 * x.dot(Ax) - x.dot(b);
            }

            void ProvideStartingPoint(std::vector<double>& x0) const
            {
                std::fill(x0.begin(), x0.end(), 0.0);
            }
        };

        // Minimize 1/2 x^T A x - x^Tb with A positive-definite subject to unit simplex constraints on blocks of variables.
        template<typename TTraits, typename TErrorTerm=Loss::NoErrorTerm<TTraits> >
        class ConstrainedQuadratic : public Minimization::ProjectableProblem<typename TTraits::ValueType>
        {
        public:
            typedef typename TTraits::ValueType TValue;
            typedef typename Classify::LinearSystem<TTraits, TErrorTerm>::Type TLinearSystem;
            typedef typename TLinearSystem::VectorType TVector;

        private:
            const TLinearSystem& system;
            TVector b;

        public:
            ConstrainedQuadratic(const TLinearSystem& system_) : system(system_), b(system_.Dimensions())
            {
                system.ProvideRightHandSide(b);
            }

            // Note: The gradient with respect to x is given by Ax - b.
            TValue Eval(const TVector& x, TVector& g)
            {
                TVector Ax(Dimensions());
                system.MultiplySystemMatrixBy(Ax, x);
                g = Ax - b;
                return 0.5 * x.dot(Ax) - x.dot(b) + TErrorTerm::ConstantContribution(system.GroundTruth());
            }

            TVector Project(const TVector& x) const
            {
                TVector xproj_;
                Compute::SystemVectorRef<TValue, TTraits::UnaryGroundLabel::Size> xproj(system.Width(), system.Height(), xproj_);
                xproj_ += x;
                #pragma omp parallel for
                for( int i = 0; i < system.NumPixels(); ++i )
                {
                    Minimization::ProjectOntoUnitSimplex( xproj[i] );
                }
                return xproj_;
            }

            const TLinearSystem& System() const
            {
                return system;
            }

            bool IsFeasible(const TVector& x) const
            {
                return Project(x).isApprox(x, TValue(1e-6));
            }

            unsigned int Dimensions() const
            {
                return system.Dimensions();
            }

            virtual TValue Norm(const TVector& g) const
            {
                return g.template lpNorm<Eigen::Infinity>();
            }

            void ProvideStartingPoint(TVector& x0) const
            {
                x0 = Project(TVector::Zero(Dimensions()));
            }

            void Report(const char* fmt, ...) const
            {
                va_list args;
                va_start(args, fmt);
                TTraits::Monitor::ReportVA(fmt, args);
                va_end(args);
            }

            void ProvideStartingPoint(std::vector<double>& x0) const
            {
                std::fill(x0.begin(), x0.end(), 0.0);
            }
        };

    }

    template <typename TTraits, typename TErrorTerm>
    struct LinearSystem
    {
        typedef typename Detail::LinearSystemDispatcher<TTraits, TErrorTerm, TTraits::CachingMode>::Type Type;
    };


    // Minimize the unconstrained quadratic resulting from an RTF model at test time using limited-memory BFGS.
    template <typename TTraits, size_t M>
    ImageRefC<typename TTraits::UnaryGroundLabel> RegressViaLBFGS(const typename TTraits::UnaryFactorTypeVector& Us,
            const typename TTraits::PairwiseFactorTypeVector& Ps,
            const typename TTraits::LinearOperatorVector& Ls,
            const ImageRefC<typename TTraits::InputLabel> image,
            typename TTraits::ValueType residualTol = 1e-4,
            unsigned maxNumIt = 5000)

    {
        typedef typename LinearSystem<TTraits, Loss::NoErrorTerm<TTraits>>::Type TLinearSystem;
        Detail::UnconstrainedQuadratic<TTraits> quadratic(TLinearSystem(image, Us, Ps, Ls));
        typename Detail::UnconstrainedQuadratic<TTraits>::TVector solution(quadratic.Dimensions());
        Minimization::LBFGSMinimize<M>(quadratic, solution, maxNumIt, residualTol, true, 10);
        return Utility::LabelingFromSolution<TTraits>(image.Width(), image.Height(), solution);
    }

    // Solve the sparse linear system Ax = b resulting from an RTF model using conjugate gradient.
    template <typename TTraits, bool instantiate>
    ImageRefC<typename TTraits::UnaryGroundLabel> RegressViaConjugateGradient(const typename TTraits::UnaryFactorTypeVector& Us,
            const typename TTraits::PairwiseFactorTypeVector& Ps,
            const ImageRefC<typename TTraits::InputLabel>& image,
            typename TTraits::ValueType residualTol = 1e-4,
            unsigned maxNumIt = 5000)
    {
        Detail::LinearSystem<TTraits, Loss::NoErrorTerm<TTraits>, instantiate> system(image, Us, Ps);
        return Utility::LabelingFromSolution<TTraits>(image.Width(), image.Height(),
                Minimization::CGSolve<typename TTraits::ValueType>(system, maxNumIt, residualTol, false));
    }

    template <typename TTraits, bool instantiate>
    ImageRefC<typename TTraits::UnaryGroundLabel> RegressViaConjugateGradient(const typename TTraits::UnaryFactorTypeVector& Us,
            const typename TTraits::PairwiseFactorTypeVector& Ps,
            const typename TTraits::LinearOperatorVector& Ls,
            const ImageRefC<typename TTraits::InputLabel>& image,
            typename TTraits::ValueType residualTol = 1e-4,
            unsigned maxNumIt = 5000)
    {
        Detail::LinearSystem<TTraits, Loss::NoErrorTerm<TTraits>, instantiate> system(image, Us, Ps, Ls);
        return Utility::LabelingFromSolution<TTraits>(image.Width(), image.Height(),
                Minimization::CGSolve<typename TTraits::ValueType>(system, maxNumIt, residualTol, false));
    }

    // Efficient code path for a model that only consists of unary factors
    template<typename TTraits>
    ImageRefC<typename TTraits::UnaryGroundLabel> PredictUnariesOnly(const typename TTraits::UnaryFactorTypeVector& Us,
            const typename TTraits::PairwiseFactorTypeVector& Ps,
            const ImageRefC<typename TTraits::InputLabel>& image)
    {
        ImageRef<typename TTraits::UnaryGroundLabel> ret(image.Width(), image.Height());
        Compute::for_each_subgraph<TTraits, true>(TTraits::Feature::PreProcess(image),
                image.Width(), image.Height(), Us, Ps,
                [&](const Compute::Subgraph<TTraits>& G)
        {
            ret(G.PosX(), G.PosY()) = G.Solve();
        });
        return ret;
    }

    // Poly-algorithm that selects the proper inference algorithm based on the model definition.
    template<typename TTraits>
    ImageRefC<typename TTraits::UnaryGroundLabel> Predict(const typename TTraits::UnaryFactorTypeVector& Us,
            const typename TTraits::PairwiseFactorTypeVector& Ps,
            const typename TTraits::LinearOperatorVector& Ls,
            const ImageRefC<typename TTraits::InputLabel>& image,
            typename TTraits::ValueType residualTol = 1e-4,
            unsigned maxNumIt = 5000,
            bool enforceSimplexConstraints = false)
    {
        if( ! enforceSimplexConstraints )
        {
            if( Ps.size() == 0 && Ls.size() == 0 )
                return PredictUnariesOnly<TTraits>(Us, Ps, image);

            typename LinearSystem<TTraits, Loss::NoErrorTerm<TTraits>>::Type system(image, Us, Ps, Ls);
#ifdef USE_GPU
            auto solver = system.GetGPUSolver();
            return Utility::LabelingFromSolution<TTraits>(image.Width(), image.Height(), system.SolveOnGPU(solver, maxNumIt, residualTol));
#else
            return Utility::LabelingFromSolution<TTraits>(image.Width(), image.Height(),
                    residualTol < 0 ?
                    Compute::ForwardGaussJacobi<TTraits>(system, maxNumIt)
                    : Minimization::CGSolve<typename TTraits::ValueType>(system, maxNumIt, residualTol, false));
#endif
        }
        else
        {
            typedef typename LinearSystem<TTraits, Loss::NoErrorTerm<TTraits>>::Type TLinearSystem;
            TLinearSystem system(image, Us, Ps, Ls);
            Classify::Detail::ConstrainedQuadratic<TTraits, Loss::NoErrorTerm<TTraits>> quadratic(system);
            typename Classify::Detail::ConstrainedQuadratic<TTraits, Loss::NoErrorTerm<TTraits>>::TVector solution(quadratic.Dimensions());
            Minimization::SPGMinimizeCQ(quadratic, solution, maxNumIt, residualTol, false, false, 15);
            return Utility::LabelingFromSolution<TTraits>(image.Width(), image.Height(), solution);
        }
    }
}

#endif // H_RTF_CLASSIFY_H
