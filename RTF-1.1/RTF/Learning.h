/* This file is part of the "Regression Tree Fields" (RTF) source code distribution,
 * obtained from http://research.microsoft.com/downloads.
 * It is provided to you under the terms of the Microsoft Research License Agreement
 * (MSR-LA). Please see License.txt for details.
 *
 *
 * File: Learning.h
 * Implements a high-level procedural interface for estimation of model parameters.
 *
 */

#ifndef H_RTF_LEARNING_H
#define H_RTF_LEARNING_H

#include <vector>
#include <numeric>

#include <Eigen/Dense>

#include "Types.h"
#include "Rect.h"
#include "Trees.h"
#include "Compute.h"
#include "Monitor.h"
#include "Training.h"
#include "Utility.h"
#include "Minimization.h"

namespace Learning
{
    namespace Detail
    {

        // Bogus tag for the negative log-pseudolikelihood; this is not a valid loss in the sense of Loss.h
        // since it is not defined in terms of the prediction of the model, but we use it internally for dispatching.
        class NLPLLoss;
    }

    // This class implements the Minimization::ProjectableProblem interface and represents our Pseudolikelihood
    // maximization problem that is solved to determine the optimal model parameters.
    template<typename TTraits, bool Subsample = false>
    class RegressionTreeFieldProblemBase : public Minimization::ProjectableProblem<typename TTraits::ValueType>
    {
    public:
        typedef typename Minimization::ProjectableProblem<typename TTraits::ValueType>::TVector TVector;
        typedef typename TTraits::ValueType                    TValue;
        typedef typename TTraits::DataSampler                  TDataSampler;
        typedef typename Compute::FactorType<TValue>           TFactorType;
        typedef typename TTraits::UnaryFactorType              TUnaryFactorType;
        typedef typename TTraits::PairwiseFactorType           TPairwiseFactorType;

    protected:
        mutable std::vector<TUnaryFactorType> Us;
        mutable std::vector<TPairwiseFactorType> Ps;
        mutable typename TTraits::LinearOperatorVector Ls;
        const TDataSampler& traindb;

        // Convenience method that provides a (non-modifying) functor to each factor type of our model
        void for_each_factor_type(std::function<void (const TFactorType&)> op) const
        {
            std::for_each(Us.begin(), Us.end(), [&](const TFactorType & U)
            {
                op(U);
            });
            std::for_each(Ps.begin(), Ps.end(), [&](const TFactorType & P)
            {
                op(P);
            });
            std::for_each(Ls.begin(), Ls.end(), [&](const TFactorType & L)
            {
                op(L);
            });
        }

        // Same as above, but for functors that can actually modify the factor types
        void for_each_factor_type(std::function<void (TFactorType&)> op)
        {
            std::for_each(Us.begin(), Us.end(), [&](TFactorType & U)
            {
                op(U);
            });
            std::for_each(Ps.begin(), Ps.end(), [&](TFactorType & P)
            {
                op(P);
            });
            std::for_each(Ls.begin(), Ls.end(), [&](TFactorType & L)
            {
                op(L);
            });
        }

    public:

        // Sets up our training problem. The factor types and the dataset must be passed in.
        RegressionTreeFieldProblemBase(const std::vector<TUnaryFactorType>& Us_,
                                       const std::vector<TPairwiseFactorType>& Ps_,
                                       const typename TTraits::LinearOperatorVector& Ls_,
                                       const TDataSampler& traindb_)
            : Us(Us_), Ps(Ps_), Ls(Ls_), traindb(traindb_)
        {
#ifdef _OPENMP
            for_each_factor_type([&](TFactorType & T)
            {
                T.InitializeLocks();
            });
#endif
            Report("Optimizing %u weights.\n", Dimensions());
        }

        virtual ~RegressionTreeFieldProblemBase()
        {
#ifdef _OPENMP
            for_each_factor_type([&](TFactorType & T)
            {
                T.DestroyLocks();
            });
#endif
        }

        // Obtains a feasible point from the given infeasible point.
        TVector Project(const TVector& infeasible) const
        {
            TVector feasible = infeasible;
            TValue *fptr = feasible.data();
            for_each_factor_type([&](const TFactorType & T)
            {
                fptr = T.Project(fptr);
            });
            return feasible;
        }

        // Checks if the given point is feasible.
        bool IsFeasible(const TVector& point) const
        {
            const TValue *pptr = point.data();
            bool feasible = true;
            for_each_factor_type([&](const TFactorType & T)
            {
                pptr = T.CheckFeasibility(pptr, feasible);
            });
            return feasible;
        }

        // Overall dimensionality of our problem; corresponds to the overall number of weights.
        unsigned Dimensions() const
        {
            size_t dim = 0;
            for_each_factor_type([&](const TFactorType & T)
            {
                dim += T.NumWeights();
            });
            return (unsigned) dim;
        }

        // Provides a feasible starting point in the given output parameter 'point'.
        void ProvideStartingPoint(TVector& point) const
        {
            // Collect the current weights of the factor types
            TValue *wptr = point.data();
            for_each_factor_type([&](const TFactorType & T)
            {
                wptr = T.GetWeights(wptr);
            });
        }

        // We measure the norm of the gradient according to L1 norm --
        // This ensures that the initial gradient norm of the problem is the same
        // for any depth of the trees and thus allows to set the optimization tolerance
        // in a meaningful manner.
        TValue Norm(const TVector& g) const
        {
            //return g.template lpNorm<1>();
            return g.norm();
        }

        void Report(const char* fmt, ...) const
        {
            va_list args;
            va_start(args, fmt);
            TTraits::Monitor::ReportVA(fmt, args);
            va_end(args);
        }

        void ReportVA(const char* fmt, va_list args) const
        {
            TTraits::Monitor::ReportVA(fmt, args);
        }
    };

    // Class that represent a loss-based RTF learning problem.
    template<typename TTraits, typename TLossTag, bool Subsample = false>
    class RegressionTreeFieldProblem : public RegressionTreeFieldProblemBase<TTraits, Subsample>
    {
    private:
        typedef RegressionTreeFieldProblemBase<TTraits, Subsample> Base;
        typedef typename Base::TValue TValue;
        typedef typename Base::TVector TVector;
        typedef typename Base::TFactorType TFactorType;
        typedef typename Base::TUnaryFactorType TUnaryFactorType;
        typedef typename Base::TPairwiseFactorType TPairwiseFactorType;
        typedef typename Base::TDataSampler TDataSampler;

        typename TTraits::ValueType residualTolCG;
        size_t                      maxNumItCG;

    public:
        typedef typename Compute::FactorGraph<TTraits> TFactorGraph;

        RegressionTreeFieldProblem(const std::vector<TUnaryFactorType>& Us_,
                                   const std::vector<TPairwiseFactorType>& Ps_,
                                   const typename TTraits::LinearOperatorVector& Ls_,
                                   const TDataSampler& traindb_,
                                   typename TTraits::ValueType residualTolCG_,
                                   size_t maxNumItCG_) : Base::RegressionTreeFieldProblemBase(Us_, Ps_, Ls_, traindb_),
            residualTolCG(residualTolCG_), maxNumItCG(maxNumItCG_)
        {
        }

#ifdef USE_MPI
        // Evaluates the loss specified by TLossTag and stores the gradient with
        // respect to the model parameters in the given flat gradient vector.
        TValue Eval(const TVector& point, TVector& gradient)
        {
            const auto normC = Loss::Loss<TTraits, TLossTag>::NormalizationConstant(Base::traindb);

            // Initialize the factor types using the given parameters
            const TValue *wptr = point.data();
            this->for_each_factor_type([&](TFactorType & T)
            {
                T.ClearGradient();
                wptr = T.SetWeights(wptr);
            });
            // Iterate over all images to accumulate the objective and the gradient
            TValue myObj = TValue(0), obj = TValue(0);
            Compute::for_each_factor_graph<TTraits>(Base::traindb, Base::Us, Base::Ps, Base::Ls, [&](const TFactorGraph & G)
            {
                myObj += G.template ComputeObjectiveAccumulateGradient<TLossTag>(normC, maxNumItCG, residualTolCG);
            });
            boost::mpi::all_reduce(MPI::Communicator(), myObj, obj, std::plus<TValue>());

            // Collect the gradient that was accumulated while iterating over the conditioned subgraphs,
            // and add in the contributions of the prior
            TVector myGradient(TVector::Zero(this->Dimensions()));
            TValue *gptr = myGradient.data();
            this->for_each_factor_type([&](const TFactorType & T)
            {
                gptr = T.GetGradientAddPrior(gptr, obj);
            });
            gradient.setZero(this->Dimensions());
            boost::mpi::all_reduce(MPI::Communicator(), myGradient.data(), this->Dimensions(), gradient.data(), std::plus<TValue>());

            // An overflow occurred during computation of the objective; we signal this to the linesearch
            // routine by returning a very large value
            if(! Utility::isfinite(obj))
                return std::numeric_limits<TValue>::max();
            else
                return obj;
        }
#else // USE_MPI
        // Evaluates the loss specified by TLossTag and stores the gradient with
        // respect to the model parameters in the given flat gradient vector.
        TValue Eval(const TVector& point, TVector& gradient)
        {
            const auto normC = Loss::Loss<TTraits, TLossTag>::NormalizationConstant(Base::traindb);
            // Initialize the factor types using the given parameters
            const TValue *wptr = point.data();
            this->for_each_factor_type([&](TFactorType & T)
            {
                T.ClearGradient();
                wptr = T.SetWeights(wptr);
            });
            // Iterate over all images to accumulate the objective and the gradient
            TValue obj = TValue();
            Compute::for_each_factor_graph<TTraits>(Base::traindb, Base::Us, Base::Ps, Base::Ls, [&](const TFactorGraph & G)
            {
                obj += G.template ComputeObjectiveAccumulateGradient<TLossTag>(normC, maxNumItCG, residualTolCG);
            });

            // Collect the gradient that was accumulated while iterating over the conditioned subgraphs,
            // and add in the contributions of the prior
            TValue *gptr = gradient.data();
            this->for_each_factor_type([&](const TFactorType & T)
            {
                gptr = T.GetGradientAddPrior(gptr, obj);
            });

            // An overflow occurred during computation of the objective; we signal this to the linesearch
            // routine by returning a very large value
            if(! Utility::isfinite(obj))
                return std::numeric_limits<TValue>::max();
            else
                return obj;
        }
#endif // USE_MPI
    };

    // Class that represent a pseudolikelihood-based RTF learning problem.
    // We need a specialization because the learning approach is quite different from those
    // objective functions that are based on complete factor graphs and require inference.
    template<typename TTraits, bool Subsample>
    class RegressionTreeFieldProblem<TTraits, Detail::NLPLLoss, Subsample> : public RegressionTreeFieldProblemBase<TTraits, Subsample>
    {
    public:
        typedef RegressionTreeFieldProblemBase<TTraits, Subsample> Base;
        typedef typename Base::TValue TValue;
        typedef typename Base::TVector TVector;
        typedef typename Base::TFactorType TFactorType;
        typedef typename Base::TUnaryFactorType TUnaryFactorType;
        typedef typename Base::TPairwiseFactorType TPairwiseFactorType;
        typedef typename Base::TDataSampler TDataSampler;

        typedef typename Compute::ConditionedSubgraph<TTraits> TConditionedSubgraph;

        RegressionTreeFieldProblem(const std::vector<TUnaryFactorType>& Us_,
                                   const std::vector<TPairwiseFactorType>& Ps_,
                                   const TDataSampler& traindb_)
            : Base::RegressionTreeFieldProblemBase(Us_, Ps_, typename TTraits::LinearOperatorVector(), traindb_)
        {
        }

        // Evaluates the negative log-pseudolikelihood objective and stores the gradient with
        // respect to the model parameters in the given flat gradient vector.
        TValue Eval(const TVector& point, TVector& gradient)
        {
            const auto numSubgraphs = Compute::num_subgraphs<TTraits, Subsample>(Base::Us, Base::Ps, Base::traindb);
            // Initialize the factor types using the given parameters
            const TValue *wptr = point.data();
            this->for_each_factor_type([&](TFactorType & T)
            {
                T.ClearGradient();
                wptr = T.SetWeights(wptr);
            });
            // Iterate over the conditioned subgraphs to accumulate the objective and the gradient
            TValue obj = TValue();
            Compute::for_each_conditioned_subgraph<TTraits, Subsample, true>(Base::traindb, Base::Us, Base::Ps,
                    [&](const TConditionedSubgraph & G_j)
            {
                const auto subObj = G_j.ComputeObjectiveAccumulateGradient(numSubgraphs);
                #pragma omp atomic
                obj += subObj;
            });
            auto before = obj;
            // Collect the gradient that was accumulated while iterating over the conditioned subgraphs,
            // and add in the contributions of the prior
            TValue *gptr = gradient.data();
            this->for_each_factor_type([&](const TFactorType & T)
            {
                gptr = T.GetGradientAddPrior(gptr, obj);
            });

            // An overflow occurred during computation of the objective; we signal this to the linesearch
            // routine by returning a very large value
            if(! Utility::isfinite(obj))
                return std::numeric_limits<TValue>::max();
            else
                return obj;
        }
    };


    // Helper method that returns a tree storing Weights intances at its leaves, rather than a mean
    // vector (as the trees returned by regression tree training do).
    template <typename TFeature, typename TLabel, typename TBasis>
    typename Traits_<TFeature, TLabel, TBasis>::ModelTreeRef
    ConvertTree(const typename Traits_<TFeature, TLabel, TBasis>::RegressionTreeCRef tree)
    {
        typename Traits_<TFeature, TLabel, TBasis>::ModelTreeRef modelTree;
        auto j = modelTree.begin_breadth_first();

        for(auto i = tree.begin_breadth_first(); i != tree.end_breadth_first(); ++i, ++j)
        {
            if(i == tree.begin())
            {
                modelTree.set_head(NodeData<TFeature, typename Traits_<TFeature, TLabel, TBasis>::Weights>());
                j = modelTree.begin_breadth_first();
            }

            j->feature = i->feature;

            for(size_t k = 0; k < i.number_of_children(); k++)
                modelTree.append_child(j);
        }

        return modelTree;
    }

    // Given a unary regression tree obtained from regression tree training, constructs a unary factor type suitable for
    // use in a regression tree field model.
    template <typename TTraits>
    typename TTraits::UnaryFactorType
    MakeUnaryFactorType(typename TTraits::UnaryTreeCRef tree,
                        typename TTraits::ValueType smallestEigenvalue       = 1e-2,
                        typename TTraits::ValueType largestEigenvalue        = 1e2,
                        int quadraticBasisIndex                              = -1,
                        typename TTraits::ValueType linearRegularizationC    = TTraits::UnaryPrior::DefaultLinearConstant(),
                        typename TTraits::ValueType quadraticRegularizationC = TTraits::UnaryPrior::DefaultQuadraticConstant())
    {
        const VecCRef<Vector2D<int>> offsets(1);
        return typename TTraits::UnaryFactorType(ConvertTree < typename TTraits::Feature,
                typename TTraits::UnaryGroundLabel,
                typename TTraits::UnaryBasis > (tree), offsets,
                smallestEigenvalue, largestEigenvalue,
                linearRegularizationC, quadraticRegularizationC, quadraticBasisIndex);
    }

    // Creates a unary factor type the tree of which is given by a single root node. See above for a description
    // of the optional parameters.
    // The instantiated factor type is suitable as an argument to later invocation of 'LearnTreesAndWeightsJointly'.
    template<typename TTraits>
    typename TTraits::UnaryFactorType
    MakeUnaryFactorType(typename TTraits::ValueType smallestEigenvalue       = 1e-2,
                        typename TTraits::ValueType largestEigenvalue        = 1e2,
                        int quadraticBasisIndex                              = -1,
                        typename TTraits::ValueType linearRegularizationC    = TTraits::UnaryPrior::DefaultLinearConstant(),
                        typename TTraits::ValueType quadraticRegularizationC = TTraits::UnaryPrior::DefaultQuadraticConstant())
    {
        auto tree = typename Traits_<typename TTraits::Feature, typename TTraits::UnaryGroundLabel, typename TTraits::UnaryBasis>::ModelTreeRef();
        tree.set_head(NodeData<typename TTraits::Feature, typename TTraits::UnaryWeights>(smallestEigenvalue));
        const VecCRef<Vector2D<int>> offsets(1);
        return typename TTraits::UnaryFactorType(tree, offsets, smallestEigenvalue, largestEigenvalue,
                linearRegularizationC, quadraticRegularizationC, quadraticBasisIndex);
    }

    // Given a pairwise regression tree obtained from regression tree training, constructs a pairwise factor type suitable for
    // use in a regression tree field model.
    template <typename TTraits>
    typename TTraits::PairwiseFactorType
    MakePairwiseFactorType(typename TTraits::PairwiseTreeCRef tree, const VecCRef<Vector2D<int>>& offsets,
                           typename TTraits::ValueType smallestEigenvalue       = 1e-2,
                           typename TTraits::ValueType largestEigenvalue        = 1e2,
                           int quadraticBasisIndex=-1,
                           typename TTraits::ValueType linearRegularizationC    = TTraits::PairwisePrior::DefaultLinearConstant(),
                           typename TTraits::ValueType quadraticRegularizationC = TTraits::PairwisePrior::DefaultQuadraticConstant())
    {
        return typename TTraits::PairwiseFactorType(ConvertTree < typename TTraits::Feature,
                typename TTraits::PairwiseGroundLabel,
                typename TTraits::PairwiseBasis > (tree), offsets,
                smallestEigenvalue, largestEigenvalue,
                linearRegularizationC, quadraticRegularizationC, quadraticBasisIndex);
    }

    // Creates a pairwise factor type the tree of which is given by a single root node. See above for a description
    // of the optional parameters.
    // The instantiated factor type is suitable as an argument to latter invocation of 'LearnTreesAndWeightsJointly'.
    template <typename TTraits>
    typename TTraits::PairwiseFactorType
    MakePairwiseFactorType(const VecCRef<Vector2D<int>>& offsets,
                           typename TTraits::ValueType smallestEigenvalue       = 1e-2,
                           typename TTraits::ValueType largestEigenvalue        = 1e2,
                           int quadraticBasisIndex=-1,
                           typename TTraits::ValueType linearRegularizationC    = TTraits::PairwisePrior::DefaultLinearConstant(),
                           typename TTraits::ValueType quadraticRegularizationC = TTraits::PairwisePrior::DefaultQuadraticConstant())
    {
        auto tree = typename Traits_<typename TTraits::Feature, typename TTraits::PairwiseGroundLabel, typename TTraits::PairwiseBasis>::ModelTreeRef();
        tree.set_head(NodeData<typename TTraits::Feature, typename TTraits::PairwiseWeights>(smallestEigenvalue));
        return typename TTraits::PairwiseFactorType(tree, offsets, smallestEigenvalue, largestEigenvalue,
                linearRegularizationC, quadraticRegularizationC, quadraticBasisIndex);
    }

    namespace Detail
    {
        // Utility function for internal use by serialization
        template <typename TTraits>
        typename TTraits::UnaryFactorType
        MakeUnaryFactorType(typename TTraits::ValueType smallestEigenvalue       = 1e-2,
                            typename TTraits::ValueType largestEigenvalue        = 1e2,
                            typename TTraits::ValueType linearRegularizationC    = TTraits::UnaryPrior::DefaultLinearConstant(),
                            typename TTraits::ValueType quadraticRegularizationC = TTraits::UnaryPrior::DefaultQuadraticConstant())
        {
            const VecCRef<Vector2D<int>> offsets(1);
            return typename TTraits::UnaryFactorType(typename Traits_ < typename TTraits::Feature,
                    typename TTraits::UnaryGroundLabel,
                    typename TTraits::UnaryBasis >::ModelTreeRef(), offsets, smallestEigenvalue, largestEigenvalue, linearRegularizationC, quadraticRegularizationC);
        }

        // Utility function for internal use by serialization
        template <typename TTraits>
        typename TTraits::PairwiseFactorType
        MakePairwiseFactorType(const VecCRef<Vector2D<int>>& offsets,
                               typename TTraits::ValueType smallestEigenvalue       = 1e-2,
                               typename TTraits::ValueType largestEigenvalue        = 1e2,
                               typename TTraits::ValueType linearRegularizationC    = TTraits::UnaryPrior::DefaultLinearConstant(),
                               typename TTraits::ValueType quadraticRegularizationC = TTraits::UnaryPrior::DefaultQuadraticConstant())
        {
            return typename TTraits::PairwiseFactorType(typename Traits_ < typename TTraits::Feature,
                    typename TTraits::PairwiseGroundLabel,
                    typename TTraits::PairwiseBasis >::ModelTreeRef(), offsets, smallestEigenvalue, largestEigenvalue, linearRegularizationC, quadraticRegularizationC);
        }
    }

    template <typename TTraits, bool Subsample, size_t m>
    typename TTraits::ValueType
    OptimizeWeights(typename TTraits::UnaryFactorTypeVector& Us,
                    typename TTraits::PairwiseFactorTypeVector& Ps,
                    const typename TTraits::DataSampler& traindb,
                    typename TTraits::ValueType breakEps = 5e-2,
                    size_t maxNumIt = 5000)
    {
        RegressionTreeFieldProblem<TTraits, Detail::NLPLLoss, Subsample> problem(Us, Ps, traindb);
        typename Minimization::ProjectableProblem<typename TTraits::ValueType>::TVector solution(problem.Dimensions());
        return Minimization::RestartingLBFGSMinimize<m>(problem, solution, maxNumIt, breakEps, true, 5);
    }

    // Implementation of joint training of trees and weights. See also class Criteria::GradientNormCriterion,
    // Training::GrowRegressionTree() and Compute::ConditionedSubgraph::ForEachGradientContributionOfType()
    // for further implementation bits.
    namespace Detail
    {

        template<typename TTraits, typename TLossTag>
        struct LearningTraits
        {
            typedef typename TTraits::ValueType TValue;

            class MeanParametersRef
            {
            public:
                struct MeanParameterEntry
                {
                    std::vector<Eigen::Matrix<TValue, Eigen::Dynamic, 1>> muPrediction;
                    std::vector<Eigen::Matrix<TValue, Eigen::Dynamic, 1>> muLossGradient;
                };

            private:
                VecRef< MeanParameterEntry > entries;

            public:
                MeanParametersRef(size_t num) : entries(num) {}

                MeanParameterEntry& operator[](size_t index)
                {
                    return entries[index];
                }
                const MeanParameterEntry& operator[](size_t index) const
                {
                    return entries[index];
                }

                static MeanParametersRef Empty()
                {
                    return MeanParametersRef(0);
                }

                size_t size() const
                {
                    return entries.size();
                }
            };

        };

        template<typename TTraits>
        struct LearningTraits<TTraits, NLPLLoss>
        {
            typedef typename TTraits::ValueType TValue;
            static const size_t                 VarDim = TTraits::UnaryGroundLabel::Size;

            class MeanParametersRef
            {
            public:
                struct MeanParameterEntry
                {
                    Eigen::Matrix<TValue, VarDim, 1>       mu;
                    Eigen::Matrix<TValue, VarDim, VarDim>  Sigma;

                    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
                };

            private:
                class MeanParameters
                {
                private:
                    MeanParameterEntry * const entries;
                    size_t num;
                public:
                    MeanParameters(size_t num_) : num(num_), entries(new MeanParameterEntry[num_]) {}
                    ~MeanParameters()
                    {
                        delete[] entries;
                    }

                    friend class MeanParametersRef;
                };

                std::shared_ptr<MeanParameters> ptr;

            public:
                MeanParametersRef(size_t num) : ptr(new MeanParameters(num)) {}

                MeanParameterEntry& operator[](size_t index)
                {
                    return ptr->entries[index];
                }
                const MeanParameterEntry& operator[](size_t index) const
                {
                    return ptr->entries[index];
                }

                static MeanParametersRef Empty()
                {
                    return MeanParametersRef(0);
                }

                size_t size() const
                {
                    return ptr->num;
                }
            };
        };

        // Iterates over all conditioned subgraphs and collects their mean parameters given the
        // current weights.
        template<typename TTraits, typename TLossTag, bool Subsample>
        struct GatherMeanParameters
        {
            typedef Eigen::Matrix<typename TTraits::ValueType, Eigen::Dynamic, 1> TSolution;

            static typename LearningTraits<TTraits, TLossTag>::MeanParametersRef
            Compute(const typename TTraits::UnaryFactorTypeVector& Us,
                    const typename TTraits::PairwiseFactorTypeVector& Ps,
                    const typename TTraits::LinearOperatorVector& Ls,
                    const typename TTraits::DataSampler& traindb,
                    typename TTraits::ValueType residualTolCG, size_t maxNumItCG)
            {
                typename LearningTraits<TTraits, TLossTag>::MeanParametersRef meanParams(traindb.GetImageCount());
                Compute::for_each_factor_graph_with_index<TTraits>(traindb, Us, Ps, Ls,
                        [&](size_t id, const Compute::FactorGraph<TTraits>& G)
                {
                    G.template ComputeMeanParameters<TLossTag>(maxNumItCG, residualTolCG,
                            meanParams[id].muPrediction, meanParams[id].muLossGradient);
                });
                return meanParams;
            }
        };

        // Specialization for negative log-pseudolikelihood
        template<typename TTraits, bool Subsample>
        struct GatherMeanParameters<TTraits, NLPLLoss, Subsample>
        {
            static typename LearningTraits<TTraits, NLPLLoss>::MeanParametersRef
            Compute(const typename TTraits::UnaryFactorTypeVector& Us,
                    const typename TTraits::PairwiseFactorTypeVector& Ps,
                    const typename TTraits::LinearOperatorVector& Ls,
                    const typename TTraits::DataSampler& traindb,
                    typename TTraits::ValueType residualTolCG, size_t maxNumItCG)
            {
                const auto nParams = Compute::num_subgraphs<TTraits, Subsample>(Us, Ps, traindb);
                typename LearningTraits<TTraits, NLPLLoss>::MeanParametersRef meanParams(nParams);
                Compute::for_each_conditioned_subgraph_with_index<TTraits, Subsample, true>(traindb, Us, Ps,
                        [&](size_t id, const Compute::ConditionedSubgraph<TTraits>& G_j)
                {
                    assert(id < nParams);
                    G_j.ComputeMeanParameters(meanParams[id].mu, meanParams[id].Sigma);
                });
                return meanParams;
            }
        };

        //
        // The following higher-order types decide whether we actually need to compute the mean
        // parameters. These are only needed by the GradientNorm criterion, so in general,
        // we can omit this step.
        //

        // General criterion, don't need mean parameters
        template<typename TTraits, typename TLossTag, bool Subsample, typename TUnarySplitCriterionTag, typename TPairwiseSplitCriterionTag>
        struct MeanParameterDispatcher
        {
            static typename LearningTraits<TTraits, TLossTag>::MeanParametersRef
            Compute(const typename TTraits::UnaryFactorTypeVector& Us, const typename TTraits::PairwiseFactorTypeVector& Ps,
                    const typename TTraits::LinearOperatorVector& Ls,
                    const typename TTraits::DataSampler& traindb, typename TTraits::ValueType residualTolCG, size_t maxNumItCG)
            {
                return LearningTraits<TTraits, TLossTag>::MeanParametersRef::Empty();
            }
        };

        // Unary splitting criterion based on gradient norm -> need mean parameters
        template<typename TTraits, typename TLossTag, bool Subsample, typename TPairwiseSplitCriterionTag>
        struct MeanParameterDispatcher<TTraits, TLossTag, Subsample, GradientNormCriterion, TPairwiseSplitCriterionTag>
        {
            static typename LearningTraits<TTraits, TLossTag>::MeanParametersRef
            Compute(const typename TTraits::UnaryFactorTypeVector& Us, const typename TTraits::PairwiseFactorTypeVector& Ps,
                    const typename TTraits::LinearOperatorVector& Ls,
                    const typename TTraits::DataSampler& traindb, typename TTraits::ValueType residualTolCG, size_t maxNumItCG)
            {
                return GatherMeanParameters<TTraits, TLossTag, Subsample>::Compute(Us, Ps, Ls, traindb, residualTolCG, maxNumItCG);
            }
        };

        // Pairwise splitting criterion based on gradient norm -> need mean parameters
        template<typename TTraits, typename TLossTag, bool Subsample, typename TUnarySplitCriterionTag>
        struct MeanParameterDispatcher<TTraits, TLossTag, Subsample, TUnarySplitCriterionTag, GradientNormCriterion>
        {
            static typename LearningTraits<TTraits, TLossTag>::MeanParametersRef
            Compute(const typename TTraits::UnaryFactorTypeVector& Us, const typename TTraits::PairwiseFactorTypeVector& Ps,
                    const typename TTraits::LinearOperatorVector& Ls,
                    const typename TTraits::DataSampler& traindb, typename TTraits::ValueType residualTolCG, size_t maxNumItCG)
            {
                return GatherMeanParameters<TTraits, TLossTag, Subsample>::Compute(Us, Ps, Ls, traindb, residualTolCG, maxNumItCG);
            }
        };

        // Both splitting criteria based on gradient norm -> need mean parameters
        template<typename TTraits, typename TLossTag, bool Subsample>
        struct MeanParameterDispatcher<TTraits, TLossTag, Subsample, GradientNormCriterion, GradientNormCriterion>
        {
            static typename LearningTraits<TTraits, TLossTag>::MeanParametersRef
            Compute(const typename TTraits::UnaryFactorTypeVector& Us, const typename TTraits::PairwiseFactorTypeVector& Ps,
                    const typename TTraits::LinearOperatorVector& Ls,
                    const typename TTraits::DataSampler& traindb, typename TTraits::ValueType residualTolCG, size_t maxNumItCG)
            {
                return GatherMeanParameters<TTraits, TLossTag, Subsample>::Compute(Us, Ps, Ls, traindb, residualTolCG, maxNumItCG);
            }
        };


        // Provides the training points to tree training if the GradientNorm criterion is used;
        // towards this end, the gradient contributions of the specified factor type are collected
        // over either all, or a subsample of, the conditioned subgraphs in the dataset.
        // The mean parameters must be pre-computed prior to instantiation of this class for
        // maximum efficiency, because all factor types share the same set of mean parameters.

        template <typename TTraits, typename TLossTag, typename TOurFactorType, bool Subsample = false>
        struct GradientNormPointSamplerDiscriminative
        {
            typedef typename TOurFactorType::TWeights TWeights;
            typedef Training::LabelVector<typename TWeights::TValue, TWeights::NumCoefficients> TLabel;
            typedef typename TWeights::TVarBasisMatrix TVarBasisMatrix;
            typedef typename TWeights::TVarVarMatrix TVarVarMatrix;

            static
            size_t AddPoints(const typename TTraits::DataSampler& sampler,
                             const typename TTraits::UnaryFactorTypeVector& Us,
                             const typename TTraits::PairwiseFactorTypeVector& Ps,
                             const typename TTraits::LinearOperatorVector& Ls,
                             const TOurFactorType& T,
                             const typename LearningTraits<TTraits, TLossTag>::MeanParametersRef& meanParameters,
                             const std::function<bool (const TLabel&, int, int, const typename TTraits::PreProcessType&)>& addOp,
                             size_t maxNumItCG, typename TTraits::ValueType residualTolCG)
            {
                int nAdded = 0;
                Compute::for_each_factor_graph_with_index<TTraits>(sampler, Us, Ps, Ls,
                        [&](size_t id, const Compute::FactorGraph<TTraits>& G)
                {
                    G.template ForEachGradientContributionOfType<TLossTag>(T, meanParameters[id].muPrediction, meanParameters[id].muLossGradient, 1,
                            [&](const int posX, const int posY, const TVarBasisMatrix & Gl, const TVarVarMatrix & Gq)
                    {
                        addOp(T.MakeLabel(Gl, Gq), posX, posY, G.Prep());
                        #pragma omp atomic
                        nAdded++;
                    });
                });
                return nAdded;
            }

            static
            size_t TotalNumPoints(const typename TTraits::DataSampler& sampler,
                                  const typename TTraits::UnaryFactorTypeVector& Us,
                                  const typename TTraits::PairwiseFactorTypeVector& Ps,
                                  const typename TTraits::LinearOperatorVector& Ls,
                                  const TOurFactorType& T)
            {
                size_t nContributions = 0;
                Compute::for_each_factor_graph_with_index<TTraits>(sampler, Us, Ps, Ls,
                        [&](size_t id, const Compute::FactorGraph<TTraits>& G)
                {
                    nContributions += G.NumGradientContributionsOfType(T);
                });
                return nContributions;
            }
        };

        template <typename TTraits, typename TLossTag, typename TOurFactorType>
        struct GradientNormPointSamplerDiscriminative<TTraits, TLossTag, TOurFactorType, true>
        {
            typedef typename TOurFactorType::TWeights TWeights;
            typedef Training::LabelVector<typename TWeights::TValue, TWeights::NumCoefficients> TLabel;
            typedef typename TWeights::TVarBasisMatrix TVarBasisMatrix;
            typedef typename TWeights::TVarVarMatrix TVarVarMatrix;

            static
            size_t AddPoints(const typename TTraits::DataSampler& sampler,
                             const typename TTraits::UnaryFactorTypeVector& Us,
                             const typename TTraits::PairwiseFactorTypeVector& Ps,
                             const typename TTraits::LinearOperatorVector& Ls,
                             const TOurFactorType& T,
                             const typename LearningTraits<TTraits, TLossTag>::MeanParametersRef& meanParameters,
                             const std::function<bool (const TLabel&, int, int, const typename TTraits::PreProcessType&)>& addOp,
                             size_t maxNumItCG, typename TTraits::ValueType residualTolCG)
            {
                typedef Eigen::Matrix<typename TTraits::ValueType, Eigen::Dynamic, 1> TSolution;

                int nAdded = 0;
                Compute::for_each_factor_graph_with_index<TTraits>(sampler, Us, Ps, Ls,
                        [&](size_t id, const Compute::FactorGraph<TTraits>& G)
                {
                    G.template ForEachGradientContributionOfType<TLossTag>(T, meanParameters[id].muPrediction, meanParameters[id].muLossGradient, 1,
                            sampler.GetSubsampledVariables(id),
                            [&](const int posX, const int posY, const TVarBasisMatrix & Gl, const TVarVarMatrix & Gq)
                    {
                        addOp(T.MakeLabel(Gl, Gq), posX, posY, G.Prep());
                        #pragma omp atomic
                        nAdded++;
                    });
                });
                return nAdded;
            }

            static
            size_t TotalNumPoints(const typename TTraits::DataSampler& sampler,
                                  const typename TTraits::UnaryFactorTypeVector& Us,
                                  const typename TTraits::PairwiseFactorTypeVector& Ps,
                                  const typename TTraits::LinearOperatorVector& Ls,
                                  const TOurFactorType& T)
            {
                size_t nContributions = 0;
                Compute::for_each_factor_graph_with_index<TTraits>(sampler, Us, Ps, Ls,
                        [&](size_t id, const Compute::FactorGraph<TTraits>& G)
                {
                    nContributions += G.NumGradientContributionsOfType(T, sampler.GetSubsampledVariables(id));
                });
                return nContributions;
            }
        };

        template <typename TTraits, typename TLossTag, typename TOurFactorType, bool Subsample = false>
        struct GradientNormPointSampler
        {
            typedef typename TOurFactorType::TWeights TWeights;
            typedef Training::LabelVector<typename TWeights::TValue, TWeights::NumCoefficients> TLabel;

            typedef GradientNormPointSamplerDiscriminative<TTraits, TLossTag, TOurFactorType, Subsample> TImpl;

        private:
            const typename TTraits::DataSampler& sampler;
            const typename TTraits::UnaryFactorTypeVector& Us;
            const typename TTraits::PairwiseFactorTypeVector& Ps;
            const typename TTraits::LinearOperatorVector& Ls;
            const TOurFactorType& T;
            typename LearningTraits<TTraits, TLossTag>::MeanParametersRef meanParameters;
            size_t maxNumItCG;
            typename TTraits::ValueType residualTolCG;

        public:
            GradientNormPointSampler(const typename TTraits::DataSampler& sampler_,
                                     const typename TTraits::UnaryFactorTypeVector& Us_,
                                     const typename TTraits::PairwiseFactorTypeVector& Ps_,
                                     const typename TTraits::LinearOperatorVector& Ls_,
                                     const TOurFactorType& T_,
                                     const typename LearningTraits<TTraits, TLossTag>::MeanParametersRef& meanParameters_,
                                     size_t maxNumItCG_, typename TTraits::ValueType residualTolCG_)
                : sampler(sampler_), Us(Us_), Ps(Ps_), Ls(Ls_), T(T_), meanParameters(meanParameters_), maxNumItCG(maxNumItCG_), residualTolCG(residualTolCG_) {}

            size_t AddPoints(const std::function<bool (const TLabel&, int, int, const typename TTraits::PreProcessType&)>& addOp) const
            {
                return TImpl::AddPoints(sampler, Us, Ps, Ls, T, meanParameters, addOp, maxNumItCG, residualTolCG);
            }

            size_t TotalNumPoints() const
            {
                return TImpl::TotalNumPoints(sampler, Us, Ps, Ls, T);
            }

            VecCRef<Vector2D<int>> Offsets() const
            {
                return T.Offsets();
            }
        };

        // Implementation for negative log-pseudolikelihood gradient
        template <typename TTraits, typename TOurFactorType, bool Subsample>
        struct GradientNormPointSampler <TTraits, NLPLLoss, TOurFactorType, Subsample>
        {
        public:
            typedef typename TOurFactorType::TWeights TWeights;
            typedef Training::LabelVector<typename TWeights::TValue, TWeights::NumCoefficients> TLabel;
            typedef typename TWeights::TVarBasisMatrix TVarBasisMatrix;
            typedef typename TWeights::TVarVarMatrix TVarVarMatrix;

        private:
            const typename TTraits::DataSampler& sampler;
            const typename TTraits::UnaryFactorTypeVector& Us;
            const typename TTraits::PairwiseFactorTypeVector& Ps;
            const TOurFactorType& T;
            typename LearningTraits<TTraits, NLPLLoss>::MeanParametersRef meanParameters;

        public:
            GradientNormPointSampler(const typename TTraits::DataSampler& sampler_,
                                     const typename TTraits::UnaryFactorTypeVector& Us_,
                                     const typename TTraits::PairwiseFactorTypeVector& Ps_,
                                     const typename TTraits::LinearOperatorVector& Ls_,
                                     const TOurFactorType& T_,
                                     const typename LearningTraits<TTraits, NLPLLoss>::MeanParametersRef& meanParameters_,
                                     size_t maxNumItCG_, typename TTraits::ValueType residualTolCG_)
                : sampler(sampler_), Us(Us_), Ps(Ps_), T(T_), meanParameters(meanParameters_) {}

            size_t AddPoints(const std::function<bool (const TLabel&, int, int, const typename TTraits::PreProcessType&)>& addOp) const
            {
                int nAdded = 0;
                Compute::for_each_conditioned_subgraph_with_index<TTraits, Subsample, false>(sampler, Us, Ps,
                        [&](size_t id, const Compute::ConditionedSubgraph<TTraits>& G_j)
                {
                    G_j.ForEachGradientContributionOfType(T, meanParameters[id].mu, meanParameters[id].Sigma, 1,     // Could use numParams instead of 1, but the loss reported by
                                                          [&](const TVarBasisMatrix & Gl, const TVarVarMatrix & Gq)  // tree training is already averaged over the pix.
                    {
                        addOp(T.MakeLabel(Gl, Gq), G_j.PosX(), G_j.PosY(), G_j.Prep());
                        #pragma omp atomic
                        nAdded++;
                    });
                });
                return nAdded;
            }

            size_t TotalNumPoints() const
            {
                size_t nConnected = 0;
                Compute::for_each_conditioned_subgraph<TTraits, Subsample, true>(sampler, Us, Ps,
                        [&](const Compute::ConditionedSubgraph<TTraits>& G_j)
                {
                    #pragma omp atomic
                    nConnected += G_j.NumConnected(T);
                });
                return nConnected;
            }

            VecCRef<Vector2D<int>> Offsets() const
            {
                return T.Offsets();
            }
        };

        template <typename TTraits, typename TLossTag, bool Subsample, typename TWeights>
        struct LinearOperatorPointSampler
        {
            typedef Training::LabelVector<typename TWeights::TValue, TWeights::NumCoefficients> TLabel;

        private:
            const typename TTraits::DataSampler& sampler;
            const typename TTraits::LinearOperatorRef L;
            typename LearningTraits<TTraits, TLossTag>::MeanParametersRef meanParameters;

        public:
            LinearOperatorPointSampler(const typename TTraits::DataSampler& sampler_,
                                       const typename TTraits::LinearOperatorRef& L_,
                                       const typename LearningTraits<TTraits, TLossTag>::MeanParametersRef& meanParameters_)
                : sampler(sampler_), L(L_), meanParameters(meanParameters_) {}

            size_t AddPoints(const std::function<bool (const TLabel&, int, int, const typename TTraits::PreProcessType&)>& addOp) const
            {
                size_t nAdded = 0;

                for( size_t i = 0; i < sampler.GetImageCount(); ++ i )
                {
                    auto img  = sampler.GetInputImage(i);
                    auto prep = TTraits::Feature::PreProcess(img);

                    if( Subsample )
                    {
                        L.ForEachGradientContribution(prep, meanParameters[i].muLossGradient, meanParameters[i].muPrediction, sampler.GetSubsampledVariables(i),
                                                      [&](const int posX, const int posY, const TLabel& label)
                        {
                            addOp(label, posX, posY, prep);
                            #pragma omp atomic
                            nAdded++;
                        });
                    }
                    else
                    {
                        L.ForEachGradientContribution(prep, meanParameters[i].muLossGradient, meanParameters[i].muPrediction,
                                                      [&](const int posX, const int posY, const TLabel& label)
                        {
                            addOp(label, posX, posY, prep);
                            #pragma omp atomic
                            nAdded++;
                        });
                    }
                }
                return nAdded;
            }

            size_t TotalNumPoints() const
            {
                size_t nAdded = 0;
                for( size_t i = 0; i < sampler.GetImageCount(); ++i )
                {
                    if( Subsample )
                    {
                        nAdded += sampler.GetSubsampledVariables(i).size();
                    }
                    else
                    {
                        auto img = sampler.GetInputImage(i);
                        nAdded += img.Width() * img.Height();
                    }
                }
                return nAdded;
            }

            VecCRef<Vector2D<int>> Offsets() const
            {
                return VecRef<Vector2D<int>>();
            }
        };



        template <typename TTraits, bool Subsample, typename TWeights>
        struct LinearOperatorPointSampler<TTraits, NLPLLoss, Subsample, TWeights>
        {
            typedef Training::LabelVector<typename TWeights::TValue, TWeights::NumCoefficients> TLabel;

        public:
            LinearOperatorPointSampler(const typename TTraits::DataSampler& sampler_,
                                       const typename TTraits::LinearOperatorRef& L_,
                                       const typename LearningTraits<TTraits, NLPLLoss>::MeanParametersRef& meanParameters_) {}

            size_t AddPoints(const std::function<bool (const TLabel&, int, int, const typename TTraits::PreProcessType&)>& addOp) const
            {
                return 0;
            }

            size_t TotalNumPoints() const
            {
                return 0;
            }

            VecCRef<Vector2D<int>> Offsets() const
            {
                return VecRef<Vector2D<int>>();
            }
        };

        //
        // The following higher-order types instantiate a point sampler of the proper type,
        // based on the splitting criterion and whether subsampling is activated or not.
        // For general splitting criteria, we can simply re-use the point samplers from
        // Training.h. If the pseudolikelihood gradient norm is to be used as a criterion,
        // we instantiate GradientNormPointSampler with the appropriate type parameters.
        //

        // Common split criterion, without subsampling
        template<typename TTraits, typename TLossTag, typename TSplitCritTag, bool Subsample = false>
        struct UnaryPointSamplerDispatcher
        {
            static Training::Detail::UnaryPointSampler<TTraits>
            Instantiate(const typename TTraits::DataSampler& sampler,
                        const typename TTraits::UnaryFactorTypeVector& Us,
                        const typename TTraits::PairwiseFactorTypeVector& Ps,
                        const typename TTraits::LinearOperatorVector& Ls,
                        const typename TTraits::UnaryFactorType& U,
                        const typename LearningTraits<TTraits, TLossTag>::MeanParametersRef& meanParameters,
                        size_t maxNumItCG = 10000, typename TTraits::ValueType residualTolCG = 1e-4)
            {
                return Training::Detail::UnaryPointSampler<TTraits>(sampler);
            }
        };

        // Common split criterion, with subsampling
        template<typename TTraits, typename TLossTag, typename TSplitCritTag>
        struct UnaryPointSamplerDispatcher<TTraits, TLossTag, TSplitCritTag, true>
        {
            static Training::Detail::UnaryPointSubsampler<TTraits>
            Instantiate(const typename TTraits::DataSampler& sampler,
                        const typename TTraits::UnaryFactorTypeVector& Us,
                        const typename TTraits::PairwiseFactorTypeVector& Ps,
                        const typename TTraits::LinearOperatorVector& Ls,
                        const typename TTraits::UnaryFactorType& U,
                        const typename LearningTraits<TTraits, TLossTag>::MeanParametersRef& meanParameters,
                        size_t maxNumItCG = 10000, typename TTraits::ValueType residualTolCG = 1e-4)
            {
                return Training::Detail::UnaryPointSubsampler<TTraits>(sampler);
            }
        };

        // Common split criterion, without subsampling
        template<typename TTraits, typename TLossTag, typename TSplitCritTag, bool Subsample = false>
        struct PairwisePointSamplerDispatcher
        {
            static Training::Detail::PairwisePointSampler<TTraits>
            Instantiate(const typename TTraits::DataSampler& sampler,
                        const typename TTraits::UnaryFactorTypeVector& Us,
                        const typename TTraits::PairwiseFactorTypeVector& Ps,
                        const typename TTraits::LinearOperatorVector& Ls,
                        const typename TTraits::PairwiseFactorType& P,
                        const typename LearningTraits<TTraits, TLossTag>::MeanParametersRef& meanParameters,
                        size_t maxNumItCG = 10000, typename TTraits::ValueType residualTolCG = 1e-4)
            {
                return Training::Detail::PairwisePointSampler<TTraits>(sampler, P.Offsets());
            }
        };

        // Common split criterion, with subsampling
        template<typename TTraits, typename TLossTag, typename TSplitCritTag>
        struct PairwisePointSamplerDispatcher<TTraits, TLossTag, TSplitCritTag, true>
        {
            static Training::Detail::PairwisePointSubsampler<TTraits>
            Instantiate(const typename TTraits::DataSampler& sampler,
                        const typename TTraits::UnaryFactorTypeVector& Us,
                        const typename TTraits::PairwiseFactorTypeVector& Ps,
                        const typename TTraits::LinearOperatorVector& Ls,
                        const typename TTraits::PairwiseFactorType& P,
                        const typename LearningTraits<TTraits, TLossTag>::MeanParametersRef& meanParameters,
                        size_t maxNumItCG = 10000, typename TTraits::ValueType residualTolCG = 1e-4)
            {
                return Training::Detail::PairwisePointSubsampler<TTraits>(sampler, P.Offsets());
            }
        };

        // Gradient norm splitting, without subsampling
        template <typename TTraits, typename TLossTag>
        struct UnaryPointSamplerDispatcher<TTraits, TLossTag, ::GradientNormCriterion, false>
        {
            static GradientNormPointSampler<TTraits, TLossTag, typename TTraits::UnaryFactorType, false>
            Instantiate(const typename TTraits::DataSampler& sampler,
                        const typename TTraits::UnaryFactorTypeVector& Us,
                        const typename TTraits::PairwiseFactorTypeVector& Ps,
                        const typename TTraits::LinearOperatorVector& Ls,
                        const typename TTraits::UnaryFactorType& U,
                        const typename LearningTraits<TTraits, TLossTag>::MeanParametersRef& meanParameters,
                        size_t maxNumItCG = 10000, typename TTraits::ValueType residualTolCG = 1e-4)
            {
                return GradientNormPointSampler<TTraits, TLossTag, typename TTraits::UnaryFactorType, false>(sampler, Us, Ps, Ls, U, meanParameters, maxNumItCG, residualTolCG);
            }
        };

        // Gradient norm splitting, with subsampling
        template <typename TTraits, typename TLossTag>
        struct UnaryPointSamplerDispatcher<TTraits, TLossTag, ::GradientNormCriterion, true>
        {
            static GradientNormPointSampler<TTraits, TLossTag, typename TTraits::UnaryFactorType, true>
            Instantiate(const typename TTraits::DataSampler& sampler,
                        const typename TTraits::UnaryFactorTypeVector& Us,
                        const typename TTraits::PairwiseFactorTypeVector& Ps,
                        const typename TTraits::LinearOperatorVector& Ls,
                        const typename TTraits::UnaryFactorType& U,
                        const typename LearningTraits<TTraits, TLossTag>::MeanParametersRef& meanParameters,
                        size_t maxNumItCG = 10000, typename TTraits::ValueType residualTolCG = 1e-4)
            {
                return GradientNormPointSampler<TTraits, TLossTag, typename TTraits::UnaryFactorType, true>(sampler, Us, Ps, Ls, U, meanParameters, maxNumItCG, residualTolCG);
            }
        };

        // Gradient norm splitting, without subsampling
        template <typename TTraits, typename TLossTag>
        struct PairwisePointSamplerDispatcher<TTraits, TLossTag, ::GradientNormCriterion, false>
        {
            static GradientNormPointSampler<TTraits, TLossTag, typename TTraits::PairwiseFactorType, false>
            Instantiate(const typename TTraits::DataSampler& sampler,
                        const typename TTraits::UnaryFactorTypeVector& Us,
                        const typename TTraits::PairwiseFactorTypeVector& Ps,
                        const typename TTraits::LinearOperatorVector& Ls,
                        const typename TTraits::PairwiseFactorType& P,
                        const typename LearningTraits<TTraits, TLossTag>::MeanParametersRef& meanParameters,
                        size_t maxNumItCG = 10000, typename TTraits::ValueType residualTolCG = 1e-4)
            {
                return GradientNormPointSampler<TTraits, TLossTag, typename TTraits::PairwiseFactorType, false>(sampler, Us, Ps, Ls, P, meanParameters, maxNumItCG, residualTolCG);
            }
        };

        // Gradient norm splitting, with subsampling
        template <typename TTraits, typename TLossTag>
        struct PairwisePointSamplerDispatcher<TTraits, TLossTag, ::GradientNormCriterion, true>
        {
            static GradientNormPointSampler<TTraits, TLossTag, typename TTraits::PairwiseFactorType, true>
            Instantiate(const typename TTraits::DataSampler& sampler,
                        const typename TTraits::UnaryFactorTypeVector& Us,
                        const typename TTraits::PairwiseFactorTypeVector& Ps,
                        const typename TTraits::LinearOperatorVector& Ls,
                        const typename TTraits::PairwiseFactorType& P,
                        const typename LearningTraits<TTraits, TLossTag>::MeanParametersRef& meanParameters,
                        size_t maxNumItCG = 10000, typename TTraits::ValueType residualTolCG = 1e-4)
            {
                return GradientNormPointSampler<TTraits, TLossTag, typename TTraits::PairwiseFactorType, true>(sampler, Us, Ps, Ls, P, meanParameters, maxNumItCG, residualTolCG);
            }
        };

        //
        // The following higher-order types instantiate a splitting criterion of the proper class,
        // based on the splitting criterion tag and the actual splitting criterion type.
        // They are used to abstract away the fact that the constructor of GradientNormCriterion
        // requires additional arguments that the others don't require.
        //

        // Common split criterion
        template <typename TSplitCriterionTag, typename TSplitCriterion, typename TFactorType>
        struct CriterionDispatcher
        {
            static TSplitCriterion
            Instantiate(const TFactorType& T, typename TFactorType::TWeights::TValue purityEpsilon)
            {
                return TSplitCriterion(purityEpsilon);
            }
        };

        // Gradient norm splitting
        template <typename TSplitCriterion, typename TFactorType>
        struct CriterionDispatcher<GradientNormCriterion, TSplitCriterion, TFactorType>
        {
            static TSplitCriterion
            Instantiate(const TFactorType& T, typename TFactorType::TWeights::TValue purityEpsilon)
            {
                return TSplitCriterion(purityEpsilon, T.WeightsInBreadthFirstOrder(), T.SmallestEigenvalue(), T.LargestEigenvalue());
            }
        };

        // Optimizes the parameters of our model for the specified loss
        template <typename TTraits, typename TLossTag, bool Subsample, size_t m>
        struct WeightOptimization
        {
            static typename TTraits::ValueType
            Run(typename TTraits::UnaryFactorTypeVector& Us,
                typename TTraits::PairwiseFactorTypeVector& Ps,
                typename TTraits::LinearOperatorVector& Ls,
                const typename TTraits::DataSampler& traindb,
                typename TTraits::ValueType breakEps = 5e-2,
                size_t maxNumIt = 5000,
                bool stagedTraining = false,
                typename TTraits::ValueType residualTolCG = 1e-4,
                size_t maxNumItCG = 10000)
            {

                if(stagedTraining)     // Optimize (convex) negative log-pseudolikelihood first, to avoid a bad local minimum of the true loss
                {
                    RegressionTreeFieldProblem<TTraits, Loss::ContinuousPerceptron, Subsample> problem(Us, Ps, Ls, traindb, residualTolCG, maxNumItCG);
                    typename Minimization::ProjectableProblem<typename TTraits::ValueType>::TVector solution(problem.Dimensions());
                    Minimization::PQNMinimize<m>(problem, solution, maxNumIt, breakEps, 50, true, 16);
                }

                // Now optimize for the true loss
                RegressionTreeFieldProblem<TTraits, TLossTag, Subsample> problem(Us, Ps, Ls, traindb, residualTolCG, maxNumItCG);

                // The below code is useful to check the derivative with respect to the model parameters.
                // Such checks should be performed, for instance, when implementing a new loss function.
#if 0
                Minimization::CheckDerivative<double>(problem, 1e-4, 1000, 1e-6, 1e-8);
                Sleep(10000);
#endif
                typename Minimization::ProjectableProblem<typename TTraits::ValueType>::TVector solution(problem.Dimensions());

                // Find the optimal model parameters using the specified optimization algorithm.
#if defined(LEARNING_USE_PQN)
                return Minimization::PQNMinimize<m>(problem, solution, maxNumIt, breakEps, 50, true, 16);
#elif defined(LEARNING_USE_SPG)
                return Minimization::SPGMinimize(problem, solution, maxNumIt, breakEps, true, false, 16);
#else
                return Minimization::RestartingLBFGSMinimize<m>(problem, solution, maxNumIt, breakEps, true, 4);
#endif
            }
        };

        // Specialization for negative log-pseudolikelihood
        template <typename TTraits, bool Subsample, size_t m>
        struct WeightOptimization<TTraits, NLPLLoss, Subsample, m>
        {
            static typename TTraits::ValueType
            Run(typename TTraits::UnaryFactorTypeVector& Us,
                typename TTraits::PairwiseFactorTypeVector& Ps,
                typename TTraits::LinearOperatorVector& Ls,
                const typename TTraits::DataSampler& traindb,
                typename TTraits::ValueType breakEps = 5e-2,
                size_t maxNumIt = 5000,
                bool stagedTraining = false,
                typename TTraits::ValueType residualTolCG = 1e-4,
                size_t maxNumItCG = 10000)
            {
                RegressionTreeFieldProblem<TTraits, NLPLLoss, Subsample> problem(Us, Ps, traindb);
                typename Minimization::ProjectableProblem<typename TTraits::ValueType>::TVector solution(problem.Dimensions());
                return Minimization::RestartingLBFGSMinimize<m>(problem, solution, maxNumIt, breakEps, true, 5);
            }
        };

        template<typename TValue>
        struct FactorTypeInfo
        {
            TValue                 purityEpsilon;
            int                    nFeatureCount;
            int                    nDepthLevels;
            int                    nMinDataPointsForSplitConsideration;

            FactorTypeInfo()
            {
            }

            FactorTypeInfo(int nFeatureCount_, int nDepthLevels_, int nMinDataPointsForSplitConsideration_, TValue purityEpsilon_)
                : purityEpsilon(purityEpsilon_), nFeatureCount(nFeatureCount_), nDepthLevels(nDepthLevels_), nMinDataPointsForSplitConsideration(nMinDataPointsForSplitConsideration_)
            {
            }
        };

        // Adds one additional level to each tree, unless the maximum depth of that type is exceeded.
        // The splitting criterion specified in the model traits is used for that purpose.
        template<typename TTraits, typename TLossTag, bool Subsample>
        int
        GrowTreesByOneLevel(int level,
                            const typename TTraits::DataSampler& traindb, typename TTraits::FeatureSampler& featureSampler,
                            typename TTraits::UnaryTreeRefVector& Uts, typename TTraits::UnaryFactorTypeVector& Us,
                            const std::vector<FactorTypeInfo<typename TTraits::ValueType>>& UInfos,
                            typename TTraits::PairwiseTreeRefVector& Pts, typename TTraits::PairwiseFactorTypeVector& Ps,
                            const std::vector<FactorTypeInfo<typename TTraits::ValueType>>& PInfos,
                            typename TTraits::LinearOperatorVector& Ls, typename TTraits::LinearOperatorTreeRefVector& Lts,
                            size_t maxNumItCG = 10000, typename TTraits::ValueType residualTolCG = 1e-4)
        {
            typedef typename TTraits::Monitor TMonitor;
            TMonitor::Report("  Computing mean parameters.\n");
            auto meanParameters = MeanParameterDispatcher < TTraits, TLossTag, Subsample,
                 typename TTraits::UnarySplitCriterionTag,
                 typename TTraits::PairwiseSplitCriterionTag >::Compute(Us, Ps, Ls, traindb, residualTolCG, maxNumItCG);
            TMonitor::Report("  Done, got %lu mean parameters.\n", meanParameters.size());
            int nGrown = 0;
            std::vector<typename TTraits::Feature> features;

            for(size_t u = 0; u < Uts.size(); ++u)
            {
                features.resize(UInfos[u].nFeatureCount);
                std::generate(features.begin(), features.end(), [&]()
                {
                    return featureSampler(level);
                } );
                TMonitor::Report("  Growing unary tree no. %u, maximum depth %u\n", u, UInfos[u].nDepthLevels);
                auto grown = Training::GrowRegressionTree<TTraits::UseExplicitThresholding, typename TTraits::Monitor>(
                                 Uts[u],
                                 UnaryPointSamplerDispatcher < TTraits, TLossTag, typename TTraits::UnarySplitCriterionTag,
                                 Subsample >::Instantiate(traindb, Us, Ps, Ls, Us[u], meanParameters, maxNumItCG, residualTolCG),
                                 features,
                                 UInfos[u].nDepthLevels,
                                 UInfos[u].nMinDataPointsForSplitConsideration,
                                 Detail::CriterionDispatcher < typename TTraits::UnarySplitCriterionTag,
                                 typename TTraits::UnarySplitCriterion,
                                 typename TTraits::UnaryFactorType >::Instantiate(Us[u], UInfos[u].purityEpsilon));
                if( grown )
                {
                    TMonitor::Report("  Done, merging new leaves.\n");
                    Us[u].MergeTree(Uts[u]);
                    nGrown += grown;
                }
                else
                {
                    TMonitor::Report("  Done, tree has reached maximum depth.\n");
                }
            }

            for(size_t p = 0; p < Pts.size(); ++p)
            {
                features.resize(PInfos[p].nFeatureCount);
                std::generate(features.begin(), features.end(), [&]()
                {
                    return featureSampler(level);
                });
                TMonitor::Report("  Growing pairwise tree no. %u, maximum depth %u\n", p, PInfos[p].nDepthLevels);
                auto grown = Training::GrowRegressionTree<TTraits::UseExplicitThresholding, typename TTraits::Monitor>(
                                 Pts[p],
                                 PairwisePointSamplerDispatcher < TTraits, TLossTag, typename TTraits::PairwiseSplitCriterionTag,
                                 Subsample >::Instantiate(traindb, Us, Ps, Ls, Ps[p], meanParameters, maxNumItCG, residualTolCG),
                                 features,
                                 PInfos[p].nDepthLevels,
                                 PInfos[p].nMinDataPointsForSplitConsideration,
                                 Detail::CriterionDispatcher < typename TTraits::PairwiseSplitCriterionTag,
                                 typename TTraits::PairwiseSplitCriterion,
                                 typename TTraits::PairwiseFactorType >::Instantiate(Ps[p], PInfos[p].purityEpsilon));
                if( grown )
                {
                    TMonitor::Report("  Done, merging new leaves.\n");
                    Ps[p].MergeTree(Pts[p]);
                    nGrown += grown;
                }
                else
                {
                    TMonitor::Report("  Done, tree has reached maximum depth.\n");
                }
            }

            for(size_t l = 0; l < Lts.size(); ++l)
            {
                const auto info = Ls[l].GetInfo();
                if( info.hasTree )
                {
                    features.resize(info.nFeatureCount);
                    std::generate(features.begin(), features.end(), [&]()
                    {
                        return featureSampler(level);
                    });
                    TMonitor::Report("  Growing linear operator tree no. %u, maximum depth %u\n", l, info.nDepthLevels);
                    auto grown = Training::GrowRegressionTree<TTraits::UseExplicitThresholding, typename TTraits::Monitor>(
                                     Lts[l],
                                     LinearOperatorPointSampler < TTraits, TLossTag,
                                     Subsample,
                                     typename TTraits::LinearOperatorWeights >(traindb, Ls[l], meanParameters),
                                     features,
                                     info.nDepthLevels,
                                     info.nMinDataPointsForSplitConsideration,
                                     Criteria::GradientNormCriterion<typename TTraits::LinearOperatorWeights>(
                                         info.purityEpsilon,
                                         Ls[l].WeightsInBreadthFirstOrder(),
                                         info.smallestEigenvalue,
                                         info.largestEigenvalue)
                                 );
                    if( grown )
                    {
                        TMonitor::Report("  Done, merging new leaves.\n");
                        Ls[l].MergeTree(Lts[l]);
                        nGrown += grown;
                    }
                    else
                    {
                        TMonitor::Report("  Done, tree has reached maximum depth.\n");
                    }
                }
            }

            return nGrown;
        }

        template <typename TTraits, typename TLossTag, bool Subsample, size_t m>
        void
        LearnTreesAndWeightsJointly(typename TTraits::UnaryFactorTypeVector& Us,
                                    const std::vector<FactorTypeInfo<typename TTraits::ValueType>>& UInfos,
                                    typename TTraits::PairwiseFactorTypeVector& Ps,
                                    const std::vector<FactorTypeInfo<typename TTraits::ValueType>>& PInfos,
                                    typename TTraits::LinearOperatorVector& Ls,
                                    const typename TTraits::DataSampler& traindb,
                                    size_t maxNumOptimItPerRound = 50,
                                    size_t maxNumOptimItFinal = 50,
                                    typename TTraits::ValueType finalBreakEps = 1e-2,
                                    bool stagedTraining = false,
                                    size_t maxNumItCG = 10000,
                                    typename TTraits::ValueType residualTolCG = 1e-4)
        {
            typedef typename TTraits::Monitor TMonitor;
            typename TTraits::UnaryTreeRefVector Uts(Us.size());
            typename TTraits::PairwiseTreeRefVector Pts(Ps.size());
            typename TTraits::LinearOperatorTreeRefVector Lts(Ls.size());
            typename TTraits::FeatureSampler featureSampler;
            TMonitor::Report("Starting joint training of trees and weights.\n");
            typename TTraits::ValueType prevObj = 0.0;
            int nGrown, level = 1;

            do
            {
                TMonitor::Report("Processing level %d.\n", level);

                if(level == 1)
                    TMonitor::Report("  Optimizing weights.\n");
                else
                    TMonitor::Report("  Re-optimizing weights (previous objective was %.5f).\n", prevObj);

                prevObj = WeightOptimization<TTraits, TLossTag, Subsample, m>::Run(Us, Ps, Ls, traindb, finalBreakEps, maxNumOptimItPerRound,
                          stagedTraining, residualTolCG, maxNumItCG);
                TMonitor::Report("  Splitting leaves at level %d.\n", level);
                nGrown = GrowTreesByOneLevel<TTraits, TLossTag, Subsample>(level++, traindb, featureSampler, Uts, Us, UInfos, Pts, Ps, PInfos, Ls, Lts,
                         maxNumItCG, residualTolCG);
            }
            while(nGrown > 0);

            TMonitor::Report("Commencing final optimization of weights.\n");
            WeightOptimization<TTraits, TLossTag, Subsample, m>::Run(Us, Ps, Ls, traindb, finalBreakEps, maxNumOptimItFinal,
                    stagedTraining, residualTolCG, maxNumItCG);
            TMonitor::Report("Joint training of trees and weights finished.\n");
        }
    }

    // ===========================
    // LearnTreesAndWeightsJointly
    // ===========================
    //
    // Learns regression trees for the specified factor types and optimizes their weights in an intermingled fashion.
    // If the split criteria are set to GradientNormCriterion in the model traits, the split decisions are based on
    // the actual increase in the norm of the projected negative log-pseudolikelihood gradient incurred by adding
    // the new leaves. However, it is also possible to specify any other split criterion in the model traits.
    //
    // The algorithm works in rounds. At each round,
    //
    //   a) The weights of the existing tree nodes are (re-)optimized. Initially, each tree consists solely of its root.
    //
    //   b) One level of nodes is added to each tree by means of the selected split criterion.
    //      If GradientNormCriterion is selected for either unary or pairwise trees, this involves:
    //        1) Computation of the current mean parameters (mu, Sigma) of each conditioned subgraph in the training data.
    //        2) Computation of the gradient contributions of each factor instance by means of the pre-computed mean parameters.
    //      Subsequently, splitting of the leaf nodes is procured as usual, using the routines of Training.h.
    //
    //   c) For each factor type, the new leaves of the corresponding tree are merged into the model tree as new Weights nodes,
    //      the parameters of which are set to those of their parents. As a result, the objective value remains unaffected
    //      by this operation. If the newly added Weight nodes were chosen according to GradientNormCriterion, it is expected
    //      that the objective can be decreased significantly during the next round of optimization.
    //
    //   d) Initiate next round, go to a).
    //
    // Since the addition of new Weights nodes leaves the objective unaffected, this scheme achieves monotonic descent in the
    // the negative log-pseudolikelihood. This property holds both for GradientNormCriterion and any other split criterion.
    //
    // The accuracy of the mean parameters computed in step b) depends on the optimization tolerance in step a). If the problem
    // is solved to optimality, the mean parameters will be exact, yielding the exact gradient with respect to the new candidate
    // Weight nodes for the other weights set to their optimum. However, in general, this is wasteful, and the gradient norm
    // of the candidate Weight nodes will be almost equally informative if the existing Weights lie in the vicinity of the optimum.
    // Hence, (re-)optimization of the weights in step a) is usually truncated after a finite number of iterations.
    //
    // To ensure a solution of high quality, one final optimization step is conducted subsequent to the greedy construction
    // scheme outlined above. There, it is advisable to use the gradient norm as the stopping condition, as one would normally
    // do when optimizing the weights.
    //
    // This function takes the following template parameters:
    //
    //  - TTraits
    //             The model traits class of your application.
    //
    //  - Subsample
    //             Determines whether training should subsample from the points contained in the training set.
    //             Note that your dataset class must implement the subsampling interface if this option is turned on,
    //             otherwise you will experience a compilation error.
    //
    //  - m        (Recommended value: 5 - 10)
    //             Number of previous iterates and gradients employed by LBFGS. Setting m to a large number can result
    //             in significant memory requirements if your model has many weights. On the other hand, a larger value
    //             can speed up convergence.
    //
    // This function takes the following arguments:
    //
    //  - Us       A vector containing the unary factor types to be learned/optimized. The instances must have been created
    //             using Learning::MakeUnaryFactorType(), which ensures that a single Weights node (the root of the tree)
    //             is present. Note that any options regarding the factor type (eigenvalue restrictions, etc.) must be passed
    //             in during this step.
    //
    //  - nUnaryFeatureCount
    //             The number of features to be sampled during growing of the unary trees.
    //
    //  - nUnaryDepthLevels
    //             The maximum depth to which unary trees will be grown.
    //
    //  - nUnaryMinDataPointsForSplitConsideration
    //             The minimum number of points that must go into the leaf of a unary tree for it to be split even further.
    //
    //  - unaryPurityEpsilon
    //             Split criterion-specific small number that determines whether a unary tree node is considered pure.
    //
    //  - Ps       Same as Us, but for the pairwise factor types. Use Learning::MakePairwiseFactorType() to construct the
    //             factor type instances.
    //
    //  - nPairwiseFeatureCount
    //             The number of features to be sampled during growing of the pairwise trees.
    //
    //  - nPairwiseDepthLevels
    //             The maximum depth to which pairwise trees will be grown.
    //
    //  - nPairwiseMinDataPointsForSplitConsideration
    //             The minimum number of points that must go into the leaf of a pairwise tree for it to be split even further.
    //
    //  - pairwisePurityEpsilon
    //             Split criterion-specific small number that determines whether a pairwise tree node is considered pure.
    //
    //  - traindb
    //             The dataset on which the trees/weights are to be trained.
    //
    //  - maxNumOptimItPerRound
    //             The number of LBFGS iterations that are performed at each round for (re-)optimization of the weights.
    //             (Re-)optimization is also cancelled if the norm of the projected gradient drops below finalBreakEps.
    //
    //  - maxNumOptimItFinal
    //             Maximum number of LBFGS iterations that are performed after greedy construction of the trees for
    //             final optimization of the weights.
    //             Final optimization is also stopped if the norm of the projected gradient drops below finalBreakEps.
    //
    //  - finalBreakEps
    //             The (re-)optimization process at each round is stopped if the norm of the projected gradient drops
    //             below this number.
    //             Moreover, final optimization of the weights is stopped based on this criterion. You can think of it
    //             as specifying the accuracy to which the weights of your model are ultimately trained.
    //
    template <typename TTraits, bool Subsample, size_t m>
    void
    LearnTreesAndWeightsJointly(typename TTraits::UnaryFactorTypeVector& Us,
                                int nUnaryFeatureCount,
                                int nUnaryDepthLevels,
                                int nUnaryMinDataPointsForSplitConsideration,
                                typename TTraits::ValueType unaryPurityEpsilon,
                                typename TTraits::PairwiseFactorTypeVector& Ps,
                                int nPairwiseFeatureCount,
                                int nPairwiseDepthLevels,
                                int nPairwiseMinDataPointsForSplitConsideration,
                                typename TTraits::ValueType pairwisePurityEpsilon,
                                const typename TTraits::DataSampler& traindb,
                                size_t maxNumOptimItPerRound = 50,
                                size_t maxNumOptimItFinal = 50,
                                typename TTraits::ValueType finalBreakEps = 1e-2)
    {
        typedef Detail::FactorTypeInfo<typename TTraits::ValueType> TInfo;
        std::vector<TInfo> UInfos(Us.size());
        std::fill(UInfos.begin(), UInfos.end(), TInfo(nUnaryFeatureCount, nUnaryDepthLevels,
                  nUnaryMinDataPointsForSplitConsideration, unaryPurityEpsilon));
        std::vector<TInfo> PInfos(Ps.size());
        std::fill(PInfos.begin(), PInfos.end(), TInfo(nPairwiseFeatureCount, nPairwiseDepthLevels,
                  nPairwiseMinDataPointsForSplitConsideration, pairwisePurityEpsilon));
        typename TTraits::LinearOperatorVector Ls;
        Detail::LearnTreesAndWeightsJointly < TTraits,
               Detail::NLPLLoss,
               Subsample, m > (Us, UInfos, Ps, PInfos, Ls,
                               traindb, maxNumOptimItPerRound, maxNumOptimItFinal, finalBreakEps);
    }

    template <typename TTraits, bool Subsample, size_t m>
    void
    LearnTreesAndWeightsJointly(typename TTraits::UnaryFactorTypeVector& Us,
                                const std::vector<Detail::FactorTypeInfo<typename TTraits::ValueType>>& UInfos,
                                typename TTraits::PairwiseFactorTypeVector& Ps,
                                const std::vector<Detail::FactorTypeInfo<typename TTraits::ValueType>>& PInfos,
                                const typename TTraits::DataSampler& traindb,
                                size_t maxNumOptimItPerRound = 50,
                                size_t maxNumOptimItFinal = 50,
                                typename TTraits::ValueType finalBreakEps = 1e-2)
    {
        typename TTraits::LinearOperatorVector Ls;
        Detail::LearnTreesAndWeightsJointly < TTraits,
               Detail::NLPLLoss,
               Subsample, m > (Us, UInfos, Ps, PInfos, Ls, traindb,
                               maxNumOptimItPerRound, maxNumOptimItFinal, finalBreakEps);
    }

    template <typename TTraits, typename TLossTag, bool Subsample, size_t m>
    void
    LearnTreesAndWeightsJointlyDiscriminative(typename TTraits::UnaryFactorTypeVector& Us,
            int nUnaryFeatureCount,
            int nUnaryDepthLevels,
            int nUnaryMinDataPointsForSplitConsideration,
            typename TTraits::ValueType unaryPurityEpsilon,
            typename TTraits::PairwiseFactorTypeVector& Ps,
            int nPairwiseFeatureCount,
            int nPairwiseDepthLevels,
            int nPairwiseMinDataPointsForSplitConsideration,
            typename TTraits::ValueType pairwisePurityEpsilon,
            const typename TTraits::DataSampler& traindb,
            size_t maxNumOptimItPerRound = 50,
            size_t maxNumOptimItFinal = 50,
            typename TTraits::ValueType finalBreakEps = 1e-2,
            bool stagedTraining = false,
            size_t maxNumCGIt = 10000,
            typename TTraits::ValueType residualTolCG = 1e-4)
    {
        typedef Detail::FactorTypeInfo<typename TTraits::ValueType> TInfo;
        std::vector<TInfo> UInfos(Us.size());
        std::fill(UInfos.begin(), UInfos.end(), TInfo(nUnaryFeatureCount, nUnaryDepthLevels,
                  nUnaryMinDataPointsForSplitConsideration, unaryPurityEpsilon));
        std::vector<TInfo> PInfos(Ps.size());
        std::fill(PInfos.begin(), PInfos.end(), TInfo(nPairwiseFeatureCount, nPairwiseDepthLevels,
                  nPairwiseMinDataPointsForSplitConsideration, pairwisePurityEpsilon));
        typename TTraits::LinearOperatorVector Ls;
        Detail::LearnTreesAndWeightsJointly < TTraits,
               TLossTag,
               Subsample, m > (Us, UInfos, Ps, PInfos, Ls,
                               traindb, maxNumOptimItPerRound, maxNumOptimItFinal, finalBreakEps, stagedTraining,
                               maxNumCGIt, residualTolCG);
    }

    template <typename TTraits, typename TLossTag, bool Subsample, size_t m>
    void
    LearnTreesAndWeightsJointlyDiscriminative(typename TTraits::UnaryFactorTypeVector& Us,
            const std::vector<Detail::FactorTypeInfo<typename TTraits::ValueType>>& UInfos,
            typename TTraits::PairwiseFactorTypeVector& Ps,
            const std::vector<Detail::FactorTypeInfo<typename TTraits::ValueType>>& PInfos,
            typename TTraits::LinearOperatorVector& Ls,
            const typename TTraits::DataSampler& traindb,
            size_t maxNumOptimItPerRound = 50,
            size_t maxNumOptimItFinal = 50,
            typename TTraits::ValueType finalBreakEps = 1e-2,
            bool stagedTraining = false,
            size_t maxNumCGIt = 10000,
            typename TTraits::ValueType residualTolCG = 1e-4)
    {
        Detail::LearnTreesAndWeightsJointly < TTraits,
               TLossTag,
               Subsample, m > (Us, UInfos, Ps, PInfos, Ls, traindb,
                               maxNumOptimItPerRound, maxNumOptimItFinal, finalBreakEps, stagedTraining,
                               maxNumCGIt, residualTolCG);
    }
}

#endif // _H_LEARNING_H
