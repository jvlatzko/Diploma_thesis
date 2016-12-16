/* This file is part of the "Regression Tree Fields" (RTF) source code distribution,
 * obtained from http://research.microsoft.com/downloads.
 * It is provided to you under the terms of the Microsoft Research License Agreement
 * (MSR-LA). Please see License.txt for details.
 *
 *
 * File: Types.h
 * Defines various type definitions and declarations that are used by the RTF code.
 *
 */

#ifndef H_RTF_TYPES_H
#define H_RTF_TYPES_H

#include <vector>
#include <Eigen/Eigen>

template <typename T, unsigned int nBands>
class ImageRef;

template <typename T, unsigned int nBands>
class ImageRefC;

template <typename T>
class VecCRef;

template <typename T>
struct Vector2D;

template <typename TFeature, typename TNode> struct NodeData;
template <typename TNode> class tree_node_;

template <typename TFeature, typename TNode, typename TAllocator>
class TreeRef;

template <typename TFeature, typename TNode, typename TAllocator>
class TreeCRef;

namespace Training
{
    template <typename TLabel>
    class RegressionTreeNode;

    template<typename TValue, size_t Dim>
    class LabelVector;
}

namespace LinearOperator
{
    template<typename TValue>
    class DefaultWeights;
}

namespace Compute
{
    template< typename TFeature, typename TLabel, typename TPrior, typename TBasis> class FactorTypeBase;
    template< typename TValue, size_t VarDim> class Factor;
    template< typename TValue, size_t VarDim> class ConnectedFactor;
    template< typename TValue, size_t VarDim > class ConditionedFactor;
    template< typename TValue, size_t VarDim, size_t BasisDim > class Weights;
    template< typename TTraits > class PrecomputedSubgraph;
    template< typename TValue, size_t VarDim > class SystemVectorRef;
    template< typename TValue, size_t VarDim > class BlockDiagonalRef;
    template< typename TValue, size_t VarDim > class SystemVectorCRef;
    template< typename TValue, size_t VarDim > class SystemMatrixRow;
}

namespace LinearOperator
{
    template< typename TFeature, typename TUnaryGroundLabel, typename TWeights > class OperatorRef;
}

namespace Criteria
{
    template <typename TLabel> class SquaredResidualsCriterion;
    template <typename TLabel> class MSTDiffEntropyCriterion;
    template <typename TLabel> class GradientNormCriterion;
}
class SquaredResidualsCriterion;
class MSTDiffEntropyCriterion;
class GradientNormCriterion;

template <typename TCriterionTag, typename TLabel, typename TBasis>
class CriterionType;
template <typename TLabel, typename TBasis>
struct CriterionType<SquaredResidualsCriterion, TLabel, TBasis>
{
    typedef Criteria::SquaredResidualsCriterion<TLabel> Type;
};
template <typename TLabel, typename TBasis>
struct CriterionType<MSTDiffEntropyCriterion, TLabel, TBasis>
{
    typedef Criteria::MSTDiffEntropyCriterion<TLabel> Type;
};
template <typename TLabel, typename TBasis>
struct CriterionType<GradientNormCriterion, TLabel, TBasis>
{
    typedef Criteria::GradientNormCriterion<Compute::Weights<typename TLabel::ValueType, TLabel::Size, TBasis::Size>> Type;
};

namespace Priors
{
    template <typename TValue, size_t VarDim, size_t BasisDim> class NullPrior;
    template <typename TValue, size_t VarDim, size_t BasisDim> class LargestEigenvaluePrior;
    template <typename TValue, size_t VarDim, size_t BasisDim> class SumOfEigenvaluesPrior;
    template <typename TValue, size_t VarDim, size_t BasisDim> class SpreadOfEigenvaluesPrior;
    template <typename TValue, size_t VarDim, size_t BasisDim> class FrobeniusPrior;
    template <typename TValue, size_t VarDim, size_t BasisDim> class ConjugatePrior;
}
class NullPrior;
class LargestEigenvaluePrior;
class SumOfEigenvaluesPrior;
class SpreadOfEigenvaluesPrior;
class FrobeniusPrior;
class ConjugatePrior;

template <typename TPriorTag, typename TValue, size_t VarDim, size_t BasisDim>
class PriorType;
template <typename TValue, size_t VarDim, size_t BasisDim>
struct PriorType<NullPrior, TValue, VarDim, BasisDim>
{
    typedef Priors::NullPrior<TValue, VarDim, BasisDim> Type;
};
template <typename TValue, size_t VarDim, size_t BasisDim>
struct PriorType<LargestEigenvaluePrior, TValue, VarDim, BasisDim>
{
    typedef Priors::LargestEigenvaluePrior<TValue, VarDim, BasisDim> Type;
};
template <typename TValue, size_t VarDim, size_t BasisDim>
struct PriorType<SumOfEigenvaluesPrior, TValue, VarDim, BasisDim>
{
    typedef Priors::SumOfEigenvaluesPrior<TValue, VarDim, BasisDim> Type;
};
template <typename TValue, size_t VarDim, size_t BasisDim>
struct PriorType<SpreadOfEigenvaluesPrior, TValue, VarDim, BasisDim>
{
    typedef Priors::SpreadOfEigenvaluesPrior<TValue, VarDim, BasisDim> Type;
};
template <typename TValue, size_t VarDim, size_t BasisDim>
struct PriorType<FrobeniusPrior, TValue, VarDim, BasisDim>
{
    typedef Priors::FrobeniusPrior<TValue, VarDim, BasisDim> Type;
};
template <typename TValue, size_t VarDim, size_t BasisDim>
struct PriorType<ConjugatePrior, TValue, VarDim, BasisDim>
{
    typedef Priors::ConjugatePrior<TValue, VarDim, BasisDim> Type;
};

namespace Unary
{
    template< typename TFeature, typename TLabel, typename TPrior, typename TBasis> class FactorType;
}

namespace Pairwise
{
    template< typename TFeature, typename TLabel, typename TPrior, typename TBasis> class FactorType;
}

namespace Monitor
{
    class DefaultMonitor;
    class NullMonitor;
}

namespace Classify
{
    namespace Detail
    {
        template< typename TTraits, typename TErrorTerm, bool instantiate > class LinearSystem;
        template< typename TTraits, typename TErrorTerm > class OnTheFlySystem;
        template< typename TTraits, typename TErrorTerm> class ConstrainedQuadratic;
        template<typename TTraits, typename TErrorTerm>
        class UnconstrainedQuadratic;
    }
    template <typename TTraits, typename TErrorTerm> struct LinearSystem;

    template <typename TTraits>
    Eigen::Matrix<typename TTraits::ValueType, Eigen::Dynamic, 1>
    SolutionFromLabeling(const ImageRefC<typename TTraits::UnaryGroundLabel, 1u>& labeling);
}

namespace Detail
{
    template <typename TFeature, typename TValue, bool UseBasis = false>
    struct UnaryBasis
    {
        typedef TValue ValueType;
        static const size_t Size = 1;

        static TValue* Compute(int x, int y, const typename TFeature::PreProcessType& prep, const VecCRef<Vector2D<int>>& offsets, TValue* basis)
        {
            *basis = TValue(1);
            return basis;
        }

        static TValue ComputeQuadratic(const typename TFeature::PreProcessType& prep, const Vector2D<int>& i, int basisIndex)
        {
            return 1.0;
        }
    };

    template <typename TFeature, typename TValue>
    struct UnaryBasis<TFeature, TValue, true>
    {
        typedef TValue ValueType;
        static const size_t Size = TFeature::UnaryBasisSize;

        static TValue* Compute(int x, int y, const typename TFeature::PreProcessType& prep, const VecCRef<Vector2D<int>>& offsets, TValue* basis)
        {
            TFeature::ComputeBasis(x, y, prep, offsets, basis);
            return basis;
        }

        static TValue ComputeQuadratic(const typename TFeature::PreProcessType& prep, const Vector2D<int>& i, int basisIndex)
        {
            if( basisIndex < 0 )
                return 1.0;
            else
                return TFeature::ComputeQuadraticBasis(prep, i, basisIndex);
        }
    };

    template <typename TFeature, typename TValue, bool UseBasis = false>
    struct PairwiseBasis
    {
        typedef TValue ValueType;
        static const size_t Size = 1;

        static TValue* Compute(int x, int y, const typename TFeature::PreProcessType& prep, const VecCRef<Vector2D<int>>& offsets, TValue* basis)
        {
            *basis = TValue(1);
            return basis;
        }

        static TValue ComputeQuadratic(const typename TFeature::PreProcessType& prep, const Vector2D<int>& i, const Vector2D<int>& j, int basisIndex)
        {
            return 1.0;
        }
    };

    template <typename TFeature, typename TValue>
    struct PairwiseBasis<TFeature, TValue, true>
    {
        typedef TValue ValueType;
        static const size_t Size = TFeature::PairwiseBasisSize;

        static TValue* Compute(int x, int y, const typename TFeature::PreProcessType& prep, const VecCRef<Vector2D<int>>& offsets, TValue* basis)
        {
            TFeature::ComputeBasis(x, y, prep, offsets, basis);
            return basis;
        }

        static TValue ComputeQuadratic(const typename TFeature::PreProcessType& prep, const Vector2D<int>& i, const Vector2D<int>& j, int basisIndex)
        {
            if( basisIndex < 0 )
                return 1.0;
            else
                return TFeature::ComputeQuadraticBasis(prep, i, j, basisIndex);
        }
    };
}

template<typename TFeature, typename TLabel, typename TBasis>
struct Traits_
{
    typedef Compute::Weights<typename TLabel::ValueType, TLabel::Size, TBasis::Size> Weights;

    typedef std::allocator<tree_node_<NodeData<TFeature, Training::RegressionTreeNode<TLabel> > > > RegressionTreeNodeAllocator;
    typedef Eigen::aligned_allocator<tree_node_<NodeData<TFeature, Weights> > > WeightsNodeAllocator;

    typedef TreeRef<TFeature, Training::RegressionTreeNode<TLabel>, RegressionTreeNodeAllocator> RegressionTreeRef;
    typedef TreeCRef<TFeature, Training::RegressionTreeNode<TLabel>, RegressionTreeNodeAllocator> RegressionTreeCRef;

    typedef TreeRef<TFeature, Weights, WeightsNodeAllocator> ModelTreeRef;
    typedef TreeCRef<TFeature, Weights, WeightsNodeAllocator> ModelTreeCRef;
};

enum CachingType { WEIGHTS_AND_BASIS_PRECOMPUTED,
                   WEIGHTS_AND_BASIS_AND_MATRIX_PRECOMPUTED,
                   ON_THE_FLY
                 };

template < typename TFeatureSampler, typename TDataSampler,
         typename TUSplitCritTag = SquaredResidualsCriterion, typename TPSplitCritTag = SquaredResidualsCriterion,
         typename TUPriorTag = NullPrior, typename TPPriorTag = NullPrior,
         bool UseBasis = false,
         bool UseExplicitThresholdTesting = false,
         typename TMonitor = Monitor::DefaultMonitor,
         int TCachingMode = WEIGHTS_AND_BASIS_PRECOMPUTED,
         typename TCustomOperatorWeights = LinearOperator::DefaultWeights<typename TDataSampler::UnaryGroundLabel::ValueType>
         >
struct Traits
{
    static const bool UseExplicitThresholding = UseExplicitThresholdTesting;
    static const bool UseBasisFunctions       = UseBasis;
    static const int  CachingMode             = TCachingMode;

    typedef TUPriorTag UnaryPriorTag;
    typedef TPPriorTag PairwisePriorTag;

    typedef TFeatureSampler FeatureSampler;
    typedef typename FeatureSampler::TFeature Feature;
    typedef typename Feature::PreProcessType PreProcessType;
    typedef TDataSampler DataSampler;

    typedef typename TDataSampler::UnaryGroundLabel UnaryGroundLabel;
    typedef typename TDataSampler::PairwiseGroundLabel PairwiseGroundLabel;
    typedef typename TDataSampler::InputLabel InputLabel;

    typedef typename TDataSampler::UnaryGroundLabel::ValueType ValueType;

    typedef Detail::UnaryBasis<Feature, typename UnaryGroundLabel::ValueType, UseBasis> UnaryBasis;
    typedef Detail::PairwiseBasis<Feature, typename PairwiseGroundLabel::ValueType, UseBasis> PairwiseBasis;

    typedef typename PriorType<TUPriorTag, typename UnaryGroundLabel::ValueType, UnaryGroundLabel::Size, UnaryBasis::Size>::Type UnaryPrior;
    typedef typename PriorType<TPPriorTag, typename PairwiseGroundLabel::ValueType, PairwiseGroundLabel::Size, PairwiseBasis::Size>::Type PairwisePrior;

    typedef typename Unary::FactorType<Feature, UnaryGroundLabel, UnaryPrior, UnaryBasis> UnaryFactorType;
    typedef typename std::vector<UnaryFactorType> UnaryFactorTypeVector;

    typedef typename Pairwise::FactorType<Feature, PairwiseGroundLabel, PairwisePrior, PairwiseBasis> PairwiseFactorType;
    typedef typename std::vector<PairwiseFactorType> PairwiseFactorTypeVector;

    typedef typename CriterionType<TUSplitCritTag, UnaryGroundLabel, UnaryBasis>::Type UnarySplitCriterion;
    typedef TUSplitCritTag UnarySplitCriterionTag;
    typedef typename CriterionType<TPSplitCritTag, PairwiseGroundLabel, PairwiseBasis>::Type PairwiseSplitCriterion;
    typedef TPSplitCritTag PairwiseSplitCriterionTag;

    // Note: It is important that UnarySplitCriterion::TLabel is used here (rather than UnaryGroundLabel), since
    // the use of GradientNormCriterion mandates a specific label type. This here _exclusively_ concerns regression
    // tree types, since the node data of model trees is not affected by the split criterion.
    typedef typename Traits_<Feature, typename UnarySplitCriterion::TLabel, UnaryBasis>::RegressionTreeRef UnaryTreeRef;
    typedef typename Traits_<Feature, typename UnarySplitCriterion::TLabel, UnaryBasis>::RegressionTreeCRef UnaryTreeCRef;
    typedef typename std::vector<UnaryTreeRef> UnaryTreeRefVector;

    // Note: The above caveats hold equally for the use of PairwiseSplitCriterion::TLabel.
    typedef typename Traits_<Feature, typename PairwiseSplitCriterion::TLabel, PairwiseBasis>::RegressionTreeRef PairwiseTreeRef;
    typedef typename Traits_<Feature, typename PairwiseSplitCriterion::TLabel, PairwiseBasis>::RegressionTreeCRef PairwiseTreeCRef;
    typedef typename std::vector<PairwiseTreeRef> PairwiseTreeRefVector;

    typedef typename Training::LabelVector<ValueType, TCustomOperatorWeights::NumCoefficients> LinearOperatorLabel;
    typedef std::allocator<tree_node_<NodeData<Feature, Training::RegressionTreeNode<LinearOperatorLabel> > > > LOPRegressionTreeNodeAllocator;
    typedef Eigen::aligned_allocator<tree_node_<NodeData<Feature, TCustomOperatorWeights> > > LOPWeightsNodeAllocator;

    typedef TreeRef<Feature, Training::RegressionTreeNode<LinearOperatorLabel>, LOPRegressionTreeNodeAllocator> LinearOperatorTreeRef;
    typedef TreeCRef<Feature, Training::RegressionTreeNode<LinearOperatorLabel>, LOPRegressionTreeNodeAllocator> LinearOperatorTreeCRef;
    typedef typename std::vector<LinearOperatorTreeRef> LinearOperatorTreeRefVector;

    typedef TreeRef<Feature, TCustomOperatorWeights, LOPWeightsNodeAllocator> LinearOperatorModelTreeRef;
    typedef TreeCRef<Feature, TCustomOperatorWeights, LOPWeightsNodeAllocator> LinearOperatorModelTreeCRef;

    typedef Compute::Weights<typename UnaryGroundLabel::ValueType, UnaryGroundLabel::Size, UnaryBasis::Size> UnaryWeights;
    typedef Compute::Weights<typename PairwiseGroundLabel::ValueType, PairwiseGroundLabel::Size, PairwiseBasis::Size> PairwiseWeights;
    typedef TCustomOperatorWeights LinearOperatorWeights;

    typedef ImageRef<UnaryWeights*, 1u> UnaryWeightsImageRef;
    typedef ImageRefC<UnaryWeights*, 1u> UnaryWeightsImageRefC;
    typedef std::vector<UnaryWeightsImageRef> UnaryWeightsImageVector;

    typedef ImageRef<PairwiseWeights*, 1u> PairwiseWeightsImageRef;
    typedef ImageRefC<PairwiseWeights*, 1u> PairwiseWeightsImageRefC;
    typedef std::vector<PairwiseWeightsImageRef> PairwiseWeightsImageVector;

    typedef ImageRef<typename UnaryBasis::ValueType, UnaryBasis::Size> UnaryBasisImageRef;
    typedef ImageRefC<typename UnaryBasis::ValueType, UnaryBasis::Size> UnaryBasisImageRefC;
    typedef std::vector<UnaryBasisImageRef> UnaryBasisImageVector;

    typedef ImageRef<typename PairwiseBasis::ValueType, PairwiseBasis::Size> PairwiseBasisImageRef;
    typedef ImageRefC<typename PairwiseBasis::ValueType, PairwiseBasis::Size> PairwiseBasisImageRefC;
    typedef std::vector<PairwiseBasisImageRef> PairwiseBasisImageVector;

    typedef LinearOperator::OperatorRef<Feature, UnaryGroundLabel, LinearOperatorWeights> LinearOperatorRef;
    typedef std::vector< LinearOperatorRef > LinearOperatorVector;

    typedef TMonitor Monitor;
};

#endif // H_RTF_TYPES_H
