/* This file is part of the "Regression Tree Fields" (RTF) source code distribution,
 * obtained from http://research.microsoft.com/downloads.
 * It is provided to you under the terms of the Microsoft Research License Agreement
 * (MSR-LA). Please see License.txt for details.
 *
 *
 * File: LinearOperator.h
 * Defines the abstract base class for custom linear operator support.
 *
 */

#ifndef H_RTF_LINEAR_OPERATOR_H
#define H_RTF_LINEAR_OPERATOR_H

#include <memory>
#include <iostream>

#include <Eigen/Eigen>

#include "Types.h"
#include "Compute.h"
#include "Serialization.h"

#ifndef INSTANTIATE_CUSTOM_OPERATOR
#define INSTANTIATE_CUSTOM_OPERATOR(type) NULL
#endif

namespace LinearOperator
{
    template<typename ValueT>
    class DefaultWeights
    {
    public:
        static const size_t NumCoefficients = 1;
        typedef ValueT TValue;

        TValue* GetWeights(TValue *ws) const
        {
            *ws = 0.0;
            return ++ws;
        }

        static TValue* Project(TValue *as, TValue, TValue)
        {
            *as = 0.0;
            return ++as;
        }
    };

    struct LinearOperatorInfo
    {
        bool hasTree;
        int nFeatureCount;
        int nDepthLevels;
        int nMinDataPointsForSplitConsideration;
        double purityEpsilon;
        double smallestEigenvalue;
        double largestEigenvalue;
    };

    template <typename TFeature, typename TUnaryGroundLabel, typename TWeights>
    class OperatorBase : public Compute::FactorType<typename TUnaryGroundLabel::ValueType>
    {
    public:
        typedef typename TUnaryGroundLabel::ValueType TValue;
        static const size_t VarDim = TUnaryGroundLabel::Size;

        typedef Compute::SystemVectorRef<TValue, VarDim>  SystemVectorRef;
        typedef Compute::BlockDiagonalRef<TValue, VarDim> BlockDiagonalRef;
        typedef Compute::SystemVectorCRef<TValue, VarDim> SystemVectorCRef;
        typedef Eigen::Matrix<TValue, Eigen::Dynamic, 1>  TVector;

        typedef Eigen::aligned_allocator<tree_node_<NodeData<TFeature, TWeights> > > LOPWeightsNodeAllocator;
        typedef TreeRef<TFeature, TWeights, LOPWeightsNodeAllocator> ModelTreeRef;

    private:
        ModelTreeRef tree;

    public:
        OperatorBase()
        {
            tree.set_head(NodeData<TFeature, TWeights>());
        }

        virtual void AddInImplicitMatrixMultipliedBy(const typename TFeature::PreProcessType& prep,
                const SystemVectorRef& Qy, const SystemVectorCRef& y) const = 0;

        virtual void AddInLinearContribution(const typename TFeature::PreProcessType& prep, const SystemVectorRef& l) const = 0;

        virtual void AccumulateGradient(const typename TFeature::PreProcessType& prep,
                                        const SystemVectorCRef& muLeftRef, const SystemVectorCRef& muRightRef, TValue normC) const
        {
        }

        virtual void AccumulateGradient(const typename TFeature::PreProcessType& prep,
                                        const std::vector<TVector>& muLeft, const std::vector<TVector>& muRight, TValue normC) const
        {
        }

        virtual void Print() const = 0;

        virtual void ResetWeights() = 0;

        virtual int Type() const = 0;

        virtual size_t NumPairwise() const
        {
            return 0;
        }

        virtual TValue* GetNonDifferentiableParameters(TValue* flatWeights) const
        {
            return flatWeights;
        }

        virtual const TValue* SetNonDifferentiableParameters(const TValue* flatWeights)
        {
            return flatWeights;
        }

        virtual size_t NumNonDifferentiableParameters() const
        {
            return 0;
        }

        virtual std::pair<TValue, TValue>* GetNonDifferentiableBoxConstraints(std::pair<TValue, TValue>* bounds) const
        {
            return bounds;
        }

        virtual std::istream& Deserialize(std::istream& in) = 0;

        virtual std::ostream& Serialize(std::ostream& out) const = 0;

        virtual LinearOperatorInfo GetInfo() const
        {
            return LinearOperatorInfo();
        }

        virtual void AddInDiagonal(const typename TFeature::PreProcessType& prep, const SystemVectorRef& diag) const
        {
        }

        virtual void AddInDiagonal(const typename TFeature::PreProcessType& prep, const BlockDiagonalRef& diag) const
        {
        }

        virtual void AccumulateMeanFieldGradient(const typename TFeature::PreProcessType& prep,
                const SystemVectorCRef& yhat, const SystemVectorCRef& yref, const SystemVectorCRef& invDiag, TValue normC) const
        {
        }

        virtual void AddPrecisionBlocks(int x, int y, Compute::SystemMatrixRow<TValue, VarDim>& row) const
        {
        }

        // Returns a vector of pointers to node weights in breadth-first order.
        VecCRef<TWeights*> WeightsInBreadthFirstOrder() const
        {
            VecRef<TWeights*> weights;

            for(auto it = tree.begin_breadth_first(); it != tree.end_breadth_first(); ++it)
                weights.push_back(&(it->data));

            return weights;
        }

        template<typename TRegressionTreeRef>
        void MergeTree(const TRegressionTreeRef& regressionTree)
        {
            auto j = tree.begin_breadth_first();

            for(auto i = regressionTree.begin_breadth_first(); i != regressionTree.end_breadth_first(); ++i, ++j)
            {
                // We've reached the leaf level of the model tree; add any nodes that are present in
                // the regression tree but not in the model tree, and copy over the branching feature
                // of the parent node.
                if(j.number_of_children() != i.number_of_children())
                {
                    assert(j.number_of_children() == 0);
                    j->feature = i->feature;

                    for(size_t k = 0; k < i.number_of_children(); k++)
                    {
                        auto child  = tree.append_child(j);
                        child->data = j->data;
                    }
                }
            }
        }

        virtual void ForEachGradientContribution(const typename TFeature::PreProcessType& prep,
                const std::vector<TVector>& muLeft,
                const std::vector<TVector>& muRight,
                std::function<void(const int posX,
                                   const int posY,
                                   Training::LabelVector<TValue,
                                   TWeights::NumCoefficients>)> op) const
        {
        }

        virtual void ForEachGradientContribution(const typename TFeature::PreProcessType& prep,
                const std::vector<TVector>& muLeft,
                const std::vector<TVector>& muRight,
                const VecCRef<Vector2D<int>>& subsample,
                std::function<void(const int posX,
                                   const int posY,
                                   Training::LabelVector<TValue,
                                   TWeights::NumCoefficients>)> op) const
        {
        }
    };

    template <typename TFeature, typename TUnaryGroundLabel, typename TWeights=DefaultWeights<typename TUnaryGroundLabel::ValueType>>
        class OperatorRef : public Compute::FactorType<typename TUnaryGroundLabel::ValueType>
            {
            private:
                std::shared_ptr< OperatorBase<TFeature, TUnaryGroundLabel, TWeights> > ptr;
                typedef OperatorRef<TFeature, TUnaryGroundLabel, TWeights> ReferenceType;

            public:
                static const size_t VarDim = TUnaryGroundLabel::Size;
                typedef typename TUnaryGroundLabel::ValueType TValue;

                typedef Compute::SystemVectorRef<TValue, VarDim>  SystemVectorRef;
                typedef Compute::BlockDiagonalRef<TValue, VarDim> BlockDiagonalRef;
                typedef Compute::SystemVectorCRef<TValue, VarDim> SystemVectorCRef;
                typedef Eigen::Matrix<TValue, Eigen::Dynamic, 1>  TVector;

    OperatorRef(OperatorBase<TFeature, TUnaryGroundLabel, TWeights>* ptr_) : ptr(ptr_) {}
    OperatorRef(const OperatorRef<TFeature, TUnaryGroundLabel, TWeights>& other) : ptr(other.ptr) {}

    static OperatorRef<TFeature, TUnaryGroundLabel, TWeights> Instantiate(int type)
    {
        return ReferenceType(INSTANTIATE_CUSTOM_OPERATOR(type));
    }

    size_t NumPairwise() const
    {
        return ptr->NumPairwise();
    }

    void AddInImplicitMatrixMultipliedBy(const typename TFeature::PreProcessType& prep,
                                         const SystemVectorRef& Qy, const SystemVectorCRef& y) const
    {
        return ptr->AddInImplicitMatrixMultipliedBy(prep, Qy, y);
    }

    void AddInLinearContribution(const typename TFeature::PreProcessType& prep, const SystemVectorRef& l) const
    {
        return ptr->AddInLinearContribution(prep, l);
    }

    virtual void AddInDiagonal(const typename TFeature::PreProcessType& prep, const SystemVectorRef& diag) const
    {
        ptr->AddInDiagonal(prep, diag);
    }

    virtual void AddInDiagonal(const typename TFeature::PreProcessType& prep, const BlockDiagonalRef& diag) const
    {
        ptr->AddInDiagonal(prep, diag);
    }

    void AccumulateGradient(const typename TFeature::PreProcessType& prep,
                            const SystemVectorCRef& muLeftRef, const SystemVectorCRef& muRightRef, TValue normC) const
    {
        return ptr->AccumulateGradient(prep, muLeftRef, muRightRef, normC);
    }

    void AccumulateGradient(const typename TFeature::PreProcessType& prep,
                            const std::vector<TVector>& muLeft, const std::vector<TVector>& muRight, TValue normC) const
    {
        return ptr->AccumulateGradient(prep, muLeft, muRight, normC);
    }

    void AccumulateMeanFieldGradient(const typename TFeature::PreProcessType& prep,
                                     const SystemVectorCRef& yhat, const SystemVectorCRef& yref, const SystemVectorCRef& invDiag, TValue normC) const
    {
        return ptr->AccumulateMeanFieldGradient(prep, yhat, yref, invDiag, normC);
    }

    const TValue* CheckFeasibility(const TValue *flatWeights, bool& feasible) const
    {
        return ptr->CheckFeasibility(flatWeights, feasible);
    }

    TValue* Project(TValue *flatWeights) const
    {
        return ptr->Project(flatWeights);
    }

    size_t NumWeights() const
    {
        return ptr->NumWeights();
    }

    const TValue* SetWeights(const TValue *flatWeights)
    {
        return ptr->SetWeights(flatWeights);
    }

    TValue* GetWeights(TValue *flatWeights) const
    {
        return ptr->GetWeights(flatWeights);
    }

    TValue* GetNonDifferentiableParameters(TValue* flatWeights) const
    {
        return ptr->GetNonDifferentiableParameters(flatWeights);
    }

    const TValue* SetNonDifferentiableParameters(const TValue* flatWeights) const
    {
        return ptr->SetNonDifferentiableParameters(flatWeights);
    }

    size_t NumNonDifferentiableParameters() const
    {
        return ptr->NumNonDifferentiableParameters();
    }

    std::pair<TValue, TValue>* GetNonDifferentiableBoxConstraints(std::pair<TValue, TValue>* bounds) const
    {
        return ptr->GetNonDifferentiableBoxConstraints(bounds);
    }

    TValue* GetGradientAddPrior(TValue *flatGradient, TValue& objective) const
    {
        return ptr->GetGradientAddPrior(flatGradient, objective);
    }

    void AddPrecisionBlocks(int x, int y, Compute::SystemMatrixRow<TValue, VarDim>& row) const
    {
        ptr->AddPrecisionBlocks(x, y, row);
    }

    void ClearGradient()
    {
        ptr->ClearGradient();
    }

    void ResetWeights()
    {
        ptr->ResetWeights();
    }

    void Print() const
    {
        return ptr->Print();
    }

    int Type() const
    {
        return ptr->Type();
    }

    std::istream& Deserialize(std::istream& in)
    {
        return ptr->Deserialize(in);
    }

    std::ostream& Serialize(std::ostream& out) const
    {
        return ptr->Serialize(out);
    }

    LinearOperatorInfo GetInfo() const
    {
        return ptr->GetInfo();
    }

    VecCRef<TWeights*> WeightsInBreadthFirstOrder() const
    {
        return ptr->WeightsInBreadthFirstOrder();
    }

    template<typename TRegressionTreeRef>
    void MergeTree(const TRegressionTreeRef& regressionTree)
    {
        return ptr->template MergeTree<TRegressionTreeRef>(regressionTree);
    }

    void ForEachGradientContribution(const typename TFeature::PreProcessType& prep,
                                     const std::vector<TVector>& muLeft,
                                     const std::vector<TVector>& muRight,
                                     std::function<void(const int posX,
                                             const int posY,
                                             Training::LabelVector<TValue,
                                             TWeights::NumCoefficients>)> op) const
    {
        ptr->ForEachGradientContribution(prep, muLeft, muRight, op);
    }

    void ForEachGradientContribution(const typename TFeature::PreProcessType& prep,
                                     const std::vector<TVector>& muLeft,
                                     const std::vector<TVector>& muRight,
                                     const VecCRef<Vector2D<int>>& subsample,
                                     std::function<void(const int posX,
                                             const int posY,
                                             Training::LabelVector<TValue,
                                             TWeights::NumCoefficients>)> op) const
    {
        ptr->ForEachGradientContribution(prep, muLeft, muRight, subsample, op);
    }

#ifdef _OPENMP
    void InitializeLocks()
    {
        ptr->InitializeLocks();
    }

    void DestroyLocks()
    {
        ptr->DestroyLocks();
    }
#endif
            };
}

#endif // H_RTF_LINEAR_OPERATOR_H
