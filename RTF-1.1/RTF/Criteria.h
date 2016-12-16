/* This file is part of the "Regression Tree Fields" (RTF) source code distribution,
 * obtained from http://research.microsoft.com/downloads.
 * It is provided to you under the terms of the Microsoft Research License Agreement
 * (MSR-LA). Please see License.txt for details.
 *
 *
 * File: Criteria.h
 * Implements splitting criteria to be used for regression tree training.
 *
 */

#ifndef H_RTF_CRITERIA_H
#define H_RTF_CRITERIA_H

#include <algorithm>
#include <limits>
#include <tuple>
#include <vector>
#include <random>
#include <cassert>
#include <cmath>

#include "Types.h"
#include "Utility.h"

namespace Criteria
{
    // The classical criterion that was introduced for CART trees; it is simple and robust.
    template<typename LabelT>
    class SquaredResidualsCriterion
    {
    public:
        typedef LabelT                     TLabel;
        typedef typename TLabel::ValueType TValue;

        // Allows for compact accumulation of the relevant statistics of the data points,
        // such that the criterion can be evaluated very efficiently.
        struct SufficientStatistics
        {
            typedef typename TLabel::ValueType ValueType;
            static const size_t Size = 2 * TLabel::Size + 1;

            TLabel sum;
            TLabel sumSquares;
            TValue num;

            SufficientStatistics() : sum(), sumSquares(), num(0) {}


            // Merge our statistics with other statistics that were accumulated on a different set of data points
            SufficientStatistics operator+(const SufficientStatistics& other) const
            {
                SufficientStatistics ret;
                ret.sum        += this->sum;
                ret.sum        += other.sum;
                ret.sumSquares += this->sumSquares;
                ret.sumSquares += other.sumSquares;
                ret.num        += this->num;
                ret.num        += other.num;
                return ret;
            }

            // Add a data point to the statistics
            SufficientStatistics& operator+=(const TLabel& label)
            {
                const auto dim = TLabel::Size;

                for(size_t d = 0; d < dim; ++d)
                {
                    sum[d] += label[d];
                    sumSquares[d] += label[d] * label[d];
                }

                num++;
                return *this;
            }

            // Compute the squared residuals by means of the compact statistics
            TValue sumSquaredResiduals() const
            {
                auto ssr = TValue(0);
                const auto dim = TLabel::Size;

                for(size_t d = 0; d < dim; ++d)
                    ssr += (sumSquares[d] - (sum[d] * sum[d]) / num);

                return ssr;
            }

            template<typename ValueT>
            ValueT* Serialize(ValueT* outbuf) const
            {
                for( size_t i = 0; i < TLabel::Size; ++i, ++outbuf )
                    *outbuf = static_cast<ValueT>(sum[i]);
                for( size_t i = 0; i < TLabel::Size; ++i, ++outbuf )
                    *outbuf = static_cast<ValueT>(sumSquares[i]);
                *outbuf = static_cast<ValueT>(num);
                return ++outbuf;
            }

            template<typename ValueT>
            const ValueT* Deserialize(const ValueT* inbuf)
            {
                for( size_t i = 0; i < TLabel::Size; ++i, ++inbuf )
                    sum[i] = static_cast<TValue>(*inbuf);
                for( size_t i = 0; i < TLabel::Size; ++i, ++inbuf )
                    sumSquares[i] = static_cast<TValue>(*inbuf);
                num = static_cast<TValue>(*inbuf);
                return ++inbuf;
            }

            size_t NumPoints() const
            {
                return num;
            }
        };

    private:
        TValue epsilon;

        // Numerically stable computation of the sum of squared residuals given all points
        TValue SumSquaredResiduals(const std::vector<const TLabel*>& points) const
        {
            const size_t num = points.size();
            const size_t dim = TLabel::Size;
            TValue ssr = TValue(0);

            for(unsigned d = 0; d < dim; ++d)
            {
                TValue mean = TValue(0), m2 = TValue(0);

                for(unsigned n = 0; n < num; ++n)
                {
                    const TValue x     = (*points[n])[d];
                    const TValue delta = x - mean;
                    mean = mean + delta / (n + 1);
                    m2   = m2 + delta * (x - mean);
                }

                ssr = ssr + m2;
            }

            return ssr;
        }

    public:
        SquaredResidualsCriterion(TValue purityEpsilon = TValue()) : epsilon(purityEpsilon) {}

        // Compute the criterion score given an explicit split (all points that go left and right)
        TValue operator()(size_t nodeIndex, const std::vector<const TLabel*>& left, const std::vector<const TLabel*>& right) const
        {
            if(left.size() == 0 || right.size() == 0)
                return -std::numeric_limits<TValue>::max();
            else
                return (- SumSquaredResiduals(left) - SumSquaredResiduals(right));
        }

        // Compute the criterion score efficiently by means of the accumulated sufficient statistics for the points that go left and right
        TValue operator()(size_t nodeIndex, const SufficientStatistics& left, const SufficientStatistics& right) const
        {
            if(left.num == 0 || right.num == 0)
                return -std::numeric_limits<TValue>::max();
            else
                return (- left.sumSquaredResiduals() - right.sumSquaredResiduals());
        }

        // Determine the prediction of our model given the provided data points
        static TLabel Average(const std::vector<const TLabel*>& points)
        {
            TLabel mean;
            const size_t num = points.size();
            const size_t dim = TLabel::Size;

            for(size_t n = 0; n < num; ++n)
            {
                const TLabel& point = *points[n];

                for(unsigned d = 0; d < dim; ++d)
                    mean[d] += point[d] / num;
            }

            return mean;
        }

        // Determine the prediction of our model given the provided data points
        static TLabel Average(const SufficientStatistics& stats)
        {
            TLabel mean = stats.sum;
            mean /= stats.num;
            return mean;
        }

        // Check if the given data points are pure. For this criterion this is the case of the sum of squared residuals is tiny.
        bool IsPure(const std::vector<const TLabel*>& all) const
        {
            return std::abs(SumSquaredResiduals(all)) <= epsilon;
        }
    };

    // Gradient norm splitting criterion
    //
    // The criterion for joint training of weights and trees; we measure the squared norm of
    // the projected gradient with respect to the new model parameters introduced by the split.
    // The points passed to operator() must be the gradient contributions by the instances of the
    // considered factor type in the negative log pseudolikelihood objective.
    // The new model parameters are assumed to be equal to those of their parent node for
    // computation of the gradient.
    // The measure is expected to be correlated with the decrease in the negative log
    // pseudolikelihood objective enabled by the split.
    template<typename TWeights>
    class GradientNormCriterion
    {
    public:
        typedef typename TWeights::TValue                                TValue;
        typedef Training::LabelVector<TValue, TWeights::NumCoefficients> TLabel;

        struct SufficientStatistics
        {
            typedef typename TLabel::ValueType ValueType;
            static const size_t Size = TLabel::Size + 1;

            TLabel sum;
            TValue num;

            SufficientStatistics() : sum(), num(0) {}


            // Merge our statistics with other statistics that were accumulated on a different set of data points
            SufficientStatistics operator+(const SufficientStatistics& other) const
            {
                SufficientStatistics ret;
                ret.sum        += this->sum;
                ret.sum        += other.sum;
                ret.num        += this->num;
                ret.num        += other.num;
                return ret;
            }

            // Add a data point to the statistics
            SufficientStatistics& operator+=(const TLabel& label)
            {
                sum += label;
                num += 1;
                return *this;
            }

            // Compute ||Project(x - g) - x||_2^2
            TValue ProjectedGradientNorm(const TLabel& params, TValue smallestEigenvalue, TValue largestEigenvalue) const
            {
#ifndef CRITERIA_NORM_OF_UNPROJECTED_GRADIENT
                TLabel gradient(params);
                gradient -= sum;
                TWeights::Project(&gradient[0], smallestEigenvalue, largestEigenvalue);
                gradient -= params;
                return gradient.SquaredNorm();
#else
                return sum.SquaredNorm();
#endif
            }

            template<typename ValueT>
            ValueT* Serialize(ValueT* outbuf) const
            {
                for( size_t i = 0; i < TLabel::Size; ++i, ++outbuf )
                    *outbuf = static_cast<ValueT>(sum[i]);
                *outbuf = static_cast<ValueT>(num);
                return ++outbuf;
            }

            template<typename ValueT>
            const ValueT* Deserialize(const ValueT* inbuf)
            {
                for( size_t i = 0; i < TLabel::Size; ++i, ++inbuf )
                    sum[i] = static_cast<TValue>(*inbuf);
                num = static_cast<TValue>(*inbuf);
                return ++inbuf;
            }

            size_t NumPoints() const
            {
                return static_cast<size_t>(num);
            }
        };

    private:
        TValue epsilon;
        VecCRef<TWeights*> weights; // lookup table for nodeIndex
        TValue smallestEigenvalue;
        TValue largestEigenvalue;

        // Compute ||Project(x - g) - x||_2^2
        TValue ProjectedGradientNorm(const TLabel& params, const std::vector<const TLabel*>& points) const
        {

            TLabel gradient(params);

            for(size_t i = 0; i < points.size(); ++i)
                gradient -= *(points[i]);

#ifndef CRITERIA_NORM_OF_UNPROJECTED_GRADIENT
            TWeights::Project(&gradient[0], smallestEigenvalue, largestEigenvalue);
#endif
            gradient -= params;
            return gradient.SquaredNorm();
        }

        GradientNormCriterion(TValue epsilon_);

    public:
        GradientNormCriterion(TValue epsilon_, const VecCRef<TWeights*> weights_,
                              TValue smallestEigenvalue_, TValue largestEigenvalue_)
            : epsilon(epsilon_), weights(weights_), smallestEigenvalue(smallestEigenvalue_), largestEigenvalue(largestEigenvalue_) {}

        TValue operator()(size_t nodeIndex, const std::vector<const TLabel*>& left, const std::vector<const TLabel*>& right) const
        {
            // Determine the parameters of the parent node, which are equal to the initial parameters of the child node candidates
            assert(nodeIndex < weights.size());
            TLabel params;
            weights[nodeIndex]->GetWeights(&params[0]);

            if(left.size() == 0 || right.size() == 0)
                return -std::numeric_limits<TValue>::max();
            else
                return std::sqrt(ProjectedGradientNorm(params, left) + ProjectedGradientNorm(params, right));
        }

        // Compute the criterion score efficiently by means of the accumulated sufficient statistics for the points that go left and right
        TValue operator()(size_t nodeIndex, const SufficientStatistics& left, const SufficientStatistics& right) const
        {
            assert(nodeIndex < weights.size());
            TLabel params;
            weights[nodeIndex]->GetWeights(&params[0]);

            if(left.num == 0 || right.num == 0)
                return -std::numeric_limits<TValue>::max();
            else
                return std::sqrt(left.ProjectedGradientNorm(params, smallestEigenvalue, largestEigenvalue)
                                 + right.ProjectedGradientNorm(params, smallestEigenvalue, largestEigenvalue));
        }

        // The average is not meaningful for this criterion
        static TLabel Average(const std::vector<const TLabel*>& points)
        {
            return TLabel();
        }

        // The average is not meaningful for this criterion
        static TLabel Average(const SufficientStatistics& stats)
        {
            return TLabel();
        }

        // A leaf is considered pure if the component of each gradient contribution is close to zero; this means
        // that our model distribution perfectly matches the empirical distribution, so there is no more gain in splitting.
        bool IsPure(const std::vector<const TLabel*>& all) const
        {
            const size_t dim = TLabel::Size;
            TValue sumSquares = TValue(0);

            for(size_t i = 0; i < all.size(); ++i)
            {
                const auto& entry = *(all[i]);

                for(size_t d = 0; d < dim; ++d)
                    sumSquares += entry[d] * entry[d];
            }

            return sumSquares < epsilon;
        }
    };

    template<typename TSufficientStatistics, typename TValue>
    TValue* SerializeStatistics(const std::vector<TSufficientStatistics>& statistics, TValue* outbuf)
    {
        for( size_t i = 0; i < statistics.size(); ++i )
            outbuf = statistics[i].template Serialize<TValue>(outbuf);
        return outbuf;
    }

    template<typename TSufficientStatistics, typename TValue>
    const TValue* DeserializeStatistics(const TValue *inbuf, std::vector<TSufficientStatistics>& statistics )
    {
        for( size_t i = 0; i < statistics.size(); ++i )
            inbuf = statistics[i].template Deserialize<TValue>(inbuf);
        return inbuf;
    }
}
#endif // H_RTF_CRITERIA_H
