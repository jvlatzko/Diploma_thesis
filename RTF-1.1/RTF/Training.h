/* This file is part of the "Regression Tree Fields" (RTF) source code distribution,
 * obtained from http://research.microsoft.com/downloads.
 * It is provided to you under the terms of the Microsoft Research License Agreement
 * (MSR-LA). Please see License.txt for details.
 *
 *
 * File: Training.h
 * Implements algorithms for training of regression trees.
 *
 */

#ifndef H_RTF_TRAINING_H
#define H_RTF_TRAINING_H

#include <array>
#include <limits>
#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>

#include <Eigen/Dense>

#include "Array.h"
#include "Trees.h"
#include "Image.h"
#include "Criteria.h"
#include "Utility.h"

namespace Training
{
    // Default implementation of a ground truth label vector
    template<typename TValue, size_t Dim>
    class LabelVector
    {
    private:
        TValue values[Dim];

    public:
        typedef TValue ValueType;
        static const size_t Size = Dim;

        const size_t size() const
        {
            return Size;
        }

        TValue& operator[](size_t index)
        {
            return values[index];
        }

        const TValue& operator[](size_t index) const
        {
            assert(index < Dim * 2);
            return values[index];
        }

        LabelVector < TValue, Dim / 2 > first() const
        {
            return LabelVector < TValue, Dim / 2 > (&values[0]);
        }

        LabelVector < TValue, Dim / 2 > second() const
        {
            return LabelVector < TValue, Dim / 2 > (&values[Dim / 2]);
        }

        LabelVector<TValue, Dim>& operator=(const LabelVector<TValue, Dim>& rhs)
        {
            if(&rhs != this)
            {
                memcpy(&values[0], &rhs.values[0], Dim * sizeof(TValue));
            }

            return *this;
        }

        LabelVector<TValue, Dim>& operator+=(const LabelVector<TValue, Dim>& rhs)
        {
            for(size_t i = 0; i < Dim; ++i)
                values[i] += rhs[i];

            return *this;
        }

        LabelVector<TValue, Dim>& operator-=(const LabelVector<TValue, Dim>& rhs)
        {
            for(size_t i = 0; i < Dim; ++i)
                values[i] -= rhs[i];

            return *this;
        }

        LabelVector<TValue, Dim>& operator*=(TValue alpha)
        {
            for(size_t i = 0; i < Dim; ++i)
                values[i] *= alpha;

            return *this;
        }

        LabelVector<TValue, Dim>& operator/=(TValue alpha)
        {
            return operator*=(1.0 / alpha);
        }

        TValue SquaredNorm() const
        {
            TValue nrm = TValue(0);

            for(size_t i = 0; i < Dim; ++i)
                nrm += values[i] * values[i];

            return nrm;
        }

        LabelVector(const LabelVector<TValue, Dim>& rhs)
        {
            memcpy(&values[0], &rhs.values[0], Dim * sizeof(TValue));
        }

        LabelVector(const LabelVector < TValue, Dim / 2 > & first, const LabelVector < TValue, Dim / 2 > & second)
        {
            memcpy(&values[0], &first.values[0], Dim / 2 * sizeof(TValue));
            memcpy(&values[Dim / 2], &second.values[0], Dim / 2 * sizeof(TValue));
        }

        LabelVector()
        {
            for(size_t i = 0; i < Dim; ++i)
                values[i] = TValue(0);
        }

        friend class LabelVector<TValue, Dim*2>;
    };

    template <typename TLabel>
    ImageRef<TLabel>& operator+=(ImageRef<TLabel>& lhs, const ImageRef<TLabel>& rhs)
    {
        const int cx = lhs.Width(), cy = lhs.Height();

        for(int y = 0; y < cy; ++y)
            for(int x = 0; x < cx; ++x)
                lhs(x, y) += rhs(x, y);

        return lhs;
    }

    template <typename TLabel>
    ImageRef<TLabel>& operator*=(ImageRef<TLabel>& lhs, const typename TLabel::ValueType& alpha)
    {
        const int cx = lhs.Width(), cy = lhs.Height();

        for(int y = 0; y < cy; ++y)
            for(int x = 0; x < cx; ++x)
                lhs(x, y) *= alpha;

        return lhs;
    }

    template <typename TValue, size_t Dim>
    std::ostream& operator << (std::ostream& out, const LabelVector<TValue, Dim>& label)
    {
        for(size_t i = 0; i < label.size(); ++i)
        {
            out << label[i];

            if(i != label.size())
                out << ' ';
        }

        return out;
    }

    // Information that is permanently associated with a node of our regression tree. It is filled into the newly
    // created child nodes from temporary information when a node is split. Notably, this permanent information
    // consists of the number of points that went through the node when it was created, as well as the average
    // of these points. The average can later be used to predict the response vector for a given input.
    template<typename TLabel>
    class RegressionTreeNode
    {
    public:
        RegressionTreeNode() : trained(false), numDataPoints(0) {}
        RegressionTreeNode(size_t numPoints, TLabel avg, bool trained_ = false) : trained(trained_), numDataPoints(numPoints), average(avg) {}
        bool     trained;
        size_t numDataPoints;
        TLabel   average;
    };

    // Feature evaluation kernels: Over a number of invocations, determine the feature/threshold
    // combination that yields the highest score according to our criterion and record it.
    // The kernels are designed such that one kernel can be used per thread. Hence, the best
    // approach is to first find the best feature/threshold pair _per thread_, and afterwards
    // reduce the individual results of the threads.
    //
    template<typename TFeature, typename TLabel, typename TSplitCriterion, typename TEntry, bool UseExplicitThresholdTesting = false>
    class FeatureEvaluationKernelBase
    {
    public:
        const std::vector<TEntry>& entries;
        TSplitCriterion            criterion;
        size_t                     nodeIndex;
        VecCRef<Vector2D<int>>     offsets;

        TFeature                   bestFeature;
        typename TSplitCriterion::TValue bestScore;

        std::mt19937                           mt;
        std::uniform_int_distribution<size_t>  uniform;

        size_t RandomEntryIndex()
        {
            return uniform(mt);
        }

    public:
        FeatureEvaluationKernelBase(const std::vector<TEntry>& entries_, const TSplitCriterion& criterion_, size_t nodeIndex_, const VecCRef<Vector2D<int>>& offsets_)
            : entries(entries_), criterion(criterion_), nodeIndex(nodeIndex_), offsets(offsets_), bestScore(- std::numeric_limits<typename TSplitCriterion::TValue>::max()),
              uniform(0, entries.size() - 1)
        {
        }

        const TFeature& BestFeature() const
        {
            return bestFeature;
        }

        typename TSplitCriterion::TValue BestScore() const
        {
            return bestScore;
        }
    };

    // Plain kernel that samples one threshold per feature
    template<typename TFeature, typename TLabel, typename TSplitCriterion, typename TEntry, bool UseExplicitThresholdTesting = false>
    class FeatureEvaluationKernel : public FeatureEvaluationKernelBase<TFeature, TLabel, TSplitCriterion, TEntry>
    {
    private:
        typedef FeatureEvaluationKernelBase<TFeature, TLabel, TSplitCriterion, TEntry> Base;

        std::vector<const TLabel*> left;
        std::vector<const TLabel*> right;

    public:
        FeatureEvaluationKernel(const std::vector<TEntry>& entries, const TSplitCriterion& criterion, size_t nodeIndex, const VecCRef<Vector2D<int>>& offsets)
            : Base::FeatureEvaluationKernelBase(entries, criterion, nodeIndex, offsets)
        {
            left.reserve(entries.size() / 2);
            right.reserve(entries.size() / 2);
        }

        void operator()(const TFeature& candidateFeature)
        {
            const TEntry& entrySample = Base::entries[Base::RandomEntryIndex()];
            const TFeature feature    = candidateFeature.WithThresholdFromSample(entrySample.x, entrySample.y, entrySample.prep, Base::offsets);
            left.clear();
            right.clear();
            std::for_each(Base::entries.begin(), Base::entries.end(), [&](const TEntry & entry)
            {
                if(feature(entry.x, entry.y, entry.prep, this->offsets))
                    right.push_back(&entry.label);
                else
                    left.push_back(&entry.label);
            });
            auto score = Base::criterion(Base::nodeIndex, left, right);

            if(score > Base::bestScore)
            {
                Base::bestScore   = score;
                Base::bestFeature = feature;
            }
        }
    };

    // More advanced kernel that samples a number of thresholds per feature and evaluates them.
    // This is cheaper if computation of the feature response is expensive, since the response
    // only needs to be computed once irrespective of the number of thresholds that are checked.
    template<typename TFeature, typename TLabel, typename TSplitCriterion, typename TEntry>
    class FeatureEvaluationKernel<TFeature, TLabel, TSplitCriterion, TEntry, true> : public FeatureEvaluationKernelBase<TFeature, TLabel, TSplitCriterion, TEntry>
    {
    private:
        typedef FeatureEvaluationKernelBase<TFeature, TLabel, TSplitCriterion, TEntry> Base;
        std::function<void(const TFeature&)> kernel;

        static const size_t NumTests = TFeature::NumThresholdTests;

        typedef typename TSplitCriterion::TValue TValue;
        typedef typename TSplitCriterion::SufficientStatistics SufficientStatistics;


        template<typename TArray1, typename TArray2>
        void CheckThresholds(const TFeature& feature, const TArray1& leftUpTo, const TArray1& rightUpTo, const TArray2& rightBinBoundaries, size_t num)
        {
            long bestIndex = -1;

            for(size_t i = 0; i < num; ++i)
            {
                const auto score = this->criterion(Base::nodeIndex, leftUpTo[i], rightUpTo[i + 1]);

                if(score > Base::bestScore)
                {
                    Base::bestScore = score;
                    bestIndex = static_cast<long>(i);
                }
            }

            if(bestIndex >= 0)
                Base::bestFeature = feature.WithThreshold(rightBinBoundaries[bestIndex]);
        }

        void CheckForBetterSplitDegenerate(const TFeature& feature)
        {
            const size_t NumEntries = Base::entries.size();

            std::vector<TValue>               responses(NumEntries);
            std::vector<TValue>               rightBinBoundaries(NumEntries);
            std::vector<SufficientStatistics> statistics(NumEntries);
            std::vector<SufficientStatistics> leftUpTo(NumEntries);
            std::vector<SufficientStatistics> rightUpTo(NumEntries);

            // Pre-compute the response values of all entries
            for(size_t i = 0; i < NumEntries; ++i)
            {
                const auto& entry = Base::entries[i];
                responses[i] = feature.Response(entry.x, entry.y, entry.prep, Base::offsets);
            }

            // Construct right bin boundaries:
            // Note that we do not create a bin corresponding to the smallest response, since
            // no entry is going to end up in that bin anyway (no response is smaller than the smallest response)
            for(size_t i = 0; i < NumEntries - 1; ++i)
                rightBinBoundaries[i] = responses[i + 1];

            rightBinBoundaries[NumEntries - 1] = std::numeric_limits<TValue>::max();
            // Sort bins for efficient lookup
            std::sort(rightBinBoundaries.begin(), rightBinBoundaries.begin() + NumEntries);

            // Fill in statistics
            for(size_t i = 0; i < NumEntries; ++i)
            {
                auto iterator       = std::upper_bound(rightBinBoundaries.begin(), rightBinBoundaries.begin() + NumEntries, responses[i]);
                auto offset         = iterator - rightBinBoundaries.begin();
                statistics[offset] += Base::entries[i].label;
            }

            // Form partial sums for the left and the right part of the split
            std::partial_sum(statistics.begin(), statistics.begin() + NumEntries, leftUpTo.begin());
            std::partial_sum(statistics.rend() - NumEntries, statistics.rend(), rightUpTo.rend() - NumEntries);

            // Check if there is a split that is better than the best feature/threshold combination recorded thus far
            CheckThresholds(feature,
                            leftUpTo,
                            rightUpTo,
                            rightBinBoundaries,
                            NumEntries - 1);
        }

        void CheckForBetterSplit(const TFeature& feature)
        {
            std::vector < TValue > rightBinBoundaries(NumTests+1);
            std::vector < SufficientStatistics > statistics(NumTests+1);
            std::vector < SufficientStatistics > leftUpTo(NumTests+1);
            std::vector < SufficientStatistics > rightUpTo(NumTests+1);

            // Determine bin boundaries
            for(size_t t = 0; t < NumTests; ++t)
            {
                const auto index      = Base::RandomEntryIndex();
                const auto &entry     = Base::entries[index];
                rightBinBoundaries[t] = feature.Response(entry.x, entry.y, entry.prep, Base::offsets);
            }

            rightBinBoundaries.back() = std::numeric_limits<typename TSplitCriterion::TValue>::max();

            // Protect against worst-case scenario where all sampled responses are exactly equal;
            // this can occur quite frequently for imbalanced discrete labels (but then again, you shouldn't use the thresholding
            // training variant for discrete problems)
            const auto front    = rightBinBoundaries.front();
            auto unequalToFront = std::count_if(rightBinBoundaries.begin() + 1, rightBinBoundaries.end() - 1, [ = ](TValue v)->bool { return v != front; });

            if(unequalToFront == 0)
            {
                // Simply scan through entries one after another until we found a different response.
                for(size_t t = 0; t < Base::entries.size() && front == rightBinBoundaries.front(); ++t)
                {
                    const auto& entry = Base::entries[t];
                    rightBinBoundaries.front() = feature.Response(entry.x, entry.y, entry.prep, Base::offsets);
                }
            }

            // Sort bins for efficient lookup
            std::sort(rightBinBoundaries.begin(), rightBinBoundaries.end());

            // Accumulate sufficient statistics of the bins
            std::for_each(Base::entries.begin(), Base::entries.end(), [&](const TEntry & entry)
            {
                auto iterator       = std::upper_bound(rightBinBoundaries.begin(), rightBinBoundaries.end(),
                                                       feature.Response(entry.x, entry.y, entry.prep, this->offsets));
                auto offset         = iterator - rightBinBoundaries.begin();
                statistics[offset] += entry.label;
            });

            // Form partial sums for the left and the right part of the split
            std::partial_sum(statistics.begin(), statistics.end(), leftUpTo.begin());
            std::partial_sum(statistics.rbegin(), statistics.rend(), rightUpTo.rbegin());

            // Check if there is a split that is better than the best feature/threshold combination recorded thus far
            CheckThresholds(feature, leftUpTo, rightUpTo, rightBinBoundaries, NumTests);
        }

    public:
        FeatureEvaluationKernel(const std::vector<TEntry>& entries, const TSplitCriterion& criterion, size_t nodeIndex, const VecCRef<Vector2D<int>>& offsets)
            : Base::FeatureEvaluationKernelBase(entries, criterion, nodeIndex, offsets)
        {
            // Choose actual kernel depending on number of entries
            if(entries.size() > TFeature::NumThresholdTests)
            {
                // Normal case - we have more entries than threshold tests
                kernel = [&](const TFeature & feature)
                {
                    this->CheckForBetterSplit(feature);
                };
            }
            else
            {
                // Degenerate case - we need not sample but can actually consider each single entry as a possible split point.
                kernel = [&](const TFeature & feature)
                {
                    this->CheckForBetterSplitDegenerate(feature);
                };
            }
        }

        void operator()(const TFeature& feature)
        {
            return kernel(feature);
        }
    };

    // Holds temporary information associated with a leaf node during one round of tree growing. Essentially, at each
    // round, the data points are sorted into the respective leaves. One instance of this class will hold the data
    // points that belong to a particular leaf. The set of points can then be used to compute measures of impurity
    // and evaluate which feature out of a given set is most promising. Finally, the class provides means of splitting
    // the associated leaf node in case a useful split is found. After the round is over, some condensed information
    // (such as the average over the points) will be held in RegressionTreeNodes of the children, whereas the bulk
    // of information is discarded and re-built (for different leaves) during the next round.
    template <typename TFeature, typename TLabel, typename TMonitor, bool UseExplicitThresholding>
    class DataPoints
    {
    public:
        typedef typename TFeature::PreProcessType        TPrep;
        typedef RegressionTreeNode<TLabel>               TRegressionTreeNode;
        typedef NodeData<TFeature, TRegressionTreeNode>  TNodeData;

    private:
        // Describes one data point, consisting of the location in the image (x,y), the offsets of the factor, the
        // pre-processed data we condition on, and finally the ground truth label for the factor at that position.
        // For regression trees, the label is most likely a vector of real numbers.
        struct Entry
        {
            Entry(const TLabel& l_, int x_, int y_, const TPrep& p_)
                : label(l_), x(x_), y(y_), prep(p_)
            {
            }
            TLabel label;
            TPrep prep;
            int x;
            int y;
        };
        // Holds our data points.
        std::vector<Entry> entries;

        // Index of the node in breadth-first order.
        int nodeIndex;

        // Iterator pointing to the node in the tree.
        typename TreeRef<TFeature, TRegressionTreeNode>::iterator_base leaf;

        // Offsets of the second factor variable relative to the currently considered variable
        VecCRef<Vector2D<int>> offsets;

        // We follow the following approach towards parallelization:
        // We parallelize over the features.
        // First, we find the best feature for each thread individually (each thread is assigned a subset of features to evaluate)
        // Afterwards, we find the globally best feature over the set of thread-locally best features.
        // Only the second step requires synchronization.
        // This is efficient as long as the number of features is (much) larger than the number of threads.
        template<typename TSplitCriterion>
        TFeature BestSplitParallel(const std::vector<TFeature>& features,
                                   const TSplitCriterion& criterion,
                                   size_t &numLeft, TLabel& averageLeft,
                                   size_t &numRight, TLabel& averageRight,
                                   TLabel& averageAll, typename TSplitCriterion::TValue& bestScore) const
        {
            typedef typename TSplitCriterion::TValue TValue;
            // Shared variables
            bestScore = - std::numeric_limits<TValue>::max();
            TFeature bestFeature;
            #pragma omp parallel shared(bestScore, bestFeature)
            {
                // Thread-local kernel
                FeatureEvaluationKernel<TFeature, TLabel, TSplitCriterion, Entry, UseExplicitThresholding> threadKernel(entries, criterion, nodeIndex, offsets);
                const size_t numFeatures = features.size();
                // for each thread, determine the best feature out of the subset of features assigned to that thread
                #pragma omp for

                for(int f = 0; f < numFeatures; ++f)
                    threadKernel(features[f]);

                // find the globally best feature over the set of thread-locally best features
                #pragma omp critical
                {
                    if(threadKernel.BestScore() > bestScore)
                    {
                        bestScore   = threadKernel.BestScore();
                        bestFeature = threadKernel.BestFeature();
                    }
                }
            }

            // Reconstruct the split induced by the best feature; this step is very cheap so it is typically
            // more efficient to reconstruct the split rather than preserve it during the first step.
            typename TSplitCriterion::SufficientStatistics left, right;
            std::for_each(entries.begin(), entries.end(), [&](const Entry & entry)
            {
                if(bestFeature(entry.x, entry.y, entry.prep, offsets))
                    right += entry.label;
                else
                    left  += entry.label;
            });
            numLeft      = left.NumPoints();
            averageLeft  = TSplitCriterion::Average(left);
            numRight     = right.NumPoints();
            averageRight = TSplitCriterion::Average(right);
            averageAll   = TSplitCriterion::Average(left + right);
            return bestFeature;
        }

        template<typename TSplitCriterion>
        TFeature BestSplit(const std::vector<TFeature>& features,
                           const TSplitCriterion& criterion,
                           size_t &numLeft, TLabel& averageLeft,
                           size_t &numRight, TLabel& averageRight,
                           TLabel& averageAll, typename TSplitCriterion::TValue& bestScore) const
        {
#ifdef USE_MPI
#ifdef MPI_N_BEST
            return BestSplitNBestMPI(features, criterion, numLeft, averageLeft, numRight, averageRight, averageAll, bestScore);
#else
            return BestSplitMPI(features, criterion, numLeft, averageLeft, numRight, averageRight, averageAll, bestScore);
#endif
#else
            return BestSplitParallel(features, criterion, numLeft, averageLeft, numRight, averageRight, averageAll, bestScore);
#endif
        }

#ifdef USE_MPI
        template<typename TSplitCriterion>
        TFeature BestSplitMPI(const std::vector<TFeature>& features,
                              const TSplitCriterion& criterion,
                              size_t &numLeft, TLabel& averageLeft,
                              size_t &numRight, TLabel& averageRight,
                              TLabel& averageAll, typename TSplitCriterion::TValue& bestScore) const
        {
            TMonitor::Report("BestSplitMPI(%d) for %d entries ...\n", MPI::Communicator().rank(), entries.size());
            typedef typename TLabel::ValueType TValue;
            typedef typename TSplitCriterion::SufficientStatistics TSufficientStatistics;

            const size_t BlockSize = TFeature::NumThresholdTests + 1;
            std::vector<TSufficientStatistics> statistics(features.size() * BlockSize);
            std::vector<TValue> rightBinBoundaries(features.size() * BlockSize);
            std::vector<TValue> bestThresholdPerFeature(features.size());
            std::vector<TValue> bestScorePerFeature(features.size());
            std::vector< std::pair<TSufficientStatistics, TSufficientStatistics> > bestStatisticsPerFeature( features.size() );

            // Generate random numbers (the indices of the entries that will be used to compute the bin boundaries);
            // towards this end, we (ab)-use the vector of bin boundaries itself
            if( entries.size() > 0 )
            {
                std::vector< size_t > randomIndices( rightBinBoundaries.size() );
                std::mt19937 mt;
                std::uniform_int_distribution<size_t>  uniform(0, entries.size() - 1);
                for( int idx = 0; idx < randomIndices.size(); ++idx)
                    randomIndices[idx] = uniform(mt);

                // For each feature
                const auto csize = MPI::Communicator().size();
                const auto crank = MPI::Communicator().rank();
                #pragma omp parallel for
                for( int f = 0; f < features.size(); ++f )
                {
                    const auto start    = f * BlockSize;
                    const auto end      = start + BlockSize;
                    const auto &feature = features[f];

                    // Determine bin boundaries, each process determines a subset of the thresholds to make sure
                    // all training examples spread over the processes are considered
                    for( size_t t = start; t < end; ++t )
                    {
                        if( t % csize == crank && entries.size() > 0 )
                        {
                            const auto &entry     = entries[randomIndices[t]];
                            rightBinBoundaries[t] = feature.Response(entry.x, entry.y, entry.prep, offsets);
                        }
                        else
                        {
                            rightBinBoundaries[t] = 0.0;
                        }
                    }
                }
            }
            else
            {
                std::fill(rightBinBoundaries.begin(), rightBinBoundaries.end(), TValue(0));
            }

            // Use the bin boundaries determined by the root process, and send them to all other processes
            {
                TMonitor::Report("Reducing bin boundaries ...\n");
                std::vector<TValue> globalBinBoundaries(rightBinBoundaries.size());
                boost::mpi::all_reduce(MPI::Communicator(), &rightBinBoundaries[0], rightBinBoundaries.size(), &globalBinBoundaries[0], std::plus<TValue>());
                rightBinBoundaries = globalBinBoundaries;
                TMonitor::Report("Done.\n");
            }

            // For each feature
            #pragma omp parallel for
            for( int f = 0; f < features.size(); ++f )
            {
                const auto start    = f * BlockSize;
                const auto end      = start + BlockSize;
                const auto &feature = features[f];

                // Terminate bin and sort for efficient lookup
                rightBinBoundaries[end-1] = std::numeric_limits<typename TSplitCriterion::TValue>::max();
                std::sort(&rightBinBoundaries[start], &rightBinBoundaries[end]);

                // Accumulate sufficient statistics of the bins
                std::for_each(entries.begin(), entries.end(), [&](const Entry& entry)
                {
                    auto hit            = std::upper_bound(&rightBinBoundaries[start], &rightBinBoundaries[end],
                                                           feature.Response(entry.x, entry.y, entry.prep, offsets));
                    auto offset         = std::distance(&rightBinBoundaries[0], hit);
                    statistics[offset] += entry.label;
                });
            }

            // Reduce sufficient statistics
            {
                TMonitor::Report("Reducing sufficient statistics ...\n");
                std::vector < float > localBuffer( features.size() * BlockSize * TSufficientStatistics::Size );
                std::vector < float > globalBuffer( features.size() * BlockSize * TSufficientStatistics::Size );
                Criteria::SerializeStatistics(statistics, &localBuffer[0]);
                boost::mpi::all_reduce(MPI::Communicator(), &localBuffer[0], localBuffer.size(), &globalBuffer[0], std::plus<float>());
                Criteria::DeserializeStatistics(&globalBuffer[0], statistics);
                TMonitor::Report("Done.\n");
            }

            // For each feature
            #pragma omp parallel for
            for( int f = 0; f < features.size(); ++f )
            {
                const auto start    = f * BlockSize;
                const auto end      = start + BlockSize;

                // Determine the best threshold for that feature
                std::vector < TSufficientStatistics > leftUpTo(BlockSize);
                std::vector < TSufficientStatistics > rightUpTo(BlockSize);

                // Form partial sums for the left and the right part of the split
                std::partial_sum(&statistics[start], &statistics[end], leftUpTo.begin());
                std::partial_sum(std::reverse_iterator<TSufficientStatistics*>(&statistics[end]),
                                 std::reverse_iterator<TSufficientStatistics*>(&statistics[start]), rightUpTo.rbegin());

                // Check if there is a split that is better than the best threshold found thus far
                bestScorePerFeature[f] = -std::numeric_limits<TValue>::max();
                size_t bestT = 0;
                for(size_t t = 0; t < BlockSize-1; ++t)
                {
                    const auto score = criterion(nodeIndex, leftUpTo[t], rightUpTo[t + 1]);

                    if(score > bestScorePerFeature[f])
                    {
                        bestScorePerFeature[f] = score;
                        bestT = t;
                    }
                }
                bestThresholdPerFeature[f]         = rightBinBoundaries[start + bestT];
                bestStatisticsPerFeature[f].first  = leftUpTo[bestT];
                bestStatisticsPerFeature[f].second = rightUpTo[bestT + 1];
            }

            // Determine the index of the best feature
            const auto bestFeatureIndex = std::distance(bestScorePerFeature.begin(),
                                          std::max_element(bestScorePerFeature.begin(), bestScorePerFeature.end()));
            bestScore = bestScorePerFeature[bestFeatureIndex];
            const auto bestFeature = features[bestFeatureIndex].WithThreshold(bestThresholdPerFeature[bestFeatureIndex]);

            // Determine the average labels of the points according to the new split
            numLeft      = bestStatisticsPerFeature[bestFeatureIndex].first.NumPoints();
            averageLeft  = criterion.Average(bestStatisticsPerFeature[bestFeatureIndex].first);
            numRight     = bestStatisticsPerFeature[bestFeatureIndex].second.NumPoints();
            averageRight = criterion.Average(bestStatisticsPerFeature[bestFeatureIndex].second);
            averageAll   = criterion.Average( bestStatisticsPerFeature[bestFeatureIndex].first + bestStatisticsPerFeature[bestFeatureIndex].second );

            return bestFeature;
        }

#ifdef MPI_N_BEST
        void DetermineBinBoundaries(const std::vector<TFeature>& features, std::vector<typename TLabel::ValueType>& binBoundaries) const
        {
            // Typedefs
            typedef typename TLabel::ValueType TValue;

            // Constants
            const size_t FeatureBlockSize = TFeature::NumThresholdTests + 1;
            const size_t OverallTableSize = FeatureBlockSize * features.size();
            const size_t NumProcesses     = MPI::Communicator().size();
            const size_t ProcessBlockSize = static_cast<size_t>(std::ceil(OverallTableSize / static_cast<double>(NumProcesses)));
            const size_t MyProcessIdx     = MPI::Communicator().rank();

            // Random number generator
            std::mt19937 mt;
            std::uniform_int_distribution<size_t> uniform(0, entries.size() - 1);

            // Each node is responsible for a number of bin boundary entries; we iterate
            // over these and compute the feature responses to obtain the bin boundaries
            // determined by this process.
            std::vector<TValue> myBoundaries(ProcessBlockSize);
            for( size_t blockIdx = 0; blockIdx < ProcessBlockSize; ++blockIdx )
            {
                // The last block index may be invalid if the overall number of table entries
                // is not a multiple of the number of processes; if so, simply skip it.
                const size_t tableIdx = blockIdx * NumProcesses + MyProcessIdx;
                if( tableIdx >= OverallTableSize )
                {
                    myBoundaries[blockIdx] = std::numeric_limits<TValue>::max();
                    break;
                }

                // Determine the boundary as the response of a randomly chosen data point entry
                const size_t featureIdx = tableIdx / FeatureBlockSize;
                const auto &feature     = features[featureIdx];
                const size_t binIdx     = tableIdx - featureIdx * FeatureBlockSize;
                if( binIdx != TFeature::NumThresholdTests )
                {
                    if( entries.size() == 0 )
                    {
                        myBoundaries[blockIdx]  = 0.0;
                    }
                    else
                    {
                        const size_t entryIdx   = uniform(mt);
                        const auto &entry       = entries[entryIdx];
                        myBoundaries[blockIdx]  = feature.Response(entry.x, entry.y, entry.prep, offsets);
                    }
                }
                else // The last element of each feature block terminates the bins
                {
                    myBoundaries[blockIdx]  = std::numeric_limits<TValue>::max();
                }
            }

            // We have now collected the boundaries determined by this process.
            // In the next step, we gather all boundaries determined by any process.
            std::vector<TValue> allBoundaries(ProcessBlockSize * NumProcesses);
            boost::mpi::all_gather(MPI::Communicator(), &myBoundaries[0], ProcessBlockSize, &allBoundaries[0]);

            // The collected boundaries are arranged by MPI in blocks stemming from each process,
            // ordered by the rank of the process. We used the data structure so obtained to
            // populate the bin boundaries arranged by feature and threshold:
            //
            //   Process 0, threshold 0
            //   Process 0, threshold 1
            //   ...
            //   Process 0, threshold ProcessBlockSize
            //   Process 1, threshold 0
            //   ...
            //   Process NumProcesses, threshold ProcessBlockSize
            //
            binBoundaries.resize(OverallTableSize);
            #pragma omp parallel for
            for( int processIdx = 0; processIdx < NumProcesses; ++processIdx )
            {
                for( size_t blockIdx = 0; blockIdx < ProcessBlockSize; ++blockIdx )
                {
                    const size_t tableIdx = blockIdx * NumProcesses + processIdx;
                    // The last block index may be invalid if the overall number of table entries
                    // is not a multiple of the number of processes; if so, simply skip it.
                    if( tableIdx >= OverallTableSize )
                        break;

                    const size_t gatheredIndex = processIdx * ProcessBlockSize + blockIdx;
                    binBoundaries[tableIdx] = allBoundaries[gatheredIndex];
                }
            }
            // Sort the bin boundaries for efficient lookup
            #pragma omp parallel for
            for( int featureIdx = 0; featureIdx < features.size(); ++featureIdx )
            {
                const auto start    = featureIdx * FeatureBlockSize;
                const auto end      = start + FeatureBlockSize;
                std::sort(&binBoundaries[start], &binBoundaries[end]);
            }
        }

        template<typename TSufficientStatistics>
        void DetermineLocalSufficientStatistics(const std::vector<TFeature>& features,
                                                const std::vector<typename TLabel::ValueType>& rightBinBoundaries,
                                                std::vector<TSufficientStatistics>& statistics) const
        {
            // Constants
            const size_t FeatureBlockSize = TFeature::NumThresholdTests + 1;
            const size_t OverallTableSize = FeatureBlockSize * features.size();
            // For each feature in parallel, determine the sufficient statistics of the data point entries
            // for the thresholds determined in the previous step
            statistics.resize(OverallTableSize);
            #pragma omp parallel for
            for( int featureIdx = 0; featureIdx < features.size(); ++featureIdx )
            {
                const auto start    = featureIdx * FeatureBlockSize;
                const auto end      = start + FeatureBlockSize;
                const auto &feature = features[featureIdx];

                // Accumulate sufficient statistics of the bins
                std::for_each(entries.begin(), entries.end(), [&](const Entry& entry)
                {
                    const auto hitIdx            = std::upper_bound(&rightBinBoundaries[start], &rightBinBoundaries[end],
                                                   feature.Response(entry.x, entry.y, entry.prep, offsets));
                    const auto tableIdx          = std::distance(&rightBinBoundaries[0], hitIdx);
                    statistics[tableIdx]        += entry.label;
                });
            }
        }

        template<typename TSplitCriterion>
        void
        DetermineLocallyBestFeatures(const std::vector<TFeature>& features,
                                     const TSplitCriterion& criterion,
                                     const std::vector<typename TSplitCriterion::SufficientStatistics>& statistics,
                                     const std::vector<typename TLabel::ValueType>& rightBinBoundaries,
                                     std::vector<typename TSplitCriterion::SufficientStatistics>& locallyBestLeftStatistics,
                                     std::vector<typename TSplitCriterion::SufficientStatistics>& locallyBestRightStatistics,
                                     std::vector<size_t>& locallyBestTableIndices) const
        {
            // Typedefs
            typedef typename TLabel::ValueType TValue;
            typedef typename TSplitCriterion::SufficientStatistics TSufficientStatistics;

            // Constants
            const size_t FeatureBlockSize = TFeature::NumThresholdTests + 1;
            const size_t NumProcesses     = MPI::Communicator().size();

            assert(MPI_N_BEST <= rightBinBoundaries.size());

            // For each feature in parallel, determine the scores
            std::vector<TValue> allScores(rightBinBoundaries.size());
            std::vector<TSufficientStatistics> leftStatistics(rightBinBoundaries.size()),
                rightStatistics(rightBinBoundaries.size());
            #pragma omp parallel for
            for( int featureIdx = 0; featureIdx < features.size(); ++featureIdx )
            {
                const auto start    = featureIdx * FeatureBlockSize;
                const auto end      = start + FeatureBlockSize;

                // Allocate space for the accumulated statistics to the left and right
                std::vector < TSufficientStatistics > leftUpTo(FeatureBlockSize);
                std::vector < TSufficientStatistics > rightUpTo(FeatureBlockSize);

                // Form partial sums for the left and the right part of the split
                std::partial_sum(&statistics[start], &statistics[end], leftUpTo.begin());
                std::partial_sum(std::reverse_iterator<const TSufficientStatistics*>(&statistics[end]),
                                 std::reverse_iterator<const TSufficientStatistics*>(&statistics[start]), rightUpTo.rbegin());

                // For each threshold, determine the score achieved by the (feature, threshold pair) and
                // store the statistics of the points ending up in the left and right branch for latter use
                for(size_t t = 0; t < FeatureBlockSize-1; ++t)
                {
                    leftStatistics [start + t] = leftUpTo[t];
                    rightStatistics[start + t] = rightUpTo[t + 1];
                    allScores      [start + t] = criterion(nodeIndex, leftUpTo[t], rightUpTo[t + 1]);
                }
                allScores[end-1] = -std::numeric_limits<TValue>::max();
            }

            // Now, from the previously computed scores, find the indices of the MPI_N_BEST
            // locally best (feature, threshold) pairs of this node.
            std::vector<size_t> allIndices(allScores.size());
            for( size_t idx = 0; idx < allIndices.size(); ++idx )
                allIndices[idx] = idx;
            std::nth_element( allIndices.begin(), allIndices.begin() + MPI_N_BEST, allIndices.end(),
                              [&](size_t i1, size_t i2) -> bool {return allScores[i1] > allScores[i2];} );

            // Finally, gather the locally best indices of all processes: MPI_N_BEST indices per process
            locallyBestTableIndices.resize(NumProcesses * MPI_N_BEST);
            boost::mpi::all_gather(MPI::Communicator(), &allIndices[0], MPI_N_BEST, &locallyBestTableIndices[0]);

            // And store the statistics belonging to these indices
            locallyBestLeftStatistics.resize(locallyBestTableIndices.size());
            locallyBestRightStatistics.resize(locallyBestTableIndices.size());
            for( size_t idx = 0; idx < locallyBestTableIndices.size(); ++idx )
            {
                locallyBestLeftStatistics[idx]  = leftStatistics[locallyBestTableIndices[idx]];
                locallyBestRightStatistics[idx] = rightStatistics[locallyBestTableIndices[idx]];
            }
        }

        template<typename TSplitCriterion>
        size_t DetermineGloballyBestIndex(const std::vector<TFeature>& features,
                                          const TSplitCriterion& criterion,
                                          std::vector<typename TSplitCriterion::SufficientStatistics>& locallyBestLeftStatistics,
                                          std::vector<typename TSplitCriterion::SufficientStatistics>& locallyBestRightStatistics,
                                          const std::vector<size_t>& locallyBestTableIndices,
                                          size_t &numLeft, TLabel& averageLeft,
                                          size_t &numRight, TLabel& averageRight,
                                          TLabel& averageAll, typename TSplitCriterion::TValue& bestScore) const
        {
            // Typedefs
            typedef typename TLabel::ValueType TValue;
            typedef typename TSplitCriterion::SufficientStatistics TSufficientStatistics;

            // Reduce the sufficient statistics that are locally best to finally choose the
            // one that is globally best among these.
            std::vector<float> localBuffer((locallyBestLeftStatistics.size() + locallyBestRightStatistics.size()) * TSufficientStatistics::Size),
                globalBuffer((locallyBestLeftStatistics.size() + locallyBestRightStatistics.size()) * TSufficientStatistics::Size);

            auto outptr = Criteria::SerializeStatistics(locallyBestLeftStatistics, &localBuffer[0]);
            Criteria::SerializeStatistics(locallyBestRightStatistics, outptr);

            //const auto t1 = GetTickCount64();
            boost::mpi::all_reduce(MPI::Communicator(), &localBuffer[0], localBuffer.size(), &globalBuffer[0], std::plus<float>());

            auto inptr = Criteria::DeserializeStatistics(&globalBuffer[0], locallyBestLeftStatistics);
            Criteria::DeserializeStatistics(inptr, locallyBestRightStatistics);

            // From the global sufficient statistics for the few promising candidates, determine the best one
            std::vector<TValue> globalScores(locallyBestLeftStatistics.size());
            #pragma omp parallel for
            for( int idx = 0; idx < globalScores.size(); ++idx )
                globalScores[idx] = criterion(nodeIndex, locallyBestLeftStatistics[idx], locallyBestRightStatistics[idx]);

            const auto bestIndex = std::distance(globalScores.begin(),
                                                 std::max_element(globalScores.begin(), globalScores.end()));

            numLeft      = locallyBestLeftStatistics[bestIndex].NumPoints();
            averageLeft  = criterion.Average(locallyBestLeftStatistics[bestIndex]);
            numRight     = locallyBestRightStatistics[bestIndex].NumPoints();
            averageRight = criterion.Average(locallyBestRightStatistics[bestIndex]);
            averageAll   = criterion.Average( locallyBestLeftStatistics[bestIndex] + locallyBestRightStatistics[bestIndex] );
            bestScore    = globalScores[bestIndex];

            // Convert the index in low-dimensional space to the index in the global table
            return locallyBestTableIndices[bestIndex];
        }

        template<typename TSplitCriterion>
        TFeature BestSplitNBestMPI(const std::vector<TFeature>& features,
                                   const TSplitCriterion& criterion,
                                   size_t &numLeft, TLabel& averageLeft,
                                   size_t &numRight, TLabel& averageRight,
                                   TLabel& averageAll, typename TSplitCriterion::TValue& bestScore) const
        {
            // Typedefs
            typedef typename TLabel::ValueType TValue;
            typedef typename TSplitCriterion::SufficientStatistics TSufficientStatistics;

            // Constants
            const size_t FeatureBlockSize = TFeature::NumThresholdTests + 1;

            // Determine bin boundaries
            std::vector<TValue> rightBinBoundaries;
            DetermineBinBoundaries(features, rightBinBoundaries);

            // Determine the sufficient statistics for this process
            std::vector<TSufficientStatistics> statistics;
            DetermineLocalSufficientStatistics(features, rightBinBoundaries, statistics);

            // For each process, find the MPI_N_BEST locally best (feature, threshold) table indices,
            // recording the statistics of the left and right banches for latter use
            std::vector<TSufficientStatistics> locallyBestLeftStatistics, locallyBestRightStatistics;
            std::vector<size_t> locallyBestTableIndices;
            DetermineLocallyBestFeatures(features, criterion, statistics, rightBinBoundaries,
                                         locallyBestLeftStatistics, locallyBestRightStatistics, locallyBestTableIndices);

            // Now, from these locally best (feature, threshold) table indices, find the one that is globally the best
            const auto bestTableIdx = DetermineGloballyBestIndex(features, criterion,
                                      locallyBestLeftStatistics, locallyBestRightStatistics,    locallyBestTableIndices,
                                      numLeft, averageLeft, numRight, averageRight, averageAll, bestScore);

            // And return the feature corresponding to the best (feature, threshold) table index
            return features[bestTableIdx/FeatureBlockSize].WithThreshold(rightBinBoundaries[bestTableIdx]);
        }
#endif // MPI_N_BEST
#endif // USE_MPI

    public:
        // Constructs the set of data points. Needs a pointer to the node the collected information applies to.
        DataPoints(int n, typename TreeRef<TFeature, TRegressionTreeNode>::iterator_base it, const VecCRef<Vector2D<int>>& off)
            : nodeIndex(n), leaf(it), offsets(off)
        {
        }

        // Add a particular data point that was sorted into our node.
        void Add(const TLabel& label, int x, int y, const TPrep& prep)
        {
            entries.push_back(Entry(label, x, y, prep));
        }

        size_t NumPoints() const
        {
            return entries.size();
        }

        int Depth() const
        {
            return TreeRef<TFeature, TRegressionTreeNode>::Depth(leaf);
        }

        // Check if the leaf is pure
        // TODO: ADAPT FOR MPI
        template<typename TSplitCriterion>
        bool IsPure(const TSplitCriterion& criterion, typename TSplitCriterion::TValue& score) const
        {
#ifdef USE_MPI
            return false;
#endif
            if(entries.size() < 2)
                return true;

            std::vector<const TLabel*> all;
            std::for_each(entries.begin(), entries.end(), [&](const Entry & entry)
            {
                all.push_back(&entry.label);
            });
            score = typename TSplitCriterion::TValue(0);
            return criterion.IsPure(all);
        }

        // Splits the associated node into two children according to the feature that gives the largest gain according
        // to some criterion. If no useful split is found, this step is skipped. The associated node will be marked
        // as 'trained', such that it won't be considered for splitting again in one of the rounds to come.
        // If the node is split, useful information is passed on to its children, such as the average and the number
        // of datapoints that would be sorted into them. The datapoints themselves will be discarded.
        template<typename TSplitCriterion>
        bool AddSplit(const std::vector<TFeature>& features, const TreeRef<TFeature, TRegressionTreeNode>& tree,
                      const TSplitCriterion& criterion, typename TSplitCriterion::TValue& bestScore)
        {
            // Mark this leaf as trained, so we won't consider splitting it again in upcoming iterations.
            leaf->data.trained = true;

            // Check if the points that fall into our node are pure; if so, we will not split the node.
            if(IsPure(criterion, bestScore))
            {
                TMonitor::Report("INFO: Node is pure, will not split.\n");
                return false;
            }

            // Determine the best split according to our criterion
            TLabel averageLeft, averageRight, averageAll;
            size_t numLeft, numRight;
            leaf->feature = BestSplit(features, criterion, numLeft, averageLeft, numRight, averageRight, averageAll, bestScore);

            // If we are the root node, set the average (has not been set by parent, because there is none)
            if(tree.size() == 1)
                leaf->data.average = averageAll;

#ifndef TRAINING_TOLERATE_DEGENERATE_SPLITS
            // Check for degenerate split
            if(numLeft == 0 || numRight == 0)
            {
                TMonitor::Report("WARN: Unable to find split for set of %u points\n", (numLeft + numRight));
                return false;
            }
#endif

            // Split the leaf and pass the children the number of points that went into them and the average of these points
            tree.append_child(leaf, TNodeData(TFeature(), TRegressionTreeNode(numLeft, averageLeft)));
            tree.append_child(leaf, TNodeData(TFeature(), TRegressionTreeNode(numRight, averageRight)));
            return true;
        }
    };

    // Perform one round of regression tree growing. This works roughly as follows:
    //   1) Choose the eligible leaf nodes
    //       (i.e. those that can be split, because they receive a sufficient number points, haven't been considered before, etc.)
    //   2) Sort the data points into temporarily allocated storage (a DataPoints object) for each selected leaf.
    //       A data point consists of a particular position within an image, along with the associated label and
    //       the image data we condition on.
    //   3) For each eligible leaf,
    //       Find the most useful feature and split the leaf into a left and a right child according to that feature.
    //       Store information that is of permanent use (such as the average of the labels) in each child.
    //   4) Discard the temporarily allocated storage. It will be rebuilt for a new set of leaves during the next round.
    // If no eligible leaves are found, this is signalled to the caller by returning a false value.
    //
    // The design enables us to sample from the training data and from the feature space at each round.
    template <bool UseExplicitThresholding, typename TMonitor, typename TFeature, typename TPointSampler, typename TSplitCriterion>
    bool GrowRegressionTree(
        TreeRef<TFeature, RegressionTreeNode<typename TPointSampler::TLabel>> tree,
        const TPointSampler& pointSampler,
        const std::vector<TFeature>& features,
        size_t nDepthLevels,
        size_t nMinDataPointsForSplitConsideration,
        const TSplitCriterion& criterion)
    {
        typedef typename TSplitCriterion::TValue        TValue;
        typedef typename TPointSampler::TLabel          TLabel;
        typedef DataPoints < TFeature, TLabel, TMonitor,
                UseExplicitThresholding >     TTableStoreValue;
        typedef std::shared_ptr<TTableStoreValue>       TTableStore;
        typedef RegressionTreeNode<TLabel>              TRegressionTreeNode;

        // Initialize the tree if it is still empty
        if(tree.begin_leaf() == tree.end_leaf())
        {
            NodeData<TFeature, TRegressionTreeNode> head;
#ifdef USE_MPI
            size_t globalNumPoints;
            boost::mpi::all_reduce(MPI::Communicator(), pointSampler.TotalNumPoints(), globalNumPoints, std::plus<size_t>());
            head.data.numDataPoints = globalNumPoints;
#else
            head.data.numDataPoints = pointSampler.TotalNumPoints();
#endif
            tree.set_head(head);
        }

        // Choose the leaf nodes that will be split and prepare the temporary leaf-local storage for this iteration
        std::vector<TTableStore> selectedNodes;
        TreeTableT<TFeature, TTableStore> treeTable;
        size_t index = 0; // index according to breadth-first-order traversal by Fill()
        treeTable.template Fill<TRegressionTreeNode>(tree, [&](const typename TreeCRef<TFeature, TRegressionTreeNode>::iterator_base & it)->TTableStore
        {
            const size_t nDepth = TreeRef<TFeature, TRegressionTreeNode>::depth(it);
            if( it->data.trained == false && it.number_of_children() == 0 &&
            (nDepth+1) < nDepthLevels && it->data.numDataPoints >= nMinDataPointsForSplitConsideration)
            {
                selectedNodes.push_back(TTableStore(new TTableStoreValue(index++, it, pointSampler.Offsets())));
                return selectedNodes.back();
            }
            else {
                index++;
                return TTableStore(NULL);
            }
        });
        TMonitor::Display("  > Found %u leaves (out of %u nodes) that are eligible for splitting\n", selectedNodes.size(), index);

        // No more leaves that are eligible for splitting
        if(selectedNodes.size() == 0)
            return false;
        // Sort the data points into the respective leaves
        size_t numPoints = pointSampler.AddPoints([&](const TLabel & label,
                           int x, int y,
                           const typename TFeature::PreProcessType & prep)->bool
        {
            TTableStore store = treeTable.GetLeafData(x, y, prep, pointSampler.Offsets());

            if(store)
            {
                #pragma omp critical
                {
                    store->Add(label, x, y, prep);
                }
            }

            return store ? true : false;
        });

#ifdef USE_MPI
        size_t globalNumPoints;
        boost::mpi::all_reduce(MPI::Communicator(), numPoints, globalNumPoints, std::plus<size_t>());
        TMonitor::Display("  > Sorted %u points into these leaves (%u in this process).\n",  globalNumPoints, numPoints);
#else
        TMonitor::Display("  > Sorted %u points into these leaves.\n",  numPoints);
#endif
        // Now, for each chosen leaf, determine the most promising split.
        TMonitor::Display("  > Processing leaves: ");
        TValue score = TValue();

        for(size_t n = 0; n < selectedNodes.size(); ++n)
        {
            TMonitor::Display(".");
            TValue nscore;
            selectedNodes[n]->AddSplit(features, tree, criterion, nscore);
            score += nscore;
        }
        TMonitor::Display(" done.\n");

#ifdef USE_MPI
        if(globalNumPoints > 0)
            TMonitor::Display("  > Loss at current level is %.9f.\n", (-score / globalNumPoints));
#else
        if(numPoints > 0)
            TMonitor::Display("  > Loss at current level is %.9f.\n", (-score / numPoints));
#endif
        return true;
    }

    // Perform several rounds of regression tree learning. Each round works as outlined in the documentation
    // of the 'GrowRegressionTree' function. The depth of the regression tree is increased in a breadth-first
    // manner until there are no more eligible leaves that could be split.
    template <bool UseExplicitThresholding, typename TMonitor, typename TFeatureSampler, typename TPointSampler, typename TSplitCriterion>
    TreeRef<typename TFeatureSampler::TFeature, RegressionTreeNode<typename TPointSampler::TLabel>> LearnRegressionTree(
                TPointSampler& pointSampler,
                TFeatureSampler& featureSampler,
                int nFeatureCount,
                int nDepthLevels,
                int nMinDataPointsForSplitConsideration,
                const TSplitCriterion& criterion)
    {
        TreeRef<typename TFeatureSampler::TFeature, RegressionTreeNode<typename TPointSampler::TLabel>> tree;
        unsigned level = 1;

        while(level < (unsigned) nDepthLevels)
        {
            // Sample features
            std::vector<typename TFeatureSampler::TFeature> features(nFeatureCount);

            for(int i = 0; i < nFeatureCount; i++)
            {
                try
                {
                    features[i] = featureSampler(level);
                }
                catch(...)
                {
                    features.resize(i);
                    break;
                }
            }

            TMonitor::Report("Processing level %u.\n", level);

            if(!GrowRegressionTree<UseExplicitThresholding, TMonitor>(tree, pointSampler, features, nDepthLevels,
                    nMinDataPointsForSplitConsideration, criterion))
                break;

            level++;
        }

        TMonitor::Report("Finished training regression tree (depth %u).\n", level);
        return tree;
    }

    namespace Detail
    {
        // Provides all unary labels of the given dataset to tree training
        template <typename TTraits>
        class UnaryPointSampler
        {
        public:
            typedef typename TTraits::Feature           TFeature;
            typedef typename TTraits::UnaryGroundLabel  TLabel;
            typedef typename TTraits::PreProcessType    TPrep;
            typedef typename TTraits::DataSampler       TDataSampler;

        protected:
            const TDataSampler& sampler;
            const VecCRef<Vector2D<int>> offsets;

        public:

            UnaryPointSampler(const TDataSampler & sampler_) : sampler(sampler_), offsets(1) {}

            VecCRef<Vector2D<int>> Offsets() const
            {
                return offsets;
            }

            size_t TotalNumPoints() const
            {
                size_t numPoints = 0;

                for(size_t i = 0; i < sampler.GetImageCount(); i++)
                {
                    auto groundTruthImage = sampler.GetGroundTruthImage(i);
                    numPoints += groundTruthImage.Width() * groundTruthImage.Height();
                }
                return numPoints;
            }

            size_t AddPoints(const std::function<bool (const TLabel&, int, int, const TPrep&)>& addOp) const
            {
                const size_t numImages = sampler.GetImageCount();
                size_t       numPoints = 0;

                for(size_t i = 0; i < numImages; i++)
                {
                    const auto trainingImage = sampler.GetInputImage(i);
                    const auto groundTruthImage = sampler.GetGroundTruthImage(i);
                    const auto prep = TFeature::PreProcess(trainingImage);
                    const int cx = groundTruthImage.Width(), cy = groundTruthImage.Height();

                    for(int y = 0; y < cy; ++y)
                    {
                        for(int x = 0; x < cx; ++x)
                        {
                            if(addOp(groundTruthImage(x, y), x, y, prep))
                                ++numPoints;
                        }
                    }
                }
                return numPoints;
            }
        };

        // Provides a subset of the unary labels of the given dataset to tree training
        template <typename TTraits>
        class UnaryPointSubsampler
        {
        public:
            typedef typename TTraits::Feature           TFeature;
            typedef typename TTraits::UnaryGroundLabel  TLabel;
            typedef typename TTraits::PreProcessType    TPrep;
            typedef typename TTraits::DataSampler       TDataSampler;

        protected:
            const TDataSampler& sampler;
            const VecCRef<Vector2D<int>> offsets;

        public:
            UnaryPointSubsampler(const TDataSampler & sampler_) : sampler(sampler_), offsets(1) {}

            VecCRef<Vector2D<int>> Offsets() const
            {
                return offsets;
            }

            size_t TotalNumPoints() const
            {
                size_t numPoints = 0;

                for(size_t i = 0; i < sampler.GetImageCount(); i++)
                    numPoints += sampler.GetSubsampledVariables(i).size();
                return numPoints;
            }

            size_t AddPoints(const std::function<bool (const TLabel&, int, int, const TPrep&)>& addOp) const
            {
                const size_t  numImages = sampler.GetImageCount();
                size_t        numPoints = 0;

                for(size_t i = 0; i < numImages; i++)
                {
                    const auto trainingImage    = sampler.GetInputImage(i);
                    const auto prep             = TFeature::PreProcess(trainingImage);
                    const auto groundTruthImage = sampler.GetGroundTruthImage(i);
                    const auto pixelSamples     = sampler.GetSubsampledVariables(i);

                    for(size_t s = 0; s < pixelSamples.size(); ++s)
                    {
                        const auto point = pixelSamples[s];

                        if(addOp(groundTruthImage(point.x, point.y), point.x, point.y, prep))
                            ++numPoints;
                    };
                }
                return numPoints;
            }
        };

        // Provides all pairwise labels of the pairwise factor specified via 'offsets' and contained
        // in the specified dataset to tree training.
        template <typename TTraits>
        class PairwisePointSampler
        {
        public:
            typedef typename TTraits::Feature             TFeature;
            typedef typename TTraits::PairwiseGroundLabel TLabel;
            typedef typename TTraits::PreProcessType      TPrep;
            typedef typename TTraits::DataSampler         TDataSampler;

        protected:
            const TDataSampler& sampler;
            const VecCRef<Vector2D<int>> offsets;
            const Rect<int> deflateRect;

        public:
            PairwisePointSampler(const TDataSampler & sampler_, const VecCRef<Vector2D<int>>& offsets_)
                : sampler(sampler_), offsets(offsets_), deflateRect(Utility::ComputeDeflateRect(offsets_))
            {
            }

            VecCRef<Vector2D<int>> Offsets() const
            {
                return offsets;
            }

            size_t TotalNumPoints() const
            {
                size_t numPoints = 0;

                for(size_t i = 0; i < sampler.GetImageCount(); i++)
                {
                    auto groundTruthImage = sampler.GetGroundTruthImage(i);
                    auto processRect      = Utility::ComputeProcessRect(deflateRect, groundTruthImage.Width(), groundTruthImage.Height());
                    numPoints            += (processRect.bottom - processRect.top) * (processRect.right - processRect.left);
                }
                return numPoints;
            }

            size_t AddPoints(const std::function<bool (const TLabel&, int, int, const TPrep&)>& addOp) const
            {
                const size_t numImages = sampler.GetImageCount();
                size_t       numPoints = 0;

                for(size_t i = 0; i < numImages; i++)
                {
                    const auto trainingImage    = sampler.GetInputImage(i);
                    const auto groundTruthImage = sampler.GetGroundTruthImage(i);
                    const auto prep             = TFeature::PreProcess(trainingImage);
                    const auto processRect      = Utility::ComputeProcessRect(deflateRect, groundTruthImage.Width(), groundTruthImage.Height());

                    for(int y = processRect.top; y < processRect.bottom; ++y)
                    {
                        for(int x = processRect.left; x < processRect.right; ++x)
                        {
                            const TLabel label = TLabel(groundTruthImage(x + offsets[0].x, y + offsets[0].y),
                                                        groundTruthImage(x + offsets[1].x, y + offsets[1].y));

                            if(addOp(label, x, y, prep))
                                ++numPoints;
                        }
                    }
                }
                return numPoints;
            }
        };

        // Provides a subset of the pairwise labels of the pairwise factor specified via 'offsets' and
        // contained in the specified dataset to tree training.
        // For each subsampled variable, we visit the factor instance which has this variable as its
        // first connected variable, provided all connected variables are within the image bounds.
        template <typename TTraits>
        class PairwisePointSubsampler
        {
        public:
            typedef typename TTraits::Feature             TFeature;
            typedef typename TTraits::PairwiseGroundLabel TLabel;
            typedef typename TTraits::PreProcessType      TPrep;
            typedef typename TTraits::DataSampler         TDataSampler;

        protected:
            const TDataSampler&          sampler;
            const VecCRef<Vector2D<int>> offsets;
            const Rect<int>              deflateRect;

        public:
            PairwisePointSubsampler(const TDataSampler & sampler_, const VecCRef<Vector2D<int>>& offsets_)
                : sampler(sampler_), offsets(offsets_), deflateRect(Utility::ComputeDeflateRect(offsets_))
            {
            }

            VecCRef<Vector2D<int>> Offsets() const
            {
                return offsets;
            }

            size_t TotalNumPoints() const
            {
                size_t numPoints = 0;

                for(size_t i = 0; i < sampler.GetImageCount(); i++)
                {
                    auto groundTruthImage = sampler.GetGroundTruthImage(i);
                    auto processRect      = Utility::ComputeProcessRect(deflateRect, groundTruthImage.Width(), groundTruthImage.Height());
                    auto pixelSamples     = sampler.GetSubsampledVariables(i);

                    for(size_t s = 0; s < pixelSamples.size(); ++s)
                        numPoints += processRect.PtInRect(pixelSamples[s]);
                }
                return numPoints;
            }

            size_t AddPoints(const std::function<bool (const TLabel&, int, int, const TPrep&)>& addOp) const
            {
                size_t numPoints = 0;

                for(size_t i = 0; i < sampler.GetImageCount(); i++)
                {
                    const auto trainingImage    = sampler.GetInputImage(i);
                    const auto groundTruthImage = sampler.GetGroundTruthImage(i);
                    const auto prep             = TFeature::PreProcess(trainingImage);
                    const auto processRect      = Utility::ComputeProcessRect(deflateRect, groundTruthImage.Width(), groundTruthImage.Height());
                    const auto pixelSamples     = sampler.GetSubsampledVariables(i);

                    for(size_t s = 0; s < pixelSamples.size(); ++s)
                    {
                        const auto point = pixelSamples[s];

                        if(processRect.PtInRect(point))
                        {
                            const TLabel label = TLabel(groundTruthImage(point.x + offsets[0].x, point.y + offsets[0].y),
                                                        groundTruthImage(point.x + offsets[1].x, point.y + offsets[1].y));

                            if(addOp(label, point.x, point.y, prep))
                                ++numPoints;
                        }
                    }
                }
                return numPoints;
            }
        };

        // Allows for convenient static_assert expressions to verify that a proper criterion is used;
        // The GradientNormCriterion is not suitable for training via the legacy LearnRegressionTree() functions.
        template<typename TSplitCritTag>
        struct CriterionCheck
        {
            static const int ok = 1;
        };

        template<>
        struct CriterionCheck<GradientNormCriterion>
        {
            static const int ok = 0;
        };
    }

    // Learns a unary regression tree from all data points provided by 'dataSampler'.
    //
    // Options:
    //   - nFeatureCount   How many features to sample at each split
    //   - nDepthLevels    Maximum depth of the tree to be trained
    //   - MinDataPointsForSplitConsideration
    //                     The minmum number of data points that must fall into a leave in
    //                       order for it to be considered for splitting.
    //   - purityEpsilon   A small number that determines the maximum distance between data points
    //                       in order for them to be considered pure. Pure node are not split any further.
    template <typename TTraits>
    typename TTraits::UnaryTreeRef
    LearnUnaryRegressionTree(const typename TTraits::DataSampler& dataSampler,
                             int nFeatureCount,
                             int nDepthLevels,
                             int nMinDataPointsForSplitConsideration = 8,
                             typename TTraits::ValueType purityEpsilon = 0.0)
    {
        static_assert(Detail::CriterionCheck<typename TTraits::UnarySplitCriterionTag>::ok,
                      "Unsupported unary split criterion specified in model traits.");
        typedef typename TTraits::Monitor TMonitor;
        Detail::UnaryPointSampler<TTraits> pointSampler(dataSampler);
        typename TTraits::FeatureSampler featureSampler;
        typename TTraits::UnarySplitCriterion criterion(purityEpsilon);
        return LearnRegressionTree<TTraits::UseExplicitThresholding, TMonitor>(pointSampler, featureSampler, nFeatureCount, nDepthLevels,
                nMinDataPointsForSplitConsideration, criterion);
    }

    // Learns a random forest from all data points provided by provided by 'dataSampler'.
    // The data sampler class must implement the subsampling interface.
    //
    // Options:
    //   - nTrees          How many trees to add to the forest
    //   - nFeatureCount   How many features to sample at each split
    //   - nDepthLevels    Maximum depth of the tree to be trained
    //   - MinDataPointsForSplitConsideration
    //                     The minmum number of data points that must fall into a leave in
    //                       order for it to be considered for splitting.
    //   - purityEpsilon   A small number that determines the maximum distance between data points
    //                       in order for them to be considered pure. Pure node are not split any further.
    template <typename TTraits>
    void
    LearnRegressionForest(typename TTraits::UnaryTreeRefVector& forest,
                          const typename TTraits::DataSampler& dataSampler,
                          int nTrees,
                          int nFeatureCount,
                          int nDepthLevels,
                          int nMinDataPointsForSplitConsideration = 8,
                          typename TTraits::ValueType purityEpsilon = 0.0)
    {
        forest.resize(nTrees);

        for(int t = 0; t < nTrees; ++t)
        {
            forest[t] = LearnUnaryRegressionTree<TTraits>(dataSampler, nFeatureCount, nDepthLevels, nMinDataPointsForSplitConsideration, purityEpsilon);
        }
    }

    // Learns a unary regression tree from a number of subsampled data points provided by 'dataSampler'.
    // The data sampler class must implement the subsampling interface.
    //
    // Options:
    //   - nFeatureCount   How many features to sample at each split
    //   - nDepthLevels    Maximum depth of the tree to be trained
    //   - MinDataPointsForSplitConsideration
    //                     The minmum number of data points that must fall into a leave in
    //                       order for it to be considered for splitting.
    //   - purityEpsilon   A small number that determines the maximum distance between data points
    //                       in order for them to be considered pure. Pure node are not split any further.
    template <typename TTraits>
    typename TTraits::UnaryTreeRef
    LearnUnaryRegressionTreeSubsample(const typename TTraits::DataSampler& dataSampler,
                                      int nFeatureCount,
                                      int nDepthLevels,
                                      int nMinDataPointsForSplitConsideration = 8,
                                      typename TTraits::ValueType purityEpsilon = 0.0)
    {
        static_assert(Detail::CriterionCheck<typename TTraits::UnarySplitCriterionTag>::ok,
                      "Unsupported unary split criterion specified in model traits.");
        typedef typename TTraits::Monitor TMonitor;
        Detail::UnaryPointSubsampler<TTraits> pointSampler(dataSampler);
        typename TTraits::FeatureSampler featureSampler;
        typename TTraits::UnarySplitCriterion criterion(purityEpsilon);
        return LearnRegressionTree<TTraits::UseExplicitThresholding, TMonitor>(pointSampler, featureSampler, nFeatureCount, nDepthLevels,
                nMinDataPointsForSplitConsideration, criterion);
    }

    // Learns a random forest from a number of subsampled data points provided by 'dataSampler'.
    // The data sampler class must implement the subsampling interface.
    //
    // Options:
    //   - nTrees          How many trees to add to the forest
    //   - nFeatureCount   How many features to sample at each split
    //   - nDepthLevels    Maximum depth of the tree to be trained
    //   - MinDataPointsForSplitConsideration
    //                     The minmum number of data points that must fall into a leave in
    //                       order for it to be considered for splitting.
    //   - purityEpsilon   A small number that determines the maximum distance between data points
    //                       in order for them to be considered pure. Pure node are not split any further.
    template <typename TTraits>
    void
    LearnRegressionForestSubsample(typename TTraits::UnaryTreeRefVector& forest,
                                   const typename TTraits::DataSampler& dataSampler,
                                   int nTrees,
                                   int nFeatureCount,
                                   int nDepthLevels,
                                   int nMinDataPointsForSplitConsideration = 8,
                                   typename TTraits::ValueType purityEpsilon = 0.0)
    {
        forest.resize(nTrees);

        for(int t = 0; t < nTrees; ++t)
        {
            forest[t] = LearnUnaryRegressionTreeSubsample<TTraits>(dataSampler, nFeatureCount, nDepthLevels, nMinDataPointsForSplitConsideration, purityEpsilon);
        }
    }

    // Learns a pairwise regression tree from all data points provided by 'dataSampler'.
    //
    // Options:
    //   - nFeatureCount   How many features to sample at each split
    //   - nDepthLevels    Maximum depth of the tree to be trained
    //   - MinDataPointsForSplitConsideration
    //                     The minmum number of data points that must fall into a leave in
    //                       order for it to be considered for splitting.
    //   - purityEpsilon   A small number that determines the maximum distance between data points
    //                       in order for them to be considered pure. Pure node are not split any further.
    template <typename TTraits>
    typename TTraits::PairwiseTreeRef
    LearnPairwiseRegressionTree(const VecCRef<Vector2D<int>>& variables,
                                const typename TTraits::DataSampler& dataSampler,
                                int nFeatureCount,
                                int nDepthLevels,
                                int nMinDataPointsForSplitConsideration = 8,
                                typename TTraits::ValueType purityEpsilon = 0.0)
    {
        static_assert(Detail::CriterionCheck<typename TTraits::PairwiseSplitCriterionTag>::ok,
                      "Unsupported pairwise split criterion specified in model traits.");
        assert(variables.size() == 2 && variables[0].x == 0 && variables[0].y == 0);
        typedef typename TTraits::Monitor TMonitor;
        Detail::PairwisePointSampler<TTraits> pointSampler(dataSampler, variables);
        typename TTraits::FeatureSampler featureSampler;
        typename TTraits::PairwiseSplitCriterion criterion(purityEpsilon);
        return LearnRegressionTree<TTraits::UseExplicitThresholding, TMonitor>(pointSampler, featureSampler, nFeatureCount, nDepthLevels,
                nMinDataPointsForSplitConsideration, criterion);
    }

    // Learns a pairwise regression tree from a number of subsampled data points provided by 'dataSampler'.
    // The data sampler class must implement the subsampling interface.
    //
    // Options:
    //   - nFeatureCount   How many features to sample at each split
    //   - nDepthLevels    Maximum depth of the tree to be trained
    //   - MinDataPointsForSplitConsideration
    //                     The minmum number of data points that must fall into a leave in
    //                       order for it to be considered for splitting.
    //   - purityEpsilon   A small number that determines the maximum distance between data points
    //                       in order for them to be considered pure. Pure node are not split any further.
    template <typename TTraits>
    typename TTraits::PairwiseTreeRef
    LearnPairwiseRegressionTreeSubsample(const VecCRef<Vector2D<int>>& variables,
                                         const typename TTraits::DataSampler& dataSampler,
                                         int nFeatureCount,
                                         int nDepthLevels,
                                         int nMinDataPointsForSplitConsideration = 8,
                                         typename TTraits::ValueType purityEpsilon = 0.0)
    {
        static_assert(Detail::CriterionCheck<typename TTraits::PairwiseSplitCriterionTag>::ok,
                      "Unsupported pairwise split criterion specified in model traits.");
        assert(variables.size() == 2 && variables[0].x == 0 && variables[0].y == 0);
        typedef typename TTraits::Monitor TMonitor;
        Detail::PairwisePointSubsampler<TTraits> pointSampler(dataSampler, variables);
        typename TTraits::FeatureSampler featureSampler;
        typename TTraits::PairwiseSplitCriterion criterion(purityEpsilon);
        return LearnRegressionTree<TTraits::UseExplicitThresholding, TMonitor>(pointSampler, featureSampler, nFeatureCount, nDepthLevels,
                nMinDataPointsForSplitConsideration, criterion);
    }

    // The following array of functions can be used to re-estimate the means at the leaves of a regression tree.
    // This is useful if the full dataset is too large to train trees from. Re-estimation of the means is very
    // efficient, and hence can be done for the full dataset, reducing the danger of overfitting.

    // Resets the data point observations for each leaf.
    template<typename TTraits>
    void SetMeansAtLeavesToZero(typename TTraits::UnaryTreeRef tree)
    {
        // Clear leaf data
        for(auto it = tree.begin_leaf(); it != tree.end_leaf(); ++it)
        {
            it->data.numDataPoints = 0;
            it->data.average       = typename TTraits::UnaryGroundLabel();
        }
    }

    // Sorts the given data points into the leafs of the tree, for latter estimation of the means.
    template<typename TTraits>
    void AccumulateMeansAtLeaves(typename TTraits::UnaryTreeRef tree, const typename TTraits::DataSampler& dataSampler)
    {
        // Accumulate points
        Detail::UnaryPointSampler<TTraits> sampler(dataSampler);
        auto nPoints = sampler.AddPoints([&](const typename TTraits::UnaryGroundLabel & label, int x, int y,
                                             const typename TTraits::PreProcessType & prep)->bool
        {
            auto it = tree.goto_leaf(x, y, prep, sampler.Offsets());
            it->data.numDataPoints += 1;
            it->data.average       += label;
            return true;
        });
    }

    // Estimates the means at the leaves of the tree from the data points that were previously sorted into the leaves.
    // NB: Each leaf must have seen at least one data point, otherwise the mean cannot be estimated and an exception
    // will be thrown.
    template<typename TTraits>
    void EstimateMeansAtLeaves(typename TTraits::UnaryTreeRef tree)
    {
        // Build average
        for(auto it = tree.begin_leaf(); it != tree.end_leaf(); ++it)
        {
            if(it->data.numDataPoints == 0)
                throw std::exception("Data point count of zero at leaf, cannot estimate mean");

            it->data.average /= it->data.numDataPoints;
        }
    }

    // Forest version of the above
    template<typename TTraits>
    void SetMeansAtLeavesToZero(typename TTraits::UnaryTreeRefVector& forest)
    {
        const int nt = static_cast<int>(forest.size());
        #pragma omp parallel for

        for(int t = 0; t < nt; ++t)
            SetMeansAtLeavesToZero<TTraits>(forest[t]);
    }

    // Forest version of the above
    template<typename TTraits>
    void AccumulateMeansAtLeaves(typename TTraits::UnaryTreeRefVector& forest, const typename TTraits::DataSampler& dataSampler)
    {
        const int nt = static_cast<int>(forest.size());
        #pragma omp parallel for

        for(int t = 0; t < nt; ++t)
            AccumulateMeansAtLeaves<TTraits>(forest[t], dataSampler);
    }

    // Forest version of the above
    template<typename TTraits>
    void EstimateMeansAtLeaves(typename TTraits::UnaryTreeRefVector& forest)
    {
        const int nt = static_cast<int>(forest.size());
        #pragma omp parallel for

        for(int t = 0; t < nt; ++t)
            EstimateMeansAtLeaves<TTraits>(forest[t]);
    }

} // namespace Training


#endif // H_RTF_TRAINING_H
