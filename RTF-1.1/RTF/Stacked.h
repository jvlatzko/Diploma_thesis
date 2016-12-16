/* This file is part of the "Regression Tree Fields" (RTF) source code distribution,
 * obtained from http://research.microsoft.com/downloads.
 * It is provided to you under the terms of the Microsoft Research License Agreement
 * (MSR-LA). Please see License.txt for details.
 *
 *
 * File: Stacked.h
 * Exposes an object-oriented interface to STACKED training of regression tree fields.
 *
 */

#ifndef H_RTF_STACKED_H
#define H_RTF_STACKED_H

#include <string>
#include <random>

#include "Types.h"
#include "Classify.h"
#include "Learning.h"
#include "Serialization.h"
#include "LinearOperator.h"

// A stacked regression tree field consists of several basic regression tree fields (see Basic.h),
// stacked on top of each other, each of which can use the prediction of the previous RTF as an
// additional input feature. These RTFs form a "prediction cascade".
// As an example, in a natural image denoising task, the first RTF would remove the dominant noise,
// whereas a second RTF, building on the prediction of the first, would remove all remaining artefacts
// that the first RTF failed to remove.
// Such cascades can often be surprisingly powerful, as each additional model only needs to correct
// any remaining biases in the predictions of the previous model.
//
// The interface of the Stacked::RTF class,
//
//   namespace Stacked
//   {
//      template < typename TFeatureSampler,
//                 typename TDataSampler,
//                 typename TSplitCritTag           = SquaredResidualsCriterion,
//                 bool UseBasis                    = false,
//                 bool UseExplicitThresholdTesting = false,
//                 typename TPrior                  = NullPrior,
//                 typename TMonitor                = Monitor::DefaultMonitor,
//                 int CachingMode                  = WEIGHTS_AND_BASIS_PRECOMPUTED,
//                 typename TLinearOperatorWeights  = LinearOperator::DefaultWeights<typename TDataSampler::UnaryGroundLabel::ValueType> >
//      class RTF;
//   }
//
// is very similar to that of a basic RTF. Please see Basic.h for a full description of each of the
// above 'model traits' template parameters.
//
// The main difference is that the TDataSampler class used in a Stacked::RTF is required to implement
// one additional method,
//
//   class Dataset
//   {
//      /* ... */
//
//      void InitializeForCascadeLevel(size_t level,
//                                     VecCRef<ImageRefC<UnaryGroundLabel>> previousPrediction) const;
//
//      /* ... */
//   };
//
// which is invoked by the RTF code before tructure and model parameters of each each layer of the
// prediction cascade are trained. For instance, for the first layer, the 'level' parameter would be 0,
// and the vector of previous predictions would be empty.  Starting from the second layer, the same
// vector would contain the predictions by the previous layer on the training data.  This gives the
// user the opportunity to store such previous predictions in the dataset class, allowing the next
// layer in the cascade to extract features from them.
//
// To add several basic RTF models to the stacked cascade, the method
//
//   LearnOneMore[Discriminative]()
//
// can be invoked multiple times. For each invocation, one RTF model is added to the ensemble. It is
// often a good idea, if sufficient training data is available, to use disjoint training sets for each
// invocation of the LearnOneMore() method. Alternatively, a re-sampling strategy should be used,
// whereby only a fraction of the training images is sampled with replacement from the overall pool
// of training data. This strategy typically aids generalization significantly, since prediction
// cascades are prone to overfitting.
//
// Please see below for a description of all methods of the Stacked::RTF class.
//
namespace Stacked
{

    namespace Detail
    {
        // Convenience class that adds a subsampling method to any user-supplied dataset class.
        template<typename TWrapped>
        class DatasetAdapter
        {
        public:
            typedef typename TWrapped::UnaryGroundLabel     UnaryGroundLabel;     // INTERFACE: the type to be used for unary groundtruth labels
            typedef typename TWrapped::PairwiseGroundLabel  PairwiseGroundLabel;  // INTERFACE: the type to be used for pairwise groundtruth labels
            typedef typename TWrapped::InputLabel           InputLabel;           // INTERFACE: the label type of input images

        private:
            const TWrapped&                                 original;

            mutable std::vector<VecRef<Vector2D<int>>>      variableSubsamples;
            mutable std::mt19937                            mt;
            mutable std::uniform_int_distribution<int>      dpos;

            double                                          pixelFraction;

        public:

            DatasetAdapter(const TWrapped& original_,
                           double pixelFraction_ = 0.5)
                : original(original_), variableSubsamples(original.GetImageCount()), pixelFraction(pixelFraction_)
            {
                ResampleVariables();
            }

            // INTERFACE: returns the number of images in the dataset
            size_t GetImageCount() const
            {
                return original.GetImageCount();
            }

            // INTERFACE: returns the idx'th ground truth image
            ImageRefC<UnaryGroundLabel> GetGroundTruthImage(size_t idx) const
            {
                return original.GetGroundTruthImage(idx);
            }

            // INTERFACE: returns the idx'th input image.
            ImageRefC<InputLabel> GetInputImage(size_t idx) const
            {
                return original.GetInputImage(idx);
            }

            // SUBSAMPLING INTERFACE: returns a number of subsampled data points
            // TODO: Right now, this samples with replacement; Breimann suggests sampling w/o replacement.
            const VecRef<Vector2D<int>>& GetSubsampledVariables(size_t idx) const
            {
                assert(idx < variableSubsamples.size());

                if(variableSubsamples[idx].empty())
                {
                    auto groundTruth  = GetGroundTruthImage(idx);
                    const auto cx     = groundTruth.Width(), cy = groundTruth.Height();
                    int numSamples    = static_cast<int>(cx * cy * pixelFraction + .5);

                    for(int s = 0; s < numSamples; ++s)
                        variableSubsamples[idx].push_back(Vector2D<int>(dpos(mt) % cx, dpos(mt) % cy));
                }

                return variableSubsamples[idx];
            }

            // Causes a new subsample of variables to be drawn upon the next invocation of GetSubsampledVariables()
            void ResampleVariables() const
            {
                const size_t ci = GetImageCount();

                for(size_t i = 0; i < ci; ++i)
                    variableSubsamples[i].resize(0);
            }
        };
    }

    // Class that represents a cascade of RTF models.
    template < typename TFeatureSampler,
             typename TDataSampler,
             typename TSplitCritTag           = SquaredResidualsCriterion,
             bool UseBasis                    = false,
             bool UseExplicitThresholdTesting = false,
             typename TPrior                  = NullPrior,
             typename TMonitor                = Monitor::DefaultMonitor,
             int CachingMode                  = WEIGHTS_AND_BASIS_PRECOMPUTED,
             typename TLinearOperatorWeights  = LinearOperator::DefaultWeights<typename TDataSampler::UnaryGroundLabel::ValueType> >
    class RTF
    {
    public:
        typedef Traits < TFeatureSampler,
                Detail::DatasetAdapter<TDataSampler>,
                TSplitCritTag,
                TSplitCritTag,
                NullPrior,
                NullPrior,
                UseBasis,
                UseExplicitThresholdTesting,
                Monitor::DefaultMonitor,
                CachingMode,
                TLinearOperatorWeights> TTraits;

        typedef typename TTraits::ValueType TValue;

        bool discreteInference;

        static const int LBFGS_M = 64;

    private:
        typedef std::function<typename TTraits::UnaryFactorType()>    UnaryTypeInstantiator;
        typedef std::function<typename TTraits::PairwiseFactorType()> PairwiseTypeInstantiator;
        typedef std::function<typename TTraits::LinearOperatorRef()>  LinearOperatorInstantiator;


        std::vector<typename TTraits::UnaryFactorTypeVector>          utypes;
        std::vector<typename TTraits::PairwiseFactorTypeVector>       ptypes;
        std::vector<typename TTraits::LinearOperatorVector>           ltypes;

        std::vector<Learning::Detail::FactorTypeInfo<TValue>>         uinfos;
        std::vector<UnaryTypeInstantiator>                            ucreat;
        std::vector<Learning::Detail::FactorTypeInfo<TValue>>         pinfos;
        std::vector<PairwiseTypeInstantiator>                         pcreat;
        std::vector<LinearOperatorInstantiator>                       lcreat;

        void AddFactorTypes()
        {
            typename TTraits::UnaryFactorTypeVector ut;

            for(size_t u = 0; u < ucreat.size(); ++u)
                ut.push_back(ucreat[u]());

            utypes.push_back(ut);
            typename TTraits::PairwiseFactorTypeVector pt;

            for(size_t p = 0; p < pcreat.size(); ++p)
                pt.push_back(pcreat[p]());

            ptypes.push_back(pt);

            typename TTraits::LinearOperatorVector lt;

            for( size_t l = 0; l < lcreat.size(); ++l )
                lt.push_back(lcreat[l]());

            ltypes.push_back(lt);
        }

        void ReadModel(std::istream& in)
        {
            size_t usize;
            in >> usize;
            uinfos.resize(usize);
            for( size_t u = 0; u < usize; ++u )
                in >> uinfos[u];

            size_t psize;
            in >> psize;
            pinfos.resize(psize);
            for( size_t p = 0; p < psize; ++p )
                in >> pinfos[p];

            size_t ensembleSize;
            in >> ensembleSize;

            utypes.resize(ensembleSize);
            ptypes.resize(ensembleSize);
            ltypes.resize(ensembleSize);
            for( size_t idx = 0; idx < ensembleSize; ++idx )
                Serialization::ReadModel<TTraits>(in, utypes[idx], ptypes[idx], ltypes[idx]);
        }

        void WriteModel(std::ostream& out) const
        {
            out << uinfos.size() << std::endl;
            for( size_t u = 0; u < uinfos.size(); ++u )
                out << uinfos[u];

            out << pinfos.size() << std::endl;
            for( size_t p = 0; p < pinfos.size(); ++p )
                out << pinfos[p];

            out << EnsembleSize() << std::endl;

            for( size_t idx = 0; idx < EnsembleSize(); ++idx )
                Serialization::WriteModel<TTraits>(out, utypes[idx], ptypes[idx], ltypes[idx]);
        }

        ImageRefC<typename TTraits::UnaryGroundLabel>
        RegressWith(const ImageRefC<typename TTraits::InputLabel>& input, size_t m, size_t maxNumItCG = 10000, TValue residualTolCG = 1e-6) const
        {
            return Classify::Predict<TTraits>(utypes[m], ptypes[m], ltypes[m],
                                              input, residualTolCG, (unsigned) maxNumItCG, discreteInference);
        }

        VecCRef<ImageRefC<typename TTraits::UnaryGroundLabel>>
                RegressWith(const TDataSampler& testdb, size_t m, size_t maxNumItCG = 10000, TValue residualTolCG = 1e-6) const
        {
            VecRef<ImageRefC<typename TTraits::UnaryGroundLabel>> predictions(testdb.GetImageCount());
            for(int i = 0; i < predictions.size(); ++i)
                predictions[i] = RegressWith(testdb.GetInputImage(i), m, maxNumItCG, residualTolCG);

            return predictions;
        }

        void InitializeDataset(const TDataSampler& ds, size_t maxNumItCG, TValue residualTolCG, int upToLevel=-1) const
        {
            if( upToLevel < 0 )
                upToLevel = static_cast<int>(EnsembleSize())-1;

            // Evaluate any previously trained models on the new data and give the user
            // the chance to store the predictions of these models within the dataset.
            for( size_t m = 0; m <= upToLevel; ++m )
            {
                if( m == 0 )
                    ds.InitializeForCascadeLevel(0, VecCRef<ImageRefC<typename TTraits::UnaryGroundLabel>>() );
                else
                    ds.InitializeForCascadeLevel(m, RegressWith(ds, m-1, maxNumItCG, residualTolCG));
            }
        }

    public:

        // Deserialization constructor: Load RTF from the file at the given path
        RTF(const std::string& fname) : discreteInference(false)
        {
            std::cerr << "Reading " << fname << std::endl;
            std::ifstream ifs(fname.c_str());
            if( ! ifs )
                throw std::runtime_error("Could not read input file " + fname);
            ReadModel(ifs);
        }

        // Deserialization constructor: Load RTF from the file at the given path
        RTF(const char* fname) : discreteInference(false)
        {
            std::ifstream ifs(fname);
            if( ! ifs )
                throw std::runtime_error("Could not read input file");
            ReadModel(ifs);
        }

        // Deserialization constructor: Load RTF from provided stream
        RTF(std::istream& in) : discreteInference(false)
        {
            ReadModel(in);
        }

        // Default constructor
        RTF() : discreteInference(false)
        {
        }

        // Number of models in the stacked ensemble
        size_t EnsembleSize() const
        {
            return utypes.size();
        }

        // Adds a unary factor type. The parameters specify the characteristics of the underlying regression
        // tree that is to be trained, as well as regularization of the model parameters.
        void AddUnaryFactorType(int nFeatureCount, int nDepthLevels, int nMinDataPointsForSplitConsideration,
                                TValue smallestEigenValue = TValue(1e-2), TValue largestEigenValue = TValue(1e2),
                                TValue linearRegularizationC = TValue(0), TValue quadraticRegularizationC = TValue(0),
                                TValue purityEpsilon = 0, int quadraticBasisIndex=-1)
        {
            ucreat.push_back([ = ]()
            {
                return Learning::MakeUnaryFactorType<TTraits>(smallestEigenValue, largestEigenValue, quadraticBasisIndex,
                        linearRegularizationC, quadraticRegularizationC);
            });
            uinfos.push_back(Learning::Detail::FactorTypeInfo<TValue>(nFeatureCount, nDepthLevels,
                             nMinDataPointsForSplitConsideration, purityEpsilon));
        }


        // Similar to the above, but for pairwise factor types. The offsets vector specifies which variables (relative to
        // a given pixel) are covered by the factor.
        void AddPairwiseFactorType(const Vector2D<int>& offsets,
                                   int nFeatureCount, int nDepthLevels, int nMinDataPointsForSplitConsideration,
                                   TValue smallestEigenValue = TValue(1e-2), TValue largestEigenValue = TValue(1e2),
                                   TValue linearRegularizationC = TValue(0), TValue quadraticRegularizationC = TValue(0),
                                   TValue purityEpsilon = 0, int quadraticBasisIndex=-1)
        {
            VecRef<Vector2D<int>> offvec;
            offvec.push_back(Vector2D<int>(0, 0));   // the first variable is always 0,0 by convention
            offvec.push_back(offsets);               // offsets of the second variable, relative to 0,0
            pcreat.push_back([ = ]()
            {
                return Learning::MakePairwiseFactorType<TTraits>(offvec,
                        smallestEigenValue, largestEigenValue, quadraticBasisIndex,
                        linearRegularizationC, quadraticRegularizationC);
            });
            pinfos.push_back(Learning::Detail::FactorTypeInfo<TValue>(nFeatureCount, nDepthLevels,
                             nMinDataPointsForSplitConsideration, purityEpsilon));
        }

        // Adds a custom, user-specified linear operator. This method is useful if the usual way
        // of instantiating pairwise factors repetitively at offsets relative to each pixel does
        // not capture the desired semantics.
        // The type of the custom linear operator must have been defined prior to calling this method,
        // as in the following example:
        //
        //   template<typename TFeature, typename TUnaryGroundLabel> class CompleteInteractionOperator;
        //   #define TYPE_COMPLETE_INTERACTION_OPERATOR -1
        //   #define INSTANTIATE_CUSTOM_OPERATOR(typeid) new MultiLabel::CompleteInteractionOperator<TFeature, TUnaryGroundLabel>()
        //
        // The declaration of the class and definition of the macros should be at the very top of the
        // C++ implementation file (.cpp), prior to inclusion of any RTF headers, because the headers
        // will expect the macros to be defined.
        //
        // The actual implementation class of the operator must be derived from
        //
        //   LinearOperator::OperatorBase
        //
        // and implement its interface.
        //
        void AddLinearOperator(int type)
        {
            lcreat.push_back([ = ]()
            {
                return TTraits::LinearOperatorRef::Instantiate(type);
            });
        }

        // Add one more basic RTF model to the prediction cascade; and
        // learn the structure of the regression trees associated with the factor types as well as
        // the model parameters residing at the leaves.
        // The parameters (and the tree structure, if GradientNormCriterion is chosen) will be
        // optimized for maximum pseudo-likelihood.
        template<bool Subsample>
        void LearnOneMore(const TDataSampler& traindb,
                          size_t maxNumOptimItPerRound = 50,
                          size_t maxNumOptimItFinal    = 50,
                          TValue finalBreakEps         = 1e-3,
                          TValue subsampleFactor       = 0.3,
                          size_t maxNumItCG            = 10000,
                          TValue residualTolCG         = 1e-4)
        {
            AddFactorTypes();
            InitializeDataset(traindb, maxNumItCG, residualTolCG);

            Detail::DatasetAdapter<TDataSampler> adapter(traindb, subsampleFactor);
            Learning::LearnTreesAndWeightsJointly<TTraits, Subsample, LBFGS_M>(utypes.back(), uinfos,
                    ptypes.back(), pinfos,
                    adapter, maxNumOptimItPerRound,
                    maxNumOptimItFinal, finalBreakEps);
        }

        // Same as the above, except that the model is optimized for the specified loss function
        // (TLossTag). This requires repeated inference during training and is generally much
        // slower than pseudo-likelihood training.  On the other hand, the predictive performance
        // in terms of the selected loss function is typically much better.
        // See Loss.h for an overview of the loss functions that are currently implemented.
        // A special case of particular interest is
        //
        //   Loss::DiscreteHamming.
        //
        // If this loss function is selected, a discrete conditional random field will be trained
        // (rather than the usual Gaussian CRF), where the discrete energy is approximated via the
        // convex quadratic programming relaxation described in Jancsary et al. (ICML 2013).
        // This is the method of choice for discrete output labels. Rather than conjugate gradient,
        // a spectral projected gradient (SPG) method is then used to solve the inference problem;
        // the parameters relating to CG have a similar meaning for SPG.
        template<typename TLossTag, bool Subsample>
        void LearnOneMoreDiscriminative(TDataSampler& traindb,
                                        size_t maxNumOptimItPerRound = 50,
                                        size_t maxNumOptimItFinal    = 50,
                                        TValue finalBreakEps         = 1e-3,
                                        bool stagedTraining          = false,
                                        size_t maxNumItCG            = 10000,
                                        TValue residualTolCG         = 1e-4,
                                        TValue subsampleFactor       = 0.3)
        {
            discreteInference = Loss::Loss<TTraits, TLossTag>::RequiresDiscreteInference();

            AddFactorTypes();
            InitializeDataset(traindb, maxNumItCG, residualTolCG);

            Detail::DatasetAdapter<TDataSampler> adapter(traindb, subsampleFactor);
            Learning::LearnTreesAndWeightsJointlyDiscriminative<TTraits, TLossTag, Subsample, LBFGS_M>(utypes.back(), uinfos,
                    ptypes.back(), pinfos, ltypes.back(),
                    adapter, maxNumOptimItPerRound,
                    maxNumOptimItFinal, finalBreakEps,
                    stagedTraining, maxNumItCG, residualTolCG);
        }

        // Predict the labels of several new, unseen examples; by default, the examples are passed
        // through all layers of the prediction cascade. This behaviour can be controlled by setting
        // the 'upToLevel' parameter.
        VecCRef<ImageRefC<typename TTraits::UnaryGroundLabel>>
                Regress(TDataSampler& testdb, size_t maxNumItCG = 10000, TValue residualTolCG = 1e-6, int upToLevel=-1) const
        {
            if( upToLevel < 0 )
                upToLevel = static_cast<int>(EnsembleSize()) - 1;

            InitializeDataset(testdb, maxNumItCG, residualTolCG, upToLevel);

            VecRef<ImageRefC<typename TTraits::UnaryGroundLabel>> predictions(testdb.GetImageCount());
            for(int i = 0; i < predictions.size(); ++i)
                predictions[i] = RegressWith(testdb.GetInputImage(i), upToLevel, maxNumItCG, residualTolCG);

            return predictions;
        }

        // Evaluate the performance of the model on the specified unseen examples in terms of the
        // specified loss function, micro-averaged.
        template<typename TLossTag>
        typename TTraits::ValueType
        EvaluateMicroAveraged(const TDataSampler& testdb, size_t maxNumItCG, TValue residualTolCG) const
        {
            return Loss::MicroAveraged<TTraits, TLossTag>(testdb, [&](const ImageRefC<typename TTraits::InputLabel>& img, size_t idx) -> ImageRefC<typename TTraits::UnaryGroundLabel>
            {
                auto pred = Regress(img, maxNumItCG, residualTolCG);
                return pred;
            });
        }

        // Evaluate the performance of the model on the specified unseen examples in terms of the
        // specified loss function, macro-averaged.
        template<typename TLossTag>
        typename TTraits::ValueType
        EvaluateMacroAveraged(const TDataSampler& testdb, size_t maxNumItCG, TValue residualTolCG) const
        {
            return Loss::MacroAveraged<TTraits, TLossTag>(testdb, [&](const ImageRefC<typename TTraits::InputLabel>& img, size_t idx) -> ImageRefC<typename TTraits::UnaryGroundLabel>
            {
                auto pred = Regress(img, maxNumItCG, residualTolCG);
                return pred;
            });
        }

        // Evaluate the performance of the model on the specified unseen examples in terms of the
        // specified loss function, micro-averaged, invoking the user-specified callback after
        // each example has been processed.
        template<typename TLossTag, typename TOp>
        typename TTraits::ValueType
        EvaluateMicroAveraged(const TDataSampler& testdb, size_t maxNumItCG, TValue residualTolCG, const TOp& op) const
        {
            return Loss::MicroAveraged<TTraits, TLossTag>(testdb, [&](const ImageRefC<typename TTraits::InputLabel>& img, size_t idx) -> ImageRefC<typename TTraits::UnaryGroundLabel>
            {
                auto pred = Regress(img, maxNumItCG, residualTolCG);
                op(pred, idx);
                return pred;
            });
        }

        // Evaluate the performance of the model on the specified unseen examples in terms of the
        // specified loss function, macro-averaged, invoking the user-specified callback after
        // each example has been processed.
        template<typename TLossTag, typename TOp>
        typename TTraits::ValueType
        EvaluateMacroAveraged(const TDataSampler& testdb, size_t maxNumItCG, TValue residualTolCG, const TOp& op) const
        {
            return Loss::MacroAveraged<TTraits, TLossTag>(testdb, [&](const ImageRefC<typename TTraits::InputLabel>& img, size_t idx) -> ImageRefC<typename TTraits::UnaryGroundLabel>
            {
                auto pred = Regress(img, maxNumItCG, residualTolCG);
                op(pred, idx);
                return pred;
            });
        }

        // Evaluate the performance of the given predictions versus the ground truth, micro-averaged.
        template<typename TLossTag>
        typename TTraits::ValueType
        EvaluateMicroAveraged(const TDataSampler& testdb, const VecCRef<ImageRefC<typename TTraits::UnaryGroundLabel>>& prediction) const
        {
            return Loss::MicroAveraged<TTraits, TLossTag>(testdb, prediction);
        }

        // Evaluate the performance of the given predictions versus the ground truth, macro-averaged.
        template<typename TLossTag>
        typename TTraits::ValueType
        EvaluateMacroAveraged(const TDataSampler& testdb, const VecCRef<ImageRefC<typename TTraits::UnaryGroundLabel>>& prediction) const
        {
            return Loss::MacroAveraged<TTraits, TLossTag>(testdb, prediction);
        }

        // Evaluate the given prediction versus the actual ground truth.
        template<typename TLossTag>
        typename TTraits::ValueType
        Evaluate(const ImageRefC<typename TTraits::UnaryGroundLabel>& ground,
                 const ImageRefC<typename TTraits::UnaryGroundLabel>& prediction)
        {
            return Loss::PerImage<TTraits, TLossTag>(ground, prediction);
        }

        // Write the model to disk in a portable format.
        void Serialize(const std::string& fname) const
        {
            std::ofstream ofs(fname.c_str());
            WriteModel(ofs);
        }

        // Write the model to the specified output stream, in a portable format.
        // Note that your feature class must implement operator>> and also operator<<
        // for later deserialization.
        void Serialize(std::ostream& out) const
        {
            WriteModel(out);
        }
    };
}

#endif // H_RTF_STACKED_H
