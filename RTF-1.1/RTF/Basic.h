/* This file is part of the "Regression Tree Fields" (RTF) source code distribution,
 * obtained from http://research.microsoft.com/downloads.
 * It is provided to you under the terms of the Microsoft Research License Agreement
 * (MSR-LA). Please see License.txt for details.
 *
 *
 * File: Basic.h
 * Exposes a basic object-oriented interface to training of regression tree fields.
 *
 */

#ifndef H_RTF_BASIC_H
#define H_RTF_BASIC_H

#include <string>
#include <random>

#include "Types.h"
#include "Classify.h"
#include "Monitor.h"
#include "Learning.h"
#include "Serialization.h"
#include "LinearOperator.h"

// A regression tree field is a conditional random field consisting of a number of unary and
// pairwise factors, or, in rare cases, a user-specified custom linear operator that models the
// the interactions between variables.
//
// In most cases, the factors are specified by means of unary and pairwise factor *types*. Factors
// of a common type share the same regression tree, which stores factor potentials at its leaves.
// The path that the input image takes through the regression tree determines the leaf node and
// hence the potentials of a factor.
//
// A factor type also specifies how factors are instantiated: For unary factor types, one factor
// is instantiated per pixel/variables.  Pairwise factors are instantiated repetitively around
// each pixel/variable at user-specified spatial offsets.  An example would be a typical grid
// layout, where pairwise factors are instantiated to the left, right, top, and bottom of each
// pixel.
//
// When a regression tree field model is trained, the tree associated with each factor type is
// 'grown', that is, its nodes are split until a user-specified depth is reached, and the
// parameters residing at the leaves are optimized for a user-specified criterion, the so-called
// loss function.  Various different loss functions are implemented and can be specified.
//
// Crucially, the training algorithm needs access to various 'model traits', which must be specified
// by the user in terms of C++ classes.  Most importantly, the algorithm needs access to the
// training data in terms of a standardized interface which must be implemented by the user. Second,
// the algorithm requires the user to offer a number of features that can be computed from the
// input image: a) Features that are used to split tree nodes; b) Features that are used in the
// 'linear basis' part of factor (e.g. filter responses), and which are weighted by model parameters;
// and c) quadratic basis scalars that can weight the quadratic (interaction) part of a factor and
// are computed from the input image relative to a factor.
//
// These, and other 'model traits', are specified via C++ template parameters when instantiating
// an RTF model:
//
//   namespace Basic
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
// The template parameters have the following meaning:
//
// *TFeatureSampler*
//
//   The responsibility of the feature sampler class is to generate a number of features for
//   splitting of tree nodes, the utility of which will be evaluated by the tree training algorithm.
//   A feature sampler must implement the following interface:
//
//     struct FeatureSampler {
//       typedef Feature TFeature;                           ... typedef that identifies the feature class
//       FeatureSampler();                                   ... default constructor
//       TFeature operator()(int level);                     ... draw a new feature instance at tree level 'level'
//     };
//
//   The feature class itself must adhere to the following interface:
//
//     struct Feature {
//       typedef ImageRefC<InputLabel> PreProcessType;       ... type of a pre-processed input image
//       Feature();                                          ... default constructor
//       static PreProcessType PreProcess(const ImageRefC<Dataset::InputLabel>& input)
//                                                           ... return a pre-processed input image
//       bool operator()(int x, int y,
//                       const PreProcessType& data,
//                       const VecCRef<Vector2D<int>>& offsets) const
//                                                           ... decide whether to branch left or right
//
//       Feature WithThresholdFromSample(int x, int y,
//                                       const PreProcessType& sample,
//                                       const VecCRef<Vector2D<int>>& offsets) const
//                                                           ... can be used to adjust the feature threshold (if any)
//                                                               from the provided sample (otherwise, just return a copy)
//
//       friend std::ostream& operator<<(std::ostream& os, const Feature& feat);
//       friend std::istream& operator>>(std::istream& is, Feature& feat);
//                                                           ... (optional) serialization interface, must be implemented to
//                                                               use the serialization routines in Serialization.h
//
//       static const size_t UnaryBasisSize    = constant1;
//       static const size_t PairwiseBasisSize = constant2;
//       static void ComputeBasis(int x, int y, const PreProcessType& prep,
//                                const VecCRef<Vector2D<int>>& offsets, TValue* basis);
//                                                           ... (optional) linear basis function interface; this must be
//                                                               implemented if 'UseBasis' is set to 'true' in the Traits class.
//                                                               The feature vector computed from prep must be written to the
//                                                               'basis' array, the length of which is either 'constant1' if
//                                                               offsets.size()==1 and 'constant2' otherwise.
//
//       static TValue ComputeQuadraticBasis(const PreProcessType& prep, const Vector2D<int>& i, size_t basisIndex);
//                                                           ... can be used to specify a positive scalar by which the user-specified
//                                                               unary factor is weighted. The 'basisIndex' parameter is a property of
//                                                               the factor type which is specified by the user when setting up that
//                                                               type. It can be used to specify, for instance, from which channel of
//                                                               the pre-processed image the weighting scalar should be computed.
//
//       static TValue ComputeQuadraticBasis(const PreProcessType& prep, const Vector2D<int>& i, const Vector2D<int>& j, size_t basisIndex);
//                                                           ... same as the above, but for pairwise factors.
//     };
//     std::ostream & operator<<(std::ostream& os, const Feature& feat);
//     inline std::istream& operator>>(std::istream& is, Feature& feat);
//                                                           ... (optional) serialization operators for writing to and reading a feature
//                                                               instance from a stream. These operators must be implemented if you want
//                                                               to be able to serialize a model.
//
// *TDataSampler*
//
//   Again, the interface of the data sampler class is by and large based on the decision tree field API. The following
//   methods must be implemented by a conformant class:
//
//   struct DataSampler {
//      typedef Training::LabelVector<double, dim>    UnaryGroundLabel;     ... type of a single-pixel ground label; it is recommended to use the
//                                                                              default implementation Training::LabelVector, but an alternative
//                                                                              implementaiton may be used as long as it exposes the same interface
//                                                                              as Training::LabelVector. The first type parameter of Training::LabelVector
//                                                                              specifies the floating point type to use ... 'double' is strongly
//                                                                              recommended, as this type specifies the precision of all internal computations.
//                                                                              'dim' specifies the variable cardinality of y_i's.
//
//      typedef Training::LabelVector<double, dim*2>  PairwiseGroundLabel;  ... type of a pairwise pixel ground label; the dimension must be twice that of
//                                                                              a unary pixel label; it is recommended to use the same floating point type.
//
//      typedef unsigned char                         InputLabel;           ... the label type of a single input pixel; this type can be almost arbitrary
//                                                                              and depends on the input data.
//
//      size_t GetImageCount() const;                                       ... return the number of images in the dataset
//
//      ImageRefC<UnaryGroundLabel> GetGroundTruthImage(size_t idx) const   ... return the i'th ground truth image
//
//      ImageRefC<InputLabel> GetInputImage(size_t idx) const               ... return the i'th input image
//   };
//
// *TUSplitCritTag* / *TPSplitCritTag*
//
//   Specifies the symbolic tag that chooses the split criterion for unary/pairwise trees
//   (see Criteria.h).
//
//
// *TUPriorTag* / *TPPriorTag*
//
//   Specifies the symbolic tag that chooses the unary/pairwise prior class
//   (see Priors.h).
//
// *UseBasis*
//
//   Boolean flag that specifies whether data-dependent linear basis functions are to be used to form the
//   linear offset vector in the canonical parameterization of the local Gaussian models. If so, the feature
//   class must implement the interface outlined under *TFeatureSampler*.
//
// *UseExplicitThresholdTesting*
//
//   Uses a specialized tree training routine for continuous threshold-style features. This allows to check a
//   large number of potential split thresholds very efficiently. The thresholds are sampled from the actual
//   data points going into a split candidate. If set to true, the feature class must expose the following
//   interface _in addition_ to the regular feature interface:
//
//     class Feature {
//        // ..
//        /* Number of thresholds sampled from the data points */
//        static const size_t NumThresholdTests = 64;
//
//        /* Computes a real-valued feature response; TValue must be the floating point type used for the ground labels */
//        TValue Response(int x, int y, const PreProcessType& data, const VecCRef<Vector2D<int>>& variables) const;
//
//        /* Returns a copy of the feature with its threshold (used by operator()) set to 'threshold' */
//        Feature WithThreshold(double threshold) const;
//        // ...
//     };
//
//   Also, note that the method
//
//      bool operator()(int x, int y, const PreProcessType& data, const VecCRef<Vector2D<int>>& variables) const
//      {
//          return Response(x, y, data, variables) < threshold;
//      }
//
//   of your feature class must be implemented as above to ensure compatibility with the Response() method.
//
// *TMonitor*
//
//   Specifies the monitor class that is used by the implementation of algorithms to display progress
//   information. By default, progress information is written to stderr along with further details
//   such as the current time, CPU and memory usage, etc. An alternative implementation must expose the
//   following interface:
//
//     struct DefaultMonitor {
//       static void Display(const char* fmt, ...);
//       static void Report(const char* fmt, ...);
//       static void ReportVA(const char *fmt, va_list argptr);
//     };
//
// *CachingMode*
//
//   Specifies how the linear system that describes the RTF inference problem should be set up.
//
//    - WEIGHTS_AND_BASIS_AND_MATRIX_PRECOMPUTED
//      The sparse matrix of the linear system is actually instantiated. All terms are pre-computed
//      for maximum efficiency of the conjugate gradient iterations.
//
//    - WEIGHTS_AND_BASIS_PRECOMPUTED
//      All factor potentials (the linear and quadratic parts) are pre-computed when setting up the
//      linear system, but they are not arranged in a compressed sparse matrix format to save memory.
//
//    - ON_THE_FLY
//      All terms, including the factor potentials, are computed on the fly at each iteration of
//      the conjugate gradient algorithm.
//
//    Which option is most efficient depends on various factors, including the memory consumption and
//    the number of conjugate gradient iterations actually needed. Setting up the sparse matrix can
//    require quite some time, so if only few iterations are performed, it may be more efficient to
//    skip this step.
//
// *TLinearOperatorWeights*
//
//  (Experimental). If a regression tree shall be trained and used to specify the weights of a
//  custom, user-implemented linear operator, then the type of the weights must be specified
//  using this template argument. In most cases, a user need not specify this parameter.
//
// Once all 'model traits' have been specified, various methods can be invoked on an RTF instance
// in order to learn the tree structure and optimize parameters, predict on unseen images, write
// to and read from disk, etc. Please see the method definitions below for details.
namespace Basic
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

    // Class that represents a single RTF model.
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
                TPrior,
                TPrior,
                UseBasis,
                UseExplicitThresholdTesting,
                TMonitor,
                CachingMode,
                TLinearOperatorWeights> TTraits;

        typedef typename TTraits::ValueType         TValue;

        bool discreteInference;

        static const int LBFGS_M = 64;

    private:
        typename TTraits::UnaryFactorTypeVector                 utypes;
        typename TTraits::PairwiseFactorTypeVector              ptypes;
        typename TTraits::LinearOperatorVector                  ltypes;

        std::vector<Learning::Detail::FactorTypeInfo<TValue>>   uinfos;
        std::vector<Learning::Detail::FactorTypeInfo<TValue>>   pinfos;


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

            Serialization::ReadModel<TTraits>(in, utypes, ptypes, ltypes);
        }

        void WriteModel(std::ostream& out) const
        {
            out << uinfos.size() << std::endl;
            for( size_t u = 0; u < uinfos.size(); ++u )
                out << uinfos[u];

            out << pinfos.size() << std::endl;
            for( size_t p = 0; p < pinfos.size(); ++p )
                out << pinfos[p];

            Serialization::WriteModel<TTraits>(out, utypes, ptypes, ltypes);
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

        // Adds a unary factor type. The parameters specify the characteristics of the underlying regression
        // tree that is to be trained, as well as regularization of the model parameters.
        typename TTraits::UnaryFactorType&
        AddUnaryFactorType(int nFeatureCount, int nDepthLevels, int nMinDataPointsForSplitConsideration,
                           TValue smallestEigenValue = TValue(1e-2), TValue largestEigenValue = TValue(1e2),
                           TValue linearRegularizationC = TValue(0), TValue quadraticRegularizationC = TValue(0),
                           TValue purityEpsilon = 0, int quadraticBasisIndex=-1)
        {
            utypes.push_back(Learning::MakeUnaryFactorType<TTraits>(smallestEigenValue, largestEigenValue, quadraticBasisIndex,
                             linearRegularizationC, quadraticRegularizationC));
            uinfos.push_back(Learning::Detail::FactorTypeInfo<TValue>(nFeatureCount, nDepthLevels,
                             nMinDataPointsForSplitConsideration, purityEpsilon));
            return utypes.back();
        }

        // Similar to the above, but for pairwise factor types. The offsets vector specifies which variables (relative to
        // a given pixel) are covered by the factor.
        typename TTraits::PairwiseFactorType&
        AddPairwiseFactorType(const Vector2D<int>& offsets,
                              int nFeatureCount, int nDepthLevels, int nMinDataPointsForSplitConsideration,
                              TValue smallestEigenValue = TValue(1e-2), TValue largestEigenValue = TValue(1e2),
                              TValue linearRegularizationC = TValue(0), TValue quadraticRegularizationC = TValue(0),
                              TValue purityEpsilon = 0, int quadraticBasisIndex=-1)
        {
            VecRef<Vector2D<int>> offvec;
            offvec.push_back(Vector2D<int>(0, 0));   // the first variable is always 0,0 by convention
            offvec.push_back(offsets);               // offsets of the second variable, relative to 0,0
            ptypes.push_back(Learning::MakePairwiseFactorType<TTraits>(offvec, smallestEigenValue, largestEigenValue, quadraticBasisIndex,
                             linearRegularizationC, quadraticRegularizationC));
            pinfos.push_back(Learning::Detail::FactorTypeInfo<TValue>(nFeatureCount, nDepthLevels,
                             nMinDataPointsForSplitConsideration, purityEpsilon));
            return ptypes.back();
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
        typename TTraits::LinearOperatorRef
        AddLinearOperator(int type)
        {
            ltypes.push_back(TTraits::LinearOperatorRef::Instantiate(type));
            return ltypes.back();
        }

        // Learn the structure of the regression trees associated with the factor types as well as
        // the model parameters residing at the leaves.
        // The parameters (and the tree structure, if GradientNormCriterion is chosen) will be
        // optimized for maximum pseudo-likelihood.
        template<bool Subsample>
        void Learn(const TDataSampler& traindb,
                   size_t maxNumOptimItPerRound = 50,
                   size_t maxNumOptimItFinal    = 50,
                   TValue finalBreakEps         = 1e-3,
                   TValue subsampleFactor       = 0.3)
        {
            Detail::DatasetAdapter<TDataSampler> adapter(traindb, subsampleFactor);
            Learning::LearnTreesAndWeightsJointly<TTraits, Subsample, LBFGS_M>(utypes, uinfos,
                    ptypes, pinfos,
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
        void LearnDiscriminative(const TDataSampler& traindb,
                                 size_t maxNumOptimItPerRound = 50,
                                 size_t maxNumOptimItFinal    = 50,
                                 TValue finalBreakEps         = 1e-3,
                                 bool stagedTraining          = false,
                                 size_t maxNumItCG            = 10000,
                                 TValue residualTolCG         = 1e-4,
                                 TValue subsampleFactor       = 0.3)
        {
            discreteInference = Loss::Loss<TTraits, TLossTag>::RequiresDiscreteInference();
            Detail::DatasetAdapter<TDataSampler> adapter(traindb, subsampleFactor);
            Learning::LearnTreesAndWeightsJointlyDiscriminative<TTraits, TLossTag, Subsample, LBFGS_M>(utypes, uinfos,
                    ptypes, pinfos, ltypes,
                    adapter, maxNumOptimItPerRound,
                    maxNumOptimItFinal, finalBreakEps,
                    stagedTraining, maxNumItCG, residualTolCG);
        }

        // Predict the labels of a new, unseen example
        ImageRefC<typename TTraits::UnaryGroundLabel>
        Regress(const ImageRefC<typename TTraits::InputLabel>& input, size_t maxNumItCG = 10000, TValue residualTolCG = 1e-6) const
        {
            return Classify::Predict<TTraits>(utypes, ptypes, ltypes, input, residualTolCG, maxNumItCG, discreteInference);
        }

        // Predict the labels of several new, unseen examples
        VecCRef<ImageRefC<typename TTraits::UnaryGroundLabel>>
                Regress(const TDataSampler& testdb, size_t maxNumItCG = 10000, TValue residualTolCG = 1e-6) const
        {
            VecRef<ImageRefC<typename TTraits::UnaryGroundLabel>> predictions(testdb.GetImageCount());

            for(int i = 0; i < predictions.size(); ++i)
                predictions[i] = Regress(testdb.GetInputImage(i), maxNumItCG, residualTolCG);

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

#endif // H_RTF_BASIC_H
