/* This file is part of the "Regression Tree Fields" (RTF) source code distribution,
 * obtained from http://research.microsoft.com/downloads.
 * It is provided to you under the terms of the Microsoft Research License Agreement
 * (MSR-LA). Please see License.txt for details.
 *
 *
 * File: TOFDenoising.cpp
 * Implements a natural image denoising example application.
 *
 */

#ifdef USE_MPI
#pragma message("MPI support enabled")
#endif

// Use the filterbank by Qi Gao and Stefan Roth, DAGM 2012.
#define USE_QI_DAGM12_FILTERBANK 1
//#define USE_RFS_FILTERBANK 1

// Do not instantiate the sparse linear system, but perform all computation on the fly
#define CACHING_TYPE ON_THE_FLY

#include "RTF/Types.h"
#include "RTF/Basic.h"
#include "RTF/Stacked.h"
#include "RTF/Monitor.h"
#include "RTF/Utility.h"
#include "RTF/Criteria.h"
#include "RTF/Learning.h"
#include "RTF/Training.h"
#include "RTF/Classify.h"
#include "RTF/Serialization.h"

#include "Dataset.h"
#include "Features.h"
 
// lambda weight for MyLorentzian loss function
#define LORENTZIAN_LAMBDA 50.0
#include "Loss.h"

#include <boost/filesystem.hpp>

// Console output channel
typedef Monitor::DefaultMonitor MyMonitor;

// Important training parameters
int NumFeatures              = 512;
int PairwiseDepth            = 7;
int UnaryDepth               = 7;
int PairwiseMinPoints        = 64;
int UnaryMinPoints           = 16;
double UnarySmallestEigen    = 1e-4;
double UnaryLargestEigen     = 1e4;
double PairwiseSmallestEigen = 1e-4;
double PairwiseLargestEigen  = 1e4;
bool JointTraining           = true;
int StackDepth               = -1;

const auto interimit         = 100;
const auto finalit           = 0; // no final optimisation: might want to use intermediate trees
const auto gradeps           = 1e-6;
const auto staged            = false;
const auto cgit              = 1000;
const auto cgeps             = 1e-6;
const auto jointsubsample    = 0.50; // Subsampling: use only 50% of the pixels
// TOF-context: input channels (4 raw frames, 2 differences, 1 quotient, 1 mean, 1 arctan(quotient))
// must be const for project to build because of template parameters to ReadDLM call in Dataset.h
const int nChann                  = 9; 

// Various options for spatial layout of the pairwise factors
const Vector2D<int>
PairwiseOffsets3x3[]         = { Vector2D<int>(1,0), Vector2D<int>(0,1),
                                 Vector2D<int>(1,1), Vector2D<int>(-1,1)
                               };

const Vector2D<int>
PairwiseOffsets5x5[]         = { Vector2D<int>(-2,-2), Vector2D<int>(-1,-2), Vector2D<int>(0, -2), Vector2D<int>(1, -2), Vector2D<int>(2, -2),
                                 Vector2D<int>(-2,-1), Vector2D<int>(-1,-1), Vector2D<int>(0, -1), Vector2D<int>(1, -1), Vector2D<int>(2, -1),
                                 Vector2D<int>(-2, 0), Vector2D<int>(-1, 0)
                               };

const Vector2D<int>
PairwiseOffsets7x7[]         = { Vector2D<int>(-3,-3), Vector2D<int>(-2,-3), Vector2D<int>(-1, -3), Vector2D<int>(0, -3), Vector2D<int>(1, -3), Vector2D<int>(2, -3), Vector2D<int>(3, -3),
                                 Vector2D<int>(-3,-2), Vector2D<int>(-2,-2), Vector2D<int>(-1, -2), Vector2D<int>(0, -2), Vector2D<int>(1, -2), Vector2D<int>(2, -2), Vector2D<int>(3, -2),
                                 Vector2D<int>(-3,-1), Vector2D<int>(-2,-1), Vector2D<int>(-1, -1), Vector2D<int>(0, -1), Vector2D<int>(1, -1), Vector2D<int>(2, -1), Vector2D<int>(3, -1),
                                 Vector2D<int>(-3, 0), Vector2D<int>(-2, 0), Vector2D<int>(-1,  0)
                               };

const Vector2D<int>*
PairwiseOffsets              = PairwiseOffsets5x5;
size_t NumPairwise           = sizeof(PairwiseOffsets5x5)/sizeof(Vector2D<int>);

// Input and output paths
std::string DataPath         = "";
std::string OutputPath       = "";

std::string ModelFile        = "";
std::string InputFile        = "";
std::string OutputFile       = "";

// Returns the name of a loss
template<typename TLossTag> std::string LossName()
{
    return Loss::Loss<Traits<Denoising::FeatureSampler, Denoising::Dataset>, TLossTag>::Name();
}

// Clear existing performance statistics files before appending
void ClearStatistics(Denoising::Dataset& testds, const std::string& systemName, const std::string& dsName)
{
    boost::filesystem::path outp = OutputPath;

#ifdef USE_MPI
    if( MPI::Communicator().rank() == 0 )
    {
#endif
        std::ofstream overall((outp / ("Overall " + dsName + " " + systemName + ".txt")).string());
        overall.close();
#ifdef USE_MPI
    }
#endif

    for( int i = 0; i < testds.GetImageCount(); ++i )
    {
        std::ofstream individual((outp / (testds.GetImageName(i) + " " + systemName + ".txt")).string());
        individual.close();
    }
}

// Collects performance statistics given a dataset and the predictions made thereon
template<typename TLossTag, typename TModel>
void
Evaluate(Denoising::Dataset& testds, const VecCRef<ImageRefC<Denoising::Dataset::UnaryGroundLabel>>& prediction,
         TModel& model, const std::string& systemName, const std::string& dsName)
{
    boost::filesystem::path outp = OutputPath;

    // Compute overall performance statistics
    const auto overallLoss = model.template EvaluateMicroAveraged<TLossTag>(testds, prediction);
#ifdef USE_MPI
    if( MPI::Communicator().rank() == 0 )
    {
#endif
        std::ofstream overall((outp / ("Overall " + dsName + " " + systemName + ".txt")).string(), std::ios_base::app);
        overall << LossName<TLossTag>() << ": " << overallLoss << std::endl;
        overall.close();
#ifdef USE_MPI
    }
#endif

    // Compute per-image performance statistics
    for( int i = 0; i < testds.GetImageCount(); ++i )
    {
        auto loss = model.Evaluate<TLossTag>(testds.GetGroundTruthImage(i), prediction[i]);
        std::ofstream individual((outp / (testds.GetImageName(i) + " " + systemName + ".txt")).string(), std::ios_base::app);
        individual << LossName<TLossTag>() << ": " << loss << std::endl;
//        testds.SaveGroundTruthImage(prediction[i], (outp / (testds.GetImageName(i) + " " + systemName + ".png")).string());
        testds.SaveGroundTruthImage(prediction[i], (outp / (testds.GetImageName(i) + " " + systemName + ".dlm")).string());
    }
}

// Reports all relevant options and collects performance statistics on test and training data
template<typename TModel>
void
EvaluateAll(TModel& model, const std::string& systemName, long long trainTicks)
{
    boost::filesystem::path outp = OutputPath;


#ifdef USE_MPI
    if( MPI::Communicator().rank() == 0 )
    {
#endif
        std::ofstream options((outp / ("Options " + systemName + ".txt")).string());
        options << "NumFeatures: " << NumFeatures << std::endl;
        options << "UnaryDepth : " << UnaryDepth << std::endl;
        options << "PairwiseDepth: " << PairwiseDepth << std::endl;
        options << "UnaryMinPoints: " << UnaryMinPoints << std::endl;
        options << "PairwiseMinPoints: " << PairwiseMinPoints << std::endl;
        options << "InterimIterations: " << interimit << std::endl; 
        options << "FinalIterations: " << finalit << std::endl; 
        options << "CG Iterations: " << cgit << std::endl; 
        options << "UnarySmallestEigen: " << UnarySmallestEigen << std::endl;
        options << "UnaryLargestEigen: " << UnaryLargestEigen << std::endl;
        options << "PairwiseSmallestEigen: " << PairwiseSmallestEigen << std::endl;
        options << "PairwiseLargestEigen: " << PairwiseLargestEigen << std::endl;
        options << "JointTraining: " << JointTraining << std::endl;
        if( PairwiseOffsets == PairwiseOffsets3x3 )
            options << "Connectivity: " << "3x3" << std::endl;
        else if( PairwiseOffsets == PairwiseOffsets5x5 )
            options << "Connectivity: " << "5x5" << std::endl;
        else if( PairwiseOffsets == PairwiseOffsets7x7)
            options << "Connectivity: " << "7x7" << std::endl; 
        else
            options << "Connectivity: " << "1x1" << std::endl;
        options << "TOF-channels: " << nChann << std::endl; 
        options << "***Model only uses depth***" << std::endl; 
        options << "***Stacked configuration possible***" << std::endl;
        options << "Filterbank: QI on depth" << std::endl; 
        options << "Loss-lambda: " << LORENTZIAN_LAMBDA << std::endl; 
        options << "DataPath: " << DataPath << std::endl;
        options << "OutputPath: " << OutputPath << std::endl;
        options.close();
#ifdef USE_MPI
    }
#endif

    Denoising::Dataset testds(DataPath, "test");
    auto startTick = GetTickCountPortable();
    auto testPrediction = model.Regress(testds, cgit, cgeps);
    const auto timePrediction = (GetTickCountPortable() - startTick);

#ifdef USE_MPI
    if( MPI::Communicator().rank() == 0 )
    {
#endif
        std::ofstream timing((outp / ("Timing " + systemName + ".txt")).string());
        timing << "Training: " << trainTicks << " ms" << std::endl;
        timing << "Predicting for " << testds.GetImageCount() << " test images: " << timePrediction << " ms" << std::endl;
        timing.close();
#ifdef USE_MPI
    }
#endif

    ClearStatistics(testds, systemName, "Test");
    //Evaluate<Loss::PSNR>(testds, testPrediction, model, systemName, "Test");
    //Evaluate<Loss::SSIM>(testds, testPrediction, model, systemName, "Test");
    Evaluate<Loss::MyLorentzian>(testds, testPrediction, model, systemName, "Test");
    testds.ReleaseCache();

    Denoising::Dataset trainds(DataPath, "train");
    auto trainPrediction = model.Regress(trainds, cgit, cgeps);
    ClearStatistics(trainds, systemName, "Train");
    //Evaluate<Loss::PSNR>(trainds, trainPrediction, model, systemName, "Train");
    //Evaluate<Loss::SSIM>(trainds, trainPrediction, model, systemName, "Train");
    Evaluate<Loss::MyLorentzian>(trainds, trainPrediction, model, systemName, "Train");
    trainds.ReleaseCache();

#ifdef USE_MPI
    if( MPI::Communicator().rank() == 0 )
    {
#endif
        MyMonitor::Report("Serializing model ...\n");
        model.Serialize((outp / ("Model " + systemName + ".txt")).string());
#ifdef USE_MPI
    }
#endif
}

// Trains an RTF in a loss-specific manner
template<typename TLossTag>
void
TrainAndEvaluate_DiscriminativeRTF(const std::string& name)
{
    Basic::RTF<Denoising::FeatureSampler,
          Denoising::Dataset,
          GradientNormCriterion,
          true,
          true,
          NullPrior,
          Monitor::DefaultMonitor,
          CACHING_TYPE> rtf;

    rtf.AddUnaryFactorType(NumFeatures, UnaryDepth, UnaryMinPoints, UnarySmallestEigen, UnaryLargestEigen);

    for( size_t i = 0; i < NumPairwise; ++i )
        rtf.AddPairwiseFactorType(PairwiseOffsets[i], NumFeatures, PairwiseDepth, PairwiseMinPoints,
                                  PairwiseSmallestEigen, PairwiseLargestEigen);

    auto startTicks = GetTickCountPortable();

    Denoising::Dataset trainds(DataPath, "train");
    rtf.template LearnDiscriminative<TLossTag, true>(trainds, interimit, finalit, gradeps, staged, cgit, cgeps, jointsubsample);
    trainds.ReleaseCache();
    EvaluateAll(rtf, name, GetTickCountPortable()-startTicks);
}


// Trains a cascade of RTFs in a loss-specific manner
template<typename TLossTag>
void
TrainAndEvaluate_StackedRTF(const std::string& name)
{
    Stacked::RTF<Denoising::FeatureSampler,
            Denoising::Dataset,
            GradientNormCriterion,
            true,
            true,
            NullPrior,
            Monitor::DefaultMonitor,
            CACHING_TYPE> rtf;

    rtf.AddUnaryFactorType(NumFeatures, UnaryDepth, UnaryMinPoints, UnarySmallestEigen, UnaryLargestEigen);

    for( size_t i = 0; i < NumPairwise; ++i )
        rtf.AddPairwiseFactorType(PairwiseOffsets[i], NumFeatures, PairwiseDepth, PairwiseMinPoints,
                                  PairwiseSmallestEigen, PairwiseLargestEigen);

    auto startTicks = GetTickCountPortable();
    MyMonitor::Report("Total Stacksize: %u\n\n", StackDepth); 

    for( size_t n = 0; n < StackDepth; ++n )
    {
        std::ostringstream os;
        os << n;
        Denoising::Dataset traindb(DataPath, "train" + os.str());
        MyMonitor::Report("Using train%s at %s\n", os.str().c_str(), DataPath.c_str());

        rtf.template LearnOneMoreDiscriminative<TLossTag, true>(traindb, interimit, finalit, gradeps, staged, cgit, cgeps, jointsubsample);
        traindb.ReleaseCache();
        MyMonitor::Report("Learned model!\n Evaluating now...\n\n"); // vl: DEBUG

        EvaluateAll(rtf, name + "-stage" + os.str(), GetTickCountPortable()-startTicks);

        MyMonitor::Report("Successfully trained and evaluated model %u, now learning next model to add to the stack.\n", n);
    }
}

// Write ground truth and input images to output directory (this is useful for browsing the results, but not really needed)
void SaveGroundTruthAndInput(Denoising::Dataset& ds)
{
    boost::filesystem::path outp = OutputPath;

    for( int i = 0; i < ds.GetImageCount(); ++i )
    {
        //ds.SaveGroundTruthImage(ds.GetGroundTruthImage(i), (outp / (ds.GetImageName(i) + " Ground.png")).string());
        ds.SaveGroundTruthImage(ds.GetGroundTruthImage(i), (outp / (ds.GetImageName(i) + " Ground.dlm")).string());
        //ds.SaveInputImage(ds.GetInputImage(i), (outp / (ds.GetImageName(i) + " Input.png")).string());
        ds.SaveInputImage(ds.GetInputImage(i), (outp / (ds.GetImageName(i) + " Input.dlm")).string());
    }
}

// Run the selected experiment
template<typename TOp>
void Run(TOp TrainAndEvaluate)
{
    TrainAndEvaluate();

    // Save ground truth and input along with predictions
    Denoising::Dataset trainds(DataPath, "train");
    SaveGroundTruthAndInput(trainds);
    trainds.ReleaseCache();

    Denoising::Dataset testds(DataPath, "test");
    SaveGroundTruthAndInput(testds);
    testds.ReleaseCache();
}

// Create a prediction for the single given input file (stacked model)
void PredictStacked()
{
    Stacked::RTF<Denoising::FeatureSampler,
            Denoising::Dataset,
            GradientNormCriterion,
            true,
            true,
            NullPrior,
            Monitor::DefaultMonitor,
            ON_THE_FLY> rtf(ModelFile);

    //Denoising::Dataset testds(InputFile, Utility::ReadDLM<float, nChann>(InputFile));
    Denoising::Dataset testds(DataPath, InputFile);

    auto prediction = rtf.Regress(testds, 1000, 1e-6, StackDepth-1);
    auto systemName = "Prediction";

    //testds.SaveGroundTruthImageDLM(prediction[0], OutputFile);
    ClearStatistics(testds, systemName, InputFile);
    Evaluate<Loss::MyLorentzian>(testds, prediction, rtf, systemName, InputFile);
    SaveGroundTruthAndInput(testds);
    testds.ReleaseCache();
}

// Create a prediction for the single given input file (basic model)
void PredictDiscriminative()
{
    Basic::RTF<Denoising::FeatureSampler,
          Denoising::Dataset,
          GradientNormCriterion,
          true,
          true,
          NullPrior,
          Monitor::DefaultMonitor,
          ON_THE_FLY> rtf(ModelFile);


    Denoising::Dataset testds(DataPath, InputFile);
    auto prediction = rtf.Regress(testds, 1000, 1e-6);

    auto systemName = "Prediction";

    ClearStatistics(testds, systemName, InputFile);
    Evaluate<Loss::MyLorentzian>(testds, prediction, rtf, systemName, InputFile);
    SaveGroundTruthAndInput(testds);
    testds.ReleaseCache();

    // boost::filesystem::path outpath = OutputPath;

    // testds.SaveGroundTruthImageDLM(prediction[0], (outpath / (OutputFile + ".dlm")).string());
	//testds.SaveGroundTruthImageDLM(prediction[0], OutputFile);
}

// Command line parsing
const static int DISCRIMINATIVE = 0;
const static int STACKED        = 1;

void Usage(const std::string& name, int type)
{
    if( type == DISCRIMINATIVE )
        std::cerr << "Usage: TOFDenoising " << name << " <inputdir> <outputdir> <connectivity> <treedepth>" << std::endl;
    else if ( type == STACKED )
        std::cerr << "Usage: TOFDenoising " << name << " <inputdir> <outputdir> <connectivity> <treedepth> <stackdepth>" << std::endl;
    else
        assert(0); // invalid

    ::exit(1);
}

void UsagePredict(const std::string& name, int type)
{
    if( type == DISCRIMINATIVE )
        std::cerr << "Usage: TOFDenoising " << name << " <modelfile> <inputdir> <inputfile> <outputdir>" << std::endl;
    else if ( type == STACKED )
        std::cerr << "Usage: TOFDenoising " << name << " <modelfile> <inputdir> <inputfile> <outputdir> [<stackdepth>]" << std::endl;
    else
        assert(0); // invalid

    ::exit(1);
}

void SetConnectivity(const std::string& conn, const std::string& name, int type)
{
    if( conn == "1x1" )
    {
        PairwiseOffsets = NULL;
        NumPairwise     = 0;
    }
    else if( conn == "3x3" )
    {
        PairwiseOffsets = PairwiseOffsets3x3;
        NumPairwise     = sizeof(PairwiseOffsets3x3)/sizeof(Vector2D<int>);
    }
    else if ( conn == "5x5" )
    {
        PairwiseOffsets = PairwiseOffsets5x5;
        NumPairwise     = sizeof(PairwiseOffsets5x5)/sizeof(Vector2D<int>);
    }
    else if ( conn == "7x7" )
    {
        PairwiseOffsets = PairwiseOffsets7x7;
        NumPairwise     = sizeof(PairwiseOffsets7x7)/sizeof(Vector2D<int>);
    }
    else
    {
        MyMonitor::Report("Failed to parse <connectivity> argument (must be one of: 1x1, 3x3, 5x5, 7x7).\n");
        Usage(name, type);
    }
}

void SetTreeDepth(const std::string& depth, const std::string& name, int type)
{
    int max_depth;
    int success = sscanf(depth.c_str(), "%d", &max_depth);
    if (success != 1)
    {
        MyMonitor::Report("Failed to parse <treedepth> argument.\n");
        Usage(name, type);
    }
    else
    {
        UnaryDepth = max_depth;
        PairwiseDepth = max_depth;
    }
}

void SetStackDepth(const std::string& depth, const std::string& name, int type)
{
    int stack_depth;
    int success = sscanf(depth.c_str(), "%d", &stack_depth);
    if (success != 1)
    {
        MyMonitor::Report("Failed to parse <stackdepth> argument.\n");
        Usage(name, type);
    }
    else
    {
        if( stack_depth < 1 )
        {
            MyMonitor::Report("Stack depth must be at least 1\n");
            Usage(name, type);
        }
        MyMonitor::Report("Stackdepth %u confirmed\n\n", stack_depth); // ...
        StackDepth = stack_depth;
    }
}

bool DirectoryExists(const std::string& path)
{
    return boost::filesystem::exists(path);
}

void SetDataPath(const std::string& rawpath, const std::string& name, int type)
{
    DataPath = rawpath;
    if( ! DirectoryExists(DataPath) )
    {
        throw Denoising::IOException("Data path " + DataPath + " does not exist!");
    }
}

void SetOutputPath(const std::string& rawpath, const std::string& name, int type)
{
    OutputPath = rawpath;

    if( ! DirectoryExists(OutputPath) )
    {
        throw Denoising::IOException("Output path " + OutputPath + " does not exist!");
    }
}

void SetModelFile(const std::string& rawpath, const std::string& name, int type)
{
    ModelFile = rawpath;

    if( ! boost::filesystem::exists(rawpath) )
    {
        throw Denoising::IOException("Model file " + ModelFile + " does not exist!");
    }
}

void SetInputFile(const std::string& rawpath, const std::string& name, int type)
{
    InputFile = rawpath;

    //if( ! boost::filesystem::exists(rawpath) )
    //{
    //    throw Denoising::IOException("Input file " + InputFile + " does not exist!");
    //}
}

void SetOutputFile(const std::string& rawpath, const std::string& name, int type)
{
    OutputFile = rawpath;

    std::ofstream test(rawpath);
    if( ! test )
    {
        throw Denoising::IOException("Cannot write to output file " + InputFile + "!");
    }
}

std::string ParseArgumentsDiscriminative(int argc, char* argv[], const std::string& name)
{
    int arg = 2;

    if( argc <= arg )
    {
        MyMonitor::Report("Missing <inputdir> argument\n");
        Usage(name, DISCRIMINATIVE);
    }
    std::string input = (argv[arg++]);
    SetDataPath(input, name, DISCRIMINATIVE);

    if( argc <= arg )
    {
        MyMonitor::Report("Missing <outputdir> argument\n");
        Usage(name, DISCRIMINATIVE);
    }
    std::string output = (argv[arg++]);
    SetOutputPath(output, name, DISCRIMINATIVE);

    if( argc <= arg )
    {
        MyMonitor::Report("Missing <connectivity> argument\n");
        Usage(name, DISCRIMINATIVE);
    }
    std::string conn = (argv[arg++]);
    SetConnectivity(conn, name, DISCRIMINATIVE);

    if( argc <= arg )
    {
        MyMonitor::Report("Missing <treedepth> argument\n");
        Usage(name, DISCRIMINATIVE);
    }
    std::string depth = (argv[arg++]);
    SetTreeDepth(depth, name, DISCRIMINATIVE);

    return name + "_" + conn + "_" + depth;
}

std::string ParseArgumentsPredictDiscriminative(int argc, char* argv[], const std::string& name)
{
    int arg = 2;

    if( argc <= arg )
    {
        MyMonitor::Report("Missing <modelfile> argument\n");
        UsagePredict(name, DISCRIMINATIVE);
    }
    std::string model = (argv[arg++]);
    SetModelFile(model, name, DISCRIMINATIVE);

    if( argc <= arg )
    {
        MyMonitor::Report("Missing <inputdir> argument\n");
        UsagePredict(name, DISCRIMINATIVE);
    }

    if( argc <= arg )
    {
        MyMonitor::Report("Missing <inputfile> argument\n"); // Meaning, e.g. "test" to read test.txt
        UsagePredict(name, DISCRIMINATIVE);
    }
    std::string inputf = (argv[arg++]);
    SetInputFile(inputf, name, DISCRIMINATIVE);

    if( argc <= arg )
    {
        MyMonitor::Report("Missing <outputdir> argument\n");
        UsagePredict(name, DISCRIMINATIVE);
    }
    std::string output = (argv[arg++]);
    SetOutputPath(output, name, DISCRIMINATIVE);

    // if( argc <= arg )
    // {
    //     MyMonitor::Report("Missing <outputfile> argument\n");
    //     UsagePredict(name, DISCRIMINATIVE);
    // }
    // std::string output = (argv[arg++]);
    // SetOutputFile(output, name, DISCRIMINATIVE);

    return name;
}

std::string ParseArgumentsPredictStacked(int argc, char* argv[], const std::string& name)
{
    int arg = 2;

    if( argc <= arg )
    {
        MyMonitor::Report("Missing <modelfile> argument\n");
        UsagePredict(name, STACKED);
    }
    std::string model = (argv[arg++]);
    SetModelFile(model, name, STACKED);

    if( argc <= arg )
    {
        MyMonitor::Report("Missing <inputdir> argument\n");
        UsagePredict(name, STACKED);
    }
    std::string input = (argv[arg++]);
    SetDataPath(input, name, STACKED);

    if( argc <= arg )
    {
        MyMonitor::Report("Missing <inputfile> argument\n");
        UsagePredict(name, STACKED);
    }
    std::string input = (argv[arg++]);
    SetInputFile(input, name, STACKED);

    if( argc <= arg )
    {
        MyMonitor::Report("Missing <outputdir> argument\n");
        UsagePredict(name, STACKED);
    }
    std::string output = (argv[arg++]);
    SetOutputPath(output, name, STACKED);

    // if( argc <= arg )
    // {
    //     MyMonitor::Report("Missing <outputfile> argument\n");
    //     UsagePredict(name, STACKED);
    // }
    // std::string output = (argv[arg++]);
    // SetOutputFile(output, name, STACKED);

    if( argc <= arg )
    {
        return name;
    }
    else
    {
        std::string depth = (argv[arg++]);
        SetStackDepth(depth, name, STACKED);
    }

    return name;
}

std::string ParseArgumentsStacked(int argc, char* argv[], const std::string& name)
{
    int arg = 2;

    if( argc <= arg )
    {
        MyMonitor::Report("Missing <inputdir> argument\n");
        Usage(name, DISCRIMINATIVE);
    }
    std::string input = (argv[arg++]);
    SetDataPath(input, name, DISCRIMINATIVE);

    if( argc <= arg )
    {
        MyMonitor::Report("Missing <outputdir> argument\n");
        Usage(name, DISCRIMINATIVE);
    }
    std::string output = (argv[arg++]);
    SetOutputPath(output, name, DISCRIMINATIVE);

    if( argc <= arg )
    {
        MyMonitor::Report("Missing <connectivity> argument\n");
        Usage(name, DISCRIMINATIVE);
    }
    std::string conn = (argv[arg++]);
    SetConnectivity(conn, name, DISCRIMINATIVE);

    if( argc <= arg )
    {
        MyMonitor::Report("Missing <treedepth> argument\n");
        Usage(name, DISCRIMINATIVE);
    }
    std::string tree_depth = (argv[arg++]);
    SetTreeDepth(tree_depth, name, DISCRIMINATIVE);

    if( argc <= arg )
    {
        MyMonitor::Report("Missing <stackdepth> argument\n");
        Usage(name, DISCRIMINATIVE);
    }
    std::string stack_depth = (argv[arg++]);
    SetStackDepth(stack_depth, name, DISCRIMINATIVE);

    return name + "_" + conn + "_" + tree_depth + "_" + stack_depth;
}

// Reads pairs of filenames from stdin and calculates the specified loss
// A pair is the filename of the ground truth followed by a tab followed by the filename of the prediction
template<typename TLossTag>
void Compare()
{
    typedef ::Loss::Loss<Traits<Denoising::FeatureSampler, Denoising::Dataset>, TLossTag> TLoss;

    std::string line;
    int line_num = 0;
    double micro = 0.0, macro = 0.0, norm_total = 0.0;
    while( std::getline(std::cin, line) )
    {
        line_num += 1;

        auto pos = line.find('\t');
        if( pos == std::string::npos )
        {
            MyMonitor::Report("Malformed input line number %d - must be: <ground truth file><tab><prediction file>\n", line_num);
            ::exit(1);
        }
        double normC = 0.0, line_loss = 0.0;
        try
        {
            auto ground     = Denoising::Dataset::LoadGroundTruthImage(line.substr(0,pos));
            auto prediction = Denoising::Dataset::LoadGroundTruthImage(line.substr(pos+1, line.length()-(pos+1)));
            normC           = TLoss::NormalizationConstant(ground);
            line_loss       = TLoss::Objective(ground, prediction);
        }
        catch( ... )
        {
            MyMonitor::Report("Failed to read file pair at line %d.\n", line_num);
            ::exit(1);
        }

        fprintf(stdout, "%.8f\n", (line_loss/normC));

        norm_total += normC;
        micro      += line_loss;
        macro      += line_loss / normC;
    }
    micro /= norm_total;
    macro /= line_num;

    fprintf(stderr, "%.8f\n", macro);
}

int main(int argc, char* argv[])
{
#ifdef USE_MPI
    // Initialize MPI
    MPI::Environment(argc, argv);
    auto& comm = MPI::Communicator();

    MyMonitor::Report("This is MPI process no. %d to ground control.\n", comm.rank());
#endif

    try
    {
        const char * systemStrings =    "Compare, "
                                        "PredictTrained, "
                                        "PredictStacked, "
                                        "RTFTrainedForMyLorentzian, "
                                        "RTFStackedForMyLorentzian"
                                        // "RTFTrainedForSSIM, "
//                                        "RTFTrainedForIWSSIM, "
//                                        "RTFTrainedForMSE, "
//                                        "RTFTrainedForPSNR"//, "
//                                        "RTFTrainedForMAD, "
//                                        "RTFStackedForSSIM, "
//                                        "RTFStackedForIWSSIM, "
//                                        "RTFStackedForMSE, "
//                                        "RTFStackedForPSNR, "
//                                        "RTFStackedForMAD"
                                        ;
        if( argc < 2 )
        {
            MyMonitor::Report("Missing <system> argument (must be one of: %s)\n", systemStrings);
            ::exit(1);
        }
        std::string sys = (argv[1]);

        // Plain discriminative configurations
        // if ( sys == "RTFTrainedForSSIM" )
        // {
        //     auto name = ParseArgumentsDiscriminative(argc, argv, sys);
        //     Run([&]()
        //     {
        //         TrainAndEvaluate_DiscriminativeRTF<Loss::SSIM>(name);
        //     });
        // }
        if ( sys == "RTFTrainedForMyLorentzian" )
        {
            auto name = ParseArgumentsDiscriminative(argc, argv, sys);
            Run([&]()
            {
                TrainAndEvaluate_DiscriminativeRTF<Loss::MyLorentzian>(name);
            });
        }
        // Stacked configuration
        if ( sys == "RTFStackedForMyLorentzian" )
        {
            auto name = ParseArgumentsStacked(argc, argv, sys);
            Run([&]()
            {
                TrainAndEvaluate_StackedRTF<Loss::MyLorentzian>(name);
            });
        }
        // else if ( sys == "RTFTrainedForMSE" )
        // {
        //     auto name = ParseArgumentsDiscriminative(argc, argv, sys);
        //     Run([&]()
        //     {
        //         TrainAndEvaluate_DiscriminativeRTF<Loss::MSE>(name);
        //     });
        // }
        // else if ( sys == "RTFTrainedForPSNR" )
        // {
        //     auto name = ParseArgumentsDiscriminative(argc, argv, sys);
        //     Run([&]()
        //     {
        //         TrainAndEvaluate_DiscriminativeRTF<Loss::PSNR>(name);
        //     });
        // }
        // else if ( sys == "RTFTrainedForMAD" )
        // {
        //     auto name = ParseArgumentsDiscriminative(argc, argv, sys);
        //     Run([&]()
        //     {
        //         TrainAndEvaluate_DiscriminativeRTF<Loss::MAD>(name);
        //     });
        // }
        // // Stacked configurations
        // else if ( sys == "RTFStackedForSSIM" )
        // {
        //     auto name = ParseArgumentsStacked(argc, argv, sys);
        //     Run([&]()
        //     {
        //         TrainAndEvaluate_StackedRTF<Loss::SSIM>(name);
        //     });
        // }
        // else if ( sys == "RTFStackedForIWSSIM" )
        // {
        //     auto name = ParseArgumentsStacked(argc, argv, sys);
        //     Run([&]()
        //     {
        //         TrainAndEvaluate_StackedRTF<Loss::IWSSIM>(name);
        //     });
        // }
        // else if ( sys == "RTFStackedForMSE" )
        // {
        //     auto name = ParseArgumentsStacked(argc, argv, sys);
        //     Run([&]()
        //     {
        //         TrainAndEvaluate_StackedRTF<Loss::MSE>(name);
        //     });
        // }
        // else if ( sys == "RTFStackedForPSNR" )
        // {
        //     auto name = ParseArgumentsStacked(argc, argv, sys);
        //     Run([&]()
        //     {
        //         TrainAndEvaluate_StackedRTF<Loss::PSNR>(name);
        //     });
        // }
        // else if ( sys == "RTFStackedForMAD" )
        // {
        //     auto name = ParseArgumentsStacked(argc, argv, sys);
        //     Run([&]()
        //     {
        //         TrainAndEvaluate_StackedRTF<Loss::MAD>(name);
        //     });
        // }
        // Evaluation mode
        else if ( sys == "Compare" )
        {
            if( argc < 3 )
            {
                MyMonitor::Report("Usage: TOFDenoising Compare <loss>");
                ::exit(1);
            }
            auto loss = (argv[2]);
            if( loss == "MSE" )
                Compare<Loss::MSE>();
            else if ( loss == "PSNR" )
                Compare<Loss::PSNR>();
            else if ( loss == "MyLorentzian" )
                Compare<Loss::MyLorentzian>();
            else if ( loss == "SSIM" )
                Compare<Loss::SSIM>();
            // else if ( loss == "IWSSIM" )
            //     Compare<Loss::IWSSIM>();
            else
            {
                MyMonitor::Report("Invalid <loss> argument (must be one of MSE, PSNR, MyLorentzian, SSIM)\n");
                ::exit(1);
            }
        }
        else if ( sys == "PredictStacked" )
        {
            ParseArgumentsPredictStacked(argc, argv, sys);
            PredictStacked();
        }
        else if ( sys == "PredictTrained" )
        {
            ParseArgumentsPredictDiscriminative(argc, argv, sys);
            PredictDiscriminative();
        }
        else
        {
            MyMonitor::Report("Invalid <system> argument (must be one of: %s)\n", systemStrings);
            ::exit(1);
        }

    }
    catch ( std::exception& e )
    {
        std::cerr << e.what() << std::endl;
        ::exit(1);
    }
    return 0;
}
