/* This file is part of the "Regression Tree Fields" (RTF) source code distribution,
 * obtained from http://research.microsoft.com/downloads.
 * It is provided to you under the terms of the Microsoft Research License Agreement
 * (MSR-LA). Please see License.txt for details.
 *
 *
 * File: Dataset.h
 * Implements the dataset class of the image denoising example.
 *
 */

#ifndef H_DENOISING_DATASET_H
#define H_DENOISING_DATASET_H

#include <string>
#include <random>
#include <vector>
#include <cassert>
#include <fstream>
#include <iostream>
#include <exception>
#include <algorithm>

#include <boost/filesystem.hpp>

#include "RTF/Image.h"
#include "RTF/Array.h"
#include "RTF/Unary.h"
#include "RTF/Types.h"
#include "RTF/Training.h"

#ifdef USE_QI_DAGM12_FILTERBANK
#include "QiDAGM12Filterbank.h"
#define Filterbank QiDAGM12Filterbank
#else
#include "RFSFilterbank.h"
#define Filterbank RFSFilterbank
#endif

// TOF denoising
namespace Denoising
{
    class IOException : public std::exception
    {
    private:
        std::string w;
    public:
        IOException(const std::string& what_) : w(what_) {}

        virtual const char *what() const throw()
        {
            return w.c_str();
        }

        virtual ~IOException() throw() {}
    };

    // vl::
    // Helper for various for loops. Number of channels as **input** in TOF-context
    // 4 (C_k) + 2 ((C_3-C_1), (C_0-C_2)) + 1 ((C_3-C_1)/(C_0-C_2)) + 1 (arctan(<<)) + 1 (1/4*(C_0+C_1+C_2+C_3))
    // total of: 
    const int nChann = 9; 

    // template<typename TFilter, typename TStoreOp, typename TGetOp>
    // void ApplyFilter(int cx, int cy, int filter_size_x, int filter_size_y, TFilter filter_values, TStoreOp storeout, TGetOp getin)
    // {
    //     const int fcy_offset = filter_size_y/2;
    //     const int fcx_offset = filter_size_x/2;

    //     #pragma omp parallel for
    //     for (int y = 0; y < cy; ++y)
    //     {
    //         for (int x = 0; x < cx; ++x)
    //         {
    //             float pixel_resp[1] = { 0.0 };

    //             for (int fy = 0; fy < filter_size_y; ++fy)
    //             {
    //                 int eff_y = y+fy-fcy_offset;
    //                 eff_y = std::max(0, std::min(eff_y, cy-1));

    //                 for (int fx = 0; fx < filter_size_x; ++fx)
    //                 {
    //                     int eff_x = x+fx-fcx_offset;
    //                     eff_x = std::max(0, std::min(eff_x, cx-1));

    //                     const auto filter_value = filter_values[fy][fx];

    //                     for( int c = 0; c < 1; ++c )
    //                         pixel_resp[c] += getin(eff_x, eff_y, c) * filter_value;
    //                 }
    //             }
    //             for( int c = 0; c < 1; ++c )
    //                 storeout(x, y, c) = (float) pixel_resp[c];
    //         }
    //     }
    // }

    // The dataset class for denoising; methods and typedefs marked by INTERFACE
    // are required by the RTF code and must be implemented by any dataset class.
    class Dataset
    {
    public:

        // INTERFACE: define the output label of each pixel to be three-dimensional
        typedef Training::LabelVector<double, 1>   UnaryGroundLabel;
        typedef Training::LabelVector<double, 2>   PairwiseGroundLabel;

        // INTERFACE: define the type of data stored for each input pixel
        typedef Training::LabelVector<float,
                // vl::change
                4 + // Original Correlation input, four raw frames
                2 + // Differences:     C3 - C1, C0 - C2
                1 + // Quotient:        (C3-C1)/(C0-C2)
                1 + // Intensity:       (C0+C1+C2+C3)/4
                1 + // depth:           c0/(4*pi*f)*arctan{(C3-C1)/(C0-C2)}
                1 + // denoised depth map from previous layer (copied depth channel of first layer)
                // add filters via toggling following line
                // only apply filterbank on depth
                1 * Filterbank::filter_count + // Filter response
				0 > InputLabel;

        // vl::change: Not sure filter helps; Adapting file anyway
        static size_t filter_input_index(size_t c)
        {
            // returns index for insertion of filterbank-operation (after/behind depth channel)
            assert(c==0); // no shift
            return nChann; 
        }

        static size_t filter_output_index(size_t c, size_t fi)
        {
            // index after filter-channels
            return nChann + 1 + fi; 
        }

        static size_t channel_to_filter_index(size_t c) 
        {
            assert( c== 0); 
            return nChann - 1; // on depth only
        }

        // static size_t noise_level_index(size_t c)
        // {
        //     return (nChann + 4 * Filterbank::filter_count) + c;
        // }

        Dataset(const std::string& path_, const std::string& type_)
            : path(path_), type(type_)
        {
            ReadDescriptor();
        }

        Dataset(const std::string& name, const ImageRefC<float, nChann>& img)
        {
            assert( false ); //don't want this called
            // inputImages.resize(0);
            // groundTruthImages.resize(0);
            // fileNames.resize(0);
            // ImageRef<InputLabel> input(img.Width(), img.Height());

            // for( int y = 0; y < input.Height(); ++y )
            //     for( int x = 0; x < input.Width(); ++x )
            //         for( int c = 0; c < nChann; ++c )
            //             input(x,y)[c] = *(img.Ptr(x,y)+c);

            // // inputImages.push_back(PreProcessInput(input));
            // inputImages.push_back((input));
            // groundTruthImages.push_back(ImageRefC<UnaryGroundLabel>());
            // fileNames.push_back(name);
        }

        void ReleaseCache() const
        {
            for( size_t i = 0; i < GetImageCount(); ++i )
            {
                groundTruthImages[i] = ImageRefC<UnaryGroundLabel>();
                inputImages[i] = ImageRef<InputLabel>();
            }
        }

        // INTERFACE: return the number of images in the dataset
        size_t GetImageCount() const
        {
            return inputImages.size();
        }

        std::string GetImageName(size_t idx) const
        {
            assert( idx < fileNames.size() );
            return fileNames[idx];
        }

        // INTERFACE: return the idx'th ground truth image
        ImageRefC<UnaryGroundLabel> GetGroundTruthImage(size_t idx) const
        {

            assert(idx < groundTruthImages.size());
            if( ! groundTruthImages[idx] )
            {
                groundTruthImages[idx] = LoadGroundTruthImage(GroundTruthImagePath(idx));
            }

            return groundTruthImages[idx];
        }

        static void SaveGroundTruthImage(const ImageRefC<UnaryGroundLabel>& ground, const std::string& path)
        {
            // if( path.back() == 'g' || path.back() == 'G' )
            //     SaveGroundTruthImagePNG(ground, path);
            // else
                SaveGroundTruthImageDLM(ground, path);
        }

        // INTERFACE: return the idx'th input image.
        ImageRefC<InputLabel> GetInputImage(size_t idx) const
        {
            assert(idx < inputImages.size());
            if( ! inputImages[idx] )
            {
                inputImages[idx] = LoadInputImage(InputImagePath(idx));
            }

            return inputImages[idx];
        }

        static void SaveInputImage(const ImageRefC<InputLabel>& input, const std::string& path)
        {
            // if( path.back() == 'g' || path.back() == 'G' )
            //     return SaveInputImagePNG(input, path);
            // else
                return SaveInputImageDLM(input, path);
        }

        // static void SaveGroundTruthImagePNG(const ImageRefC<UnaryGroundLabel>& ground, const std::string& path)
        // {
        //     std::printf("Warning: Saving depth as PNG in TOF-context"); 
        //     ImageRef<unsigned char, 3> img(ground.Width(), ground.Height());
        //     for( int y = 0; y < img.Height(); ++y )
        //         for( int x = 0; x < img.Width(); ++x )
        //             for( int c = 0; c < 3; ++c )
        //                 *(img.Ptr(x,y) + c) = (unsigned char) (std::max(0.0f, std::min((float) ground(x,y)[c], 1.0f))*255.0f + .5f );
        //     Utility::WritePNG(img, path);
        // }

        static void SaveGroundTruthImageDLM(const ImageRefC<UnaryGroundLabel>& ground, const std::string& path)
        {
            ImageRef<float, 1> img(ground.Width(), ground.Height());
            for( int y = 0; y < img.Height(); ++y )
                for( int x = 0; x < img.Width(); ++x )
                    for( int c = 0; c < 1; ++ c )
                        *(img.Ptr(x,y)+c) = (float) ground(x,y)[c];
            Utility::WriteDLM(img, path);
        }

        // static void SaveInputImagePNG(const ImageRefC<InputLabel>& input, const std::string& path)
        // {
        //     std::printf("Warning: Saving %d channel frame as PNG in TOF-context", nChann); 

        //     ImageRef<unsigned char, 3> img(input.Width(), input.Height());
        //     for( int y = 0; y < img.Height(); ++y )
        //         for( int x = 0; x < img.Width(); ++x )
        //             for( int c = 0; c < 3; ++ c )
        //                 *(img.Ptr(x,y)+c) = (unsigned char) ( std::max(0.0f, std::min((float) input(x,y)[c], 1.0f))*255.0f + .5f );
        //     Utility::WritePNG(img, path);
        // }

        static void SaveInputImageDLM(const ImageRefC<InputLabel>& input, const std::string& path)
        {
            ImageRef<float, nChann> img(input.Width(), input.Height());
            for( int y = 0; y < img.Height(); ++y )
                for( int x = 0; x < img.Width(); ++x )
                    for( int c = 0; c < nChann; ++ c )
                        *(img.Ptr(x,y)+c) = input(x,y)[c];
            Utility::WriteDLM(img, path);
        }

        std::string GetImagePath(size_t idx, const std::string& prefix) const
        {
            assert(idx < GetImageCount());

            boost::filesystem::path dlmPath = path;
            dlmPath = dlmPath / prefix / (fileNames[idx] + ".dlm");
            // boost::filesystem::path pngPath = path;
            // pngPath = pngPath / prefix / (fileNames[idx] + ".png");

            if(  ! boost::filesystem::exists(dlmPath) )
                    throw IOException("Can't find " + prefix + " image: " + dlmPath.string());
                else
                    return dlmPath.string();
            // else
            // {
            //     if( ! boost::filesystem::exists(pngPath) )
            //         throw IOException("Can't find " + prefix + " image: " + pngPath.string());
            //     else
            //         return pngPath.string();
            // }
            // return pngPath.string();
        }

        std::string GroundTruthImagePath(size_t idx) const
        {
            return GetImagePath(idx, "labels");
        }

        static ImageRefC<UnaryGroundLabel> LoadGroundTruthImage(const std::string& path)
        {
            // if( path.back() == 'g' || path.back() == 'G' )
            //     return LoadGroundTruthImagePNG(path);
            // else
                return LoadGroundTruthImageDLM(path);
        }

       // static ImageRefC<UnaryGroundLabel> LoadGroundTruthImagePNG(const std::string& path)
        // {
        //     auto img = Utility::ReadPNG<unsigned char, 3>(path);
        //     ImageRef<UnaryGroundLabel> ret(img.Width(), img.Height());

        //     for( int y = 0; y < ret.Height(); ++y )
        //         for( int x = 0; x < ret.Width(); ++x )
        //             for( int c = 0; c < 3; ++c )
        //                 ret(x,y)[c] = *(img.Ptr(x,y)+c) / 255.0;
        //     return ret;
        // }

        static ImageRefC<UnaryGroundLabel> LoadGroundTruthImageDLM(const std::string& path)
        {
            auto img = Utility::ReadDLM<float, 1>(path);
            ImageRef<UnaryGroundLabel> ret(img.Width(), img.Height());

            for( int y = 0; y < ret.Height(); ++y )
                for( int x = 0; x < ret.Width(); ++x )
                    for( int c = 0; c < 1; ++c ) //vl.change: single channel depth map
                        ret(x,y)[c] = *(img.Ptr(x,y)+c);
            return ret;
        }

        std::string InputImagePath(size_t idx) const
        {
            return GetImagePath(idx, "images");
        }

        ImageRef<InputLabel> LoadInputImage(const std::string& path) const
        {
            // if( path.back() == 'g' || path.back() == 'G' )
            //     return LoadInputImagePNG(path);
            // else
                return LoadInputImageDLM(path);
        }

        // ImageRef<InputLabel> LoadInputImagePNG(const std::string& path) const
        // {
        //     std::cerr << path << std::endl;
        //     auto img = Utility::ReadPNG<unsigned char, 3>(path);
        //     ImageRef<InputLabel> ret(img.Width(), img.Height());

        //     for( int y = 0; y < ret.Height(); ++y )
        //         for( int x = 0; x < ret.Width(); ++x )
        //             for( int c = 0; c < 3; ++c )
        //                 ret(x,y)[c] = *(img.Ptr(x,y)+c) / 255.0f;
        //     return PreProcessInput(ret);
        // }

        ImageRef<InputLabel> LoadInputImageDLM(const std::string& path) const
        {
            auto img = Utility::ReadDLM<float, nChann>(path);
            ImageRef<InputLabel> ret(img.Width(), img.Height());

            for( int y = 0; y < ret.Height(); ++y )
                for( int x = 0; x < ret.Width(); ++x )
                    for( int c = 0; c < nChann; ++c ) 
                    {
                        if( c < 8)
                            ret(x,y)[c] = 0; // depth only
                        else
                            ret(x,y)[c] = *(img.Ptr(x,y)+c);
                    }
            return PreProcessInput(ret);
        }

        // INTERFACE: in a stacked RTF ensemble, perform pre-processing before each new layer
        void InitializeForCascadeLevel(size_t level, VecCRef<ImageRefC<UnaryGroundLabel>> previousPrediction) const
        {
            if( level > 0 ) // Initial pre-processing is done already while first loading the input images
                for( size_t idx = 0; idx < GetImageCount(); ++idx )
                    PreProcessInput(inputImages[idx], previousPrediction[idx]);
        }

        static ImageRef<InputLabel>
        PreProcessInput(ImageRef<InputLabel> input,
                        ImageRefC<UnaryGroundLabel> previousPrediction = ImageRef<UnaryGroundLabel>(0,0))
        {
            // 1) Determine the input for the filters
            // ... if this is the first layer in the stack, we simply use the RGB channels of the original input image
            // ... otherwise, we use the RGB channels of the prediction of the previous layer
            const int cx = input.Width();
            const int cy = input.Height();

            if( previousPrediction.Width() == 0 )
            {
                #pragma omp parallel for
                for( int y = 0; y < cy; ++y )
                    for( int x = 0; x < cx; ++x )
                        for( int c = 0; c < 1; ++c )
                            input(x,y)[filter_input_index(c)] = input(x,y)[channel_to_filter_index(c)];
                //UpdateNoiseLevels(input);
            }
            else
            {
                //std::printf("Warning: TOF-context does not support stacked RTF"); 
                #pragma omp parallel for
                for( int y = 0; y < cy; ++y )
                    for( int x = 0; x < cx; ++x )
                        for( int c = 0; c < 1; ++c ) //vl::change : depth map is single layer
                            input(x,y)[filter_input_index(c)] = previousPrediction(x,y)[c];
            }

            // 2) Update the filter responses and estimate the noise level
            return UpdateFilterResponses(input);
        }

        static ImageRef<InputLabel>
        UpdateFilterResponses(ImageRef<InputLabel> input)
        {
            // Computes a 2D convolution with each filter (single channel)
            // and stores the response in the input image for later use in the linear basis and the feature checks
            const int cx = input.Width();
            const int cy = input.Height();

            const int fcy_offset = Filterbank::filter_size_y/2;
            const int fcx_offset = Filterbank::filter_size_x/2;

            std::vector<float> filter_values(Filterbank::filter_size_y * Filterbank::filter_size_x * Filterbank::filter_count);
            int idx = 0;
            for (int fy = 0; fy < Filterbank::filter_size_y; ++fy)
                for (int fx = 0; fx < Filterbank::filter_size_x; ++fx)
                    for (int fi = 0; fi < Filterbank::filter_count; ++fi)
                        filter_values[idx++] = (float) Filterbank::filter_values[fi][fy][fx];

            #pragma omp parallel for
            for (int y = 0; y < cy; ++y)
            {
                for (int x = 0; x < cx; ++x)
                {
                    const float* filter_ptr = &(filter_values[0]);

                    auto &output_label = input(x,y);
                    //vl::change: this should be it - only apply to one input channel
                    memset(&output_label[filter_output_index(0,0)], 0, 1 * Filterbank::filter_count * sizeof(float));

                    for (int fy = 0; fy < Filterbank::filter_size_y; ++fy)
                    {
                        int eff_y = y+fy-fcy_offset;
                        eff_y = std::max(0, std::min(eff_y, cy-1));

                        for (int fx = 0; fx < Filterbank::filter_size_x; ++fx)
                        {
                            int eff_x = x+fx-fcx_offset;
                            eff_x = std::max(0, std::min(eff_x, cx-1));

                            const float* const input_ptr = &(input(eff_x, eff_y)[filter_input_index(0)]);
                            float* output_ptr = &output_label[filter_output_index(0,0)];

                            for (int fi = 0; fi < Filterbank::filter_count; ++fi)
                            {
                                const auto filter_value = *filter_ptr++;

                                for( int c = 0; c < 1; ++c ) 
                                    *output_ptr++ += input_ptr[c] * filter_value;
                            }
                        }
                    }
                }
            }

            return input;
        }

        // static ImageRef<InputLabel>
        // UpdateNoiseLevels(ImageRef<InputLabel> input)
        // {
        //     // Set up filter input
        //     const int cx = input.Width();
        //     const int cy = input.Height();

        //     ImageT<float, nChann> I1(cx, cy);
        //     #pragma omp parallel for
        //     for( int y = 0; y < cy; ++y )
        //         for( int x = 0; x < cx; ++x )
        //             for( int c = 0; c < nChann; ++c ) 
        //                 *(I1.Ptr(x,y)+c) = input(x,y)[filter_input_index(c)];

        //     // Apply horizontal high-pass filter
        //     const float oneOverSqrtTwo = 0.7071067811865475;
        //     float high_pass_filter_horz[1][2] = { {-oneOverSqrtTwo, oneOverSqrtTwo} };

        //     ImageT<float, nChann> I2(cx, cy);
        //     ApplyFilter(cx, cy, 2, 1, high_pass_filter_horz,
        //                 [&](int x, int y, int c) -> float& { return *(I2.Ptr(x,y)+c); },
        //                 [&](int x, int y, int c) -> float  { return *(I1.Ptr(x,y)+c); });

        //     // Apply vertical high-pass filter
        //     float high_pass_filter_vert[2][1] = { {-0.7071067811865475}, {0.7071067811865475} };

        //     ApplyFilter(cx, cy, 1, 2, high_pass_filter_vert,
        //                 [&](int x, int y, int c) -> float& { return *(I1.Ptr(x,y)+c); },
        //                 [&](int x, int y, int c) -> float  { return *(I2.Ptr(x,y)+c); });

        //     // Estimate local 3x3 means
        //     const float oneOverNine = 0.1111111111111111;
        //     float mean_filter[3][3] =
        //     {
        //         {oneOverNine, oneOverNine, oneOverNine},
        //         {oneOverNine, oneOverNine, oneOverNine},
        //         {oneOverNine, oneOverNine, oneOverNine}
        //     };

        //     ApplyFilter(cx, cy, 3, 3, mean_filter,
        //                 [&](int x, int y, int c) -> float& { return *(I2.Ptr(x,y)+c); },
        //                 [&](int x, int y, int c) -> float  { return *(I1.Ptr(x,y)+c); });

        //     // Compute pixel-wise squares of high-frequency filtered image (I1) and mean image (I2)
        //     #pragma omp parallel for
        //     for( int y = 0; y < cy; ++y )
        //         for( int x = 0; x < cx; ++x )
        //             for( int c = 0; c < nChann; ++c ) //vl::change
        //             {
        //                 *(I1.Ptr(x,y)+c) *= *(I1.Ptr(x,y)+c);
        //                 *(I2.Ptr(x,y)+c) *= *(I2.Ptr(x,y)+c);
        //             }

        //     // Add up squares in 3x3 windows
        //     float const_filter[3][3] =
        //     {
        //         {1.0, 1.0, 1.0},
        //         {1.0, 1.0, 1.0},
        //         {1.0, 1.0, 1.0}
        //     };

        //     ImageT<float, nChann> I3(cx, cy);
        //     ApplyFilter(cx, cy, 3, 3, const_filter,
        //                 [&](int x, int y, int c) -> float& { return *(I3.Ptr(x,y)+c); },
        //                 [&](int x, int y, int c) -> float  { return *(I1.Ptr(x,y)+c); });

        //     // Subtract squared mean (stored in I2) from sum of squares image (in I3), normalize
        //     // and take square root to obtain standard deviation
        //     const auto local_norm = 1.0/8.0;
        //     #pragma omp parallel for
        //     for( int y = 0; y < cy; ++y )
        //         for( int x = 0; x < cx; ++x )
        //             for( int c = 0; c < nChann; ++c ) 
        //                 input(x,y)[noise_level_index(c)] = std::sqrt(std::max(0.0, (*(I3.Ptr(x,y)+c) - 9.0 * *(I2.Ptr(x,y)+c)) * local_norm));

        //     return input;
        // }

    private:

#ifdef USE_MPI
        void ReadDescriptor()
        {
            boost::filesystem::path p = path;
            const std::string dpath = (p / (type + ".txt")).string();
            std::ifstream ifs(dpath);

            std::cerr << "reading " << dpath << std::endl;

            if( ifs.fail() )
                throw IOException("failed to open '" + dpath + "'");

            std::string file;
            size_t line = 0;
            while(std::getline(ifs, file))
            {
                if( (file != "") && (line++ % MPI::Communicator().size() == MPI::Communicator().rank()) )
                {
                    fileNames.push_back(file);
                }
            }
            inputImages.resize(fileNames.size());
            groundTruthImages.resize(fileNames.size());
        }
#else
        void ReadDescriptor()
        {
            boost::filesystem::path p = path;
            const std::string dpath = (p / (type + ".txt")).string();

            std::ifstream ifs(dpath);

            if( ifs.fail() )
                throw IOException("failed to open '" + dpath + "'");

            std::string file;
            while(std::getline(ifs, file))
            {
                if( file != "" )
                    fileNames.push_back(file);
            }
            groundTruthImages.resize(fileNames.size());
            inputImages.resize(fileNames.size());
        }
#endif

        const std::string                                path;
        const std::string                                type;

        std::vector<std::string>                         fileNames;
        mutable std::vector<ImageRefC<UnaryGroundLabel>> groundTruthImages;
        mutable std::vector<ImageRef<InputLabel>>        inputImages;
    };

} // namespace Denoising

#endif // H_DENOISING_DATASET_H
