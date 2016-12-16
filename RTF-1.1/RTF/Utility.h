/* This file is part of the "Regression Tree Fields" (RTF) source code distribution,
 * obtained from http://research.microsoft.com/downloads.
 * It is provided to you under the terms of the Microsoft Research License Agreement
 * (MSR-LA). Please see License.txt for details.
 *
 *
 * File: Utility.h
 * Implements various utility routines, mostly related to I/O and data type conversion.
 *
 */

#ifndef H_RTF_UTILITY_H
#define H_RTF_UTILITY_H

#include <stdexcept>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <array>
#include <vector>
#include <string>
#include <ctime>

#ifdef WIN32
#include <float.h>
#endif

#include <Eigen/Dense>

#include <itkImage.h>
#include <itkRGBPixel.h>
#include <itkRGBAPixel.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>

#include "Rect.h"
#include "Image.h"
#include "Array.h"
#include "Types.h"

namespace Constants
{
    inline double gamma()
    {
        return 0.57721566490153286061;
    }

    inline double pi()
    {
        return 3.14159265358979323846;
    }

    inline double e()
    {
        return 2.71828182845904523536;
    }
}

namespace Utility
{
    // Returns \log \sum_i \exp( v_i ).
    // Stores exp( v_j ) / (\sum_i \exp( v_i )) in \mu_i.
    //   (\mu can be interpreted as the marginals resulting from v and satisfies
    //    \mu_i \geq 0, \sum_i \mu_i = 1)
    template <typename TVector>
    typename TVector::RealScalar LogSumExp(const TVector & v, TVector &mu)
    {
        auto max = v.maxCoeff();
        auto lse = max + log((v.array() - max).exp().sum());
        mu = (v.array() - lse).exp();
        return lse;
    }

    template<typename TOutValue, typename TInValue>
    std::vector<TOutValue>
    vector_cast(const Eigen::Matrix<TInValue, Eigen::Dynamic, 1>& in)
    {
        std::vector<TOutValue> out(in.rows());
        const size_t size = static_cast<size_t>(in.rows());

        for(size_t i = 0; i < size; ++i)
            out[i] = static_cast<TOutValue>(in[i]);

        return out;
    }

    template<typename TOutValue, typename TInValue>
    Eigen::Matrix<TOutValue, Eigen::Dynamic, 1>
    vector_cast(const std::vector<TInValue>& in)
    {
        Eigen::Matrix<TOutValue, Eigen::Dynamic, 1> out(in.size());

        for(size_t i = 0; i < in.size(); ++i)
            out[i] = static_cast<TOutValue>(in[i]);

        return out;
    }

    template<typename TLabel>
    Eigen::Matrix<typename TLabel::ValueType, TLabel::Size, 1> LabelToVector(const TLabel& label)
    {
        Eigen::Matrix<typename TLabel::ValueType, TLabel::Size, 1> ret;
        memcpy(ret.data(), &label[0], sizeof(typename TLabel::ValueType) * TLabel::Size);
        return ret;
    }

    template<typename TLabel>
    Eigen::Matrix<typename TLabel::ValueType, 2*TLabel::Size, 1> LabelPairToVector(const TLabel& l1, const TLabel& l2)
    {
        typedef typename TLabel::ValueType TValue;
        Eigen::Matrix<TValue, 2*TLabel::Size, 1> ret;
        memcpy(ret.data(),              &l1[0], sizeof(TValue) * TLabel::Size);
        memcpy(ret.data() + TLabel::Size, &l2[0], sizeof(TValue) * TLabel::Size);
    }

    template <typename TLabel>
    TLabel VectorToLabel(const Eigen::Matrix<typename TLabel::ValueType, TLabel::Size, 1>& vec)
    {
        TLabel ret;
        memcpy(&ret[0], vec.data(), TLabel::Size * sizeof(typename TLabel::ValueType));
        return ret;
    }

    // Given a single stacked vector of cx * cy entries containing the means of all variables,
    // return a cx * cy image the labels of which represent these means. This is just a helper
    // function for the actual API routines.
    template <typename TTraits>
    ImageRefC<typename TTraits::UnaryGroundLabel> LabelingFromSolution(const int cx, const int cy,
            const Eigen::Matrix < typename TTraits::ValueType,
            Eigen::Dynamic, 1 > & solution)
    {
        typedef typename TTraits::ValueType TValue;
        typedef typename TTraits::UnaryGroundLabel TUnaryGroundLabel;
        const int VarDim = TTraits::UnaryGroundLabel::Size;
        ImageRef<TUnaryGroundLabel> map(cx, cy);
        Compute::SystemVectorCRef<TValue, VarDim> sref(cx, cy, solution);
        #pragma omp parallel for

        for(int y = 0; y < cy; ++y)
        {
            for(int x = 0; x < cx; ++x)
            {
                map(x, y) = VectorToLabel<TUnaryGroundLabel>(sref(x, y));
            }
        }

        return map;
    }

    // Convert a cx * cy image of labels into a single stacked vector of dimensionality cx * cy * VarDim.
    template <typename TTraits>
    Eigen::Matrix<typename TTraits::ValueType, Eigen::Dynamic, 1>
    SolutionFromLabeling(const ImageRefC<typename TTraits::UnaryGroundLabel>& labeling)
    {
        const int cx = labeling.Width(), cy = labeling.Height(), VarDim = TTraits::UnaryGroundLabel::Size;
        Eigen::Matrix<typename TTraits::ValueType, Eigen::Dynamic, 1> solution(cx * cy * VarDim);
        Compute::SystemVectorRef<typename TTraits::ValueType, VarDim> sref(cx, cy, solution);
        #pragma omp parallel for

        for(int y = 0; y < cy; ++y)
        {
            for(int x = 0; x < cx; ++x)
            {
                sref(x, y) = LabelToVector(labeling(x, y));
            }
        }

        return solution;
    }

    inline Rect<int> ComputeDeflateRect(const VecCRef<Vector2D<int>>& variables)
    {
        Rect<int> deflateRect;

        for(size_t v = 0; v < variables.size(); v++)
            deflateRect |= variables[v];

        return deflateRect;
    }

    inline Rect<int> ComputeProcessRect(const Rect<int>& deflateRect, int cx, int cy)
    {
        return Rect<int>(0, 0, cx, cy).DeflateRect(deflateRect);
    }

    template<typename TValue>
    TValue isfinite(TValue v)
    {
#ifdef WIN32
        return _finite(v);
#else
        return std::isfinite(v);
#endif // WIN32
    }

    template<typename TValue, unsigned NumChannels>
    ImageRef<TValue, NumChannels>
    ReadPNG(const std::string& path)
    {
        if( NumChannels == 1 )
        {
            typedef TValue PixelType;
            typedef itk::Image<PixelType> ImageType;
            typedef itk::ImageFileReader<ImageType> ReaderType;

            typename ReaderType::Pointer reader = ReaderType::New();
            reader->SetFileName(path);
            reader->Update();

            typename ImageType::Pointer img = reader->GetOutput();

            ImageRef<TValue, NumChannels> ret(img->GetLargestPossibleRegion().GetSize()[0], img->GetLargestPossibleRegion().GetSize()[1]);
            for(int y = 0; y < ret.Height(); ++y)
            {
                for(int x = 0; x < ret.Width(); ++x)
                {
                    typename ImageType::IndexType idx = {{x, y}};
                    ret(x, y) = img->GetPixel(idx);
                }
            }
            return ret;
        }
        else if ( NumChannels == 3 )
        {
            typedef itk::RGBPixel<TValue> PixelType;
            typedef itk::Image<PixelType> ImageType;
            typedef itk::ImageFileReader<ImageType> ReaderType;

            typename ReaderType::Pointer reader = ReaderType::New();
            reader->SetFileName(path);
            reader->Update();

            typename ImageType::Pointer img = reader->GetOutput();

            ImageRef<TValue, NumChannels> ret(img->GetLargestPossibleRegion().GetSize()[0], img->GetLargestPossibleRegion().GetSize()[1]);
            for(int y = 0; y < ret.Height(); ++y)
            {
                for(int x = 0; x < ret.Width(); ++x)
                {
                    typename ImageType::IndexType idx = {{x, y}};
                    *(ret.Ptr(x, y)+0) = img->GetPixel(idx).GetRed();
                    *(ret.Ptr(x, y)+1) = img->GetPixel(idx).GetGreen();
                    *(ret.Ptr(x, y)+2) = img->GetPixel(idx).GetBlue();
                }
            }
            return ret;
        }
        else if ( NumChannels == 4 )
        {
            typedef itk::RGBAPixel<TValue> PixelType;
            typedef itk::Image<PixelType> ImageType;
            typedef itk::ImageFileReader<ImageType> ReaderType;

            typename ReaderType::Pointer reader = ReaderType::New();
            reader->SetFileName(path);
            reader->Update();

            typename ImageType::Pointer img = reader->GetOutput();

            ImageRef<TValue, NumChannels> ret(img->GetLargestPossibleRegion().GetSize()[0], img->GetLargestPossibleRegion().GetSize()[1]);
            for(int y = 0; y < ret.Height(); ++y)
            {
                for(int x = 0; x < ret.Width(); ++x)
                {
                    typename ImageType::IndexType idx = {{x, y}};
                    *(ret.Ptr(x, y)+0) = img->GetPixel(idx).GetRed();
                    *(ret.Ptr(x, y)+1) = img->GetPixel(idx).GetGreen();
                    *(ret.Ptr(x, y)+2) = img->GetPixel(idx).GetBlue();
                    *(ret.Ptr(x, y)+3) = img->GetPixel(idx).GetAlpha();
                }
            }
            return ret;
        }
        else
        {
            throw std::runtime_error("ReadPNG: Can only read 1 (grayscale), 3 (RGB), or 4-channel (RGBA) input.");
            return ImageRef<TValue, NumChannels>();
        }
    }

    // Normalize the values to lie within [0,1]
    template<typename TValue>
    void Normalize(ImageRef<TValue, 1>& img)
    {
        // Determine maximum and minimum
        TValue maxval = -std::numeric_limits<TValue>::max();
        TValue minval = std::numeric_limits<TValue>::max();

        for(int y = 0; y < img.Height(); ++y)
        {
            for(int x = 0; x < img.Width(); ++x)
            {
                if( img(x, y) > maxval )
                    maxval = img(x,y);
                if( img(x, y) < minval )
                    minval = img(x,y);
            }
        }

        // Scale
        for(int y = 0; y < img.Height(); ++y)
        {
            for(int x = 0; x < img.Width(); ++x)
            {
                img(x, y) = (img(x, y) - minval)/(maxval - minval);
            }
        }
    }

    template<typename TValue, unsigned NumChannels>
    ImageRef<TValue, NumChannels>
    ReadDLM(const std::string& path)
    {
        std::vector<std::vector<TValue> > img;
        std::string line;
        std::ifstream in(path);

        if(! in)
            throw std::runtime_error("Could not open input file: " + path);

        while(std::getline(in, line))
        {
            std::vector<TValue> row;
            if(img.size() > 0)
                row.reserve(img.back().size());

            const char *pixptr = line.c_str();
            char * next;

            while(pixptr != (line.c_str() + line.length()))
            {
                const auto pixval = strtod(pixptr, &next);
                if( next == pixptr )
                    break;
                pixptr = next;
                row.push_back(pixval);
            }

            img.push_back(row);
        }

        if(img.size() == 0)
            throw std::runtime_error("Got empty ground truth file: " + path);

        if(img.front().size() % NumChannels != 0)
            throw std::runtime_error("Input file " + path + " seems to have wrong number of channels.");

        ImageRef<TValue, NumChannels> ret((int)(img.front().size() / NumChannels), (int)(img.size()));

        for(int y = 0; y < ret.Height(); ++y)
        {
            size_t idx = 0;

            for(int x = 0; x < ret.Width(); ++x)
                for(int c = 0; c < NumChannels; ++c)
                    *(ret.Ptr(x, y) + c) = img[y][idx++];
        }

        return ret;
    }

    template<typename TValue, unsigned NumChannels>
    ImageRef<TValue, NumChannels>
    ReadCommaSeparatedDLM(const std::string& path)
    {
        std::vector<std::vector<TValue> > img;
        std::string line;
        std::ifstream in(path);

        if(! in)
            throw std::runtime_error("Could not open input file: " + path);

        while(std::getline(in, line))
        {
            std::stringstream stream(line);
            std::vector<TValue> row;
            TValue pixval;

            std::string item;

            while(std::getline(stream, item, ','))
            {
                std::stringstream pixstr(item);
                pixstr >> pixval;
                row.push_back(pixval);
            }

            img.push_back(row);
        }

        if(img.size() == 0)
            throw std::runtime_error("Got empty ground truth file: " + path);

        if(img.front().size() % NumChannels != 0)
            throw std::runtime_error("Input file " + path + " seems to have wrong number of channels.");

        ImageRef<TValue, NumChannels> ret((int)(img.front().size() / NumChannels), (int)(img.size()));

        for(int y = 0; y < ret.Height(); ++y)
        {
            size_t idx = 0;

            for(int c = 0; c < NumChannels; ++c)
                for(int x = 0; x < ret.Width(); ++x)
                    *(ret.Ptr(x, y) + c) = img[y][idx++];
        }

        std::cerr << "Read comma-separated DLM image of size " << ret.Width() << ", " << ret.Height() << std::endl;
        return ret;
    }

    // write PNG from ImageRefC.
    template<typename TValue, unsigned NumChannels>
    void WritePNG(const ImageRefC<TValue, NumChannels>& img, const std::string& path)
    {
        if( NumChannels == 1 )
        {
            typedef TValue PixelType;
            typedef itk::Image<PixelType> ImageType;

            typename ImageType::IndexType start = {{0, 0}};
            typename ImageType::SizeType size;
            size[0] = img.Width();
            size[1] = img.Height();

            typename ImageType::RegionType region;
            region.SetSize(size);
            region.SetIndex(start);

            typename ImageType::Pointer out = ImageType::New();
            out->SetRegions(region);
            out->Allocate();

            for( int y = 0; y < img.Height(); ++y )
            {
                for( int x = 0; x < img.Width(); ++x )
                {
                    typename ImageType::IndexType idx = {{x, y}};
                    PixelType p = img(x, y);
                    out->SetPixel(idx, p);
                }
            }

            typedef itk::ImageFileWriter< ImageType > WriterType;
            typename WriterType::Pointer writer = WriterType::New();

            writer->SetFileName(path);
            writer->SetInput(out);
            writer->Update();
        }
        else if ( NumChannels == 3 )
        {
            typedef itk::RGBPixel<TValue> PixelType;
            typedef itk::Image<PixelType> ImageType;

            PixelType p;

            typename ImageType::IndexType start = {{0, 0}};
            typename ImageType::SizeType size;
            size[0] = img.Width();
            size[1] = img.Height();

            typename ImageType::RegionType region;
            region.SetSize(size);
            region.SetIndex(start);

            typename ImageType::Pointer out = ImageType::New();
            out->SetRegions(region);
            out->Allocate();

            for( int y = 0; y < img.Height(); ++y )
            {
                for( int x = 0; x < img.Width(); ++x )
                {
                    typename ImageType::IndexType idx = {{x, y}};
                    PixelType p;
                    p.SetRed(*(img.Ptr(x, y)+0));
                    p.SetGreen(*(img.Ptr(x, y)+1));
                    p.SetBlue(*(img.Ptr(x, y)+2));
                    out->SetPixel(idx, p);
                }
            }

            typedef itk::ImageFileWriter < ImageType > WriterType;
            typename WriterType::Pointer writer = WriterType::New();

            writer->SetFileName(path);
            writer->SetInput(out);
            writer->Update();
        }
        else if ( NumChannels == 4 )
        {
            typedef itk::RGBAPixel<TValue> PixelType;
            typedef itk::Image<PixelType> ImageType;

            typename ImageType::IndexType start = {{0, 0}};
            typename ImageType::SizeType size;
            size[0] = img.Width();
            size[1] = img.Height();

            typename ImageType::RegionType region;
            region.SetSize(size);
            region.SetIndex(start);

            typename ImageType::Pointer out = ImageType::New();
            out->SetRegions(region);
            out->Allocate();

            for( int y = 0; y < img.Height(); ++y )
            {
                for( int x = 0; x < img.Width(); ++x )
                {
                    typename ImageType::IndexType idx = {{x, y}};
                    PixelType p;
                    p.SetRed(*(img.Ptr(x, y)+0));
                    p.SetGreen(*(img.Ptr(x, y)+1));
                    p.SetBlue(*(img.Ptr(x, y)+2));
                    p.SetAlpha(*(img.Ptr(x, y)+3));
                    out->SetPixel(idx, p);
                }
            }

            typedef itk::ImageFileWriter< ImageType > WriterType;
            typename WriterType::Pointer writer = WriterType::New();

            writer->SetFileName(path);
            writer->SetInput(out);
            writer->Update();
        }
        else
        {
            throw std::runtime_error("WritePNG: Can only write 1 (grayscale), 3 (RGB), or 4-channel (RGBA) output.");
        }
    }

    template<typename TValue, unsigned NumChannels>
    void WritePNG(const ImageRef<TValue, NumChannels>& img, const std::string& path)
    {
        return WritePNG(ImageRefC<TValue, NumChannels>(img), path);
    }


    template<typename TValue, unsigned NumChannels>
    void WriteDLM(const ImageRefC<TValue, NumChannels>& img, const std::string& path)
    {
        const auto cx = img.Width();
        std::ofstream out(path);

        for(int y = 0; y < img.Height(); ++y)
        {
            for(int x = 0; x < cx; ++x)
            {
                for(int c = 0; c < NumChannels; ++c)
                {
                    out << std::fixed << std::setprecision(6) << *(img.Ptr(x, y) + c);

                    if((x < (cx - 1)) || (c < (NumChannels - 1)))
                        out << '\t';
                }
            }
            out << std::endl;
        }
        out.close();
    }

    template<typename TValue, unsigned NumChannels>
    void WriteDLM(const ImageRef<TValue, NumChannels>& img, const std::string& path)
    {
        return WriteDLM(ImageRefC<TValue, NumChannels>(img), path);
    }
}

size_t GetTickCountPortable()
{
    return static_cast<size_t>(((std::clock() / static_cast<double>(CLOCKS_PER_SEC))*1000));
}


#endif // H_RTF_UTILITY_H
