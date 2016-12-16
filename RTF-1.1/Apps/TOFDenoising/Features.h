/* This file is part of the "Regression Tree Fields" (RTF) source code distribution,
 * obtained from http://research.microsoft.com/downloads.
 * It is provided to you under the terms of the Microsoft Research License Agreement
 * (MSR-LA). Please see License.txt for details.
 *
 *
 * File: Features.h
 * Implements the feature class of the image denoising example.
 *
 */

#ifndef H_DENOISING_FEATURES_H
#define H_DENOISING_FEATURES_H

#include <random>
#include <cassert>
#include <fstream>
#include <iostream>
#include <exception>
#include <algorithm>

#include "RTF/Image.h"
#include "RTF/Array.h"
#include "RTF/Unary.h"
#include "RTF/Types.h"
#include "RTF/Training.h"

#include "Dataset.h"

namespace Denoising
{

    // Responsible for making the branching decisions within tree construction
    class Feature
    {
    public:
        // INTERFACE: the type of a pre-processed image (can be the same as the input image)
        typedef ImageRefC<Dataset::InputLabel> PreProcessType;

        // INTERFACE: linear basis function support - dimensionality of the bases
        static const size_t UnaryBasisSize    = 1 + Dataset::InputLabel::Size;
        static const size_t PairwiseBasisSize = 1 + 2*Dataset::InputLabel::Size;

        // INTERFACE: explicit threshold testing - how many thresholds to check
        static const size_t NumThresholdTests = 128;

        // Types of feature checks used to compute responses
        static const int UnaryType    = 0;
        static const int PairwiseType = 1;
        static const int SpatialType  = 2;

        Feature(int type_, int channel_, int offx1_, int offy1_,
                int offx2_, int offy2_, double threshold_ = 0.0)
            : type(type_), channel(channel_),
              offx1(offx1_), offy1(offy1_), offx2(offx2_), offy2(offy2_),
              threshold(threshold_)
        {
        }

        // INTERFACE: the feature class must have a default constructor
        Feature() : type(0), channel(0), offx1(0), offy1(0),
            offx2(0), offy2(0), threshold(0.0) {}

        // INTERFACE: create a new feature instance with the threshold set from the provided value
        Feature WithThreshold(double threshold_) const
        {
            return Feature(type, channel, offx1, offy1, offx2, offy2, threshold_);
        }

        // INTERFACE: decide whether to branch left or right
        bool operator()(int x, int y, const PreProcessType& image,
                        const VecCRef<Vector2D<int>>& offsets) const
        {
            return Response(x, y, image, offsets) < threshold;
        }

        // INTERFACE: compute the feature response that will be compared against a threshold
        double Response(int x, int y, const PreProcessType& image,
                        const VecCRef<Vector2D<int>>& offsets) const
        {
            switch( type )
            {
            case UnaryType:
            {
                return EvaluatePixelValue(x+offsets[0].x+offx1, y+offsets[0].y+offy1, channel, image);
            }
            case PairwiseType:
            {
                const auto pval1 = EvaluatePixelValue(x+offsets[0].x+offx1, y+offsets[0].y+offy1, channel, image);
                const auto pval2 = (offsets.size() < 2) ?
                                   EvaluatePixelValue(x+offx2, y+offy2, channel, image) :
                                   EvaluatePixelValue(x+offsets[1].x+offx2, y+offsets[1].y+offy2, channel, image);
                return (pval1 - pval2);
            }
            case SpatialType: // Distance from centre
            {
                const auto pos_y = (double) y / image.Height();
                const auto pos_x = (double) x / image.Width();
                const auto dis_y = 0.5 - pos_y;
                const auto dis_x = 0.5 - pos_x;
                return std::sqrt(dis_y*dis_y + dis_x*dis_x);
            }
            default:
                assert(0);
                return 0;
            }
        }

        static double EvaluatePixelValue(int x, int y, int c,
                                         const PreProcessType& prep)
        {
            // clamp
            x = std::min(std::max(0, x), prep.Width()-1);
            y = std::min(std::max(0, y), prep.Height()-1);

            return prep(x,y)[c];
        }

        // INTERFACE: fill in the linear basis
        static void ComputeBasis(int x, int y, const PreProcessType& prep,
                                 const VecCRef<Vector2D<int>>& offsets, double* basis)
        {
            // 1) Constant element
            basis[0] = 1.0;

            // 2) Unary basis
            std::copy(&(prep(x,y)[0]), &(prep(x,y)[0]) + Dataset::InputLabel::Size, &basis[1]);

            // 3) Second part of pairwise basis (if applicable)
            if( offsets.size() > 1 )
            {
                std::copy(&(prep(x + offsets[1].x, y + offsets[1].y)[0]),
                          &(prep(x + offsets[1].x, y + offsets[1].y)[0]) + Dataset::InputLabel::Size, &basis[1 + Dataset::InputLabel::Size]);
            }
        }

        // INTERFACE: return a prep-processed input image of type PreProcessType
        static PreProcessType PreProcess(const ImageRefC<Dataset::InputLabel>& input)
        {
            return input;
        }

        // INTERFACE: return a positive weight for the quadratic part of the specified unary factor
        static double ComputeQuadraticBasis(const PreProcessType& prep, const Vector2D<int>& i, size_t basisIndex)
        {
            return 1.0;
        }

        // INTERFACE: return a positive weight for the quadratic part of the specified pairwise factor
        static double ComputeQuadraticBasis(const PreProcessType& prep, const Vector2D<int>& i, const Vector2D<int>& j, size_t basisIndex)
        {
            return 1.0;
        }

        friend std::ostream& operator<<(std::ostream& os, const Feature& feat);
        friend std::istream& operator>>(std::istream& is, Feature& feat);

    private:
        int type, channel;
        int offx1, offy1, offx2, offy2;
        double threshold;
    };

    // INTERFACE (for serializatin): writes a feature instance to a stream
    inline std::ostream & operator<<(std::ostream& os, const Feature& feat)
    {
        os << feat.type << " " << feat.channel << " "
           << feat.offx1 << " " << feat.offy1 << " "
           << feat.offx2 << " " << feat.offy2 << " "
           << feat.threshold << std::endl;
        return os;
    }

    // INTERFACE (for serialization): reads a feature instance from a stream
    inline std::istream& operator>>(std::istream& is, Feature& feat)
    {
        is >> feat.type >> feat.channel >> feat.offx1 >> feat.offy1
           >> feat.offx2 >> feat.offy2 >> feat.threshold;
        return is;
    }

    // Repeatedly invoked to create feature instances that are then used for branching
    class FeatureSampler
    {
    private:
        std::mt19937                                        mt;
        std::normal_distribution<double>                    dthreshold;
        std::uniform_int_distribution<int>                  doffset;
        std::uniform_int_distribution<int>                  dtype;
        std::uniform_int_distribution<int>                  dchannel_index;
        std::bernoulli_distribution                         btest;

    public:
        FeatureSampler()
            : dthreshold(0.0f,1.0f), doffset(-10, +10), dtype(0, 100),
              dchannel_index(0 + 8, Dataset::InputLabel::Size-1), btest(0.5) // for splits, do not use early channels
        {
        }

        // INTERFACE: The type of features
        typedef Feature TFeature;

        // INTERFACE: Instantiates a new randomly drawn feature
        TFeature operator()(int)
        {
            int type = dtype(mt);

            int channel_index = dchannel_index(mt);
            bool centered_test1 = btest(mt);
            bool centered_test2 = btest(mt);

            if( type < 47 ) // about half of the checks are of some unary type
            {
                return Feature(Feature::UnaryType, channel_index,
                               centered_test1 ? 0 : doffset(mt),
                               centered_test1 ? 0 : doffset(mt),
                               0,
                               0);
            }
            else if ( type < 94 ) // almost half are of some pairwise type
            {
                return Feature(Feature::PairwiseType, channel_index,
                               centered_test1 ? 0 : doffset(mt),
                               centered_test1 ? 0 : doffset(mt),
                               centered_test2 ? 0 : doffset(mt),
                               centered_test2 ? 0 : doffset(mt));
            }
            else // and some of the features should always be of the simplest type
            {
                return Feature(Feature::UnaryType, 0, 0, 0, 0, 0);
            }
        }
    };
}

#endif // H_DENOISING_FEATURES_H
