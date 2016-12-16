#ifndef H_TOFDENOISING_LOSS_H
#define H_TOFDENOISING_LOSS_H

#include "RTF/Loss.h"

#ifndef LORENTZIAN_LAMBDA
#define LORENTZIAN_LAMBDA 50.0
#pragma message ( "LORENTZIAN_LAMBDA should be defined before including Loss.h" )
#endif

namespace Loss
{
    class MyLorentzian;

    namespace Detail
    {

        template<typename TTraits>
        struct MyLorentzian
        {
            static typename TTraits::ValueType
            Objective(const ImageRefC<typename TTraits::UnaryGroundLabel>& ground,
                      const ImageRefC<typename TTraits::UnaryGroundLabel>& prediction)
            {
                assert(TTraits::UnaryGroundLabel::Size == 1);    // only single channel output supported
                const auto cy = ground.Height(), cx = ground.Width();

                // compute mean of delta
                typename TTraits::ValueType deltaMean = 0;
                #pragma omp parallel for
                for(int y = 0; y < cy; ++y)
                {
                    typename TTraits::ValueType lineMean = 0;
                    for(int x = 0; x < cx; ++x)
                    {
                        lineMean += (prediction(x,y)[0] - ground(x,y)[0]);
                    }
                    #pragma omp atomic
                    deltaMean += lineMean;
                }
                deltaMean /= (cy*cx); // normalize to obtain mean
                
                // compute loss with mean subtracted
                typename TTraits::ValueType loss = 0;
                #pragma omp parallel for
                for(int y = 0; y < cy; ++y)
                {
                    typename TTraits::ValueType lineLoss = 0;
                    for(int x = 0; x < cx; ++x)
                    {
                        auto delta  = prediction(x,y)[0] - ground(x,y)[0] - deltaMean;
                        lineLoss += std::log(1.0 + LORENTZIAN_LAMBDA * (delta*delta));
                    }
                    #pragma omp atomic
                    loss += lineLoss;
                }

                return loss;
            }

            static void
            Gradient(const ImageRefC<typename TTraits::UnaryGroundLabel>& ground,
                     const ImageRefC<typename TTraits::UnaryGroundLabel>& prediction,
                     const ImageRef<typename TTraits::UnaryGroundLabel>& gradient)
            {
                assert(TTraits::UnaryGroundLabel::Size == 1);    // only single channel output supported
                const auto cy = ground.Height(), cx = ground.Width();

                // compute mean of delta
                typename TTraits::ValueType deltaMean = 0;
                #pragma omp parallel for
                for(int y = 0; y < cy; ++y)
                {
                    typename TTraits::ValueType lineMean = 0;
                    for(int x = 0; x < cx; ++x)
                    {
                        lineMean += (prediction(x,y)[0] - ground(x,y)[0]);
                    }
                    #pragma omp atomic
                    deltaMean += lineMean;
                }
                deltaMean /= (cy*cx); // normalize to obtain mean

                // compute gradient and its mean
                typename TTraits::ValueType gradMean = 0;
                #pragma omp parallel for
                for(int y = 0; y < cy; ++y)
                {
                    typename TTraits::ValueType lineMean = 0;
                    for(int x = 0; x < cx; ++x)
                    {
                        auto delta  = prediction(x,y)[0] - ground(x,y)[0] - deltaMean;
                        gradient(x,y)[0] = (2.0 * LORENTZIAN_LAMBDA * delta) / (1.0 + LORENTZIAN_LAMBDA * (delta*delta));
                        lineMean += gradient(x,y)[0];
                    }
                    #pragma omp atomic
                    gradMean += lineMean;
                }
                gradMean /= (cy*cx); // normalize to obtain mean

                // adjust gradient
                #pragma omp parallel for
                for(int y = 0; y < cy; ++y)
                {
                    for(int x = 0; x < cx; ++x)
                    {
                        gradient(x,y)[0] -= gradMean;
                    }
                }

            }

            static typename TTraits::ValueType NormalizationConstant(const ImageRefC<typename TTraits::UnaryGroundLabel>& ground)
            {
                return ground.Width() * ground.Height();
            }

            static const char* Name()
            {
                return ("MyLorentzian(" + std::to_string(LORENTZIAN_LAMBDA) + ")").c_str();
            }

            static bool RequiresDiscreteInference()
            {
                return false;
            }
        };
   

        // Template specializations that choose the proper loss class based on a loss tag
        template<typename TTraits>
        struct LossDispatcher<TTraits, ::Loss::MyLorentzian>
        {
            typedef ::Loss::Detail::MyLorentzian<TTraits> Loss;
        };

    }

}

#endif // H_TOFDENOISING_LOSS_H