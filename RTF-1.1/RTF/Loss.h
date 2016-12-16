/* This file is part of the "Regression Tree Fields" (RTF) source code distribution,
 * obtained from http://research.microsoft.com/downloads.
 * It is provided to you under the terms of the Microsoft Research License Agreement
 * (MSR-LA). Please see License.txt for details.
 *
 *
 * File: Loss.h
 * Defines common loss functions and methods for evaluating them on images.
 *
 */

#ifndef H_RTF_LOSS_H
#define H_RTF_LOSS_H

#include <iostream>
#include <stdexcept>
#include <algorithm>

#include <Eigen/Eigen>
#include "Array.h"
#include "Image.h"
#include "Types.h"
#include "Utility.h"

#ifdef USE_MPI
#include "MPI.h"
#endif

namespace Loss
{
    // Loss tags. Use these as the template argument of the Learn() method of the BoostedRTF class
    // to specify the objective function that is optimized by functional gradient descent.

    // The classic mean squared error criterion for continuous prediction tasks
    class MSE;


    // Robust outlier-resistant "Lorentzian loss" with respect to the prediction,
    //
    //     rho(x) = log(1 + 0.5 ||x||^2)
    //
    // Reference: [Black1996] Black, Rangarajan, "On the unification of line processes, outlier rejection,
    //                        and robust statistics with applications in early vision", IJCV 96.
    class Lorentzian;

    // Smooth approximation of the discrete multi-class 0-1 error. The loss function expects your
    // ground truth data to be encoded as follows:
    //
    //   For a k-label problem, each ground truth label must be a k-dimensional vector, with each
    //   component corresponding to a particular labelling. The component corresponding to the
    //   observed labelling must be given the largest individual value of the vector.
    //   The loss function will then train the ensemble such as to predict vectors where the component
    //   of the predicted class attains a substantially larger value than all other classes. The proper
    //   way of decoding a prediction to a discrete label is hence to choose the class corresponding to
    //   the component with the largest value.
    //   The initial encoding *does* matter as it influences the capability of the RTF model to fit the
    //   response. Empirically, encoding the ground truth as orthonormal basis vectors has been found to
    //   work well. Note that the final ensemble will *not* produce this initial ground truth faithfully,
    //   but rather strive to find predictions that separate the correct from the wrong labellings in
    //   the sense outlined above.
    //   If you want to re-produce the exact *values* of the ground labels, MSE loss should be used instead.
    class MultiNomialLogistic;

    // Classification error, supports unlabelled pixels indicated by all-zero vector
    // MultiNomialLogistic is a smoothed version of this.
    class PerPixelError;

    // Structural similarity - pointers:
    // https://ece.uwaterloo.ca/~z70wang/publications/iciar10.pdf
    // https://ece.uwaterloo.ca/~z70wang/publications/MAD.pdf (gradient derived in appendix B)
    class SSIM;

    // And an information-content weighted version of SSIM:
    class IWSSIM;

    // Peak signal-to-noise ratio
    class PSNR;

    // Mean absolute error, differentiable
    class MAD;


    // True mean absolute error
    class MAE;

    // Energy-based convex surrogate loss functions leading to a convex parameter estimation problem;
    // these only make sense at training time.
    class ContinuousPerceptron;
    class ContinuousMeanField;

    // The loss of the max-margin learning objective based on the convex QP relaxation described in
    // Jancsary et al. (ICML 2013). Meant to be used for discrete variables.
    class DiscreteHamming;

    template<typename TTraits, typename TLoss>
    struct ErrorTerm;

    namespace Detail
    {
        /////////////////////////////////////
        // Loss interface given by:
        // Objective(ground, prediction)
        // Gradient(groud, prediction, &gradient)
        // static NormalizationConstant(img)
        // static Name()
        // static RequiresDiscreteInference()

        // Accumulates pointwise losses using function
        template<typename TTraits, typename TFunc>
        typename TTraits::ValueType PointwiseObjectiveHelper(const
                ImageRefC<typename TTraits::UnaryGroundLabel>& ground, const
                ImageRefC<typename TTraits::UnaryGroundLabel>& prediction,
                const TFunc& func)
        {
            const auto cy = ground.Height(), cx = ground.Width();
            const auto cc = TTraits::UnaryGroundLabel::Size;
            typename TTraits::ValueType loss = 0;

            #pragma omp parallel for
            for(int y = 0; y < cy; ++y)
            {
                typename TTraits::ValueType lineLoss = 0;

                for(int x = 0; x < cx; ++x)
                {
                    for(int c = 0; c < cc; ++c)
                    {
                        lineLoss +=  func(ground(x, y)[c], prediction(x, y)[c]);
                    }
                }

                #pragma omp atomic
                loss += lineLoss;
            }

            return loss;
        }

        template<typename TTraits>
        struct MSE
        {
            static typename TTraits::ValueType
            Objective(const ImageRefC<typename TTraits::UnaryGroundLabel>& ground,
                      const ImageRefC<typename TTraits::UnaryGroundLabel>& prediction)
            {
                const auto cy = ground.Height(), cx = ground.Width();
                const auto cc = TTraits::UnaryGroundLabel::Size;
                typename TTraits::ValueType loss = 0;

                #pragma omp parallel for
                for(int y = 0; y < cy; ++y)
                {
                    typename TTraits::ValueType lineLoss = 0;

                    for(int x = 0; x < cx; ++x)
                    {
                        for(int c = 0; c < cc; ++c)
                        {
                            auto delta = ground(x, y)[c] - prediction(x, y)[c];
                            lineLoss  += delta * delta;
                        }
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
                const auto cy = ground.Height(), cx = ground.Width();
                const auto cc = TTraits::UnaryGroundLabel::Size;

                #pragma omp parallel for
                for(int y = 0; y < cy; ++y)
                {
                    for(int x = 0; x < cx; ++x)
                    {
                        for(int c = 0; c < cc; ++c)
                        {
                            auto delta       = ground(x, y)[c] - prediction(x, y)[c];
                            gradient(x, y)[c] = -2.0 * delta;
                        }
                    }
                }
            }

            static typename TTraits::ValueType NormalizationConstant(const ImageRefC<typename TTraits::UnaryGroundLabel>& ground)
            {
                return ground.Width() * ground.Height() * TTraits::UnaryGroundLabel::Size;
            }

            static const char* Name()
            {
                return "MSE";
            }

            static bool RequiresDiscreteInference()
            {
                return false;
            }
        };

        // Mean absolute error, differentiable - assumes pixel values range from 0.0 to 1.0
        // This is straight from the Tappen et al. (CVPR 2007)
        template<typename TTraits>
        struct MAD
        {
            static typename TTraits::ValueType
            Objective(const ImageRefC<typename TTraits::UnaryGroundLabel>& ground,
                      const ImageRefC<typename TTraits::UnaryGroundLabel>& prediction)
            {
                const auto eps = 0.1 / 255.0;
                const auto cy  = ground.Height(), cx = ground.Width();
                const auto cc  = TTraits::UnaryGroundLabel::Size;
                typename TTraits::ValueType loss = 0;

                #pragma omp parallel for
                for(int y = 0; y < cy; ++y)
                {
                    typename TTraits::ValueType lineLoss = 0;

                    for(int x = 0; x < cx; ++x)
                    {
                        for(int c = 0; c < cc; ++c)
                        {
                            auto delta = ground(x, y)[c] - prediction(x, y)[c];
                            lineLoss  += std::sqrt(delta * delta + eps);
                        }
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
                const auto eps = 0.1 / 255.0;
                const auto cy  = ground.Height(), cx = ground.Width();
                const auto cc  = TTraits::UnaryGroundLabel::Size;

                #pragma omp parallel for
                for(int y = 0; y < cy; ++y)
                {
                    for(int x = 0; x < cx; ++x)
                    {
                        for(int c = 0; c < cc; ++c)
                        {
                            auto delta       = ground(x, y)[c] - prediction(x, y)[c];
                            gradient(x, y)[c] = -delta / std::sqrt(delta * delta + eps);
                        }
                    }
                }
            }

            static typename TTraits::ValueType NormalizationConstant(const ImageRefC<typename TTraits::UnaryGroundLabel>& ground)
            {
                return ground.Width() * ground.Height() * TTraits::UnaryGroundLabel::Size;
            }

            static int MaxLineSearchEvaluations()
            {
                return 200;
            }
            static const char* Name()
            {
                return "MAD";
            }

            static bool RequiresDiscreteInference()
            {
                return false;
            }
        };

        // Mean absolute error - this can only be used to *evaluate* predictions, since
        // the loss is *not* differentiable!
        template<typename TTraits>
        struct MAE
        {
            static typename TTraits::ValueType
            Objective(const ImageRefC<typename TTraits::UnaryGroundLabel>& ground,
                      const ImageRefC<typename TTraits::UnaryGroundLabel>& prediction)
            {
                const auto eps = 0.1 / 255.0;
                const auto cy  = ground.Height(), cx = ground.Width();
                const auto cc  = TTraits::UnaryGroundLabel::Size;
                typename TTraits::ValueType loss = 0;

                #pragma omp parallel for
                for(int y = 0; y < cy; ++y)
                {
                    typename TTraits::ValueType lineLoss = 0;

                    for(int x = 0; x < cx; ++x)
                    {
                        for(int c = 0; c < cc; ++c)
                        {
                            auto delta = ground(x, y)[c] - prediction(x, y)[c];
                            lineLoss  += std::abs(delta);
                        }
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
                throw std::runtime_error("MAE loss is not differentiable - use only to evaluate predictions.");
            }

            static typename TTraits::ValueType NormalizationConstant(const ImageRefC<typename TTraits::UnaryGroundLabel>& ground)
            {
                return ground.Width() * ground.Height() * TTraits::UnaryGroundLabel::Size;
            }

            static const char* Name()
            {
                return "MAE";
            }
        };

        template<typename TTraits>
        struct PSNR
        {
            static typename TTraits::ValueType
            Objective(const ImageRefC<typename TTraits::UnaryGroundLabel>& ground,
                      const ImageRefC<typename TTraits::UnaryGroundLabel>& prediction)
            {
                const auto cy = ground.Height(), cx = ground.Width();
                const auto cc = TTraits::UnaryGroundLabel::Size;
                const auto loss = MSE<TTraits>::Objective(ground, prediction);
                return 10.0 * (std::log10(loss) - std::log10(typename TTraits::ValueType(cx * cy * cc)));
            }

            static void
            Gradient(const ImageRefC<typename TTraits::UnaryGroundLabel>& ground,
                     const ImageRefC<typename TTraits::UnaryGroundLabel>& prediction,
                     const ImageRef<typename TTraits::UnaryGroundLabel>& gradient)
            {
                const auto cy = ground.Height(), cx = ground.Width();
                const auto cc = TTraits::UnaryGroundLabel::Size;
                MSE<TTraits>::Gradient(ground, prediction, gradient);
                const auto loss  = MSE<TTraits>::Objective(ground, prediction);
                const auto scale = 10.0 / (loss * std::log(10.0));

                #pragma omp parallel for
                for(int y = 0; y < cy; ++y)
                {
                    for(int x = 0; x < cx; ++x)
                    {
                        for(int c = 0; c < cc; ++c)
                        {
                            gradient(x, y)[c] *= scale;
                        }
                    }
                }
            }

            static typename TTraits::ValueType NormalizationConstant(const ImageRefC<typename TTraits::UnaryGroundLabel>& ground)
            {
                return 1;
            }

            static const char* Name()
            {
                return "PSNR";
            }

            static bool RequiresDiscreteInference()
            {
                return false;
            }
        };

        template<typename TTraits>
        struct Lorentzian
        {
            static typename TTraits::ValueType
            Objective(const ImageRefC<typename TTraits::UnaryGroundLabel>& ground,
                      const ImageRefC<typename TTraits::UnaryGroundLabel>& prediction)
            {
                const auto cy = ground.Height(), cx = ground.Width();
                const auto cc = TTraits::UnaryGroundLabel::Size;
                typename TTraits::ValueType loss = 0;

                for(int y = 0; y < cy; ++y)
                {
                    typename TTraits::ValueType lineLoss = 0;

                    for(int x = 0; x < cx; ++x)
                    {
                        typename TTraits::ValueType deltaNorm2 = 0;

                        for(int c = 0; c < cc; ++c)
                        {
                            auto delta  = ground(x, y)[c] - prediction(x, y)[c];
                            deltaNorm2 += delta * delta;
                        }

                        lineLoss += std::log(1.0 + 0.5 * deltaNorm2);
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
                const auto cy = ground.Height(), cx = ground.Width();
                const auto cc = TTraits::UnaryGroundLabel::Size;

                #pragma omp parallel for
                for(int y = 0; y < cy; ++y)
                {
                    for(int x = 0; x < cx; ++x)
                    {
                        typename TTraits::ValueType deltaNorm2 = 0;

                        for(int c = 0; c < cc; ++c)
                        {
                            auto delta  = ground(x, y)[c] - prediction(x, y)[c];
                            deltaNorm2 += delta * delta;
                        }

                        for(int c = 0; c < cc; ++c)
                        {
                            auto delta  = ground(x, y)[c] - prediction(x, y)[c];
                            gradient(x, y)[c] = -(delta / (1.0 + 0.5 * deltaNorm2));
                        }
                    }
                }
            }

            static typename TTraits::ValueType NormalizationConstant(const ImageRefC<typename TTraits::UnaryGroundLabel>& ground)
            {
                return ground.Width() * ground.Height() * TTraits::UnaryGroundLabel::Size;
            }

            static const char* Name()
            {
                return "Lorentzian";
            }

            static bool RequiresDiscreteInference()
            {
                return false;
            }
        };

        template<typename TTraits>
        struct MultiNomialLogistic
        {
            static typename TTraits::ValueType
            Objective(const ImageRefC<typename TTraits::UnaryGroundLabel>& ground,
                      const ImageRefC<typename TTraits::UnaryGroundLabel>& prediction)
            {
                const auto cy = ground.Height(), cx = ground.Width();
                const auto cc = TTraits::UnaryGroundLabel::Size;
                typename TTraits::ValueType loss = 0;

                #pragma omp parallel for
                for(int y = 0; y < cy; ++y)
                {
                    typename TTraits::ValueType lineLoss = 0;

                    for(int x = 0; x < cx; ++x)
                    {
                        // Ignore unobserved pixels, as encoded by an all-zero ground truth vector
                        if( std::count(&ground(x, y)[0], &ground(x, y)[0] + cc, 0.0) == cc )
                            continue;

                        // Compute log-sum-exp and find observed state (largest component of ground truth)
                        auto maxGround     = -std::numeric_limits<typename TTraits::ValueType>::max();
                        auto maxPred       = maxGround;
                        auto observedState = 0;

                        for(int c = 0; c < cc; ++c)
                        {
                            if(ground(x, y)[c] > maxGround)
                            {
                                maxGround     = ground(x, y)[c];
                                observedState = c;
                            }

                            if(prediction(x, y)[c] > maxPred)
                                maxPred       = prediction(x, y)[c];
                        }

                        auto sumExp = typename TTraits::ValueType(0);

                        for(int c = 0; c < cc; ++c)
                            sumExp += std::exp(prediction(x, y)[c] - maxPred);

                        const auto logSumExp = maxPred + std::log(sumExp);
                        // Loss is negative log-likelihood of observed state
                        lineLoss += (logSumExp - prediction(x, y)[observedState]);
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
                const auto cy = ground.Height(), cx = ground.Width();
                const auto cc = TTraits::UnaryGroundLabel::Size;

                #pragma omp parallel for
                for(int y = 0; y < cy; ++y)
                {
                    for(int x = 0; x < cx; ++x)
                    {
                        // Ignore unobserved pixels, as encoded by an all-zero ground truth vector
                        if( std::count(&ground(x, y)[0], &ground(x, y)[0] + cc, 0.0) == cc )
                            continue;

                        // Compute log-sum-exp and find observed state (largest component of ground truth)
                        auto maxGround     = -std::numeric_limits<typename TTraits::ValueType>::max();
                        auto maxPred       = maxGround;
                        auto observedState = 0;

                        for(int c = 0; c < cc; ++c)
                        {
                            if(ground(x, y)[c] > maxGround)
                            {
                                maxGround     = ground(x, y)[c];
                                observedState = c;
                            }

                            if(prediction(x, y)[c] > maxPred)
                                maxPred       = prediction(x, y)[c];
                        }

                        auto sumExp = typename TTraits::ValueType(0);

                        for(int c = 0; c < cc; ++c)
                            sumExp += std::exp(prediction(x, y)[c] - maxPred);

                        const auto logSumExp = maxPred + std::log(sumExp);

                        // Gradient with respect to prediction is model distribution - empirical distribution
                        for(int c = 0; c < cc; ++c)
                        {
                            const auto mu  = std::exp(prediction(x, y)[c] - logSumExp);
                            const auto obs = (c == observedState) ? 1.0 : 0.0;
                            gradient(x, y)[c] = mu - obs;
                        }
                    }
                }
            }

            static typename TTraits::ValueType NormalizationConstant(const ImageRefC<typename TTraits::UnaryGroundLabel>& ground)
            {
                const auto cy = ground.Height(), cx = ground.Width();
                const auto cc = TTraits::UnaryGroundLabel::Size;
                typename TTraits::ValueType observed = 0;

                #pragma omp parallel for
                for(int y = 0; y < cy; ++y)
                {
                    typename TTraits::ValueType lineObserved = 0;

                    for(int x = 0; x < cx; ++x)
                    {
                        // Ignore unobserved pixels, as encoded by an all-zero ground truth vector
                        if( std::count(&(ground(x,y)[0]), &(ground(x,y)[0]) + cc, 0.0) == cc )
                            continue;

                        lineObserved += 1.0;
                    }

                    #pragma omp atomic
                    observed += lineObserved;
                }
                return observed;
            }

            static const char* Name()
            {
                return "MultiNomialLogistic";
            }

            static bool RequiresDiscreteInference()
            {
                return false;
            }
        };

        template<typename TTraits>
        struct PerPixelError
        {
            static typename TTraits::ValueType
            Objective(const ImageRefC<typename TTraits::UnaryGroundLabel>& ground,
                      const ImageRefC<typename TTraits::UnaryGroundLabel>& prediction)
            {
                const auto cy = ground.Height(), cx = ground.Width();
                const auto cc = TTraits::UnaryGroundLabel::Size;
                typename TTraits::ValueType loss = 0;

                #pragma omp parallel for
                for(int y = 0; y < cy; ++y)
                {
                    typename TTraits::ValueType lineLoss = 0;

                    for(int x = 0; x < cx; ++x)
                    {
                        // Ignore unobserved pixels, as encoded by an all-zero ground truth vector
                        if( std::count(&ground(x, y)[0], &ground(x, y)[0] + cc, 0.0) == cc )
                            continue;

                        // Find predicted and observed state (largest component of ground truth vectors)
                        auto maxGround      = -std::numeric_limits<typename TTraits::ValueType>::max();
                        auto maxPred        = maxGround;
                        auto observedState  = 0;
                        auto predictedState = 0;

                        for(int c = 0; c < cc; ++c)
                        {
                            if(ground(x, y)[c] > maxGround)
                            {
                                maxGround     = ground(x, y)[c];
                                observedState = c;
                            }

                            if(prediction(x, y)[c] > maxPred)
                            {
                                maxPred        = prediction(x, y)[c];
                                predictedState = c;
                            }
                        }

                        // Every mis-predicted pixel counts equally as 1.0
                        lineLoss += (observedState == predictedState) ? 0.0 : 1.0;
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
                // Non-differentiable, only used at test-time
                assert(0);
            }

            static typename TTraits::ValueType NormalizationConstant(const ImageRefC<typename TTraits::UnaryGroundLabel>& ground)
            {
                const auto cy = ground.Height(), cx = ground.Width();
                const auto cc = TTraits::UnaryGroundLabel::Size;
                typename TTraits::ValueType observed = 0;

                #pragma omp parallel for
                for(int y = 0; y < cy; ++y)
                {
                    typename TTraits::ValueType lineObserved = 0;

                    for(int x = 0; x < cx; ++x)
                    {
                        // Ignore unobserved pixels, as encoded by an all-zero ground truth vector
                        if( std::count(&ground(x, y)[0], &ground(x, y)[0] + cc, 0.0) == cc )
                            continue;

                        lineObserved++;
                    }

                    #pragma omp atomic
                    observed += lineObserved;
                }

                return observed;
            }

            static const char* Name()
            {
                return "PerPixelError";
            }
        };

        template<typename TTraits>
        struct SSIM
        {
            const static int DynamicRange = 1;
            const static int WD           = 8;

            typedef Eigen::Matrix<typename TTraits::ValueType, WD*WD, 1> TWindowCol;

            static typename TTraits::ValueType
            Objective(const ImageRefC<typename TTraits::UnaryGroundLabel>& ground,
                      const ImageRefC<typename TTraits::UnaryGroundLabel>& prediction)
            {
                assert(ground.Width() >= WD && ground.Height() >= WD);    // image must be at least WDxWD
                const auto cy = ground.Height(), cx = ground.Width();
                const typename TTraits::ValueType C1 = (0.01 * DynamicRange) * (0.01 * DynamicRange);
                const typename TTraits::ValueType C2 = (0.03 * DynamicRange) * (0.03 * DynamicRange);
                typename TTraits::ValueType loss = 0;

                for( int c = 0; c < TTraits::UnaryGroundLabel::Size; ++c )
                {
                    #pragma omp parallel for
                    for(int y = 0; y <= cy - WD; ++y)
                    {
                        typename TTraits::ValueType lineLoss = 0;

                        for(int x = 0; x <= cx - WD; ++x)
                        {
                            TWindowCol a, b;

                            for(int offy = 0; offy < WD; ++offy)
                            {
                                for(int offx = 0; offx < WD; ++offx)
                                {
                                    a[offy * WD + offx] = ground(x + offx, y + offy)[c];
                                    b[offy * WD + offx] = prediction(x + offx, y + offy)[c];
                                }
                            }

                            const auto mean_a = TWindowCol::Constant(1.0).dot(a) / (WD * WD);
                            const auto mean_b = TWindowCol::Constant(1.0).dot(b) / (WD * WD);
                            const auto var_a  = (a.array() - mean_a).matrix().dot((a.array() - mean_a).matrix()) / (WD * WD - 1);
                            const auto var_b  = (b.array() - mean_b).matrix().dot((b.array() - mean_b).matrix()) / (WD * WD - 1);
                            const auto cov_ab = (a.array() - mean_a).matrix().dot((b.array() - mean_b).matrix()) / (WD * WD - 1);
                            lineLoss         += -((2.0 * mean_a * mean_b + C1) * (2.0 * cov_ab + C2)) / ((mean_a * mean_a + mean_b * mean_b + C1) * (var_a + var_b + C2));
                        }

                        #pragma omp atomic
                        loss += lineLoss;
                    }
                }
                return loss;
            }

            static void
            Gradient(const ImageRefC<typename TTraits::UnaryGroundLabel>& ground,
                     const ImageRefC<typename TTraits::UnaryGroundLabel>& prediction,
                     const ImageRef<typename TTraits::UnaryGroundLabel>& gradient)
            {
                assert(ground.Width() >= WD && ground.Height() >= WD);    // image must be at least WDxWD
                const auto cy = ground.Height(), cx = ground.Width();
                const typename TTraits::ValueType C1 = (0.01 * DynamicRange) * (0.01 * DynamicRange);
                const typename TTraits::ValueType C2 = (0.03 * DynamicRange) * (0.03 * DynamicRange);
                gradient.Clear();

                for( int c = 0; c < TTraits::UnaryGroundLabel::Size; ++c )
                {
                    // The idea is that lines that are > WD apart from each other are independent
                    // (i.e. do not affect each other's components of the gradient) and can thus be parallelized over.
                    for( int line = 0; line < WD; ++line )
                    {
                        #pragma omp parallel for
                        for(int y = line; y <= cy - WD; y += WD)
                        {
                            for(int x = 0; x <= cx - WD; ++x)
                            {
                                TWindowCol a, b;

                                for(int offy = 0; offy < WD; ++offy)
                                {
                                    for(int offx = 0; offx < WD; ++offx)
                                    {
                                        a[offy * WD + offx] = ground(x + offx, y + offy)[c];
                                        b[offy * WD + offx] = prediction(x + offx, y + offy)[c];
                                    }
                                }

                                const auto mean_a = TWindowCol::Constant(1.0).dot(a) / (WD * WD);
                                const auto mean_b = TWindowCol::Constant(1.0).dot(b) / (WD * WD);
                                const auto var_a  = (a.array() - mean_a).matrix().dot((a.array() - mean_a).matrix()) / (WD * WD - 1);
                                const auto var_b  = (b.array() - mean_b).matrix().dot((b.array() - mean_b).matrix()) / (WD * WD - 1);
                                const auto cov_ab = (a.array() - mean_a).matrix().dot((b.array() - mean_b).matrix()) / (WD * WD - 1);
                                const auto A1     = 2.0 * mean_a * mean_b + C1;
                                const auto A2     = 2.0 * cov_ab + C2;
                                const auto B1     = mean_a * mean_a + mean_b * mean_b + C1;
                                const auto B2     = var_a + var_b + C2;
                                const auto I       = TWindowCol::Constant(1.0);
                                TWindowCol nabla_b = 2.0 / (WD * WD * B1 * B1 * B2 * B2) * (A1 * B1 * (B2 * a - A2 * b) + B1 * B2 * (A2 - A1) * mean_a * I + A1 * A2 * (B1 - B2) * mean_b * I);

                                for(int offy = 0; offy < WD; ++offy)
                                {
                                    for(int offx = 0; offx < WD; ++offx)
                                    {
                                        gradient(x + offx, y + offy)[c] += -nabla_b[offy * WD + offx];
                                    }
                                }
                            }
                        }
                    }
                }
            }

            static typename TTraits::ValueType NormalizationConstant(const ImageRefC<typename TTraits::UnaryGroundLabel>& ground)
            {
                return (ground.Width() - (WD - 1)) * (ground.Height() - (WD - 1)) * TTraits::UnaryGroundLabel::Size;
            }

            static const char* Name()
            {
                return "SSIM";
            }

            static bool RequiresDiscreteInference()
            {
                return false;
            }
        };

        template<typename TTraits>
        struct IWSSIM
        {
            const static int DynamicRange = 1;
            const static int WD           = 8;

            typedef Eigen::Matrix<typename TTraits::ValueType, WD*WD, 1> TWindowCol;

            static typename TTraits::ValueType
            Objective(const ImageRefC<typename TTraits::UnaryGroundLabel>& ground,
                      const ImageRefC<typename TTraits::UnaryGroundLabel>& prediction)
            {
                assert(ground.Width() >= WD && ground.Height() >= WD);    // image must be at least WDxWD
                const auto cy = ground.Height(), cx = ground.Width();
                const typename TTraits::ValueType C1 = (0.01 * DynamicRange) * (0.01 * DynamicRange);
                const typename TTraits::ValueType C2 = (0.03 * DynamicRange) * (0.03 * DynamicRange);
                typename TTraits::ValueType loss = 0;

                for( int c = 0; c < TTraits::UnaryGroundLabel::Size; ++ c )
                {
                    #pragma omp parallel for
                    for(int y = 0; y <= cy - WD; ++y)
                    {
                        typename TTraits::ValueType lineLoss = 0;

                        for(int x = 0; x <= cx - WD; ++x)
                        {
                            TWindowCol a, b;

                            for(int offy = 0; offy < WD; ++offy)
                            {
                                for(int offx = 0; offx < WD; ++offx)
                                {
                                    a[offy * WD + offx] = ground(x + offx, y + offy)[c];
                                    b[offy * WD + offx] = prediction(x + offx, y + offy)[c];
                                }
                            }

                            const auto mean_a = TWindowCol::Constant(1.0).dot(a) / (WD * WD);
                            const auto mean_b = TWindowCol::Constant(1.0).dot(b) / (WD * WD);
                            const auto var_a  = (a.array() - mean_a).matrix().dot((a.array() - mean_a).matrix()) / (WD * WD - 1);
                            const auto var_b  = (b.array() - mean_b).matrix().dot((b.array() - mean_b).matrix()) / (WD * WD - 1);
                            const auto cov_ab = (a.array() - mean_a).matrix().dot((b.array() - mean_b).matrix()) / (WD * WD - 1);
                            const auto w      = std::log(1.0 + C2 + var_a / C2);
                            const auto ssim   = ((2.0 * mean_a * mean_b + C1) * (2.0 * cov_ab + C2)) / ((mean_a * mean_a + mean_b * mean_b + C1) * (var_a + var_b + C2));
                            lineLoss         += -(w * ssim);
                        }

                        #pragma omp atomic
                        loss += lineLoss;
                    }
                }

                return loss;
            }

            static void
            Gradient(const ImageRefC<typename TTraits::UnaryGroundLabel>& ground,
                     const ImageRefC<typename TTraits::UnaryGroundLabel>& prediction,
                     const ImageRef<typename TTraits::UnaryGroundLabel>& gradient)
            {
                assert(ground.Width() >= WD && ground.Height() >= WD);    // image must be at least WDxWD
                const auto cy = ground.Height(), cx = ground.Width();
                const typename TTraits::ValueType C1 = (0.01 * DynamicRange) * (0.01 * DynamicRange);
                const typename TTraits::ValueType C2 = (0.03 * DynamicRange) * (0.03 * DynamicRange);
                gradient.Clear();

                for( int c = 0; c < TTraits::UnaryGroundLabel::Size; ++c )
                {
                    // The idea is that lines that are > WD apart from each other are independent
                    // (i.e. do not affect each other's components of the gradient) and can thus be parallelized over.
                    for( int line = 0; line < WD; ++line )
                    {
                        #pragma omp parallel for
                        for(int y = line; y <= cy - WD; y += WD)
                        {
                            for(int x = 0; x <= cx - WD; ++x)
                            {
                                TWindowCol a, b;

                                for(int offy = 0; offy < WD; ++offy)
                                {
                                    for(int offx = 0; offx < WD; ++offx)
                                    {
                                        a[offy * WD + offx] = ground(x + offx, y + offy)[c];
                                        b[offy * WD + offx] = prediction(x + offx, y + offy)[c];
                                    }
                                }

                                const auto mean_a = TWindowCol::Constant(1.0).dot(a) / (WD * WD);
                                const auto mean_b = TWindowCol::Constant(1.0).dot(b) / (WD * WD);
                                const auto var_a  = (a.array() - mean_a).matrix().dot((a.array() - mean_a).matrix()) / (WD * WD - 1);
                                const auto var_b  = (b.array() - mean_b).matrix().dot((b.array() - mean_b).matrix()) / (WD * WD - 1);
                                const auto cov_ab = (a.array() - mean_a).matrix().dot((b.array() - mean_b).matrix()) / (WD * WD - 1);
                                const auto w      = std::log(1.0 + C2 + var_a / C2);
                                const auto A1     = 2.0 * mean_a * mean_b + C1;
                                const auto A2     = 2.0 * cov_ab + C2;
                                const auto B1     = mean_a * mean_a + mean_b * mean_b + C1;
                                const auto B2     = var_a + var_b + C2;
                                const auto I       = TWindowCol::Constant(1.0);
                                TWindowCol nabla_s = 2.0 / (WD * WD * B1 * B1 * B2 * B2) * (A1 * B1 * (B2 * a - A2 * b) + B1 * B2 * (A2 - A1) * mean_a * I + A1 * A2 * (B1 - B2) * mean_b * I);
                                TWindowCol nabla   = -(w * nabla_s);

                                for(int offy = 0; offy < WD; ++offy)
                                {
                                    for(int offx = 0; offx < WD; ++offx)
                                    {
                                        gradient(x + offx, y + offy)[c] += nabla[offy * WD + offx];
                                    }
                                }
                            }
                        }
                    }
                }
            }

            static typename TTraits::ValueType NormalizationConstant(const ImageRefC<typename TTraits::UnaryGroundLabel>& ground)
            {
                assert(ground.Width() >= WD && ground.Height() >= WD);    // image must be at least WDxWD
                const auto cy = ground.Height(), cx = ground.Width();
                const typename TTraits::ValueType C1 = (0.01 * DynamicRange) * (0.01 * DynamicRange);
                const typename TTraits::ValueType C2 = (0.03 * DynamicRange) * (0.03 * DynamicRange);
                typename TTraits::ValueType sum = 0;

                for( int c = 0; c < TTraits::UnaryGroundLabel::Size; ++c )
                {
                    #pragma omp parallel for
                    for(int y = 0; y <= cy - WD; ++y)
                    {
                        typename TTraits::ValueType lineSum = 0;

                        for(int x = 0; x <= cx - WD; ++x)
                        {
                            TWindowCol a;

                            for(int offy = 0; offy < WD; ++offy)
                                for(int offx = 0; offx < WD; ++offx)
                                    a[offy * WD + offx] = ground(x + offx, y + offy)[c];

                            const auto mean_a = TWindowCol::Constant(1.0).dot(a) / (WD * WD);
                            const auto var_a  = (a.array() - mean_a).matrix().dot((a.array() - mean_a).matrix()) / (WD * WD - 1);
                            const auto w      = std::log(1.0 + C2 + var_a / C2);
                            lineSum          += w;
                        }

                        #pragma omp atomic
                        sum += lineSum;
                    }
                }

                return sum;
            }

            static const char* Name()
            {
                return "IWSSIM";
            }

            static bool RequiresDiscreteInference()
            {
                return false;
            }
        };



        // Template specializations that choose the proper loss class based on a loss tag
        template<typename TTraits, typename TLossTag =::Loss::MSE>
        struct LossDispatcher
        {
            typedef ::Loss::Detail::MSE<TTraits> Loss;
        };

        template<typename TTraits>
        struct LossDispatcher<TTraits, ::Loss::PSNR>
        {
            typedef ::Loss::Detail::PSNR<TTraits> Loss;
        };

        template<typename TTraits>
        struct LossDispatcher<TTraits, ::Loss::MAD>
        {
            typedef ::Loss::Detail::MAD<TTraits> Loss;
        };

        template<typename TTraits>
        struct LossDispatcher<TTraits, ::Loss::MAE>
        {
            typedef ::Loss::Detail::MAE<TTraits> Loss;
        };

        template<typename TTraits>
        struct LossDispatcher<TTraits, ::Loss::Lorentzian>
        {
            typedef ::Loss::Detail::Lorentzian<TTraits> Loss;
        };

        template<typename TTraits>
        struct LossDispatcher<TTraits, ::Loss::MultiNomialLogistic>
        {
            typedef ::Loss::Detail::MultiNomialLogistic<TTraits> Loss;
        };

        template<typename TTraits>
        struct LossDispatcher<TTraits, ::Loss::PerPixelError>
        {
            typedef ::Loss::Detail::PerPixelError<TTraits> Loss;
        };

        template<typename TTraits>
        struct LossDispatcher<TTraits, ::Loss::SSIM>
        {
            typedef ::Loss::Detail::SSIM<TTraits> Loss;
        };

        template<typename TTraits>
        struct LossDispatcher<TTraits, ::Loss::IWSSIM>
        {
            typedef ::Loss::Detail::IWSSIM<TTraits> Loss;
        };

        template<typename TTraits>
        struct LossDispatcher<TTraits, ::Loss::ContinuousPerceptron>
        {
            typedef ::Loss::ErrorTerm<TTraits, ContinuousPerceptron> Loss;
        };

        template<typename TTraits>
        struct LossDispatcher<TTraits, ::Loss::DiscreteHamming>
        {
            typedef ::Loss::ErrorTerm<TTraits, DiscreteHamming> Loss;
        };


        template<typename TTraits, typename TLossTag, typename ImageRefT1, typename ImageRefT2>
        typename TTraits::ValueType
        Objective(const VecCRef<ImageRefT1>& ground,
                  const VecCRef<ImageRefT2>& prediction)
        {
            auto loss = typename TTraits::ValueType(0);

            for(size_t i = 0; i < ground.size(); ++i)
                loss += LossDispatcher<TTraits, TLossTag>::Loss::Objective(ground[i], prediction[i]);

            typename TTraits::ValueType globalLoss = 0.0;
#ifndef USE_MPI
            globalLoss = loss;
#else
            boost::mpi::all_reduce(MPI::Communicator(), loss, globalLoss, std::plus<typename TTraits::ValueType>());
#endif
            return globalLoss;
        }

        template<typename TTraits, typename TLossTag, typename ImageRefT1, typename ImageRefT2, typename ImageRefT3>
        static void
        PseudoResponse(const VecCRef<ImageRefT1>& ground,
                       const VecCRef<ImageRefT2>& prediction,
                       const VecCRef<ImageRefT3>& pseudoResponse)
        {
            const auto cc = TTraits::UnaryGroundLabel::Size;

            for(size_t i = 0; i < ground.size(); ++i)
            {
                LossDispatcher<TTraits, TLossTag>::Loss::Gradient(ground[i], prediction[i], pseudoResponse[i]);
                auto iPseudo = pseudoResponse[i];
                const int cx = iPseudo.Width(), cy = iPseudo.Height();

                #pragma omp parallel for
                for(int y = 0; y < cy; ++y)
                {
                    for(int x = 0; x < cx; ++x)
                        for(int c = 0; c < cc; ++c)
                            iPseudo(x, y)[c] *= typename TTraits::ValueType(-1); // pseudo response is anti-gradient
                }
            }
        }
    }

    template<typename TTraits, typename TLossTag>
    class Loss
    {
    private:
        typedef typename Detail::LossDispatcher<TTraits, TLossTag>::Loss LossImpl;

    public:

        static typename TTraits::ValueType
        Objective(const VecCRef<ImageRefC<typename TTraits::UnaryGroundLabel>>& ground,
                  const VecCRef<ImageRefC<typename TTraits::UnaryGroundLabel>>& prediction)
        {
            return Detail::Objective<TTraits, TLossTag>(ground, prediction);
        }

        static typename TTraits::ValueType
        Objective(const VecCRef<ImageRefC<typename TTraits::UnaryGroundLabel>>& ground,
                  const VecCRef<ImageRef<typename TTraits::UnaryGroundLabel>>& prediction)
        {
            return Detail::Objective<TTraits, TLossTag>(ground, prediction);
        }

        static typename TTraits::ValueType
        Objective(const ImageRefC<typename TTraits::UnaryGroundLabel>& ground,
                  const ImageRefC<typename TTraits::UnaryGroundLabel>& prediction)
        {
            return LossImpl::Objective(ground, prediction);
        }

        static ImageRefC<typename TTraits::UnaryGroundLabel>
        Gradient(const ImageRefC<typename TTraits::UnaryGroundLabel>& ground,
                 const ImageRefC<typename TTraits::UnaryGroundLabel>& prediction)
        {
            ImageRef<typename TTraits::UnaryGroundLabel> gradient(ground.Width(), ground.Height());
            LossImpl::Gradient(ground, prediction, gradient);
            return gradient;
        }

        static void
        PseudoResponse(const VecCRef<ImageRefC<typename TTraits::UnaryGroundLabel>>& ground,
                       const VecCRef<ImageRef<typename TTraits::UnaryGroundLabel>>& prediction,
                       const VecCRef<ImageRef<typename TTraits::UnaryGroundLabel>>& pseudoResponse)
        {
            return Detail::PseudoResponse<TTraits, TLossTag>(ground, prediction, pseudoResponse);
        }

        static void
        PseudoResponse(const VecCRef<ImageRefC<typename TTraits::UnaryGroundLabel>>& ground,
                       const VecCRef<ImageRefC<typename TTraits::UnaryGroundLabel>>& prediction,
                       const VecCRef<ImageRef<typename TTraits::UnaryGroundLabel>>& pseudoResponse)
        {
            return Detail::PseudoResponse<TTraits, TLossTag>(ground, prediction, pseudoResponse);
        }

        static const char* Name()
        {
            return LossImpl::Name();
        }

        static int MaxLineSearchEvaluations()
        {
            return LossImpl::MaxLineSearchEvaluations();
        }

        static typename TTraits::ValueType NormalizationConstant(const typename TTraits::DataSampler& db)
        {
            typename TTraits::ValueType loss = 0;

            for(size_t i = 0; i < db.GetImageCount(); ++i)
            {
                const auto gt = db.GetGroundTruthImage(i);
                loss += LossImpl::NormalizationConstant(gt);
            }

            typename TTraits::ValueType globalLoss = 0;
#ifndef USE_MPI
            globalLoss = loss;
#else
            boost::mpi::all_reduce(MPI::Communicator(), loss, globalLoss, std::plus<typename TTraits::ValueType>());
#endif
            return globalLoss;
        }

        static typename TTraits::ValueType NormalizationConstant(const VecCRef<ImageRefC<typename TTraits::UnaryGroundLabel>>& ground)
        {
            typename TTraits::ValueType loss = 0;

            for(size_t i = 0; i < ground.size(); ++i)
                loss += LossImpl::NormalizationConstant(ground[i]);

            typename TTraits::ValueType globalLoss = 0;
#ifndef USE_MPI
            globalLoss = loss;
#else
            boost::mpi::all_reduce(MPI::Communicator(), loss, globalLoss, std::plus<typename TTraits::ValueType>());
#endif
            return globalLoss;
        }

        static typename TTraits::ValueType NormalizationConstant(const ImageRefC<typename TTraits::UnaryGroundLabel>& ground)
        {
            return LossImpl::NormalizationConstant(ground);
        }

        static bool RequiresDiscreteInference()
        {
            return LossImpl::RequiresDiscreteInference();
        }
    };

#ifdef USE_MPI
    template<typename TTraits, typename TLossTag, typename TDataSampler>
    typename TTraits::ValueType
    MacroAveraged(const TDataSampler& testdb, const VecCRef<ImageRefC<typename TTraits::UnaryGroundLabel>>& prediction)
    {
        typedef ::Loss::Loss<TTraits, TLossTag> TLoss;
        typename TTraits::ValueType loss = 0.0;
        typename TTraits::ValueType numImages = testdb.GetImageCount();

        for(size_t i = 0; i < testdb.GetImageCount(); ++i)
        {
            const auto gt = testdb.GetGroundTruthImage(i);
            const auto numValues = TLoss::NormalizationConstant(gt);
            loss += TLoss::Objective(gt, prediction[i]) / numValues;
        }

        typename TTraits::ValueType globalLoss = 0.0;
        typename TTraits::ValueType globalNumImages = 0.0;

        boost::mpi::all_reduce(MPI::Communicator(), numImages, globalNumImages, std::plus<typename TTraits::ValueType>());
        boost::mpi::all_reduce(MPI::Communicator(), loss, globalLoss, std::plus<typename TTraits::ValueType>());

        return globalLoss/globalNumImages;
    }
#else
    template<typename TTraits, typename TLossTag, typename TDataSampler>
    typename TTraits::ValueType
    MacroAveraged(const TDataSampler& testdb, const VecCRef<ImageRefC<typename TTraits::UnaryGroundLabel>>& prediction)
    {
        typedef ::Loss::Loss<TTraits, TLossTag> TLoss;
        typename TTraits::ValueType loss = 0.0;

        for(size_t i = 0; i < testdb.GetImageCount(); ++i)
        {
            const auto gt = testdb.GetGroundTruthImage(i);
            const auto numValues = TLoss::NormalizationConstant(gt);
            loss += TLoss::Objective(gt, prediction[i]) / numValues;
        }

        return loss / testdb.GetImageCount();
    }
#endif

#ifdef USE_MPI
    template<typename TTraits, typename TLossTag, typename TDataSampler, typename TRegressOp>
    typename TTraits::ValueType
    MacroAveraged(const TDataSampler& testdb, const TRegressOp& regress)
    {
        typedef ::Loss::Loss<TTraits, TLossTag> TLoss;
        typename TTraits::ValueType loss = 0.0;
        typename TTraits::ValueType numImages = testdb.GetImageCount();

        for(size_t i = 0; i < testdb.GetImageCount(); ++i)
        {
            const auto gt = testdb.GetGroundTruthImage(i);
            const auto in = testdb.GetInputImage(i);
            const auto numValues = TLoss::NormalizationConstant(gt);
            loss += TLoss::Objective(gt, regress(in, i)) / numValues;
        }

        typename TTraits::ValueType globalLoss = 0.0;
        typename TTraits::ValueType globalNumImages = 0.0;

        boost::mpi::all_reduce(MPI::Communicator(), numImages, globalNumImages, std::plus<typename TTraits::ValueType>());
        boost::mpi::all_reduce(MPI::Communicator(), loss, globalLoss, std::plus<typename TTraits::ValueType>());

        return globalLoss/globalNumImages;
    }
#else
    template<typename TTraits, typename TLossTag, typename TDataSampler, typename TRegressOp>
    typename TTraits::ValueType
    MacroAveraged(const TDataSampler& testdb, const TRegressOp& regress)
    {
        typedef ::Loss::Loss<TTraits, TLossTag> TLoss;
        typename TTraits::ValueType loss = 0.0;

        for(size_t i = 0; i < testdb.GetImageCount(); ++i)
        {
            const auto gt = testdb.GetGroundTruthImage(i);
            const auto in = testdb.GetInputImage(i);
            const auto numValues = TLoss::NormalizationConstant(gt);
            loss += TLoss::Objective(gt, regress(in, i)) / numValues;
        }

        return loss / testdb.GetImageCount();
    }
#endif // USE_MPI

#ifdef USE_MPI
    template<typename TTraits, typename TLossTag, typename TDataSampler>
    typename TTraits::ValueType
    MicroAveraged(const TDataSampler& testdb, const VecCRef<ImageRefC<typename TTraits::UnaryGroundLabel>>& prediction)
    {
        typedef ::Loss::Loss<TTraits, TLossTag> TLoss;
        typename TTraits::ValueType numValues = 0;
        typename TTraits::ValueType loss = 0.0;

        for(size_t i = 0; i < testdb.GetImageCount(); ++i)
        {
            const auto gt = testdb.GetGroundTruthImage(i);
            const auto in = testdb.GetInputImage(i);
            numValues += TLoss::NormalizationConstant(gt);
            loss      += TLoss::Objective(gt, prediction[i]);
        }

        typename TTraits::ValueType globalNumValues = 0;
        typename TTraits::ValueType globalLoss = 0.0;

        boost::mpi::all_reduce(MPI::Communicator(), numValues, globalNumValues, std::plus<typename TTraits::ValueType>());
        boost::mpi::all_reduce(MPI::Communicator(), loss, globalLoss, std::plus<typename TTraits::ValueType>());

        return globalLoss / globalNumValues;
    }
#else
    template<typename TTraits, typename TLossTag, typename TDataSampler>
    typename TTraits::ValueType
    MicroAveraged(const TDataSampler& testdb, const VecCRef<ImageRefC<typename TTraits::UnaryGroundLabel>>& prediction)
    {
        typedef ::Loss::Loss<TTraits, TLossTag> TLoss;
        typename TTraits::ValueType numValues = 0;
        typename TTraits::ValueType loss = 0.0;

        for(size_t i = 0; i < testdb.GetImageCount(); ++i)
        {
            const auto gt = testdb.GetGroundTruthImage(i);
            const auto in = testdb.GetInputImage(i);
            numValues += TLoss::NormalizationConstant(gt);
            loss      += TLoss::Objective(gt, prediction[i]);
        }

        return loss / numValues;
    }
#endif

#ifdef USE_MPI
    template<typename TTraits, typename TLossTag, typename TDataSampler, typename TRegressOp>
    typename TTraits::ValueType
    MicroAveraged(const TDataSampler& testdb, const TRegressOp& regress)
    {
        typedef ::Loss::Loss<TTraits, TLossTag> TLoss;
        typename TTraits::ValueType numValues = 0;
        typename TTraits::ValueType loss = 0.0;

        for(size_t i = 0; i < testdb.GetImageCount(); ++i)
        {
            const auto gt = testdb.GetGroundTruthImage(i);
            const auto in = testdb.GetInputImage(i);
            numValues += TLoss::NormalizationConstant(gt);
            loss      += TLoss::Objective(gt, regress(in, i));
        }

        typename TTraits::ValueType globalNumValues = 0;
        typename TTraits::ValueType globalLoss = 0.0;

        boost::mpi::all_reduce(MPI::Communicator(), numValues, globalNumValues, std::plus<typename TTraits::ValueType>());
        boost::mpi::all_reduce(MPI::Communicator(), loss, globalLoss, std::plus<typename TTraits::ValueType>());

        return globalLoss / globalNumValues;
    }
#else // USE_MPI

    template<typename TTraits, typename TLossTag, typename TDataSampler, typename TRegressOp>
    typename TTraits::ValueType
    MicroAveraged(const TDataSampler& testdb, const TRegressOp& regress)
    {
        typedef ::Loss::Loss<TTraits, TLossTag> TLoss;
        typename TTraits::ValueType numValues = 0;
        typename TTraits::ValueType loss = 0.0;

        for(size_t i = 0; i < testdb.GetImageCount(); ++i)
        {
            const auto gt = testdb.GetGroundTruthImage(i);
            const auto in = testdb.GetInputImage(i);
            numValues += TLoss::NormalizationConstant(gt);
            loss      += TLoss::Objective(gt, regress(in, i));
        }

        return loss / numValues;
    }
#endif // USE_MPI

    template<typename TTraits, typename TLossTag>
    typename TTraits::ValueType
    PerImage(const ImageRefC<typename TTraits::UnaryGroundLabel>& ground,
             const ImageRefC<typename TTraits::UnaryGroundLabel>& prediction)
    {
        typedef ::Loss::Loss<TTraits, TLossTag> TLoss;
        return TLoss::Objective(ground, prediction) / TLoss::NormalizationConstant(ground);
    }

    template<typename TTraits>
    struct NoErrorTerm
    {
        static typename TTraits::ValueType
        ConstantContribution(const ImageRefC<typename TTraits::UnaryGroundLabel>& y)
        {
            return .0;
        }

        static void
        AddInImplicitMatrixMultipliedBy(const ImageRefC<typename TTraits::UnaryGroundLabel>& y,
                                        Compute::SystemVectorRef<typename TTraits::ValueType, TTraits::UnaryGroundLabel::Size>& Ax,
                                        Compute::SystemVectorCRef<typename TTraits::ValueType, TTraits::UnaryGroundLabel::Size>& x)
        {
        }

        static void
        AddInLinearContribution(const ImageRefC<typename TTraits::UnaryGroundLabel>& y,
                                Compute::SystemVectorRef<typename TTraits::ValueType, TTraits::UnaryGroundLabel::Size>& l)
        {
        }
    };


    template<typename TTraits, typename TLoss=ContinuousPerceptron>
    struct ErrorTerm
    {
        static typename TTraits::ValueType
        ConstantContribution(const ImageRefC<typename TTraits::UnaryGroundLabel>& ystar)
        {
            return 0.0;
        }

        static void
        AddInImplicitMatrixMultipliedBy(const ImageRefC<typename TTraits::UnaryGroundLabel>& ystar,
                                        Compute::SystemVectorRef<typename TTraits::ValueType, TTraits::UnaryGroundLabel::Size>& Qyhat,
                                        Compute::SystemVectorCRef<typename TTraits::ValueType, TTraits::UnaryGroundLabel::Size>& yhat)
        {
        }

        static void
        AddInLinearContribution(const ImageRefC<typename TTraits::UnaryGroundLabel>& ystar,
                                Compute::SystemVectorRef<typename TTraits::ValueType, TTraits::UnaryGroundLabel::Size>& l)
        {
        }

        static typename TTraits::ValueType NormalizationConstant(const ImageRefC<typename TTraits::UnaryGroundLabel>& ground)
        {
            return ground.Width() * ground.Height() * TTraits::UnaryGroundLabel::Size ;
        }

        static bool RequiresDiscreteInference()
        {
            return false;
        }
    };

    template<typename TTraits>
    struct ErrorTerm <TTraits, DiscreteHamming>
    {
        static typename TTraits::ValueType
        ConstantContribution(const ImageRefC<typename TTraits::UnaryGroundLabel>& ystar)
        {
            return 0.0;
        }

        static void
        AddInImplicitMatrixMultipliedBy(const ImageRefC<typename TTraits::UnaryGroundLabel>& ystar,
                                        Compute::SystemVectorRef<typename TTraits::ValueType, TTraits::UnaryGroundLabel::Size>& Qyhat,
                                        Compute::SystemVectorCRef<typename TTraits::ValueType, TTraits::UnaryGroundLabel::Size>& yhat)
        {
        }

        static void
        AddInLinearContribution(const ImageRefC<typename TTraits::UnaryGroundLabel>& ystar,
                                Compute::SystemVectorRef<typename TTraits::ValueType, TTraits::UnaryGroundLabel::Size>& l)
        {
            const auto ground = Utility::SolutionFromLabeling<TTraits>(ystar);
            const auto ones   = Compute::SystemVectorRef<typename TTraits::ValueType, TTraits::UnaryGroundLabel::Size>::VectorType::Ones(l.Raw().size());
            l.Raw() += (ones - ground);
        }

        static typename TTraits::ValueType NormalizationConstant(const ImageRefC<typename TTraits::UnaryGroundLabel>& ground)
        {
            return ground.Width() * ground.Height();
        }

        static bool RequiresDiscreteInference()
        {
            return true;
        }
    };
}

#endif // H_RTF_LOSS_H
