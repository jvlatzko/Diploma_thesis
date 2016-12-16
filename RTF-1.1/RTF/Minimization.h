/* This file is part of the "Regression Tree Fields" (RTF) source code distribution,
 * obtained from http://research.microsoft.com/downloads.
 * It is provided to you under the terms of the Microsoft Research License Agreement
 * (MSR-LA). Please see License.txt for details.
 *
 *
 * File: Minimization.h
 * Implements various algorithms for constrained and unconstrained optimization.
 *
 */

#ifndef H_RTF_MINIMIZATION_H
#define H_RTF_MINIMIZATION_H

#include <cmath>
#include <ctime>
#include <iomanip>
#include <limits>
#include <random>
#include <deque>
#include <iostream>

#include <Eigen/Dense>

namespace Minimization
{
    template<typename TValue>
    class Types
    {
    public:
        typedef Eigen::Matrix<TValue, Eigen::Dynamic, 1> TVector;
    };

    template<typename TValue>
    class ProjectableProblem
    {
    public:
        typedef typename Types<TValue>::TVector TVector;

        virtual ~ProjectableProblem() { }

        virtual TValue Eval(const TVector& x, TVector& g) = 0;

        virtual unsigned Dimensions() const = 0;

        virtual void ProvideStartingPoint(TVector& x0) const = 0;

        virtual TVector Project(const TVector& x) const = 0;

        virtual bool IsFeasible(const TVector& x) const = 0;

        virtual TValue Norm(const TVector& g) const
        {
            return g.norm();
        }

        virtual void Report(const char* fmt, ...) const
        {
            va_list args;
            va_start(args, fmt);
            vfprintf(stderr, fmt, args);
            va_end(args);
        }

        virtual void ReportVA(const char* fmt, va_list args) const
        {
            vfprintf(stderr, fmt, args);
        }
    };

    template<typename TValue>
    class UnconstrainedProblem : public ProjectableProblem<TValue>
    {
    public:
        typedef ProjectableProblem<TValue> Base;

        typename Base::TVector Project(const typename Base::TVector& x) const
        {
            return x;
        }

        bool IsFeasible(const typename Base::TVector& x) const
        {
            return true;
        }
    };

    // Port of senowozi's CheckDerivative tool to our function minimization interface
    template<typename TValue>
    bool CheckDerivative(ProjectableProblem<TValue>& prob,
                         TValue x_range, unsigned int test_count, TValue dim_eps, TValue grad_tol)
    {
        assert(dim_eps > 0.0);
        assert(grad_tol > 0.0);
        typedef typename ProjectableProblem<TValue>::TVector TVector;
        // Random number generation, for random perturbations
        std::mt19937 rgen; //(static_cast<unsigned>(std::time(0)) + 1);
        std::uniform_real_distribution<TValue> rdestu;  // range [0,1]
        // Random number generation, for random dimensions
        unsigned int dim = prob.Dimensions();
        std::mt19937 rgen2; //(static_cast<unsigned>(std::time(0)) + 2);
        std::uniform_int_distribution<unsigned int> rdestd(0, dim - 1);
        // Get base
        TVector x0(dim);
        prob.ProvideStartingPoint(x0);
        TVector xtest(dim);
        TVector grad(dim);
        TVector grad_d(dim);    // dummy

        for(unsigned int test_id = 0; test_id < test_count; ++test_id)
        {
            xtest = x0;

            for(unsigned int d = 0; d < dim; ++d)
                xtest[d] += 2.0 * x_range * rdestu(rgen) - x_range;

            //xtest = prob.Project(xtest); // ensure the point is feasible
            // Get exact derivative
            TValue xtest_fval = prob.Eval(xtest, grad);
            // Compute first-order finite difference approximation
            unsigned int test_dim = rdestd(rgen);
            xtest[test_dim] += dim_eps;
            TValue xtest_d_fval = prob.Eval(xtest, grad_d);
            TValue deriv_fd = (xtest_d_fval - xtest_fval) / dim_eps;

            // Check accuracy
            if(std::abs(deriv_fd - grad[test_dim]) > grad_tol)
            {
                std::ios_base::fmtflags original_format = std::cout.flags();
                std::streamsize original_prec = std::cout.precision();
                std::cout << std::endl;
                std::cout << "### DERIVATIVE CHECKER WARNING" << std::endl;
                std::cout << "### during test " << (test_id + 1) << " a violation "
                          << "in gradient computation was found:" << std::endl;
                std::cout << std::setprecision(6)
                          << std::setiosflags(std::ios::scientific);
                std::cout << "### dim " << test_dim << ", exact " << grad[test_dim]
                          << ", finite-diff " << deriv_fd
                          << ", absdiff " << fabs(deriv_fd - grad[test_dim])
                          << std::endl;
                std::cout << std::endl;
                std::cout.precision(original_prec);
                std::cout.flags(original_format);
                return (false);
            }
            else
            {
                std::cout << "### dim " << test_dim << " passed!" << std::endl;
            }
        }

        return (true);
    }

    template<typename VectorType, int k>
    void
    ProjectOntoUnitSimplex(Eigen::VectorBlock<VectorType, k> v)
    {
        typedef typename VectorType::Scalar TValue;
        Eigen::Matrix<TValue, k, 1> mu = v;
        std::sort(mu.data(), mu.data()+k); // sorted in ascending order
        size_t j  = 1;
        TValue s  = 0.0;
        TValue r  = 0.0;
        TValue sr = 0.0;

        while( j <= k )
        {
            const auto m = mu(k-j);
            s += m;
            if( (m - (1.0/j)*(s - 1.0)) > 0.0)
            {
                r  = j;
                sr = s;
            }
            j += 1;
        }
        const auto theta = (1.0/r)*(sr-1.0);
        j = 0;
        while( j < k )
        {
            v(j) = std::max(0.0, v(j) - theta);
            j += 1;
        }
    }


    template<typename TValue>
    TValue ProjectedGradientNorm(const ProjectableProblem<TValue>& problem,
                                 const typename Types<TValue>::TVector& x,
                                 const typename Types<TValue>::TVector& g)
    {
        return problem.Norm((problem.Project(x - g) - x));
    }

    template<typename TValue>
    TValue SPGComputeDirection(const ProjectableProblem<TValue>& problem,
                               const typename Types<TValue>::TVector& x,
                               const typename Types<TValue>::TVector& g,
                               TValue alpha,
                               typename Types<TValue>::TVector& d)
    {
        d = problem.Project(x - alpha * g) - x;
        return d.dot(g);
    }

    template<typename TValue>
    TValue SPGComputeStepSize(const ProjectableProblem<TValue>& problem,
                              const typename Types<TValue>::TVector& newx, const typename Types<TValue>::TVector& oldx,
                              const typename Types<TValue>::TVector& newg, const typename Types<TValue>::TVector& oldg)
    {
        auto s = newx - oldx;
        auto y = newg - oldg;
        return s.dot(s) / s.dot(y);
    }

    template<typename TValue>
    TValue SPGMinimize(ProjectableProblem<TValue> &problem,
                       typename ProjectableProblem<TValue>::TVector& x,
                       size_t maxNumIt  = 5000,
                       TValue geps      = (TValue) 1e-3,
                       bool verbose     = true,
                       bool fixedNumIt  = false,
                       size_t maxSrchIt = 100,
                       TValue gamma     = (TValue) 1e-4,
                       TValue maxAlpha  = (TValue) 1e4,
                       TValue minAlpha  = (TValue) 1e-10)
    {
        typedef typename Types<TValue>::TVector TVector;
        const size_t dim = problem.Dimensions();
        TVector g(dim), candx(dim), candg(dim), d(dim);
        problem.ProvideStartingPoint(x);
        TValue f      = problem.Eval(x, g);
        TValue gnorm  = ProjectedGradientNorm(problem, x, g);
        TValue alpha  = TValue(1.0) / (gnorm * x.norm());
        TValue gTd    = SPGComputeDirection(problem, x, g, alpha, d);
        size_t t      = 1;
        size_t fevals = 1;
        problem.Report("SPG: Initially  : f %-10.8f ||g|| %-10.8f\n", f, gnorm);

        while(gnorm > geps && t < maxNumIt)
        {
            TValue lambda = (TValue) 1.0;
            bool accepted = false;
            size_t srchIt = 0;

            do
            {
                TVector candx   = x + lambda * d;
                TValue  candf   = problem.Eval(candx, candg);
                TValue  suffdec = gamma * lambda * gTd;

                if(srchIt > 0 && verbose)
                    problem.Report("SPG:    SrchIt %4d: f %-10.8f t %-10.8f\n", srchIt, candf, alpha * lambda);

                if(candf <= f + suffdec)
                {
                    alpha    = std::min(maxAlpha, std::max(minAlpha, SPGComputeStepSize(problem, candx, x, candg, g)));
                    f        = candf;
                    accepted = true;
                    x        = candx;
                    g        = candg;
                }
                else if(srchIt >= maxSrchIt)
                {
                    accepted = true;
                }
                else
                {
                    lambda  *= 0.5;
                    srchIt++;
                }

                fevals++;
            }
            while(! accepted);

            if(srchIt >= maxSrchIt)
            {
                problem.Report("SPG: Linesearch cannot make further progress.\n");
                break;
            }

            if((! fixedNumIt) || (t % 10 == 0 || t == maxNumIt))
                gnorm = ProjectedGradientNorm(problem, x, g);

            gTd   = SPGComputeDirection(problem, x, g, alpha, d);

            if(verbose && !fixedNumIt)
                problem.Report("SPG: MainIt %4d: f %-10.8f ||g|| %-10.8f\n", t, f, gnorm);

            t++;
        }

        problem.Report("SPG: FinIt  %4d: f %-10.8f ||g|| %-10.8f fevals: %d\n", t - 1, f, gnorm, fevals);
        return f;
    }

    template<typename TConstrainedQuadratic>
    typename TConstrainedQuadratic::TValue
    SPGMinimizeCQ(TConstrainedQuadratic &problem,
                  typename TConstrainedQuadratic::TVector& x,
                  size_t maxNumIt  = 5000,
                  typename TConstrainedQuadratic::TValue geps = 1e-3,
                  bool verbose     = true,
                  bool fixedNumIt  = false,
                  size_t maxSrchIt = 100,
                  typename TConstrainedQuadratic::TValue gamma     = 1e-4,
                  typename TConstrainedQuadratic::TValue maxAlpha  = 1e4,
                  typename TConstrainedQuadratic::TValue minAlpha  = 1e-10)
    {
        typedef typename TConstrainedQuadratic::TVector TVector;
        typedef typename TConstrainedQuadratic::TValue  TValue;
        std::deque<TValue> f_hist;
        const size_t dim = problem.Dimensions();
        TVector g(dim), candx(dim), candg(dim), d(dim), Qd(dim);
        problem.ProvideStartingPoint(x);
        TValue f      = problem.Eval(x, g);
        TValue gnorm  = ProjectedGradientNorm(problem, x, g);
        TValue alpha  = TValue(1.0) / (gnorm * x.norm());
        TValue gTd    = SPGComputeDirection(problem, x, g, alpha, d);
        size_t t      = 1;
        size_t fevals = 1;
        problem.Report("SPGCQ: Initially  : f %-10.8f ||g|| %-10.8f\n", f, gnorm);

        f_hist.push_back(f);
        while(gnorm > geps && t < maxNumIt)
        {
            TValue lambda = 1.0;
            bool accepted = false;
            size_t srchIt = 0;

            do
            {
                TVector candx   = x + lambda * d;
                TValue  candf   = problem.Eval(candx, candg);
                TValue  suffdec = gamma * lambda * gTd;

                if(srchIt > 0 && verbose)
                    problem.Report("SPGCQ:    SrchIt %4d: f %-10.8f t %-10.8f\n", srchIt, candf, alpha * lambda);

                if(candf <= *std::max_element(f_hist.begin(), f_hist.end()) + suffdec)
                {
                    alpha    = std::min(maxAlpha, std::max(minAlpha, SPGComputeStepSize(problem, candx, x, candg, g)));
                    f        = candf;
                    accepted = true;
                    x        = candx;
                    g        = candg;
                    f_hist.push_back(f);
                    if( f_hist.size() > 8 )
                        f_hist.pop_front();
                }
                else if(srchIt >= maxSrchIt)
                {
                    accepted = true;
                }
                else
                {
                    lambda  *= 0.01;
                    srchIt++;
                }

                fevals++;
            }
            while(! accepted);

            if(srchIt >= maxSrchIt)
            {
                problem.Report("SPGCQ: Linesearch cannot make further progress.\n");
                break;
            }

            if((! fixedNumIt) || (t % 10 == 0 || t == maxNumIt))
                gnorm = ProjectedGradientNorm(problem, x, g);

            gTd   = SPGComputeDirection(problem, x, g, alpha, d);

            if(verbose && !fixedNumIt)
                problem.Report("SPGCQ: MainIt %4d: f %-10.8f ||g|| %-10.8f\n", t, f, gnorm);

            t++;
        }

        problem.Report("SPGCQ: FinIt  %4d: f %-10.8f ||g|| %-10.8f fevals: %d\n", t - 1, f, gnorm, fevals);
        return f;
    }

    // Compact representation of an n x n Hessian, maintained via L-BFGS updates
    template <typename TValue, size_t m>
    class CompactHessian
    {
    public:
        typedef Eigen::Matrix<TValue, Eigen::Dynamic, 1> TVector;
        typedef Eigen::Matrix<TValue, Eigen::Dynamic, Eigen::Dynamic> TMatrix;

    private:
        TValue sigma;
        std::deque<TVector> Y;
        std::deque<TVector> S;
        Eigen::FullPivHouseholderQR< TMatrix > MQr;
#if 0
        TMatrix N;
#endif
        // Returns the product of the tranpose of N (a 2k x n matrix)
        // with an n x 1 vector v. We never instantiate N explicitly to save memory.
        const TVector NTv(const TVector& v) const
        {
            const int k = (int) Y.size();
            TVector ntv(2 * k);
            //#pragma omp parallel
            {
                //#pragma omp for nowait

                for(int i = 0; i < k; ++i)
                    ntv[i] = sigma * S[i].dot(v);

                //#pragma omp for

                for(int i = k; i < 2 * k; ++i)
                    ntv[i] = Y[i - k].dot(v);
            }
            return ntv;
        }

        // Returns the product of N (a n x 2k matrix) with a 2k x 1 vector v
        const TVector Nv(const TVector& v) const
        {
            const int n = (int) Y.front().size(), k = (int) Y.size();
            TVector nv1 = TVector::Zero(n), nv2 = TVector::Zero(n);
            //#pragma omp parallel sections
            {
                //#pragma omp section

                for(int i = 0; i < k; ++i)
                    nv1 += v[i] * sigma * S[i];

                //#pragma omp section

                for(int i = k; i < 2 * k; ++i)
                    nv2 += v[i] * Y[i - k];
            }
            return nv1 + nv2;
        }


#if 0
        // Returns the product of N (a n x 2k matrix) with a 2k x 1 vector v
        const TVector Nv(const TVector& v) const
        {
            const int n = (int) Y.front().size(), k = (int) Y.size();
            TVector nv = TVector::Zero(n), threadnv = TVector::Zero(n);
            #pragma omp parallel firstprivate(threadnv) shared(nv)
            {
                #pragma omp for

                for(int i = 0; i < k; ++i)
                    threadnv += (v[i] * sigma) * S[i] + v[k + i] * Y[i];

                #pragma omp critical
                {
                    nv += threadnv;
                }
            }
            return nv;
        }
#endif
    public:

        TVector Times(const TVector& v) const
        {
            if(Y.empty())
                return v;
            else
                return sigma * v - Nv(MQr.solve(NTv(v)));
        }

        void Update(const TVector& y, const TVector& s)
        {
            // Compute scaling factor for initial Hessian, which we choose as
            const TValue yTs = y.dot(s);

            if(yTs < 1e-10)   // Ensure B remains strictly positive definite
                return;

            if( (yTs / y.dot(y)) < 1e-10 ) // Ensure numerical stability
                return;

            if( TValue(1.0) / (yTs / y.dot(y)) > 1e10 ||
                    TValue(1.0) / (yTs / y.dot(y)) < 1e-10 )
                return;

            sigma = TValue(1.0) / (yTs / y.dot(y));

            if(Y.size() >= m)
            {
                Y.pop_front();
                S.pop_front();
            }

            Y.push_back(y);
            S.push_back(s);

            const size_t k = Y.size(), n = Y.front().size();
            // D_k is the k x k diagonal matrix D_k = diag [s_0^Ty_0, ...,s_{k-1}^Ty_{k-1}].
            TVector minusd(k);
            //#pragma omp parallel for

            for(int i = 0; i < (int) k; ++ i)
                minusd[i] = - S[i].dot(Y[i]);

            const auto minusD = minusd.asDiagonal();
            // L_k is the k x k matrix with (L_k)_{i,j} = if( i > j ) s_i^T y_j else 0
            // (this is a lower triangular matrix with the main diagonal set to all zeroes)
            TMatrix L = TMatrix::Zero(k, k);
            //#pragma omp parallel for

            for(int j = 0; j < (int) k; ++j)
                for(size_t i = j + 1; i < k; ++i)
                    L(i, j) = S[i].dot(Y[j]);

            // S_k^T S_k is the symmetric k x k matrix with element (i,j) given by <s_i, s_j>
            TMatrix STS(k, k);
            //#pragma omp parallel for

            for(int j = 0; j < (int) k; ++j)
            {
                for(size_t i = j; i < k; ++i)
                {
                    const TValue sTs = S[i].dot(S[j]);
                    STS(i, j) = sTs;
                    STS(j, i) = sTs;
                }
            }

            // M is the 2k x 2k matrix given by: M = [ \sigma * S_k^T S_k    L_k ]
            //                                       [         L_k^T        -D_k ]
            TMatrix M(2 * k, 2 * k);
            M.topLeftCorner(k, k)     = sigma * STS;
            M.bottomLeftCorner(k, k)  = L.transpose();
            M.topRightCorner(k, k)    = L;
            M.bottomRightCorner(k, k) = minusD;
            // Save QR decomposition of M for later use in left-multiplication by M^{-1}
            MQr = M.fullPivHouseholderQr();
#if 0
            // N is the n x 2k matrix given by: N = [ \sigma * s_1  ... \sigma * s_k  y_1 ... y_k ],
            // where s_i and y_i are n x 1 column vectors.
            N.resize(n, 2 * k);

            for(int j = 0; j < k; ++j)
            {
                N.col(j)   = sigma * S[j];
                N.col(k + j) = Y[j];
            }

#endif
        }
    };

    template <typename TValue, size_t M>
    Eigen::Matrix<TValue, Eigen::Dynamic, 1> operator * (const CompactHessian<TValue, M>& B,
            const Eigen::Matrix<TValue, Eigen::Dynamic, 1>& v)
    {
        return B.Times(v);
    }

    // Forms a quadratic model around fun, the argmin of which then determines a feasible
    // quasi-Newton descent direction
    template <typename TValue, size_t M>
    class PQNSubproblem : public ProjectableProblem<TValue>
    {
    public:
        typedef Eigen::Matrix<TValue, Eigen::Dynamic, 1> TVector;

    private:
        ProjectableProblem<TValue>& fun;
        const TValue f_k;
        const TVector& x_k;
        const TVector& g_k;
        const CompactHessian<TValue, M>& B_k;

    public:
        PQNSubproblem(ProjectableProblem<TValue>& fun_,
                      TValue f_k_,
                      const TVector& x_k_,
                      const TVector& g_k_,
                      const CompactHessian<TValue, M>& B_k_) : fun(fun_), f_k(f_k_), x_k(x_k_), g_k(g_k_), B_k(B_k_)
        {
        }

        unsigned Dimensions() const
        {
            return fun.Dimensions();
        }

        // Compute objective and gradient of the quadratic model at the current iterate:
        //  q_k(p)         = f_k + (p-x_k)^T g_k + 1/2 (p-x_k)^T B_k(p-x_k)
        //  \nabla q_k(p)  = g_k + B_k(p-x_k)
        TValue Eval(const TVector& p, TVector& nabla)
        {
            const TVector d  = p - x_k;
            const TVector Bd = B_k * d;
            const TValue q_k = f_k + d.dot(g_k) + 0.5 * d.dot(Bd);
            nabla = g_k + Bd;
            return q_k;
        }

        TVector Project(const TVector& point) const
        {
            return fun.Project(point);
        }

        bool IsFeasible(const TVector& point) const
        {
            return fun.IsFeasible(point);
        }

        void ProvideStartingPoint(TVector& point) const
        {
            point = Project(x_k);
        }

        virtual TValue Norm(const TVector& g) const
        {
            return fun.Norm(g);
        }

        virtual void Report(const char* fmt, ...) const
        {
            va_list args;
            va_start(args, fmt);
            fun.ReportVA(fmt, args);
            va_end(args);
        }

        virtual void ReportVA(const char* fmt, va_list args) const
        {
            fun.ReportVA(fmt, args);
        }
    };

    template <size_t M, typename TValue>
    TValue PQNMinimize(ProjectableProblem<TValue>& prob, Eigen::Matrix<TValue, Eigen::Dynamic, 1>& x,
                       size_t maxNumIt   = 1000,
                       TValue optTol     = TValue(1e-3),
                       size_t numInnerIt = 50,
                       bool verbose      = true,
                       size_t maxSrchIt  = 100,
                       TValue gamma      = TValue(1e-6))
    {
        typedef Eigen::Matrix<TValue, Eigen::Dynamic, 1> TVector;
        const unsigned Dim = prob.Dimensions();
        TVector candx(Dim), g(Dim), candg(Dim), d(Dim), s(Dim), y(Dim);
        prob.ProvideStartingPoint(x);
        TValue f     = prob.Eval(x, g), candf = TValue(0.0), suffdec = TValue(0.0);
        TValue gnorm = ProjectedGradientNorm(prob, x, g);
        size_t fevals = 1;
        size_t t      = 1;
        prob.Report("PQN: Initially  : f %-10.8f ||g|| %-10.8f\n", f, gnorm);
        CompactHessian<TValue, M> B;

        while(gnorm > optTol && t < maxNumIt)
        {
            // Find descent direction
            if(t == 1)
            {
                // Initial direction, plain steepest descent
                d = prob.Project(x - 1e-4 * g / gnorm) - x;
            }
            else
            {
                // Update the limited-memory BFGS approximation to the Hessian
                B.Update(y, s);
                // Solve the quadratic subproblem approximately; we use the current iterate x as a guess
                // (note that this guarantees d being a descent direction if we perform at least
                //  one successful step of SPG - see Schmidt et al.)
                PQNSubproblem<TValue, M> subprob(prob, f, x, g, B);
                SPGMinimize(subprob, d, numInnerIt, optTol / TValue(10.0), false, true);
                d -= x;
            }

            // Backtracking line-search
            bool   accepted = false;
            TValue lambda   = TValue(1.0);
            TValue gTd      = g.dot(d);
            size_t srchit   = 0;

            do
            {
                candx   = x + lambda * d;
                candf   = prob.Eval(candx, candg);
                suffdec = gamma * lambda * gTd;

                if(srchit > 0 && verbose)
                    prob.Report("PQN:   SrchIt %4d: f %-10.8f t %-10.8f\n", srchit, candf, lambda);

                if(candf < f + suffdec)
                {
                    s = candx - x;
                    y = candg - g;
                    x = candx;
                    g = candg;
                    f = candf;
                    accepted = true;
                }
                else if(srchit >= maxSrchIt)
                {
                    accepted = true;
                }
                else
                {
                    lambda *= 0.5;
                    srchit++;
                }

                fevals++;
            }
            while(! accepted);

            if(srchit >= maxSrchIt)
            {
                prob.Report("PQN: Line search cannot make further progress");
                break;
            }

            gnorm = ProjectedGradientNorm(prob, x, g);

            if(verbose)
                prob.Report("PQN: MainIt %4d: f %-10.8f ||g|| %-10.8f\n", t, f, gnorm);

            t++;
        }

        prob.Report("PQN: FinIt  %4d: f %-10.8f ||g|| %-10.8f fevals: %d\n", t - 1, f, gnorm, fevals);
        return f;
    }

    // Represents a linear system Ax = b.
    template<typename TValue>
    class LinearSystem
    {
    public:
        typedef Eigen::Matrix<TValue, Eigen::Dynamic, 1> VectorType;

        // Store the right-hand side of the linear system, i.e. vector b, in the provided argument
        virtual void ProvideRightHandSide(VectorType& b) const = 0;

        // Store the inverse of the diagonal of the system matrix in the provided argument
        virtual void ProvideInverseDiagonal(VectorType& invDiag) const = 0;

        // Compute y = Ax and store the result in the provided output argument
        virtual void MultiplySystemMatrixBy(VectorType& y, const VectorType& x) const = 0;

        // Returns the number of components of b (or, equivalently, of x)
        virtual unsigned Dimensions() const = 0;

        virtual void Report(const char* fmt, ...) const
        {
            va_list args;
            va_start(args, fmt);
            vfprintf(stderr, fmt, args);
            va_end(args);
        }
    };

    template<typename TValue>
    Eigen::Matrix<TValue, Eigen::Dynamic, 1> CGSolve(const LinearSystem<TValue>& system,
            unsigned maxNumIt = 5000,
            TValue breakEps = 1e-6,
            bool verbose = true,
            bool *converged = NULL)
    {
        const unsigned Dim = system.Dimensions();
        typedef Eigen::Matrix<TValue, Eigen::Dynamic, 1> TVector;
        TVector Ap(Dim);
        TVector r(Dim);
        system.ProvideRightHandSide(r);
        TVector p     = r;
        TValue  rsold = r.dot(r);
        TVector x     = TVector::Zero(Dim);
        system.Report("CG: Initially  : ||r|| %-10.6f dim %u\n", sqrt(rsold), Dim);

        if( converged )
            *converged = true;

        if(sqrt(rsold) < breakEps)
            return x;

        unsigned t;

        for(t = 1; t <= maxNumIt; ++t)
        {
            system.MultiplySystemMatrixBy(Ap, p);
            TValue alpha = rsold / p.dot(Ap);
            x += alpha *  p;
            r -= alpha * Ap;
            const TValue rsnew = r.dot(r);

            if(verbose)
                system.Report("CG: MainIt %4d: ||r|| %-10.6f\n", t, sqrt(rsnew));

            if(sqrt(rsnew) < breakEps)
            {
                rsold = rsnew;
                break;
            }

            p *= rsnew / rsold;
            p += r;
            rsold = rsnew;
        }
        if( t > maxNumIt )
        {
            if( converged )
                *converged = false;
        }

        system.Report("CG: FinIt  %4d: ||r|| %-10.6f\n", t, sqrt(rsold));
        return x;
    }

    template<typename TValue>
    Eigen::Matrix<TValue, Eigen::Dynamic, 1> PCGSolve(const LinearSystem<TValue>& system,
            unsigned maxNumIt = 5000,
            TValue breakEps = 1e-6,
            bool verbose = true,
            bool *converged = NULL)
    {
        const unsigned Dim = system.Dimensions();
        typedef Eigen::Matrix<TValue, Eigen::Dynamic, 1> TVector;
        TVector Minv_(Dim);
        system.ProvideInverseDiagonal(Minv_);
        const auto Minv = Minv_.asDiagonal();
        TVector Ap(Dim);
        TVector r(Dim);
        system.ProvideRightHandSide(r);
        TVector z = Minv * r;
        TVector p = z;
        TValue  rsold = r.dot(z);
        TValue  nrm   = r.norm();
        TVector x     = TVector::Zero(Dim);
        system.Report("PCG: Initially  : ||r|| %-10.6f dim %u\n", nrm, Dim);

        if( converged )
            *converged = true;

        if(sqrt(rsold) < breakEps)
            return x;

        unsigned t;

        for(t = 1; t <= maxNumIt; ++t)
        {
            system.MultiplySystemMatrixBy(Ap, p);
            TValue alpha = rsold / p.dot(Ap);
            x += alpha *  p;
            r -= alpha * Ap;

            nrm = r.norm();
            if(verbose)
                system.Report("PCG: MainIt %4d: ||r|| %-10.6f\n", t, nrm);
            if(nrm < breakEps)
                break;

            z = Minv * r;
            const TValue rsnew = r.dot(z);

            p *= rsnew / rsold;
            p += z;
            rsold = rsnew;
        }
        if( t > maxNumIt )
        {
            if( converged )
                *converged = false;
        }

        system.Report("PCG: FinIt  %4d: ||r|| %-10.6f\n", t, nrm);
        return x;
    }

    template<typename TValue, size_t VarDim>
    Eigen::Matrix<TValue, Eigen::Dynamic, 1>
    ScaleByBlockDiagonal(const Eigen::Matrix<TValue, Eigen::Dynamic, VarDim>& D,
                         const Eigen::Matrix<TValue, Eigen::Dynamic, 1>& v)
    {
        Eigen::Matrix<TValue, Eigen::Dynamic, 1> ret(v.size());
        const auto NumBlocks = v.size() / VarDim;

        #pragma omp parallel for
        for( int i = 0; i < NumBlocks; ++i )
        {
            ret.template segment<VarDim>(VarDim*i) = D.template block<VarDim, VarDim>(VarDim*i, 0) * v.template segment<VarDim>(VarDim*i);
        }
        return ret;
    }

    template<typename TLinearSystem>
    typename TLinearSystem::VectorType BlockPCGSolve(const TLinearSystem &system,
            unsigned maxNumIt = 5000,
            typename TLinearSystem::TValue breakEps = 1e-6,
            bool verbose = true,
            bool *converged = NULL)
    {
        typedef typename TLinearSystem::TValue TValue;
        typedef typename TLinearSystem::VectorType TVector;
        typedef typename TLinearSystem::BlockDiagonalType TBlockDiagonal;
        static const size_t VarDim = TLinearSystem::Dim;

        const auto Dim = system.Dimensions();

        TBlockDiagonal Minv(Dim, VarDim);
        system.ProvideInverseBlockDiagonal(Minv);
        TVector Ap(Dim);
        TVector r(Dim);
        system.ProvideRightHandSide(r);
        TVector z = ScaleByBlockDiagonal<TValue, VarDim>(Minv, r);
        TVector p = z;
        TValue  rsold = r.dot(z);
        TValue  nrm   = r.norm();
        TVector x     = TVector::Zero(Dim);
        system.Report("PCG: Initially  : ||r|| %-10.6f dim %u\n", nrm, Dim);

        if( converged )
            *converged = true;

        if(sqrt(rsold) < breakEps)
            return x;

        unsigned t;

        for(t = 1; t <= maxNumIt; ++t)
        {
            system.MultiplySystemMatrixBy(Ap, p);
            TValue alpha = rsold / p.dot(Ap);
            x += alpha *  p;
            r -= alpha * Ap;

            nrm = r.norm();
            if(verbose)
                system.Report("PCG: MainIt %4d: ||r|| %-10.6f\n", t, nrm);
            if(nrm < breakEps)
                break;

            z = ScaleByBlockDiagonal<TValue, VarDim>(Minv, r);
            const TValue rsnew = r.dot(z);

            p *= rsnew / rsold;
            p += z;
            rsold = rsnew;
        }
        if( t > maxNumIt )
        {
            if( converged )
                *converged = false;
        }

        system.Report("PCG: FinIt  %4d: ||r|| %-10.6f\n", t, nrm);
        return x;
    }

    template <typename TValue, size_t m>
    class CompactInverseHessian
    {
    public:
        typedef Eigen::Matrix<TValue, Eigen::Dynamic, 1> TVector;

    private:
        std::deque<TVector> Y;
        std::deque<TVector> S;

    public:
        bool Empty() const
        {
            return Y.empty();
        }

        void Clear()
        {
            Y.clear();
            S.clear();
        }

        void Update(const TVector& y, const TVector& s)
        {
            // Compute scaling factor for initial Hessian, which we choose as
            const TValue yTs = y.dot(s);

            if(yTs < 1e-12)   // Ensure B remains strictly positive definite
                return;

            if(Y.size() >= m)
            {
                Y.pop_front();
                S.pop_front();
            }

            Y.push_back(y);
            S.push_back(s);
        }

        // Returns the product of vector g with the compact inverse Hessian;
        // this is computed using the classic two-loop LBFGS formula.
        TVector Times(const TVector& v) const
        {
            const size_t k = Y.size();

            if(k == 0)
                return v;

            TVector p(v), alphas(k);

            for(int i = (int) k - 1; i  >= 0; --i)
            {
                const auto alpha = S[i].dot(p) / S[i].dot(Y[i]);
                p -= alpha * Y[i];
                alphas[i] = alpha;
            }

            p *= S.back().dot(Y.back()) / Y.back().squaredNorm();

            for(size_t i = 0; i < k; ++i)
            {
                const auto beta = Y[i].dot(p) / Y[i].dot(S[i]);
                p += (alphas[i] - beta) * S[i];
            }

            return p;
        }
    };

    template <typename TValue, size_t M>
    Eigen::Matrix<TValue, Eigen::Dynamic, 1> operator * (const CompactInverseHessian<TValue, M>& H,
            const Eigen::Matrix<TValue, Eigen::Dynamic, 1>& v)
    {
        return H.Times(v);
    }

    template <size_t M, typename TValue>
    TValue LBFGSMinimize(UnconstrainedProblem<TValue>& prob, Eigen::Matrix<TValue, Eigen::Dynamic, 1>& x,
                         size_t maxNumIt   = 1000,
                         TValue optTol     = TValue(1e-3),
                         bool verbose      = true,
                         size_t maxSrchIt  = 10,
                         TValue gamma      = TValue(1e-6))
    {
        typedef Eigen::Matrix<TValue, Eigen::Dynamic, 1> TVector;
        const unsigned Dim = prob.Dimensions();
        TVector candx(Dim), g(Dim), candg(Dim), d(Dim), s(Dim), y(Dim);
        prob.ProvideStartingPoint(x);
        TValue f     = prob.Eval(x, g), candf = TValue(0.0), suffdec = TValue(0.0);
        TValue gnorm = prob.Norm(g);
        size_t fevals = 1;
        size_t t      = 1;
        prob.Report("LBFGS: Initially  : f %-10.8f ||g|| %-10.8f\n", f, gnorm);
        CompactInverseHessian<TValue, M> H;

        while(gnorm > optTol && t < maxNumIt)
        {
            // Find descent direction
            if(H.Empty())
                d = - g / gnorm; // Initial direction, plain steepest descent
            else
                d = - (H * g);   // Scale by inverse Hessian

            // Backtracking line-search
            bool   accepted = false;
            TValue lambda   = TValue(1.0);
            TValue gTd      = g.dot(d);
            size_t srchit   = 0;

            do
            {
                candx   = x + lambda * d;
                candf   = prob.Eval(candx, candg);
                suffdec = gamma * lambda * gTd;

                if(srchit > 0 && verbose)
                    prob.Report("LBFGS:   SrchIt %4d: f %-10.8f t %-10.8f\n", srchit, candf, lambda);

                if(candf < f + suffdec)
                {
                    s = candx - x;
                    y = candg - g;
                    x = candx;
                    g = candg;
                    f = candf;
                    accepted = true;
                }
                else if(srchit >= maxSrchIt)
                {
                    accepted = true;
                }
                else
                {
                    lambda *= 0.5;
                    srchit++;
                }

                fevals++;
            }
            while(! accepted);

            if(srchit >= maxSrchIt)
            {
                prob.Report("LBFGS: Line search cannot make further progress\n");
                break;
            }

            // Valid step - update the L-BFGS approximation to the inverse Hessian
            H.Update(y, s);
            gnorm = prob.Norm(g);

            if(verbose)
                prob.Report("LBFGS: MainIt %4d: f %-10.8f ||g|| %-10.4f\n", t, f, gnorm);

            t++;
        }

        prob.Report("LBFGS: FinIt  %4d: f %-10.4f ||g|| %-10.8f fevals: %d\n", t - 1, f, gnorm, fevals);
        return f;
    }

    template <size_t M, typename TValue>
    TValue RestartingLBFGSMinimize(ProjectableProblem<TValue>& prob, Eigen::Matrix<TValue, Eigen::Dynamic, 1>& x,
                                   size_t maxNumIt   = 1000,
                                   TValue optTol     = TValue(1e-3),
                                   bool verbose      = true,
                                   size_t maxSrchIt  = 10,
                                   TValue gamma      = TValue(1e-6))
    {
        typedef Eigen::Matrix<TValue, Eigen::Dynamic, 1> TVector;
        const unsigned Dim = prob.Dimensions();
        TVector candx(Dim), g(Dim), candg(Dim), d(Dim), s(Dim), y(Dim);
        prob.ProvideStartingPoint(x);
        TValue f     = prob.Eval(x, g), candf = TValue(0.0), suffdec = TValue(0.0);
        TValue gnorm = ProjectedGradientNorm(prob, x, g);
        size_t fevals = 1;
        size_t t      = 1;
        prob.Report("LBFGS: Initially  : f %-10.8f ||g|| %-10.8f\n", f, gnorm);
        CompactInverseHessian<TValue, M> H;

        while(gnorm > optTol && t < maxNumIt)
        {
            // Find descent direction
            if( H.Empty() )//if(t == 1)
            {
                // Initial direction, plain steepest descent
                d = prob.Project(x - g * (1e-3 / gnorm)) - x;
            }
            else
            {
                // Scale the gradient by the L-BFGS approximation to the inverse Hessian
                d = prob.Project(x - (H * g)) - x;
            }

            // Backtracking line-search
            bool   accepted = false;
            TValue lambda   = TValue(1.0);
            TValue gTd      = g.dot(d);
            size_t srchit   = 0;

            do
            {
                candx   = x + lambda * d;
                candf   = prob.Eval(candx, candg);
                suffdec = gamma * lambda * gTd;

                if(srchit > 0 && verbose)
                {
                    prob.Report("LBFGS:   SrchIt %4d: f %-10.8f t %-10.8f\n", srchit, candf, lambda);
                }

                if(candf < f + suffdec)
                {
                    s = candx - x;
                    y = candg - g;
                    x = candx;
                    g = candg;
                    f = candf;
                    accepted = true;
                    srchit = 0;
                }                                               // Allow more line search steps initially
                else if((!H.Empty() && srchit >= maxSrchIt) || (H.Empty() && srchit >= maxSrchIt * 10))
                {
                    accepted = true;
                    H.Clear();
                    srchit++;
                }
                else
                {
                    lambda *= 0.125;
                    srchit++;
                }

                fevals++;
            }
            while(! accepted);

            if(srchit >= maxSrchIt * 10)
            {
                prob.Report("LBFGS: Line search cannot make further progress\n");
                break;
            }

            if(srchit > maxSrchIt)
            {
                prob.Report("LBFGS: Projected direction is bad - resetting approximation to inverse Hessian\n");
                continue;
            }

            // Valid step - update the L-BFGS approximation to the inverse Hessian
            H.Update(y, s);
            gnorm = ProjectedGradientNorm(prob, x, g);

            if(verbose)
            {
                prob.Report("LBFGS: MainIt %4d: f %-10.8f ||g|| %-10.8f\n", t, f, gnorm);
            }

            t++;
        }

        prob.Report("LBFGS: FinIt  %4d: f %-10.8f ||g|| %-10.8f fevals: %d\n", t - 1, f, gnorm, fevals);
        return f;
    }

    // Ternary search algorithm: Simple method for 1d-optimization
    //
    // Can be used to solve one-dimensional linesearch problems efficiently and without
    // the derivative with respect to the parameter. Linear convergence.
    template<typename TValue, typename TFunction>
    TValue TernarySearch(TFunction f, TValue left, TValue right, TValue absolutePrecision, bool verbose = false)
    {
        // left and right are the current bounds; the minimum is between them
        if((right - left) < absolutePrecision)
            return (left + right) / TValue(2);

        const auto leftThird = (TValue(2) * left + right) / TValue(3);
        const auto rightThird = (left + TValue(2) * right) / TValue(3);
        auto flt = f(leftThird);
        auto frt = f(rightThird);

        if(verbose)
        {
            std::cout << "  TernarySearch: f(" << leftThird << ")=" << flt
                      << "  f(" << rightThird << ")=" << frt << std::endl;
        }

        if(flt > frt)
            return TernarySearch(f, leftThird, right, absolutePrecision);
        else
            return TernarySearch(f, left, rightThird, absolutePrecision);
    }

    template<typename TValue, typename TFunction>
    TValue SafeDescentSearch(TFunction f, TValue left, TValue right, bool verbose = false)
    {
        // 1. Assert the minimal stepsize is descent
        auto alpha = left;
        auto f_0 = f(0);
        auto f_alpha = f(alpha);

        if(f_alpha > f_0)
        {
            if(verbose)
            {
                std::cout << "   SafeDescentSearch: minimum step size " << left << " failed to provide descent."
                          << std::endl;
            }

            return std::numeric_limits<TValue>::signaling_NaN();
        }

        double growth_factor = std::sqrt(2.0);
        auto f_prev = f_alpha;
        auto alpha_prev = alpha;

        do
        {
            if(verbose)
            {
                std::cout << "   SafeDescentSearch: alpha=" << alpha << ", f(alpha)=" << f_alpha
                          << std::endl;
            }

            // Safe previous
            alpha_prev = alpha;
            f_prev = f_alpha;
            // Grow
            alpha *= growth_factor;

            if(alpha > right)
                break;

            f_alpha = f(alpha);
        }
        while(f_alpha < f_prev);

        return (alpha_prev);
    }
}

#endif // H_RTF_MINIMIZATION_H
