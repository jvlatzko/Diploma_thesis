/* This file is part of the "Regression Tree Fields" (RTF) source code distribution,
 * obtained from http://research.microsoft.com/downloads.
 * It is provided to you under the terms of the Microsoft Research License Agreement
 * (MSR-LA). Please see License.txt for details.
 *
 *
 * File: Compute.h
 * Implements several core internal routines for estimating the model parameters and
 * setting up the sparse linear system of the inference problem.
 *
 */

#ifndef H_RTF_COMPUTE_H
#define H_RTF_COMPUTE_H

#include <Eigen/StdVector>

#include <random>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <Eigen/Sparse>

#include "Types.h"
#include "Loss.h"
#include "Unary.h"
#include "Classify.h"
#include "Pairwise.h"
#include "Training.h"
#include "Minimization.h"


namespace Compute
{
    // This class represents the parameters stored at a single leaf node of a regression tree.
    // It consists of a VarDim x 1 vector w, and a VarDim x VarDim matrix W. The former is used in the
    // linear term of the Gaussian density function, whereas the latter is used to determine
    // the coefficients of the quadratic term.
    template<typename ValueT, size_t VarDim, size_t BasisDim>
    class Weights
    {
    public:
        typedef ValueT TValue;
    private:
#ifdef _OPENMP
        omp_lock_t                                     lock;
        bool                                           dolock;
#endif
    public:
        static const size_t                             NumCoefficients = VarDim*BasisDim + VarDim*VarDim;

        typedef Eigen::Matrix<TValue, VarDim, VarDim>   TVarVarMatrix;
        typedef Eigen::Matrix<TValue, VarDim, BasisDim> TVarBasisMatrix;

        TVarBasisMatrix         Wl; // Coefficients for the linear term.
        TVarVarMatrix           Wq; // Coefficients for the quadratic term.

        mutable TVarBasisMatrix Gl; // Temporary storage that is used to accumulate the gradient during training. It is
        mutable TVarVarMatrix   Gq; // perhaps not particularly elegant to store this information here, but it makes life much simpler.

#ifdef _OPENMP
        std::vector< TVarBasisMatrix > Gls;
        std::vector< TVarVarMatrix > Gqs;
#endif

    public:

        // Initialize the linear coefficients to zero, and the quadratic coefficients to the negative identity.
        // Note that this guarantees feasibility, as the negative identity is negative definite.
        Weights() : Wl(TVarBasisMatrix::Zero()), Wq(-TVarVarMatrix::Identity()), Gl(TVarBasisMatrix::Zero()), Gq(TVarVarMatrix::Zero())
        {
#ifdef _OPENMP
            dolock = false;
#endif
        }

        Weights(TValue v) : Wl(TVarBasisMatrix::Zero()), Wq(-v*TVarVarMatrix::Identity()), Gl(TVarBasisMatrix::Zero()), Gq(TVarVarMatrix::Zero())
        {
#ifdef _OPENMP
            dolock = false;
#endif
        }

        void Reset(TValue smallestEigenvalue)
        {
            Wl = TVarBasisMatrix::Zero();
            Wq = -smallestEigenvalue*TVarVarMatrix::Identity();
            Gl = TVarBasisMatrix::Zero();
            Gq = TVarVarMatrix::Zero();
        }

        // Load a flat representation of the the coefficients into the corresponding vector and matrix.
        static const TValue* LoadCoefficients(TVarBasisMatrix *Al, TVarVarMatrix* Aq, const TValue *as)
        {
            memcpy(Al->data(), as, VarDim * BasisDim * sizeof(TValue));
            as += VarDim * BasisDim;
            memcpy(Aq->data(), as, VarDim * VarDim * sizeof(TValue));
            as += VarDim * VarDim;
            return as;
        }

        // Store the coefficient vector and matrix in a flat representation.
        static TValue* StoreCoefficients(TValue *as, const TVarBasisMatrix *Al, const TVarVarMatrix *Aq)
        {
            if( as )
            {
                memcpy(as, Al->data(), VarDim * BasisDim * sizeof(TValue));
                as += VarDim * BasisDim;
                memcpy(as, Aq->data(), VarDim * VarDim * sizeof(TValue));
                as += VarDim * VarDim;
            }
            return as;
        }

#ifdef _OPENMP
        void Locked(const std::function<void (void)>& op)
        {
            if(dolock)
                omp_set_lock(&lock);

            op();

            if(dolock)
                omp_unset_lock(&lock);
        }
#else
        void Locked(const std::function<void (void)>& op)
        {
            op();
        }
#endif

        // Checks if the coefficients of this instance are feasible. The condition that must be satisifed
        // is for the coefficient matrix W to be negative definite. We first check if W is symmetric
        // (a necessary condition), and afterwards check if all eigenvalues are negative (a sufficient
        // condition for negative definiteness).
        static const TValue* CheckFeasibility(const TValue* ts, bool &feasible)
        {
            TVarBasisMatrix Tl;
            TVarVarMatrix Tq;
            ts = LoadCoefficients(&Tl, &Tq, ts);
            const auto B = (Tq + Tq.transpose()) / TValue(2.0); // check for symmetry

            if(! B.isApprox(Tq, TValue(1e-10)))
            {
                feasible = false;
                return ts;
            }

            auto eigenvalues = Tq.template selfadjointView<Eigen::Lower>().eigenvalues(); // check for negative eigenvalues

            if((eigenvalues.array() > TValue(-1e-10)).any())
                feasible = false;

            return ts;
        }

        // Projects the coefficients to lie within the convex set of negative-definite symmetric matrices
        // whose smallest eigenvalue is <= -smallestEigenvalue and whose largest eigenvalue is
        // >= -largestEigenvalue. Again, this is involves an eigen decomposition. The projection finds
        // the closest feasible point according to the Frobenius norm and is based on
        //
        // Higham, N. J. Computing a nearest symmetric positive semidefinite matrix. Linear algebra and
        //   its applications, Vol. 103:103--118, 1988.
        static TValue* Project(TValue *as, TValue smallestEigenvalue, TValue largestEigenvalue)
        {
            TVarBasisMatrix Al;
            TVarVarMatrix Aq;
            LoadCoefficients(&Al, &Aq, as);
            auto B = (Aq + Aq.transpose()) / TValue(2.0);
            Eigen::SelfAdjointEigenSolver<TVarVarMatrix> solver(B);
            auto Z        = solver.eigenvectors();
            auto lambda   = solver.eigenvalues();

            for(auto i   = 0; i < lambda.size(); ++i)
            {
                if(lambda[i] > -smallestEigenvalue)
                    lambda[i] = -smallestEigenvalue;
                else if(lambda[i] < -largestEigenvalue)
                    lambda[i] = -largestEigenvalue;
            }

            TVarVarMatrix X = (Z * lambda.asDiagonal() * Z.transpose());
            TValue * ret    = StoreCoefficients(as, &Al, &X);
            return ret;
        }

        static TValue* ProjectDiagForm(TValue *as, TValue smallestEigenvalue, TValue largestEigenvalue)
        {
            TVarBasisMatrix Al;
            TVarVarMatrix Aq;
            LoadCoefficients(&Al, &Aq, as);

            auto& var  = Aq.template topLeftCorner < VarDim/2, VarDim/2 > ();
            auto& varT = Aq.template bottomRightCorner < VarDim/2, VarDim/2 > ();
            auto& cov  = Aq.template topRightCorner < VarDim/2, VarDim/2 > ();
            auto& covT = Aq.template bottomLeftCorner < VarDim/2, VarDim/2 > ();

            var  = Eigen::Matrix<TValue, VarDim/2, 1>(var.diagonal()).asDiagonal();
            varT = Eigen::Matrix<TValue, VarDim/2, 1>(varT.diagonal()).asDiagonal();
            cov  = Eigen::Matrix<TValue, VarDim/2, 1>(cov.diagonal()).asDiagonal();
            covT = Eigen::Matrix<TValue, VarDim/2, 1>(covT.diagonal()).asDiagonal();

            auto B = (Aq + Aq.transpose()) / TValue(2.0);
            Eigen::SelfAdjointEigenSolver<TVarVarMatrix> solver(B);
            auto Z        = solver.eigenvectors();
            auto lambda   = solver.eigenvalues();

            for(auto i   = 0; i < lambda.size(); ++i)
            {
                if(lambda[i] > -smallestEigenvalue)
                    lambda[i] = -smallestEigenvalue;
                else if(lambda[i] < -largestEigenvalue)
                    lambda[i] = -largestEigenvalue;
            }

            TVarVarMatrix X = (Z * lambda.asDiagonal() * Z.transpose());
            TValue * ret    = StoreCoefficients(as, &Al, &X);
            return ret;
        }

        // Store all coefficients to the given flat array.
        TValue* GetWeights(TValue *ws) const
        {
            return StoreCoefficients(ws, &Wl, &Wq);
        }

        // Load all coefficients from the given flat array.
        const TValue* SetWeights(const TValue *ws)
        {
            const auto ret = LoadCoefficients(&Wl, &Wq, ws);
            return ret;
        }

        static size_t NumWeights()
        {
            return NumCoefficients;
        }

        // Store the gradient.
        TValue* GetGradient(TValue *gs) const
        {
#ifdef _OPENMP
            Gl = TVarBasisMatrix::Zero();
            Gq = TVarVarMatrix::Zero();

            for( size_t i = 0; i < Gls.size(); ++i )
                Gl += Gls[i];
            for( size_t i = 0; i < Gqs.size(); ++i )
                Gq += Gqs[i];
#endif
            auto ret = StoreCoefficients(gs, &Gl, &Gq);
            return ret;
        }

        // Load the gradient.
        void ClearGradient()
        {
            Gl = TVarBasisMatrix::Zero();
            Gq = TVarVarMatrix::Zero();

#ifdef _OPENMP
            Gls.resize(omp_get_max_threads());
            for( size_t i = 0; i < Gls.size(); ++i )
                Gls[i] = TVarBasisMatrix::Zero();
            Gqs.resize(omp_get_max_threads());
            for( size_t i = 0; i < Gqs.size(); ++i )
                Gqs[i] = TVarVarMatrix::Zero();
#endif
        }

        TVarVarMatrix& GetGq()
        {
#ifdef _OPENMP
            const auto id = omp_get_thread_num();
            if( Gqs.size() > id )
                return Gqs[id];
            else
                return Gq;
#else
            return Gq;
#endif
        }

        TVarBasisMatrix& GetGl()
        {
#ifdef _OPENMP
            const auto id = omp_get_thread_num();
            if( Gls.size() > id )
                return Gls[id];
            else
                return Gl;
#else
            return Gl;
#endif
        }

#ifdef _OPENMP
        void InitializeLock()
        {
            omp_init_lock(&lock);
            dolock = true;
        }
#endif

#ifdef _OPENMP
        void DestroyLock()
        {
            omp_destroy_lock(&lock);
            dolock = false;
        }
#endif

        // Friends that are used for serialization.
        template<typename TValue1, size_t VarDim1, size_t BasisDim1>
        friend std::ostream& operator<<(std::ostream& out, const Compute::Weights<TValue1, VarDim, BasisDim>& weights);

        template<typename TValue1, size_t VarDim1, size_t BasisDim1>
        friend std::istream& operator>>(std::istream& in, Compute::Weights<TValue1, VarDim1, BasisDim1>& weights);

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    };

    // Abstract base class that defines the common interface of unary and pairwise factor types.
    template<typename TValue>
    struct FactorType
    {
        // Check the feasibility of a particular weights setting, which must be passed in as
        // a flat array. The output parameter 'feasible' is set accordingly.
        virtual const TValue* CheckFeasibility(const TValue *flatWeights, bool& feasible) const = 0;

        // Projects the given weights for this factor type onto the set of symmetric negative definite
        // matrices that fulfill the eigenvalue restrictions imposed by this factor type.
        virtual TValue* Project(TValue *flatWeights) const = 0;

        // Returns the total number of weights stored by this factor type.
        virtual size_t NumWeights() const = 0;

        // Sets all weights of the factor type from a flat array.
        virtual const TValue* SetWeights(const TValue *flatWeights) = 0;

        // Stores all weights of the factor type to a flat array.
        virtual TValue* GetWeights(TValue *flatWeights) const = 0;

        // Stores the accumulated gradient of this factor type in the given flat array, and
        // adds in the gradient of the associated prior. The objective contribution of the
        // prior is returned in the output parameter 'objective'.
        virtual TValue* GetGradientAddPrior(TValue *flatGradient, TValue& objective) const = 0;

        // Clears the temporary storage for the gradient. This must be called prior to accumulating
        // the gradient.
        virtual void ClearGradient() = 0;

        virtual void ResetWeights() = 0;

#ifdef _OPENMP
        virtual void InitializeLocks() = 0;
        virtual void DestroyLocks() = 0;
#endif
        // This is required such that weights can be allocated on the heap while guaranteeing
        // the alignment required for SIMD instructions.
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    };

    // Forward declarations
    template<typename TTraits> class Subgraph;

    template<typename TTraits, bool Parallel>
    void
    for_each_subgraph(const typename TTraits::PreProcessType& prep,
                      const int cx, const int cy,
                      const typename TTraits::UnaryFactorTypeVector& Us,
                      const typename TTraits::PairwiseFactorTypeVector& Ps,
                      const std::function<void (const Subgraph<TTraits>&)>& op);



    // Concrete base class of unary and pairwise factor types that implements some common
    // operations.
    template<typename TFeature, typename TLabel, typename TPrior, typename TBasis>
    class FactorTypeBase : public FactorType<typename TLabel::ValueType>
    {
    public:
        static const size_t                                                   VarDim = TLabel::Size;
        static const size_t                                                   BasisDim = TBasis::Size;
        typedef typename TLabel::ValueType                                    TValue;
        typedef typename Traits_<TFeature, TLabel, TBasis>::ModelTreeRef      TModelTreeRef;
        typedef Weights<TValue, VarDim, BasisDim>                             TWeights;

    protected:
        TModelTreeRef            tree;                      // Reference to the underlying regression tree.
        VecCRef<Vector2D<int>>   offsets;                   // Offsets of the variables of a factor instance of this type, relative to the center variable.
        TValue                   smallestEigenvalue;        // Determines the restriction on the eigenvalues of the matrix parameters
        TValue                   largestEigenvalue;         // of this factor type, see comments of the Project() method.
        TValue                   linearRegularizationC;     // Regularization constant for the prior over the matrix coefficients.
        TValue                   quadraticRegularizationC;  // Regularization constant for the prior over the linear coefficients.
        bool                     fixParameters;

    public:
        FactorTypeBase(const TModelTreeRef& tree_,
                       const VecCRef<Vector2D<int>>& offsets_,
                       TValue smallestEigenvalue_, TValue largestEigenvalue_,
                       TValue linearRegularizationC_, TValue quadraticRegularizationC_)
            : tree(tree_), offsets(offsets_),
              smallestEigenvalue(smallestEigenvalue_), largestEigenvalue(largestEigenvalue_),
              linearRegularizationC(linearRegularizationC_), quadraticRegularizationC(quadraticRegularizationC_),
              fixParameters(false)
        {
        }

        void FixParameters(bool fix)
        {
            fixParameters = fix;
        }

        void ResetWeights()
        {
            for(auto it = tree.begin_leaf(); it != tree.end_leaf(); ++it)
                it->data.Reset(smallestEigenvalue);
        }

        // Return the variable offsets of this factor type.
        VecCRef<Vector2D<int>> Offsets() const
        {
            return offsets;
        }

        // Check feasibility of all weights.
        const TValue* CheckFeasibility(const TValue *flatWeights, bool& feasible) const
        {
            if( fixParameters )
                return flatWeights;

            for(auto it = tree.begin_leaf(); it != tree.end_leaf(); ++it)
                flatWeights = it->data.CheckFeasibility(flatWeights, feasible);

            return flatWeights;
        }

        // This method is one of the hotspots of training with PQN; the SPG routine
        // in the inner loop calls Project() very frequently while minimizing the
        // quadratic approximation subject to the parameter constraints.
        TValue* Project(TValue *flatWeights) const
        {
            if( fixParameters )
                return flatWeights;

            size_t numLeaves = 0;

            for(auto it = tree.begin_leaf(); it != tree.end_leaf(); ++it)
                numLeaves++;

            #pragma omp parallel for
            for(int i = 0; i < (int) numLeaves; ++i)
                TWeights::Project(flatWeights + i * TWeights::NumWeights(), smallestEigenvalue, largestEigenvalue);

            return flatWeights + numLeaves * TWeights::NumWeights();
        }

        // Simply sum the number of coefficients of all weights instances at the leaves.
        size_t NumWeights() const
        {
            if( fixParameters )
                return 0;

            size_t n = 0;

            for(auto it = tree.begin_leaf(); it != tree.end_leaf(); ++it)
                n += TWeights::NumWeights();

            return n;
        }

        // Iterate through all leaves and set the weights from the given flat array.
        const TValue* SetWeights(const TValue *flatWeights)
        {
            if( fixParameters )
                return flatWeights;

            for(auto it = tree.begin_leaf(); it != tree.end_leaf(); ++it)
                flatWeights = it->data.SetWeights(flatWeights);

            return flatWeights;
        }

        // Iterate through all leaves and write out the weights to the flat array.
        TValue* GetWeights(TValue *flatWeights) const
        {
            if( fixParameters )
                return flatWeights;

            for(auto it = tree.begin_leaf(); it != tree.end_leaf(); ++it)
                flatWeights = it->data.GetWeights(flatWeights);

            return flatWeights;
        }

        // Returns a vector of pointers to node weights in breadth-first order.
        VecCRef<TWeights*> WeightsInBreadthFirstOrder() const
        {
            VecRef<TWeights*> weights;

            for(auto it = tree.begin_breadth_first(); it != tree.end_breadth_first(); ++it)
                weights.push_back(&(it->data));

            return weights;
        }

        // Returns the smallest admissible eigenvalue of the Precision matrix of each Weights block.
        TValue SmallestEigenvalue() const
        {
            return smallestEigenvalue;
        }

        // Returns the largest admissible eigenvalue of the Precision matrix of each Weights block.
        TValue LargestEigenvalue() const
        {
            return largestEigenvalue;
        }

        // Iterate through all leaves and collect the previously accumulated gradient
        // and the contribution by the priors, if any.
        TValue* GetGradientAddPrior(TValue *flatGradient, TValue& objective) const
        {
            if( fixParameters )
                return flatGradient;

            for(auto it = tree.begin_leaf(); it != tree.end_leaf(); ++it)
            {
                objective   += TPrior::ComputeObjectiveAddGradient(linearRegularizationC, quadraticRegularizationC, it->data);
                flatGradient = it->data.GetGradient(flatGradient);
            }

            return flatGradient;
        }

        // Iterate through all leaves and clear the temporary storage for the gradient.
        void ClearGradient()
        {
            if( fixParameters )
                return;

            for(auto it = tree.begin_leaf(); it != tree.end_leaf(); ++it)
                it->data.ClearGradient();
        }

        // Add any new leaves contained in the specified regression tree to our model tree, setting the
        // feature attribute of the respective parent and setting the Weights node of the newly allocated
        // children to the weights of the parent.
        // This is can be used to update the model tree if the corresponding regression tree was grown
        // by one level subsequent to instantiation of the factor type.
        template<typename TRegressionTreeRef>
        void MergeTree(const TRegressionTreeRef& regressionTree)
        {
            auto j = tree.begin_breadth_first();

            for(auto i = regressionTree.begin_breadth_first(); i != regressionTree.end_breadth_first(); ++i, ++j)
            {
                // We've reached the leaf level of the model tree; add any nodes that are present in
                // the regression tree but not in the model tree, and copy over the branching feature
                // of the parent node.
                if(j.number_of_children() != i.number_of_children())
                {
                    assert(j.number_of_children() == 0);
                    j->feature = i->feature;

                    for(size_t k = 0; k < i.number_of_children(); k++)
                    {
                        auto child  = tree.append_child(j);
                        child->data = j->data;
                    }
                }
            }
        }

        TModelTreeRef
        GetTree()
        {
            return tree;
        }

        Training::LabelVector<TValue, TWeights::NumCoefficients>
        MakeLabel(const typename TWeights::TVarBasisMatrix& Gl, const typename TWeights::TVarVarMatrix& Gq) const
        {
            Training::LabelVector<TValue, TWeights::NumCoefficients> label;
            TWeights::StoreCoefficients(&label[0], &Gl, &Gq);
            return label;
        }

#ifdef _OPENMP
        void InitializeLocks()
        {
            for(auto it = tree.begin_leaf(); it != tree.end_leaf(); ++it)
                it->data.InitializeLock();
        }

        void DestroyLocks()
        {
            for(auto it = tree.begin_leaf(); it != tree.end_leaf(); ++it)
                it->data.DestroyLock();
        }
#endif

        template<typename TFeature1, typename TLabel1, typename TPrior1, typename TBasis1>
        friend std::ostream& operator<<(std::ostream& out, const FactorTypeBase<TFeature1, TLabel1, TPrior1, TBasis1>& type);

        template<typename TFeature1, typename TLabel1, typename TPrior1, typename TBasis1>
        friend std::istream& operator>>(std::istream& in, FactorTypeBase<TFeature1, TLabel1, TPrior1, TBasis1>& type);
    };

    // Abstract base class for a factor that is conditioned on all of its variables,
    // except for one. This functionality is required by Pseudolikelihood training
    // as well as Gibbs sampling.
    template <typename TValue, size_t CondDim>
    class ConditionedFactor
    {
    public:
        typedef Eigen::Matrix<TValue, CondDim, 1>         TCondVector;
        typedef Eigen::Matrix<TValue, CondDim, CondDim>   TCondMatrix;

        // Given mean parameters mu and Sigma, as well as the overall number of subgraphs,
        // add in this factor's contribution to the gradient with respect to the model parameters.
        virtual void AccumulateGradient(const TCondVector& mu, const TCondMatrix& Sigma, size_t numSubgraphs) const = 0;

        // Return the coefficients of this factor that occur in the linear term of the Gaussian density
        // of the conditioned subgraph.
        virtual TCondVector LinearCoefficients() const = 0;

        // Return the coefficients of this factor that occur in the quadratic term of the Gaussian density
        // of the conditioned subgraph.
        virtual TCondMatrix QuadraticCoefficients() const = 0;
    };

    // Represents the Gaussian distribution of a single vertex conditioned on its neighborhood.
    // We use the notation/terminology of Wainwright (2008):
    //   y_j     ... label of vertex j
    //   mu_j    ... first-order mean parameter E[y_j]
    //   Sigma_j ... second-order mean parameter E[y_j y_j^T]
    //   theta_j ... linear part of the canonical parameters
    //   Theta_j ... quaratic part of the canonical parameters
    // Note that in this notation, the covariance matrix of the conditioned Gaussian is given by
    //   C_j := Sigma_j - mu_j mu_j^T
    template<typename TTraits>
    class ConditionedSubgraph
    {
    public:
        static const size_t                                   VarDim = TTraits::UnaryGroundLabel::Size;

        typedef typename TTraits::UnaryGroundLabel            TLabel;
        typedef typename TTraits::ValueType                   TValue;
        typedef typename TTraits::PreProcessType              TPrep;

        typedef typename TTraits::UnaryFactorType             TUnaryFactorType;
        typedef typename TTraits::PairwiseFactorType          TPairwiseFactorType;
        typedef ConditionedFactor<TValue, VarDim>             TConditionedFactor;

        typedef Eigen::Matrix<TValue, VarDim, 1>              TVector;
        typedef Eigen::Matrix<TValue, VarDim, VarDim>         TMatrix;

    protected:
        Vector2D<int>     j;     // the center variable
        TPrep             prep;  // the preprocessed input image
        ImageRefC<TLabel> y;     // the groundtruth labels

        const std::vector<TUnaryFactorType>    &Us; // reference to the vector of unary factor types
        const std::vector<TPairwiseFactorType> &Ps; // reference to the vector of pairwise factor types

        // Convenience method that invokes a given functor for each conditioned factor connected to the
        // center variable of this subgraph.
        virtual void for_each_conditioned_factor(const std::function<void (const TConditionedFactor&)>& op) const
        {
            std::for_each(Us.begin(), Us.end(), [&](const TUnaryFactorType & U)
            {
                U.ForEachConditionedInstance(j, prep, y, op);
            });
            std::for_each(Ps.begin(), Ps.end(), [&](const TPairwiseFactorType & P)
            {
                P.ForEachConditionedInstance(j, prep, y, op);
            });
        }

    public:

        // Contructs a subgraph from the given parameters - see the descriptions above for details.
        ConditionedSubgraph(const Vector2D<int>& j_, const TPrep& prep_, const ImageRefC<TLabel>& y_,
                            const std::vector<TUnaryFactorType>& Us_, const std::vector<TPairwiseFactorType>& Ps_)
            : j(j_), prep(prep_), y(y_), Us(Us_), Ps(Ps_)
        {
        }

        virtual ~ConditionedSubgraph() {}

        // Returns the x coordinate of the center variable in the overall grid.
        int PosX() const
        {
            return j.x;
        }

        // Returns the y coordinate of the center variable in the overall grid.
        int PosY() const
        {
            return j.y;
        }

        const TPrep& Prep() const
        {
            return prep;
        }

        // To draw a random sample from a multivariate normal distribution, we need three ingredients:
        // 1) The mean \mu of the multivariate distribution
        // 2) A cholesky decomposition C := AA^T of the covariance of the multivariate distribution
        // 3) A vector z of VarDim independent standard normal variates
        //
        // The sample is then obtained as \mu + Az .
        // See http://en.wikipedia.org/wiki/Multivariate_normal_distribution
        template<typename TEngine>
        TLabel DrawSample(TEngine& engine, std::normal_distribution<TValue>& normal) const
        {
            // Obtain the parameters of the multivariate Gaussian according to which our conditioned subgraph is distributed
            TVector mu_j;
            TMatrix Sigma_j;
            ComputeMeanParameters(mu_j, Sigma_j);
            const TMatrix C_j = Sigma_j - mu_j * mu_j.transpose();
            // Obtain VarDim independent standard normal variates
            typename std::normal_distribution<TValue>::param_type param(0, 1);
            TVector z;

            for(int i = 0; i < z.rows(); ++i)
                z[i] = normal(engine, param);

            // Obtain the multivariate sample according to the text book formula
            return Utility::VectorToLabel<TLabel>(mu_j + C_j.llt().matrixL() * z);
        }

        // Compute the mean parameters of the conditioned Gaussian, mu_i := E[y_i] and Sigma_i := E[y_i y_i^T]
        // Note that the covariance matrix is given by C := Sigma_i - mu_i mu_i^T.
        // Do not mistake Sigma_i for the covariance of the Gaussian!
        void ComputeMeanParameters(TVector &mu_j, TMatrix &Sigma_j) const
        {
            TVector theta_j;
            TMatrix Theta_j;
            TValue logDetC_j; // temporary storage of the canonical parameters
            return ComputeMeanParameters(theta_j, Theta_j, mu_j, Sigma_j, logDetC_j);
        }

        void ComputeMeanParameters(TVector &theta_j, TMatrix &Theta_j,
                                   TVector &mu_j, TMatrix &Sigma_j, TValue& logDetC_j) const
        {
            theta_j = TVector::Zero(); // Initialize the canonical parameters to zero for latter accumulation
            Theta_j = TMatrix::Zero();
            TVector* ptheta_i = &theta_j; // Somehow the compiler doesn't like these as references in the lambda abstraction,
            TMatrix* pTheta_i = &Theta_j; // so we're passing them as pointers
            // Accumulate the linear and quadratic weights to initialize the canonical parameters
            for_each_conditioned_factor([ = ](const TConditionedFactor & factor)
            {
                *ptheta_i += factor.LinearCoefficients();
                *pTheta_i += factor.QuadraticCoefficients();
            });
            // Compute the mean parameters
#if 0
            auto Theta_j_Inv = Theta_j.inverse();
            mu_j            = - Theta_j_Inv * theta_j;
            Sigma_j         = - Theta_j_Inv + mu_j * mu_j.transpose();
            logDetC_j       = - log(Theta_j.determinant());
#else
            const auto Qr          = Theta_j.fullPivHouseholderQr();       // This variant should be more stable numerically
            const auto Theta_j_Inv = Qr.inverse();
            mu_j                   = - Theta_j_Inv * theta_j;
            Sigma_j                = - Theta_j_Inv + mu_j * mu_j.transpose();
            logDetC_j              = - Qr.logAbsDeterminant();
#endif
        }

        // Computes the contribution of the subgraph to the pseudolikelihood object and accumulates
        // the gradient contribution in the temporary storage provides by the weights nodes.
        // The objective is scaled by 1/numSubgraphs to allow for meaningful comparisons among
        // datasets of different sizes.
        TValue ComputeObjectiveAccumulateGradient(size_t numSubgraphs) const
        {
            const TValue two     = static_cast<TValue>(2);
            const TValue oneHalf = 1 / two;
            // Canonical parameters and mean parameters of the conditioned Gaussian
            TVector theta_j, mu_j;
            TMatrix Theta_j, Sigma_j;
            TValue  logDetC_j;
            // The observed labeling of vertex i
            auto y_j = Utility::LabelToVector(y(j.x, j.y));
            // Accumulate the canonical parameters and compute the mean parameters
            ComputeMeanParameters(theta_j, Theta_j, mu_j, Sigma_j, logDetC_j);
            // Pass the mean parameters on to the conditioned factors so as to accumulate the gradient
            for_each_conditioned_factor([&](const TConditionedFactor & factor)
            {
                factor.AccumulateGradient(mu_j, Sigma_j, numSubgraphs);
            });
            // Compute the energy term
            auto E_j         = theta_j.dot(y_j)  + oneHalf * y_j.dot(Theta_j * y_j);
            // Compute the log partition function according to its variational representation
            auto A_j         = theta_j.dot(mu_j) + oneHalf * (Theta_j.array() * Sigma_j.array()).sum()
                               + oneHalf * logDetC_j //log( (Sigma_j - mu_j * mu_j.transpose()).determinant() )
                               + oneHalf * VarDim * log(two * static_cast<TValue>(Constants::pi() * Constants::e()));
            // Return the negative log-likelihood
            return (A_j - E_j) / numSubgraphs;
        }

        // For the given factor type, determines the gradient contribution(s) of this subgraph given the current
        // mean parameters mu_j and Sigma_j, and invokes op on the gradient contribution.
        // For a unary type, there is exactly one gradient contribution, so op is guaranteed to be invoked only
        // once. For a pairwise type, there can be up to two gradient contributions.
        template<typename TFactorType, typename TOp>
        void
        ForEachGradientContributionOfType(const TFactorType& T, const TVector &mu_j, const TMatrix &Sigma_j,
                                          size_t numSubgraphs, const TOp& op) const
        {
            typename TFactorType::TWeights w;
            T.ForEachConditionedInstance(j, prep, y, &w, [&](const TConditionedFactor & factor)
            {
                w = typename TFactorType::TWeights();
                factor.AccumulateGradient(mu_j, Sigma_j, numSubgraphs);
                op(w.Gl, w.Gq);
            });
        }

        template<typename TFactorType>
        size_t
        NumConnected(const TFactorType& T) const
        {
            return T.NumConnected(j, y.Width(), y.Height());
        }
    };

    // Subclass of ConditionedSubgraph that works using pre-computed 'weights images' that
    // store a pointer to the applicable weights for each factor instance. This avoids
    // the need to sort the data points in the leaves of the regression tree repeatedly if
    // we iterate over the same subgraph multiple times, and is useful to implement
    // Gibbs sampling, among other applications.
    template<typename TTraits>
    class PrecomputedConditionedSubgraph : public ConditionedSubgraph<TTraits>
    {
    private:
        typedef ConditionedSubgraph<TTraits> Base;
        const typename TTraits::UnaryWeightsImageVector& Uws;
        const typename TTraits::PairwiseWeightsImageVector& Pws;
        const typename TTraits::UnaryBasisImageVector& Ubs;
        const typename TTraits::PairwiseBasisImageVector& Pbs;

    protected:
        virtual void for_each_conditioned_factor(const std::function<void (const typename Base::TConditionedFactor&)>& op) const
        {
            for(size_t u = 0; u < Base::Us.size(); ++u)
                Base::Us[u].ForEachConditionedInstance(Base::j, ConditionedSubgraph<TTraits>::Prep(), Uws[u], Ubs[u], Base::y, op);

            for(size_t p = 0; p < Base::Ps.size(); ++p)
                Base::Ps[p].ForEachConditionedInstance(Base::j, ConditionedSubgraph<TTraits>::Prep(), Pws[p], Pbs[p], Base::y, op);
        }

    public:
        PrecomputedConditionedSubgraph(const Vector2D<int>& j_, const typename TTraits::PreProcessType& prep_, const ImageRefC<typename Base::TLabel>& y_,
                                       const typename TTraits::UnaryFactorTypeVector& Us_, const typename TTraits::PairwiseFactorTypeVector& Ps_,
                                       const typename TTraits::UnaryWeightsImageVector& Uws_, const typename TTraits::PairwiseWeightsImageVector& Pws_,
                                       const typename TTraits::UnaryBasisImageVector& Ubs_, const typename TTraits::PairwiseBasisImageVector& Pbs_)
            : Base::ConditionedSubgraph(j_, prep_, y_, Us_, Ps_), Uws(Uws_), Pws(Pws_), Ubs(Ubs_), Pbs(Pbs_)
        {
        }
    };

    // Back-propagation style training, where the gradient is traced back over a finite number
    // of iterations of an inference algorithm.
    // We implement here the classic (blocked) Gaussi-Jacobi iterations, which tend to converge
    // very rapidly if the system matrix is diagonally dominant.

    // Forward inference pass which keeps a tape of the relevant statistics for the gradient.
    template<typename TTraits>
    void
    ForwardGaussJacobi(typename Classify::LinearSystem<TTraits, Loss::NoErrorTerm<TTraits>>::Type& system, size_t N,
                       std::vector<Eigen::Matrix<typename TTraits::ValueType, Eigen::Dynamic, 1>> &y)
    {
        const size_t BlockDim = TTraits::UnaryGroundLabel::Size;
        typedef Eigen::Matrix<typename TTraits::ValueType, Eigen::Dynamic, 1> TVector;
        typedef Eigen::Matrix<typename TTraits::ValueType, Eigen::Dynamic, TTraits::UnaryGroundLabel::Size> TBlockDiagonal;
        const auto Dim = system.Dimensions();

        TVector b(Dim), Ay(Dim);
        TBlockDiagonal D(Dim, BlockDim);
        system.ProvideRightHandSide(b);
        system.ProvideInverseBlockDiagonal(D);

        y.resize(N+1);
        y[0] = TVector::Zero(Dim);

        for( size_t k = 0; k < N; ++ k )
        {
            system.MultiplySystemMatrixBy(Ay, y[k]);
            y[k+1] = y[k] - Minimization::ScaleByBlockDiagonal<typename TTraits::ValueType, TTraits::UnaryGroundLabel::Size>(D, Ay - b);
        }
        system.MultiplySystemMatrixBy(Ay, y[N]);
        return;
    }

    // Forward inference pass which does not keep a tape (used at test time for better efficiency)
    template<typename TTraits>
    Eigen::Matrix<typename TTraits::ValueType, Eigen::Dynamic, 1>
    ForwardGaussJacobi(typename Classify::LinearSystem<TTraits, Loss::NoErrorTerm<TTraits>>::Type& system, size_t N)
    {
        const size_t BlockDim = TTraits::UnaryGroundLabel::Size;
        typedef Eigen::Matrix<typename TTraits::ValueType, Eigen::Dynamic, TTraits::UnaryGroundLabel::Size> TBlockDiagonal;
        typedef Eigen::Matrix<typename TTraits::ValueType, Eigen::Dynamic, 1> TVector;
        const auto Dim = system.Dimensions();

        TVector y = TVector::Zero(Dim);
        TVector b(Dim), Ay(Dim);
        TBlockDiagonal D(Dim, BlockDim);
        system.ProvideRightHandSide(b);
        system.ProvideInverseBlockDiagonal(D);

        for( size_t k = 0; k < N; ++ k )
        {
            system.MultiplySystemMatrixBy(Ay, y);
            y = y - Minimization::ScaleByBlockDiagonal<typename TTraits::ValueType, TTraits::UnaryGroundLabel::Size>(D, Ay - b);;
        }
        return y;
    }

    // Backward pass which recovers the gradient from the tape recorded by the forward pass.
    template<typename TTraits>
    void
    BackwardGaussJacobi(typename Classify::LinearSystem<TTraits, Loss::NoErrorTerm<TTraits>>::Type& system, int N,
                        const Eigen::Matrix<typename TTraits::ValueType, Eigen::Dynamic, 1>& g,
                        std::vector<Eigen::Matrix<typename TTraits::ValueType, Eigen::Dynamic, 1>> &yBack)
    {
        const size_t BlockDim = TTraits::UnaryGroundLabel::Size;
        typedef Eigen::Matrix<typename TTraits::ValueType, Eigen::Dynamic, TTraits::UnaryGroundLabel::Size> TBlockDiagonal;
        typedef Eigen::Matrix<typename TTraits::ValueType, Eigen::Dynamic, 1> TVector;
        const auto Dim = system.Dimensions();
        TVector Ay(Dim);
        TBlockDiagonal D(Dim, BlockDim);
        system.ProvideInverseBlockDiagonal(D);

        yBack.resize(N+1);
        yBack[N] = g;

        for( int k = N - 1; k >= 0; --k )
        {
            system.MultiplySystemMatrixBy(Ay, yBack[k+1]);
            yBack[k] = yBack[k+1] - Minimization::ScaleByBlockDiagonal<typename TTraits::ValueType, TTraits::UnaryGroundLabel::Size>(D, Ay);
        }

        // Pre-scale yBack by D^{-1}, as required for subsequent computation of the gradient wrt the model parameters
        for( int k = 0; k <= N; ++k )
            yBack[k] = Minimization::ScaleByBlockDiagonal<typename TTraits::ValueType, TTraits::UnaryGroundLabel::Size>(D, yBack[k]);
    }

    // Forward pass of backprop "heavy ball" algorithm by Domke (AISTATS 2012);
    // not used right now because Gauss-Jacobi is more efficient.
    template<typename TTraits>
    void
    ForwardHeavyBall(typename Classify::LinearSystem<TTraits, Loss::NoErrorTerm<TTraits>>::Type& system, size_t N,
                     std::vector<Eigen::Matrix<typename TTraits::ValueType, Eigen::Dynamic, 1>> &y)
    {
        typedef Eigen::Matrix<typename TTraits::ValueType, Eigen::Dynamic, 1> TVector;
        const auto Dim = system.Dimensions();

        TVector p = TVector::Zero(Dim);
        TVector b(Dim), Ay(Dim);
        system.ProvideRightHandSide(b);

        y.resize(N+1);
        y[0] = TVector::Zero(Dim);

        for( size_t k = 0; k < N; ++ k )
        {
            system.MultiplySystemMatrixBy(Ay, y[k]);
            p = - (Ay - b) + 0.5 * p;
            y[k+1] = y[k] + p;
        }
        return;
    }

    // Backward pass of backprop "heavy ball".
    template<typename TTraits>
    void
    BackwardHeavyBall(typename Classify::LinearSystem<TTraits, Loss::NoErrorTerm<TTraits>>::Type& system, size_t N,
                      const Eigen::Matrix<typename TTraits::ValueType, Eigen::Dynamic, 1>& g,
                      std::vector<Eigen::Matrix<typename TTraits::ValueType, Eigen::Dynamic, 1>> &pBack)
    {
        typedef Eigen::Matrix<typename TTraits::ValueType, Eigen::Dynamic, 1> TVector;
        const auto Dim = system.Dimensions();
        TVector Ap(Dim);

        TVector yBack = g;
        pBack.resize(N+1);
        pBack[N] = TVector::Zero(Dim);

        for( int k = N - 1; k >= 0; --k )
        {
            std::cerr << "HB: BackwardIt " << k << std::endl;
            pBack[k+1] = pBack[k+1] + yBack;
            system.MultiplySystemMatrixBy(Ap, pBack[k+1]);
            yBack = yBack - Ap;

            pBack[k] = 0.5 * pBack[k+1];
        }
    }

    // LossTraits: Given a loss function, implement functionality such as gradient computation for
    // discriminative, loss-based training of RTFs. The class abstracts away the differences
    // in treatment of difference loss functions.

    // The below specialization is for direct loss minimization.
    template<typename TTraits, typename TLossTag>
    struct LossTraits
    {
        static const size_t VarDim = TTraits::UnaryGroundLabel::Size;

        typedef Loss::Loss<TTraits, TLossTag>                   TLoss;
        typedef typename TTraits::ValueType                     TValue;
        typedef Eigen::Matrix<TValue, Eigen::Dynamic, 1>        TSolution;
        typedef typename TTraits::UnaryFactorType::Factor       TUnaryFactor;
        typedef typename TTraits::PairwiseFactorType::Factor    TPairwiseFactor;
        typedef typename TTraits::UnaryFactorType::TWeights     TUnaryWeights;
        typedef typename TTraits::PairwiseFactorType::TWeights  TPairwiseWeights;

        typedef typename Classify::LinearSystem<TTraits, Loss::NoErrorTerm<TTraits>>::Type TLinearSystem;


        static std::shared_ptr<TLinearSystem>
        ComputeMeanParametersUnaryOnly(const typename TTraits::PreProcessType& x,
                                       const ImageRefC<typename TTraits::UnaryGroundLabel>& y,
                                       const std::vector<typename TTraits::UnaryFactorType>    &Us,
                                       const std::vector<typename TTraits::PairwiseFactorType> &Ps,
                                       TSolution& muPrediction,
                                       TSolution& muLossGradient)
        {
            ImageRef<typename TTraits::UnaryGroundLabel> mu(y.Width(), y.Height());
            Compute::for_each_subgraph<TTraits, true>(x, y.Width(), y.Height(), Us, Ps,
                    [&](const Compute::Subgraph<TTraits>& G)
            {
                mu(G.PosX(), G.PosY()) = G.Solve();
            });
            auto dloss = TLoss::Gradient(y, mu);
            SystemVectorRef<TValue, VarDim> muPredictionRef(y.Width(), y.Height(), muPrediction);
            SystemVectorRef<TValue, VarDim> muLossGradientRef(y.Width(), y.Height(), muLossGradient);
            Compute::for_each_subgraph<TTraits, true>(x, y.Width(), y.Height(), Us, Ps,
                    [&](const Compute::Subgraph<TTraits>& G)
            {
                muLossGradientRef(G.PosX(), G.PosY()) = Utility::LabelToVector(G.Solve(dloss(G.PosX(), G.PosY())));
                muPredictionRef(G.PosX(), G.PosY())   = Utility::LabelToVector(mu(G.PosX(), G.PosY()));
            });
            return nullptr;
        }

        static std::shared_ptr<TLinearSystem>
        ComputeMeanParameters(const typename TTraits::PreProcessType& x,
                              const ImageRefC<typename TTraits::UnaryGroundLabel>& y,
                              const std::vector<typename TTraits::UnaryFactorType>    &Us,
                              const std::vector<typename TTraits::PairwiseFactorType> &Ps,
                              const typename TTraits::LinearOperatorVector& Ls,
                              size_t maxNumIt, TValue residualTol,
                              std::vector<TSolution>& muPrediction,
                              std::vector<TSolution>& muLossGradient)
        {
            if( Ps.size() == 0 && Ls.size() == 0 )
            {
                muPrediction.resize(1);
                muLossGradient.resize(1);
                return ComputeMeanParametersUnaryOnly(x, y, Us, Ps, muPrediction.back(), muLossGradient.back());
            }

            // Backprop-style training: Consider the inference algorithm as no more than
            // as some iterative scheme that keeps improving the solution. We stop after a
            // finite number of steps and trace back the gradient.
            if( residualTol < 0 )
            {
                std::shared_ptr<TLinearSystem> system( new TLinearSystem(x, y.Width(), y.Height(), Us, Ps, Ls) );

                ForwardGaussJacobi<TTraits>(*system, maxNumIt, muPrediction);
                const auto g = Utility::SolutionFromLabeling<TTraits>(TLoss::Gradient(y, Utility::LabelingFromSolution<TTraits>(y.Width(), y.Height(), muPrediction.back())));
                BackwardGaussJacobi<TTraits>(*system, maxNumIt, g, muLossGradient);

                return system;
            }
            // Otherwise, we rely on the algebraic result for differentiation of the inverse of a matrix;
            // this requires solving the system to optimality.
            muPrediction.resize(1);
            muLossGradient.resize(1);

            // Determine the mean under our model distribution (the prediction)
            std::shared_ptr<TLinearSystem> system( new TLinearSystem(x, y.Width(), y.Height(), Us, Ps, Ls) );

#ifdef USE_GPU
            auto solver = system->GetGPUSolver();
            muPrediction[0] = system->SolveOnGPU(solver, maxNumIt, residualTol);
            system->SetRightHandSide(TLoss::Gradient(y, Utility::LabelingFromSolution<TTraits>(y.Width(), y.Height(), muPrediction[0])));
            muLossGradient[0] = system->SolveOnGPU(solver, maxNumIt, residualTol);
#else
            muPrediction[0] = Minimization::CGSolve<TValue>(*system, (unsigned) maxNumIt, residualTol, false);
            system->SetRightHandSide(TLoss::Gradient(y, Utility::LabelingFromSolution<TTraits>(y.Width(), y.Height(), muPrediction[0])));
            muLossGradient[0] = Minimization::CGSolve<TValue>(*system, (unsigned) maxNumIt, residualTol, false);
#endif

            // Return the system for potential latter use
            return system;
        }

        static TValue
        ComputeObjective(const ImageRefC<typename TTraits::UnaryGroundLabel>& y,
                         std::shared_ptr<TLinearSystem> system,
                         const std::vector<TSolution>& muPrediction,
                         const std::vector<TSolution>& muLossGradient,
                         TValue normC)
        {
            const auto prediction = Utility::LabelingFromSolution<TTraits>(y.Width(), y.Height(), muPrediction.back());

            return TLoss::Objective(y, prediction) / normC;
        }

        static void
        AccumulateFactorGradient(const SystemVectorCRef<TValue, VarDim>& yRef,
                                 const std::vector<TSolution>& muPrediction,
                                 const std::vector<TSolution>& muLossGradient,
                                 TValue normC,
                                 TUnaryFactor &factor)
        {
            if( muPrediction.size() == 1 )
                factor.AccumulateGradient(SystemVectorCRef<TValue, VarDim>(yRef.Width(), yRef.Height(), muPrediction[0]),
                                          SystemVectorCRef<TValue, VarDim>(yRef.Width(), yRef.Height(), muLossGradient[0]), normC);
            else
                factor.AccumulateGradient(yRef, muPrediction, muLossGradient, normC);
        }

        static void
        AccumulateFactorGradient(const SystemVectorCRef<TValue, VarDim>& yRef,
                                 const std::vector<TSolution>& muPrediction,
                                 const std::vector<TSolution>& muLossGradient,
                                 TValue normC,
                                 TPairwiseFactor &factor)
        {
            if( muPrediction.size() == 1 )
                factor.AccumulateGradient(SystemVectorCRef<TValue, VarDim>(yRef.Width(), yRef.Height(), muPrediction[0]),
                                          SystemVectorCRef<TValue, VarDim>(yRef.Width(), yRef.Height(), muLossGradient[0]), normC);
            else
                factor.AccumulateGradient(yRef, muPrediction, muLossGradient, normC);
        }

        static void
        AccumulateFactorGradient(const SystemVectorCRef<TValue, VarDim>& yRef,
                                 const std::vector<TSolution>& muPrediction,
                                 const std::vector<TSolution>& muLossGradient,
                                 TValue normC,
                                 TUnaryFactor &factor, TUnaryWeights& w)
        {
            if( muPrediction.size() == 1 )
                factor.AccumulateGradient(SystemVectorCRef<TValue, VarDim>(yRef.Width(), yRef.Height(), muPrediction[0]),
                                          SystemVectorCRef<TValue, VarDim>(yRef.Width(), yRef.Height(), muLossGradient[0]), normC, &w);
            else
                factor.AccumulateGradient(yRef, muPrediction, muLossGradient, normC, &w);
        }

        static void
        AccumulateFactorGradient(const SystemVectorCRef<TValue, VarDim>& yRef,
                                 const std::vector<TSolution>& muPrediction,
                                 const std::vector<TSolution>& muLossGradient,
                                 TValue normC,
                                 TPairwiseFactor &factor, TPairwiseWeights& w)
        {
            if( muPrediction.size() == 1 )
                factor.AccumulateGradient(SystemVectorCRef<TValue, VarDim>(yRef.Width(), yRef.Height(), muPrediction[0]),
                                          SystemVectorCRef<TValue, VarDim>(yRef.Width(), yRef.Height(), muLossGradient[0]), normC, &w);
            else
                factor.AccumulateGradient(yRef, muPrediction, muLossGradient, normC, &w);
        }

        static void
        AccumulateOperatorGradient(const typename TTraits::PreProcessType& x,
                                   const SystemVectorCRef<TValue, VarDim>& yRef,
                                   const std::vector<TSolution>& muPrediction,
                                   const std::vector<TSolution>& muLossGradient,
                                   TValue normC,
                                   const typename TTraits::LinearOperatorRef& op)
        {
            if( muPrediction.size() == 1 )
                op.AccumulateGradient(x, SystemVectorCRef<TValue, VarDim>(yRef.Width(), yRef.Height(), muPrediction[0]),
                                      SystemVectorCRef<TValue, VarDim>(yRef.Width(), yRef.Height(), muLossGradient[0]), normC);
            else
                op.AccumulateGradient(x, muPrediction, muLossGradient, normC);
        }
    };

    // Specialization for continuous structured perceptron loss, an energy-based training approach.
    template<typename TTraits>
    struct LossTraits<TTraits, Loss::ContinuousPerceptron>
    {
        static const size_t VarDim = TTraits::UnaryGroundLabel::Size;

        typedef typename TTraits::ValueType                     TValue;
        typedef Eigen::Matrix<TValue, Eigen::Dynamic, 1>        TSolution;
        typedef typename TTraits::UnaryFactorType::Factor       TUnaryFactor;
        typedef typename TTraits::PairwiseFactorType::Factor    TPairwiseFactor;
        typedef typename TTraits::UnaryFactorType::TWeights     TUnaryWeights;
        typedef typename TTraits::PairwiseFactorType::TWeights  TPairwiseWeights;
        typedef typename Classify::LinearSystem<TTraits, Loss::ErrorTerm<TTraits, Loss::ContinuousPerceptron>>::Type TLinearSystem;

        static std::shared_ptr<TLinearSystem>
        ComputeMeanParameters(const typename TTraits::PreProcessType& x,
                              const ImageRefC<typename TTraits::UnaryGroundLabel>& y,
                              const std::vector<typename TTraits::UnaryFactorType>    &Us,
                              const std::vector<typename TTraits::PairwiseFactorType> &Ps,
                              const typename TTraits::LinearOperatorVector& Ls,
                              size_t maxNumIt, TValue residualTol,
                              std::vector<TSolution>& yhat,
                              std::vector<TSolution>& ystar)
        {
            yhat.resize(1);
            ystar.resize(1);

            // Determine the mean under our model distribution (the prediction)
            std::shared_ptr<TLinearSystem> system( new TLinearSystem(x, y.Width(), y.Height(), Us, Ps, Ls, y) );
            yhat[0]  = Minimization::CGSolve<TValue>(*system, (unsigned) maxNumIt, residualTol, false);
            ystar[0] = Utility::SolutionFromLabeling<TTraits>(y);

            // Return the system for potential latter use
            return system;
        }

        static TValue
        ComputeObjective(const ImageRefC<typename TTraits::UnaryGroundLabel>& y,
                         std::shared_ptr<TLinearSystem> system,
                         const std::vector<TSolution>& yhat,
                         const std::vector<TSolution>& ystar,
                         TValue normC)
        {
            Classify::Detail::UnconstrainedQuadratic<TTraits, Loss::ErrorTerm<TTraits, Loss::ContinuousPerceptron>> quadratic(*system);
            TSolution g;
            auto predicted = quadratic.Eval(yhat[0], g);
            auto observed  = quadratic.Eval(ystar[0], g);

            return (observed - predicted)/normC;
        }

        static void
        AccumulateFactorGradient(const SystemVectorCRef<TValue, VarDim>& yRef,
                                 const std::vector<TSolution>& yhat,
                                 const std::vector<TSolution>& ystar,
                                 TValue normC,
                                 TUnaryFactor &factor)
        {
            factor.AccumulateEnergyBasedGradient(SystemVectorCRef<TValue, VarDim>(yRef.Width(), yRef.Height(), yhat[0]),
                                                 SystemVectorCRef<TValue, VarDim>(yRef.Width(), yRef.Height(), ystar[0]), normC);
        }

        static void
        AccumulateFactorGradient(const SystemVectorCRef<TValue, VarDim>& yRef,
                                 const std::vector<TSolution>& yhat,
                                 const std::vector<TSolution>& ystar,
                                 TValue normC,
                                 TPairwiseFactor &factor)
        {
            factor.AccumulateEnergyBasedGradient(SystemVectorCRef<TValue, VarDim>(yRef.Width(), yRef.Height(), yhat[0]),
                                                 SystemVectorCRef<TValue, VarDim>(yRef.Width(), yRef.Height(), ystar[0]), normC);
        }

        static void
        AccumulateFactorGradient(const SystemVectorCRef<TValue, VarDim>& yRef,
                                 const std::vector<TSolution>& yhat,
                                 const std::vector<TSolution>& ystar,
                                 TValue normC,
                                 TUnaryFactor &factor, TUnaryWeights& w)
        {
            factor.AccumulateEnergyBasedGradient(SystemVectorCRef<TValue, VarDim>(yRef.Width(), yRef.Height(), yhat[0]),
                                                 SystemVectorCRef<TValue, VarDim>(yRef.Width(), yRef.Height(), ystar[0]), normC, &w);
        }

        static void
        AccumulateFactorGradient(const SystemVectorCRef<TValue, VarDim>& yRef,
                                 const std::vector<TSolution>& yhat,
                                 const std::vector<TSolution>& ystar,
                                 TValue normC,
                                 TPairwiseFactor &factor, TPairwiseWeights& w)
        {
            factor.AccumulateEnergyBasedGradient(SystemVectorCRef<TValue, VarDim>(yRef.Width(), yRef.Height(), yhat[0]),
                                                 SystemVectorCRef<TValue, VarDim>(yRef.Width(), yRef.Height(), ystar[0]), normC, &w);
        }

        static void
        AccumulateOperatorGradient(const typename TTraits::PreProcessType& x,
                                   const SystemVectorCRef<TValue, VarDim>& yRef,
                                   const std::vector<TSolution>& yhats,
                                   const std::vector<TSolution>& ystars,
                                   TValue normC,
                                   const typename TTraits::LinearOperatorRef& op)
        {
            const auto cx = yRef.Width(), cy = yRef.Height();

            const auto& yhat = yhats[0], ystar = ystars[0];
            SystemVectorCRef<TValue, VarDim> yhatRef(cx, cy, yhat);
            const TSolution hyhat  = 0.5 * yhat;
            SystemVectorCRef<TValue, VarDim> hyhatRef(cx, cy, hyhat);
            const TSolution mystar = -ystar;
            SystemVectorCRef<TValue, VarDim> mystarRef(cx, cy, mystar);
            const TSolution hystar = 0.5 * ystar;
            SystemVectorCRef<TValue, VarDim> hystarRef(cx, cy, hystar);

            // 1/2 yhat - 1/2 ystar
            op.AccumulateGradient(x, hyhatRef,  yhatRef, normC);
            op.AccumulateGradient(x, hystarRef, mystarRef, normC);
        }
    };

    // Specialization for Gaussian mean field training: Same as continuous perceptron, but with
    // an additional (factorized) entropy term that enforces "peakedness" of the energy.
    template<typename TTraits>
    struct LossTraits<TTraits, Loss::ContinuousMeanField>
    {
        static const size_t VarDim = TTraits::UnaryGroundLabel::Size;

        typedef typename TTraits::ValueType                     TValue;
        typedef Eigen::Matrix<TValue, Eigen::Dynamic, 1>        TSolution;
        typedef typename TTraits::UnaryFactorType::Factor       TUnaryFactor;
        typedef typename TTraits::PairwiseFactorType::Factor    TPairwiseFactor;
        typedef typename TTraits::UnaryFactorType::TWeights     TUnaryWeights;
        typedef typename TTraits::PairwiseFactorType::TWeights  TPairwiseWeights;
        typedef typename Classify::LinearSystem<TTraits, Loss::NoErrorTerm<TTraits>>::Type TLinearSystem;

        static std::shared_ptr<TLinearSystem>
        ComputeMeanParameters(const typename TTraits::PreProcessType& x,
                              const ImageRefC<typename TTraits::UnaryGroundLabel>& y,
                              const std::vector<typename TTraits::UnaryFactorType>    &Us,
                              const std::vector<typename TTraits::PairwiseFactorType> &Ps,
                              const typename TTraits::LinearOperatorVector& Ls,
                              size_t maxNumIt, TValue residualTol,
                              std::vector<TSolution>& yhat,
                              std::vector<TSolution>& invDiag)
        {
            yhat.resize(1);
            invDiag.resize(1);

            // Determine the mean under our model distribution (the prediction)
            std::shared_ptr<TLinearSystem> system( new TLinearSystem(x, y.Width(), y.Height(), Us, Ps, Ls, y) );
            yhat[0] = Minimization::CGSolve(*system, (unsigned) maxNumIt, residualTol, false);
            system->ProvideInverseDiagonal(invDiag[0]);

            // Return the system for potential latter use
            return system;
        }

        static TValue
        ComputeObjective(const ImageRefC<typename TTraits::UnaryGroundLabel>& y,
                         std::shared_ptr<TLinearSystem> system,
                         const std::vector<TSolution>& yhat,
                         const std::vector<TSolution>& invDiag,
                         TValue normC)
        {
            Classify::Detail::UnconstrainedQuadratic<TTraits, Loss::NoErrorTerm<TTraits>> quadratic(*system);
            TSolution g;
            const auto m         = invDiag[0].size();
            const auto predicted = quadratic.Eval(yhat[0], g) + 0.5 * m; // (invDiag.dot(invDiag.array().cwiseInverse().matrix())) == m;
            const auto entropy   = TValue(0.5) * invDiag[0].array().log().sum() + 0.5 * m * std::log(TValue(2.0) * Constants::pi() * Constants::e());
            const auto logpart   = predicted - entropy;
            const auto observed  = quadratic.Eval(Utility::SolutionFromLabeling<TTraits>(y), g);

            return (observed - logpart)/normC;
        }

        static void
        AccumulateFactorGradient(const SystemVectorCRef<TValue, VarDim>& yRef,
                                 const std::vector<TSolution>& yhat,
                                 const std::vector<TSolution>& invDiag,
                                 TValue normC,
                                 TUnaryFactor &factor)
        {
            factor.AccumulateMeanFieldGradient(SystemVectorCRef<TValue, VarDim>(yRef.Width(), yRef.Height(), yhat[0]), yRef,
                                               SystemVectorCRef<TValue, VarDim>(yRef.Width(), yRef.Height(), invDiag[0]), normC);
        }

        static void
        AccumulateFactorGradient(const SystemVectorCRef<TValue, VarDim>& yRef,
                                 const std::vector<TSolution>& yhat,
                                 const std::vector<TSolution>& invDiag,
                                 TValue normC,
                                 TPairwiseFactor &factor)
        {
            factor.AccumulateMeanFieldGradient(SystemVectorCRef<TValue, VarDim>(yRef.Width(), yRef.Height(), yhat[0]), yRef,
                                               SystemVectorCRef<TValue, VarDim>(yRef.Width(), yRef.Height(), invDiag[0]), normC);
        }

        static void
        AccumulateFactorGradient(const SystemVectorCRef<TValue, VarDim>& yRef,
                                 const std::vector<TSolution>& yhat,
                                 const std::vector<TSolution>& invDiag,
                                 TValue normC,
                                 TUnaryFactor &factor, TUnaryWeights& w)
        {
            factor.AccumulateMeanFieldGradient(SystemVectorCRef<TValue, VarDim>(yRef.Width(), yRef.Height(), yhat[0]), yRef,
                                               SystemVectorCRef<TValue, VarDim>(yRef.Width(), yRef.Height(), invDiag[0]), normC, &w);
        }

        static void
        AccumulateFactorGradient(const SystemVectorCRef<TValue, VarDim>& yRef,
                                 const std::vector<TSolution>& yhat,
                                 const std::vector<TSolution>& invDiag,
                                 TValue normC,
                                 TPairwiseFactor &factor, TPairwiseWeights& w)
        {
            factor.AccumulateMeanFieldGradient(SystemVectorCRef<TValue, VarDim>(yRef.Width(), yRef.Height(), yhat[0]), yRef,
                                               SystemVectorCRef<TValue, VarDim>(yRef.Width(), yRef.Height(), invDiag[0]), normC, &w);
        }

        static void
        AccumulateOperatorGradient(const typename TTraits::PreProcessType& x,
                                   const SystemVectorCRef<TValue, VarDim>& yRef,
                                   const std::vector<TSolution>& yhat,
                                   const std::vector<TSolution>& invDiag,
                                   TValue normC,
                                   const typename TTraits::LinearOperatorRef& op)
        {
            op.AccumulateMeanFieldGradient(x, SystemVectorCRef<TValue, VarDim>(yRef.Width(), yRef.Height(), yhat[0]), yRef,
                                           SystemVectorCRef<TValue, VarDim>(yRef.Width(), yRef.Height(), invDiag[0]), normC);
        }
    };

    // Structured Hamming loss for discrete output variables. This is used along with the convex QP
    // relaxation described in Jancsary et al. (ICML 2013). The Hamming loss is injected into the
    // "loss-augmented" inference problem of the max-margin learning formulation.
    template<typename TTraits>
    struct LossTraits<TTraits, Loss::DiscreteHamming>
    {
        static const size_t VarDim = TTraits::UnaryGroundLabel::Size;

        typedef typename TTraits::ValueType                     TValue;
        typedef Eigen::Matrix<TValue, Eigen::Dynamic, 1>        TSolution;
        typedef typename TTraits::UnaryFactorType::Factor       TUnaryFactor;
        typedef typename TTraits::PairwiseFactorType::Factor    TPairwiseFactor;
        typedef typename TTraits::UnaryFactorType::TWeights     TUnaryWeights;
        typedef typename TTraits::PairwiseFactorType::TWeights  TPairwiseWeights;
        typedef typename Classify::LinearSystem<TTraits, Loss::ErrorTerm<TTraits, Loss::DiscreteHamming>>::Type  TLinearSystem;
        typedef Classify::Detail::ConstrainedQuadratic<TTraits, Loss::ErrorTerm<TTraits, Loss::DiscreteHamming>> TQuadratic;

        static std::shared_ptr<TLinearSystem>
        ComputeMeanParameters(const typename TTraits::PreProcessType& x,
                              const ImageRefC<typename TTraits::UnaryGroundLabel>& y,
                              const std::vector<typename TTraits::UnaryFactorType>    &Us,
                              const std::vector<typename TTraits::PairwiseFactorType> &Ps,
                              const typename TTraits::LinearOperatorVector& Ls,
                              size_t maxNumIt, TValue residualTol,
                              std::vector<TSolution>& yhat,
                              std::vector<TSolution>& ystar)
        {
            yhat.resize(1);
            ystar.resize(1);

            // Determine the mean under our model distribution (the prediction)
            std::shared_ptr<TLinearSystem> system( new TLinearSystem(x, y.Width(), y.Height(), Us, Ps, Ls, y) );
            TQuadratic quadratic(*system);
            Minimization::SPGMinimizeCQ(quadratic, yhat[0], maxNumIt, residualTol, false, false, 3);
            ystar[0] = Utility::SolutionFromLabeling<TTraits>(y);

            // Return the system for potential latter use
            return system;
        }

        static TValue
        ComputeObjective(const ImageRefC<typename TTraits::UnaryGroundLabel>& y,
                         std::shared_ptr<TLinearSystem> system,
                         const std::vector<TSolution>& yhat,
                         const std::vector<TSolution>& ystar,
                         TValue normC)
        {
            TQuadratic quadratic(*system);
            TSolution g;
            auto predicted = quadratic.Eval(yhat[0], g);
            auto observed  = quadratic.Eval(ystar[0], g);

            return (observed - predicted)/normC;
        }

        static void
        AccumulateFactorGradient(const SystemVectorCRef<TValue, VarDim>& yRef,
                                 const std::vector<TSolution>& yhat,
                                 const std::vector<TSolution>& ystar,
                                 TValue normC,
                                 TUnaryFactor &factor)
        {
            factor.AccumulateEnergyBasedGradient(SystemVectorCRef<TValue, VarDim>(yRef.Width(), yRef.Height(), yhat[0]),
                                                 SystemVectorCRef<TValue, VarDim>(yRef.Width(), yRef.Height(), ystar[0]), normC);
        }

        static void
        AccumulateFactorGradient(const SystemVectorCRef<TValue, VarDim>& yRef,
                                 const std::vector<TSolution>& yhat,
                                 const std::vector<TSolution>& ystar,
                                 TValue normC,
                                 TPairwiseFactor &factor)
        {
            factor.AccumulateEnergyBasedGradient(SystemVectorCRef<TValue, VarDim>(yRef.Width(), yRef.Height(), yhat[0]),
                                                 SystemVectorCRef<TValue, VarDim>(yRef.Width(), yRef.Height(), ystar[0]), normC);
        }

        static void
        AccumulateFactorGradient(const SystemVectorCRef<TValue, VarDim>& yRef,
                                 const std::vector<TSolution>& yhat,
                                 const std::vector<TSolution>& ystar,
                                 TValue normC,
                                 TUnaryFactor &factor, TUnaryWeights& w)
        {
            factor.AccumulateEnergyBasedGradient(SystemVectorCRef<TValue, VarDim>(yRef.Width(), yRef.Height(), yhat[0]),
                                                 SystemVectorCRef<TValue, VarDim>(yRef.Width(), yRef.Height(), ystar[0]), normC, &w);
        }

        static void
        AccumulateFactorGradient(const SystemVectorCRef<TValue, VarDim>& yRef,
                                 const std::vector<TSolution>& yhat,
                                 const std::vector<TSolution>& ystar,
                                 TValue normC,
                                 TPairwiseFactor &factor, TPairwiseWeights& w)
        {
            factor.AccumulateEnergyBasedGradient(SystemVectorCRef<TValue, VarDim>(yRef.Width(), yRef.Height(), yhat[0]),
                                                 SystemVectorCRef<TValue, VarDim>(yRef.Width(), yRef.Height(), ystar[0]), normC, &w);
        }

        static void
        AccumulateOperatorGradient(const typename TTraits::PreProcessType& x,
                                   const SystemVectorCRef<TValue, VarDim>& yRef,
                                   const std::vector<TSolution>& yhats,
                                   const std::vector<TSolution>& ystars,
                                   TValue normC,
                                   const typename TTraits::LinearOperatorRef& op)
        {
            const auto& yhat = yhats[0], ystar = ystars[0];
            const auto cx = yRef.Width(), cy = yRef.Height();
            SystemVectorCRef<TValue, VarDim> yhatRef(cx, cy, yhat);
            const TSolution hyhat  = 0.5 * yhat;
            SystemVectorCRef<TValue, VarDim> hyhatRef(cx, cy, hyhat);
            const TSolution mystar = -ystar;
            SystemVectorCRef<TValue, VarDim> mystarRef(cx, cy, mystar);
            const TSolution hystar = 0.5 * ystar;
            SystemVectorCRef<TValue, VarDim> hystarRef(cx, cy, hystar);

            // 1/2 yhat - 1/2 ystar
            op.AccumulateGradient(x, hyhatRef,  yhatRef, normC);
            op.AccumulateGradient(x, hystarRef, mystarRef, normC);
        }
    };

    // Class that describes a factor graph and provides functionality for evaluating a given loss
    // function given the current model and for computing the gradient with respect to the
    // model parameters.
    template <typename TTraits>
    class FactorGraph
    {
    public:
        static const size_t                                    VarDim = TTraits::UnaryGroundLabel::Size;

        typedef typename TTraits::UnaryGroundLabel             TLabel;
        typedef typename TTraits::ValueType                    TValue;
        typedef typename TTraits::PreProcessType               TPrep;

        typedef typename TTraits::UnaryFactorType              TUnaryFactorType;
        typedef typename TTraits::PairwiseFactorType           TPairwiseFactorType;
        typedef typename TUnaryFactorType::Factor              TUnaryFactor;
        typedef typename TPairwiseFactorType::Factor           TPairwiseFactor;
        typedef typename TUnaryFactorType::TWeights            TUnaryWeights;
        typedef typename TPairwiseFactorType::TWeights         TPairwiseWeights;

        typedef std::pair < typename TTraits::UnaryWeightsImageVector,
                typename TTraits::PairwiseWeightsImageVector > TWeightsImages;

        typedef std::pair < typename TTraits::UnaryBasisImageVector,
                typename TTraits::PairwiseBasisImageVector >   TBasisImages;

        typedef Eigen::Matrix<TValue, VarDim, 1>               TVector;
        typedef Eigen::Matrix<TValue, VarDim, VarDim>          TMatrix;

        typedef Eigen::Matrix<TValue, Eigen::Dynamic, 1>       TSolution;

    protected:
        const TPrep       x;     // the preprocessed input image
        ImageRefC<TLabel> y;     // the groundtruth labels

        const std::vector<TUnaryFactorType>    &Us; // reference to the vector of unary factor types
        const std::vector<TPairwiseFactorType> &Ps; // reference to the vector of pairwise factor types
        const typename TTraits::LinearOperatorVector& Ls; // reference to the vector of custom linear operators

        // Convenience method that invokes a given functor for each conditioned factor connected to the
        // center variable of this subgraph.
        template<typename TOp, typename TErrorTerm, bool instantiate>
        void for_each_unary_factor(std::shared_ptr<Classify::Detail::LinearSystem<TTraits, TErrorTerm, instantiate>> system, const TOp& op) const
        {
            if( system )
            {
                for(int idx = 0; idx < Us.size(); ++idx)
                    Us[idx].ForEachInstance(system->Prep(), system->UnaryWeightsImage(idx), system->UnaryBasisImage(idx), op);
            }
            else
            {
                for(int idx = 0; idx < Us.size(); ++idx)
                    Us[idx].ForEachInstance(x, y, op);
            }
        }

        template<typename TOp, typename TErrorTerm>
        void for_each_unary_factor(std::shared_ptr<Classify::Detail::OnTheFlySystem<TTraits, TErrorTerm>> system, TOp op) const
        {
            for(int idx = 0; idx < Us.size(); ++idx)
                Us[idx].ForEachInstance(x, y, op);
        }

        template<typename TOp, typename TErrorTerm, bool instantiate>
        void for_each_pairwise_factor(std::shared_ptr<Classify::Detail::LinearSystem<TTraits, TErrorTerm, instantiate>> system, const TOp& op) const
        {
            if( system )
            {
                for(int idx = 0; idx < Ps.size(); ++idx)
                    Ps[idx].ForEachInstance(system->Prep(), system->PairwiseWeightsImage(idx), system->PairwiseBasisImage(idx), op);
            }
            else
            {
                for(int idx = 0; idx < Ps.size(); ++idx)
                    Ps[idx].ForEachInstance(x, y, op);
            }
        }

        template<typename TOp, typename TErrorTerm>
        void for_each_pairwise_factor(std::shared_ptr<Classify::Detail::OnTheFlySystem<TTraits, TErrorTerm>> system, const TOp& op) const
        {
            for(int idx = 0; idx < Ps.size(); ++idx)
                Ps[idx].ForEachInstance(x, y, op);
        }

        template<typename TOp>
        void for_each_linear_operator(const TOp& op) const
        {
            for( int idx = 0; idx < Ls.size(); ++idx )
                op(Ls[idx]);
        }

    public:
        FactorGraph(const TPrep& x_, const ImageRefC<TLabel>& y_,
                    const std::vector<TUnaryFactorType>& Us_, const std::vector<TPairwiseFactorType>& Ps_,
                    const typename TTraits::LinearOperatorVector& Ls_)
            : x(x_), y(y_), Us(Us_), Ps(Ps_), Ls(Ls_)
        {
        }

        virtual ~FactorGraph() {}

        const TPrep& Prep() const
        {
            return x;
        }

        // With some abuse of terminology, we use "mean parameters" to refer to statistics obtained from the solution
        // of the energy minimization problem that allow for efficient computation of the gradient.
        // The concrete content of the mean parameters depends on the training objected (i.e. the loss) in use.
        template<typename TLossTag>
        void
        ComputeMeanParameters(size_t maxNumIt, TValue residualTol, std::vector<TSolution>& muPrediction, std::vector<TSolution>& muLossGradient) const
        {
            LossTraits<TTraits, TLossTag>::ComputeMeanParameters(x, y, Us, Ps, Ls,
                    maxNumIt, residualTol, muPrediction, muLossGradient);
        }

        // Evaluate the training objective on this factor graph, for the current model parameters,
        // and accumulate the gradient with respect to the model parameters.
        template<typename TLossTag>
        TValue ComputeObjectiveAccumulateGradient(TValue normC, size_t maxNumIt, TValue residualTol) const
        {
            // Compute mean parameters
            std::vector<TSolution> muPrediction, muLossGradient;
            auto system = LossTraits<TTraits, TLossTag>::ComputeMeanParameters(x, y, Us, Ps, Ls,
                          maxNumIt, residualTol,
                          muPrediction, muLossGradient);

            // Accumulate gradient
            const auto yVec = Utility::SolutionFromLabeling<TTraits>(y);
            SystemVectorCRef<TValue, VarDim> yRef(y.Width(), y.Height(), yVec);

            for_each_unary_factor(system, [&](TUnaryFactor & factor)
            {
                LossTraits<TTraits, TLossTag>::AccumulateFactorGradient(yRef, muPrediction, muLossGradient, normC, factor);
            });
            for_each_pairwise_factor(system, [&](TPairwiseFactor & factor)
            {
                LossTraits<TTraits, TLossTag>::AccumulateFactorGradient(yRef, muPrediction, muLossGradient, normC, factor);
            });
            for_each_linear_operator([&](const typename TTraits::LinearOperatorRef& op)
            {
                LossTraits<TTraits, TLossTag>::AccumulateOperatorGradient(x, yRef, muPrediction, muLossGradient, normC, op);
            });

            // Return objective
            const auto obj = LossTraits<TTraits, TLossTag>::ComputeObjective(y, system, muPrediction, muLossGradient, normC);

            return obj;
        }

        // For each gradient contribution by a factor of the specified type, perform the specified action.
        // This is used to collect the "data points" for tree training, i.e. splitting of nodes.
        template<typename TLossTag, typename TOp>
        void
        ForEachGradientContributionOfType(const TUnaryFactorType& T, const std::vector<TSolution> &muPrediction, const std::vector<TSolution>& muLossGradient,
                                          TValue normC, const TOp& op) const
        {
            const auto yVec = Utility::SolutionFromLabeling<TTraits>(y);
            SystemVectorCRef<TValue, VarDim> yRef(y.Width(), y.Height(), yVec);

            T.ForEachInstance(x, y, [&](TUnaryFactor & factor)
            {
                TUnaryWeights w;
                LossTraits<TTraits, TLossTag>::AccumulateFactorGradient(yRef, muPrediction, muLossGradient, normC, factor, w);
                if( ! ( w.Gl.isConstant(0) && w.Gq.isConstant(0) ) )
                    op(factor.PosX(), factor.PosY(), w.Gl, w.Gq);
            });
        }

        template<typename TLossTag, typename TOp>
        void
        ForEachGradientContributionOfType(const TUnaryFactorType& T, const std::vector<TSolution> &muPrediction, const std::vector<TSolution>& muLossGradient,
                                          TValue normC, const VecCRef<Vector2D<int>>& subsample, const TOp& op) const
        {
            const auto yVec = Utility::SolutionFromLabeling<TTraits>(y);
            SystemVectorCRef<TValue, VarDim> yRef(y.Width(), y.Height(), yVec);

            T.ForEachInstance(x, y, subsample, [&](TUnaryFactor & factor)
            {
                TUnaryWeights w;
                LossTraits<TTraits, TLossTag>::AccumulateFactorGradient(yRef, muPrediction, muLossGradient, normC, factor, w);
                if( ! ( w.Gl.isConstant(0) && w.Gq.isConstant(0) ) )
                    op(factor.PosX(), factor.PosY(), w.Gl, w.Gq);
            });
        }

        template<typename TLossTag, typename TOp>
        void
        ForEachGradientContributionOfType(const TPairwiseFactorType& T, const std::vector<TSolution> &muPrediction, const std::vector<TSolution>& muLossGradient,
                                          TValue normC, const TOp& op) const
        {
            const auto yVec = Utility::SolutionFromLabeling<TTraits>(y);
            SystemVectorCRef<TValue, VarDim> yRef(y.Width(), y.Height(), yVec);

            T.ForEachInstance(x, y, [&](TPairwiseFactor & factor)
            {
                TPairwiseWeights w;
                LossTraits<TTraits, TLossTag>::AccumulateFactorGradient(yRef, muPrediction, muLossGradient, normC, factor, w);
                if( ! ( w.Gl.isConstant(0) && w.Gq.isConstant(0) ) )
                    op(factor.PosX(), factor.PosY(), w.Gl, w.Gq);
            });
        }


        template<typename TLossTag, typename TOp>
        void
        ForEachGradientContributionOfType(const TPairwiseFactorType& T, const std::vector<TSolution> &muPrediction, const std::vector<TSolution>& muLossGradient,
                                          TValue normC, const VecCRef<Vector2D<int>>& subsample, const TOp& op) const
        {
            const auto yVec = Utility::SolutionFromLabeling<TTraits>(y);
            SystemVectorCRef<TValue, VarDim> yRef(y.Width(), y.Height(), yVec);

            T.ForEachInstance(x, y, subsample, [&](TPairwiseFactor & factor)
            {
                TPairwiseWeights w;
                LossTraits<TTraits, TLossTag>::AccumulateFactorGradient(yRef, muPrediction, muLossGradient, normC, factor, w);
                if( ! ( w.Gl.isConstant(0) && w.Gq.isConstant(0) ) )
                    op(factor.PosX(), factor.PosY(), w.Gl, w.Gq);
            });
        }

        template<typename TFactorType>
        size_t
        NumGradientContributionsOfType(const TFactorType& T) const
        {
            return T.NumFactors(y.Width(), y.Height());
        }

        template<typename TFactorType>
        size_t
        NumGradientContributionsOfType(const TFactorType& T, const VecCRef<Vector2D<int>>& subsample) const
        {
            return T.NumFactors(y.Width(), y.Height(), subsample);
        }
    };

    template<typename TTraits>
    void
    for_each_factor_graph(const typename TTraits::DataSampler& traindb,
                          const typename TTraits::UnaryFactorTypeVector& Us,
                          const typename TTraits::PairwiseFactorTypeVector& Ps,
                          const typename TTraits::LinearOperatorVector& Ls,
                          const std::function<void (const FactorGraph<TTraits>&)>& op)
    {
        const auto size = traindb.GetImageCount();

        for(int i = 0; i < size; ++i)
        {
            auto prep   = TTraits::Feature::PreProcess(traindb.GetInputImage(i));
            auto ground = traindb.GetGroundTruthImage(i);
            op(FactorGraph<TTraits>(prep, ground, Us, Ps, Ls));
        }
    }

    template<typename TTraits>
    void
    for_each_factor_graph_with_index(const typename TTraits::DataSampler& traindb,
                                     const typename TTraits::UnaryFactorTypeVector& Us,
                                     const typename TTraits::PairwiseFactorTypeVector& Ps,
                                     const typename TTraits::LinearOperatorVector& Ls,
                                     const std::function<void (size_t id, const FactorGraph<TTraits>&)>& op)
    {
        const auto size = traindb.GetImageCount();

        for(int i = 0; i < size; ++i)
        {
            auto prep   = TTraits::Feature::PreProcess(traindb.GetInputImage(i));
            auto ground = traindb.GetGroundTruthImage(i);
            op(i, FactorGraph<TTraits>(prep, ground, Us, Ps, Ls));
        }
    }

    // CODE FOR SETTING UP LINEAR SYSTEMS

    // Determines the offset of a particular pixel of an image in a stacked vector representation
    // of all mean vectors.
    template< int VarDim >
    int VarOffset(int cx, int cy, const Vector2D<int>& v)
    {
        return VarDim * (cx * v.y + v.x);
    }

    template< int VarDim >
    int VarOffset(int cx, int cy, int x, int y)
    {
        return VarDim * (cx * y + x);
    }


    // Base class of a wrapper around a stacked vector representation of the individual
    // variable means. This can be used to efficiently find the offset of a particular
    // pixel in our image into the stacked mean vector.
    template< int VarDim >
    class SystemVectorBase
    {
    private:
        const int cx;
        const int cy;

    public:
        SystemVectorBase(int cx_, int cy_) : cx(cx_), cy(cy_)
        {
        }

        inline int VarOffset(const Vector2D<int>& v) const
        {
            return Compute::VarOffset<VarDim>(cx, cy, v);
        }

        inline int VarOffset(int x, int y) const
        {
            return Compute::VarOffset<VarDim>(cx, cy, x, y);
        }

        inline size_t NumPixels() const
        {
            return cx * cy;
        }

        int Width() const
        {
            return cx;
        }

        int Height() const
        {
            return cy;
        }
    };

    // Wrapper that provides write access to the underlying stacked mean vector.
    template< typename TValue, size_t VarDim >
    class SystemVectorRef : public SystemVectorBase<VarDim>
    {
    public:
        typedef SystemVectorBase<VarDim> Base;
        typedef Eigen::Matrix<TValue, Eigen::Dynamic, 1> VectorType;

    private:
        VectorType& values;

    public:
        SystemVectorRef(int cx_, int cy_, VectorType& values_) : Base::SystemVectorBase(cx_, cy_), values(values_)
        {
            values = VectorType::Zero(VarDim * cx_ * cy_);
        }

        Eigen::VectorBlock<VectorType, VarDim> operator()(const Vector2D<int>& v) const
        {
            return values.template segment<VarDim>(Base::VarOffset(v));
        }

        Eigen::VectorBlock<VectorType, VarDim> operator()(int offset) const
        {
            return values.template segment<VarDim>(offset);
        }

        Eigen::VectorBlock<VectorType, VarDim> operator[](int offset) const
        {
            return values.template segment<VarDim>(VarDim*offset);
        }

        Eigen::VectorBlock<VectorType, VarDim> operator()(int x, int y) const
        {
            return values.template segment<VarDim>(Base::VarOffset(x, y));
        }

        VectorType& Raw() const
        {
            return values;
        }
    };

    // Wrapper that provides write access to the underlying stacked block diagonal.
    template< typename TValue, size_t VarDim >
    class BlockDiagonalRef : public SystemVectorBase<VarDim>
    {
    public:
        typedef SystemVectorBase<VarDim> Base;
        typedef Eigen::Matrix<TValue, Eigen::Dynamic, VarDim> VectorType;

    private:
        VectorType& values;

    public:
        BlockDiagonalRef(int cx_, int cy_, VectorType& values_) : Base::SystemVectorBase(cx_, cy_), values(values_)
        {
            values = VectorType::Zero(VarDim * cx_ * cy_, VarDim);
        }

        Eigen::Block<VectorType, VarDim, VarDim> operator()(const Vector2D<int>& v) const
        {
            return values.template block<VarDim, VarDim>(Base::VarOffset(v), 0);
        }

        Eigen::Block<VectorType, VarDim, VarDim> operator[](int offset) const
        {
            return values.template block<VarDim, VarDim>(VarDim*offset, 0);
        }

        Eigen::Block<VectorType, VarDim, VarDim> operator()(int x, int y) const
        {
            return values.template block<VarDim, VarDim>(Base::VarOffset(x, y), 0);
        }

        VectorType& Raw() const
        {
            return values;
        }
    };

    // Wrapper that provides write access to the underlying stacked block diagonal.
    template< typename TValue, size_t VarDim >
    class BlockDiagonalCRef : public SystemVectorBase<VarDim>
    {
    public:
        typedef SystemVectorBase<VarDim> Base;
        typedef Eigen::Matrix<TValue, Eigen::Dynamic, VarDim> VectorType;

    private:
        const VectorType& values;

    public:
        BlockDiagonalCRef(int cx_, int cy_, const VectorType& values_) : Base::SystemVectorBase(cx_, cy_), values(values_)
        {
        }

        const Eigen::Block<const VectorType, VarDim, VarDim> operator()(const Vector2D<int>& v) const
        {
            return values.template block<VarDim, VarDim>(Base::VarOffset(v), 0);
        }

        const Eigen::Block<const VectorType, VarDim, VarDim> operator[](int offset) const
        {
            return values.template block<VarDim, VarDim>(VarDim*offset, 0);
        }

        const Eigen::Block<const VectorType, VarDim, VarDim> operator()(int x, int y) const
        {
            return values.template block<VarDim, VarDim>(Base::VarOffset(x, y), 0);
        }

        const VectorType& Raw() const
        {
            return values;
        }
    };

    // Wrapper that provides const access to the underlying stacked mean vector.
    template< typename TValue, size_t VarDim >
    class SystemVectorCRef : public SystemVectorBase<VarDim>
    {
    public:
        typedef SystemVectorBase<VarDim> Base;
        typedef Eigen::Matrix<TValue, Eigen::Dynamic, 1> VectorType;

    private:
        const VectorType& values;

    public:
        SystemVectorCRef(int cx_, int cy_, const VectorType& values_) : Base::SystemVectorBase(cx_, cy_), values(values_)
        {
        }

        const Eigen::VectorBlock<const VectorType, VarDim> operator()(const Vector2D<int>& v) const
        {
            return values.template segment<VarDim>(Base::VarOffset(v));
        }

        const Eigen::VectorBlock<const VectorType, VarDim> operator()(int offset) const
        {
            return values.template segment<VarDim>(offset);
        }

        const Eigen::VectorBlock<const VectorType, VarDim> operator[](int offset) const
        {
            return values.template segment<VarDim>(VarDim*offset);
        }

        const Eigen::VectorBlock<const VectorType, VarDim> operator()(int x, int y) const
        {
            return values.template segment<VarDim>(Base::VarOffset(x, y));
        }

        const VectorType& Raw() const
        {
            return values;
        }
    };

    // One row of an instantiated system matrix A of a linear system Ax = b resulting from a regression tree field
    // model at test time.
    // Each row corresponds to one variable of the Markov Random Field/one pixel in the input image.
    // Essentially, this implementation uses a coordinate storage scheme that exploits the special block structure
    // of the sparse matrix.
    template <typename TValue, size_t VarDim>
    class SystemMatrixRow
    {
    public:
        typedef Eigen::Matrix<TValue, VarDim, VarDim> TMatrix;
        typedef SystemVectorRef<TValue, VarDim>       TSystemVectorRef;
        typedef SystemVectorCRef<TValue, VarDim>      TSystemVectorCRef;

    private:
        // Precision blocks of variables adjacent to the variable represented by this row.
        std::vector< TMatrix, Eigen::aligned_allocator<TMatrix> > blocks;

        // Offsets of the adjacents variables/columns into the flat vector representation.
        std::vector< int > colOffsets;

        // Size of the underlying image.
        int cx;
        int cy;

        // Offset of this row into the flat vector representation.
        int rowOffset;

    public:

        // This must be called prior to multiplying the row with a dense vector, as it provides important
        // information that is used to determine the offsets into the dense vector space.
        SystemMatrixRow<TValue, VarDim>& Init(int cx_, int cy_, const Vector2D<int>& v, size_t numPairwise)
        {
            cx = cx_;
            cy = cy_;
            rowOffset = VarOffset<VarDim>(cx, cy, v);
            blocks.reserve(1 + 2 * numPairwise); // one for ourself, plus two adjacent variables for each pairwise factor type
            return *this;
        }

        // Add one precision block describing the interaction of the variable represented by this row
        // with the given variable 'v'.
        template <typename Derived>
        void Add(const Vector2D<int>& v, const Eigen::MatrixBase<Derived>& negPrecisionBlock)
        {
            const auto colOffset = VarOffset<VarDim>(cx, cy, v);
            const auto position  = static_cast<size_t>(std::lower_bound(colOffsets.begin(),
                                   colOffsets.end(), colOffset) - colOffsets.begin());

            if(position >= colOffsets.size())
            {
                colOffsets.push_back(colOffset);
                blocks.push_back(- negPrecisionBlock);
            }
            else if(colOffsets[position] == colOffset)
            {
                blocks[ position ] -= negPrecisionBlock;
            }
            else
            {
                colOffsets.insert(colOffsets.begin() + position, colOffset);
                blocks.insert(blocks.begin() + position, - negPrecisionBlock);
            }
        }

        template <typename Derived>
        void Add(int x, int y, const Eigen::MatrixBase<Derived>& precisionBlock)
        {
            TMatrix tmp = -precisionBlock;
            Add(Vector2D<int>(x, y), tmp);
        }

        // Multiply this row by the given vector 'x' and accumulate the result in 'Ax'.
        void MultiplyBy(const TSystemVectorRef& Ax, const TSystemVectorCRef& x) const
        {
            for(size_t c = 0; c < colOffsets.size(); ++c)
                Ax(rowOffset) += blocks[c] * x(colOffsets[c]);
        }

        // Returns the number of precision blocks stored by this row.
        int NumBlocks() const
        {
            return static_cast<int>(blocks.size());
        }

        const int Width() const
        {
            return cx;
        }

        const int Height() const
        {
            return cy;
        }

        // Exports the row into a low-level flat coordinate storage representation.
        template< typename TExportValue >
        void Export(std::vector<int>& rowIndices,
                    std::vector<int>& colIndices,
                    std::vector<TExportValue>& values,
                    int entryOffset) const
        {
            for(int row = 0; row < VarDim; ++row)
            {
                for(int block = 0; block < blocks.size(); ++block)
                {
                    const int colOffset = colOffsets[block];

                    for(int col = 0; col < VarDim; ++col)
                    {
                        rowIndices[entryOffset] = rowOffset + row;
                        colIndices[entryOffset] = colOffset + col;
                        values[entryOffset++]   = blocks[block](row, col);
                    }
                }
            }
        }

        // Exports the row into a low-level flat coordinate storage representation.
        template< typename TExportValue >
        void Export(std::vector<Eigen::Triplet<TExportValue>>& triplets,
                    int entryOffset) const
        {
            for(int row = 0; row < VarDim; ++row)
            {
                for(int block = 0; block < blocks.size(); ++block)
                {
                    const int colOffset = colOffsets[block];

                    for(int col = 0; col < VarDim; ++col)
                    {
                        triplets[entryOffset++] = Eigen::Triplet<TExportValue>(rowOffset+row, colOffset+col, blocks[block](row, col));
                    }
                }
            }
        }
    };

    // A couple of forward declaration
    template<typename TTraits>
    void
    for_each_precomputed_subgraph(const typename TTraits::PreProcessType& prep, const int cx, const int cy,
                                  const typename TTraits::UnaryFactorTypeVector& Us,
                                  const typename TTraits::PairwiseFactorTypeVector& Ps,
                                  const typename TTraits::UnaryWeightsImageVector& Uws,
                                  const typename TTraits::PairwiseWeightsImageVector& Pws,
                                  const typename TTraits::UnaryBasisImageVector& Ubs,
                                  const typename TTraits::PairwiseBasisImageVector& Pbs,
                                  const std::function<void (const PrecomputedSubgraph<TTraits>&)>& op);

    template<typename TTraits, bool Subsample, bool Parallel>
    void
    for_each_conditioned_subgraph(const typename TTraits::DataSampler& traindb,
                                  const typename TTraits::UnaryFactorTypeVector& Us,
                                  const typename TTraits::PairwiseFactorTypeVector& Ps,
                                  const std::function<void (const ConditionedSubgraph<TTraits>&)>& op);


    // Implements an explicit instantiation of the sparse system matrix A of the sparse linear
    // system Ax = b resulting from an RTF model at test time.
    template <typename TTraits>
    class SystemMatrix
    {
    public:
        static const int                          VarDim = TTraits::UnaryGroundLabel::Size;
        typedef typename TTraits::ValueType       TValue;
        typedef SystemVectorRef<TValue, VarDim>   TSystemVectorRef;
        typedef SystemVectorCRef<TValue, VarDim>  TSystemVectorCRef;

    private:
        const int cx; // Number of columns of the input image
        const int cy; // Number of rows of the input image

        std::vector<SystemMatrixRow<TValue, VarDim>> rows; // Rows of the system matrix, which store the precision blocks
        // describing interactions with variables that are adjacent to
        // the variable represented by the row itself.
        // The is one row/variable per pixel of the input image.

    public:

        SystemMatrix(const typename TTraits::PreProcessType& prep, const int cx_, const int cy_,
                     const typename TTraits::UnaryFactorTypeVector& Us,
                     const typename TTraits::PairwiseFactorTypeVector& Ps,
                     const typename TTraits::UnaryWeightsImageVector& Uws,
                     const typename TTraits::PairwiseWeightsImageVector& Pws,
                     const typename TTraits::UnaryBasisImageVector& Ubs,
                     const typename TTraits::PairwiseBasisImageVector& Pbs,
                     const typename TTraits::LinearOperatorVector& Ls) : cx(cx_), cy(cy_), rows(cx * cy)
        {
            size_t numPairwise = Ps.size();
            for( int l = 0; l < Ls.size(); ++l )
                numPairwise += Ls[l].NumPairwise();

            if( Us.size() > 0 || Ps.size() > 0 )
            {
#ifdef PARALLEL_SYSMATRIX_ALLOCATION
#pragma message ( "Allocating system matrix in parallel" )
                // Construct the rows of the matrix.
                for_each_precomputed_subgraph<TTraits, true>(prep, cx, cy, Us, Ps, Uws, Pws, Ubs, Pbs, [&](const PrecomputedSubgraph<TTraits>& G_j)
                {
                    const int x = G_j.PosX(), y = G_j.PosY();
                    G_j.AddPrecisionBlocks(rows[ cx * y + x ].Init(cx, cy, Vector2D<int>(x, y), numPairwise));
                    for( int l = 0; l < Ls.size(); ++l )
                        Ls[l].AddPrecisionBlocks(x, y, rows[ cx * y + x]);
                });
#else
#pragma message ( "Allocating system matrix sequentially" )
                // Construct the rows of the matrix.
                for_each_precomputed_subgraph<TTraits, false>(prep, cx, cy, Us, Ps, Uws, Pws, Ubs, Pbs, [&](const PrecomputedSubgraph<TTraits>& G_j)
                {
                    const int x = G_j.PosX(), y = G_j.PosY();
                    G_j.AddPrecisionBlocks(rows[ cx * y + x ].Init(cx, cy, Vector2D<int>(x, y), numPairwise));
                    for( int l = 0; l < Ls.size(); ++l )
                        Ls[l].AddPrecisionBlocks(x, y, rows[ cx * y + x]);
                });
#endif
            }
            else
            {
                #pragma omp parallel for
                for( int y = 0; y < cy; ++y )
                {
                    for( int x = 0; x < cx; ++x )
                    {
                        auto& row = rows[ cx * y + x ];
                        row.Init(cx, cy, Vector2D<int>(x, y), numPairwise);
                        for( int l = 0; l < Ls.size(); ++l )
                            Ls[l].AddPrecisionBlocks(x, y, row);
                    }
                }
            }
        }

        // Multiplies the matrix A described by this object with the given dense vector 'x' and stores the result in 'Ax'.
        void MultiplyBy(const TSystemVectorRef& Ax, const TSystemVectorCRef& x) const
        {
            #pragma omp parallel for
            for(int r = 0; r < (int) rows.size(); ++r)
                rows[r].MultiplyBy(Ax, x);
        }

        // The total number of non-zero precision blocks stored by this sparse matrix.
        int NumBlocks() const
        {
            int blocks = 0;

            for(size_t r = 0; r < rows.size(); ++r)
                blocks += rows[r].NumBlocks();

            return blocks;
        }

        SystemMatrixRow<TValue, VarDim>& GetRow(int x, int y)
        {
            return rows[ cx * y + x ];
        }

        // Exports the sparse matrix into a low-level coordinate storage scheme, which is useful for
        // interfacing with other sparse linear algebra libraries.
        template< typename TExportValue >
        void Export(int& numRows, std::vector<int>& rowIndices,
                    int& numCols, std::vector<int>& colIndices,
                    std::vector<TExportValue>& values) const
        {
            numRows = cx * cy * VarDim;
            numCols = cx * cy * VarDim;
            const int numEntries = NumBlocks() * VarDim * VarDim;
            rowIndices.resize(numEntries);
            colIndices.resize(numEntries);
            values.resize(numEntries);
            std::vector<int> entryOffsets(rows.size());
            int offset = 0;

            for(size_t r = 0; r < rows.size(); ++r)
            {
                entryOffsets[r] = offset;
                offset          += rows[r].NumBlocks() * VarDim * VarDim;
            }

            #pragma omp parallel for
            for(int r = 0; r < (int) rows.size(); ++r)
                rows[r].Export(rowIndices, colIndices, values, entryOffsets[r]);

            DropZeros(rowIndices, colIndices, values);
        }

        template< typename TExportValue >
        void DropZeros(std::vector<int>& rowIndices,
                       std::vector<int>& colIndices,
                       std::vector<TExportValue>& values) const
        {
            std::vector<int> newRowIndices;
            newRowIndices.reserve(rowIndices.size());
            std::vector<int> newColIndices;
            newColIndices.reserve(colIndices.size());
            std::vector<TExportValue> newValues;
            newValues.reserve(newValues.size());

            for( int i = 0; i < rowIndices.size(); ++i )
            {
                if( values[i] != TExportValue(0) )
                {
                    newRowIndices.push_back(rowIndices[i]);
                    newColIndices.push_back(colIndices[i]);
                    newValues.push_back(values[i]);
                }
            }
            rowIndices = newRowIndices;
            colIndices = newColIndices;
            values     = newValues;
        }

        template< typename TExportValue >
        void Export(Eigen::SparseMatrix<TExportValue>& Q) const
        {
            const auto numRows = cx * cy * VarDim;
            const auto numCols = cx * cy * VarDim;
            Q.resize(numRows, numCols); // matrix is square and symmetric

            const auto numEntries = NumBlocks() * VarDim * VarDim;
            std::vector<Eigen::Triplet<TExportValue>> triplets(numEntries);

            std::vector<int> entryOffsets(rows.size());
            int offset = 0;

            for(size_t r = 0; r < rows.size(); ++r)
            {
                entryOffsets[r] = offset;
                offset          += rows[r].NumBlocks() * VarDim * VarDim;
            }

            #pragma omp parallel for
            for(int r = 0; r < (int) rows.size(); ++r)
                rows[r].Export(triplets, entryOffsets[r]);

            Q.setFromTriplets(triplets.begin(), triplets.end());
        }
    };

    // Abstract class that describes the common functionality of factors that are adjacent
    // to the center variable of a particular subgraph.
    template <typename TValue, size_t VarDim>
    class ConnectedFactor
    {
    public:
        typedef SystemVectorRef<TValue, VarDim>  TSystemVectorRef;
        typedef BlockDiagonalRef<TValue, VarDim> TBlockDiagonalRef;
        typedef SystemVectorCRef<TValue, VarDim> TSystemVectorCRef;
        typedef SystemMatrixRow<TValue, VarDim>  TSystemMatrixRow;

        // Adds the precision block(s) describing the the interactions between the center variable of the subgraph
        // and the variables of this factor.
        virtual void AddPrecisionBlocks(TSystemMatrixRow& row) const = 0;

        // Adds the coefficients of this factor that are relevant to the righthand side of the sparse linear system
        // determined by the RTF model.
        virtual void AddInSiteLinearCoefficients(const TSystemVectorRef& b) const = 0;

        // Stores the product of the precision block(s) describing the interactions between the center variable of the subgraph
        // and the variables of this factor with the given vector 'x' in the given vector 'Ax'. This can be used to
        // multiply the sparse matrix of the linear system by a dense vector without ever actually instantiating it.
        virtual void AddInSiteQuadraticCoefficientsMultipliedBy(const TSystemVectorRef& Ax, const TSystemVectorCRef& x) const = 0;

        virtual void AddInDiagonal(const TSystemVectorRef& invDiag) const = 0;
        virtual void AddInDiagonal(const TBlockDiagonalRef& invDiag) const = 0;
    };

    // Represents one variable in a given image and all its connected factors.
    template< typename TTraits >
    class Subgraph
    {
    public:
        static const int                              VarDim = TTraits::UnaryGroundLabel::Size;
        typedef typename TTraits::ValueType           TValue;
        typedef typename TTraits::UnaryGroundLabel    TUnaryGroundLabel;
        typedef BlockDiagonalRef<TValue, VarDim>      TBlockDiagonalRef;
        typedef SystemVectorRef<TValue, VarDim>       TSystemVectorRef;
        typedef SystemVectorCRef<TValue, VarDim>      TSystemVectorCRef;
        typedef SystemMatrixRow<TValue, VarDim>       TSystemMatrixRow;
        typedef Eigen::Matrix<TValue, VarDim, VarDim> TMatrix;
        typedef Eigen::Matrix<TValue, VarDim, 1>      TVector;

    private:
        const Vector2D<int>& j;                                   // The center variable
        const typename TTraits::PreProcessType prep;              // The preprocessed input image
        const int cx;                                             // Width of the input image
        const int cy;                                             // Height of the input image
        const typename TTraits::UnaryFactorTypeVector& Us;        // Reference to the vector of unary factor types
        const typename TTraits::PairwiseFactorTypeVector& Ps;     // Reference to the vector of pairwise factor types

        // Applies a given functor to all instances of connected factors.
        void for_each_connected_factor(const std::function<void (const ConnectedFactor<TValue, VarDim>&)>& op) const
        {
            for(size_t u = 0; u < Us.size(); ++u)
                Us[u].ForEachConnectedInstance(j, prep, op);

            for(size_t p = 0; p < Ps.size(); ++p)
                Ps[p].ForEachConnectedInstance(j, prep, cx, cy, op);
        }

    public:
        Subgraph(const Vector2D<int>& j_, const typename TTraits::PreProcessType& prep_, const int cx_, const int cy_,
                 const typename TTraits::UnaryFactorTypeVector& Us_, const typename TTraits::PairwiseFactorTypeVector& Ps_)
            : j(j_), prep(prep_), cx(cx_), cy(cy_), Us(Us_), Ps(Ps_)
        {
        }

        // If the model consists only of unaries, we can solve in-place
        TUnaryGroundLabel Solve() const
        {
            assert(Ps.size() == 0);

            TMatrix Q = TMatrix::Zero();
            TVector l = TVector::Zero();

            for( size_t u = 0; u < Us.size(); ++u )
            {
                Us[u].ForEachConnectedInstance(j, prep, [&](const typename TTraits::UnaryFactorType::ConnectedFactor& f)
                {
                    Q -= f.GetPrecisionBlock();
                    l += f.GetLinearCoefficients();
                });
            }
            auto llt = Q.llt();
            return Utility::VectorToLabel<TUnaryGroundLabel>(llt.solve(l));
        }

        TUnaryGroundLabel Solve(const TUnaryGroundLabel& rhs) const
        {
            assert(Ps.size() == 0);

            TMatrix Q = TMatrix::Zero();

            for( size_t u = 0; u < Us.size(); ++u )
            {
                Us[u].ForEachConnectedInstance(j, prep, [&](const typename TTraits::UnaryFactorType::ConnectedFactor& f)
                {
                    Q -= f.GetPrecisionBlock();
                });
            }
            auto llt = Q.llt();
            return Utility::VectorToLabel<TUnaryGroundLabel>(llt.solve(Utility::LabelToVector(rhs)));
        }

        // Returns the x coordinate of the center variable of this subgraph.
        int PosX() const
        {
            return j.x;
        }

        // Returns the y coordinate of the center variable of this subgraph.
        int PosY() const
        {
            return j.y;
        }

        // For each connected factor, add the coefficients that are relevant to the righthand side of the sparse linear system
        // determined by the RTF model.
        void AddInSiteLinearCoefficients(const TSystemVectorRef& b) const
        {
            for_each_connected_factor([&](const ConnectedFactor<TValue, VarDim>& factor)
            {
                factor.AddInSiteLinearCoefficients(b);
            });
        }

        // Stores the product of the precision block(s) describing the interactions between the center variable of the subgraph
        // and the variables of all connected factors with the given vector 'x' in the given vector 'Ax'. This can be used to
        // multiply the sparse matrix of the linear system by a dense vector without ever actually instantiating it.
        void AddInSiteQuadraticCoefficientsMultipliedBy(const TSystemVectorRef& Ax, const TSystemVectorCRef& x) const
        {
            for_each_connected_factor([&](const ConnectedFactor<TValue, VarDim>& factor)
            {
                factor.AddInSiteQuadraticCoefficientsMultipliedBy(Ax, x);
            });
        }

        void AddInDiagonal(TSystemVectorRef& invDiag) const
        {
            for_each_connected_factor([&](const ConnectedFactor<TValue, VarDim>& factor)
            {
                factor.AddInDiagonal(invDiag);
            });
        }

        void AddInDiagonal(TBlockDiagonalRef& invDiag) const
        {
            for_each_connected_factor([&](const ConnectedFactor<TValue, VarDim>& factor)
            {
                factor.AddInDiagonal(invDiag);
            });
        }
    };

    // Similar to the 'Subgraph' class, but allows to specify precomputed 'weights images' that
    // contain pointers to the relevant weights nodes for each factor instance. This avoids having
    // to sort the data points into the leaves of the regression trees over and over again, if
    // we perform multiple iterations over the subgraphs of an image.
    template< typename TTraits >
    class PrecomputedSubgraph
    {
    public:
        static const int                           VarDim = TTraits::UnaryGroundLabel::Size;
        typedef typename TTraits::ValueType        TValue;
        typedef typename TTraits::UnaryGroundLabel TUnaryGroundLabel;
        typedef SystemVectorRef<TValue, VarDim>    TSystemVectorRef;
        typedef BlockDiagonalRef<TValue, VarDim>   TBlockDiagonalRef;
        typedef SystemVectorCRef<TValue, VarDim>   TSystemVectorCRef;
        typedef SystemMatrixRow<TValue, VarDim>    TSystemMatrixRow;

    private:
        const Vector2D<int>& j;                                    // The center variable
        const typename TTraits::PreProcessType& prep;
        const typename TTraits::UnaryFactorTypeVector& Us;         // Reference to the unary factor types
        const typename TTraits::PairwiseFactorTypeVector& Ps;      // Reference to the pairwise factor types
        const typename TTraits::UnaryWeightsImageVector& Uws;      // Reference to the precomputed 'weights images' of the unary types
        const typename TTraits::PairwiseWeightsImageVector& Pws;   // Same for the pairwise types.
        const typename TTraits::UnaryBasisImageVector& Ubs;        // Reference to the precomputed 'basis images' of the unary types
        const typename TTraits::PairwiseBasisImageVector& Pbs;     // Same for the pairwise types.

        // Applies the given functor to all instances of connected factors, drawing on the pre-computed 'weights images'.
        template<typename TOp>
        void for_each_connected_factor(const TOp& op) const
        {
            for(size_t u = 0; u < Us.size(); ++u)
                Us[u].ForEachConnectedInstance(j, prep, Uws[u], Ubs[u], op);

            for(size_t p = 0; p < Ps.size(); ++p)
                Ps[p].ForEachConnectedInstance(j, prep, Pws[p], Pbs[p], op);
        }

    public:
        PrecomputedSubgraph(const Vector2D<int>& j_, const typename TTraits::PreProcessType& prep_,
                            const typename TTraits::UnaryFactorTypeVector& Us_, const typename TTraits::PairwiseFactorTypeVector& Ps_,
                            const typename TTraits::UnaryWeightsImageVector& Uws_, const typename TTraits::PairwiseWeightsImageVector& Pws_,
                            const typename TTraits::UnaryBasisImageVector& Ubs_, const typename TTraits::PairwiseBasisImageVector& Pbs_)
            : j(j_), prep(prep_), Us(Us_), Ps(Ps_), Uws(Uws_), Pws(Pws_), Ubs(Ubs_), Pbs(Pbs_)
        {
        }

        // Returns the x coordinate of the center variable
        int PosX() const
        {
            return j.x;
        }

        // Returns the y coordinate of the center variable
        int PosY() const
        {
            return j.y;
        }

        // For each connected factor, add the coefficients that are relevant to the righthand side of the sparse linear system
        // determined by the RTF model.
        void AddInSiteLinearCoefficients(const TSystemVectorRef& b) const
        {
            for_each_connected_factor([&](const ConnectedFactor<TValue, VarDim>& factor)
            {
                factor.AddInSiteLinearCoefficients(b);
            });
        }

        // Stores the product of the precision block(s) describing the interactions between the center variable of the subgraph
        // and the variables of all connected factors with the given vector 'x' in the given vector 'Ax'. This can be used to
        // multiply the sparse matrix of the linear system by a dense vector without ever actually instantiating it.
        void AddInSiteQuadraticCoefficientsMultipliedBy(const TSystemVectorRef& Ax, const TSystemVectorCRef& x) const
        {
            for_each_connected_factor([&](const ConnectedFactor<TValue, VarDim>& factor)
            {
                factor.AddInSiteQuadraticCoefficientsMultipliedBy(Ax, x);
            });
        }

        // For each connected factor, adds the precision block(s) describing the the interactions between the center variable
        // of the subgraph and the variables of the factor to the given row of the sparse linear system.
        void AddPrecisionBlocks(TSystemMatrixRow& row) const
        {
            for_each_connected_factor([&](const ConnectedFactor<TValue, VarDim>& factor)
            {
                factor.AddPrecisionBlocks(row);
            });
        }

        void AddInDiagonal(TSystemVectorRef& invDiag) const
        {
            for_each_connected_factor([&](const ConnectedFactor<TValue, VarDim>& factor)
            {
                factor.AddInDiagonal(invDiag);
            });
        }

        void AddInDiagonal(TBlockDiagonalRef& invDiag) const
        {
            for_each_connected_factor([&](const ConnectedFactor<TValue, VarDim>& factor)
            {
                factor.AddInDiagonal(invDiag);
            });
        }
    };

    template<typename TTraits, bool Subsample>
    size_t num_subgraphs(const typename TTraits::UnaryFactorTypeVector& Us,
                         const typename TTraits::PairwiseFactorTypeVector& Ps,
                         const typename TTraits::DataSampler& traindb)
    {
        size_t ret = 0;
        Compute::for_each_conditioned_subgraph<TTraits, Subsample, true>(traindb, Us, Ps, [&](const ConditionedSubgraph<TTraits>&)
        {
            #pragma omp atomic
            ret++;
        });
        return ret;
    }

    // Iterates over all conditioned subgraphs of a given image, using the precomputed 'weights images'.
    template<typename TTraits>
    void
    for_each_precomputed_conditioned_subgraph(const typename TTraits::PreProcessType& prep,
            const ImageRefC<typename TTraits::UnaryGroundLabel>& y,
            const typename TTraits::UnaryFactorTypeVector& Us,
            const typename TTraits::PairwiseFactorTypeVector& Ps,
            const typename TTraits::UnaryWeightsImageVector& Uws,
            const typename TTraits::PairwiseWeightsImageVector& Pws,
            const typename TTraits::UnaryBasisImageVector& Ubs,
            const typename TTraits::PairwiseBasisImageVector& Pbs,
            const std::function<void (const PrecomputedConditionedSubgraph<TTraits>&)>& op)
    {
        const int cx = y.Width(), cy = y.Height();
        Vector2D<int> i;

        for(i.y = 0; i.y < cy; ++i.y)
        {
            for(i.x = 0; i.x < cx; ++i.x)
            {
                op(PrecomputedConditionedSubgraph<TTraits>(i, prep, y, Us, Ps, Uws, Pws, Ubs, Pbs));
            }
        }
    }

    // Type metafunctions that delegate to the actual implementation of the above functions
    // based on the template parameters.
    namespace Detail
    {
        template<typename TTraits, bool Subsample = false, bool Parallel = false>
        struct for_each_conditioned_subgraph
        {
            for_each_conditioned_subgraph(const typename TTraits::DataSampler& traindb,
                                          const typename TTraits::UnaryFactorTypeVector& Us,
                                          const typename TTraits::PairwiseFactorTypeVector& Ps,
                                          const std::function<void (const ConditionedSubgraph<TTraits>&)>& op)
            {
                for(size_t i = 0; i < traindb.GetImageCount(); ++i)
                {
                    const auto prep   = TTraits::Feature::PreProcess(traindb.GetInputImage(i));
                    const auto ground = traindb.GetGroundTruthImage(i);
                    const int  cx     = ground.Width(), cy = ground.Height();

                    for(int y = 0; y < cy; ++y)
                    {
                        for(int x = 0; x < cx; ++x)
                            op(ConditionedSubgraph<TTraits>(Vector2D<int>(x, y), prep, ground, Us, Ps));
                    }
                }
            }
        };

        template<typename TTraits>
        struct for_each_conditioned_subgraph<TTraits, true, false>
        {
            for_each_conditioned_subgraph(const typename TTraits::DataSampler& traindb,
                                          const typename TTraits::UnaryFactorTypeVector& Us,
                                          const typename TTraits::PairwiseFactorTypeVector& Ps,
                                          const std::function<void (const ConditionedSubgraph<TTraits>&)>& op)
            {
                for(size_t i = 0; i < traindb.GetImageCount(); ++i)
                {
                    const auto samples = traindb.GetSubsampledVariables(i);
                    const auto prep    = TTraits::Feature::PreProcess(traindb.GetInputImage(i));
                    const auto ground  = traindb.GetGroundTruthImage(i);

                    for(int s = 0; s < samples.size(); ++s)
                    {
                        op(ConditionedSubgraph<TTraits>(samples[s], prep, ground, Us, Ps));
                    }
                }
            }
        };

        template<typename TTraits>
        struct for_each_conditioned_subgraph<TTraits, false, true>
        {
            for_each_conditioned_subgraph(const typename TTraits::DataSampler& traindb,
                                          const typename TTraits::UnaryFactorTypeVector& Us,
                                          const typename TTraits::PairwiseFactorTypeVector& Ps,
                                          const std::function<void (const ConditionedSubgraph<TTraits>&)>& op)
            {
                for(size_t i = 0; i < traindb.GetImageCount(); ++i)
                {
                    const auto prep   = TTraits::Feature::PreProcess(traindb.GetInputImage(i));
                    const auto ground = traindb.GetGroundTruthImage(i);
                    const int  cx     = ground.Width(), cy = ground.Height();

                    #pragma omp parallel for
                    for(int y = 0; y < cy; ++y)
                    {
                        for(int x = 0; x < cx; ++x)
                            op(ConditionedSubgraph<TTraits>(Vector2D<int>(x, y), prep, ground, Us, Ps));
                    }
                }
            }
        };

        template<typename TTraits>
        struct for_each_conditioned_subgraph<TTraits, true, true>
        {
            for_each_conditioned_subgraph(const typename TTraits::DataSampler& traindb,
                                          const typename TTraits::UnaryFactorTypeVector& Us,
                                          const typename TTraits::PairwiseFactorTypeVector& Ps,
                                          const std::function<void (const ConditionedSubgraph<TTraits>&)>& op)
            {
                for(size_t i = 0; i < traindb.GetImageCount(); ++i)
                {
                    const auto samples = traindb.GetSubsampledVariables(i);
                    const auto prep    = TTraits::Feature::PreProcess(traindb.GetInputImage(i));
                    const auto ground  = traindb.GetGroundTruthImage(i);

                    #pragma omp parallel for
                    for(int s = 0; s < samples.size(); ++s)
                    {
                        op(ConditionedSubgraph<TTraits>(samples[s], prep, ground, Us, Ps));
                    }
                }
            }
        };

        template<typename TTraits, bool Subsample = false, bool Parallel = true>
        struct for_each_conditioned_subgraph_with_index
        {
            for_each_conditioned_subgraph_with_index(const typename TTraits::DataSampler& traindb,
                    const typename TTraits::UnaryFactorTypeVector& Us,
                    const typename TTraits::PairwiseFactorTypeVector& Ps,
                    const std::function<void (size_t id, const ConditionedSubgraph<TTraits>&)>& op)
            {
                size_t offset = 0;

                for(size_t i = 0; i < traindb.GetImageCount(); ++i)
                {
                    const auto prep   = TTraits::Feature::PreProcess(traindb.GetInputImage(i));
                    const auto ground = traindb.GetGroundTruthImage(i);
                    const int  cx     = ground.Width(), cy = ground.Height();

                    #pragma omp parallel for
                    for(int y = 0; y < cy; ++y)
                    {
                        for(int x = 0; x < cx; ++x)
                            op(offset + y * cx + x, ConditionedSubgraph<TTraits>(Vector2D<int>(x, y), prep, ground, Us, Ps));
                    }

                    offset += cx * cy;
                }
            }
        };

        template<typename TTraits>
        struct for_each_conditioned_subgraph_with_index<TTraits, false, false>
        {
            for_each_conditioned_subgraph_with_index(const typename TTraits::DataSampler& traindb,
                    const typename TTraits::UnaryFactorTypeVector& Us,
                    const typename TTraits::PairwiseFactorTypeVector& Ps,
                    const std::function<void (size_t id, const ConditionedSubgraph<TTraits>&)>& op)
            {
                size_t offset = 0;

                for(size_t i = 0; i < traindb.GetImageCount(); ++i)
                {
                    const auto prep   = TTraits::Feature::PreProcess(traindb.GetInputImage(i));
                    const auto ground = traindb.GetGroundTruthImage(i);
                    const int  cx     = ground.Width(), cy = ground.Height();

                    for(int y = 0; y < cy; ++y)
                    {
                        for(int x = 0; x < cx; ++x)
                            op(offset + y * cx + x, ConditionedSubgraph<TTraits>(Vector2D<int>(x, y), prep, ground, Us, Ps));
                    }

                    offset += cx * cy;
                }
            }
        };

        template<typename TTraits>
        struct for_each_conditioned_subgraph_with_index<TTraits, true, true>
        {
            for_each_conditioned_subgraph_with_index(const typename TTraits::DataSampler& traindb,
                    const typename TTraits::UnaryFactorTypeVector& Us,
                    const typename TTraits::PairwiseFactorTypeVector& Ps,
                    const std::function<void (size_t id, const ConditionedSubgraph<TTraits>&)>& op)
            {
                size_t offset = 0;

                for(size_t i = 0; i < traindb.GetImageCount(); ++i)
                {
                    const auto samples = traindb.GetSubsampledVariables(i);
                    const auto prep    = TTraits::Feature::PreProcess(traindb.GetInputImage(i));
                    const auto ground  = traindb.GetGroundTruthImage(i);

                    #pragma omp parallel for
                    for(int s = 0; s < samples.size(); ++s)
                    {
                        op(offset + s, ConditionedSubgraph<TTraits>(samples[s], prep, ground, Us, Ps));
                    }

                    offset += samples.size();
                }
            }
        };

        template<typename TTraits>
        struct for_each_conditioned_subgraph_with_index<TTraits, true, false>
        {
            for_each_conditioned_subgraph_with_index(const typename TTraits::DataSampler& traindb,
                    const typename TTraits::UnaryFactorTypeVector& Us,
                    const typename TTraits::PairwiseFactorTypeVector& Ps,
                    const std::function<void (size_t id, const ConditionedSubgraph<TTraits>&)>& op)
            {
                size_t offset = 0;

                for(size_t i = 0; i < traindb.GetImageCount(); ++i)
                {
                    const auto samples = traindb.GetSubsampledVariables(i);
                    const auto prep    = TTraits::Feature::PreProcess(traindb.GetInputImage(i));
                    const auto ground  = traindb.GetGroundTruthImage(i);

                    for(int s = 0; s < samples.size(); ++s)
                        op(offset + s, ConditionedSubgraph<TTraits>(samples[s], prep, ground, Us, Ps));

                    offset += samples.size();
                }
            }
        };

        template<typename TTraits, bool Parallel = false>
        struct for_each_precomputed_subgraph
        {
            for_each_precomputed_subgraph(const typename TTraits::PreProcessType& prep, const int cx, const int cy,
                                          const typename TTraits::UnaryFactorTypeVector& Us,
                                          const typename TTraits::PairwiseFactorTypeVector& Ps,
                                          const typename TTraits::UnaryWeightsImageVector& Uws,
                                          const typename TTraits::PairwiseWeightsImageVector& Pws,
                                          const typename TTraits::UnaryBasisImageVector& Ubs,
                                          const typename TTraits::PairwiseBasisImageVector& Pbs,
                                          std::function<void (const PrecomputedSubgraph<TTraits>&)> op)
            {
                Vector2D<int> i;

                for(i.y = 0; i.y < cy; ++i.y)
                    for(i.x = 0; i.x < cx; ++i.x)
                        op(PrecomputedSubgraph<TTraits>(i, prep, Us, Ps, Uws, Pws, Ubs, Pbs));
            }
        };

        template<typename TTraits>
        struct for_each_precomputed_subgraph<TTraits, true>
        {
            for_each_precomputed_subgraph(const typename TTraits::PreProcessType& prep, const int cx, const int cy,
                                          const typename TTraits::UnaryFactorTypeVector& Us,
                                          const typename TTraits::PairwiseFactorTypeVector& Ps,
                                          const typename TTraits::UnaryWeightsImageVector& Uws,
                                          const typename TTraits::PairwiseWeightsImageVector& Pws,
                                          const typename TTraits::UnaryBasisImageVector& Ubs,
                                          const typename TTraits::PairwiseBasisImageVector& Pbs,
                                          std::function<void (const PrecomputedSubgraph<TTraits>&)> op)
            {
                #pragma omp parallel for
                for(int y = 0; y < cy; ++y)
                {
                    for(int x = 0; x < cx; ++x)
                        op(PrecomputedSubgraph<TTraits>(Vector2D<int>(x, y), prep, Us, Ps, Uws, Pws, Ubs, Pbs));
                }
            }
        };

        template<typename TTraits, bool Parallel = false>
        struct for_each_subgraph
        {
            for_each_subgraph(const typename TTraits::PreProcessType& prep,
                              const int cx, const int cy,
                              const typename TTraits::UnaryFactorTypeVector& Us,
                              const typename TTraits::PairwiseFactorTypeVector& Ps,
                              std::function<void (const Subgraph<TTraits>&)> op)
            {
                Vector2D<int> i;

                for(i.y = 0; i.y < cy; ++i.y)
                    for(i.x = 0; i.x < cx; ++i.x)
                        op(Subgraph<TTraits>(i, prep, cx, cy, Us, Ps));
            }
        };

        template<typename TTraits>
        struct for_each_subgraph<TTraits, true>
        {
            for_each_subgraph(const typename TTraits::PreProcessType& prep,
                              const int cx, const int cy,
                              const typename TTraits::UnaryFactorTypeVector& Us,
                              const typename TTraits::PairwiseFactorTypeVector& Ps,
                              std::function<void (const Subgraph<TTraits>&)> op)
            {
                #pragma omp parallel for
                for(int y = 0; y < cy; ++y)
                {
                    for(int x = 0; x < cx; ++x)
                        op(Subgraph<TTraits>(Vector2D<int>(x, y), prep, cx, cy, Us, Ps));
                }
            }
        };
    } // namespace Detail


    // Iterates over conditioned subgraphs of a given image. The template parameters determine whether
    // only a subsample of the subgraphs is to be processed, and if the subgraphs are to be processed
    // in parallel, respectively.
    template<typename TTraits, bool Subsample, bool Parallel>
    void
    for_each_conditioned_subgraph(const typename TTraits::DataSampler& traindb,
                                  const typename TTraits::UnaryFactorTypeVector& Us,
                                  const typename TTraits::PairwiseFactorTypeVector& Ps,
                                  const std::function<void (const ConditionedSubgraph<TTraits>&)>& op)
    {
        Detail::for_each_conditioned_subgraph<TTraits, Subsample, Parallel>(traindb, Us, Ps, op);
    }

    // Iterates over conditioned subgraphs of a given dataset, associating with each conditioned subgraph
    // a unique ID numbered from 0 to nSubraphs-1. The subgraphs are processed in parallel. The 'Subsample'
    // template parameter determines whether only a subsample of the subgraphs is to be processed. If so,
    // The ID is still guaranteed to be contiguous and ranges from 0 to the number of subraphs in the
    // subsample minus one.
    template<typename TTraits, bool Subsample, bool Parallel>
    void
    for_each_conditioned_subgraph_with_index(const typename TTraits::DataSampler& traindb,
            const typename TTraits::UnaryFactorTypeVector& Us,
            const typename TTraits::PairwiseFactorTypeVector& Ps,
            const std::function<void (size_t id, const ConditionedSubgraph<TTraits>&)>& op)
    {
        Detail::for_each_conditioned_subgraph_with_index<TTraits, Subsample, Parallel>(traindb, Us, Ps, op);
    }

    // Iterates over all subgraphs of a given image, using the precomputed 'weights images'.
    // The template parameter 'parallel' specifies whether the subgraphs are to be processed concurrently.
    template<typename TTraits, bool Parallel>
    void
    for_each_precomputed_subgraph(const typename TTraits::PreProcessType& prep, const int cx, const int cy,
                                  const typename TTraits::UnaryFactorTypeVector& Us,
                                  const typename TTraits::PairwiseFactorTypeVector& Ps,
                                  const typename TTraits::UnaryWeightsImageVector& Uws,
                                  const typename TTraits::PairwiseWeightsImageVector& Pws,
                                  const typename TTraits::UnaryBasisImageVector& Ubs,
                                  const typename TTraits::PairwiseBasisImageVector& Pbs,
                                  const std::function<void (const PrecomputedSubgraph<TTraits>&)>& op)
    {
        Detail::for_each_precomputed_subgraph<TTraits, Parallel>(prep, cx, cy, Us, Ps, Uws, Pws, Ubs, Pbs, op);
    }

    // Same as the above, but without employing precomputed 'weights images'.
    template<typename TTraits, bool Parallel>
    void
    for_each_subgraph(const typename TTraits::PreProcessType& prep,
                      const int cx, const int cy,
                      const typename TTraits::UnaryFactorTypeVector& Us,
                      const typename TTraits::PairwiseFactorTypeVector& Ps,
                      const std::function<void (const Subgraph<TTraits>&)>& op)
    {
        Detail::for_each_subgraph<TTraits, Parallel>(prep, cx, cy, Us, Ps, op);
    }

    // Returns precomputed 'weights images' for each factor type, which contain
    // pointers to the relevant weights of the regression tree for each factor instance
    // of a grid. This avoids having to re-sort the pixels into the leaves of the regression
    // trees if multiple iterations over the pixels of an image a performed.
    template<typename TTraits>
    std::pair < typename TTraits::UnaryWeightsImageVector,
        typename TTraits::PairwiseWeightsImageVector >
        ComputeWeightsImages(const typename TTraits::PreProcessType& prep,
                             const int cx, const int cy,
                             const typename TTraits::UnaryFactorTypeVector& Us,
                             const typename TTraits::PairwiseFactorTypeVector& Ps)
    {
        typename TTraits::UnaryWeightsImageVector Uws(Us.size());
        typename TTraits::PairwiseWeightsImageVector Pws(Ps.size());

        for(size_t u = 0; u < Us.size(); ++u)
            Uws[u] = Us[u].WeightsImage(prep, cx, cy);

        for(size_t p = 0; p < Ps.size(); ++p)
            Pws[p] = Ps[p].WeightsImage(prep, cx, cy);

        return std::make_pair(Uws, Pws);
    }

    // Returns precomputed 'basis images' for each factor type, which contain
    // the relevant basis vectors for each factor instance of a grid. This avoids
    // having to re-compute the basis vectors if multiple iterations over an image
    // are performed.
    template<typename TTraits>
    std::pair < typename TTraits::UnaryBasisImageVector,
        typename TTraits::PairwiseBasisImageVector >
        ComputeBasisImages(const typename TTraits::PreProcessType& prep,
                           const int cx, const int cy,
                           const typename TTraits::UnaryFactorTypeVector& Us,
                           const typename TTraits::PairwiseFactorTypeVector& Ps)
    {
        typename TTraits::UnaryBasisImageVector Ubs(Us.size());
        typename TTraits::PairwiseBasisImageVector Pbs(Ps.size());

        for(size_t u = 0; u < Us.size(); ++u)
            Ubs[u] = Us[u].BasisImage(prep, cx, cy);

        for(size_t p = 0; p < Ps.size(); ++p)
            Pbs[p] = Ps[p].BasisImage(prep, cx, cy);

        return std::make_pair(Ubs, Pbs);
    }
}

#endif // H_RTF_COMPUTE_H
