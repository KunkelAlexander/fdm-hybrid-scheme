#include "IPRInterpolationTables.h"



struct IPRContext {
    IPRContext(size_t N) : size(N), lambda(1.0) {
        printf("Allocated new IPR context of size N = %ld\n", N);

//      complex N^2 array
        changeOfBasisMatrix      = new real[2 * N * N];

//      real array with N polynomials of degree 0 - (N-1) evaluated at 2*N points
        interpolationPolynomials = new real[2 * N * N];

//      allocate memory for permutation
        p = gamer_gsl::permutation_alloc(N);

//      plan FFT of function to be interpolated
        gamer_float_complex* planner_array = (gamer_float_complex*) gamer_fftw_malloc(N * sizeof(gamer_float_complex));
        fftwPlan                           = create_gamer_fftw_1d_forward_c2c_plan(N, planner_array);
        gamer_fftw_free(planner_array);

//      compute change of basis matrix
        computeChangeOfBasisMatrix(changeOfBasisMatrix, N, lambda);

//      evaluate Gegenbauer polynomials at 2*N interpolation points
        computeInterpolationPolynomials(lambda, N, interpolationPolynomials);
    }

    ~IPRContext() {
        printf("Deallocated new IPR context of size N = %ld\n", size);

        delete [] changeOfBasisMatrix;
        delete [] interpolationPolynomials;
        gamer_gsl::permutation_free(p);
        destroy_gamer_complex_fftw_plan(fftwPlan);
    }

    double polynomial(int n, double lambda, double x) const
    {
        return gsl_sf_gegenpoly_n(n, lambda, x);
    }

    void computeChangeOfBasisMatrix(real *W, int N, double lambda) const
    {
        double       * input  = (double* )       fftw_malloc( N * 1 * sizeof( double )       );
        fftw_complex * output = (fftw_complex* ) fftw_malloc( N * N * sizeof( fftw_complex ) );

        for (int n = 0; n < N; ++n)
        {
//          evaluate polynomial in interval [-1, 1]
            for (int j = 0; j < N; ++j)
            {
                input[j] = polynomial( n, lambda, -1 + j / ((double)N / 2.0) );
            }

//          create FFTW plan for double-to-complex FFT
            fftw_plan plan = fftw_plan_dft_r2c_1d(N, input, &output[n * N], FFTW_ESTIMATE);

//          compute forward FFT
            fftw_execute(plan);

//          destroy the FFTW plan
            fftw_destroy_plan(plan);

//          real-to-complex FFT maps from n to n/2 +1 because of symmetry of real FFT
//          fill up remaining n - n/2 - 1 values with complex conjugate to obtain square matrix
            for (int j = (N / 2 + 1); j < N; ++j)
            {
                output[ n * N + j ][0] =   output[ n * N + N - j ][0];
                output[ n * N + j ][1] = - output[ n * N + N - j ][1];
            }
        }

//      transpose output array
        gsl_matrix_complex_view output_view = gsl_matrix_complex_view_array((double* ) output, N, N);
        gsl_matrix_complex_transpose(&output_view.matrix);


//      LU decomposition with pivot
        int signum;
        gamer_gsl::linalg_complex_LU_decomp(&output_view.matrix, p, &signum);


//      write output array to W with precision of real type
        for (int i = 0; i < N * N * 2; ++i) {
            W[i] = (real) ((double*) output)[i];
        }


        fftw_free(input);
        fftw_free(output);
    }

    void computeInterpolationPolynomials(double lambda, size_t N, real *poly) const
    {
//      iterate over polynomials
        for (size_t polyOrder = 0; polyOrder < N; ++polyOrder)
        {
//          iterate over cells in interval [-1, 1] and evaluate polynomial
            for (size_t cell = 0; cell < 2 * N; ++cell)
            {
                poly[polyOrder * N * 2 + cell] = (real) polynomial(polyOrder, lambda, -1 + 1.0 / (2 * N) + cell * 1.0 / N);
            }
        }
    }

    const real* getInterpolationPolynomials() const {
        return interpolationPolynomials;
    }

    const real* getChangeOfBasisMatrix() const {
        return changeOfBasisMatrix;
    }

    gamer_real_fftw_plan getFFTWPlan() const {
        return fftwPlan;
    }

    const gamer_gsl::permutation* getPermutation() const {
        return p;
    }

private:
    size_t                  size;
    real*                   changeOfBasisMatrix;
    real*                   interpolationPolynomials;
    gamer_real_fftw_plan    fftwPlan;
    double                  lambda;
    gamer_gsl::permutation* p;
};

class IPR {
public:
    enum InterpolationMode { InterpolateReal, InterpolateImag };

    void interpolateComplex(gamer_float_complex *input, real *re_output, real *im_output, real* workspace, size_t N, size_t ghostBoundary) {

        if (contexts.find(N) == contexts.end()) {
            contexts.emplace(N, N);
        }

        IPRContext& c = contexts.at(N);

//      compute forward FFT
        gamer_fftw_c2c   (c.getFFTWPlan(), (gamer_float_complex *) input);

//      initialise workspace with change of basis matrix
        memcpy(workspace, c.getChangeOfBasisMatrix(), 2 * N * N * sizeof(real));

        //gaussWithTruncation(workspace, (real*) input, N, IPR_TruncationThreshold, c.getPermutation());
        gaussWithoutTruncation(workspace, (real*) input, N, IPR_TruncationThreshold, c.getPermutation());

        interpolateFunction(input, re_output, N, ghostBoundary, c.getInterpolationPolynomials(), InterpolationMode::InterpolateReal );
        interpolateFunction(input, im_output, N, ghostBoundary, c.getInterpolationPolynomials(), InterpolationMode::InterpolateImag );
    }

    void interpolateReal(real* re_input, real *re_output, real* workspace, size_t N, size_t ghostBoundary) {

        if (contexts.find(N) == contexts.end()) {
            contexts.emplace(N, N);
        }

        IPRContext& c = contexts.at(N);

//      compute forward FFT
        gamer_float_complex* input = (gamer_float_complex*) gamer_fftw_malloc(sizeof(gamer_float_complex) * N);

        for (size_t i = 0; i < N; ++i) {
            c_re(input[i]) = re_input[i];
            c_im(input[i]) = 0.0;
        }

//      compute forward FFT
        gamer_fftw_c2c   (c.getFFTWPlan(), input);

//      initialise workspace with change of basis matrix
        memcpy(workspace, c.getChangeOfBasisMatrix(), 2 * N * N * sizeof(real));

        gaussWithTruncation(workspace, (real*) input, N, IPR_TruncationThreshold, c.getPermutation());

        interpolateFunction(input, re_output, N, ghostBoundary, c.getInterpolationPolynomials(), InterpolationMode::InterpolateReal );

        gamer_fftw_free(input);
    }



    void gaussWithTruncation(real *A, real *B, size_t N, real truncationThreshold, const gamer_gsl::permutation* p) const
    {
        /*
        Solve Ax = B using Gaussian elimination and LU decomposition with truncation after the forward substitution for stability of IPR
        */

        // create matrix views for input vector and matrix
        gamer_gsl::matrix_complex_view A_view = gamer_gsl::matrix_complex_view_array(A, N, N);
        gamer_gsl::matrix_complex          *a = &A_view.matrix;

        gamer_gsl::vector_complex_view B_view = gamer_gsl::vector_complex_view_array(B, N);
        gamer_gsl::vector_complex          *b = &B_view.vector;


        // apply permutation p to b
        gamer_gsl::permute_vector_complex(p, b);

        // forward substitution to solve for Ly = B
        // note that L is stored as longer triangular part of a without the diagonal elements
        // the diagonal elements are 1
        //
        //   (100)(y1) = (B1)
        //   (x10)(y2) = (B2)
        //   (yz1)(y3) = (B3)
        //
        // y1                    = B1
        // y1 * x + y2           = B2 -> y2 = B2 - y1 * x
        // y1 * y + y2 * z + y3  = B3 -> y3 = B3 - y1 * y - y2 * z
        //
        for (size_t m = 1; m < N; ++m)
        {
            for (size_t n = 0; n < m; ++n)
            {
                gamer_gsl::vector_complex_set(b, m, gamer_gsl::complex_sub(gamer_gsl::vector_complex_get(b, m), gamer_gsl::complex_mul(gamer_gsl::vector_complex_get(b, n), gamer_gsl::matrix_complex_get(a, m, n))));
            }
        }

        // truncation for IPR
        // necessary for convergence at large
        // ( see Short Note: On the numerical convergence with the inverse polynomial reconstruction method for the resolution of the Gibbs phenomenon, Jung and Shizgal 2007)
        for (size_t m = 0; m < N; ++m)
        {
            if (gamer_gsl::complex_abs(gamer_gsl::vector_complex_get(b, m)) < truncationThreshold)
            {
                gamer_gsl::vector_complex_set(b, m, {0, 0});
            }
        }

        // backward substitution to solve for y = Ux
        //
        //   (uwk)(x1) = (y1)
        //   (0yx)(x2) = (y2)
        //   (00z)(x3) = (y3)
        //
        //                       x3 = y3
        //          y * x2 + x * x3 = y2 -> x2 = (y2 - x * x3         ) / y
        // u * x1 + w * x2 * k * x3 = y1 -> x3 = (y1 - u * x1 - w * x2) / k

        for (int m = N - 1; m >= 0; --m)
        {
            for (int n = N - 1; n > m; --n)
            {
                gamer_gsl::vector_complex_set(b, m, gamer_gsl::complex_sub(gamer_gsl::vector_complex_get(b, m), gamer_gsl::complex_mul(gamer_gsl::vector_complex_get(b, n), gamer_gsl::matrix_complex_get(a, m, n))));
            }
            gamer_gsl::vector_complex_set(b, m, gamer_gsl::complex_div(gamer_gsl::vector_complex_get(b, m), gamer_gsl::matrix_complex_get(a, m, m)));
        }

    }


    void gaussWithoutTruncation(real *A, real *B, size_t N, real truncationThreshold, const gamer_gsl::permutation* p) const
    {
        /*
        Solve Ax = B using Gaussian elimination and LU decomposition with truncation after the forward substitution for stability of IPR
        */

        // create matrix views for input vector and matrix
        gamer_gsl::matrix_complex_view A_view = gamer_gsl::matrix_complex_view_array(A, N, N);
        gamer_gsl::matrix_complex          *a = &A_view.matrix;

        gamer_gsl::vector_complex_view B_view = gamer_gsl::vector_complex_view_array(B, N);
        gamer_gsl::vector_complex          *b = &B_view.vector;
        //gamer_gsl::linalg_complex_LU_svx(a, p, b);

    }


    void interpolateFunction(gamer_float_complex* g, real* output, size_t N, size_t ghostBoundary, const real* poly, InterpolationMode mode) const
    {
//      iterate over cells
        for (size_t cell = 0; cell < 2 * N - 4 * ghostBoundary + 1; ++cell)
        {
            output[cell] = 0;

            for (size_t polyOrder = 0; polyOrder < N; ++polyOrder)
            {
                switch(mode) {
                    case InterpolationMode::InterpolateReal:
                        output[cell] += c_re(g[polyOrder]) * poly[polyOrder * N * 2 + cell - 1 + 2 * ghostBoundary];
                        break;
                    case InterpolationMode::InterpolateImag:
                        output[cell] += c_im(g[polyOrder]) * poly[polyOrder * N * 2 + cell - 1 + 2 * ghostBoundary];
                        break;
                }
            }
        }
    } // FUNCTION : interpolateFunction


private:
    std::unordered_map<size_t, IPRContext> contexts;
};


const size_t ghostBoundary = 2;


struct NewIPRContext {
    NewIPRContext(size_t N) : size(N), lambda(1.0) {
        printf("Allocated new IPR context of size N = %ld\n", N);

//      complex N^2 array
        changeOfBasisMatrix      = new double[2 * N * N];

//      real array with N polynomials of degree 0 - (N-1) evaluated at 2*N points
        interpolationPolynomials = new double[2 * (2 * N) * N];

//      evaluation matrix
        evaluationMatrix         = new double[2 * (2 * N) * N];

//      allocate memory for permutation
        p1 = gsl_permutation_alloc(N);
        p2 = gsl_permutation_alloc(N);

//      plan FFT of function to be interpolated
        gamer_float_complex* planner_array = (gamer_float_complex*) fftw_malloc(N * sizeof(gamer_float_complex));
        fftwPlan                           = create_gamer_fftw_1d_forward_c2c_plan(N, planner_array);
        gamer_fftw_free(planner_array);

//      compute W
        computeChangeOfBasisMatrix(changeOfBasisMatrix, N, lambda);

//      compute LU decomposition of W
        int signum;
        computeLUDecomposition(changeOfBasisMatrix, N, p1, &signum);

//      compute interpolation polynomials
        computeInterpolationPolynomials(lambda, N, interpolationPolynomials);

//
        computeEvaluationMatrix(evaluationMatrix, changeOfBasisMatrix, interpolationPolynomials, N, p2, &signum);


    }

    ~NewIPRContext() {
        printf("Deallocated new IPR context of size N = %ld\n", size);

        delete [] changeOfBasisMatrix;
        delete [] interpolationPolynomials;
        delete [] evaluationMatrix;
        gsl_permutation_free(p1);
        gsl_permutation_free(p2);
        fftw_destroy_plan(fftwPlan);
    }

    double polynomial(int n, double lambda, double x) const
    {
        return gsl_sf_gegenpoly_n(n, lambda, x);
    }

    void computeChangeOfBasisMatrix(double* W, int N, double lambda) const
    {
        double*       input  = (double* )       fftw_malloc( N * 1 * sizeof( double )       );
        fftw_complex* output = (fftw_complex* ) W;

        for (int n = 0; n < N; ++n)
        {
//          evaluate polynomial in interval [-1, 1]
            for (int j = 0; j < N; ++j)
            {
                input[j] = polynomial( n, lambda, -1 + j / ((double)N / 2.0) );
            }

//          create FFTW plan for double-to-complex FFT
            fftw_plan plan = fftw_plan_dft_r2c_1d(N, input, &output[n * N], FFTW_ESTIMATE);

//          compute forward FFT
            fftw_execute(plan);

//          destroy the FFTW plan
            fftw_destroy_plan(plan);

//          real-to-complex FFT maps from n to n/2 +1 because of symmetry of real FFT
//          fill up remaining n - n/2 - 1 values with complex conjugate to obtain square matrix
            for (int j = (N / 2 + 1); j < N; ++j)
            {
                output[ n * N + j ][0] =   output[ n * N + N - j ][0];
                output[ n * N + j ][1] = - output[ n * N + N - j ][1];
            }
        }

//      transpose output array
        gsl_matrix_complex_view output_view = gsl_matrix_complex_view_array((double* ) output, N, N);
        gsl_matrix_complex_transpose(&output_view.matrix);

        fftw_free(input);
    }

    void computeLUDecomposition(double* input, size_t N, gsl_permutation* p, int* signum) {
        gsl_matrix_complex_view input_view = gsl_matrix_complex_view_array(input, N, N);

        gsl_linalg_complex_LU_decomp(&input_view.matrix, p, signum);
    }

    void computeEvaluationMatrix(double* transformationMatrix, const double* changeOfBasisMatrix, const double* interpolationPolynomials, size_t N, gsl_permutation* p, int* signum) {

        double* in         = new real[2 * N * N];
        double* out        = new real[2 * N * N];
        memcpy(in, changeOfBasisMatrix, 2* N * N * sizeof(double));

        gsl_matrix_complex_view input_view  = gsl_matrix_complex_view_array(in,  N, N);
        gsl_matrix_complex_view output_view = gsl_matrix_complex_view_array(out, N, N);
        gsl_matrix_complex_const_view poly_view               = gsl_matrix_complex_const_view_array(interpolationPolynomials, 2 * N, N);
        gsl_matrix_complex_view       trans_view              = gsl_matrix_complex_view_array      (transformationMatrix,     2 * N, N);



//      set lower triangular part of matrix to zero
        for (size_t m = 1; m < N; ++m)
        {
            for (size_t n = 0; n < m; ++n)
            {
                gsl_matrix_complex_set(&input_view.matrix, m, n, {0, 0});
            }
        }

        gsl_linalg_complex_LU_decomp(&input_view.matrix, p, signum);


//      invert upper triangular part
        gsl_linalg_complex_LU_invert(&input_view.matrix, p, &output_view.matrix);

//      apply permutation
        //gsl_permute_matrix_complex(p, &output_view.matrix);


//      multiply inverted U by polynomials matrix to obtain transformation matrix
        gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, {1.0, 0.0}, &poly_view.matrix, &output_view.matrix, {0.0, 0.0}, &trans_view.matrix);

        delete[] in;
        delete[] out;
    }

    void computeInterpolationPolynomials(double lambda, size_t N, real *poly) const
    {
//      iterate over cells in interval [-1, 1] and evaluate polynomial
        for (size_t cell = 0; cell < 2 * N; ++cell)
        {
//      iterate over polynomials
            for (size_t polyOrder = 0; polyOrder < N; ++polyOrder)
            {
                poly[((cell * N) + polyOrder) * 2     ] = (real) polynomial(polyOrder, lambda, -1 + 1.0 / (2 * N) + (cell + 1 + 2 * (ghostBoundary - 1)) * 1.0 / N);
                poly[((cell * N) + polyOrder) * 2 + 1 ] = 0;
            }
        }
    }

    const real* getInterpolationPolynomials() const {
        return interpolationPolynomials;
    }

    const real* getChangeOfBasisMatrix() const {
        return changeOfBasisMatrix;
    }
    const real* getEvaluationMatrix() const {
        return evaluationMatrix;
    }

    gamer_real_fftw_plan getFFTWPlan() const {
        return fftwPlan;
    }

    const gamer_gsl::permutation* getFirstPermutation() const {
        return p1;
    }

    const gamer_gsl::permutation* getSecondPermutation() const {
        return p2;
    }

private:
    size_t                  size;
    real*                   changeOfBasisMatrix;
    real*                   interpolationPolynomials;
    real*                   evaluationMatrix;
    gamer_real_fftw_plan    fftwPlan;
    double                  lambda;
    gamer_gsl::permutation* p1;
    gamer_gsl::permutation* p2;
};

class NewIPR {
public:
    enum InterpolationMode { InterpolateReal, InterpolateImag };

    void interpolateComplex(gamer_float_complex *input, gamer_float_complex *output, size_t N, size_t ghostBoundary) {

        if (contexts.find(N) == contexts.end()) {
            contexts.emplace(N, N);
        }

        NewIPRContext& c = contexts.at(N);

//      compute forward FFT
        gamer_fftw_c2c   (c.getFFTWPlan(), (gamer_float_complex *) input);

        size_t truncationN = N;

        gaussWithTruncation(c.getChangeOfBasisMatrix(), (real*) input, N, truncationN, IPR_TruncationThreshold, c.getFirstPermutation(),  c.getSecondPermutation());

        interpolateFunction(input, output, N, truncationN, ghostBoundary, c.getEvaluationMatrix() );
    }

    void interpolateReal(real* re_input, real *re_output, size_t N, size_t ghostBoundary) {

//      compute forward FFT
        gamer_float_complex* input  = (gamer_float_complex*) gamer_fftw_malloc(sizeof(gamer_float_complex) * N);
        gamer_float_complex* output = (gamer_float_complex*) gamer_fftw_malloc(sizeof(gamer_float_complex) * N * 2);

        for (size_t i = 0; i < N; ++i) {
            c_re(input[i]) = re_input[i];
            c_im(input[i]) = 0.0;
        }

        interpolateComplex(input, output, N, ghostBoundary);

        for (size_t i = 0; i < 2 * N - 2 * ghostBoundary; ++i) {
            re_output[i] = c_re(output[i]);
        }

        gamer_fftw_free(input);
        gamer_fftw_free(output);
    }



    void gaussWithTruncation(const real *LU, real *x, size_t N, size_t& truncationN, real truncationThreshold, const gamer_gsl::permutation* p1, const gamer_gsl::permutation* p2) const
    {

        /*
        Solve Ax = B using Gaussian elimination and LU decomposition with truncation after the forward substitution for stability of IPR
        */

        // create matrix views for input vector and matrix
        gsl_matrix_complex_const_view A_view = gsl_matrix_complex_const_view_array(LU, N, N);
        const gsl_matrix_complex          *a = &A_view.matrix;

        gamer_gsl::vector_complex_view B_view = gamer_gsl::vector_complex_view_array(x, N);
        gamer_gsl::vector_complex          *b = &B_view.vector;


        // apply permutation p to b
        gamer_gsl::permute_vector_complex(p1, b);

        // forward substitution to solve for Ly = B
        gsl_blas_ztrsv(CblasLower, CblasNoTrans, CblasUnit, a, b);


        // truncation for IPR
        // necessary for convergence at large
        // ( see Short Note: On the numerical convergence with the inverse polynomial reconstruction method for the resolution of the Gibbs phenomenon, Jung and Shizgal 2007)
        for (size_t m = 0; m < N; ++m)
        {
            if (gamer_gsl::complex_abs(gamer_gsl::vector_complex_get(b, m)) < truncationThreshold)
            {
                //gamer_gsl::vector_complex_set(b, m, {0, 0});
                truncationN = m;
                break;
            }
        }

        // apply second permutation
        //gamer_gsl::permute_vector_complex(p2, b);

        //gsl_blas_ztrsv(CblasUpper, CblasNoTrans, CblasNonUnit, a, b);


    }


    void interpolateFunction(const gamer_float_complex* g, gamer_float_complex* output, size_t N, size_t truncationN, size_t ghostBoundary, const real* poly) const
    {

        gsl_matrix_complex_const_view A_view          = gsl_matrix_complex_const_view_array((double*) poly, 2 * N, N);
        gsl_matrix_complex_const_view truncatedA_view = gsl_matrix_complex_const_submatrix(&A_view.matrix, 0, 0, 2 * N, truncationN);
        const gsl_matrix_complex          *a = &truncatedA_view.matrix;

        gamer_gsl::vector_complex_view B_view = gamer_gsl::vector_complex_view_array((double*) g, truncationN);
        gamer_gsl::vector_complex          *b = &B_view.vector;

        gamer_gsl::vector_complex_view C_view = gamer_gsl::vector_complex_view_array((double*) output, 2 * N);
        gamer_gsl::vector_complex          *c = &C_view.vector;

        gsl_blas_zgemv(CblasNoTrans, {1.0, 0.0}, a, b, {0.0, 0.0}, c);

    } // FUNCTION : interpolateFunction


private:
    std::unordered_map<size_t, NewIPRContext> contexts;
};



struct FastIPRContext {
    FastIPRContext(size_t N) : size(N), lambda(0.5) {
        printf("Allocated new Fast IPR context of size N = %ld\n", N);

//      plan FFT of function to be interpolated
        gamer_float_complex* planner_array = (gamer_float_complex*) gamer_fftw_malloc(N * sizeof(gamer_float_complex));
        fftwPlan                           = create_gamer_fftw_1d_forward_c2c_plan(N, planner_array);
        gamer_fftw_free(planner_array);

//      load interpolation table from binary file
        char filename[100];
        sprintf(filename, "fdm-hybrid-scheme/demonstration/interpolation/Interpolant_N=%ld_lambda=0.5.bin", N);

        size_t arraySize = 2 * N * 2 * N;

        double* array;
        readBinaryFile(filename, &array, arraySize);

        interpolationPolynomials = (real*)malloc(arraySize * sizeof(real));

        // Print the array for verification
        for (int i = 0; i < arraySize; i++) {
            interpolationPolynomials[i] = (real) array[i];
        }
        //printf("\n");

        free(array);


    }

    ~FastIPRContext() {
        printf("Deallocated new IPR context of size N = %ld\n", size);

        destroy_gamer_complex_fftw_plan(fftwPlan);
        free(interpolationPolynomials);

    }


    void readBinaryFile(const char* filename, double** array, int size)
    {
        FILE* file = fopen(filename, "rb");
        if (file == NULL) {
            printf("Failed to open the file.\n");
            return;
        }

        // Allocate memory for the array
        *array = (double*)malloc(size * sizeof(double));

        // Read the array from the binary file
        fread(*array, sizeof(double), size, file);

        // Close the file
        fclose(file);
    }


    const real* getInterpolationPolynomials() const {
        return interpolationPolynomials;
    }

    gamer_real_fftw_plan getFFTWPlan() const {
        return fftwPlan;
    }

private:
    size_t                  size;
    real*                   interpolationPolynomials;
    gamer_real_fftw_plan    fftwPlan;
    double                  lambda;
};

class FastIPR {
public:
    void interpolateComplex(gamer_float_complex *input, gamer_float_complex *output, size_t N, size_t ghostBoundary) {

        if (contexts.find(N) == contexts.end()) {
            contexts.emplace(N, N);
        }

        FastIPRContext& c = contexts.at(N);

//      compute forward FFT
        gamer_fftw_c2c   (c.getFFTWPlan(), (gamer_float_complex *) input);

        interpolateFunction(input, output, N, ghostBoundary, c.getInterpolationPolynomials() );
    }


    void interpolateFunction(gamer_float_complex* fhat, gamer_float_complex* output, size_t N, size_t ghostBoundary, const real* iprTable) const
    {

        size_t index;

//      iterate over cells
        for (size_t cell = 0; cell < 2 * N - 4 * ghostBoundary + 1; ++cell)
        {
            c_re(output[cell]) = 0;
            c_im(output[cell]) = 0;

            for (size_t polyOrder = 0; polyOrder < N; ++polyOrder)
            {
                index = (cell - 1 + 2 * ghostBoundary) * N + polyOrder;
                gamer_float_complex a = {iprTable[index * 2], iprTable[index * 2 + 1]};
                gamer_float_complex b = {fhat[polyOrder][0], fhat[polyOrder][1]};
                //c_re(b) = c_re(b) < 1e-17 ? 0 : c_re(b);
                //c_im(b) = c_im(b) < 1e-17 ? 0 : c_im(b);
                c_re(output[cell]) += c_re(a) * c_re(b) - c_im(a) * c_im(b);
                c_im(output[cell]) += c_re(a) * c_im(b) + c_im(a) * c_re(b);
            }
        }
    } // FUNCTION : interpolateFunction

private:
    std::unordered_map<size_t, FastIPRContext> contexts;
};


// Quartic interpolation in GAMER
const float QuarticR[ 1 + 2*2 ] = { +35.0/2048.0, -252.0/2048.0, +1890.0/2048.0, +420.0/2048.0, -45.0/2048.0 };
const float QuarticL[ 1 + 2*2 ] = { -45.0/2048.0, +420.0/2048.0, +1890.0/2048.0, -252.0/2048.0, +35.0/2048.0 };

void interpolateQuartic(float *input, float *output, size_t N, size_t ghostBoundary) {
    for (int i = 0; i < N - 2 * ghostBoundary; ++i) {
        real r = 0, l = 0;
        for ( int j = 0; j < 5; ++j ) {
            r += QuarticR[j] * input[i + j];
            l += QuarticL[j] * input[i + j];
        }
        output[i * 2    ] = l;
        output[i * 2 + 1] = r;
    }
}


double testFunc(double x) {
    return sin(10.0 * x);
}

void fastIPR(size_t N, int printInput) {
    FastIPR ipr;

    const int ghostBoundary = 2;

//  linear input test function
    gamer_float_complex* input     = new gamer_float_complex[N];
    gamer_float_complex* output    = new gamer_float_complex[2 * N];

    if (printInput)
        printf("Input: ");

    for (size_t i = 0; i < N; ++i) {
        c_re(input[i]) = testFunc(1.0 * i/N);
        c_im(input[i]) = testFunc(1.0 * i/N);

        if (printInput)
            printf("%f ", c_re(input[i]));
    }

    if (printInput)
        printf("\n");


    ipr.interpolateComplex(input, output, N, ghostBoundary);

//  interpolated input function
    if (printInput) {
        printf("IPR interpolated function g:");
        for (size_t i = 0; i < 2 * N - 4 * ghostBoundary; ++i)
        {
            printf("%f + i %f ", c_re(output[i]), c_im(output[i]));
        }
        printf("\n");
    }



    double error = 0, ana, dx;
    dx = 1.0 / N;
    for (size_t i = 0; i < 2 * N - 4 * ghostBoundary; ++i)
    {
        ana = testFunc((-0.25 + ghostBoundary) * dx + i * dx * 0.5);
        error += abs(c_re(output[i]) - ana);
    }

    printf("Interpolation error: %5.15f\n", error/N);
}

void IPRVsGAMER(size_t N, int printInput) {

    IPR ipr;
    NewIPR newIpr;

    const int ghostBoundary = 2;

//  linear input test function
    real* input     = new real[N];
    real* output    = new real[2 * N];
    float* finput   = new float[N];
    float* foutput  = new float[2 * N];
    real* workspace = new real[2 * N * N];

    if (printInput)
        printf("Input: ");

    for (size_t i = 0; i < N; ++i) {
        input[i] = testFunc(1.0 * i/N);

        if (printInput)
            printf("%f ", input[i]);
    }

    if (printInput)
        printf("\n");


    ipr.interpolateReal(input, output, workspace, N, ghostBoundary);

//  interpolated input function
    if (printInput) {
        printf("IPR interpolated function g:");
        for (size_t i = 0; i < 2 * N - 4 * ghostBoundary; ++i)
        {
            printf("%f ", output[i]);
        }
        printf("\n");
    }



    double error = 0, ana, dx;
    dx = 1.0 / N;
    for (size_t i = 0; i < 2 * N - 4 * ghostBoundary; ++i)
    {
        ana = testFunc((-0.25 + ghostBoundary) * dx + i * dx * 0.5);
        error += abs(output[i] - ana);
    }

    printf("Interpolation error: %5.15f\n", error/N);

    if (printInput)
        printf("Input: ");

    for (size_t i = 0; i < N; ++i) {
        input[i] = testFunc(1.0 * i/N);

        if (printInput)
            printf("%f ", input[i]);
    }

    if (printInput)
        printf("\n");


    newIpr.interpolateReal(input, output, N, ghostBoundary);

//  interpolated input function
    if (printInput) {
        printf("IPR interpolated function g:");
        for (size_t i = 0; i < 2 * N - 4 * ghostBoundary; ++i)
        {
            printf("%f ", output[i]);
        }
        printf("\n");
    }



    error = 0, ana, dx;
    dx = 1.0 / N;
    for (size_t i = 0; i < 2 * N - 4 * ghostBoundary; ++i)
    {
        ana = testFunc((-0.25 + ghostBoundary) * dx + i * dx * 0.5);
        error += abs(output[i] - ana);
    }

    printf("Interpolation error: %5.15f\n", error/N);


//  interpolate using quartic interpolation
    if (printInput)
        printf("Input: ");

    for (size_t i = 0; i < N; ++i) {
        finput[i] = testFunc(1.0 * i/N);

        if (printInput)
            printf("%f ", finput[i]);
    }

    if (printInput)
        printf("\n");

    interpolateQuartic(finput, foutput, N, 2);


//  interpolated input function
    if (printInput) {
        printf("Quartic interpolated function g:");
        for (size_t i = 0; i < 2 * N - 4 * ghostBoundary; ++i)
        {
            printf("%f ", foutput[i]);
        }
        printf("\n");
    }


//  compare interpolation errors

    error = 0;
    dx = 1.0 / N;
    for (size_t i = 0; i < 2 * N - 4 * ghostBoundary; ++i)
    {
        ana = testFunc((-0.25 + ghostBoundary) * dx + i * dx * 0.5);
        error += abs(foutput[i] - ana);
    }

    printf("Interpolation error: %5.15f\n", error/N);


//  fast IPR interpolation
    gamer_float_complex* crinput    = new gamer_float_complex[N];
    gamer_float_complex* coutput    = new gamer_float_complex[2 * N];

    for (size_t i = 0; i < N; ++i) {
        c_re(crinput[i]) = testFunc(1.0 * i/N);
        c_im(crinput[i]) = testFunc(1.0 * i/N);
    }
    FastIPR fastIpr;

    fastIpr.interpolateComplex(crinput, coutput, N, ghostBoundary);


//  interpolated input function
    if (printInput) {
        printf("Fast IPR interpolated function g:");
        for (size_t i = 0; i < 2 * N - 4 * ghostBoundary; ++i)
        {
            printf("%f ", c_re(coutput[i]));
        }
        printf("\n");
    }


//  compare interpolation errors

    error = 0;
    dx = 1.0 / N;
    for (size_t i = 0; i < 2 * N - 4 * ghostBoundary; ++i)
    {
        ana = testFunc((-0.25 + ghostBoundary) * dx + i * dx * 0.5);
        error += abs(c_re(coutput[i]) - ana);
    }

    printf("Interpolation error: %5.15f\n", error/N);




    const int iterations = 10000;

    printf("Now timing %d interpolations using IPR\n", iterations);

    gamer_float_complex* cinput     = new gamer_float_complex[N];
    {
    clock_t begin = clock();

    for (size_t iter = 0; iter < iterations; ++iter) {
        for (size_t i = 0; i < N; ++i) {
            c_re(cinput[i]) = testFunc(1.0 * i/N);
            c_im(cinput[i]) = testFunc(1.0 * i/N);
        }

        newIpr.interpolateComplex(cinput, coutput, N, ghostBoundary);
    }
    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

    printf("Time spent: %f\n", time_spent/2);
    }

        printf("Now timing %d interpolations using quartic interpolation\n", iterations);

    {
    clock_t begin = clock();

    for (size_t iter = 0; iter < iterations; ++iter) {
        for (size_t i = 0; i < N; ++i) {
            finput[i] = testFunc(1.0 * i/N);
        }

        interpolateQuartic(finput, foutput, N, 2);
    }
    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

    printf("Time spent: %f\n", time_spent);
    }

        printf("Now timing %d interpolations using fast IPR\n", iterations);

    {

    //  linear input test function



    clock_t begin = clock();

    for (size_t iter = 0; iter < iterations; ++iter) {
        for (size_t i = 0; i < N; ++i) {
            c_re(cinput[i]) = testFunc(1.0 * i/N);
            c_im(cinput[i]) = testFunc(1.0 * i/N);
        }

        fastIpr.interpolateComplex(cinput, coutput, N, ghostBoundary);
    }
    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

    printf("Time spent: %f\n", time_spent/2.0);
    }


    delete [] input;
    delete [] finput;
    delete [] cinput;
    delete [] output;
    delete [] foutput;
    delete [] workspace;

}


void sineTest(size_t N) {
    IPR ipr;

    const int ghostBoundary = 1;

//  linear input test function
    real* input     = new real[N];
    real* output    = new real[2 * N];
    real* workspace = new real[2 * N * N];

    printf("Input: ");

    for (size_t i = 0; i < N; ++i) {
        input[i] = testFunc(1.0 * i/N);

        printf("%f ", input[i]);
    }

    printf("\n");


    ipr.interpolateReal(input, output, workspace, N, ghostBoundary);

//  interpolated input function
    printf("Interpolated function g:");
    for (size_t i = 0; i < 2 * N - 4 * ghostBoundary; ++i)
    {
        printf("%f ", output[i]);
    }
    printf("\n");


    for (size_t i = 0; i < N; ++i) {
        input[i] = testFunc(1.0 * i/N);

        printf("%f ", input[i]);
    }

    ipr.interpolateReal(input, output, workspace, N, ghostBoundary);

//  interpolated input function
    printf("Interpolated function g:");
    for (size_t i = 0; i < 2 * N - 4 * ghostBoundary; ++i)
    {
        printf("%f ", output[i]);
    }
    printf("\n");

    double error = 0, ana, dx;
    dx = 1.0 / N;
    for (size_t i = 0; i < 2 * N - 4 * ghostBoundary; ++i)
    {
        ana = testFunc(0.75 * dx + i * dx * 0.5);
        error += abs(output[i] - ana);
    }

    printf("Interpolation error: %f\n", error/N);


    delete [] input;
    delete [] output;
    delete [] workspace;

}

void complexTest() {
    IPR ipr;

    const int N = 6;
    const int ghostBoundary = 1;
    real linear[12] = {1., 1., 2., 2., 3., 3., 4., 4., 5., 5., 6., 6.0};


//  linear input test function
    real* input = (real*) gamer_fftw_malloc(N * 2 * sizeof(real));
    for (int i = 0; i < 2 * N; ++i) {
        input[i] = linear[i];
    }
    real reoutput[2 * N] = { };
    real imoutput[2 * N] = { };
    real* workspace = new real[2 * N * N];

    ipr.interpolateComplex((gamer_float_complex*) input, reoutput, imoutput, workspace, N, ghostBoundary);

//  interpolated input function
    std::cout << "Interpolated function g:" << std::endl;
    for (size_t i = 0; i < 2 * N - 4 * ghostBoundary; ++i)
    {
        std::cout << "(" << reoutput[i] << ", " << imoutput[i ] << ")" << std::endl;
    }

    for (int i = 0; i < 2 * N; ++i) {
        input[i] = linear[i];
    }

    ipr.interpolateComplex((gamer_float_complex*) input, reoutput, imoutput, workspace, N, ghostBoundary);

//  interpolated input function
    std::cout << "Interpolated function g:" << std::endl;
    for (size_t i = 0; i < 2 * N - 4 * ghostBoundary; ++i)
    {
        std::cout << "(" << reoutput[i] << ", " << imoutput[i ] << ")" << std::endl;
    }

    delete [] workspace;
    gamer_fftw_free(input);
}

void realTest() {
    IPR ipr;

    const int N = 6;
    const int ghostBoundary = 1;
    real linear[N] = {1., 2., 3., 4., 5., 6.};


//  linear input test function
    real* input = (real*) gamer_fftw_malloc(N * sizeof(real));
    for (int i = 0; i < N; ++i) {
        input[i] = linear[i];
    }

    real reoutput[2 * N] = { };
    real* workspace = new real[2 * N * N];

    ipr.interpolateReal( input, reoutput, workspace, N, ghostBoundary);

//  interpolated input function
    std::cout << "Interpolated function g:" << std::endl;
    for (size_t i = 0; i < 2 * N - 4 * ghostBoundary; ++i)
    {
        std::cout << "(" << reoutput[i] << ")" << std::endl;
    }

    for (int i = 0; i < N; ++i) {
        input[i] = linear[i];
    }

    ipr.interpolateReal( input, reoutput, workspace, N, ghostBoundary);

//  interpolated input function
    std::cout << "Interpolated function g:" << std::endl;
    for (size_t i = 0; i < 2 * N - 4 * ghostBoundary; ++i)
    {
        std::cout << "(" << reoutput[i] << ")" << std::endl;
    }

    printf("Deleteing workspace\n");
    delete [] workspace;
    printf("Deleting input\n");
    gamer_fftw_free(input);
}

int main(int argc, char** argv)
{
    int i, N = 0, printInput = 0;
    if (argc >= 2) {
        /* there is 1 parameter (or more) in the command line used */
        /* argv[0] may point to the program name */
        /* argv[1] points to the 1st parameter */
        /* argv[argc] is NULL */
        N = atoi(argv[1]); /* better to use strtol */
        if (N > 0) {
            if (argc >= 3) {
                printInput = atoi(argv[2]); /* better to use strtol */
            }
            IPRVsGAMER(N, printInput);
            //fastIPR(N, printInput);
        } else {
        fprintf(stderr, "Please use a positive integer.\n");
        }
    }

    return 0;
}