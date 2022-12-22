#include <vector>

template<typename scalar>
int solve_gmres(){
    // let's assume we hava a matrix A and rught hand side b, we can only do matrix-vector product
    // we have initial vector x, maybe all zeros
    // we have preconditioner P, we can only apply the preconditioner to a vector
    // we have target tolerance and number of iterations to do between restart
    // we also have max number of iterations


    std::vector<scalar> H, S, Z;
    std::vector<scalar> C;
    std::vector<scalar> coeffs;
    // avoids allocation by reserving the max space that we would need
    H.reserve(restart * (restart + 1)); // transformation matrix, upper triangular packed
    S.reserve(restart + 1); // sin of Givens rotation
    C.reserve(restart + 1); // cos of Givens rotation
    Z.reserve(restart + 1); // holds the coefficients of the solution

    coeffs.resize(restart); // holds the coefficients of the residual projected on the basis

    scalar inner_res = 0.0; // 0.0 is a place holder, will be overwritten
    scalar outer_res = tolerance + 1.0; // forces an initial iteration
    int total_iterations = 0;
    int outer_iterations = 0;

    // needs a scratch vector r
    // need storage for the Krylov basis: W must be num_rows x restarts
    std::vector<scalar> W(num_rows * restarts); // could be fk::matrix

    while((outer_res > tolerance) && (outer_iterations < max_outer_iterations)){
        H.resize(0); S.resize(0); C.resize(0); Z.resize(0);

        // !! computer r = b - A x, must be done matrix free, may need more workspace
        // !! computer r = P^{-1} r (again, matrix free is preferred)
        total_iterations++;

        inner_res = nrm2(r);
        scal(1.0 / inner_res, r);
        Z.push_back(inner_res);

        // !! W[0:num_rows-1] = r

        int inner_iterations = 0;
        while ((inner_res > tolerance) && (inner_iterations < restart)){

            // !! compute r = P^{-1} A r
            total_iterations++;

            gemv('T', num_rows, inner_iterations+1, 1.0, W, r, 0.0, coeffs);
            gemv('N', num_rows, inner_iterations+1, -1.0, W, coeffs, 1.0, r);

            auto nrm = norm2(engine, r);
            scal(engine, 1.0 / nrm, r);

            for(int i=0; i<inner_iterations; i++)
                rot(1, &coeffs[i], 1, &coeffs[i+1], 1, C[i], S[i]); // uses BLAS Givens rotation method

            scalar isin = 0.0; // placeholder value
            scalar beta = nrm;
            scalar icos = 0.0; // placeholder value
            rotg(coeffs[inner_iterations], beta, icos, isin); // another BLAS call

            H.insert(H.end(), coeffs.begin(), coeffs.end());
            S.push_back(isin);
            C.push_back(icos);

            inner_res = std::abs(S.back() * Z.back());
            inner_iterations++;

            // the if-condition prevents one operation, if convergence has been achieved
            if ((inner_res > tolerance) and (inner_iterations < restart)){
                // expand the Krylov basis

                // using syntax for BLAS copy, useful when using cuBLAS on the GPU and cannot rely on std::copy()
                copy(engine, num_rows, r, 1, W.data() + inner_iterations * num_rows, 1);

                Z.push_back(0.0);
                // yet another BLAS call to Given methods
                rot(1, Z.data() + inner_iterations-1, 1, Z.data() + inner_iterations, 1, C.back(), S.back());
            }
        }

        if (H.size() > 0){
            // this happens when the last iteration overestimated the residual and the new iteration
            // simply declared "converged" at iteration 0, then there is no point to adjust x
            tpsv('U', 'N', 'N', (int) Z.size(), H, Z);
            gemv('N', num_rows, num_basis, 1.0, W, coeffs, 1.0, x);
        }
        outer_iterations++;
        outer_res = inner_res;
    }

    return total_iterations; // can also return the final tolerance
}
