#include "parallel_solver.hpp"
#include "tools.hpp"
#include <iostream>
#include <type_traits>

extern "C"
{
  void pdgeadd_(char *, int *, int *, double *, double *, int *, int *, int *,
                double *, double *, int *, int *, int *);
  void psgeadd_(char *, int *, int *, float *, float *, int *, int *, int *,
                float *, float *, int *, int *, int *);
}

template<typename P>
void parallel_solver<P>::gather_matrix(P *A, int *descA, P *A_distr,
                                       int *descA_distr, int n, int m)
{
  // Useful constants
  P zero{0.0}, one{1.0};
  int i_one{1};
  char N{'N'};
  // Call pdgeadd_ to distribute matrix (i.e. copy A into A_distr)
  if constexpr (std::is_same<P, double>::value)
  {
    pdgeadd_(&N, &m, &n, &one, A_distr, &i_one, &i_one, descA_distr, &zero, A,
             &i_one, &i_one, descA);
  }
  else if constexpr (std::is_same<P, float>::value)
  {
    psgeadd_(&N, &m, &n, &one, A_distr, &i_one, &i_one, descA_distr, &zero, A,
             &i_one, &i_one, descA);
  }
  else
  { // not instantiated; should never be reached
    std::cerr << "geadd not implemented for non-floating types" << '\n';
    expect(false);
  }
}

template<typename P>
void parallel_solver<P>::scatter_matrix(P *A, int *descA, P *A_distr,
                                        int *descA_distr, int n, int m)
{
  // Useful constants
  P zero{0.0}, one{1.0};
  int i_one{1};
  char N{'N'};
  // Call pdgeadd_ to distribute matrix (i.e. copy A into A_distr)
  if constexpr (std::is_same<P, double>::value)
  {
    pdgeadd_(&N, &m, &n, &one, A, &i_one, &i_one, descA, &zero, A_distr, &i_one,
             &i_one, descA_distr);
  }
  else if constexpr (std::is_same<P, float>::value)
  {
    psgeadd_(&N, &m, &n, &one, A, &i_one, &i_one, descA, &zero, A_distr, &i_one,
             &i_one, descA_distr);
  }
  else
  { // not instantiated; should never be reached
    std::cerr << "geadd not implemented for non-floating types" << '\n';
    expect(false);
  }
}

template class parallel_solver<float>;
template class parallel_solver<double>;
