#include "cblacs_grid.hpp"
#include "distribution.hpp"
#include "parallel_solver.hpp"
#include "tests_general.hpp"

struct distribution_test_init
{
  distribution_test_init() { initialize_distribution(); }
  ~distribution_test_init() { finalize_distribution(); }
};

#ifdef ASGARD_USE_MPI
static distribution_test_init const distrib_test_info;
#endif

TEMPLATE_TEST_CASE("", "[parallel_solver]", float, double)
{
  int myrank    = get_rank();
  int num_ranks = get_num_ranks();

  fk::matrix<TestType> A{
      {0., 0., 1., 1.}, {0., 0., 1., 1.}, {2., 2., 3., 3.}, {2., 2., 3., 3.}};

  if (myrank != 0)
    A.clear_and_resize(0, 0);

  fk::vector<TestType> B{0., 0., 2., 2.};
  if (myrank != 0)
    B.resize(0);

  auto grid = std::make_shared<cblacs_grid>();
  parallel_solver<TestType> solver;

  int n = 4;
  int m = 4;
  REQUIRE(n == m);
  fk::matrix<TestType> A_distr(4, 4, 2, 2, grid);
  if (num_ranks == 1)
  {
    REQUIRE(A_distr.local_size() == A.size());
  }
  else
  {
    REQUIRE(A_distr.local_size() == 4);
  }

  int descA[9];
  if (myrank == 0)
  {
    std::copy_n(A.get_desc(), 9, descA);
  }
  MPI_Bcast(descA, 9, MPI_INT, 0, MPI_COMM_WORLD);

  solver.scatter_matrix(A.data(), descA, A_distr.data(), A_distr.get_desc(), n,
                        n);

  if (num_ranks == 1)
  {
    for (int i = 0; i < m; ++i)
    {
      for (int j = 0; j < n; ++j)
      {
        REQUIRE_THAT(A(i, j), Catch::Matchers::WithinRel(A_distr(i, j),
                                                         TestType{0.001}));
      }
    }
  }
  else
  {
    for (int i = 0; i < 2; ++i)
    {
      for (int j = 0; j < 2; ++j)
      {
        REQUIRE_THAT(A_distr(i, j),
                     Catch::Matchers::WithinRel(myrank, TestType{0.001}));
      }
    }
  }

  n = B.size();
  fk::vector<TestType> B_distr(4, 2, grid);
  if (num_ranks == 1)
  {
    REQUIRE(B_distr.local_size() == B.size());
  }
  else
  {
    REQUIRE(B_distr.local_size() == 2);
  }

  int descB[9];
  if (myrank == 0)
  {
    std::copy_n(B.get_desc(), 9, descB);
  }

  MPI_Bcast(descB, 9, MPI_INT, 0, MPI_COMM_WORLD);
  solver.scatter_matrix(B.data(), descB, B_distr.data(), B_distr.get_desc(), 1,
                        n);

  if (num_ranks == 1)
  {
    for (int i = 0; i < n; ++i)
    {
      REQUIRE_THAT(B(i),
                   Catch::Matchers::WithinRel(B_distr(i), TestType{0.001}));
    }
  }
  else if (myrank % 2 == 0)
  {
    for (int i = 0; i < 2; ++i)
    {
      REQUIRE_THAT(B_distr(i),
                   Catch::Matchers::WithinRel(myrank, TestType{0.001}));
    }
  }
}
