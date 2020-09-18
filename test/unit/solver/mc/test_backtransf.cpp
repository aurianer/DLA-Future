//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#include "dlaf/solver/mc.h"

#include <functional>
#include <sstream>
#include <tuple>
#include "gtest/gtest.h"
#include "dlaf/matrix.h"
#include "dlaf/matrix_output.h"
//#include "dlaf_test/comm_grids/grids_6_ranks.h"
#include "dlaf_test/matrix/util_matrix.h"
#include "dlaf_test/matrix/util_matrix_blas.h"
#include "dlaf_test/util_types.h"

//// TRAPPING
//#pragma STDC FENV_ACCESS ON
//#include<cfenv>
//#include<fenv.h>
//#include<signal.h>

using namespace dlaf;
using namespace dlaf::comm;
using namespace dlaf::matrix;
using namespace dlaf::matrix::test;
using namespace dlaf_test;
using namespace testing;

template <typename Type>
class BacktransfSolverLocalTest : public ::testing::Test {};
TYPED_TEST_SUITE(BacktransfSolverLocalTest, MatrixElementTypes);

const std::vector<blas::Side> blas_sides({blas::Side::Left, blas::Side::Right});
const std::vector<blas::Uplo> blas_uplos({blas::Uplo::Lower, blas::Uplo::Upper});
const std::vector<blas::Op> blas_ops({blas::Op::NoTrans, blas::Op::Trans, blas::Op::ConjTrans});
const std::vector<blas::Diag> blas_diags({blas::Diag::NonUnit, blas::Diag::Unit});

const std::vector<std::tuple<SizeType, SizeType, SizeType, SizeType>> sizes = {
    {0, 0, 1, 1},                                                // m, n = 0
    {0, 2, 1, 2}, {7, 0, 2, 1},                                  // m = 0 or n = 0
    {2, 2, 5, 5}, {10, 10, 2, 3}, {7, 7, 3, 2},                  // m = n
    {3, 2, 7, 7}, {12, 3, 5, 5},  {7, 6, 3, 2}, {15, 7, 3, 5},   // m > n
    {2, 3, 7, 7}, {4, 13, 5, 5},  {7, 8, 2, 9}, {19, 25, 6, 5},  // m < n
};

GlobalElementSize globalTestSize(const LocalElementSize& size) {
  return {size.rows(), size.cols()};
}

//// TRAPPING
// void floating_point_handler(int signal, siginfo_t *sip, void *uap) {
//  std::cerr << "floating point error at " << sip->si_addr << " : ";
//  int code=sip->si_code;
//  if (code==FPE_FLTDIV)
//    std::cerr << "division by zero\n";
//  if (code==FPE_FLTUND)
//    std::cerr << "underflow\n";
//  if (code==FPE_FLTINV)
//    std::cerr << "invalid result\n";
//  std::abort();
//}

// TYPED_TEST(BacktransfSolverLocalTest, Correctness3x3) {
//  const SizeType n = 3;
//  const SizeType nb = 1;
//
//  // DATA
//  auto el_C = [](const GlobalElementIndex& index) {
//    // ColMajor
//    static const double values[] = {12, 6, -4, -51, 167, 24, 4, -68, -41};
//    return values[index.row() + 3 * index.col()];
//  };
//
//  auto el_V = [](const GlobalElementIndex& index) {
//    // ColMajor
//    static const double values[] = {1, 0.23077, -0.15385, 0, 1, 0.055556, 0, 0, 0};
//    return values[index.row() + 3 * index.col()];
//  };
//
//  auto el_T = [](const GlobalElementIndex& index) {
//    // ColMajor
//    static const double values[] = {1.8571, 1.8571, 1.8571, 0.0, 1.9938, 1.9938, 0, 0, 0};
//    return values[index.row() + 3 * index.col()];
//  };
//
//  // RESULT
//  auto res = [](const GlobalElementIndex& index) {
//    // ColMajor
//    static const double values[] = {-14., 0., 0., -21., -175., 0., 14., 70., -35.};
//    return values[index.row() + 3 * index.col()];
//  };
//
//  LocalElementSize sizeC(n, n);
//  TileElementSize blockSizeC(nb, nb);
//  Matrix<double, Device::CPU> mat_c(sizeC, blockSizeC);
//  set(mat_c, el_C);
//
//  LocalElementSize sizeV(n, n);
//  TileElementSize blockSizeV(nb, nb);
//  Matrix<double, Device::CPU> mat_v(sizeV, blockSizeV);
//  set(mat_v, el_V);
//
//  LocalElementSize sizeT(n, n);
//  TileElementSize blockSizeT(nb, nb);
//  Matrix<double, Device::CPU> mat_t(sizeT, blockSizeT);
//  set(mat_t, el_T);
//
//  //  std::cout << "Matrix C" << std::endl;
//  //  printElements(mat_c);
//  //  std::cout << "" << std::endl;
//  //  std::cout << "Matrix V" << std::endl;
//  //  printElements(mat_v);
//  //  std::cout << "" << std::endl;
//  //  std::cout << "Matrix T" << std::endl;
//  //  printElements(mat_t);
//  //  std::cout << "" << std::endl;
//
//  Solver<Backend::MC>::backtransf(mat_c, mat_v, mat_t);
//
//  //  std::cout << "Result: " << std::endl;
//  //  printElements(mat_c);
//
//  CHECK_MATRIX_NEAR(res, mat_c, 1e13 * (mat_c.size().rows() + 1) * TypeUtilities<double>::error,
//                    1e13 * (mat_c.size().rows() + 1) * TypeUtilities<double>::error);
//}

TYPED_TEST(BacktransfSolverLocalTest, Correctness) {
  //  //TRAPPING
  //   std::feclearexcept(FE_ALL_EXCEPT);
  //  feenableexcept(FE_DIVBYZERO | FE_UNDERFLOW | FE_OVERFLOW | FE_INVALID);
  //  struct sigaction act;
  //  act.sa_sigaction=floating_point_handler;
  //  act.sa_flags=SA_SIGINFO;
  //  sigaction(SIGFPE, &act, NULL);

  //  double zero=0.0;
  //  double one=1.0;
  //  std::cout << "1.0/1.0 = " << one/one << '\n';
  //  std::cout << "1.0/0.0 = " << one/zero << '\n';

  // To be generalized
  const SizeType n = 4;
  const SizeType nb = 1;

  BaseType<TypeParam> beta = 1.0f;
  BaseType<TypeParam> gamma = 1.0f;
  BaseType<TypeParam> delta = 1.0f;

  // MODIFY
  // Note: The tile elements are chosen such that:
  // - res_ij = 1 / 2^(|i-j|) * exp(I*(-i+j)),
  // where I = 0 for real types or I is the complex unit for complex types.
  // Therefore the result should be:
  // a_ij = Sum_k(res_ik * ConjTrans(res)_kj) =
  //      = Sum_k(1 / 2^(|i-k| + |j-k|) * exp(I*(-i+j))),
  // where k = 0 .. min(i,j)
  // Therefore,
  // a_ij = (4^(min(i,j)+1) - 1) / (3 * 2^(i+j)) * exp(I*(-i+j))

  // Matric C
  auto el_c = [beta, gamma](const GlobalElementIndex& index) {
    SizeType k = index.row();
    SizeType j = index.col();

    return TypeUtilities<TypeParam>::polar(gamma / (std::exp2(k + j)), beta * (k + j));
  };

  // Matrix V
  auto el_v = [beta, delta](const GlobalElementIndex& index) {
    SizeType k = index.row();
    SizeType i = index.col();

    if (k == i)
      return TypeUtilities<TypeParam>::polar(1.0, 0.0);

    if (k < i)
      return TypeUtilities<TypeParam>::polar(-9.9, 0.0);

    return TypeUtilities<TypeParam>::polar(delta / (std::exp2(k - i)), beta * (k - i));
  };

  // Matrix T
  auto el_t = [beta, gamma](const GlobalElementIndex& index) {
    //    SizeType l = index.row();
    SizeType i = index.col();
    SizeType l = 0.0;

    TypeParam el =
        TypeUtilities<TypeParam>::polar(gamma / (std::exp2(2 * i - 2 * l)), beta * (2 * i - 2 * l));

    return el;
  };

  // Matrix Result
  auto res = [beta, gamma, delta, el_c, n](const GlobalElementIndex& index) {
    SizeType k = index.row();
    SizeType j = index.col();

    BaseType<TypeParam> sub = 1.0;
    SizeType max;

    if (k == j) {
      max = k + 1;
      if (k == (n - 1))
        max = k;
    }
    else {
      if (k < j)
        max = k + 1;
      else
        max = j + 1;
    }

    for (SizeType i = 0; i < max; i++)
      sub = sub * (1.0 - (n - i) * delta * gamma * gamma);

    TypeParam el = TypeUtilities<TypeParam>::polar((gamma * sub) / (std::exp2(k + j)), (beta * (k + j)));

    return el;
  };

  LocalElementSize size_mat(n, n);
  TileElementSize blockSize_mat(nb, nb);
  Matrix<TypeParam, Device::CPU> mat_v(size_mat, blockSize_mat);
  Matrix<TypeParam, Device::CPU> mat_t(size_mat, blockSize_mat);
  Matrix<TypeParam, Device::CPU> mat_c(size_mat, blockSize_mat);
  Matrix<TypeParam, Device::CPU> mat_res(size_mat, blockSize_mat);

  set(mat_v, el_v);
  set(mat_t, el_t);
  set(mat_c, el_c);
  set(mat_res, res);

  std::cout << "Matrix C" << std::endl;
  printElements(mat_c);
  std::cout << "" << std::endl;
  std::cout << "Matrix V" << std::endl;
  printElements(mat_v);
  std::cout << "" << std::endl;
  std::cout << "Matrix T" << std::endl;
  printElements(mat_t);
  std::cout << "" << std::endl;

  Solver<Backend::MC>::backtransf(mat_c, mat_v, mat_t);

  std::cout << "RESULT: matrix C" << std::endl;
  printElements(mat_c);
  std::cout << "" << std::endl;

  CHECK_MATRIX_NEAR(res, mat_c, 40 * (mat_c.size().rows() + 1) * TypeUtilities<TypeParam>::error,
                    40 * (mat_c.size().rows() + 1) * TypeUtilities<TypeParam>::error);
}
