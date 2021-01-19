//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#include "dlaf/solver/backtransformation.h"

#include <functional>
#include <sstream>
#include <tuple>
#include "gtest/gtest.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix_output.h"
#include "dlaf_test/comm_grids/grids_6_ranks.h"
#include "dlaf_test/matrix/util_matrix.h"
#include "dlaf_test/matrix/util_matrix_blas.h"
#include "dlaf_test/util_types.h"

using namespace dlaf;
using namespace dlaf::comm;
using namespace dlaf::matrix;
using namespace dlaf::matrix::test;
using namespace dlaf::test;
using namespace testing;

::testing::Environment* const comm_grids_env =
    ::testing::AddGlobalTestEnvironment(new CommunicatorGrid6RanksEnvironment);

template <typename Type>
class BackTransformationSolverLocalTest : public ::testing::Test {};
//TYPED_TEST_SUITE(BackTransformationSolverLocalTest, MatrixElementTypes);
TYPED_TEST_SUITE(BackTransformationSolverLocalTest, double);

template <typename Type>
class BackTransformationSolverDistributedTest : public ::testing::Test {
public:
  const std::vector<CommunicatorGrid>& commGrids() {
    return comm_grids;
  }
};
TYPED_TEST_SUITE(BackTransformationSolverDistributedTest, double);

GlobalElementSize globalTestSize(const LocalElementSize& size) {
  return {size.rows(), size.cols()};
}

const std::vector<blas::Side> blas_sides({blas::Side::Left, blas::Side::Right});
const std::vector<blas::Uplo> blas_uplos({blas::Uplo::Lower, blas::Uplo::Upper});
const std::vector<blas::Op> blas_ops({blas::Op::NoTrans, blas::Op::Trans, blas::Op::ConjTrans});
const std::vector<blas::Diag> blas_diags({blas::Diag::NonUnit, blas::Diag::Unit});

//const std::vector<std::tuple<SizeType, SizeType, SizeType, SizeType>> sizes = {
//    {0, 0, 1, 1},                                                // m, n = 0
//    {0, 2, 1, 2}, {7, 0, 2, 1},                                  // m = 0 or n = 0
//    {2, 2, 5, 5}, {10, 10, 2, 3}, {7, 7, 3, 2},                  // m = n
//    {3, 2, 7, 7}, {12, 3, 5, 5},  {7, 6, 3, 2}, {15, 7, 3, 5},   // m > n
//    {2, 3, 7, 7}, {4, 13, 5, 5},  {7, 8, 2, 9}, {19, 25, 6, 5},  // m < n
//};
//
//GlobalElementSize globalTestSize(const LocalElementSize& size) {
//  return {size.rows(), size.cols()};
//}


// TODO: for now there are 3 test cases for matrices of size 3x3, 4x4, 20x20; all with nb = 1 --> extend to general, using computeTfactor routines

//TYPED_TEST(BackTransformationSolverLocalTest, Correctness_n3_nb1) {
//  const SizeType m = 3;
//  const SizeType n = 3;
//  const SizeType mb = 1;
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
//    static const double values[] = {1.8571, 0.0, 0.0, -0.82, 1.9938, 0.0, 0., 0., 0.};
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
//  LocalElementSize sizeC(m, n);
//  TileElementSize blockSizeC(mb, nb);
//  Matrix<double, Device::CPU> mat_c(sizeC, blockSizeC);
//  set(mat_c, el_C);
//
//  LocalElementSize sizeV(m, n);
//  TileElementSize blockSizeV(mb, nb);
//  Matrix<double, Device::CPU> mat_v(sizeV, blockSizeV);
//  set(mat_v, el_V);
//
//  LocalElementSize sizeT(m, n);
//  TileElementSize blockSizeT(mb, nb);
//  Matrix<double, Device::CPU> mat_t(sizeT, blockSizeT);
//  set(mat_t, el_T);
//
//
//  std::cout << "C " << mat_c << std::endl;
//  std::cout << "V " << mat_v << std::endl;
//  std::cout << "T " << mat_t << std::endl;
//  
//  solver::backTransformation<Backend::MC>(mat_c, mat_v, mat_t);
//
//  double error = 0.1;
//  CHECK_MATRIX_NEAR(res, mat_c, error, error);
//}


TYPED_TEST(BackTransformationSolverLocalTest, Correctness_n4_n1) {
  const SizeType n = 4;
  const SizeType nb = 2;

  auto el_C = [](const GlobalElementIndex& index) {
    static const double values[] = {17.00000, 86.00000, 148.00000, 111.00000 , 1.00000, 160.00000, 155.00000, 170.00000 , 190.00000, 58.00000, 81.00000, 64.00000 , 118.00000, 176.00000, 30.00000, 105.00000 };
    return values[index.row() + 4 * index.col()];
  };

  auto el_V = [](const GlobalElementIndex& index) {
    static const double values[] = {1.00000, 0.38788, 0.66751, 0.50063 , 0.00000, 1.00000, -0.21782, 0.27163 , 0.00000, 0.00000, 1.00000, 0.19976 , 0.00000, 0.00000, 0.00000, 0.00000 };
    return values[index.row() + 4 * index.col()];
  };

  auto el_T = [](const GlobalElementIndex& index) {
    static const double values[] = {1.08304, 0.00000, 0.00000, 0.00000 , -0.73116, 1.78376, 0.00000, 0.00000 , -1.82870, 0.56111, 1.92325, 0.00000 , 0.00000, 0.00000, 0.00000, 0.00000 };
    return values[index.row() + 4 * index.col()];
  };

  auto res = [](const GlobalElementIndex& index) {
    static const double values[] = {-204.71896, 0.00002, -0.00063, 0.00075 , -271.52747, -69.27218, -0.00027, 0.00078 , -133.40188, 47.89095, 173.29409, 0.00092 , -162.35393, -96.63778, 142.80510, -27.08661 };
    return values[index.row() + 4 * index.col()];
  };


  LocalElementSize sizeC(n, n);
  TileElementSize blockSizeC(nb, nb);
  Matrix<double, Device::CPU> mat_c(sizeC, blockSizeC);
  set(mat_c, el_C);

  LocalElementSize sizeV(n, n);
  TileElementSize blockSizeV(nb, nb);
  Matrix<double, Device::CPU> mat_v(sizeV, blockSizeV);
  set(mat_v, el_V);

  LocalElementSize sizeT(n, n);
  TileElementSize blockSizeT(nb, nb);
  Matrix<double, Device::CPU> mat_t(sizeT, blockSizeT);
  set(mat_t, el_T);

  solver::backTransformation<Backend::MC>(mat_c, mat_v, mat_t);

  double error = 0.1;
  CHECK_MATRIX_NEAR(res, mat_c, error, error);
}


//TYPED_TEST(BackTransformationSolverLocalTest, Correctness_n20_n1) {
//  const SizeType n = 20;
//  const SizeType nb = 1;
//auto el_C = [](const GlobalElementIndex& index) {
//  static const double values[] = {15.00000, 9.10000, 14.50000, 10.50000, 16.10000, 14.90000, 8.80000, 14.50000, 4.20000, 1.60000, 0.70000, 15.00000, 0.00000, 0.90000, 1.50000, 18.20000, 15.30000, 12.70000, 16.70000, 16.80000 , 7.40000, 18.80000, 19.00000, 13.60000, 15.90000, 9.90000, 6.00000, 17.80000, 13.10000, 8.90000, 10.60000, 7.80000, 2.40000, 8.50000, 5.30000, 16.40000, 13.80000, 8.90000, 3.10000, 10.90000 , 10.20000, 11.80000, 17.50000, 10.20000, 6.40000, 7.50000, 11.90000, 1.50000, 6.60000, 2.80000, 2.80000, 18.50000, 8.70000, 19.40000, 5.80000, 13.40000, 19.10000, 8.00000, 0.90000, 0.20000 , 4.20000, 18.60000, 11.50000, 6.60000, 11.50000, 13.70000, 14.50000, 7.60000, 6.10000, 5.30000, 4.60000, 8.50000, 19.70000, 17.50000, 7.00000, 14.90000, 6.70000, 3.10000, 11.30000, 18.00000 , 9.90000, 12.80000, 5.70000, 6.70000, 15.00000, 14.20000, 7.90000, 4.30000, 0.20000, 7.20000, 0.50000, 14.70000, 10.60000, 3.40000, 12.50000, 10.60000, 9.00000, 2.70000, 13.20000, 19.20000 , 3.00000, 17.80000, 18.10000, 11.30000, 17.10000, 10.10000, 12.50000, 0.80000, 16.40000, 18.50000, 6.70000, 2.40000, 6.90000, 1.40000, 3.90000, 14.60000, 19.00000, 15.10000, 1.70000, 8.10000 , 0.10000, 17.90000, 8.80000, 6.70000, 5.70000, 0.20000, 19.50000, 1.40000, 17.80000, 4.50000, 18.80000, 11.60000, 2.80000, 16.00000, 10.50000, 17.90000, 7.30000, 2.60000, 2.90000, 8.80000 , 2.00000, 9.70000, 2.50000, 4.20000, 9.10000, 14.80000, 19.80000, 15.80000, 13.70000, 4.80000, 13.00000, 9.80000, 12.90000, 3.10000, 1.30000, 9.80000, 5.10000, 10.60000, 11.50000, 12.10000 , 7.40000, 19.70000, 2.00000, 9.80000, 5.20000, 12.20000, 19.40000, 14.30000, 7.00000, 2.30000, 5.90000, 19.80000, 6.00000, 1.00000, 3.70000, 3.10000, 7.00000, 10.50000, 2.90000, 17.70000 , 12.50000, 2.20000, 18.10000, 2.40000, 17.60000, 6.40000, 17.20000, 16.70000, 14.40000, 18.00000, 3.10000, 1.80000, 11.40000, 11.20000, 9.00000, 13.10000, 3.20000, 11.40000, 10.80000, 12.80000 , 12.50000, 7.40000, 8.00000, 16.80000, 1.90000, 9.00000, 4.90000, 13.60000, 17.60000, 19.80000, 18.90000, 12.30000, 0.50000, 5.40000, 5.10000, 17.40000, 16.40000, 9.80000, 15.60000, 2.00000 , 10.50000, 1.80000, 3.70000, 0.40000, 18.10000, 2.40000, 0.60000, 0.40000, 7.70000, 4.10000, 13.90000, 5.00000, 16.10000, 3.30000, 1.40000, 8.20000, 11.00000, 9.20000, 2.70000, 5.30000 , 12.80000, 3.60000, 13.50000, 19.10000, 10.00000, 3.70000, 11.90000, 18.50000, 6.70000, 3.70000, 13.70000, 18.40000, 8.40000, 17.70000, 14.60000, 1.10000, 12.40000, 17.50000, 14.70000, 15.10000 , 5.30000, 15.30000, 7.30000, 11.40000, 16.20000, 7.20000, 1.90000, 8.30000, 9.50000, 16.90000, 17.40000, 3.30000, 9.00000, 2.00000, 14.80000, 18.80000, 3.30000, 7.00000, 17.80000, 18.90000 , 4.90000, 8.20000, 4.80000, 3.70000, 7.30000, 4.90000, 8.50000, 13.20000, 13.40000, 9.40000, 6.10000, 8.40000, 0.90000, 2.70000, 7.90000, 8.50000, 19.70000, 3.60000, 0.40000, 9.30000 , 9.50000, 15.90000, 2.10000, 17.70000, 12.90000, 16.90000, 1.70000, 2.10000, 0.80000, 18.00000, 3.40000, 15.00000, 15.30000, 14.10000, 17.30000, 7.90000, 9.20000, 9.70000, 6.60000, 14.40000 , 14.10000, 12.70000, 17.30000, 8.50000, 20.00000, 16.80000, 16.10000, 2.60000, 19.90000, 6.80000, 0.50000, 15.00000, 18.30000, 13.60000, 4.20000, 10.90000, 11.50000, 13.30000, 2.20000, 10.00000 , 14.50000, 13.10000, 2.50000, 9.70000, 9.90000, 7.60000, 19.20000, 0.90000, 12.50000, 5.70000, 5.40000, 17.40000, 9.40000, 1.00000, 19.00000, 1.20000, 16.00000, 10.70000, 18.20000, 10.40000 , 19.30000, 19.80000, 7.20000, 7.60000, 18.90000, 14.50000, 11.40000, 9.30000, 5.40000, 13.60000, 10.10000, 7.60000, 7.10000, 9.60000, 19.00000, 0.80000, 16.10000, 2.40000, 2.80000, 2.40000 , 4.70000, 16.30000, 10.40000, 14.90000, 6.80000, 4.70000, 13.90000, 6.00000, 13.20000, 1.90000, 1.30000, 6.40000, 1.00000, 6.20000, 15.00000, 1.00000, 7.90000, 2.20000, 5.30000, 5.60000 };
//  return values[index.row() + 20 * index.col()];
//};
//
//auto el_V = [](const GlobalElementIndex& index) {
//  static const double values[] = {1.00000, 0.13157, 0.20965, 0.15181, 0.23278, 0.21543, 0.12723, 0.20965, 0.06073, 0.02313, 0.01012, 0.21688, 0.00000, 0.01301, 0.02169, 0.26314, 0.22121, 0.18362, 0.24146, 0.24290 , 0.00000, 1.00000, 0.19962, 0.14052, 0.08939, -0.03982, -0.01962, 0.16908, 0.25117, 0.19519, 0.25608, -0.09522, 0.06108, 0.19871, 0.10552, 0.06099, 0.05161, -0.02219, -0.24814, -0.05157 , 0.00000, 0.00000, 1.00000, 0.00220, -0.19843, -0.05001, 0.20112, -0.37288, -0.05463, -0.08040, -0.10349, 0.32963, 0.22969, 0.45032, 0.07714, -0.00416, 0.24100, -0.00051, -0.15039, -0.30968 , 0.00000, 0.00000, 0.00000, 1.00000, 0.00255, -0.17750, -0.25284, 0.12774, 0.11144, 0.02621, 0.08087, 0.02394, -0.51750, -0.25287, -0.07697, -0.04483, 0.20253, 0.14128, -0.27884, -0.34428 , 0.00000, 0.00000, 0.00000, 0.00000, 1.00000, 0.02152, -0.13924, -0.34181, -0.16996, 0.26107, -0.05258, 0.33063, 0.05735, -0.14386, 0.51802, -0.21738, 0.10312, -0.20411, -0.12550, 0.02298 , 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 1.00000, 0.27594, -0.34548, 0.24215, 0.31759, -0.08192, -0.32264, 0.06408, -0.33159, -0.24149, 0.10483, 0.20404, 0.38363, 0.08177, 0.02277 , 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 1.00000, -0.27934, 0.22261, -0.01683, 0.25375, 0.18032, -0.22703, -0.05041, 0.12945, 0.17863, 0.02768, 0.02976, 0.08083, 0.03635 , 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 1.00000, 0.06243, 0.01710, 0.16935, 0.30486, 0.27868, 0.03924, 0.06360, -0.28461, 0.03098, 0.05205, -0.14144, -0.24632 , 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 1.00000, 0.08998, 0.34551, -0.26527, -0.03916, 0.12756, 0.07008, 0.37622, 0.02268, -0.07925, 0.22844, -0.21064 , 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 1.00000, -0.23861, -0.30011, -0.08574, -0.04789, 0.01303, -0.16606, -0.21106, 0.16024, -0.17584, 0.07226 , 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 1.00000, 0.42744, 0.30738, 0.15615, 0.23674, -0.10121, 0.15439, -0.23810, 0.27548, -0.13860 , 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 1.00000, -0.32806, 0.11047, 0.41799, 0.21992, -0.09703, -0.44761, 0.44425, 0.03450 , 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 1.00000, -0.25737, -0.33546, 0.13706, 0.08504, -0.22120, -0.62337, -0.51001 , 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 1.00000, -0.16810, -0.38413, 0.39727, -0.45302, -0.13201, 0.18025 , 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 1.00000, 0.17084, 0.62918, 0.18060, -0.01956, 0.13938 , 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 1.00000, -0.38273, 0.46891, -0.17123, -0.13086 , 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 1.00000, -0.41176, 0.07785, -0.17737 , 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 1.00000, -0.34558, 0.90820 , 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 1.00000, -0.01970 , 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000 };
//  return values[index.row() + 20 * index.col()];
//};
//
//auto el_T = [](const GlobalElementIndex& index) {
//  static const double values[] = {1.27694, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000 , -0.36036, 1.42915, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000 , -0.07911, -0.33089, 1.11838, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000 , 0.04499, -0.42879, 0.17826, 1.15679, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000 , -0.16047, -0.08642, -0.03195, 0.02962, 1.14027, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000 , -0.45226, -0.01144, 0.17617, 0.18072, 0.15194, 1.08391, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000 , -0.12587, -0.00202, -0.46502, 0.16857, -0.03015, -0.57749, 1.49585, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000 , -0.52231, -0.53476, 0.04216, -0.04718, 0.33335, 0.49205, 0.63913, 1.45518, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000 , 0.12482, -0.82015, 0.26105, -0.30566, 0.49538, -0.12570, -0.66769, 0.10082, 1.36477, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000 , 0.57661, -0.54431, 0.20888, -0.23903, -0.35952, -0.80173, 0.37927, 0.19633, 0.10582, 1.54947, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000 , 0.46796, -0.10958, -0.04262, 0.24187, -0.92939, -0.02974, -0.35446, -0.71112, -0.51898, 1.02401, 1.29577, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000 , -0.64027, 0.17748, 0.18398, -0.12266, -0.06361, 0.57402, -0.23852, 0.46908, 0.42399, 0.03368, -0.93192, 1.14206, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000 , 0.24398, 0.09838, -0.42344, 0.00981, -0.09536, -0.19772, 0.07807, -0.17476, 0.46878, 0.00670, -0.84563, 0.80483, 1.05129, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000 , 0.10326, -0.32507, -0.67172, 0.35199, 0.30670, 0.23120, 0.35393, 0.01531, 0.31555, -0.13318, -0.48751, 0.03389, 0.17108, 1.25882, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000 , 0.06136, -0.07945, -0.45357, -0.09664, -0.59662, -0.23897, 0.03267, -0.05076, 0.06950, -0.14103, -0.47983, -0.08784, 0.51736, 0.06455, 1.35366, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000 , -0.24216, -0.38374, -0.64198, 0.46184, 0.33879, 0.36996, 0.33879, 0.17428, -0.59940, 0.27183, 0.19919, -0.10859, -0.07769, 1.33579, 0.00033, 1.41562, 0.00000, 0.00000, 0.00000, 0.00000 , -0.38389, 0.09229, 0.01659, -0.33815, 0.43756, -0.16217, 0.39470, 0.49746, -1.09666, 0.76195, 0.88793, -0.56510, -1.05928, 0.06647, -1.18522, 1.32742, 1.65691, 0.00000, 0.00000, 0.00000 , -0.63051, -0.08337, 0.40164, -0.26428, 0.30719, -0.03888, -0.41528, 0.27066, 0.32140, 0.41760, 0.15456, 0.79161, -0.23785, -0.22660, -1.16853, 0.22301, 1.02223, 1.02868, 0.00000, 0.00000 , -0.56363, 0.12785, -0.36671, 0.72083, 0.77975, -0.73511, 0.72987, 0.36175, 0.18875, 0.18672, -1.04962, 0.60987, 1.33173, 0.61896, -0.59590, 0.42349, 0.47336, 0.74750, 1.99922, 0.00000 , 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000 };
//  return values[index.row() + 20 * index.col()];
//};
//
//auto res = [](const GlobalElementIndex& index) {
//  static const double values[] = {-54.16388, -0.00015, -0.00017, 0.00007, 0.00003, -0.00033, 0.00003, 0.00015, 0.00021, 0.00011, -0.00010, 0.00011, -0.00002, -0.00004, -0.00009, -0.00037, -0.00052, 0.00001, 0.00002, 0.00028 , -45.81652, -27.49215, -0.00018, 0.00009, 0.00030, -0.00022, 0.00029, 0.00017, -0.00003, 0.00013, 0.00005, 0.00006, 0.00005, 0.00013, -0.00011, -0.00029, -0.00044, 0.00009, 0.00006, 0.00021 , -36.12014, -16.33695, -28.62653, 0.00013, 0.00016, -0.00027, 0.00024, 0.00018, 0.00024, 0.00013, -0.00014, -0.00005, 0.00011, 0.00001, -0.00004, -0.00023, -0.00050, 0.00002, -0.00011, 0.00018 , -40.44444, -16.28333, -10.73231, 27.23099, 0.00033, -0.00031, 0.00025, 0.00025, 0.00006, 0.00017, -0.00001, -0.00013, -0.00005, -0.00000, 0.00002, -0.00028, -0.00032, 0.00014, 0.00001, 0.00021 , -39.98721, -2.32032, -4.10783, 16.46642, -15.69434, -0.00030, 0.00015, 0.00025, 0.00014, 0.00022, -0.00003, 0.00010, -0.00020, -0.00006, -0.00005, -0.00027, -0.00037, 0.00018, 0.00010, 0.00021 , -40.51429, -25.72598, -4.87726, 1.19819, -4.15841, -24.16645, 0.00044, 0.00030, -0.00004, 0.00024, -0.00000, -0.00015, 0.00005, 0.00012, -0.00007, -0.00031, -0.00048, -0.00002, 0.00001, 0.00033 , -29.81721, -27.19694, -13.37350, 10.61873, 0.68285, -1.76396, -24.47550, 0.00029, 0.00000, 0.00024, 0.00010, -0.00024, 0.00012, -0.00005, 0.00027, -0.00019, -0.00012, 0.00031, 0.00000, 0.00000 , -36.58143, -11.29814, 0.98227, 16.05999, 4.97846, -4.56242, -9.11503, -19.45875, -0.00023, 0.00007, 0.00016, -0.00032, 0.00003, -0.00011, 0.00019, -0.00036, -0.00024, 0.00008, 0.00005, 0.00001 , -37.83076, -9.16641, -3.41291, 10.10306, -5.20116, 1.91382, -7.72162, -15.19938, 19.28067, 0.00015, 0.00027, -0.00031, 0.00006, -0.00006, 0.00020, -0.00038, -0.00018, 0.00030, 0.00010, -0.00009 , -41.30048, -16.77857, -0.53779, 12.31120, 2.73635, -7.62995, 0.45401, -5.07800, -10.37351, -23.54555, 0.00021, 0.00013, 0.00016, -0.00022, 0.00015, -0.00049, -0.00034, 0.00006, 0.00009, 0.00033 , -41.61948, -19.72521, -3.10922, -6.22854, -2.32391, -3.58850, -10.11209, -10.50148, -11.12089, 0.46762, -22.52199, 0.00015, 0.00014, -0.00001, -0.00005, -0.00030, -0.00060, -0.00011, -0.00003, 0.00026 , -23.39548, -9.81227, -4.76499, 4.72190, -4.22035, -7.17797, 0.44081, -6.47635, -11.76820, 0.91434, 4.34864, 19.42994, -0.00013, -0.00001, -0.00030, -0.00003, -0.00042, -0.00006, 0.00015, 0.00018 , -46.13419, -13.59624, -11.14499, 4.51581, -3.11389, 9.92412, -4.23282, -11.76809, -1.11002, -11.08189, -6.30287, 5.83489, 22.09861, -0.00011, 0.00009, -0.00045, -0.00040, 0.00022, -0.00001, -0.00003 , -40.58797, -19.22136, 10.31232, 15.95668, -11.97660, -4.70138, -8.69223, 2.13218, -10.68124, -0.12467, -9.26434, 4.60606, 4.33161, 7.05569, -0.00009, -0.00026, -0.00038, 0.00017, 0.00011, 0.00034 , -29.42791, -15.87622, -2.27257, -1.93556, -5.91875, -2.92501, -5.34636, -7.04791, -0.13783, -2.04022, -0.82841, -0.96705, -2.12131, -9.33457, -10.56756, -0.00022, -0.00030, 0.00005, 0.00017, -0.00001 , -38.15171, -15.06291, -11.46004, 16.22024, -21.92075, -1.08477, 6.58437, -3.37808, -0.03193, 0.38659, -10.06353, 1.75860, 5.58283, 4.70393, 2.02971, -11.41327, -0.00042, 0.00010, 0.00009, 0.00016 , -46.25226, -17.91147, -18.23999, 11.25521, -1.63913, -10.96735, 3.50810, -7.65584, -0.24131, -6.60383, 6.74655, 4.94923, -3.21422, -2.47873, 7.66843, -3.44147, 10.17219, 0.00003, 0.00009, 0.00035 , -39.99799, -3.96493, -12.52203, 9.17687, -16.52706, -7.53360, -11.96690, -11.61928, 2.02813, -4.34876, -4.21482, 1.25742, 8.34945, -2.72584, -0.50036, 5.43621, 5.99093, 11.82692, 0.00023, 0.00033 , -36.74938, -21.76823, -9.48651, 5.53511, -19.74348, -0.11819, 2.77485, -8.26029, -2.12451, -5.06058, 0.32865, 1.25374, -1.30846, -5.08452, 4.30158, 0.48785, -6.36799, 15.71669, 6.18751, 0.00035 , -26.73337, -17.17171, -6.97036, 4.20398, -6.83632, -0.53473, -6.21429, -1.19228, 6.74611, -3.79499, -0.03264, -8.08253, 7.53227, -3.53002, 3.00275, 3.36953, 6.74943, 9.28377, -1.45723, 4.07405 };
//  return values[index.row() + 20 * index.col()];
//};
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
//  solver::backTransformation<Backend::MC>(mat_c, mat_v, mat_t);
//
//  double error = 0.1;
//  CHECK_MATRIX_NEAR(res, mat_c, error, error);
//}


// TYPED_TEST(BackTransformationSolverDistributedTest, Correctness_n3_nb1_distrib) {
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
//    static const double values[] = {1.8571, 0.0, 0.0, -0.82, 1.9938, 0.0, 0., 0., 0.};
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
//  for (const auto& comm_grid : this->commGrids()) {
//    Index2D src_rank_index(std::max(0, comm_grid.size().rows() - 1),
//			   std::min(1, comm_grid.size().cols() - 1));
//    
//    LocalElementSize sizeC(n, n);
//    TileElementSize blockSizeC(nb, nb);
//    GlobalElementSize szC = globalTestSize(sizeC);
//    Distribution distributionC(szC, blockSizeC, comm_grid.size(), comm_grid.rank(), src_rank_index);
//    Matrix<double, Device::CPU> mat_c(std::move(distributionC));
//    set(mat_c, el_C);
//
//    LocalElementSize sizeV(n, n);
//    TileElementSize blockSizeV(nb, nb);
//    GlobalElementSize szV = globalTestSize(sizeV);
//    Distribution distributionV(szV, blockSizeV, comm_grid.size(), comm_grid.rank(), src_rank_index);
//    Matrix<double, Device::CPU> mat_v(std::move(distributionV));
//    set(mat_v, el_V);
//
//    LocalElementSize sizeT(n, n);
//    TileElementSize blockSizeT(nb, nb);
//    GlobalElementSize szT = globalTestSize(sizeT);
//    Distribution distributionT(szT, blockSizeT, comm_grid.size(), comm_grid.rank(), src_rank_index);
//    Matrix<double, Device::CPU> mat_t(std::move(distributionT));
//    set(mat_t, el_T);
//        
//    solver::backTransformation<Backend::MC>(comm_grid, mat_c, mat_v, mat_t);
//    
//    double error = 0.1;
//    CHECK_MATRIX_NEAR(res, mat_c, error, error);
//  }
//}
//


 TYPED_TEST(BackTransformationSolverDistributedTest, Correctness_n4_nb1_distrib) {
  
  const SizeType n = 4;
  const SizeType nb = 1;

  auto el_C = [](const GlobalElementIndex& index) {
    static const double values[] = {17.00000, 86.00000, 148.00000, 111.00000 , 1.00000, 160.00000, 155.00000, 170.00000 , 190.00000, 58.00000, 81.00000, 64.00000 , 118.00000, 176.00000, 30.00000, 105.00000 };
    return values[index.row() + 4 * index.col()];
  };

  auto el_V = [](const GlobalElementIndex& index) {
    static const double values[] = {1.00000, 0.38788, 0.66751, 0.50063 , 0.00000, 1.00000, -0.21782, 0.27163 , 0.00000, 0.00000, 1.00000, 0.19976 , 0.00000, 0.00000, 0.00000, 0.00000 };
    return values[index.row() + 4 * index.col()];
  };

  auto el_T = [](const GlobalElementIndex& index) {
    static const double values[] = {1.08304, 0.00000, 0.00000, 0.00000 , -0.73116, 1.78376, 0.00000, 0.00000 , -1.82870, 0.56111, 1.92325, 0.00000 , 0.00000, 0.00000, 0.00000, 0.00000 };
    return values[index.row() + 4 * index.col()];
  };

  auto res = [](const GlobalElementIndex& index) {
    static const double values[] = {-204.71896, 0.00002, -0.00063, 0.00075 , -271.52747, -69.27218, -0.00027, 0.00078 , -133.40188, 47.89095, 173.29409, 0.00092 , -162.35393, -96.63778, 142.80510, -27.08661 };
    return values[index.row() + 4 * index.col()];
  };

  for (const auto& comm_grid : this->commGrids()) {
    Index2D src_rank_index(std::max(0, comm_grid.size().rows() - 1),
			   std::min(1, comm_grid.size().cols() - 1));
    
    LocalElementSize sizeC(n, n);
    TileElementSize blockSizeC(nb, nb);
    GlobalElementSize szC = globalTestSize(sizeC);
    Distribution distributionC(szC, blockSizeC, comm_grid.size(), comm_grid.rank(), src_rank_index);
    Matrix<double, Device::CPU> mat_c(std::move(distributionC));
    set(mat_c, el_C);

    LocalElementSize sizeV(n, n);
    TileElementSize blockSizeV(nb, nb);
    GlobalElementSize szV = globalTestSize(sizeV);
    Distribution distributionV(szV, blockSizeV, comm_grid.size(), comm_grid.rank(), src_rank_index);
    Matrix<double, Device::CPU> mat_v(std::move(distributionV));
    set(mat_v, el_V);

    LocalElementSize sizeT(n, n);
    TileElementSize blockSizeT(nb, nb);
    GlobalElementSize szT = globalTestSize(sizeT);
    Distribution distributionT(szT, blockSizeT, comm_grid.size(), comm_grid.rank(), src_rank_index);
    Matrix<double, Device::CPU> mat_t(std::move(distributionT));
    set(mat_t, el_T);
        
    solver::backTransformation<Backend::MC>(comm_grid, mat_c, mat_v, mat_t);
    
    double error = 0.1;
    CHECK_MATRIX_NEAR(res, mat_c, error, error);
  }
}

