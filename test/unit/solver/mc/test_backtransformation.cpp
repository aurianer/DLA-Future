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
#include "dlaf/common/index2d.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/matrix/copy.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/matrix_base.h"
#include "dlaf/matrix/matrix_output.h"
#include "dlaf/util_matrix.h"
//#include "dlaf_test/comm_grids/grids_6_ranks.h"
#include "dlaf_test/matrix/util_matrix.h"
#include "dlaf_test/matrix/util_matrix_blas.h"
#include "dlaf_test/matrix/util_matrix_local.h"
#include "dlaf_test/util_types.h"

using namespace dlaf;
using namespace dlaf::comm;
using namespace dlaf::common;
using namespace dlaf::matrix;
using namespace dlaf::matrix::internal;
using namespace dlaf::matrix::test;
using namespace dlaf::test;
using namespace testing;

//::testing::Environment* const comm_grids_env =
//    ::testing::AddGlobalTestEnvironment(new CommunicatorGrid6RanksEnvironment);

template <typename Type>
class BackTransformationSolverLocalTest : public ::testing::Test {};
TYPED_TEST_SUITE(BackTransformationSolverLocalTest, MatrixElementTypes);

//template <typename Type>
//class BackTransformationSolverDistributedTest : public ::testing::Test {
//public:
//  const std::vector<CommunicatorGrid>& commGrids() {
//    return comm_grids;
//}
//};
//TYPED_TEST_SUITE(BackTransformationSolverDistributedTest, double);

GlobalElementSize globalTestSize(const LocalElementSize& size) {
  return {size.rows(), size.cols()};
}

template<class T>
void set_zero(Matrix<T, Device::CPU>& mat) {
  set(mat, [](auto&&){return static_cast<T>(0.0);});
}

const std::vector<std::tuple<SizeType, SizeType, SizeType, SizeType>> sizes = {
    {3, 3, 1, 1}, {3, 3, 3, 3},
    {4, 4, 1, 1}, {4, 4, 4, 4}, {4, 4, 2, 2}, {4, 2, 2, 2}, //{4, 1, 1, 1}, 
    {6, 6, 3, 3}, {6, 3, 3, 3},   // {6, 1, 3, 3},
    {12, 2, 2, 2}, // {12, 2, 4, 4}, {12, 2, 3, 3},
    {12, 12, 2, 2}, {12, 12, 3, 3}, {12, 12, 4, 4},
    //{12, 24, 2, 2}, {12, 24, 3, 3}, {12, 24, 4, 4},
    //{24, 12, 4, 4}
};

template<class T>
MatrixLocal<T> makeLocal(const Matrix<const T, Device::CPU>& matrix) {
  return {matrix.size(), matrix.distribution().blockSize()};
}

template <class T>
void testBacktransformationEigenv(SizeType m, SizeType n, SizeType mb, SizeType nb) {

  LocalElementSize sizeC(m, n);
  TileElementSize blockSizeC(mb, nb);
  Matrix<T, Device::CPU> mat_c(sizeC, blockSizeC);
  dlaf::matrix::util::set_random(mat_c);
  //std::cout << "Random matrix C" << std::endl;
  //printElements(mat_c);

  LocalElementSize sizeV(m, m);
  TileElementSize blockSizeV(mb, mb);
  Matrix<T, Device::CPU> mat_v(sizeV, blockSizeV);
  dlaf::matrix::util::set_random(mat_v);
  //std::cout << "Random matrix V" << std::endl;
  //printElements(mat_v);
  
  // Impose orthogonality: Q = I - v tau v^H is orthogonal (Q Q^H = I)
  // leads to tau = 2/(vT v) for real 
  // TODO: COMPLEX!!
  LocalElementSize sizeTau(m, 1);
  TileElementSize blockSizeTau(1, 1);
  Matrix<T, Device::CPU> mat_tau(sizeTau, blockSizeTau);
  
  LocalElementSize sizeT(mb, m);
  TileElementSize blockSizeT(mb, nb);
  Matrix<T, Device::CPU> mat_t(sizeT, blockSizeT);
  set_zero(mat_t);

  comm::CommunicatorGrid comm_grid(MPI_COMM_WORLD, 1, 1, common::Ordering::ColumnMajor);

  // Copy C matrix locally
  auto mat_c_loc = dlaf::matrix::test::all_gather<T>(mat_c, comm_grid);

  // Copy T matrix locally
  auto mat_t_loc = dlaf::matrix::test::all_gather<T>(mat_t, comm_grid);

  // Copy V matrix locally
  auto mat_v_loc = dlaf::matrix::test::all_gather<T>(mat_v, comm_grid);
  // Reset diagonal and upper values of V
  lapack::laset(lapack::MatrixType::General, std::min(m,mb), mat_v_loc.size().cols(), 0, 0, mat_v_loc.ptr(), mat_v_loc.ld());
  if (m > mb) {
    lapack::laset(lapack::MatrixType::Upper, mat_v_loc.size().rows()-mb, mat_v_loc.size().cols(), 0, 1, mat_v_loc.ptr(GlobalElementIndex{mb, 0}), mat_v_loc.ld());
  }

  MatrixLocal<T> taus({m, 1}, {1, 1});
  // Compute taus (real case: tau = 2 / v^H v)
  const int tottaus = (m/mb-1)*mb-1;
  for (SizeType i = tottaus; i > -1; --i) {
    const GlobalElementIndex v_offset{i, i};
    auto dotprod = blas::dot(m-i, mat_v_loc.ptr(v_offset), 1, mat_v_loc.ptr(v_offset), 1);
    auto tau = 2.0/dotprod;
    taus({i,0}) = tau;
    //std::cout << "tau (" << i << ") " << tau << std::endl;
    lapack::larf(lapack::Side::Left, m-i, n, mat_v_loc.ptr(v_offset), 1, tau, mat_c_loc.ptr(GlobalElementIndex{i,0}), mat_c_loc.ld());
  }

//  std::cout << " " << std::endl;
//  std::cout << " Result simple way " << std::endl;
//  dlaf::matrix::test::print(format::numpy{}, "mat Cloc ", mat_c_loc, std::cout);
//  std::cout << " " << std::endl;
  
  for (SizeType i = mat_t.nrTiles().cols()-1; i > -1; --i) {
      const GlobalElementIndex offset{i*nb, i*nb};
      const GlobalElementIndex tau_offset{i*nb, 0};
      auto tile_t = mat_t(LocalTileIndex{0,i}).get();
      lapack::larft(lapack::Direction::Forward, lapack::StoreV::Columnwise, mat_v.size().rows()-i*nb, nb, mat_v_loc.ptr(offset), mat_v_loc.ld(), taus.ptr(tau_offset), tile_t.ptr(), tile_t.ld());
  }
  
  //std::cout << " " << std::endl;
  //std::cout << "Matrix T: " << mat_t << std::endl;
  //printElements(mat_t);
  
  solver::backTransformation<Backend::MC>(mat_c, mat_v, mat_t);

  //std::cout << " " << std::endl;
  //std::cout << "Output " << std::endl;
  //printElements(mat_c);

  auto result = [& dist = mat_c.distribution(),
		 &mat_local = mat_c_loc](const GlobalElementIndex& element) {
    const auto tile_index = dist.globalTileIndex(element);
    const auto tile_element = dist.tileElementIndex(element);
    return mat_local.tile_read(tile_index)(tile_element);
  };
  
  double error = 0.01;
  CHECK_MATRIX_NEAR(result, mat_c, error, error);
}


TYPED_TEST(BackTransformationSolverLocalTest, Correctness_random) {

  SizeType m, n, mb, nb;

  for (auto sz : sizes) {
    std::tie(m, n, mb, nb) = sz;
    testBacktransformationEigenv<TypeParam>(m, n, mb, nb);
  }
    
}
