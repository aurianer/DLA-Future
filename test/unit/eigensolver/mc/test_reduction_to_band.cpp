//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/eigensolver/mc.h"

#include <gtest/gtest.h>

#include "dlaf/communication/communicator.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/sync/broadcast.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/matrix.h"
#include "dlaf/util_matrix.h"
#include "dlaf/matrix/copy.h"
#include "dlaf/communication/functions_sync.h"

#include "dlaf_test/comm_grids/grids_6_ranks.h"
#include "dlaf_test/matrix/util_matrix.h"
#include "dlaf_test/matrix/util_tile.h"
#include "dlaf_test/util_types.h"

using namespace dlaf;
using namespace dlaf::comm;
using namespace dlaf::matrix;
using namespace dlaf::matrix::test;
using namespace dlaf_test;
using namespace testing;

::testing::Environment* const comm_grids_env =
    ::testing::AddGlobalTestEnvironment(new CommunicatorGrid6RanksEnvironment);

template <typename Type>
class CholeskyDistributedTest : public ::testing::Test {
public:
  const std::vector<CommunicatorGrid>& commGrids() {
    return comm_grids;
  }
};

using NonComplexTypes = ::testing::Types<double, float>;
TYPED_TEST_SUITE(CholeskyDistributedTest, NonComplexTypes); //MatrixElementTypes);

const std::vector<LocalElementSize> square_sizes{{8, 8}};
const std::vector<TileElementSize> square_block_sizes{{2, 2}};

template <class T>
void print(Matrix<const T, Device::CPU>& matrix, std::string prefix) {
  using common::iterate_range2d;

  const auto& distribution = matrix.distribution();

  std::ostringstream ss;
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  ss << prefix << rank;
  prefix = ss.str();

  ss.clear();
  ss << prefix << " = np.zeros((" << distribution.size() << "))" << std::endl;

  for (const auto& index_tile : iterate_range2d(distribution.localNrTiles())) {
    const auto& tile = matrix.read(index_tile).get();

    for (const auto& index_el : iterate_range2d(tile.size())) {
      GlobalElementIndex index_g{
          distribution.template globalElementFromLocalTileAndTileElement<Coord::Row>(index_tile.row(),
                                                                                     index_el.row()),
          distribution.template globalElementFromLocalTileAndTileElement<Coord::Col>(index_tile.col(),
                                                                                     index_el.col()),
      };
      ss << prefix << "[" << index_g.row() << "," << index_g.col() << "] = " << tile(index_el)
         << std::endl;
    }
  }

  std::cout << ss.str() << std::endl;
}

template <class T, Device device>
Matrix<T, device> makeLocal(Matrix<const T, device>& matrix) {
  const auto& dist = matrix.distribution();
  return {{dist.size().rows(), dist.size().cols()}, dist.blockSize()};
}

template <class T, Device device>
Matrix<T, device> makeLocalQ(Matrix<const T, device>& matrix, const SizeType band_size) {
  const auto& dist = matrix.distribution();

  const SizeType height = dist.size().rows() - band_size;

  return {{
    height,
    std::min(height, dist.size().cols())
  }, dist.blockSize()};
}

template <class T, Device device> // TODO add tile_selector predicate
void all_gather(Matrix<const T, device>& source, Matrix<T, device>& dest,
    common::Pipeline<comm::CommunicatorGrid>& serial_comm) {
  const auto& dist_source = source.distribution();
  const auto rank = dist_source.rankIndex();

  for (const auto& ij_tile : iterate_range2d(dest.distribution().nrTiles())) {
    const auto owner = dist_source.rankGlobalTile(ij_tile);

    auto&& dest_tile = dest(ij_tile);

    if (owner == rank)
      hpx::dataflow(hpx::util::unwrapping([](auto&& source, auto&& dest, auto&& comm_wrapper) {
        comm::sync::broadcast::send(
          comm_wrapper().fullCommunicator(),
          source);
        copy(source, dest);
      }), source.read(ij_tile), dest_tile, serial_comm());
    else
      hpx::dataflow(hpx::util::unwrapping([=](auto&& dest, auto&& comm_wrapper) {
        auto&& comm = comm_wrapper();
        comm::sync::broadcast::receive_from(
          comm.rankFullCommunicator(owner),
          comm.fullCommunicator(),
          dest);
      }), dest_tile, serial_comm());
  }
}

template <class T, Device device>
void set_out_of_band(Matrix<T, device>& matrix, const SizeType& band_size) {
  DLAF_ASSERT(band_size == matrix.blockSize().rows(), "not yet implemented", band_size, matrix.blockSize().rows());

  DLAF_ASSERT(square_blocksize(matrix), matrix.blockSize());
  DLAF_ASSERT(square_size(matrix), matrix.blockSize());

  const auto k_diag = band_size / matrix.blockSize().rows();

  for (SizeType j_tile = 0; j_tile < matrix.nrTiles().cols(); ++j_tile) {
    for (SizeType i_tile = k_diag + j_tile; i_tile < matrix.nrTiles().rows(); ++i_tile) {
      LocalTileIndex ij_tile{i_tile, j_tile};

      const bool is_k_diag = i_tile == k_diag + j_tile;

      hpx::dataflow(hpx::util::unwrapping([is_k_diag](const auto& tile_lo, const auto& tile_up) {
        if (is_k_diag) {
          // set strict lower...
          lapack::laset(lapack::MatrixType::Lower,
            tile_lo.size().rows() - 1, tile_lo.size().cols() - 1,
            0, 0,
            tile_lo.ptr({1, 0}), tile_lo.ld());
          // ... and strict upper
          lapack::laset(lapack::MatrixType::Upper,
            tile_up.size().rows() - 1, tile_up.size().cols() - 1,
            0, 0,
            tile_up.ptr({0, 1}), tile_up.ld());
        }
        else {
          dlaf::matrix::test::set(tile_lo, 0);
          dlaf::matrix::test::set(tile_up, 0);
        }
      }), matrix(ij_tile), matrix(common::transposed(ij_tile)));
    }
  }
}

TYPED_TEST(CholeskyDistributedTest, Correctness) {
  constexpr Device device = Device::CPU;

  for (auto&& comm_grid : this->commGrids()) {
    for (const auto& size : square_sizes) {
      for (const auto& block_size : square_block_sizes) {
        const SizeType band_size = block_size.rows();

        // TODO setup a matrix
        Matrix<const TypeParam, device> reference = [&](){
          Matrix<TypeParam, device> reference(size, block_size);
          dlaf_test::matrix::util::set_random_hermitian(reference);
          return reference;
        }();

        Matrix<TypeParam, device> matrix_a(size, block_size);
        dlaf::copy(reference, matrix_a);

        // TODO apply reduction-to-band
        DLAF_ASSERT(band_size == matrix_a.distribution().blockSize().rows(), "not yet implemented");

        //dlaf::EigenSolver<Backend::MC>::reduction_to_band(comm_grid, matrix_a);

        // TODO check che la parte non-uplo non è cambiata
        auto check_uplo_unchanged = [&reference, &matrix_a](const GlobalElementIndex& index) {
          const auto& dist = reference.distribution();
          const auto ij_tile = dist.globalTileIndex(index);
          const auto ij_element_wrt_tile = dist.tileElementIndex(index);

          const bool is_in_upper = index.row() < index.col();

          if (!is_in_upper)
            return matrix_a.read(ij_tile).get()(ij_element_wrt_tile);
          else
            return reference.read(ij_tile).get()(ij_element_wrt_tile);
        };
        CHECK_MATRIX_NEAR(check_uplo_unchanged, matrix_a, 0, 1e-3);

        // TODO "allreduce" del risultato in due "matrici lapack" (Q e banda (salvata come general))
        // Each rank must collect the Q and the B (Q with the 1, B with both up/low and zero outside the band)
        common::Pipeline<comm::CommunicatorGrid> serial_comm(comm_grid);

        Matrix<TypeParam, device> A = makeLocal(reference);
        Matrix<TypeParam, device> B = makeLocal(matrix_a);
        Matrix<TypeParam, device> Q = makeLocalQ(matrix_a, band_size);

        all_gather(reference, A, serial_comm);

        all_gather(matrix_a, B, serial_comm);
        set_out_of_band(B, B.blockSize().rows());

        all_gather(matrix_a, Q, serial_comm);

        print(A, "A");
        print(B, "B");
        print(Q, "Q");

        // TODO su tutti i rank usiamo [[z,c]un,[s,d]or]mqr per applicare Q da sinistra e da destra

        // TODO ogni rank fa il check con la parte uplo della A originale.
      }
    }
  }
}
