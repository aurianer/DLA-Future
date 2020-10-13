//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/common/index2d.h"
#include "dlaf/eigensolver/mc.h"

#include <gtest/gtest.h>

#include "dlaf/communication/communicator.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/sync/broadcast.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/matrix.h"
#include "dlaf/memory/memory_view.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"
#include "dlaf/matrix/copy.h"
#include "dlaf/communication/functions_sync.h"

#include "dlaf_test/matrix/matrix_local.h"
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

template <class T>
using MatrixLocal = dlaf::test::MatrixLocal<T>;

::testing::Environment* const comm_grids_env =
    ::testing::AddGlobalTestEnvironment(new CommunicatorGrid6RanksEnvironment);

template <typename Type>
class ReductionToBandTest : public ::testing::Test {
public:
  const std::vector<CommunicatorGrid>& commGrids() {
    return comm_grids;
  }
};

using NonComplexTypes = ::testing::Types<double, float>;
TYPED_TEST_SUITE(ReductionToBandTest, NonComplexTypes); //MatrixElementTypes);

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

  ss.str("");
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

template <class T>
MatrixLocal<T> createLocalMatrix(SizeType rows, SizeType cols, TileElementSize blocksize) {
  return {{rows, cols}, std::move(blocksize)};
}

template <class T>
auto makeLocal(const Matrix<const T, Device::CPU>& matrix) {
  const auto from_size = matrix.size();
  return createLocalMatrix<T>(from_size.rows(), from_size.cols(), matrix.distribution().blockSize());
}

template <class T, Device device> // TODO add tile_selector predicate
void all_gather(Matrix<const T, device>& source, MatrixLocal<T>& dest, comm::CommunicatorGrid comm_grid) {
  const auto& dist_source = source.distribution();
  const auto rank = dist_source.rankIndex();

  for (const auto& ij_tile : iterate_range2d(dist_source.nrTiles())) {
    const auto owner = dist_source.rankGlobalTile(ij_tile);

    auto& dest_tile = dest(ij_tile);

    if (owner == rank) {
      const auto& source_tile = source.read(ij_tile).get();
      comm::sync::broadcast::send(
          comm_grid.fullCommunicator(),
          source_tile);
      copy(source_tile, dest_tile);
    }
    else {
      comm::sync::broadcast::receive_from(
          comm_grid.rankFullCommunicator(owner),
          comm_grid.fullCommunicator(),
          dest_tile);
    }
  }
}

template <class T>
void mirror_on_diag(const Tile<T, Device::CPU>& tile) {
  DLAF_ASSERT(square_size(tile), "");

  for (SizeType j = 0; j < tile.size().cols(); j++)
    for (SizeType i = j; i < tile.size().rows(); ++i)
      tile({j, i}) = tile({i, j});
}

template <class T>
void copy_transposed(const Tile<const T, Device::CPU>& from, const Tile<T, Device::CPU>& to) {
  DLAF_ASSERT(square_size(from), "");
  DLAF_ASSERT(equal_size(from, to), from.size(), to.size());

  for (SizeType j = 0; j < from.size().cols(); j++)
    for (SizeType i = 0; i < from.size().rows(); ++i)
      to({j, i}) = from({i, j});
}

// band_size in elements
template <class T>
void setup_sym_band(MatrixLocal<T>& matrix, const SizeType& band_size) {
  DLAF_ASSERT(band_size == matrix.blockSize().rows(), "not yet implemented", band_size, matrix.blockSize().rows());
  DLAF_ASSERT(band_size % matrix.blockSize().rows() == 0, "not yet implemented", band_size, matrix.blockSize().rows());

  DLAF_ASSERT(square_blocksize(matrix), matrix.blockSize());
  DLAF_ASSERT(square_size(matrix), matrix.blockSize());

  const auto k_diag = band_size / matrix.blockSize().rows();

  // TODO setup band edge and its tranposed
  for (SizeType j = 0; j < matrix.nrTiles().cols(); ++j) {
    const GlobalTileIndex ij{j + k_diag, j};

    if (!ij.isIn(matrix.nrTiles()))
      continue;

    const auto& tile_lo = matrix(ij);

    // setup the strictly lower to zero
    lapack::laset(
        lapack::MatrixType::Lower,
        tile_lo.size().rows() - 1, tile_lo.size().cols() - 1,
        0, 0,
        tile_lo.ptr({1, 0}), tile_lo.ld());

    copy_transposed(matrix(ij), matrix(common::transposed(ij)));
  }

  // TODO setup zeros in both lower and upper
  for (SizeType j = 0; j < matrix.nrTiles().cols(); ++j) {
    for (SizeType i = j + k_diag + 1; i < matrix.nrTiles().rows(); ++i) {
      const GlobalTileIndex ij{i, j};
      if (!ij.isIn(matrix.nrTiles()))
          continue;

      dlaf::matrix::test::set(matrix(ij), 0);
      dlaf::matrix::test::set(matrix(common::transposed(ij)), 0);
    }
  }

  // TODO mirror on_diag
  for (SizeType k = 0; k < matrix.nrTiles().rows(); ++k) {
    const GlobalTileIndex kk{k, k};
    const auto& tile = matrix(kk);

    mirror_on_diag(tile);
  }
}

TYPED_TEST(ReductionToBandTest, Correctness) {
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
        auto A = makeLocal(reference);
        auto B = makeLocal(matrix_a);
        auto Q = makeLocal(matrix_a); // TODO FIXME Q can be smaller, but then all_gather must copy a submatrix

        all_gather(reference, A, comm_grid);

        all_gather(matrix_a, B, comm_grid);
        setup_sym_band(B, B.blockSize().rows());

        all_gather(matrix_a, Q, comm_grid);

        //print(A, "A");
        //print(B, "B");
        //print(Q, "Q");

        // TODO collect also taus

        // TODO su tutti i rank usiamo [[z,c]un,[s,d]or]mqr per applicare Q da sinistra e da destra

        // TODO ogni rank fa il check con la parte uplo della A originale.
      }
    }
  }
}
