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

#include "dlaf/lapack_tile.h"  // Just for lapack.hh workaround

#include "dlaf/common/pipeline.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/functions_sync.h"
#include "dlaf/communication/sync/broadcast.h"
#include "dlaf/matrix.h"
#include "dlaf/matrix/copy.h"
#include "dlaf/memory/memory_view.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"

#include "dlaf_test/comm_grids/grids_6_ranks.h"
#include "dlaf_test/matrix/matrix_local.h"
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

TYPED_TEST_SUITE(ReductionToBandTest, MatrixElementTypes);

const std::vector<LocalElementSize> square_sizes{{3, 3}, {12, 12}, {24, 24}};
const std::vector<TileElementSize> square_block_sizes{{3, 3}};

template <class T>
MatrixLocal<T> makeLocal(const Matrix<const T, Device::CPU>& matrix) {
  return {matrix.size(), matrix.distribution().blockSize()};
}

template <class T, Device device>  // TODO add tile_selector predicate
void all_gather(Matrix<const T, device>& source, MatrixLocal<T>& dest,
                comm::CommunicatorGrid comm_grid) {
  const auto& dist_source = source.distribution();
  const auto rank = dist_source.rankIndex();

  for (const auto& ij_tile : iterate_range2d(dist_source.nrTiles())) {
    const auto owner = dist_source.rankGlobalTile(ij_tile);

    auto& dest_tile = dest(ij_tile);

    if (owner == rank) {
      const auto& source_tile = source.read(ij_tile).get();
      comm::sync::broadcast::send(comm_grid.fullCommunicator(), source_tile);
      copy(source_tile, dest_tile);
    }
    else {
      comm::sync::broadcast::receive_from(comm_grid.rankFullCommunicator(owner),
                                          comm_grid.fullCommunicator(), dest_tile);
    }
  }
}

template <class T>
void mirror_on_diag(const Tile<T, Device::CPU>& tile) {
  DLAF_ASSERT(square_size(tile), "");

  for (SizeType j = 0; j < tile.size().cols(); j++)
    for (SizeType i = j; i < tile.size().rows(); ++i)
      tile({j, i}) = dlaf::conj(tile({i, j}));
}

template <class T>
void copy_transposed(const Tile<const T, Device::CPU>& from, const Tile<T, Device::CPU>& to) {
  DLAF_ASSERT(square_size(from), "");
  DLAF_ASSERT(equal_size(from, to), from.size(), to.size());

  for (SizeType j = 0; j < from.size().cols(); j++)
    for (SizeType i = 0; i < from.size().rows(); ++i)
      to({j, i}) = dlaf::conj(from({i, j}));
}

// band_size in elements
template <class T>
void setup_sym_band(MatrixLocal<T>& matrix, const SizeType& band_size) {
  DLAF_ASSERT(band_size == matrix.blockSize().rows(), "not yet implemented", band_size,
              matrix.blockSize().rows());
  DLAF_ASSERT(band_size % matrix.blockSize().rows() == 0, "not yet implemented", band_size,
              matrix.blockSize().rows());

  DLAF_ASSERT(square_blocksize(matrix), matrix.blockSize());
  DLAF_ASSERT(square_size(matrix), matrix.blockSize());

  const auto k_diag = band_size / matrix.blockSize().rows();

  // setup band "edge" and its tranposed
  for (SizeType j = 0; j < matrix.nrTiles().cols(); ++j) {
    const GlobalTileIndex ij{j + k_diag, j};

    if (!ij.isIn(matrix.nrTiles()))
      continue;

    const auto& tile_lo = matrix(ij);

    // setup the strictly lower to zero
    // clang-format off
    lapack::laset(
        lapack::MatrixType::Lower,
        tile_lo.size().rows() - 1, tile_lo.size().cols() - 1,
        0, 0,
        tile_lo.ptr({1, 0}), tile_lo.ld());
    // clang-format on

    copy_transposed(matrix(ij), matrix(common::transposed(ij)));
  }

  // setup zeros in both lower and upper out-of-band
  for (SizeType j = 0; j < matrix.nrTiles().cols(); ++j) {
    for (SizeType i = j + k_diag + 1; i < matrix.nrTiles().rows(); ++i) {
      const GlobalTileIndex ij{i, j};
      if (!ij.isIn(matrix.nrTiles()))
        continue;

      dlaf::matrix::test::set(matrix(ij), 0);
      dlaf::matrix::test::set(matrix(common::transposed(ij)), 0);
    }
  }

  // mirror band (diagonal tiles are correctly set just in the lower part)
  for (SizeType k = 0; k < matrix.nrTiles().rows(); ++k) {
    const GlobalTileIndex kk{k, k};
    const auto& tile = matrix(kk);

    mirror_on_diag(tile);
  }
}

template <class T>
auto all_gather_taus(const SizeType k, const SizeType chunk_size, const SizeType band_size,
                     std::vector<hpx::shared_future<std::vector<T>>> fut_local_taus,
                     comm::CommunicatorGrid comm_grid) {
  using namespace dlaf::comm::sync;

  std::vector<T> taus;
  taus.reserve(to_sizet(k));

  hpx::wait_all(fut_local_taus);
  auto local_taus = hpx::util::unwrap(fut_local_taus);

  DLAF_ASSERT(band_size == chunk_size, band_size, chunk_size);
  DLAF_ASSERT(k % chunk_size == 0, k, band_size);

  const auto n_chunks = k / chunk_size;

  for (auto index_chunk = 0; index_chunk < n_chunks; ++index_chunk) {
    const auto owner = index_chunk % comm_grid.size().cols();
    const bool is_owner = owner == comm_grid.rank().col();

    std::vector<T> chunk_data;
    if (is_owner) {
      const auto index_chunk_local = to_sizet(index_chunk / comm_grid.size().cols());
      chunk_data = local_taus[index_chunk_local];
      broadcast::send(comm_grid.rowCommunicator(),
                      common::make_data(chunk_data.data(), chunk_data.size()));
    }
    else {
      chunk_data.resize(to_sizet(chunk_size));
      broadcast::receive_from(owner, comm_grid.rowCommunicator(),
                              common::make_data(chunk_data.data(), chunk_data.size()));
    }

    // copy each chunk contiguously
    std::copy(chunk_data.begin(), chunk_data.end(), std::back_inserter(taus));
  }

  return taus;
}

TYPED_TEST(ReductionToBandTest, Correctness) {
  constexpr Device device = Device::CPU;

  for (auto&& comm_grid : this->commGrids()) {
    for (const auto& size : square_sizes) {
      for (const auto& block_size : square_block_sizes) {
        const SizeType band_size = block_size.rows();
        const SizeType band_size_tiles = band_size / block_size.rows();
        const SizeType k_reflectors = size.rows() - band_size;  // -1 the last one is superflous

        Distribution distribution({size.rows(), size.cols()}, block_size, comm_grid.size(),
                                  comm_grid.rank(), {0, 0});

        // setup the reference input matrix
        Matrix<const TypeParam, device> reference = [&]() {
          Matrix<TypeParam, device> reference(distribution);
          dlaf_test::matrix::util::set_random_hermitian(reference);
          return reference;
        }();

        Matrix<TypeParam, device> matrix_a(std::move(distribution));
        dlaf::copy(reference, matrix_a);

        // Apply reduction-to-band
        DLAF_ASSERT(band_size == matrix_a.blockSize().rows(), "not yet implemented");

        auto local_taus = dlaf::EigenSolver<Backend::MC>::reduction_to_band(comm_grid, matrix_a);

        // First basic check: the reduction should not affect the strictly upper part of the input
        auto check_rest_unchanged = [&reference, &matrix_a](const GlobalElementIndex& index) {
          const auto& dist = reference.distribution();
          const auto ij_tile = dist.globalTileIndex(index);
          const auto ij_element_wrt_tile = dist.tileElementIndex(index);

          const bool is_in_upper = index.row() < index.col();

          if (!is_in_upper)
            return matrix_a.read(ij_tile).get()(ij_element_wrt_tile);
          else
            return reference.read(ij_tile).get()(ij_element_wrt_tile);
        };
        CHECK_MATRIX_NEAR(check_rest_unchanged, matrix_a, 0, 1e-3);

        // The distributed result of reduction to band has the following characteristics:
        // - just lower part is relevant (diagonal is included)
        // - it stores the lower part of the band (B)
        // - and the matrix V with all the reflectors (the 1 component of each reflector is omitted)
        //
        // Each rank collects locally the full matrix by storing each part separately
        // - the B matrix, which is then "completed" by zeroing elements out of the band, and by mirroring the
        //    existing part of the band
        // - the V matrix, whose relevant part is the submatrix underneath the band

        auto mat_b = makeLocal(matrix_a);
        all_gather(matrix_a, mat_b, comm_grid);
        setup_sym_band(mat_b, mat_b.blockSize().rows());

        auto mat_v = makeLocal(matrix_a);
        all_gather(matrix_a, mat_v, comm_grid);
        // TODO FIXME mat_v can be smaller, but then all_gather must copy a submatrix
        const GlobalTileIndex v_offset{band_size_tiles, 0};

        // Finally, collect locally tau values, which together with V allow to apply the Q transformation
        auto taus = all_gather_taus(k_reflectors, block_size.rows(), band_size, local_taus, comm_grid);
        DLAF_ASSERT(to_SizeType(taus.size()) == k_reflectors, taus.size(), k_reflectors);

        // Now that all input are collected locally, it's time to apply the transformation,
        // ...but just if there is any
        if (v_offset.isIn(mat_v.nrTiles())) {
          // Reduction to band returns a sequence of transformations applied from left and right to A
          // allowing to reduce the matrix A to a band matrix B
          //
          // Hn* ... H2* H1* A H1 H2 ... Hn
          // Q* A Q = B
          //
          // Applying the inverse of the same transformations, we can go from B to A
          // Q B Q* = A
          // Q = H1 H2 ... Hn
          // H1 H2 ... Hn B Hn* ... H2* H1*

          // apply from left...
          const GlobalTileIndex left_offset{band_size_tiles, 0};
          const GlobalElementSize left_size{mat_b.size().rows() - band_size, mat_b.size().cols()};
          // clang-format off
          lapack::unmqr(lapack::Side::Left, lapack::Op::NoTrans,
            left_size.rows(), left_size.cols(), k_reflectors,
            mat_v(v_offset).ptr(), mat_v.ld(),
            taus.data(),
            mat_b(left_offset).ptr(), mat_b.ld());
          // clang-format on

          // ... and from right
          const GlobalTileIndex right_offset = common::transposed(left_offset);
          const GlobalElementSize right_size = common::transposed(left_size);
          // clang-format off
          lapack::unmqr(lapack::Side::Right, lapack::Op::ConjTrans,
            right_size.rows(), right_size.cols(), k_reflectors,
            mat_v(v_offset).ptr(), mat_v.ld(),
            taus.data(),
            mat_b(right_offset).ptr(), mat_b.ld());
          // clang-format on
        }

        // Eventually, check the result obtained by applying the inverse transformation equals the original matrix
        for (const auto& index : common::iterate_range2d(reference.distribution().localNrTiles())) {
          const auto global_index = reference.distribution().globalTileIndex(index);
          CHECK_TILE_NEAR(reference.read(index).get(), mat_b(global_index), 1e-3, 1e-3);
        }
      }
    }
  }
}
