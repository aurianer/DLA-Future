//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include <hpx/include/parallel_executors.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/include/util.hpp>

#include "dlaf/blas_tile.h"
#include "dlaf/common/index2d.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/common/range2d.h"
#include "dlaf/common/vector.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/executor.h"
#include "dlaf/communication/functions_sync.h"
#include "dlaf/factorization/cholesky/api.h"
#include "dlaf/lapack_tile.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/panel.h"

#include "dlaf/memory/memory_view.h"
#include "dlaf/util_matrix.h"

namespace dlaf {
namespace factorization {
namespace internal {

template <class T>
void potrf_diag_tile(hpx::threads::executors::pool_executor executor_hp,
                     hpx::future<matrix::Tile<T, Device::CPU>> matrix_tile) {
  hpx::dataflow(executor_hp, hpx::util::unwrapping(tile::potrf<T, Device::CPU>), blas::Uplo::Lower,
                std::move(matrix_tile));
}

template <class T>
void trsm_panel_tile(hpx::threads::executors::pool_executor executor_hp,
                     hpx::shared_future<matrix::Tile<const T, Device::CPU>> kk_tile,
                     hpx::future<matrix::Tile<T, Device::CPU>> matrix_tile) {
  hpx::dataflow(executor_hp, hpx::util::unwrapping(tile::trsm<T, Device::CPU>), blas::Side::Right,
                blas::Uplo::Lower, blas::Op::ConjTrans, blas::Diag::NonUnit, 1.0, std::move(kk_tile),
                std::move(matrix_tile));
}

template <class T>
void herk_trailing_diag_tile(hpx::threads::executors::pool_executor trailing_matrix_executor,
                             hpx::shared_future<matrix::Tile<const T, Device::CPU>> panel_tile,
                             hpx::future<matrix::Tile<T, Device::CPU>> matrix_tile) {
  hpx::dataflow(trailing_matrix_executor, hpx::util::unwrapping(tile::herk<T, Device::CPU>),
                blas::Uplo::Lower, blas::Op::NoTrans, -1.0, panel_tile, 1.0, std::move(matrix_tile));
}

template <class T>
void gemm_trailing_matrix_tile(hpx::threads::executors::pool_executor trailing_matrix_executor,
                               hpx::shared_future<matrix::Tile<const T, Device::CPU>> panel_tile,
                               hpx::shared_future<matrix::Tile<const T, Device::CPU>> col_panel,
                               hpx::future<matrix::Tile<T, Device::CPU>> matrix_tile) {
  hpx::dataflow(trailing_matrix_executor, hpx::util::unwrapping(tile::gemm<T, Device::CPU>),
                blas::Op::NoTrans, blas::Op::ConjTrans, -1.0, std::move(panel_tile),
                std::move(col_panel), 1.0, std::move(matrix_tile));
}

template <class T>
struct Cholesky<Backend::MC, Device::CPU, T> {
  static void call_L(Matrix<T, Device::CPU>& mat_a);
  static void call_L(comm::CommunicatorGrid grid, Matrix<T, Device::CPU>& mat_a);
};

template <class T>
void Cholesky<Backend::MC, Device::CPU, T>::call_L(Matrix<T, Device::CPU>& mat_a) {
  using hpx::threads::executors::pool_executor;
  using hpx::threads::thread_priority_high;
  using hpx::threads::thread_priority_default;

  pool_executor executor_hp("default", thread_priority_high);
  pool_executor executor_normal("default", thread_priority_default);

  // Number of tile (rows = cols)
  SizeType nrtile = mat_a.nrTiles().cols();

  for (SizeType k = 0; k < nrtile; ++k) {
    auto kk = LocalTileIndex{k, k};

    potrf_diag_tile(executor_hp, mat_a(kk));

    for (SizeType i = k + 1; i < nrtile; ++i) {
      // Update panel mat_a(i,k) with trsm (blas operation), using data mat_a.read(k,k)
      trsm_panel_tile(executor_hp, mat_a.read(kk), mat_a(LocalTileIndex{i, k}));
    }

    for (SizeType j = k + 1; j < nrtile; ++j) {
      // first trailing panel gets high priority (look ahead).
      auto trailing_matrix_executor = (j == k + 1) ? executor_hp : executor_normal;

      // Update trailing matrix: diagonal element mat_a(j,j), reading mat_a.read(j,k), using herk (blas operation)
      herk_trailing_diag_tile(trailing_matrix_executor, mat_a.read(LocalTileIndex{j, k}),
                              mat_a(LocalTileIndex{j, j}));

      for (SizeType i = j + 1; i < nrtile; ++i) {
        // Update remaining trailing matrix mat_a(i,j), reading mat_a.read(i,k) and mat_a.read(j,k),
        // using gemm (blas operation)
        gemm_trailing_matrix_tile(trailing_matrix_executor, mat_a.read(LocalTileIndex{i, k}),
                                  mat_a.read(LocalTileIndex{j, k}), mat_a(LocalTileIndex{i, j}));
      }
    }
  }
}

template <class T>
struct RoundRobin {
  template <class... Args>
  RoundRobin(std::size_t n, Args&& ... args) : next_index_(0) {
    for (auto i = 0; i < n; ++i)
      pool_.emplace_back(std::forward<Args>(args)...);
  }

  T& next_resource() {
    auto idx = (next_index_ + 1) % pool_.size();
    std::swap(idx, next_index_);
    return pool_[idx];
  }

  std::size_t next_index_;
  std::vector<T> pool_;
};

template <class T>
void Cholesky<Backend::MC, Device::CPU, T>::call_L(comm::CommunicatorGrid grid,
                                                   Matrix<T, Device::CPU>& mat_a) {
  using hpx::threads::executors::pool_executor;
  using hpx::threads::thread_priority_high;
  using hpx::threads::thread_priority_default;
  using comm::internal::mpi_pool_exists;

  // Set up executor on the default queue with high priority.
  pool_executor executor_hp("default", thread_priority_high);
  // Set up executor on the default queue with default priority.
  pool_executor executor_normal("default", thread_priority_default);

  // Set up MPI executor
  const comm::Index2D this_rank = grid.rank();
  const auto grid_size = grid.size();
  auto executor_mpi = (mpi_pool_exists()) ? pool_executor("mpi", thread_priority_high) : executor_hp;
  common::Pipeline<comm::CommunicatorGrid> mpi_task_chain(std::move(grid));

  matrix::Distribution const& distr = mat_a.distribution();
  const SizeType nrtile = mat_a.nrTiles().cols();

  matrix::Panel<Coord::Row, T, Device::CPU> diag_tiles(distr, {0, 0});

  constexpr std::size_t N_WORKSPACES = 3;
  RoundRobin<matrix::Panel<Coord::Col, T, Device::CPU>> panel_cols(N_WORKSPACES, distr, LocalTileIndex(0, 0));
  RoundRobin<matrix::Panel<Coord::Row, T, Device::CPU>> panel_cols_t(N_WORKSPACES, distr, LocalTileIndex(0, 0));

  for (SizeType k = 0; k < nrtile; ++k) {
    const GlobalTileIndex kk_idx(k, k);
    const comm::Index2D kk_rank = distr.rankGlobalTile(kk_idx);

    const LocalTileIndex kk_offset{
        distr.nextLocalTileFromGlobalTile<Coord::Row>(k),
        distr.nextLocalTileFromGlobalTile<Coord::Col>(k),
    };


    auto& panel_col = panel_cols.next_resource();
    auto& panel_col_t = panel_cols_t.next_resource();

    panel_col.set_offset(kk_offset);
    panel_col_t.set_offset(kk_offset);

    //// Factorization of diagonal tile and broadcast it along the `k`-th column
    // TODO skip last step tile
    if (kk_rank.col() == this_rank.col()) {
      if (kk_rank.row() == this_rank.row()) {
        potrf_diag_tile(executor_hp, mat_a(kk_idx));
        diag_tiles.set_tile(kk_offset, mat_a.read(kk_idx));
        if (kk_idx.row() < nrtile - 1)
          comm::send_tile(executor_mpi, mpi_task_chain, Coord::Col, diag_tiles.read(kk_offset));
      }
      else {
        if (kk_idx.row() < nrtile - 1)
          comm::recv_tile<T>(executor_mpi, mpi_task_chain, Coord::Col, diag_tiles(kk_offset),
                           kk_rank.row());
      }
    }

    // Update the column panel under diagonal tile and broadcast
    if (kk_rank.col() == this_rank.col()) {
      for (SizeType i = distr.nextLocalTileFromGlobalTile<Coord::Row>(k + 1);
           i < distr.localNrTiles().rows(); ++i) {
        const LocalTileIndex local_idx(Coord::Row, i);
        const LocalTileIndex ik_idx(i, distr.localTileFromGlobalTile<Coord::Col>(k));

        trsm_panel_tile(executor_hp, diag_tiles.read(kk_offset), mat_a(ik_idx));
        panel_col.set_tile(local_idx, mat_a.read(ik_idx));
      }
    }

    // TODO skip last step tile
    matrix::broadcast(executor_mpi, kk_rank.col(), panel_col, grid_size, mpi_task_chain);
    matrix::broadcast(executor_mpi, kk_rank.col(), panel_col, panel_col_t, grid_size, mpi_task_chain);

    // Iterate over the trailing matrix
    for (SizeType j_idx = k + 1; j_idx < nrtile; ++j_idx) {
      const auto owner = distr.rankGlobalTile({j_idx, j_idx});

      if (owner.col() != this_rank.col())
        continue;

      if (this_rank.row() == owner.row()) {
        pool_executor trailing_matrix_executor = (j_idx == k + 1) ? executor_hp : executor_normal;

        const auto j = distr.localTileFromGlobalTile<Coord::Col>(j_idx);
        herk_trailing_diag_tile(trailing_matrix_executor,
            panel_col_t.read({Coord::Col, j}),
            mat_a(GlobalTileIndex{j_idx, j_idx}));
      }

      for (SizeType i_idx = j_idx + 1; i_idx < nrtile; ++i_idx) {
        const auto owner_row = distr.rankGlobalTile<Coord::Row>(i_idx);

        if (owner_row != this_rank.row())
          continue;

        const auto i = distr.localTileFromGlobalTile<Coord::Row>(i_idx);
        const auto j = distr.localTileFromGlobalTile<Coord::Col>(j_idx);
        gemm_trailing_matrix_tile(executor_normal, panel_col.read({Coord::Row, i}),
            panel_col_t.read({Coord::Col, j}), mat_a(LocalTileIndex{i, j}));

      }
    }

    panel_col_t.reset();
    panel_col.reset();
  }
}

/// ---- ETI
#define DLAF_CHOLESKY_MC_ETI(KWORD, DATATYPE) \
  KWORD template struct Cholesky<Backend::MC, Device::CPU, DATATYPE>;

DLAF_CHOLESKY_MC_ETI(extern, float)
DLAF_CHOLESKY_MC_ETI(extern, double)
DLAF_CHOLESKY_MC_ETI(extern, std::complex<float>)
DLAF_CHOLESKY_MC_ETI(extern, std::complex<double>)

}
}
}
