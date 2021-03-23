//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include <hpx/include/parallel_executors.hpp>
#include <hpx/include/resource_partitioner.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/include/util.hpp>
#include <hpx/local/future.hpp>

#include <blas.hh>

#include "dlaf/solver/backtransformation/api.h"

#include "dlaf/blas_tile.h"
#include "dlaf/common/index2d.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/common/vector.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/functions_sync.h"
#include "dlaf/communication/init.h"
#include "dlaf/executors.h"
#include "dlaf/factorization/qr.h"
#include "dlaf/lapack_tile.h"
#include "dlaf/matrix/copy.h"
#include "dlaf/matrix/copy_tile.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/layout_info.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/util_matrix.h"

#include "dlaf/matrix/matrix_output.h"

namespace dlaf {
namespace solver {
namespace internal {

using namespace dlaf::matrix;

template <class T>
void set_zero(Matrix<T, Device::CPU>& mat) {
  dlaf::matrix::util::set(mat, [](auto&&) { return static_cast<T>(0.0); });
}
 
// Implementation based on:
// 1. Part of algorithm 6 "LAPACK Algorithm for the eigenvector back-transformation", page 15, PhD thesis
// "GPU Accelerated Implementations of a Generalized Eigenvalue Solver for Hermitian Matrices with
// Systematic Energy and Time to Solution Analysis" presented by Raffaele Solcà (2016)
// 2. Report "Gep + back-transformation", Alberto Invernizzi (2020)
// 3. Report "Reduction to band + back-transformation", Raffaele Solcà (2020)
// 4. G. H. Golub and C. F. Van Loan, Matrix Computations, chapter 5, The Johns Hopkins University Press
template <class T>
struct BackTransformation<Backend::MC, Device::CPU, T> {
  static void call_FC(Matrix<T, Device::CPU>& mat_c, Matrix<const T, Device::CPU>& mat_v,
                      common::internal::vector<hpx::shared_future<common::internal::vector<T>>> taus);
  static void call_FC(comm::CommunicatorGrid grid, Matrix<T, Device::CPU>& mat_c,
                      Matrix<const T, Device::CPU>& mat_v, common::internal::vector<hpx::shared_future<common::internal::vector<T>>> taus);
};

template <class T>
void BackTransformation<Backend::MC, Device::CPU, T>::call_FC(Matrix<T, Device::CPU>& mat_c,
                                                              Matrix<const T, Device::CPU>& mat_v,
                                                              common::internal::vector<hpx::shared_future<common::internal::vector<T>>> taus) {
  constexpr auto Left = blas::Side::Left;
  constexpr auto Right = blas::Side::Right;
  constexpr auto Upper = blas::Uplo::Upper;
  constexpr auto Lower = blas::Uplo::Lower;
  constexpr auto NoTrans = blas::Op::NoTrans;
  constexpr auto ConjTrans = blas::Op::ConjTrans;
  constexpr auto NonUnit = blas::Diag::NonUnit;

  using hpx::util::unwrapping;

  using hpx::execution::parallel_executor;
  using hpx::resource::get_thread_pool;
  using hpx::threads::thread_priority;

  parallel_executor executor_hp(&get_thread_pool("default"), thread_priority::high);
  parallel_executor executor_normal(&get_thread_pool("default"), thread_priority::default_);

  const SizeType m = mat_c.nrTiles().rows();
  const SizeType n = mat_c.nrTiles().cols();
  const SizeType mb = mat_c.blockSize().rows();
  const SizeType nb = mat_c.blockSize().cols();
  const SizeType ms = mat_c.size().rows();
  const SizeType ns = mat_c.size().cols();

  // Matrix T
  comm::CommunicatorGrid comm_grid(MPI_COMM_WORLD, 1, 1, common::Ordering::ColumnMajor);
  common::Pipeline<comm::CommunicatorGrid> serial_comm(comm_grid);
  int tottaus;
  if (ms < mb || ms == 0 || ns == 0)
    tottaus = 0;
  else
    tottaus = (ms / mb - 1) * mb + ms % mb;
  
  if (tottaus == 0)
    return;

  LocalElementSize sizeT(tottaus, tottaus);
  TileElementSize blockSizeT(mb, mb);
  Matrix<T, Device::CPU> mat_t(sizeT, blockSizeT);
  set_zero(mat_t);
  
  Matrix<T, Device::CPU> mat_vv({mat_v.size().rows(), mb}, mat_v.blockSize());
  Matrix<T, Device::CPU> mat_w({mat_v.size().rows(), mb}, mat_v.blockSize());
  Matrix<T, Device::CPU> mat_w2({mb, mat_c.size().cols()}, mat_c.blockSize());

  SizeType last_mb;

  if (mat_v.blockSize().cols() == 1) {
    last_mb = 1;
  }
  else {
    if (mat_v.size().cols() % mat_v.blockSize().cols() == 0)
      last_mb = mat_v.blockSize().cols();
    else
      last_mb = mat_v.size().cols() % mat_v.blockSize().cols();
  }
  
  Matrix<T, Device::CPU> mat_vv_last({mat_v.size().rows(), last_mb}, mat_v.blockSize());
  Matrix<T, Device::CPU> mat_w_last({mat_v.size().rows(), last_mb}, mat_v.blockSize());
  Matrix<T, Device::CPU> mat_w2_last({last_mb, mat_c.size().cols()}, mat_c.blockSize());

  const SizeType reflectors = (mat_v.size().cols() < mat_v.size().rows()) ? mat_v.nrTiles().rows() - 2 : mat_v.nrTiles().cols() - 2;
  
  for (SizeType k = reflectors; k > -1; --k) {
    bool is_last = (k == reflectors) ? true : false;

    void (&cpyReg)(TileElementSize, TileElementIndex, const matrix::Tile<const T, Device::CPU>&,
                   TileElementIndex, const matrix::Tile<T, Device::CPU>&) = copy<T>;
    void (&cpy)(const matrix::Tile<const T, Device::CPU>&, const matrix::Tile<T, Device::CPU>&) =
        copy<T>;

    // Copy V panel into VV
    for (SizeType i = 0; i < mat_v.nrTiles().rows(); ++i) {
      if (is_last) {
        TileElementSize region = mat_vv_last.read(LocalTileIndex(i, 0)).get().size();
        TileElementIndex idx_in(0, 0);
        TileElementIndex idx_out(0, 0);
        hpx::dataflow(executor_hp, hpx::util::unwrapping(cpyReg), region, idx_in,
                      mat_v.read(LocalTileIndex(i, k)), idx_out, mat_vv_last(LocalTileIndex(i, 0)));
      }
      else {
        hpx::dataflow(executor_hp, hpx::util::unwrapping(cpy), mat_v.read(LocalTileIndex(i, k)),
                      mat_vv(LocalTileIndex(i, 0)));
      }

      // Fixing elements of VV and copying them into WH
      if (is_last) {
        auto tile_v = mat_vv_last(LocalTileIndex{i, 0}).get();
        if (i <= k) {
          lapack::laset(lapack::MatrixType::General, tile_v.size().rows(), tile_v.size().cols(), 0, 0,
                        tile_v.ptr(), tile_v.ld());
        }
        else if (i == k + 1) {
          lapack::laset(lapack::MatrixType::Upper, tile_v.size().rows(), tile_v.size().cols(), 0, 1,
                        tile_v.ptr(), tile_v.ld());
        }
        hpx::dataflow(executor_hp, hpx::util::unwrapping(cpy), mat_vv_last.read(LocalTileIndex(i, 0)),
                      mat_w_last(LocalTileIndex(i, 0)));
      }
      else {
        auto tile_v = mat_vv(LocalTileIndex{i, 0}).get();
        if (i <= k) {
          lapack::laset(lapack::MatrixType::General, tile_v.size().rows(), tile_v.size().cols(), 0, 0,
                        tile_v.ptr(), tile_v.ld());
        }
        else if (i == k + 1) {
          lapack::laset(lapack::MatrixType::Upper, tile_v.size().rows(), tile_v.size().cols(), 0, 1,
                        tile_v.ptr(), tile_v.ld());
        }
        hpx::dataflow(executor_hp, hpx::util::unwrapping(cpy), mat_vv.read(LocalTileIndex(i, 0)),
                      mat_w(LocalTileIndex(i, 0)));
      }
    }

    int taupan;
    // Reset W2 to zero
    if (is_last) {
      matrix::util::set(mat_w2_last, [](auto&&) { return 0; });
      taupan = last_mb;
    }
    else {
      matrix::util::set(mat_w2, [](auto&&) { return 0; });
      taupan = mat_v.blockSize().cols();
    }

  const GlobalTileIndex v_start{k+1, k};
  auto taus_panel = taus[k];
  
  dlaf::factorization::internal::computeTFactor<Backend::MC>(taupan, mat_v, v_start, taus_panel, mat_t(LocalTileIndex{k,k}), serial_comm);
  
    for (SizeType i = k + 1; i < m; ++i) {
      auto kk = LocalTileIndex{k, k};
      // WH = V T
      auto ik = LocalTileIndex{i, 0};
      if (is_last) {
        hpx::dataflow(executor_normal, hpx::util::unwrapping(tile::trmm<T, Device::CPU>), Right, Upper,
                      ConjTrans, NonUnit, 1.0, mat_t.read(kk), std::move(mat_w_last(ik)));
      }
      else {
        hpx::dataflow(executor_normal, hpx::util::unwrapping(tile::trmm<T, Device::CPU>), Right, Upper,
                      ConjTrans, NonUnit, 1.0, mat_t.read(kk), std::move(mat_w(ik)));
      }
    }

    for (SizeType j = 0; j < n; ++j) {
      auto kj = LocalTileIndex{0, j};
      for (SizeType i = k + 1; i < m; ++i) {
        auto ik = LocalTileIndex{i, 0};
        auto ij = LocalTileIndex{i, j};
        // W2 = W C
        if (is_last) {
          hpx::dataflow(executor_normal, hpx::util::unwrapping(tile::gemm<T, Device::CPU>), ConjTrans,
                        NoTrans, 1.0, std::move(mat_w_last(ik)), mat_c.read(ij), 1.0,
                        std::move(mat_w2_last(kj)));
        }
        else {
          hpx::dataflow(executor_normal, hpx::util::unwrapping(tile::gemm<T, Device::CPU>), ConjTrans,
                        NoTrans, 1.0, std::move(mat_w(ik)), mat_c.read(ij), 1.0, std::move(mat_w2(kj)));
        }
      }
    }
    
    for (SizeType i = k + 1; i < m; ++i) {
      auto ik = LocalTileIndex{i, 0};
      for (SizeType j = 0; j < n; ++j) {
        auto kj = LocalTileIndex{0, j};
        auto ij = LocalTileIndex{i, j};
        // C = C - V W2
        if (is_last) {
          hpx::dataflow(executor_normal, hpx::util::unwrapping(tile::gemm<T, Device::CPU>), NoTrans,
                        NoTrans, -1.0, mat_vv_last.read(ik), mat_w2_last.read(kj), 1.0,
                        std::move(mat_c(ij)));
        }
        else {
          hpx::dataflow(executor_normal, hpx::util::unwrapping(tile::gemm<T, Device::CPU>), NoTrans,
                        NoTrans, -1.0, mat_vv.read(ik), mat_w2.read(kj), 1.0, std::move(mat_c(ij)));
        }
      }
    }

  }
}

template <class T>
void BackTransformation<Backend::MC, Device::CPU, T>::call_FC(comm::CommunicatorGrid grid,
                                                              Matrix<T, Device::CPU>& mat_c,
                                                              Matrix<const T, Device::CPU>& mat_v,
                                                              common::internal::vector<hpx::shared_future<common::internal::vector<T>>> taus) {
  //  DLAF_UNIMPLEMENTED(grid);

  constexpr auto Left = blas::Side::Left;
  constexpr auto Right = blas::Side::Right;
  constexpr auto Upper = blas::Uplo::Upper;
  constexpr auto Lower = blas::Uplo::Lower;
  constexpr auto NoTrans = blas::Op::NoTrans;
  constexpr auto ConjTrans = blas::Op::ConjTrans;
  constexpr auto NonUnit = blas::Diag::NonUnit;

  using hpx::util::unwrapping;

  using hpx::execution::parallel_executor;
  using hpx::resource::get_thread_pool;
  using hpx::threads::thread_priority;

  auto executor_hp = dlaf::getHpExecutor<Backend::MC>();
  auto executor_np = dlaf::getNpExecutor<Backend::MC>();
  auto executor_mpi = dlaf::getMPIExecutor<Backend::MC>();
  
  const SizeType m = mat_c.nrTiles().rows();
  const SizeType n = mat_c.nrTiles().cols();
  const SizeType c_local_rows = mat_c.distribution().localNrTiles().rows();
  const SizeType c_local_cols = mat_c.distribution().localNrTiles().cols();
  const SizeType mb = mat_c.blockSize().rows();
  const SizeType nb = mat_c.blockSize().cols();
  const SizeType ms = mat_c.size().rows();
  const SizeType ns = mat_c.size().cols();

  auto dist_c = mat_c.distribution();
  auto dist_v = mat_v.distribution();

  const comm::Index2D this_rank = grid.rank();
  
  // Compute number of taus
  common::Pipeline<comm::CommunicatorGrid> serial_comm(grid);
  int tottaus;
  if (ms < mb || ms == 0 || ns == 0)
    tottaus = 0;
  else
    tottaus = (ms / mb - 1) * mb + ms % mb;
  
  if (tottaus == 0)
    return;

  // Initialize auxiliary matrices
  LocalElementSize sizeT(tottaus, tottaus);
  TileElementSize blockSizeT(mb, mb);
  Matrix<T, Device::CPU> mat_t(sizeT, blockSizeT);
  set_zero(mat_t);
  
  comm::Index2D src_rank_index = mat_v.distribution().sourceRankIndex();

  LocalElementSize size_vw(mat_v.size().rows(), mb);
  GlobalElementSize size_vv(size_vw.rows(), size_vw.cols());
  Distribution dist_vv(size_vv, mat_v.blockSize(), grid.size(), grid.rank(), src_rank_index);
  Matrix<T, Device::CPU> mat_vv(std::move(dist_vv));

  GlobalElementSize size_w(size_vw.rows(), size_vw.cols());
  Distribution dist_w(size_w, mat_v.blockSize(), grid.size(), grid.rank(), src_rank_index);
  Matrix<T, Device::CPU> mat_w(std::move(dist_w));

  LocalElementSize sz_w2(mb, mat_v.size().cols());
  GlobalElementSize size_w2(sz_w2.rows(), sz_w2.cols());
  Distribution dist_w2(size_w2, mat_v.blockSize(), grid.size(), grid.rank(), src_rank_index);
  Matrix<T, Device::CPU> mat_w2(std::move(dist_w2));

  SizeType last_mb;

  if (mat_v.blockSize().cols() == 1) {
    last_mb = 1;
  }
  else {
    if (mat_v.size().cols() % mat_v.blockSize().cols() == 0)
      last_mb = mat_v.blockSize().cols();
    else
      last_mb = mat_v.size().cols() % mat_v.blockSize().cols();
  }

  LocalElementSize sz_vv_last(mat_v.size().rows(), last_mb);
  GlobalElementSize size_vv_last(sz_vv_last.rows(), sz_vv_last.cols());
  Distribution dist_vv_last(size_vv_last, mat_v.blockSize(), grid.size(), grid.rank(), src_rank_index);
  Matrix<T, Device::CPU> mat_vv_last(std::move(dist_vv_last));

  GlobalElementSize size_w_last(sz_vv_last.rows(), sz_vv_last.cols());
  Distribution dist_w_last(size_w_last, mat_v.blockSize(), grid.size(), grid.rank(), src_rank_index);
  Matrix<T, Device::CPU> mat_w_last(std::move(dist_w_last));

  LocalElementSize sz_w2_last(last_mb, mat_v.size().cols());
  GlobalElementSize size_w2_last(sz_w2_last.rows(), sz_w2_last.cols());
  Distribution dist_w2_last(size_w2_last, mat_v.blockSize(), grid.size(), grid.rank(), src_rank_index);
  Matrix<T, Device::CPU> mat_w2_last(std::move(dist_w2_last));

 //  Compute number of reflectors
  const SizeType reflectors = (mat_v.size().cols() < mat_v.size().rows()) ? mat_v.nrTiles().rows() - 2 : mat_v.nrTiles().cols() - 2;
  

  //Main loop
  for (SizeType k = reflectors; k > -1; --k) {
    bool is_last = (k == reflectors) ? true : false;

    void (&cpyReg)(TileElementSize, TileElementIndex, const matrix::Tile<const T, Device::CPU>&,
                   TileElementIndex, const matrix::Tile<T, Device::CPU>&) = copy<T>;
    void (&cpy)(const matrix::Tile<const T, Device::CPU>&, const matrix::Tile<T, Device::CPU>&) =
        copy<T>;

    // Copy V panel into VV
      for (SizeType i = 0; i < mat_v.nrTiles().rows(); ++i) {      
	auto i_local = dist_v.template localTileFromGlobalTile<Coord::Row>(i);
	
	common::internal::vector<hpx::shared_future<Tile<const T, Device::CPU>>> ik_tile(dist_v.localNrTiles().rows());

	if (is_last) {	
	  auto i_rank_row = dist_v.template rankGlobalTile<Coord::Row>(i);
	  auto k_rank_col = dist_v.template rankGlobalTile<Coord::Col>(k);
	  //std::cout << "last i " << i << " k " << k << std::endl;
	  auto i_rr_vv = mat_vv_last.distribution().template rankGlobalTile<Coord::Row>(i);
	  auto i_rc_vv = mat_vv_last.distribution().template rankGlobalTile<Coord::Col>(0);

	  if (this_rank.row() == i_rank_row) {	    
	    // Broadcast V(i,k) row-wise
	    if (this_rank.col() == k_rank_col) {
	      //std::cout << " sending i row " << i << " col " << k << " rank " << this_rank << std::endl;
	      auto i_local_row = dist_v.template localTileFromGlobalTile<Coord::Row>(i);
	      auto k_local_col = dist_v.template localTileFromGlobalTile<Coord::Col>(k);
	      auto ik = LocalTileIndex(i_local_row, k_local_col);
	      ik_tile[i_local] = mat_v.read(ik);
	      hpx::dataflow(executor_mpi, comm::sendTile_o, serial_comm(), Coord::Row, mat_v.read(ik));
	    }
	    else {
	      //std::cout << " receiving i row " << i << " col " << k << " rank " << this_rank << std::endl;
	      ik_tile[i_local] = hpx::dataflow(executor_mpi, comm::recvAllocTile<T>, serial_comm(), Coord::Row, mat_vv_last.tileSize(GlobalTileIndex(i, 0)), k_rank_col);
	    }
	  }
	  
	  if (this_rank.row() == i_rr_vv && this_rank.col() == i_rc_vv) {
	    auto i_local_row = mat_vv_last.distribution().template localTileFromGlobalTile<Coord::Row>(i);
	    auto k_local_col = mat_vv_last.distribution().template localTileFromGlobalTile<Coord::Col>(0);
	    auto i0 = LocalTileIndex(i_local_row, k_local_col);
	    //	    std::cout << " mao " << this_rank.row() << " " << this_rank.col() << " i0 " << i0 << " mat_vv_last "  << mat_vv_last.read(i0).get()({0,0}) << std::endl;
	    TileElementSize region = mat_vv_last.read(i0).get().size();
	    TileElementIndex idx_in(0, 0);
	    TileElementIndex idx_out(0, 0);
	    //std::cout << "region " << region << " ik_tile " << ik_tile[i_local].get()({0,0}) << std::endl;
	    hpx::dataflow(executor_hp, hpx::util::unwrapping(cpyReg), region, idx_in, ik_tile[i_local_row], idx_out, mat_vv_last(LocalTileIndex(i_local, 0)));
	    //std::cout << " mao " << this_rank.row() << " " << this_rank.col() << " i0 " << i0 << " mat_vv_last "  << mat_vv_last.read(i0).get()({0,0}) << std::endl;	   
	  }
	}
	else {
	  auto i_rank_row = dist_v.template rankGlobalTile<Coord::Row>(i);
	  auto k_rank_col = dist_v.template rankGlobalTile<Coord::Col>(k);
	  //std::cout << " i " << i << " k " << k << std::endl;
	  auto i_rr_vv = mat_vv.distribution().template rankGlobalTile<Coord::Row>(i);
	  auto i_rc_vv = mat_vv.distribution().template rankGlobalTile<Coord::Col>(0);

	  if (this_rank.row() == i_rank_row) {	    
	    // Broadcast V(i,k) row-wise
	    if (this_rank.col() == k_rank_col) {
	      //std::cout << " sending i row " << i << " col " << k << " rank " << this_rank << std::endl;
	      auto i_local_row = dist_v.template localTileFromGlobalTile<Coord::Row>(i);
	      auto k_local_col = dist_v.template localTileFromGlobalTile<Coord::Col>(k);
	      auto ik = LocalTileIndex(i_local_row, k_local_col);
	      ik_tile[i_local] = mat_v.read(ik);
	      hpx::dataflow(executor_mpi, comm::sendTile_o, serial_comm(), Coord::Row, mat_v.read(ik));
	    }
	    else {
	      //std::cout << " receiving i row " << i << " col " << k << " rank " << this_rank << std::endl;
	      ik_tile[i_local] = hpx::dataflow(executor_mpi, comm::recvAllocTile<T>, serial_comm(), Coord::Row, mat_vv_last.tileSize(GlobalTileIndex(i, 0)), k_rank_col);
	    }
	  }
	  
	  if (this_rank.row() == i_rr_vv && this_rank.col() == i_rc_vv) {
	    auto i_local_row = mat_vv.distribution().template localTileFromGlobalTile<Coord::Row>(i);
	    auto k_local_col = mat_vv.distribution().template localTileFromGlobalTile<Coord::Col>(0);
	    auto i0 = LocalTileIndex(i_local_row, k_local_col);
	    //std::cout << " mao " << this_rank.row() << " " << this_rank.col() << " i0 " << i0 << " mat_vv "  << mat_vv.read(i0).get()({0,0}) << " i_tile " << ik_tile[i_local_row].get()({0,0}) << std::endl;
	    hpx::dataflow(executor_mpi, hpx::util::unwrapping(cpy), ik_tile[i_local_row], mat_vv(LocalTileIndex(i_local, 0)));
	    //std::cout << " frr " << this_rank.row() << " " << this_rank.col() << " i0 " << i0 << " mat_vv "  << mat_vv.read(i0).get()({0,0}) << std::endl;	   
	  }
	}
	
	// Fixing elements of VV and copying them into WH
	if (is_last) {
	  auto i_rank_row = mat_vv_last.distribution().template rankGlobalTile<Coord::Row>(i);
	  auto k_rank_col = mat_vv_last.distribution().template rankGlobalTile<Coord::Col>(0);
	  auto i_local_row = mat_vv_last.distribution().template localTileFromGlobalTile<Coord::Row>(i);
	  auto k_local_col = mat_vv_last.distribution().template localTileFromGlobalTile<Coord::Col>(0);
	  auto i0 = LocalTileIndex(i_local_row, k_local_col);

	  if (this_rank.row() == i_rank_row && this_rank.col() == k_rank_col) {
	    auto tile_v = mat_vv_last(i0).get();
	    if (i <= k) {
	      lapack::laset(lapack::MatrixType::General, tile_v.size().rows(), tile_v.size().cols(), 0, 0,
			    tile_v.ptr(), tile_v.ld());
	    }
	    else if (i == k + 1) {
	      lapack::laset(lapack::MatrixType::Upper, tile_v.size().rows(), tile_v.size().cols(), 0, 1,
			    tile_v.ptr(), tile_v.ld());
	    }
	    hpx::dataflow(executor_mpi, hpx::util::unwrapping(cpy), mat_vv_last.read(i0), mat_w_last(i0));
	  }
	}
	else {
	  auto i_rank_row = mat_vv.distribution().template rankGlobalTile<Coord::Row>(i);
	  auto k_rank_col = mat_vv.distribution().template rankGlobalTile<Coord::Col>(0);
	  auto i_local_row = mat_vv.distribution().template localTileFromGlobalTile<Coord::Row>(i);
	  auto k_local_col = mat_vv.distribution().template localTileFromGlobalTile<Coord::Col>(0);
	  auto i0 = LocalTileIndex(i_local_row, k_local_col);

	  if (this_rank.row() == i_rank_row && this_rank.col() == k_rank_col) {
	    auto tile_v = mat_vv(i0).get();
	    if (i <= k) {
	      lapack::laset(lapack::MatrixType::General, tile_v.size().rows(), tile_v.size().cols(), 0, 0, tile_v.ptr(), tile_v.ld());
	    }
	    else if (i == k + 1) {
	      lapack::laset(lapack::MatrixType::Upper, tile_v.size().rows(), tile_v.size().cols(), 0, 1, tile_v.ptr(), tile_v.ld());
	    }
	    hpx::dataflow(executor_mpi, hpx::util::unwrapping(cpy), mat_vv.read(i0), mat_w(i0));
	  }
	}


    int taupan;
    // Reset W2 to zero
    if (is_last) {
      matrix::util::set(mat_w2_last, [](auto&&) { return 0; });
      taupan = last_mb;
    }
    else {
      matrix::util::set(mat_w2, [](auto&&) { return 0; });
      taupan = mat_v.blockSize().cols();
    }
    
    // Matrix T
  const GlobalTileIndex v_start{k+1, k};
  auto taus_panel = taus[k];

  auto kk = LocalTileIndex(k, k);
  dlaf::factorization::internal::computeTFactor<Backend::MC>(taupan, mat_v, v_start, taus_panel, mat_t(kk), serial_comm);
  
  for (SizeType i_local = mat_w.distribution().template nextLocalTileFromGlobalTile<Coord::Row>(k + 1); i_local < c_local_rows; ++i_local) {

    //Broadcast T(k,k) column-wise
    hpx::shared_future<Tile<const T, Device::CPU>> kk_tile;
    
    auto k_rank_row = mat_t.distribution().template rankGlobalTile<Coord::Row>(k);
    auto k_rank_col = mat_t.distribution().template rankGlobalTile<Coord::Col>(k);

    if (this_rank.col() == k_rank_col) {
      if (this_rank.row() == k_rank_row) {
	auto k_local_row = mat_t.distribution().template localTileFromGlobalTile<Coord::Row>(k);
	auto k_local_col = mat_t.distribution().template localTileFromGlobalTile<Coord::Col>(k);
	auto kk = LocalTileIndex(k_local_row, k_local_col);

	kk_tile = mat_t.read(kk);
	hpx::dataflow(executor_mpi, comm::sendTile_o, serial_comm(), Coord::Col, mat_t.read(kk));
      }
      else {
	kk_tile = hpx::dataflow(executor_mpi, comm::recvAllocTile<T>, serial_comm(), Coord::Col, mat_t.tileSize(GlobalTileIndex(k, k)), k_rank_row);
      }
    }
	  
    // WH = V T
    if (is_last) {
      auto i = mat_w_last.distribution().template globalTileFromLocalTile<Coord::Row>(i_local);
      
      auto i_rank_row = mat_w_last.distribution().template rankGlobalTile<Coord::Row>(i);
      auto k_rank_col = mat_w_last.distribution().template rankGlobalTile<Coord::Col>(0);
      
      if (this_rank.row() == i_rank_row && this_rank.col() == k_rank_col) {
	auto i_local_row = mat_w_last.distribution().template localTileFromGlobalTile<Coord::Row>(i);
	auto i_local_col = mat_w_last.distribution().template localTileFromGlobalTile<Coord::Col>(0);
	auto ik = LocalTileIndex{i_local_row, i_local_col};	
	hpx::dataflow(executor_hp, hpx::util::unwrapping(tile::trmm<T, Device::CPU>), Right, Upper, ConjTrans, NonUnit, 1.0, mat_t.read(kk), std::move(mat_w_last(ik)));
      }
    }
    else {
      auto i = mat_w.distribution().template globalTileFromLocalTile<Coord::Row>(i_local);
      
      auto i_rank_row = mat_w.distribution().template rankGlobalTile<Coord::Row>(i);
      auto k_rank_col = mat_w.distribution().template rankGlobalTile<Coord::Col>(0);
      
      if (this_rank.row() == i_rank_row && this_rank.col() == k_rank_col) {
	auto i_local_row = mat_w.distribution().template localTileFromGlobalTile<Coord::Row>(i);
	auto i_local_col = mat_w.distribution().template localTileFromGlobalTile<Coord::Col>(0);
	auto ik = LocalTileIndex{i_local_row, i_local_col};
        hpx::dataflow(executor_hp, hpx::util::unwrapping(tile::trmm<T, Device::CPU>), Right, Upper, ConjTrans, NonUnit, 1.0, mat_t.read(kk), std::move(mat_w(ik)));
      }
    }
  }

    for (SizeType j_local = 0; j_local < c_local_cols; ++j_local) {

      // Broadcast W2(0,j) col-wise
      hpx::shared_future<Tile<const T, Device::CPU>> w2_tile;
      
      if (is_last) {
	auto j = mat_w2_last.distribution().template globalTileFromLocalTile<Coord::Col>(j_local);

	auto j_rank_row = mat_w2_last.distribution().template rankGlobalTile<Coord::Row>(0);
	auto j_rank_col = mat_w2_last.distribution().template rankGlobalTile<Coord::Col>(j);

	if (this_rank.col() == j_rank_col) {
	  if (this_rank.row() == j_rank_row) {
	    auto j_local_row = mat_w2_last.distribution().template localTileFromGlobalTile<Coord::Row>(0);
	    auto j_local_col = mat_w2_last.distribution().template localTileFromGlobalTile<Coord::Col>(j);
	    auto jj = LocalTileIndex(j_local_row, j_local_col);

	    w2_tile = mat_w2_last.read(jj);
	    hpx::dataflow(executor_mpi, comm::sendTile_o, serial_comm(), Coord::Col, mat_w2_last.read(jj));
	  }
	  else {
	    w2_tile = hpx::dataflow(executor_mpi, comm::recvAllocTile<T>, serial_comm(), Coord::Col, mat_w2_last.tileSize(GlobalTileIndex(0, j)), j_rank_row);
	  }
	}
      }
      else {
	auto j = mat_w2.distribution().template globalTileFromLocalTile<Coord::Col>(j_local);

	auto j_rank_row = mat_w2.distribution().template rankGlobalTile<Coord::Row>(0);
	auto j_rank_col = mat_w2.distribution().template rankGlobalTile<Coord::Col>(j);

	if (this_rank.col() == j_rank_col) {
	  if (this_rank.row() == j_rank_row) {
	    auto j_local_row = mat_w2.distribution().template localTileFromGlobalTile<Coord::Row>(0);
	    auto j_local_col = mat_w2.distribution().template localTileFromGlobalTile<Coord::Col>(j);
	    auto jj = LocalTileIndex(j_local_row, j_local_col);

	    w2_tile = mat_w2.read(jj);
	    hpx::dataflow(executor_mpi, comm::sendTile_o, serial_comm(), Coord::Col, mat_w2.read(jj));
	  }
	  else {
	    w2_tile = hpx::dataflow(executor_mpi, comm::recvAllocTile<T>, serial_comm(), Coord::Col, mat_w2.tileSize(GlobalTileIndex(0, j)), j_rank_row);
	  }
	}
      }

      //      auto kj = LocalTileIndex{0, j};

      for (SizeType i_local = mat_w.distribution().template nextLocalTileFromGlobalTile<Coord::Row>(k + 1); i_local < c_local_rows; ++i_local) {

      // Broadcast W(i,0) row-wise




//        auto ik = LocalTileIndex{i, 0};
//        auto ij = LocalTileIndex{i, j};
//        // W2 = W C
//        if (is_last) {
//          hpx::dataflow(executor_normal, hpx::util::unwrapping(tile::gemm<T, Device::CPU>), ConjTrans,
//                        NoTrans, 1.0, std::move(mat_w_last(ik)), mat_c.read(ij), 1.0,
//                        std::move(mat_w2_last(kj)));
//        }
//        else {
//          hpx::dataflow(executor_normal, hpx::util::unwrapping(tile::gemm<T, Device::CPU>), ConjTrans,
//                        NoTrans, 1.0, std::move(mat_w(ik)), mat_c.read(ij), 1.0, std::move(mat_w2(kj)));
//        }
      }
    }
    
//    for (SizeType i = k + 1; i < m; ++i) {
//      auto ik = LocalTileIndex{i, 0};
//      for (SizeType j = 0; j < n; ++j) {
//        auto kj = LocalTileIndex{0, j};
//        auto ij = LocalTileIndex{i, j};
//        // C = C - V W2
//        if (is_last) {
//          hpx::dataflow(executor_normal, hpx::util::unwrapping(tile::gemm<T, Device::CPU>), NoTrans,
//                        NoTrans, -1.0, mat_vv_last.read(ik), mat_w2_last.read(kj), 1.0,
//                        std::move(mat_c(ij)));
//        }
//        else {
//          hpx::dataflow(executor_normal, hpx::util::unwrapping(tile::gemm<T, Device::CPU>), NoTrans,
//                        NoTrans, -1.0, mat_vv.read(ik), mat_w2.read(kj), 1.0, std::move(mat_c(ij)));
//        }
//      }
    }

  }

 }

/// ---- ETI
#define DLAF_SOLVER_BACKTRANSFORMATION_MC_ETI(KWORD, DATATYPE) \
  KWORD template struct BackTransformation<Backend::MC, Device::CPU, DATATYPE>;

DLAF_SOLVER_BACKTRANSFORMATION_MC_ETI(extern, float)
DLAF_SOLVER_BACKTRANSFORMATION_MC_ETI(extern, double)
DLAF_SOLVER_BACKTRANSFORMATION_MC_ETI(extern, std::complex<float>)
DLAF_SOLVER_BACKTRANSFORMATION_MC_ETI(extern, std::complex<double>)

}
}
}
