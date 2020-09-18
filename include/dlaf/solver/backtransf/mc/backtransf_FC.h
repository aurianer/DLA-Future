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

#include "dlaf/blas_tile.h"
#include "dlaf/common/index2d.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/common/vector.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/executor.h"
#include "dlaf/communication/functions_sync.h"
#include "dlaf/lapack_tile.h"
#include "dlaf/matrix.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/layout_info.h"
#include "dlaf/util_matrix.h"

#include "dlaf/matrix_output.h"

namespace dlaf {
namespace internal {
namespace mc {

// Local implementation of Left Lower NoTrans

// Implementation based on:
// 1. Algorithm 6 "LAPACK Algorithm for the eigenvector back-transformation", page 15, PhD thesis "GPU
// Accelerated Implementations of a Generalized Eigenvalue Solver for Hermitian Matrices with Systematic
// Energy and Time to Solution Analysis" presented by Raffaele Solcà (2016)
// 2. Report "Gep + back-transformation", Alberto Invernizzi (2020)
// 3. Report "Reduction to band + back-transformation", Raffaele Solcà (2020)

template <class T>
void backtransf_FC(Matrix<T, Device::CPU>& mat_c, Matrix<const T, Device::CPU>& mat_v,
                   Matrix<T, Device::CPU>& mat_t) {
  constexpr auto Left = blas::Side::Left;
  constexpr auto Right = blas::Side::Right;
  constexpr auto Upper = blas::Uplo::Upper;
  constexpr auto Lower = blas::Uplo::Lower;
  constexpr auto NoTrans = blas::Op::NoTrans;
  constexpr auto ConjTrans = blas::Op::ConjTrans;
  constexpr auto NonUnit = blas::Diag::NonUnit;

  using hpx::threads::executors::pool_executor;
  using hpx::threads::thread_priority_high;
  using hpx::threads::thread_priority_default;

  // Set up executor on the default queue with high priority.
  pool_executor executor_hp("default", thread_priority_high);
  // Set up executor on the default queue with default priority.
  pool_executor executor_normal("default", thread_priority_default);

  SizeType m = mat_c.nrTiles().rows();
  SizeType n = mat_c.nrTiles().cols();
  SizeType mb = mat_t.blockSize().rows();
  SizeType nb = mat_t.blockSize().cols();

  // n-1 reflectors
  for (SizeType i = 0; i < (n - 1); ++i) {
    // Create a temporary matrix to store W2
    TileElementSize size(mb, nb);
    auto dist = mat_t.distribution();
    auto layout = tileLayout(dist.localSize(), size);
    Matrix<T, Device::CPU> mat_w2(std::move(dist), layout);
    // ...and fill it with zeros
    for (SizeType w2_col = 0; w2_col < n; ++w2_col) {
      for (SizeType w2_row = 0; w2_row < m; ++w2_row) {
	auto tile_index = LocalTileIndex(w2_row, w2_col);
	auto el_zero = []() {
	  return static_cast<T>(0.0);
	};
	auto tile = mat_w2(tile_index).get();
	for (SizeType w2_j = 0; w2_j < nb; ++w2_j) {
	  for (SizeType w2_i = 0; w2_i < mb; ++w2_i) {
	    TileElementIndex index(w2_i, w2_j);
	    tile(index) = el_zero();
	  }
	}
      }
    }
    
    for (SizeType k = i; k < m; ++k) {
      auto ki = LocalTileIndex{k, i};
      auto kk = LocalTileIndex{k, k};
      // W = V T 
      hpx::dataflow(executor_hp, hpx::util::unwrapping(tile::trmm<T, Device::CPU>), Left, Upper, NoTrans,
                    NonUnit, 1.0, mat_v.read(ki), std::move(mat_t(ki)));
    }

    for (SizeType k = i; k < m; ++k) {
      auto ik = LocalTileIndex{i, k};
      for (SizeType j = i; j < n; ++j) {
        auto ji = LocalTileIndex{j, i};
        auto jk = LocalTileIndex{j, k};
        // W2 = WH C
        hpx::dataflow(executor_normal, hpx::util::unwrapping(tile::gemm<T, Device::CPU>), NoTrans,
                      NoTrans, 1.0, std::move(mat_t(ji)), mat_c.read(jk), 1.0, std::move(mat_w2(ik)));
      }
    }

    for (SizeType k = i; k < m; ++k) {
      auto ki = LocalTileIndex{k, i};
      for (SizeType j = i; j < n; ++j) {
        auto ij = LocalTileIndex{i, j};
        auto kj = LocalTileIndex{k, j};
        // C = C - V W2
        hpx::dataflow(executor_normal, hpx::util::unwrapping(tile::gemm<T, Device::CPU>), NoTrans,
                      NoTrans, -1.0, mat_v.read(ki), mat_w2.read(ij), 1.0, std::move(mat_c(kj)));
      }
    }
  }
}

}
}
}
