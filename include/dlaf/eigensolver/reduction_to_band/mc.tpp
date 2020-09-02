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

#include "dlaf/communication/communicator_grid.h"
#include "dlaf/eigensolver/reduction_to_band/mc/reduction_to_band.h"
#include "dlaf/matrix.h"
#include "dlaf/util_matrix.h"

namespace dlaf {

template <class T>
void EigenSolver<Backend::MC>::reduction_to_band(comm::CommunicatorGrid grid,
                                                 Matrix<T, Device::CPU>& mat_a) {
  // DLAF_ASSERT(matrix::square_size(mat_a), mat_a);
  // DLAF_ASSERT(matrix::square_blocksize(mat_a), mat_a);
  // DLAF_ASSERT(matrix::equal_process_grid(mat_a, grid), mat_a, grid);

  internal::mc::reduction_to_band(grid, mat_a);
}

}
