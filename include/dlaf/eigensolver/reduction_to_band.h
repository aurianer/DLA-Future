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
#include "dlaf/matrix/matrix.h"
#include "dlaf/util_matrix.h"

#include "dlaf/eigensolver/reduction_to_band/mc.h"

namespace dlaf {
namespace eigensolver {

/// Reduce a Hermitian definite generalized eigenproblem to standard form, using the factorization
/// obtained from potrf (Cholesky factorization): A <- f(A, B).
/// It solves B=inv(L)*A*inv(L**H) or B=inv(U**H)*A*inv(U).
/// Implementation on local memory.
///
/// @param mat_a on entry it contains the Hermitian matrix A (if A is real, the matrix is symmetric),
/// on exit the matrix elements are overwritten with the elements of the matrix B.
/// @param mat_b contains the triangular matrix. It can be lower (L) or upper (U). Only the tiles of
/// the matrix which contain the lower triangular or the upper triangular part are accessed.
/// @pre mat_a and mat_b have a square size,
/// @pre mat_a and mat_b have a square block size,
/// @pre mat_a and mat_b are not distributed.
template <Backend backend, Device device, class T>
std::vector<hpx::shared_future<std::vector<T>>> reductionToBand(comm::CommunicatorGrid grid,
                                                                Matrix<T, device>& mat_a) {
  DLAF_ASSERT(matrix::square_size(mat_a), mat_a);
  DLAF_ASSERT(matrix::square_blocksize(mat_a), mat_a);
  // TODO fix for non-distributed
  DLAF_ASSERT(matrix::equal_process_grid(mat_a, grid), mat_a, grid);

  return internal::ReductionToBand<backend, device, T>::call(grid, mat_a);
}

}
}
