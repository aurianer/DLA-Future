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
#include "dlaf/eigensolver/internal.h"
#include "dlaf/matrix.h"

namespace dlaf {

template <>
struct EigenSolver<Backend::MC> {
  /// Reduction-to-band
  ///
  /// TODO
  template <class T>
  static void reduction_to_band(comm::CommunicatorGrid grid, Matrix<T, Device::CPU>& mat_a);
};

}

#include "dlaf/eigensolver/reduction_to_band/mc.tpp"

/// ---- ETI
namespace dlaf {

#define DLAF_EIGENSOLVER_ETI(KWORD, DATATYPE)                                   \
  KWORD template void                                                           \
  EigenSolver<Backend::MC>::reduction_to_band<DATATYPE>(comm::CommunicatorGrid, \
                                                        Matrix<DATATYPE, Device::CPU>&);

DLAF_EIGENSOLVER_ETI(extern, float)
DLAF_EIGENSOLVER_ETI(extern, double)
// TODO fix for complex types
// DLAF_EIGENSOLVER_ETI(extern, std::complex<float>)
// DLAF_EIGENSOLVER_ETI(extern, std::complex<double>)

}
