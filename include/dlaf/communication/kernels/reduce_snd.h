//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

/// @file


#include <mpi.h>

#include <pika/execution.hpp>

//#include <dlaf/common/data.h>
#include <dlaf/common/eti.h>
#include <dlaf/communication/communicator_pipeline.h>
#include <dlaf/communication/index.h>
//#include <dlaf/communication/message.h>
#include <dlaf/matrix/tile.h>

namespace dlaf::comm {
/// Schedule a reduction send.
///
/// The returned sender signals completion when the send is done. If the input
/// tile is movable it will be sent by the returned sender. Otherwise a void
/// sender is returned.
template <class T, Device D>
[[nodiscard]] pika::execution::experimental::unique_any_sender<> schedule_reduce_send(
    pika::execution::experimental::unique_any_sender<CommunicatorPipelineExclusiveWrapper> pcomm,
    comm::IndexT_MPI rank_root, MPI_Op reduce_op, dlaf::matrix::ReadOnlyTileSender<T, D> tile);

#define DLAF_SCHEDULE_REDUCE_SEND_ETI(kword, Type, Device)                                          \
  kword template pika::execution::experimental::unique_any_sender<> schedule_reduce_send(           \
      pika::execution::experimental::unique_any_sender<CommunicatorPipelineExclusiveWrapper> pcomm, \
      comm::IndexT_MPI rank_root, MPI_Op reduce_op,                                                 \
      dlaf::matrix::ReadOnlyTileSender<Type, Device> tile)

DLAF_EXPAND_ETI_SDCZ_DEVICE(DLAF_SCHEDULE_REDUCE_SEND_ETI, extern);
DLAF_SCHEDULE_REDUCE_SEND_ETI(extern, int, Device::CPU);
}
