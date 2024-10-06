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

#include <complex>

//FIXME: is mpi needed in this file?
#include <mpi.h>

#include <pika/execution.hpp>

#include <dlaf/common/assert.h>
#include <dlaf/common/eti.h>
#include <dlaf/communication/communicator_pipeline.h>
#include <dlaf/matrix/tile.h>
#include <dlaf/types.h>

namespace dlaf::comm {

/// Schedule a broadcast send.
///
/// The returned sender signals completion when the send is done. If the input
/// tile is movable it will be sent by the returned sender. Otherwise a void
/// sender is returned.
template <class T, Device D, class Comm>
[[nodiscard]] pika::execution::experimental::unique_any_sender<> schedule_bcast_send(
    pika::execution::experimental::unique_any_sender<Comm> pcomm,
    dlaf::matrix::ReadOnlyTileSender<T, D> tile);

#define DLAF_SCHEDULE_BCAST_SEND_ETI(kword, Type, Device, Comm)                          \
  kword template pika::execution::experimental::unique_any_sender<> schedule_bcast_send( \
      pika::execution::experimental::unique_any_sender<Comm> pcomm,                      \
      dlaf::matrix::ReadOnlyTileSender<Type, Device> tile)

// clang-format off
DLAF_EXPAND_ETI_SDCZ_DEVICE_VA_ARGS(DLAF_SCHEDULE_BCAST_SEND_ETI, extern, CommunicatorPipelineExclusiveWrapper);

DLAF_SCHEDULE_BCAST_SEND_ETI(extern, SizeType, Device::CPU, CommunicatorPipelineExclusiveWrapper);
#ifdef DLAF_WITH_GPU
DLAF_SCHEDULE_BCAST_SEND_ETI(extern, SizeType, Device::GPU, CommunicatorPipelineExclusiveWrapper);
#endif
// clang-format on

}
