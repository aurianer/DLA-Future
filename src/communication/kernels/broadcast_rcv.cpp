//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <utility>

#include <mpi.h>

#include <pika/execution.hpp>

#include <dlaf/common/assert.h>
#include <dlaf/common/eti.h>
#include <dlaf/communication/communicator_pipeline.h>
#include <dlaf/communication/index.h>
#include <dlaf/communication/kernels/broadcast_rcv.h>
#include <dlaf/communication/kernels/internal/broadcast_rcv.h>
#include <dlaf/communication/rdma.h>
#include <dlaf/matrix/tile.h>
#include <dlaf/sender/traits.h>
#include <dlaf/sender/with_temporary_tile.h>

namespace dlaf::comm {

template <class T, Device D, class Comm>
[[nodiscard]] dlaf::matrix::ReadWriteTileSender<T, D> schedule_bcast_recv(
    pika::execution::experimental::unique_any_sender<Comm> pcomm, comm::IndexT_MPI root_rank,
    dlaf::matrix::ReadWriteTileSender<T, D> tile) {
  using dlaf::internal::RequireContiguous;
  constexpr Device DComm = CommunicationDevice_v<D>;
  constexpr auto require_contiguous =
#if defined(DLAF_WITH_MPI_GPU_AWARE) && defined(DLAF_WITH_MPI_GPU_FORCE_CONTIGUOUS)
      DComm == Device::GPU ? RequireContiguous::Yes :
#endif
                           RequireContiguous::No;
  return internal::schedule_bcast_recv<DComm, require_contiguous>(std::move(pcomm), root_rank,
                                                                  std::move(tile));
}

// clang-format off
DLAF_EXPAND_ETI_SDCZ_DEVICE_VA_ARGS(DLAF_SCHEDULE_BCAST_RECV_ETI, , CommunicatorPipelineExclusiveWrapper);

DLAF_SCHEDULE_BCAST_RECV_ETI(, SizeType, Device::CPU, CommunicatorPipelineExclusiveWrapper);
#ifdef DLAF_WITH_GPU
DLAF_SCHEDULE_BCAST_RECV_ETI(, SizeType, Device::GPU, CommunicatorPipelineExclusiveWrapper);
#endif
// clang-format on
}
