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

/// @file

#include <hpx/include/parallel_executors.hpp>
#include <hpx/local/future.hpp>
#include <hpx/threading_base/annotated_function.hpp>
#include <hpx/tuple.hpp>
#include <hpx/functional.hpp>
#include <type_traits>

#include "dlaf/common/assert.h"
#include "dlaf/common/data.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/message.h"
#include "dlaf/matrix/tile.h"

namespace dlaf {
namespace comm {
namespace sync {
namespace broadcast {

/// MPI_Bcast wrapper for sender side accepting a Data.
///
/// For more information, see the Data concept in "dlaf/common/data.h".
template <class DataIn>
void send(Communicator& communicator, DataIn&& message_to_send) {
  auto data = common::make_data(message_to_send);
  using DataT = std::remove_const_t<typename common::data_traits<decltype(data)>::element_t>;

  auto message = comm::make_message(std::move(data));
  DLAF_MPI_CALL(MPI_Bcast(const_cast<DataT*>(message.data()), message.count(), message.mpi_type(),
                          communicator.rank(), communicator));
}

/// MPI_Bcast wrapper for receiver side accepting a dlaf::comm::Message.
///
/// For more information, see the Data concept in "dlaf/common/data.h".
template <class DataOut>
void receive_from(const int broadcaster_rank, Communicator& communicator, DataOut&& data) {
  DLAF_ASSERT_HEAVY(broadcaster_rank != communicator.rank(), "");
  auto message = comm::make_message(common::make_data(std::forward<DataOut>(data)));
  DLAF_MPI_CALL(
      MPI_Bcast(message.data(), message.count(), message.mpi_type(), broadcaster_rank, communicator));
}

}

template <Coord dir, class T>
struct UnwrapCommGrid {
  template <class U>
  static decltype(auto) call(U&& u) {
    return std::forward<U>(u);
  }
};

template <Coord dir>
struct UnwrapCommGrid<dir, common::PromiseGuard<CommunicatorGrid>> {
  template <class U>
  static Communicator& call(U&& u) {
    return u.ref().subCommunicator(dir);
  }
};

template <Coord dir, class Func>
struct Foo {
  Func f;

  template <class... Ts>
  auto operator()(Ts&&... ts) {
    return f(UnwrapCommGrid<dir, std::decay_t<Ts>>::call(std::forward<Ts>(ts))...);
  }
};

struct broadcast_send {
  template <class DataIn>
  void operator()(Communicator& communicator, DataIn&& message_to_send) {
    broadcast::send(communicator, message_to_send);
  }
};

struct broadcast_recv {
  template <class DataOut>
  void operator()(const comm::IndexT_MPI rank, Communicator& communicator, DataOut&& message_to_send) {
    broadcast::receive_from(rank, communicator, message_to_send);
  }
};

}

template <Coord rc_comm, class T>
void send_tile(hpx::threads::executors::pool_executor ex,
               common::Pipeline<comm::CommunicatorGrid>& task_chain,
               hpx::shared_future<matrix::Tile<const T, Device::CPU>> tile) {
  hpx::dataflow(ex, hpx::util::annotated_function(hpx::util::unwrapping(sync::Foo<rc_comm, sync::broadcast_send>()), "send_tile"), task_chain(), tile);
}

template <class T>
hpx::future<matrix::Tile<const T, Device::CPU>> recv_tile(
    hpx::threads::executors::pool_executor ex, common::Pipeline<comm::CommunicatorGrid>& mpi_task_chain,
    Coord rc_comm, TileElementSize tile_size, int rank) {
  using ConstTile_t = matrix::Tile<const T, Device::CPU>;
  using PromiseComm_t = common::PromiseGuard<comm::CommunicatorGrid>;
  using MemView_t = memory::MemoryView<T, Device::CPU>;
  using Tile_t = matrix::Tile<T, Device::CPU>;

  auto recv_bcast_f = hpx::util::annotated_function(
      [rank, tile_size, rc_comm](hpx::future<PromiseComm_t> fpcomm) -> ConstTile_t {
        PromiseComm_t pcomm = fpcomm.get();
        MemView_t mem_view(tile_size.linear_size());
        Tile_t tile(tile_size, std::move(mem_view), tile_size.rows());
        comm::sync::broadcast::receive_from(rank, pcomm.ref().subCommunicator(rc_comm), tile);
        return ConstTile_t(std::move(tile));
      },
      "recv_tile");
  return hpx::dataflow(ex, std::move(recv_bcast_f), mpi_task_chain());
}
}
}
