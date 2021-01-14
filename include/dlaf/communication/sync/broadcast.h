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

#include <type_traits>
#include <utility>

#include <hpx/functional.hpp>
#include <hpx/include/parallel_executors.hpp>
#include <hpx/local/future.hpp>
#include <hpx/pack_traversal/unwrap.hpp>
#include <hpx/threading_base/annotated_function.hpp>
#include <hpx/tuple.hpp>

#include "dlaf/common/assert.h"
#include "dlaf/common/data.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/message.h"
#include "dlaf/matrix/tile.h"

#include "dlaf/common/functional.h"

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
  DLAF_ASSERT_HEAVY(broadcaster_rank != communicator.rank(), "sender and receiver should be different",
                    broadcaster_rank, communicator.rank());
  auto message = comm::make_message(common::make_data(std::forward<DataOut>(data)));
  DLAF_MPI_CALL(
      MPI_Bcast(message.data(), message.count(), message.mpi_type(), broadcaster_rank, communicator));
}

DLAF_MAKE_CALLABLE_OBJECT(send);
DLAF_MAKE_CALLABLE_OBJECT(receive_from);

}
}

template <Coord dir>
struct SelectCommunicator {
  template <class T>
  decltype(auto) operator()(T&& t) {
    return std::forward<T>(t);
  }

  Communicator& operator()(CommunicatorGrid& guard) {
    return guard.subCommunicator(dir);
  }
};

template <Coord dir, class Callable>
struct selector_impl {
  Callable f;

  template <class... Ts>
  auto operator()(Ts&&... ts) {
    auto t2 = hpx::tuple<Ts...>{std::forward<Ts>(ts)...};
    auto t3 = apply(SelectCommunicator<dir>{}, t2);
    return hpx::invoke_fused(std::move(f), t3);
  }
};

template <Coord dir, class Callable>
auto selector(Callable f) {
  return selector_impl<dir, Callable>{std::move(f)};
}

template <Coord rc_comm, class T>
void send_tile(hpx::threads::executors::pool_executor ex,
               common::Pipeline<comm::CommunicatorGrid>& task_chain,
               hpx::shared_future<matrix::Tile<const T, Device::CPU>> tile) {
  using hpx::util::annotated_function;
  using hpx::util::unwrapping;

  auto send_tile_func = unwrap_guards(selector<rc_comm>(sync::broadcast::send_o));
  hpx::dataflow(ex, annotated_function(unwrapping(std::move(send_tile_func)), "send_tile"), task_chain(),
                tile);
}

template <Coord rc_comm, class T>
hpx::future<matrix::Tile<const T, Device::CPU>> recv_tile(
    hpx::threads::executors::pool_executor ex, common::Pipeline<comm::CommunicatorGrid>& mpi_task_chain,
    TileElementSize tile_size, int rank) {
  using hpx::util::annotated_function;
  using hpx::util::unwrapping;

  using ConstTile_t = matrix::Tile<const T, Device::CPU>;
  using MemView_t = memory::MemoryView<T, Device::CPU>;
  using Tile_t = matrix::Tile<T, Device::CPU>;

  auto recv_bcast_f =
      unwrap_guards(selector<rc_comm>([tile_size, rank](Communicator& comm) -> ConstTile_t {
        MemView_t mem_view(tile_size.linear_size());
        Tile_t tile(tile_size, std::move(mem_view), tile_size.rows());
        comm::sync::broadcast::receive_from(rank, comm, tile);
        return std::move(tile);
      }));

  return hpx::dataflow(ex, annotated_function(unwrapping(std::move(recv_bcast_f)), "recv_tile"),
                       mpi_task_chain());
}
}
}
