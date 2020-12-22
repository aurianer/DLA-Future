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

#define DLAF_MAKE_CALLABLE_OBJECT(fname)     \
  constexpr struct fname##_t {               \
    template <typename... Ts>                \
    decltype(auto) operator()(Ts&&... ts) {  \
      return fname(std::forward<Ts>(ts)...); \
    }                                        \
  } fname##_o {                              \
  }

template <class Func, class Tuple, std::size_t... Is>
auto apply_impl(Func&& func, Tuple&& t, std::index_sequence<Is...>) {
  return hpx::tuple<decltype(func(hpx::get<Is, std::decay_t<Tuple>>(t)))...>(func(hpx::get<Is>(t))...);
}

// TODO apply does not work with single argument (useful becase unwrap(a) does not return a tuple)
template <class Func, class Tuple>
auto apply(Func&& func, Tuple&& t) {
  return apply_impl(std::forward<Func>(func), std::forward<Tuple>(t),
                    std::make_index_sequence<std::tuple_size<std::decay_t<Tuple>>::value>());
}

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

struct UnwrapPromiseGuards {
  template <class T>
  decltype(auto) operator()(T&& t) {
    return std::forward<T>(t);
  }

  template <class T>
  T& operator()(dlaf::common::PromiseGuard<T>& u) {
    return u.ref();
  }
};

template <Coord dir>
struct SelectCommunicator {
  template <class T>
  decltype(auto) operator()(T&& t) {
    return std::move(t);
  }

  Communicator& operator()(CommunicatorGrid& guard) {
    return guard.subCommunicator(dir);
  }
};

template <Coord dir, class Callable>
struct Foo {
  Callable f;

  template <class... Ts>
  auto operator()(Ts&&... ts) {
    // extract all futures
    auto t1 = hpx::util::unwrap<Ts...>(std::forward<Ts>(ts)...);

    // Extract just PromiseGuards resources, move everything else
    auto t2 = apply(UnwrapPromiseGuards{}, t1);

    // Select row/col for CommunicatorGrid params, move everything else
    auto t3 = apply(SelectCommunicator<dir>{}, t2);

    // TODO still to investigate why unwrapping is needed
    return hpx::invoke_fused(hpx::util::unwrapping(f), t3);
  }
};

template <Coord dir, class Callable>
auto foo(Callable&& func) {
  return Foo<dir, Callable>{std::forward<Callable>(func)};
}
}

template <Coord rc_comm, class T>
void send_tile(hpx::threads::executors::pool_executor ex,
               common::Pipeline<comm::CommunicatorGrid>& task_chain,
               hpx::shared_future<matrix::Tile<const T, Device::CPU>> tile) {
  hpx::dataflow(ex, sync::foo<rc_comm>(sync::broadcast::send_o), task_chain(), tile);
}

template <Coord rc_comm, class T>
hpx::future<matrix::Tile<const T, Device::CPU>> recv_tile(
    hpx::threads::executors::pool_executor ex, common::Pipeline<comm::CommunicatorGrid>& mpi_task_chain,
    TileElementSize tile_size, int rank) {
  using ConstTile_t = matrix::Tile<const T, Device::CPU>;
  using MemView_t = memory::MemoryView<T, Device::CPU>;
  using Tile_t = matrix::Tile<T, Device::CPU>;

  auto recv_bcast_f = hpx::util::unwrapping([tile_size](int rank, Communicator& comm) -> ConstTile_t {
    MemView_t mem_view(tile_size.linear_size());
    Tile_t tile(tile_size, std::move(mem_view), tile_size.rows());
    comm::sync::broadcast::receive_from(rank, comm, tile);
    return std::move(tile);
  });
  return hpx::dataflow(ex, sync::foo<rc_comm>(std::move(recv_bcast_f)), rank,
                       mpi_task_chain());  // TODO why if I don't move it creates a problem?!
}
}
}
