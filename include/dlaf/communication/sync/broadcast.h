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

#include <hpx/functional.hpp>
#include <hpx/include/parallel_executors.hpp>
#include <hpx/local/future.hpp>
#include <hpx/threading_base/annotated_function.hpp>
#include <hpx/tuple.hpp>
#include <type_traits>

#include "dlaf/common/assert.h"
#include "dlaf/common/data.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/message.h"
#include "dlaf/matrix/tile.h"

#define DLAF_MAKE_CALLABLE_OBJECT(fname)          \
  constexpr struct fname##_t {                    \
    template <typename... Ts>                     \
    decltype(auto) operator()(Ts&&... ts) const { \
      return fname(std::forward<Ts>(ts)...);      \
    }                                             \
  } fname##_o {                                   \
  }

template <std::size_t N, class Func>
struct apply_tuple {
  template <class TupleIn>
  static decltype(auto) call(TupleIn&& t) {
    return hpx::tuple_cat(apply_tuple<N - 1, Func>::call(t),
                          hpx::make_tuple<>(Func{}(hpx::util::get<N - 1>(t))));
  }
};

template <class Func>
struct apply_tuple<1, Func> {
  template <class T>
  static decltype(auto) call(T&& t) {
    return hpx::make_tuple<>(Func{}(hpx::get<0>(std::forward<T>(t))));
  }
};

template <class Func, class Tuple>
auto map_tuple(Func&&, Tuple&& t) {
  return apply_tuple<hpx::tuple_size<std::decay_t<Tuple>>::value, Func>::call(std::forward<Tuple>(t));
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
  static decltype(auto) call(T&& u) {
    return std::forward<T>(u);
  }

  template <class T>
  static T& call(dlaf::common::PromiseGuard<T> u) {
    return u.ref();
  }
};

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
struct Foo {
  Callable f;

  template <class... Ts>
  auto operator()(Ts&&... ts) {
    auto t = hpx::make_tuple<>(UnwrapPromiseGuards::call(std::forward<Ts>(ts))...);
    auto t2 = map_tuple(SelectCommunicator<dir>{}, t);
    return hpx::invoke_fused(hpx::util::unwrapping(std::move(f)), t2);
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
  auto send_tile = [](hpx::future<common::PromiseGuard<comm::CommunicatorGrid>> fut_guard,
                      hpx::shared_future<matrix::Tile<const T, Device::CPU>> tile) {
    auto resource = fut_guard.get();
    auto& comm = sync::SelectCommunicator<rc_comm>{}(resource.ref());

    sync::broadcast::send_o(comm, tile.get());
  };
  hpx::dataflow(ex, std::move(send_tile), task_chain(), std::move(tile));
}

template <Coord rc_comm, class T>
hpx::future<matrix::Tile<const T, Device::CPU>> recv_tile(
    hpx::threads::executors::pool_executor ex, common::Pipeline<comm::CommunicatorGrid>& mpi_task_chain,
    TileElementSize tile_size, int rank) {
  using ConstTile_t = matrix::Tile<const T, Device::CPU>;
  using MemView_t = memory::MemoryView<T, Device::CPU>;
  using Tile_t = matrix::Tile<T, Device::CPU>;

  auto recv_bcast_f =
      [rank,
       tile_size](hpx::future<common::PromiseGuard<comm::CommunicatorGrid>> fut_grid) -> ConstTile_t {
    auto resource = fut_grid.get();
    auto& comm = sync::SelectCommunicator<rc_comm>{}(resource.ref());

    MemView_t mem_view(tile_size.linear_size());
    Tile_t tile(tile_size, std::move(mem_view), tile_size.rows());
    sync::broadcast::receive_from_o(rank, comm, tile);
    return std::move(tile);
  };
  return hpx::dataflow(ex, std::move(recv_bcast_f),
                       mpi_task_chain());  // TODO why if I don't move it creates a problem?!
}
}
}
