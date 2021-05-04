//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <hpx/local/future.hpp>

#include <gtest/gtest.h>
#include <mpi.h>

#include "dlaf/communication/communicator.h"
#include "dlaf/communication/executor.h"
#include "dlaf/communication/kernels.h"
#include "dlaf/matrix/matrix.h"

using namespace dlaf;

template <class T>
using Bag = hpx::tuple<
  common::Buffer<std::remove_const_t<T>>,
  common::DataDescriptor<T>
>;

template <class T>
auto makeItContiguous(const matrix::Tile<T, Device::CPU>& tile) {
  common::Buffer<std::remove_const_t<T>> buffer;
  auto tile_data = common::make_data(tile);
  auto what_to_use = common::make_contiguous(tile_data, buffer);
  if (buffer)
    common::copy(tile_data, buffer);
  return Bag<T>(std::move(buffer), std::move(what_to_use));
}

DLAF_MAKE_CALLABLE_OBJECT(makeItContiguous);

template <class T>
auto copyBack(matrix::Tile<T, Device::CPU> const& tile, Bag<T> bag) {
  auto buffer_used = std::move(hpx::get<0>(bag));
  if (buffer_used)
    common::copy(buffer_used, common::make_data(tile));
}

DLAF_MAKE_CALLABLE_OBJECT(copyBack);

template <class T>
auto reduceRecvInPlace(
    common::PromiseGuard<comm::Communicator> pcomm,
    MPI_Op reduce_op,
    Bag<T> bag,
    MPI_Request* req) {
  auto message = comm::make_message(hpx::get<1>(bag));
  auto& communicator = pcomm.ref();

  DLAF_MPI_CALL(
      MPI_Ireduce(
        MPI_IN_PLACE,
        message.data(),
        message.count(),
        message.mpi_type(),
        reduce_op,
        communicator.rank(),
        communicator,
        req));

  return std::move(bag);
}

DLAF_MAKE_CALLABLE_OBJECT(reduceRecvInPlace);

template <class T>
auto reduceSend(
    comm::IndexT_MPI rank_root,
    common::PromiseGuard<comm::Communicator> pcomm,
    MPI_Op reduce_op,
    Bag<const T> bag,
    //matrix::Tile<const T, Device::CPU> const&,
    MPI_Request* req) {
  auto message = comm::make_message(hpx::get<1>(bag));
  auto& communicator = pcomm.ref();

  DLAF_MPI_CALL(
      MPI_Ireduce(
        message.data(),
        nullptr,
        message.count(),
        message.mpi_type(),
        reduce_op,
        rank_root,
        communicator,
        req));
}

DLAF_MAKE_CALLABLE_OBJECT(reduceSend);

template <class T>
auto scheduleReduceRecvInPlace(
    comm::Executor& ex,
    hpx::future<common::PromiseGuard<comm::Communicator>> pcomm,
    MPI_Op reduce_op,
    hpx::future<matrix::Tile<T, Device::CPU>> tile) {

  hpx::future<Bag<T>> bag;
  {
    auto wrapped_res = hpx::split_future(hpx::dataflow(
        matrix::unwrapExtendTiles(makeItContiguous_o),
        std::move(tile)));
    bag = std::move(hpx::get<0>(wrapped_res));
    auto args = hpx::split_future(std::move(hpx::get<1>(wrapped_res)));
    tile = std::move(hpx::get<0>(args));
  }

  // tile still keeps the original tile
  // bag contains the info for communicating with the ireduce
  bag = hpx::dataflow(
      ex,
      hpx::util::unwrapping(reduceRecvInPlace_o),
      std::move(pcomm),
      reduce_op,
      std::move(bag));

  auto wrapped_res = hpx::split_future(hpx::dataflow(
        matrix::unwrapExtendTiles(copyBack_o),
        std::move(tile),
        std::move(bag)));
  tile = std::move(hpx::get<0>(wrapped_res));

  return std::move(tile);
}

template <class T>
auto scheduleReduceSend(
    comm::Executor& ex,
    comm::IndexT_MPI rank_root,
    hpx::future<common::PromiseGuard<comm::Communicator>> pcomm,
    MPI_Op reduce_op,
    hpx::shared_future<matrix::Tile<const T, Device::CPU>> tile) {

  hpx::future<Bag<const T>> bag;
  {
    auto wrapped_res = hpx::split_future(hpx::dataflow(
        matrix::unwrapExtendTiles(makeItContiguous_o),
        tile));
    bag = std::move(hpx::get<0>(wrapped_res));
    auto args = hpx::split_future(std::move(hpx::get<1>(wrapped_res)));
  }

  hpx::dataflow(
      ex,
      hpx::util::unwrapping(reduceSend_o),
      rank_root,
      std::move(pcomm),
      reduce_op,
      std::move(bag)/*,
      tile*/);

  return tile;
}

TEST(Reduce, Contiguous) {
  using namespace std::literals;
  comm::Communicator comm(MPI_COMM_WORLD);
  comm::CommunicatorGrid grid(comm, 1, 2, common::Ordering::ColumnMajor);
  common::Pipeline<comm::Communicator> chain(comm);

  auto ex_mpi = getMPIExecutor<Backend::MC>();

  int root = 0;
  int sz = 4;

  LocalTileIndex index(0, 0);
  dlaf::Matrix<double, Device::CPU> matrix({sz, 1}, {sz, 1});
  matrix(index).get()({0, 0}) = comm.rank() == 0 ? 1 : 3;

  if (comm.rank() == root) {
    auto t = scheduleReduceRecvInPlace(ex_mpi, chain(), MPI_SUM, matrix(index));
    EXPECT_EQ(4, t.get()({0, 0}));
  }
  else {
    auto t = scheduleReduceSend(ex_mpi, root, chain(), MPI_SUM, matrix.read(index));
    EXPECT_EQ(3, t.get()({0, 0}));
  }
}

TEST(Reduce, NotContiguous) {
  using namespace std::literals;
  comm::Communicator comm(MPI_COMM_WORLD);
  comm::CommunicatorGrid grid(comm, 1, 2, common::Ordering::ColumnMajor);
  common::Pipeline<comm::Communicator> chain(comm);

  auto ex_mpi = getMPIExecutor<Backend::MC>();

  int root = 0;
  int sz = 4;

  LocalTileIndex index(0, 0);
  dlaf::Matrix<double, Device::CPU> matrix({1, sz}, {1, sz});
  matrix(index).get()({0, 0}) = comm.rank() == 0 ? 1 : 3;

  if (comm.rank() == root) {
    auto t = scheduleReduceRecvInPlace(ex_mpi, chain(), MPI_SUM, matrix(index));
    EXPECT_EQ(4, t.get()({0, 0}));
  }
  else {
    auto t = scheduleReduceSend(ex_mpi, root, chain(), MPI_SUM, matrix.read(index));
    EXPECT_EQ(3, t.get()({0, 0}));
  }
}
