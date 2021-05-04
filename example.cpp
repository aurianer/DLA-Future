#include <hpx/local/future.hpp>
#include <hpx/hpx_init.hpp>

#include "dlaf/common/data.h"
#include "dlaf/common/index2d.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/executor.h"
#include "dlaf/executors.h"
#include "dlaf/init.h"
#include "dlaf/matrix/index.h"
#include "dlaf/memory/memory_view.h"
#include "dlaf/matrix/tile.h"
#include "dlaf/matrix/matrix.h"

#include "dlaf/communication/kernels.h"
#include "dlaf/types.h"

using T = float;
constexpr auto D = dlaf::Device::CPU;

using namespace dlaf;

template <class T>
using Bag = hpx::tuple<
  common::Buffer<std::remove_const_t<T>>,
  common::DataDescriptor<T>
>;

template <class T>
auto makeItContiguous(const matrix::Tile<T, D>& tile) {
  common::Buffer<std::remove_const_t<T>> buffer;
  auto tile_data = common::make_data(tile);
  auto what_to_use = common::make_contiguous(tile_data, buffer);
  if (buffer)
    common::copy(tile_data, buffer);
  return Bag<T>(std::move(buffer), std::move(what_to_use));
}

DLAF_MAKE_CALLABLE_OBJECT(makeItContiguous);

template <class T>
auto copyBack(matrix::Tile<T, D> const& tile, Bag<T> bag) {
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

auto scheduleReduceRecvInPlace(
    comm::Executor& ex,
    hpx::future<common::PromiseGuard<comm::Communicator>> pcomm,
    MPI_Op reduce_op,
    hpx::future<matrix::Tile<T, D>> tile) {

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

auto scheduleReduceSend(
    comm::Executor& ex,
    comm::IndexT_MPI rank_root,
    hpx::future<common::PromiseGuard<comm::Communicator>> pcomm,
    MPI_Op reduce_op,
    hpx::shared_future<matrix::Tile<const T, D>> tile) {

  hpx::future<Bag<const T>> bag;
  {
    auto wrapped_res = hpx::split_future(hpx::dataflow(
        matrix::unwrapExtendTiles(makeItContiguous_o),
        std::move(tile)));
    bag = std::move(hpx::get<0>(wrapped_res));
    auto args = hpx::split_future(std::move(hpx::get<1>(wrapped_res)));
  }

  hpx::dataflow(
      ex,
      hpx::util::unwrapping(reduceSend_o),
      rank_root,
      std::move(pcomm),
      reduce_op,
      std::move(bag));

  return tile;
}

int app(int argc, char** argv) {
  dlaf::initialize(argc, argv);

  comm::CommunicatorGrid grid({MPI_COMM_WORLD}, 2, 2, common::Ordering::ColumnMajor);

  matrix::Matrix<T, D> matrix({1, 2}, {1, 2});
  const LocalTileIndex index(0, 0);

  auto ex_mpi = getMPIExecutor<Backend::MC>();

  common::Pipeline<comm::Communicator> chain(grid.fullCommunicator());

  matrix(index).get()({0, 0}) = grid.rank().row() * 2 + grid.rank().col();

  if (grid.rank() == comm::Index2D{0, 0}) {
    auto t = scheduleReduceRecvInPlace(ex_mpi, chain(), MPI_SUM, matrix(index));
    std::cout << t.get()({0, 0}) << "\n";
  }
  else {
    scheduleReduceSend(ex_mpi, 0, chain(), MPI_SUM, matrix.read(index));
  }

  return dlaf::finalize(), hpx::finalize();
}

int main(int argc, char ** argv) {
  int threading_required = MPI_THREAD_MULTIPLE;
  int threading_provided;
  MPI_Init_thread(&argc, &argv, threading_required, &threading_provided);

  using namespace hpx::program_options;
  options_description desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");
  desc_commandline.add(dlaf::getOptionsDescription());

  hpx::init_params p;
  p.desc_cmdline = desc_commandline;
  p.rp_callback = dlaf::initResourcePartitionerHandler;

  auto ret = hpx::init(app, argc, argv, p);

  DLAF_MPI_CALL(MPI_Finalize());
}
