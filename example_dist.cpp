#include <iostream>

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include "dlaf/common/index2d.h"
#include "dlaf/common/range2d.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/init.h"
#include "dlaf/eigensolver/mc.h"
#include "dlaf/matrix.h"
#include "dlaf/matrix/index.h"
#include "dlaf/types.h"

template <class Type>
void print(dlaf::Matrix<const Type, dlaf::Device::CPU>& matrix, std::string prefix);

int miniapp(hpx::program_options::variables_map& vm) {
  using Type = double;

  using dlaf::SizeType;
  using dlaf::comm::Communicator;
  using dlaf::comm::CommunicatorGrid;

  const SizeType n = vm["matrix-rows"].as<SizeType>();
  const SizeType nb = vm["block-size"].as<SizeType>();

  const SizeType grid_rows = vm["grid-rows"].as<SizeType>();
  const SizeType grid_cols = vm["grid-cols"].as<SizeType>();

  Communicator world(MPI_COMM_WORLD);
  CommunicatorGrid comm_grid(world, grid_rows, grid_cols, dlaf::common::Ordering::ColumnMajor);

  dlaf::GlobalElementSize matrix_size(n, n);
  dlaf::TileElementSize block_size(nb, nb);

  dlaf::Matrix<Type, dlaf::Device::CPU> mat_a(matrix_size, block_size, comm_grid);
  dlaf::matrix::util::set_random_hermitian(mat_a);

  print(mat_a, "A");

  dlaf::EigenSolver<dlaf::Backend::MC>::reduction_to_band(comm_grid, mat_a);

  print(mat_a, "in-place A");

  return hpx::finalize();
}

int main(int argc, char** argv) {
  using dlaf::SizeType;

  dlaf::comm::mpi_init mpi_initter(argc, argv, dlaf::comm::mpi_thread_level::serialized);

  // options
  using namespace hpx::program_options;
  options_description desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

  // clang-format off
  desc_commandline.add_options()
    ("matrix-rows", value<SizeType>() ->default_value(4), "Matrix rows")
    ("block-size",  value<SizeType>() ->default_value(2), "Block cyclic distribution size")
    ("grid-rows",   value<int>()      ->default_value(1), "Number of row processes in the 2D communicator")
    ("grid-cols",   value<int>()      ->default_value(1), "Number of column processes in the 2D communicator");
  // clang-format on

  auto ret_code = hpx::init(miniapp, desc_commandline, argc, argv);

  std::cout << "finished" << std::endl;

  return ret_code;
}

template <class Type>
void print(dlaf::Matrix<const Type, dlaf::Device::CPU>& matrix, std::string prefix) {
  using dlaf::common::iterate_range2d;
  using dlaf::Coord;

  const auto& dist = matrix.distribution();

  std::ostringstream ss;
  ss << prefix << " = np.zeros((" << dist.size() << "))" << std::endl;

  for (const auto& index_tile : iterate_range2d(dist.localNrTiles())) {
    const auto& tile = matrix.read(index_tile).get();

    for (const auto& index_el : iterate_range2d(tile.size())) {
      dlaf::GlobalElementIndex index_g{
          dist.template globalElementFromLocalTileAndTileElement<Coord::Row>(index_tile.row(),
                                                                             index_el.row()),
          dist.template globalElementFromLocalTileAndTileElement<Coord::Col>(index_tile.col(),
                                                                             index_el.col()),
      };
      ss << prefix << "[" << index_g.row() << "," << index_g.col() << "] = " << tile(index_el)
         << std::endl;
    }
  }

  std::cout << ss.str() << std::endl;
}
