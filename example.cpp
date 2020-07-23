#include <iostream>
#include <cmath>

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include "dlaf/common/range2d.h"
#include "dlaf/matrix.h"
#include "dlaf/matrix/index.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"

using Type = double;

using dlaf::Coord;
using dlaf::LocalElementSize;
using dlaf::LocalTileIndex;
using dlaf::LocalTileSize;
using dlaf::SizeType;
using dlaf::TileElementSize;
using dlaf::TileElementIndex;

using MatrixType = dlaf::Matrix<Type, dlaf::Device::CPU>;
using ConstMatrixType = dlaf::Matrix<const Type, dlaf::Device::CPU>;
using TileType = dlaf::Tile<Type, dlaf::Device::CPU>;
using ConstTileType = dlaf::Tile<const Type, dlaf::Device::CPU>;

void print(ConstMatrixType& matrix);

int miniapp(hpx::program_options::variables_map& vm) {
  const SizeType m = vm["matrix-rows"].as<SizeType>();
  const SizeType n = vm["matrix-cols"].as<SizeType>();
  const SizeType nb = vm["block-size"].as<SizeType>();

  LocalElementSize matrix_size(m, n);
  TileElementSize block_size(nb, nb);

  MatrixType A{matrix_size, block_size};
  dlaf::matrix::util::set_random(A);

  const auto& distribution = A.distribution();

  std::cout << 'A' << std::endl;
  print(A);
  std::cout << A << std::endl;
  std::cout << A.distribution().localNrTiles() << std::endl;

  using dlaf::common::iterate_range2d;
  using hpx::util::unwrapping;

  // for each panel
  for (SizeType j_current_panel = 0; j_current_panel < distribution.nrTiles().cols(); ++j_current_panel) {
    if (j_current_panel > 0)
      break;

    std::cout << ">>> computing panel " << j_current_panel << std::endl;

    const LocalTileIndex index_tile_x0{j_current_panel, j_current_panel};

    // for each column in the panel, compute reflector and update panel
    for (SizeType j_local_reflector = 0; j_local_reflector < nb; ++j_local_reflector) {  // TODO fix tile size
      std::cout << ">>> computing local reflector " << j_local_reflector << std::endl;

      const TileElementIndex index_el_x0{j_local_reflector, j_local_reflector};

      // compute norm + identify x0 component
      Type x0;
      Type norm_x = 0;
      for (SizeType i = index_tile_x0.row(); i < distribution.nrTiles().rows(); ++i) {
        const ConstTileType& tile_v = A.read(LocalTileIndex{i, index_tile_x0.row()}).get();

        if (i == index_tile_x0.row())
          x0 = tile_v(index_el_x0);

        const SizeType first_tile_element = (i == index_tile_x0.row()) ? index_el_x0.row() : 0;
        for (SizeType i_sub = first_tile_element; i_sub < tile_v.size().rows(); ++i_sub) {
          Type x = tile_v({i_sub, index_el_x0.col()});
          norm_x += x * x;
          std::cout << "norm: " << x << std::endl;
        }
      }
      norm_x = std::sqrt(norm_x);
      std::cout << "|x| = " << norm_x << std::endl;

      std::cout << "x0 = " << x0 << std::endl;

      // compute first component of the reflector
      const Type y = std::signbit(x0) ? norm_x : -norm_x;
      A(index_tile_x0).get()(index_el_x0) = 1;

      std::cout << "y = " << y << std::endl;

      // compute tau
      const Type tau = (y - x0) / y;
      std::cout << "t" << j_local_reflector << " = " << tau << std::endl;

      // compute V (reflector components)
      for (SizeType i = index_tile_x0.row(); i < distribution.nrTiles().rows(); ++i) {
        TileType tile_v = A(LocalTileIndex{i, index_tile_x0.col()}).get();

        const SizeType first_tile_element = (i == index_tile_x0.row()) ? index_el_x0.row() + 1 : 0;
        for (SizeType i_sub = first_tile_element; i_sub < tile_v.size().rows(); ++i_sub) {
          Type& x = tile_v({i_sub, index_el_x0.col()});

          std::cout << "x " << i * nb + i_sub << " " << x << " " << x0 << " " << y << std::endl;
          x = x / (x0 - y);
          std::cout << "x " << i * nb + i_sub << "=" << x << std::endl;
        }
      }

      // compute W
      const LocalElementSize W_size{1, distribution.blockSize().cols()};
      MatrixType W(W_size, distribution.blockSize());
      {
        TileType w = W(LocalTileIndex{0, 0}).get();

        for (int sub_j = index_el_x0.col() + 1; sub_j < W.blockSize().cols(); ++sub_j)
          w({0, sub_j}) = 0;

        // for each tile in the panel
        for (SizeType h = index_tile_x0.row(); h < distribution.nrTiles().rows(); ++h) {
          std::cout << "h " << h << std::endl;

          const LocalTileIndex index_a{h, j_current_panel};
          const ConstTileType& tile = A.read(index_a).get();

          // consider just the trailing panel
          // i.e. all rows (height = reflector), just columns to the right of the current reflector
          const SizeType first_element_in_tile = (h == index_tile_x0.row()) ? index_el_x0.row() : 0;
          for (SizeType sub_i = first_element_in_tile; sub_i < tile.size().rows(); ++sub_i) {
            const TileElementIndex index_el_v{sub_i, index_el_x0.col()};
            const Type v = tile(index_el_v);

            std::cout << "v" << index_a << index_el_v << " " << v << std::endl;

            for (SizeType sub_j = index_el_x0.col() + 1; sub_j < tile.size().cols(); ++sub_j) {
              const TileElementIndex index_el_a{sub_i, sub_j};
              const Type a = tile(index_el_a);

              const TileElementIndex index_el_w{0, sub_j};
              w(index_el_w) += v * a;

              std::cout << "w_p " << index_el_a << " @ " << index_a << index_el_w << " = " << w({0, sub_j}) << " " << v << " " << a << std::endl;
            }
          }
        }
      }
      std::cout << "W" << std::endl;
      print(W);

      // update trailing panel
      {
        TileType w = W(LocalTileIndex{0, 0}).get();

        for (SizeType h = index_tile_x0.row(); h < distribution.nrTiles().rows(); ++h) {
          TileType tile_a = A(LocalTileIndex{h, j_current_panel}).get();

          const SizeType first_element_in_tile = (h == index_tile_x0.row()) ? index_el_x0.row() : 0;
          for (SizeType sub_i = first_element_in_tile; sub_i < tile_a.size().rows(); ++sub_i) {
            const Type v = tile_a({sub_i, j_local_reflector});

            std::cout << "v = " << v << std::endl;

            for (SizeType sub_j = j_local_reflector + 1; sub_j < tile_a.size().cols(); ++sub_j) {
              const TileElementIndex index_el_a{sub_i, sub_j};
              Type& a = tile_a(index_el_a);
              std::cout << "a " << index_el_a << " " << tau << " " << v << " " << w({0, sub_j}) << std::endl;
              a -= tau * v * w({0, sub_j});
              std::cout << "a " << index_el_a << "=" << a << std::endl;
            }
          }
        }
      }

      // put in place the previously computed result
      A(index_tile_x0).get()({j_local_reflector, j_local_reflector}) = y;

      std::cout << 'A' << std::endl;
      print(A);
    }

    // TODO compute T-factor
    // TODO update each panel with the T-factor
  }

  std::cout << 'A' << std::endl;
  print(A);

  return hpx::finalize();
}

int main(int argc, char** argv) {
  // options
  using namespace hpx::program_options;
  options_description desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");


  // clang-format off
  desc_commandline.add_options()
    ("matrix-rows", value<SizeType>()->default_value(4), "Matrix rows")
    ("matrix-cols", value<SizeType>()->default_value(4), "Matrix cols")
    ("block-size",  value<SizeType>()->default_value(2), "Block cyclic distribution size");
  // clang-format on

  auto ret_code = hpx::init(miniapp, desc_commandline, argc, argv);

  return ret_code;
}

void print(ConstMatrixType& matrix) {
  using dlaf::common::iterate_range2d;

  std::ostringstream ss;
  ss << "np.array([";

  const auto& distribution = matrix.distribution();
  auto matrix_size = distribution.size();
  for (const auto& index_g : iterate_range2d(matrix_size)) {
    dlaf::GlobalTileIndex tile_g = distribution.globalTileIndex(index_g);
    TileElementIndex index_e = distribution.tileElementIndex(index_g);

    const auto& tile = matrix.read(tile_g).get();

    std::cout << index_g << " " << index_e << " " << tile(index_e) << std::endl;
    ss << tile(index_e) << ", ";
  }

  bool is_vector = matrix_size.rows() == 1 || matrix_size.cols() == 1;
  if (!is_vector)
    matrix_size.transpose();
  ss << "]).reshape" << matrix_size << (is_vector ? "" : ".T") << std::endl;

  std::cout << ss.str();
}
