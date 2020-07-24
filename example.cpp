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
    for (SizeType j_local_reflector = 0; j_local_reflector < nb; ++j_local_reflector) {
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

      // is there a remaining panel?
      if (j_local_reflector < nb - 1) {
        // compute W
        const LocalElementSize W_size{1, nb};
        MatrixType W(W_size, distribution.blockSize());
        {
          TileType w = W(LocalTileIndex{0, 0}).get();

          // for each tile in the panel
          for (SizeType h = index_tile_x0.row(); h < distribution.nrTiles().rows(); ++h) {
            const LocalTileIndex index_a{h, j_current_panel};
            const ConstTileType& tile = A.read(index_a).get();

            // consider just the trailing panel
            // i.e. all rows (height = reflector), just columns to the right of the current reflector
            const SizeType first_element_in_tile = (h == index_tile_x0.row()) ? index_el_x0.row() : 0;

            const TileElementSize A_size{tile.size().rows() - first_element_in_tile,
                                         tile.size().cols() - (index_el_x0.col() + 1)};
            const TileElementIndex A_start{first_element_in_tile, index_el_x0.col() + 1};
            const TileElementIndex V_start{first_element_in_tile, index_el_x0.col()};
            const TileElementIndex W_start{0, index_el_x0.col() + 1};

            // w = P* . v
            const Type beta = (h == index_tile_x0.row() ? 0 : 1);
            blas::gemv(blas::Layout::ColMajor, blas::Op::ConjTrans, A_size.rows(), A_size.cols(), 1.0,
                       tile.ptr(A_start), tile.ld(), tile.ptr(V_start), 1, beta, w.ptr(W_start), w.ld());
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

            const TileElementSize V_size{tile_a.size().rows() - first_element_in_tile, 1};
            const TileElementSize W_size{1, tile_a.size().cols() - (index_el_x0.col() + 1)};
            const TileElementSize A_size{V_size.rows(), W_size.cols()};

            const TileElementIndex A_start{first_element_in_tile, index_el_x0.col() + 1};
            const TileElementIndex V_start{first_element_in_tile, index_el_x0.col()};
            const TileElementIndex W_start{0, index_el_x0.col() + 1};

            // Pt = Pt - tau * v * w*
            const Type alpha = -tau;
            blas::ger(blas::Layout::ColMajor, A_size.rows(), A_size.cols(), alpha, tile_a.ptr(V_start),
                      1, w.ptr(W_start), w.ld(), tile_a.ptr(A_start), tile_a.ld());
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
