#include <iostream>
#include <cmath>

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include "dlaf/common/range2d.h"
#include "dlaf/matrix.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"

using Type = double;

using dlaf::Coord;
using dlaf::LocalElementSize;
using dlaf::LocalTileIndex;
using dlaf::LocalTileSize;
using dlaf::SizeType;
using dlaf::TileElementSize;

using MatrixType = dlaf::Matrix<Type, dlaf::Device::CPU>;
using ConstMatrixType = dlaf::Matrix<const Type, dlaf::Device::CPU>;
using TileType = dlaf::Tile<Type, dlaf::Device::CPU>;
using ConstTileType = dlaf::Tile<const Type, dlaf::Device::CPU>;

void print(ConstMatrixType& matrix);

int miniapp(hpx::program_options::variables_map& vm) {
  const SizeType n = 3;
  const SizeType nb = 1;

  LocalElementSize matrix_size(n, n);
  TileElementSize block_size(nb, nb);

  // DATA
  Type dataA[n * n * nb * nb] = {12, 6, -4, -51, 167, 24, 4, -68, -41};           // A

  MatrixType A = dlaf::createMatrixFromColMajor<dlaf::Device::CPU>(
      matrix_size, block_size, matrix_size.rows(), dataA);

  const auto& distribution = A.distribution();

  std::cout << A << std::endl;
  std::cout << A.distribution().localNrTiles() << std::endl;

  using dlaf::common::iterate_range2d;
  using hpx::util::unwrapping;

  for (SizeType j = 0; j < distribution.nrTiles().cols() && distribution.nrTiles().rows() - j > 1; ++j) {
    std::cout << ">>> computing reflector " << j << std::endl;

    const LocalTileIndex reflector_x0{j, j};
    const Type x0 = A.read(reflector_x0).get()({0, 0});
    std::cout << "x0=" << x0 << std::endl;

    // TODO compute norm
    Type norm_x = 0;
    const LocalTileSize reflector_size(distribution.localNrTiles().rows() - j, 1);
    for (const auto& index_x : iterate_range2d(reflector_x0, reflector_size)) {
      const Type x = A.read(index_x).get()({0, 0});
      norm_x += x * x;
    }
    norm_x = std::sqrt(norm_x);

    std::cout << "|x|=" << norm_x << std::endl;

    // TODO compute y
    const Type y = std::signbit(x0) ? norm_x : -norm_x;
    A(reflector_x0).get()({0, 0}) = y;
    std::cout << "y=" << y << std::endl;

    // TODO compute tau
    const Type tau = (y - x0) / y;
    std::cout << "t=" << tau << std::endl;

    // TODO compute reflector components
    for (const auto& index_x : iterate_range2d(reflector_x0 + LocalTileSize{1, 0}, reflector_size - LocalTileSize{1, 0})) {
      Type& x = A(index_x).get()({0, 0});
      x = x / (x0 - y);
      std::cout << "V" << index_x << " " << x << std::endl;
    }

    // TODO compute W
    const LocalElementSize W_size{1, distribution.size().cols() - (reflector_x0.row() + 1)};
    MatrixType W(W_size, distribution.blockSize());
    for (const auto& index_w : iterate_range2d(W.distribution().localNrTiles())) {
      Type& w = W(LocalTileIndex{0, index_w.col()}).get()({0, 0});

      for (SizeType k = reflector_x0.row(); k < distribution.localNrTiles().rows(); ++k) {
        const LocalTileIndex index_v{k, reflector_x0.col()};
        const LocalTileIndex index_A{k, reflector_x0.col() + 1 + index_w.col()};

        const Type v = (k == reflector_x0.row()) ? 1 : A(index_v).get()({0, 0});
        const Type a = A.read(index_A).get()({0, 0});

        std::cout << "Wpartial" << index_w << " " << v << "*" << a << std::endl;
        w += v * a;
      }

      std::cout << "W" << index_w << " " << w << std::endl;
    }

    // TODO update trailing panel
    const LocalTileIndex tl_trailing{reflector_x0.row(), reflector_x0.col() + 1};
    const LocalTileSize trailing_matrix{
      distribution.localNrTiles().rows() - tl_trailing.row(),
      distribution.localNrTiles().cols() - tl_trailing.col(),
    };
    for (const auto& index_result : iterate_range2d(tl_trailing, trailing_matrix)) {
      const LocalTileIndex index_v{index_result.row(), tl_trailing.row()};

      const Type v = (index_result.row() == tl_trailing.row()) ? 1 : A.read(index_v).get()({0, 0});
      const Type w = W.read(LocalTileIndex{0, index_result.col() - tl_trailing.col()}).get()({0, 0});

      Type& a = A(index_result).get()({0, 0});

      std::cout << "A" << index_result << " " << a << " " << tau << "*" << v << "*" << w << std::endl;
      a -= tau * v * w;
      std::cout << "A" << index_result << "=" << a << std::endl;
    }
  }

  std::cout << 'A' << std::endl;
  print(A);

  return hpx::finalize();
}

int main(int argc, char** argv) {
  // options
  using namespace hpx::program_options;
  options_description desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

  auto ret_code = hpx::init(miniapp, desc_commandline, argc, argv);

  return ret_code;
}

void print(ConstMatrixType& matrix) {
  using dlaf::common::iterate_range2d;
  for (const auto& index : iterate_range2d(matrix.distribution().localNrTiles()))
    std::cout << index << " " << matrix.read(index).get()({0, 0}) << std::endl;
}
