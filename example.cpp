#include <cmath>
#include <iostream>

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include "dlaf/common/index2d.h"
#include "dlaf/common/range2d.h"
#include "dlaf/matrix.h"
#include "dlaf/matrix/index.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"
#include "dlaf/lapack_tile.h"

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
    // TODO just for debugging, compute just first panel
    if (j_current_panel > 0)
      break;

    const LocalTileIndex index_tile_x0{j_current_panel + 1, j_current_panel};
    const LocalTileSize Ai_size{distribution.nrTiles().rows() - index_tile_x0.row(), 1};

    std::cout << ">>> computing panel " << index_tile_x0 << " " << Ai_size << std::endl;

    MatrixType T(LocalElementSize{nb, nb}, distribution.blockSize());

    // for each column in the panel, compute reflector and update panel
    for (SizeType j_local_reflector = 0; j_local_reflector < nb; ++j_local_reflector) {
      // TODO fix check: is there a panel underneath to annihilate?
      if (Ai_size.rows() == 1 && j_local_reflector == nb - 1)
        break;

      std::cout << ">>> computing local reflector " << j_local_reflector << std::endl;

      const TileElementIndex index_el_x0{j_local_reflector, j_local_reflector};

      // compute norm + identify x0 component
      Type x0;
      Type norm_x = 0;
      for (SizeType i = index_tile_x0.row(); i < distribution.nrTiles().rows(); ++i) {
        const ConstTileType& tile_v = A.read(LocalTileIndex{i, index_tile_x0.col()}).get();

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
            const LocalTileIndex index_a{h, index_tile_x0.col()};
            const ConstTileType& tile = A.read(index_a).get();

            // consider just the trailing panel
            // i.e. all rows (height = reflector), just columns to the right of the current reflector
            const SizeType first_element_in_tile = (h == index_tile_x0.row()) ? index_el_x0.row() : 0;

            const TileElementSize A_size{tile.size().rows() - first_element_in_tile,
                                         tile.size().cols() - (index_el_x0.col() + 1)};
            const TileElementIndex A_start{first_element_in_tile, index_el_x0.col() + 1};
            const TileElementIndex V_start{first_element_in_tile, index_el_x0.col()};
            const TileElementIndex W_start{0, index_el_x0.col() + 1};

            // w += 1 . A* . v
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
            TileType tile_a = A(LocalTileIndex{h, index_tile_x0.col()}).get();

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

      // TODO compute T-factor component for this reflector
      // T(0:j, j) = T(0:j, 0:j) . -tau(j) . V(j:, 0:j)* . V(j:, j)
      const TileElementSize T_size{index_el_x0.row(), 1};
      const TileElementIndex T_start{0, index_el_x0.col()};
      {
        TileType tile_t = T(LocalTileIndex{0, 0}).get();
        for (const auto& index_v : iterate_range2d(index_tile_x0, Ai_size)) {
          std::cout << "* computing T " << index_v << std::endl;

          const SizeType first_element_in_tile = (index_v.row() == index_tile_x0.row()) ? index_el_x0.row() + 1 : 0;

          const ConstTileType& tile_v = A.read(index_v).get();

          // T(0:j, j) = -tau . V(j:, 0:j)* . V(j:, j)
          // [j x 1] = [(n-j) x j]* . [(n-j) x 1]
          const TileElementSize V_size{tile_v.size().rows() - first_element_in_tile, index_el_x0.col()};
          const TileElementIndex Va_start{first_element_in_tile, 0};
          const TileElementIndex Vb_start{first_element_in_tile, index_el_x0.col()};

          // set tau on the diagonal
          if (index_v.row() == index_tile_x0.row()) {
            std::cout << "t on diagonal " << tau << std::endl;
            tile_t(index_el_x0) = tau;

            // compute first component with implicit one
            for (const auto& index_el_t : iterate_range2d(T_start, T_size)) {
              const auto index_el_va = dlaf::common::internal::transposed(index_el_t);
              tile_t(index_el_t) = -tau * tile_v(index_el_va);

              std::cout << tile_t(index_el_t) << " " << -tau << " " << tile_v(index_el_va) << std::endl;
            }
          }

          std::cout << "GEMV?" << Va_start << " " << V_size << "  " << Vb_start << std::endl;
          if (Va_start.row() < tile_v.size().rows() && Vb_start.row() < tile_v.size().rows()) {
            std::cout << "GEMV!" << std::endl;
            for (SizeType i_loc = 0; i_loc < tile_t.size().rows(); ++i_loc)
              std::cout << "t[" << i_loc << "] " << tile_t({i_loc, j_local_reflector}) << std::endl;

            // t = -tau . V* . V
            const Type alpha = -tau;
            const Type beta = 1;
            blas::gemv(blas::Layout::ColMajor, blas::Op::ConjTrans,
                V_size.rows(), V_size.cols(),
                alpha,
                tile_v.ptr(Va_start), tile_v.ld(),
                tile_v.ptr(Vb_start), 1,
                beta, tile_t.ptr(T_start), 1);

            for (SizeType i_loc = 0; i_loc < tile_t.size().rows(); ++i_loc)
              std::cout << "t*[" << i_loc << "] " << tile_t({i_loc, j_local_reflector}) << std::endl;
          }
        }

        std::cout << "TRMV" << std::endl;
        for (SizeType i_loc = 0; i_loc < tile_t.size().rows(); ++i_loc)
          std::cout << "t[" << i_loc << "] " << tile_t({i_loc, j_local_reflector}) << std::endl;

        std::cout << tile_t(T_start) << " " << tile_t({0, 0}) << std::endl;

        // t = T . t
        blas::trmv(
            blas::Layout::ColMajor,
            blas::Uplo::Upper, blas::Op::NoTrans, blas::Diag::NonUnit,
            j_local_reflector,
            tile_t.ptr({0, 0}), tile_t.ld(),
            tile_t.ptr(T_start), 1);

        for (SizeType i_loc = 0; i_loc < tile_t.size().rows(); ++i_loc)
          std::cout << "t[" << i_loc << "] " << tile_t({i_loc, j_local_reflector}) << std::endl;
      }

      std::cout << "T" << std::endl;
      print(T);
    }

    // TODO update trailing matrix
    const LocalTileIndex At_start{index_tile_x0.row(), index_tile_x0.col() + 1};

    std::cout << "update trailing matrix " << At_start << std::endl;

    MatrixType W({Ai_size.rows() * nb, Ai_size.cols() * nb}, distribution.blockSize());

    {
      const ConstTileType& tile_t = T.read(LocalTileIndex{0, 0}).get();

      for (SizeType i_t = At_start.row(); i_t < distribution.nrTiles().rows(); ++i_t) {
        const LocalTileIndex index_tile_v{i_t, j_current_panel};
        const LocalTileIndex index_tile_w{i_t - At_start.row(), 0};

        std::cout << "computing W" << index_tile_w << " with V " << index_tile_v << std::endl;

        // TODO TRMM W = V . T
        TileType tile_w = W(index_tile_w).get();

        // copy
        const ConstTileType& tile_v = A.read(index_tile_v).get();

        const bool is_diagonal_tile = index_tile_v.row() == At_start.row();

        lapack::lacpy(
            is_diagonal_tile ? lapack::MatrixType::Lower : lapack::MatrixType::General,
            tile_v.size().rows(), tile_v.size().cols(),
            tile_v.ptr(), tile_v.ld(),
            tile_w.ptr(), tile_w.ld());

        if (is_diagonal_tile) { // is this the first one? (diagonal)
          std::cout << "setting diagonal V on W" << std::endl;
          // set upper part to zero and 1 on diagonal (reflectors)
          lapack::laset(lapack::MatrixType::Upper,
              tile_w.size().rows(), tile_w.size().cols(),
              Type(0), Type(1),
              tile_w.ptr(), tile_w.ld());
        }

        // W = V . T
        blas::trmm(
            blas::Layout::ColMajor, blas::Side::Right,
            blas::Uplo::Upper, blas::Op::NoTrans, blas::Diag::NonUnit,
            tile_w.size().rows(), tile_w.size().cols(),
            Type(1),
            tile_t.ptr(), tile_t.ld(),
            tile_w.ptr(), tile_w.ld());
      }
    }

    std::cout << "W" << std::endl;
    print(W);

    // TODO HEMM X = At . W

    // TODO GEMM W2 = W* . X
    // TODO GEMM X = X - 0.5 . V . W2
    // TODO HER2K At = At - X . V* + V . X*
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

  std::cout << "finished" << std::endl;

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

    //std::cout << index_g << " " << index_e << " " << tile(index_e) << std::endl;
    ss << tile(index_e) << ", ";
  }

  bool is_vector = matrix_size.rows() == 1 || matrix_size.cols() == 1;
  if (!is_vector)
    matrix_size.transpose();
  ss << "]).reshape" << matrix_size << (is_vector ? "" : ".T") << std::endl;

  std::cout << ss.str();
}
