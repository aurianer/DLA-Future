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

using MemoryViewType = dlaf::memory::MemoryView<Type, dlaf::Device::CPU>;

void print(ConstMatrixType& matrix);
void print_tile(const ConstTileType& tile);

int miniapp(hpx::program_options::variables_map& vm) {
  const SizeType m = vm["matrix-rows"].as<SizeType>();
  const SizeType n = vm["matrix-cols"].as<SizeType>();
  const SizeType nb = vm["block-size"].as<SizeType>();

  LocalElementSize matrix_size(m, n);
  TileElementSize block_size(nb, nb);

  MatrixType A{matrix_size, block_size};
  dlaf::matrix::util::set_random_hermitian(A);

  const auto& distribution = A.distribution();

  std::cout << 'A' << std::endl;
  print(A);
  std::cout << A << std::endl;
  std::cout << A.distribution().localNrTiles() << std::endl;

  using dlaf::common::iterate_range2d;
  using hpx::util::unwrapping;

  // for each panel
  for (SizeType j_current_panel = 0; j_current_panel < distribution.nrTiles().cols() &&
                                     (j_current_panel + 1) < distribution.nrTiles().rows();
       ++j_current_panel) {

    //if (j_current_panel > 0) break;

    const LocalTileIndex Ai_start{j_current_panel + 1, j_current_panel};
    const LocalTileSize Ai_size{distribution.nrTiles().rows() - Ai_start.row(), 1};

    const LocalTileIndex At_start{Ai_start.row(), Ai_start.col() + 1};
    const LocalTileSize At_size{Ai_size.rows(), A.nrTiles().cols() - (Ai_start.col() + 1)};

    std::cout << std::endl << ">>> computing panel" << std::endl;
    std::cout << ">>> Ai " << A.read(Ai_start).get()({0, 0}) << " " << Ai_size << " " << Ai_start << std::endl;

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
      for (SizeType i = Ai_start.row(); i < distribution.nrTiles().rows(); ++i) {
        const ConstTileType& tile_v = A.read(LocalTileIndex{i, Ai_start.col()}).get();

        if (i == Ai_start.row())
          x0 = tile_v(index_el_x0);

        const SizeType first_tile_element = (i == Ai_start.row()) ? index_el_x0.row() : 0;
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
      A(Ai_start).get()(index_el_x0) = 1;

      std::cout << "y = " << y << std::endl;

      // compute tau
      const Type tau = (y - x0) / y;
      std::cout << "t" << j_local_reflector << " = " << tau << std::endl;

      // compute V (reflector components)
      for (SizeType i = Ai_start.row(); i < distribution.nrTiles().rows(); ++i) {
        TileType tile_v = A(LocalTileIndex{i, Ai_start.col()}).get();

        const SizeType first_tile_element = (i == Ai_start.row()) ? index_el_x0.row() + 1 : 0;
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
          for (SizeType h = Ai_start.row(); h < distribution.nrTiles().rows(); ++h) {
            const LocalTileIndex index_a{h, Ai_start.col()};
            const ConstTileType& tile = A.read(index_a).get();

            // consider just the trailing panel
            // i.e. all rows (height = reflector), just columns to the right of the current reflector
            const SizeType first_element_in_tile = (h == Ai_start.row()) ? index_el_x0.row() : 0;

            const TileElementSize A_size{tile.size().rows() - first_element_in_tile,
                                         tile.size().cols() - (index_el_x0.col() + 1)};
            const TileElementIndex A_start{first_element_in_tile, index_el_x0.col() + 1};
            const TileElementIndex V_start{first_element_in_tile, index_el_x0.col()};
            const TileElementIndex W_start{0, index_el_x0.col() + 1};

            // w += 1 . A* . v
            const Type beta = (h == Ai_start.row() ? 0 : 1);
            blas::gemv(blas::Layout::ColMajor, blas::Op::ConjTrans, A_size.rows(), A_size.cols(), 1.0,
                       tile.ptr(A_start), tile.ld(), tile.ptr(V_start), 1, beta, w.ptr(W_start), w.ld());
          }
        }
        std::cout << "W" << std::endl;
        print(W);

        // update trailing panel
        {
          TileType w = W(LocalTileIndex{0, 0}).get();

          for (SizeType h = Ai_start.row(); h < distribution.nrTiles().rows(); ++h) {
            TileType tile_a = A(LocalTileIndex{h, Ai_start.col()}).get();

            const SizeType first_element_in_tile = (h == Ai_start.row()) ? index_el_x0.row() : 0;

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
      A(Ai_start).get()({j_local_reflector, j_local_reflector}) = y;

      std::cout << 'A' << std::endl;
      print(A);

      // TODO compute T-factor component for this reflector
      // T(0:j, j) = T(0:j, 0:j) . -tau(j) . V(j:, 0:j)* . V(j:, j)
      const TileElementSize T_size{index_el_x0.row(), 1};
      const TileElementIndex T_start{0, index_el_x0.col()};
      {
        TileType tile_t = T(LocalTileIndex{0, 0}).get();
        for (const auto& index_v : iterate_range2d(Ai_start, Ai_size)) {
          std::cout << "* computing T " << index_v << std::endl;

          const SizeType first_element_in_tile =
              (index_v.row() == Ai_start.row()) ? index_el_x0.row() + 1 : 0;

          const ConstTileType& tile_v = A.read(index_v).get();

          // T(0:j, j) = -tau . V(j:, 0:j)* . V(j:, j)
          // [j x 1] = [(n-j) x j]* . [(n-j) x 1]
          const TileElementSize V_size{tile_v.size().rows() - first_element_in_tile, index_el_x0.col()};
          const TileElementIndex Va_start{first_element_in_tile, 0};
          const TileElementIndex Vb_start{first_element_in_tile, index_el_x0.col()};

          // set tau on the diagonal
          if (index_v.row() == Ai_start.row()) {
            std::cout << "t on diagonal " << tau << std::endl;
            tile_t(index_el_x0) = tau;

            // compute first component with implicit one
            for (const auto& index_el_t : iterate_range2d(T_start, T_size)) {
              const auto index_el_va = dlaf::common::internal::transposed(index_el_t);
              tile_t(index_el_t) = -tau * tile_v(index_el_va);

              std::cout << tile_t(index_el_t) << " " << -tau << " " << tile_v(index_el_va) << std::endl;
            }
          }

          if (Va_start.row() < tile_v.size().rows() && Vb_start.row() < tile_v.size().rows()) {
            std::cout << "GEMV" << Va_start << " " << V_size << "  " << Vb_start << std::endl;
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

      std::cout << "T(partial)" << std::endl;
      print(T);
    }

    std::cout << "T" << std::endl;
    print(T);

    // TODO update trailing matrix
    std::cout << ">>> update trailing matrix" << std::endl;
    std::cout << ">>> At " << A.read(At_start).get()({0, 0}) << " " << At_size << " " << At_start << std::endl;

    MatrixType W({Ai_size.rows() * nb, nb}, distribution.blockSize());
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
    MatrixType X({At_size.rows() * nb, W.size().cols()}, distribution.blockSize());
    dlaf::matrix::util::set(X, [](auto&&) { return 0; });

    for (SizeType i_t = At_start.row(); i_t < distribution.nrTiles().rows(); ++i_t) {
      for (SizeType j_t = At_start.col(); j_t <= i_t; ++j_t) {
        const LocalTileIndex index_tile_at{i_t, j_t};

        std::cout << "computing X " << index_tile_at << std::endl;

        const ConstTileType& tile_a = A.read(index_tile_at).get();

        const bool is_diagonal_tile = (i_t == j_t);

        if (is_diagonal_tile) {
          // HEMM
          const LocalTileIndex index_tile_x{index_tile_at.row() - At_start.row(), 0};
          const LocalTileIndex index_tile_w = index_tile_x;

          std::cout << "HEMM " << index_tile_x << " " << index_tile_at << " " << index_tile_w << std::endl;

          TileType tile_x = X(index_tile_x).get();
          const ConstTileType& tile_w = W.read(index_tile_w).get();

          blas::hemm(blas::Layout::ColMajor, blas::Side::Left, blas::Uplo::Lower,
              tile_x.size().rows(), tile_a.size().cols(),
              Type(1),
              tile_a.ptr(), tile_a.ld(),
              tile_w.ptr(), tile_w.ld(),
              Type(1),
              tile_x.ptr(), tile_x.ld());
        }
        else {
          // A  . W
          {
            const LocalTileIndex index_tile_x{index_tile_at.row() - At_start.row(), 0};
            const LocalTileIndex index_tile_w{index_tile_at.col() - At_start.col(), 0};

            std::cout << "GEMM " << index_tile_x << " " << index_tile_at << " " << index_tile_w << std::endl;

            TileType tile_x = X(index_tile_x).get();
            const ConstTileType& tile_w = W.read(index_tile_w).get();

            blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                tile_a.size().rows(), tile_w.size().cols(), tile_a.size().cols(),
                Type(1),
                tile_a.ptr(), tile_a.ld(),
                tile_w.ptr(), tile_w.ld(),
                Type(1),
                tile_x.ptr(), tile_x.ld());
          }

          // A* . W
          {
            const LocalTileIndex index_tile_x{index_tile_at.col() - At_start.col(), 0};
            const LocalTileIndex index_tile_w{index_tile_at.row() - At_start.row(), 0};

            std::cout << "GEMM* " << index_tile_x << " " << index_tile_at << " " << index_tile_w << std::endl;

            TileType tile_x = X(index_tile_x).get();
            const ConstTileType& tile_w = W.read(index_tile_w).get();

            blas::gemm(blas::Layout::ColMajor, blas::Op::ConjTrans, blas::Op::NoTrans,
                tile_a.size().rows(), tile_w.size().cols(), tile_a.size().cols(),
                Type(1),
                tile_a.ptr(), tile_a.ld(),
                tile_w.ptr(), tile_w.ld(),
                Type(1),
                tile_x.ptr(), tile_x.ld());
          }
        }
      }
    }

    std::cout << "X" << std::endl;
    print(X);

    // TODO GEMM W2 = W* . X
    {
      // re-use T as W2
      TileType tile_w2 = T(LocalTileIndex{0, 0}).get();
      for (const auto& index_tile : iterate_range2d(W.nrTiles())) {
        const ConstTileType& tile_w = W.read(index_tile).get();
        const ConstTileType& tile_x = X.read(index_tile).get();

        const Type beta = (index_tile.row() == 0) ? 0 : 1;
        blas::gemm(blas::Layout::ColMajor,
            blas::Op::ConjTrans, blas::Op::NoTrans,
            tile_w2.size().rows(), tile_w2.size().cols(), tile_w.size().cols(),
            Type(1),
            tile_w.ptr(), tile_w.ld(),
            tile_x.ptr(), tile_x.ld(),
            beta,
            tile_w2.ptr(), tile_w2.ld());
      }
    }

    std::cout << "W2" << std::endl;
    print(T);

    // TODO GEMM X = X - 0.5 . V . W2
    {
      TileType tile_w2 = T(LocalTileIndex{0, 0}).get();

      for (const auto& index_tile_x : iterate_range2d(W.nrTiles())) {
        const LocalTileIndex index_tile_v{Ai_start.row() + index_tile_x.row(), Ai_start.col()};

        std::cout << "UPDATING X" << index_tile_x << " V" << index_tile_v << std::endl;

        hpx::shared_future<ConstTileType> fut_tile_v = A.read(index_tile_v);

        const bool is_diagonal_tile = (Ai_start.row() == index_tile_v.row());
        if (is_diagonal_tile) {
          const ConstTileType& tile_v = fut_tile_v.get();

          MemoryViewType mem_view(dlaf::util::size_t::mul(tile_v.size().rows(), tile_v.size().cols()));
          TileType tile_tmp(tile_v.size(), std::move(mem_view), tile_v.size().rows());

          std::cout << "diagonal: " << std::boolalpha << is_diagonal_tile << std::endl;
          lapack::lacpy(
              lapack::MatrixType::Lower,
              tile_v.size().rows(), tile_v.size().cols(),
              tile_v.ptr(), tile_v.ld(),
              tile_tmp.ptr(), tile_tmp.ld());

          // set upper part to zero and 1 on diagonal (reflectors)
          lapack::laset(lapack::MatrixType::Upper,
              tile_tmp.size().rows(), tile_tmp.size().cols(),
              Type(0), Type(1),
              tile_tmp.ptr(), tile_tmp.ld());

          fut_tile_v = hpx::make_ready_future<ConstTileType>(std::move(tile_tmp));
        }

        TileType tile_x = X(index_tile_x).get();
        const ConstTileType& tile_v = fut_tile_v.get();

        blas::gemm(blas::Layout::ColMajor,
            blas::Op::NoTrans, blas::Op::NoTrans,
            tile_x.size().rows(), tile_x.size().cols(), tile_v.size().cols(),
            Type(-0.5),
            tile_v.ptr(), tile_v.ld(),
            tile_w2.ptr(), tile_w2.ld(),
            Type(1),
            tile_x.ptr(), tile_x.ld());
      }
    }

    std::cout << "X" << std::endl;
    print(X);

    // TODO HER2K At = At - X . V* + V . X*
    std::cout << "At" << At_start << " size:" << At_size << std::endl;
    for (SizeType i = At_start.row(); i < A.nrTiles().cols(); ++i) {
      for (SizeType j = At_start.col(); j <= i; ++j) {
        const LocalTileIndex index_tile_at{i, j};
        TileType tile_at = A(index_tile_at).get();
        std::cout << "HER2K At" << index_tile_at << std::endl;

        const bool is_diagonal_tile = (index_tile_at.row() == index_tile_at.col());

        if (is_diagonal_tile) {
          const LocalTileIndex index_tile_v{index_tile_at.row(), Ai_start.col()};
          const LocalTileIndex index_tile_x{index_tile_at.row() - At_start.row(), 0};
          hpx::shared_future<ConstTileType> fut_tile_v = A.read(index_tile_v);

          const bool is_first_reflector_tile = (Ai_start.row() == index_tile_v.row());
          if (is_first_reflector_tile) {
            const ConstTileType& tile_v = fut_tile_v.get();

            MemoryViewType mem_view(dlaf::util::size_t::mul(tile_v.size().rows(), tile_v.size().cols()));
            TileType tile_tmp(tile_v.size(), std::move(mem_view), tile_v.size().rows());

            std::cout << "first component: " << std::boolalpha << is_first_reflector_tile << std::endl;
            lapack::lacpy(
                lapack::MatrixType::Lower,
                tile_v.size().rows(), tile_v.size().cols(),
                tile_v.ptr(), tile_v.ld(),
                tile_tmp.ptr(), tile_tmp.ld());

            // set upper part to zero and 1 on diagonal (reflectors)
            lapack::laset(lapack::MatrixType::Upper,
                tile_tmp.size().rows(), tile_tmp.size().cols(),
                Type(0), Type(1),
                tile_tmp.ptr(), tile_tmp.ld());

            fut_tile_v = hpx::make_ready_future<ConstTileType>(std::move(tile_tmp));
          }

          const ConstTileType& tile_v = fut_tile_v.get();
          const ConstTileType& tile_x = X.read(index_tile_x).get();

          std::cout << "her2k" << std::endl;
          blas::her2k(
              blas::Layout::ColMajor, blas::Uplo::Lower, blas::Op::NoTrans,
              tile_at.size().rows(), tile_v.size().cols(),
              Type(-1),
              tile_v.ptr(), tile_v.ld(),
              tile_x.ptr(), tile_x.ld(),
              Type(1),
              tile_at.ptr(), tile_at.ld());
        }
        else {
          std::cout << "double gemm" << std::endl;

          // GEMM A: X . V*
          {
            const LocalTileIndex index_tile_x{index_tile_at.row() - At_start.row(), 0};
            const LocalTileIndex index_tile_v{index_tile_at.col(), Ai_start.col()};
            hpx::shared_future<ConstTileType> fut_tile_v = A.read(index_tile_v);

            const bool is_first_reflector_tile = (Ai_start.row() == index_tile_v.row());
            if (is_first_reflector_tile) {
              const ConstTileType& tile_v = fut_tile_v.get();

              MemoryViewType mem_view(dlaf::util::size_t::mul(tile_v.size().rows(), tile_v.size().cols()));
              TileType tile_tmp(tile_v.size(), std::move(mem_view), tile_v.size().rows());

              std::cout << "first component: " << std::boolalpha << is_first_reflector_tile << std::endl;
              lapack::lacpy(
                  lapack::MatrixType::Lower,
                  tile_v.size().rows(), tile_v.size().cols(),
                  tile_v.ptr(), tile_v.ld(),
                  tile_tmp.ptr(), tile_tmp.ld());

              // set upper part to zero and 1 on diagonal (reflectors)
              lapack::laset(lapack::MatrixType::Upper,
                  tile_tmp.size().rows(), tile_tmp.size().cols(),
                  Type(0), Type(1),
                  tile_tmp.ptr(), tile_tmp.ld());

              fut_tile_v = hpx::make_ready_future<ConstTileType>(std::move(tile_tmp));
            }

            const ConstTileType& tile_v = fut_tile_v.get();
            const ConstTileType& tile_x = X.read(index_tile_x).get();

            std::cout << "At" << std::endl;
            print_tile(tile_at);
            std::cout << "X" << std::endl;
            print_tile(tile_x);
            std::cout << "V" << std::endl;
            print_tile(tile_v);

            blas::gemm(
                blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::ConjTrans,
                tile_at.size().rows(), tile_at.size().cols(), tile_x.size().rows(),
                Type(-1),
                tile_x.ptr(), tile_x.ld(),
                tile_v.ptr(), tile_v.ld(),
                Type(1),
                tile_at.ptr(), tile_at.ld());
          }

          std::cout << "At(updated-gemm1)" << std::endl;
          print_tile(tile_at);

          {
            const LocalTileIndex index_tile_v{index_tile_at.row(), Ai_start.col()};
            const LocalTileIndex index_tile_x{index_tile_at.col() - At_start.row(), 0};
            hpx::shared_future<ConstTileType> fut_tile_v = A.read(index_tile_v);

            const bool is_first_reflector_tile = (Ai_start.row() == index_tile_v.row());
            if (is_first_reflector_tile) {
              const ConstTileType& tile_v = fut_tile_v.get();

              MemoryViewType mem_view(dlaf::util::size_t::mul(tile_v.size().rows(), tile_v.size().cols()));
              TileType tile_tmp(tile_v.size(), std::move(mem_view), tile_v.size().rows());

              std::cout << "first component: " << std::boolalpha << is_first_reflector_tile << std::endl;
              lapack::lacpy(
                  lapack::MatrixType::Lower,
                  tile_v.size().rows(), tile_v.size().cols(),
                  tile_v.ptr(), tile_v.ld(),
                  tile_tmp.ptr(), tile_tmp.ld());

              // set upper part to zero and 1 on diagonal (reflectors)
              lapack::laset(lapack::MatrixType::Upper,
                  tile_tmp.size().rows(), tile_tmp.size().cols(),
                  Type(0), Type(1),
                  tile_tmp.ptr(), tile_tmp.ld());

              fut_tile_v = hpx::make_ready_future<ConstTileType>(std::move(tile_tmp));
            }

            const ConstTileType& tile_v = fut_tile_v.get();
            const ConstTileType& tile_x = X.read(index_tile_x).get();

            std::cout << "At" << std::endl;
            print_tile(tile_at);
            std::cout << "V" << std::endl;
            print_tile(tile_v);
            std::cout << "X" << std::endl;
            print_tile(tile_x);

            blas::gemm(
                blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::ConjTrans,
                tile_at.size().rows(), tile_at.size().cols(), tile_x.size().rows(),
                Type(-1),
                tile_v.ptr(), tile_v.ld(),
                tile_x.ptr(), tile_x.ld(),
                Type(1),
                tile_at.ptr(), tile_at.ld());
          }
        }

        std::cout << "At(updated)" << std::endl;
        print_tile(tile_at);
      }
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

void print_tile(const ConstTileType& tile) {
  for (SizeType i_loc = 0; i_loc < tile.size().rows(); ++i_loc) {
    for (SizeType j_loc = 0; j_loc < tile.size().cols(); ++j_loc)
      std::cout << tile({i_loc, j_loc}) << ", ";
    std::cout << std::endl;
  }
}
