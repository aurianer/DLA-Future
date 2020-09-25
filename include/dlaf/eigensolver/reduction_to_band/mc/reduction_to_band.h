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

#include <cmath>
#include <string>

#include <hpx/include/util.hpp>
//#include <hpx/local/future.hpp>

#include "dlaf/common/data.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/common/range2d.h"
#include "dlaf/common/vector.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/functions_sync.h"
#include "dlaf/lapack_tile.h"
#include "dlaf/matrix.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/util_matrix.h"

namespace dlaf {
namespace internal {
namespace mc {

namespace {

template <class Type>
using MatrixT = Matrix<Type, Device::CPU>;
template <class Type>
using ConstMatrixT = Matrix<const Type, Device::CPU>;
template <class Type>
using TileT = Tile<Type, Device::CPU>;
template <class Type>
using ConstTileT = Tile<const Type, Device::CPU>;

template <class Type>
using FutureTile = hpx::future<TileT<Type>>;
template <class Type>
using FutureConstTile = hpx::shared_future<ConstTileT<Type>>;

template <class Type>
using FutureConstPanel = common::internal::vector<FutureConstTile<Type>>;

template <class Type>
using MemViewT = memory::MemoryView<Type, Device::CPU>;

template <class T>
void print(ConstMatrixT<T>& matrix, std::string prefix = "");
template <class T>
void print_tile(const ConstTileT<T>& tile);

#define TRACE_ON

#ifdef TRACE_ON
template <class T>
void trace(T&& arg) {
  std::cout << arg << std::endl;
}
template <class T, class... Ts>
void trace(T&& arg, Ts&&... args) {
  std::cout << arg << " ";
  trace(args...);
};
#else
template <class... Ts>
void trace(Ts&&...) {}
#endif

template <class Type>
struct ReflectorParams {
  Type norm;
  Type x0;
  Type y;
  Type tau;
  Type factor;
};

template <class Type>
void set_to_zero(MatrixT<Type>& matrix) {
  dlaf::matrix::util::set(matrix, [](...) { return 0; });
}

template <class T>
void compute_w(MatrixT<T>& w, FutureConstPanel<T> v, ConstMatrixT<T>& t) {
  auto trmm_func = hpx::util::unwrapping(
      [](TileT<T>&& tile_w, const ConstTileT<T>& tile_v, const ConstTileT<T>& tile_t) -> void {
        // clang-format off
        lapack::lacpy(lapack::MatrixType::General,
            tile_v.size().rows(), tile_v.size().cols(),
            tile_v.ptr(), tile_v.ld(),
            tile_w.ptr(), tile_w.ld());
        // clang-format on

        // W = V . T
        // clang-format off
        blas::trmm(blas::Layout::ColMajor,
            blas::Side::Right, blas::Uplo::Upper, blas::Op::NoTrans, blas::Diag::NonUnit,
            tile_w.size().rows(), tile_w.size().cols(),
            static_cast<T>(1),
            tile_t.ptr(), tile_t.ld(),
            tile_w.ptr(), tile_w.ld());
        // clang-format on
      });

  const LocalTileSize Wpanel_size = w.distribution().localNrTiles();

  for (const LocalTileIndex& index_tile_w : iterate_range2d(Wpanel_size)) {
    const SizeType index_v = index_tile_w.row();

    // clang-format off
    FutureTile<T>      tile_w = w(index_tile_w);
    FutureConstTile<T> tile_v = v[index_v];
    FutureConstTile<T> tile_t = t.read(LocalTileIndex{0, 0});
    // clang-format on

    hpx::dataflow(trmm_func, std::move(tile_w), std::move(tile_v), std::move(tile_t));
  }
}

// TODO document that it stores in Xcols just the ones for which he is not on the right row,
// otherwise it directly compute also gemm2 inside Xrows
template <class T>
void compute_x(MatrixT<T>& x, MatrixT<T>& x_tmp, const LocalTileIndex at_start, ConstMatrixT<T>& a,
               ConstMatrixT<T>& w, ConstMatrixT<T>& w_tmp) {
  using hpx::util::unwrapping;

  auto hemm_func = unwrapping([](TileT<T>&& tile_x, const ConstTileT<T>& tile_a, const ConstTileT<T>& tile_w) -> void {
    trace("HEMM");

    trace("tile_a");
    print_tile(tile_a);
    trace("tile_w");
    print_tile(tile_w);
    trace("tile_x");
    print_tile(tile_x);

    // clang-format off
    blas::hemm(blas::Layout::ColMajor,
        blas::Side::Left, blas::Uplo::Lower,
        tile_x.size().rows(), tile_a.size().cols(),
        static_cast<T>(1),
        tile_a.ptr(), tile_a.ld(),
        tile_w.ptr(), tile_w.ld(),
        static_cast<T>(1),
        tile_x.ptr(), tile_x.ld());
    // clang-format on

    trace("*tile_x");
    print_tile(tile_x);
  });

  auto gemm_a_func = unwrapping([](TileT<T>&& tile_x, const ConstTileT<T>& tile_a, const ConstTileT<T>& tile_w) -> void {
    trace("GEMM1");

    trace("tile_a");
    print_tile(tile_a);
    trace("tile_w");
    print_tile(tile_w);
    trace("tile_x");
    print_tile(tile_x);

    // clang-format off
    blas::gemm(blas::Layout::ColMajor,
        blas::Op::NoTrans, blas::Op::NoTrans,
        tile_a.size().rows(), tile_w.size().cols(), tile_a.size().cols(),
        static_cast<T>(1),
        tile_a.ptr(), tile_a.ld(),
        tile_w.ptr(), tile_w.ld(),
        static_cast<T>(1),
        tile_x.ptr(), tile_x.ld());
    // clang-format on

    trace("*tile_x");
    print_tile(tile_x);
  });

  auto gemm_b_func = unwrapping([](TileT<T>&& tile_x, const ConstTileT<T>& tile_a, const ConstTileT<T>& tile_w) -> void {
    trace("GEMM2");

    trace("tile_a");
    print_tile(tile_a);
    trace("tile_w");
    print_tile(tile_w);
    trace("tile_x");
    print_tile(tile_x);

    // clang-format off
    blas::gemm(blas::Layout::ColMajor,
        blas::Op::ConjTrans, blas::Op::NoTrans,
        tile_a.size().rows(), tile_w.size().cols(), tile_a.size().cols(),
        static_cast<T>(1),
        tile_a.ptr(), tile_a.ld(),
        tile_w.ptr(), tile_w.ld(),
        static_cast<T>(1),
        tile_x.ptr(), tile_x.ld());
    // clang-format on

    trace("*tile_x");
    print_tile(tile_x);
  });

  const auto dist = a.distribution();
  const comm::Index2D rank = dist.rankIndex();

  for (SizeType i = at_start.row(); i < dist.localNrTiles().rows(); ++i) {
    const comm::IndexT_MPI limit = dist.template nextLocalTileFromGlobalTile<Coord::Col>(
        dist.template globalTileFromLocalTile<Coord::Row>(i) + 1);
    for (SizeType j = at_start.col(); j < limit; ++j) {
      const LocalTileIndex index_at{i, j};

      const GlobalTileIndex index_a = dist.globalTileIndex(index_at); // TODO possible FIXME? I would add at_start
      const bool is_diagonal_tile = (index_a.row() == index_a.col());

      if (is_diagonal_tile) {
        const LocalTileIndex index_x{index_at.row() - at_start.row(), 0};
        const LocalTileIndex index_w{index_at.row() - at_start.row(), 0};

        // clang-format off
        FutureTile<T>       tile_x = x(index_x);
        FutureConstTile<T>  tile_a = a.read(index_at);
        FutureConstTile<T>  tile_w = w.read(index_w);
        // clang-format on

        hpx::dataflow(hemm_func, std::move(tile_x), std::move(tile_a), std::move(tile_w));
      }
      else {
        // A  . W
        {
          const LocalTileIndex index_x{index_at.row() - at_start.row(), 0};

          // clang-format off
          FutureTile<T>       tile_x = x(index_x);
          FutureConstTile<T>  tile_a = a.read(index_at);
          FutureConstTile<T>  tile_w;
          // clang-format on

          const bool own_w = rank.row() == dist.template rankGlobalTile<Coord::Row>(index_a.col());
          if (own_w) {
            const SizeType index_a_row = dist.template localTileFromGlobalTile<Coord::Row>(index_a.col());
            const LocalTileIndex index_w{index_a_row - at_start.row(), 0};

            tile_w = w.read(index_w);
          }
          else {
            const SizeType index_tmp = dist.template localTileFromGlobalTile<Coord::Col>(index_a.col());
            const LocalTileIndex index_tile_w{0, index_tmp - at_start.col()};
            DLAF_ASSERT(index_tmp == index_at.col(), ""); // TODO if this is true, index_tmp can be replaced

            tile_w = w_tmp.read(index_tile_w);
          }

          hpx::dataflow(gemm_a_func, std::move(tile_x), std::move(tile_a), std::move(tile_w));
        }

        // A* . W
        {
          const LocalTileIndex index_w{index_at.row() - at_start.row(), 0};

          // clang-format off
          FutureConstTile<T>  tile_a = a.read(index_at);
          FutureConstTile<T>  tile_w = w.read(index_w);
          FutureTile<T>       tile_x_h;
          // clang-format on

          // if it will be reduced on itself, just add it now
          const bool own_x = rank.row() == dist.template rankGlobalTile<Coord::Row>(index_a.col());
          if (own_x) {
            const SizeType index_tmp = dist.template localTileFromGlobalTile<Coord::Row>(index_a.col());
            const LocalTileIndex index_x_h{index_tmp - at_start.row(), 0};
            tile_x_h = x(index_x_h);
          }
          else {
            const LocalTileIndex index_x_h{0, index_at.col() - at_start.col()};
            tile_x_h = x_tmp(index_x_h);
          }

          hpx::dataflow(gemm_b_func, std::move(tile_x_h), std::move(tile_a), std::move(tile_w));
        }
      }
    }
  }
}

template <class T>
void compute_w2(MatrixT<T>& w2, ConstMatrixT<T>& w, ConstMatrixT<T>& x,
                common::Pipeline<comm::CommunicatorGrid>& serial_comm) {
  using hpx::util::unwrapping;
  using common::make_data;
  using namespace comm::sync;

  auto gemm_func = unwrapping([](auto&& tile_w2, auto&& tile_w, auto&& tile_x, auto beta) {
    trace("GEMM W2");

    // clang-format off
      blas::gemm(blas::Layout::ColMajor,
          blas::Op::ConjTrans, blas::Op::NoTrans,
          tile_w2.size().rows(), tile_w2.size().cols(), tile_w.size().cols(),
          static_cast<T>(1),
          tile_w.ptr(), tile_w.ld(),
          tile_x.ptr(), tile_x.ld(),
          beta,
          tile_w2.ptr(), tile_w2.ld());
    // clang-format on
  });

  // GEMM W2 = W* . X
  for (const auto& index_tile : iterate_range2d(w.distribution().localNrTiles())) {
    const T beta = (index_tile.row() == 0) ? 0 : 1;

    // clang-format off
    FutureTile<T>       tile_w2 = w2(LocalTileIndex{0, 0});
    FutureConstTile<T>  tile_w  = w.read(index_tile);
    FutureConstTile<T>  tile_x  = x.read(index_tile);
    // clang-format on

    hpx::dataflow(gemm_func, std::move(tile_w2), std::move(tile_w), std::move(tile_x), beta);
  }

  // all-reduce instead of computing it on each node, everyone in the panel should have it
  auto all_reduce_w2 = unwrapping([](auto&& tile_w2, auto&& comm_wrapper) {
    all_reduce(comm_wrapper().colCommunicator(), MPI_SUM, make_data(tile_w2));
  });

  FutureTile<T> tile_w2 = w2(LocalTileIndex{0, 0});

  hpx::dataflow(std::move(all_reduce_w2), std::move(tile_w2), serial_comm());
}

template <class T>
void update_x(MatrixT<T>& x, ConstMatrixT<T>& w2, FutureConstPanel<T> v) {
  using hpx::util::unwrapping;

  // GEMM X = X - 0.5 . V . W2
  auto gemm_func = unwrapping([](TileT<T>&& tile_x, const ConstTileT<T>& tile_v, const ConstTileT<T>& tile_w2) -> void {
    trace("UPDATING X");

    // clang-format off
    blas::gemm(blas::Layout::ColMajor,
        blas::Op::NoTrans, blas::Op::NoTrans,
        tile_x.size().rows(), tile_x.size().cols(), tile_v.size().cols(),
        static_cast<T>(-0.5), // FIXME T must be a signed type
        tile_v.ptr(), tile_v.ld(),
        tile_w2.ptr(), tile_w2.ld(),
        static_cast<T>(1),
        tile_x.ptr(), tile_x.ld());
    // clang-format on
  });

  for (SizeType i = 0; i < v.size(); ++i) {
    const LocalTileIndex index_tile_x{i, 0};

    // clang-format off
    FutureTile<T>       tile_x  = x(index_tile_x);
    FutureConstTile<T>  tile_v  = v[i];
    FutureConstTile<T>  tile_w2 = w2.read(LocalTileIndex{0, 0});
    // clang-format on

    hpx::dataflow(gemm_func, std::move(tile_x), std::move(tile_v), std::move(tile_w2));
  }
}

template <class T>
void update_a(const LocalTileIndex at_start, MatrixT<T>& a, ConstMatrixT<T>& x, ConstMatrixT<T>& x_tmp,
              FutureConstPanel<T> v, FutureConstPanel<T> v_tmp) {
  using hpx::util::unwrapping;

  auto her2k_func = unwrapping(
      [](TileT<T>&& tile_at, const ConstTileT<T>& tile_v, const ConstTileT<T>& tile_x) -> void {
        trace("HER2K");

        std::cout << "tile_v\n";
        print_tile(tile_v);

        std::cout << "tile_x\n";
        print_tile(tile_x);

        std::cout << "tile_at\n";
        print_tile(tile_at);

        // clang-format off
        blas::her2k(blas::Layout::ColMajor,
            blas::Uplo::Lower, blas::Op::NoTrans,
            tile_at.size().rows(), tile_v.size().cols(),
            static_cast<T>(-1), // TODO T must be a signed type
            tile_v.ptr(), tile_v.ld(),
            tile_x.ptr(), tile_x.ld(),
            static_cast<T>(1),
            tile_at.ptr(), tile_at.ld());
        // clang-format on

        std::cout << "tile_at*\n";
        print_tile(tile_at);
      });

  auto gemm_a_func = unwrapping(
      [](TileT<T>&& tile_at, const ConstTileT<T>& tile_x, const ConstTileT<T>& tile_v) -> void {
        trace("DOUBLE GEMM-1");

        std::cout << "tile_x\n";
        print_tile(tile_x);

        std::cout << "tile_v\n";
        print_tile(tile_v);

        std::cout << "tile_at\n";
        print_tile(tile_at);

        // clang-format off
        blas::gemm(blas::Layout::ColMajor,
            blas::Op::NoTrans, blas::Op::ConjTrans,
            tile_at.size().rows(), tile_at.size().cols(), tile_x.size().rows(),
            static_cast<T>(-1), // TODO T must be a signed type
            tile_x.ptr(), tile_x.ld(),
            tile_v.ptr(), tile_v.ld(),
            static_cast<T>(1),
            tile_at.ptr(), tile_at.ld());
        // clang-format on

        std::cout << "tile_at*\n";
        print_tile(tile_at);
      });

  auto gemm_b_func = unwrapping(
      [](TileT<T>&& tile_at, const ConstTileT<T>& tile_v, const ConstTileT<T>& tile_x) -> void {
        trace("DOUBLE GEMM-2");

        std::cout << "tile_v\n";
        print_tile(tile_v);

        std::cout << "tile_x\n";
        print_tile(tile_x);

        std::cout << "tile_at\n";
        print_tile(tile_at);

        // clang-format off
        blas::gemm(blas::Layout::ColMajor,
            blas::Op::NoTrans, blas::Op::ConjTrans,
            tile_at.size().rows(), tile_at.size().cols(), tile_x.size().rows(),
            static_cast<T>(-1), // TODO check this + FIXME T must be a signed type
            tile_v.ptr(), tile_v.ld(),
            tile_x.ptr(), tile_x.ld(),
            static_cast<T>(1),
            tile_at.ptr(), tile_at.ld());
        // clang-format on

        std::cout << "tile_at*\n";
        print_tile(tile_at);
      });

  const auto dist = a.distribution();
  const comm::Index2D rank = dist.rankIndex();

  for (SizeType i = at_start.row(); i < dist.localNrTiles().rows(); ++i) {
    const comm::IndexT_MPI limit = dist.template nextLocalTileFromGlobalTile<Coord::Col>(
        dist.template globalTileFromLocalTile<Coord::Row>(i) + 1);
    for (SizeType j = at_start.col(); j < limit; ++j) {
      const LocalTileIndex index_at{i, j};

      const GlobalTileIndex index_a = dist.globalTileIndex(index_at);  // TODO possible FIXME
      const bool is_diagonal_tile = (index_a.row() == index_a.col());

      if (is_diagonal_tile) {
        // HER2K
        const SizeType index_v{index_at.row() - at_start.row()};
        const LocalTileIndex index_x{index_at.row() - at_start.row(), 0};

        // clang-format off
        FutureTile<T>       tile_a = a(index_at);
        FutureConstTile<T>  tile_v = v[index_v];
        FutureConstTile<T>  tile_x = x.read(index_x);
        // clang-format on

        hpx::dataflow(her2k_func, std::move(tile_a), std::move(tile_v), std::move(tile_x));
      }
      else {
        // GEMM A: X . V*
        {
          const LocalTileIndex index_x{index_at.row() - at_start.row(), 0};

          // clang-format off
          FutureTile<T>       tile_a = a(index_at);
          FutureConstTile<T>  tile_x = x.read(index_x);
          FutureConstTile<T>  tile_v;
          // clang-format on

          const bool own_v = rank.row() == dist.template rankGlobalTile<Coord::Row>(index_a.col());
          if (own_v) {
            const SizeType index_tmp =
                dist.template localTileFromGlobalTile<Coord::Row>(index_a.col()) - at_start.row();
            tile_v = v[index_tmp];
          }
          else {
            tile_v = v_tmp[index_at.col() - at_start.col()];
          }

          hpx::dataflow(gemm_a_func, std::move(tile_a), std::move(tile_x), tile_v);
        }

        // GEMM A: V . X*
        {
          const SizeType index_tile_v{index_at.row() - at_start.row()};

          // clang-format off
          FutureTile<T>       tile_a = a(index_at);
          FutureConstTile<T>  tile_v = v[index_tile_v];
          FutureConstTile<T>  tile_x;
          // clang-format on

          const bool own_x = rank.row() == dist.template rankGlobalTile<Coord::Row>(index_a.col());
          if (own_x) {
            const auto index_tmp =
                dist.template localTileFromGlobalTile<Coord::Row>(index_a.col()) - at_start.row();
            tile_x = x.read(LocalTileIndex{index_tmp, 0});
          }
          else {
            tile_x = x_tmp.read(LocalTileIndex{0, index_at.col() - at_start.col()});
          }

          hpx::dataflow(gemm_b_func, tile_a, tile_v, tile_x);
        }
      }
    }
  }
}
}

// Distributed implementation of reduction to band
template <class Type>
void reduction_to_band(comm::CommunicatorGrid grid, Matrix<Type, Device::CPU>& mat_a) {
  using common::iterate_range2d;
  using common::make_data;
  using hpx::util::unwrapping;

  using namespace comm;
  using namespace comm::sync;

  const auto& dist = mat_a.distribution();
  const comm::Index2D rank = dist.rankIndex();

  const SizeType nb = mat_a.blockSize().rows();

  common::Pipeline<comm::CommunicatorGrid> serial_comm(std::move(grid));

  // for each reflector panel
  for (SizeType j_panel = 0; j_panel < (dist.nrTiles().cols() - 1); ++j_panel) {
    MatrixT<Type> t(LocalElementSize{nb, nb}, dist.blockSize());   // used just by the column
    MatrixT<Type> v0(LocalElementSize{nb, nb}, dist.blockSize());  // used just by the owner

    const GlobalTileIndex Ai_start_global{j_panel + 1, j_panel};
    const GlobalTileIndex At_start_global{Ai_start_global + GlobalTileSize{0, 1}};

    const comm::Index2D rank_v0 = dist.rankGlobalTile(Ai_start_global);

    const LocalTileIndex Ai_start{
        dist.template nextLocalTileFromGlobalTile<Coord::Row>(Ai_start_global.row()),
        dist.template nextLocalTileFromGlobalTile<Coord::Col>(Ai_start_global.col()),
    };
    const LocalTileSize Ai_size{dist.localNrTiles().rows() - Ai_start.row(), 1};

    const LocalTileIndex At_start{
        Ai_start.row(),
        dist.template nextLocalTileFromGlobalTile<Coord::Col>(At_start_global.col()),
    };
    const LocalTileSize At_size{Ai_size.rows(), dist.localNrTiles().cols() - At_start.col()};

    // workspaces used as support for computations with trailing matrix At
    const LocalElementSize workspace_col_localsize{Ai_size.rows() * nb, Ai_size.cols() * nb};
    const LocalElementSize workspace_row_localsize{Ai_size.cols() * nb, At_size.cols() * nb};

    print(mat_a, std::string("A_input") + std::to_string(j_panel));

    const bool is_reflector_rank_col = rank_v0.col() == rank.col();

    // 1. PANEL
    if (is_reflector_rank_col) {
      set_to_zero(t);

      const SizeType Ai_start_row_el_global =
          dist.template globalElementFromGlobalTileAndTileElement<Coord::Row>(Ai_start_global.row(), 0);
      const SizeType Ai_el_size_rows_global = mat_a.size().rows() - Ai_start_row_el_global;

      trace(">>> COMPUTING panel");
      trace(">>> Ai", Ai_size, Ai_start);

      // for each column in the panel, compute reflector and update panel
      // if reflector would be just the first 1, skip the last column
      const SizeType last_reflector = (nb - 1) - (Ai_el_size_rows_global == nb ? 1 : 0);
      for (SizeType j_reflector = 0; j_reflector <= last_reflector; ++j_reflector) {
        const TileElementIndex index_el_x0{j_reflector, j_reflector};

        trace(">>> COMPUTING local reflector", index_el_x0);

        // 1A/1 COMPUTING NORM
        trace("COMPUTING NORM");

        hpx::future<std::pair<Type, Type>> fut_x0_and_partial_norm =
            hpx::make_ready_future<std::pair<Type, Type>>(Type(0), Type(0));

        for (const LocalTileIndex& index_tile_x : iterate_range2d(Ai_start, Ai_size)) {
          const SizeType index_tile_v_global =
              dist.template globalTileFromLocalTile<Coord::Row>(index_tile_x.row());

          const bool has_first_component = (index_tile_v_global == Ai_start_global.row());

          if (has_first_component) {
            auto compute_x0_and_partial_norm_func =
                unwrapping([index_el_x0](auto&& tile_x, std::pair<Type, Type>&& x0_and_norm) {
                  x0_and_norm.first = tile_x(index_el_x0);

                  const Type* x_ptr = tile_x.ptr(index_el_x0);
                  x0_and_norm.second =
                      blas::dot(tile_x.size().rows() - index_el_x0.row(), x_ptr, 1, x_ptr, 1);

                  trace("x = ", *x_ptr);
                  trace("x0 = ", x0_and_norm.first);

                  return std::move(x0_and_norm);
                });

            fut_x0_and_partial_norm = hpx::dataflow(compute_x0_and_partial_norm_func,
                                                    mat_a(index_tile_x), fut_x0_and_partial_norm);
          }
          else {
            auto compute_partial_norm_func =
                unwrapping([index_el_x0](auto&& tile_x, std::pair<Type, Type>&& x0_and_norm) {
                  const Type* x_ptr = tile_x.ptr({0, index_el_x0.col()});
                  x0_and_norm.second += blas::dot(tile_x.size().rows(), x_ptr, 1, x_ptr, 1);

                  trace("x = ", *x_ptr);

                  return std::move(x0_and_norm);
                });

            fut_x0_and_partial_norm = hpx::dataflow(compute_partial_norm_func, mat_a.read(index_tile_x),
                                                    fut_x0_and_partial_norm);
          }
        }

        // reduce norm
        auto reduce_norm_func = unwrapping([rank_v0](auto&& x0_and_norm, auto&& comm_wrapper) {
          const Type local_sum = x0_and_norm.second;
          Type norm = x0_and_norm.second;
          reduce(rank_v0.row(), comm_wrapper().colCommunicator(), MPI_SUM, make_data(&local_sum, 1),
                 make_data(&norm, 1));
          x0_and_norm.second = norm;
          return std::move(x0_and_norm);
        });

        fut_x0_and_partial_norm =
            hpx::dataflow(reduce_norm_func, fut_x0_and_partial_norm, serial_comm());

        // 1A/2 COMPUTE PARAMS
        hpx::shared_future<ReflectorParams<Type>> reflector_params;
        if (rank_v0 == rank) {
          auto compute_parameters_func =
              unwrapping([](const std::pair<Type, Type>& x0_and_norm, ReflectorParams<Type>&& params) {
                params.x0 = x0_and_norm.first;
                params.norm = std::sqrt(x0_and_norm.second);

                // compute first component of the reflector
                params.y = std::signbit(params.x0) ? params.norm : -params.norm;

                // compute tau
                params.tau = (params.y - params.x0) / params.y;

                // compute factor
                params.factor = 1 / (params.x0 - params.y);

                trace("COMPUTE REFLECTOR PARAMS");
                trace("|x| = ", params.norm);
                trace("x0  = ", params.x0);
                trace("y   = ", params.y);
                trace("tau = ", params.tau);

                return std::move(params);
              });

          hpx::future<ReflectorParams<Type>> rw_reflector_params =
              hpx::make_ready_future<ReflectorParams<Type>>();

          reflector_params =
              hpx::dataflow(compute_parameters_func, fut_x0_and_partial_norm, rw_reflector_params);

          auto bcast_params_func = unwrapping([](const auto& params, auto&& comm_wrapper) {
            Type data[2] = {params.y, params.factor};
            broadcast::send(comm_wrapper().colCommunicator(), make_data(data, 2));
            trace("sending params", data[0], data[1]);
          });

          hpx::dataflow(bcast_params_func, reflector_params, serial_comm());
        }
        else {
          auto bcast_params_func = unwrapping([rank = rank_v0.row()](auto&& comm_wrapper) {
            trace("waiting params");
            Type data[2];
            broadcast::receive_from(rank, comm_wrapper().colCommunicator(), make_data(data, 2));
            ReflectorParams<Type> params;
            params.y = data[0];
            params.factor = data[1];
            trace("received params", data[0], data[1]);
            return params;
          });

          reflector_params = hpx::dataflow(bcast_params_func, serial_comm());
        }

        // 1A/3 COMPUTE REFLECTOR COMPONENTs
        trace("COMPUTING REFLECTOR COMPONENT");

        for (const LocalTileIndex& index_tile_v : iterate_range2d(Ai_start, Ai_size)) {
          const SizeType index_tile_v_global =
              dist.template globalTileFromLocalTile<Coord::Row>(index_tile_v.row());

          const bool has_first_component = (index_tile_v_global == Ai_start_global.row());

          auto compute_reflector_components_func = unwrapping(
              [index_el_x0, has_first_component](const ReflectorParams<Type>& params, auto&& tile_v) {
                if (has_first_component)
                  tile_v(index_el_x0) = params.y;

                const SizeType first_tile_element = has_first_component ? index_el_x0.row() + 1 : 0;

                if (first_tile_element > tile_v.size().rows() - 1)
                  return;

                Type* v = tile_v.ptr({first_tile_element, index_el_x0.col()});
                blas::scal(tile_v.size().rows() - first_tile_element, params.factor, v, 1);
              });

          hpx::dataflow(compute_reflector_components_func, reflector_params, mat_a(index_tile_v));
        }

        // 1B UPDATE TRAILING PANEL
        // for each tile in the panel, consider just the trailing panel
        // i.e. all rows (height = reflector), just columns to the right of the current reflector
        if (index_el_x0.col() < nb - 1) {
          // 1B/1 Compute W
          MatrixT<Type> W({1, nb}, dist.blockSize());
          set_to_zero(W);

          for (const LocalTileIndex& index_tile_a : iterate_range2d(Ai_start, Ai_size)) {
            const SizeType index_tile_a_global =
                dist.template globalTileFromLocalTile<Coord::Row>(index_tile_a.row());

            const bool has_first_component = (index_tile_a_global == Ai_start_global.row());

            // GEMV w = Pt* . V
            auto compute_W_func =
                unwrapping([has_first_component, index_el_x0](auto&& tile_a, auto&& tile_w) {
                  const SizeType first_element_in_tile = has_first_component ? index_el_x0.row() : 0;

                  TileElementIndex Pt_start{first_element_in_tile, index_el_x0.col() + 1};
                  TileElementSize Pt_size{tile_a.size().rows() - Pt_start.row(),
                                          tile_a.size().cols() - Pt_start.col()};

                  TileElementIndex V_start{first_element_in_tile, index_el_x0.col()};
                  const TileElementIndex W_start{0, index_el_x0.col() + 1};

                  trace("computing W for trailing panel update");
                  trace("Pt", Pt_start);
                  print_tile(tile_a);
                  trace("V", V_start);
                  print_tile(tile_a);

                  if (has_first_component) {
                    const TileElementSize offset{1, 0};

                    Type fake_v = 1;
                    // clang-format off
                    blas::gemv(blas::Layout::ColMajor,
                        blas::Op::ConjTrans,
                        offset.rows(), Pt_size.cols(),
                        Type(1),
                        tile_a.ptr(Pt_start), tile_a.ld(),
                        &fake_v, 1,
                        0,
                        tile_w.ptr(W_start), tile_w.ld());
                    // clang-format on

                    trace("W");
                    print_tile(tile_w);

                    Pt_start = Pt_start + offset;
                    V_start = V_start + offset;
                    Pt_size = Pt_size - offset;
                  }

                  // W += 1 . A* . V
                  // clang-format off
                  blas::gemv(blas::Layout::ColMajor,
                      blas::Op::ConjTrans,
                      Pt_size.rows(), Pt_size.cols(),
                      Type(1),
                      tile_a.ptr(Pt_start), tile_a.ld(),
                      tile_a.ptr(V_start), 1,
                      1,
                      tile_w.ptr(W_start), tile_w.ld());
                  // clang-format on

                  trace("W");
                  print_tile(tile_w);
                });

            hpx::dataflow(compute_W_func, mat_a.read(index_tile_a), W(LocalTileIndex{0, 0}));
          }

          // all-reduce W
          auto reduce_w_func = unwrapping([rank_v0](auto&& tile_w, auto&& comm_wrapper) {
            all_reduce(comm_wrapper().colCommunicator(), MPI_SUM, make_data(tile_w));
          });

          hpx::dataflow(reduce_w_func, W(LocalTileIndex{0, 0}), serial_comm());

          print(W, std::string("W_red") + std::to_string(j_panel));

          // 1B/2 UPDATE TRAILING PANEL
          for (const LocalTileIndex& index_tile_a : iterate_range2d(Ai_start, Ai_size)) {
            const SizeType global_row_tile_a =
                dist.template globalTileFromLocalTile<Coord::Row>(index_tile_a.row());

            const bool has_first_component = (global_row_tile_a == Ai_start_global.row());

            // GER Pt = Pt - tau . v . w*
            auto apply_reflector_func =
                unwrapping([index_el_x0, has_first_component](const ReflectorParams<Type>& params,
                                                              auto&& tile_w, auto&& tile_a) {
                  const SizeType first_element_in_tile = has_first_component ? index_el_x0.row() : 0;

                  TileElementIndex Pt_start{first_element_in_tile, index_el_x0.col() + 1};
                  TileElementSize Pt_size{tile_a.size().rows() - Pt_start.row(),
                                          tile_a.size().cols() - Pt_start.col()};

                  TileElementIndex V_start{first_element_in_tile, index_el_x0.col()};
                  const TileElementIndex W_start{0, index_el_x0.col() + 1};

                  const Type tau = -1 / (params.factor * params.y);  // TODO FIXME
                  trace("UPDATE TRAILING PANEL, tau =", tau);
                  trace("A");
                  print_tile(tile_a);
                  trace("W");
                  print_tile(tile_w);

                  if (has_first_component) {
                    const TileElementSize offset{1, 0};

                    // Pt = Pt - tau * v[0] * w*
                    // clang-format off
                    Type fake_v = 1;
                    blas::ger(blas::Layout::ColMajor,
                        1, Pt_size.cols(),
                        -tau,
                        &fake_v, 1,
                        tile_w.ptr(W_start), tile_w.ld(),
                        tile_a.ptr(Pt_start), tile_a.ld());
                    // clang-format on

                    Pt_start = Pt_start + offset;
                    V_start = V_start + offset;
                    Pt_size = Pt_size - offset;
                  }

                  // Pt = Pt - tau * v * w*
                  // clang-format off
                  blas::ger(blas::Layout::ColMajor,
                      Pt_size.rows(), Pt_size.cols(),
                      -tau,
                      tile_a.ptr(V_start), 1,
                      tile_w.ptr(W_start), tile_w.ld(),
                      tile_a.ptr(Pt_start), tile_a.ld());
                  // clang-format on

                  trace("Pt");
                  print_tile(tile_a);
                });

            hpx::dataflow(apply_reflector_func, reflector_params, W(LocalTileIndex{0, 0}),
                          mat_a(index_tile_a));
          }
        }

        print(mat_a, std::string("A") + std::to_string(j_panel));

        // 2. CALCULATE T-FACTOR
        // T(0:j, j) = T(0:j, 0:j) . -tau(j) . V(j:, 0:j)* . V(j:, j)

        // 2A First step GEMV
        const TileElementSize T_size{index_el_x0.row(), 1};
        const TileElementIndex T_start{0, index_el_x0.col()};
        for (const auto& index_tile_v : iterate_range2d(Ai_start, Ai_size)) {
          trace("* COMPUTING T", index_tile_v);

          const SizeType index_tile_v_global =
              dist.template globalTileFromLocalTile<Coord::Row>(index_tile_v.row());

          const bool has_first_component = (index_tile_v_global == Ai_start_global.row());

          // GEMV t = V(j:mV; 0:j)* . V(j:mV;j)
          auto gemv_func = unwrapping(
              [T_start, T_size, has_first_component, index_el_x0](const ReflectorParams<Type>& params,
                                                                  auto&& tile_v, auto&& tile_t) {
                const Type tau = params.tau;

                const SizeType first_element_in_tile = has_first_component ? index_el_x0.row() + 1 : 0;

                // T(0:j, j) = -tau . V(j:, 0:j)* . V(j:, j)
                // [j x 1] = [(n-j) x j]* . [(n-j) x 1]
                const TileElementSize V_size{tile_v.size().rows() - first_element_in_tile,
                                             index_el_x0.col()};
                const TileElementIndex Va_start{first_element_in_tile, 0};
                const TileElementIndex Vb_start{first_element_in_tile, index_el_x0.col()};

                // set tau on the diagonal
                if (has_first_component) {
                  trace("t on diagonal", tau);
                  tile_t(index_el_x0) = tau;

                  // compute first component with implicit one
                  for (const auto& index_el_t : iterate_range2d(T_start, T_size)) {
                    const auto index_el_va = common::internal::transposed(index_el_t);
                    tile_t(index_el_t) = -tau * tile_v(index_el_va);

                    trace("tile_t", tile_t(index_el_t), -tau, tile_v(index_el_va));
                  }
                }

                if (Va_start.row() < tile_v.size().rows() && Vb_start.row() < tile_v.size().rows()) {
                  trace("GEMV", Va_start, V_size, Vb_start);
                  for (SizeType i_loc = 0; i_loc < tile_t.size().rows(); ++i_loc)
                    trace("t[", i_loc, "]", tile_t({i_loc, index_el_x0.col()}));

                  // t = -tau . V* . V
                  const Type alpha = -tau;
                  const Type beta = 1;
                  // clang-format off
                  blas::gemv(blas::Layout::ColMajor,
                      blas::Op::ConjTrans,
                      V_size.rows(), V_size.cols(),
                      alpha,
                      tile_v.ptr(Va_start), tile_v.ld(),
                      tile_v.ptr(Vb_start), 1,
                      beta, tile_t.ptr(T_start), 1);
                  // clang-format on

                  for (SizeType i_loc = 0; i_loc < tile_t.size().rows(); ++i_loc)
                    trace("t*[", i_loc, "] ", tile_t({i_loc, index_el_x0.col()}));
                }
              });

          hpx::dataflow(gemv_func, reflector_params, mat_a.read(index_tile_v), t(LocalTileIndex{0, 0}));
        }

        // REDUCE after GEMV
        if (!T_size.isEmpty()) {
          auto reduce_t_func =
              unwrapping([rank_v0, T_start, T_size](auto&& tile_t, auto&& comm_wrapper) {
                auto&& input_t = make_data(tile_t.ptr(T_start), T_size.rows());
                std::vector<Type> out_data(T_size.rows());
                auto&& output_t = make_data(out_data.data(), T_size.rows());
                // TODO reduce just the current, otherwise reduce all together
                reduce(rank_v0.row(), comm_wrapper().colCommunicator(), MPI_SUM, input_t, output_t);
                common::copy(output_t, input_t);
                trace("reducing", T_start, T_size.rows(), *tile_t.ptr());
              });

          // TODO just reducer needs RW
          hpx::dataflow(reduce_t_func, t(LocalTileIndex{0, 0}), serial_comm());
        }

        // 2B Second Step TRMV
        if (rank_v0 == rank) {
          // TRMV t = T . t
          auto trmv_func = unwrapping([T_start, T_size](auto&& tile_t) {
            trace("trmv");

            // clang-format off
            blas::trmv(blas::Layout::ColMajor,
                blas::Uplo::Upper, blas::Op::NoTrans, blas::Diag::NonUnit,
                T_size.rows(),
                tile_t.ptr(), tile_t.ld(),
                tile_t.ptr(T_start), 1);
            // clang-format on
          });

          hpx::dataflow(trmv_func, t(LocalTileIndex{0, 0}));
        }
      }

      // setup V0
      if (rank_v0 == rank) {
        auto setup_V0_func = unwrapping([](auto&& tile_a, auto&& tile_v) {
          // clang-format off
          lapack::lacpy(lapack::MatrixType::Lower,
              tile_v.size().rows(), tile_v.size().cols(),
              tile_a.ptr(), tile_a.ld(),
              tile_v.ptr(), tile_v.ld());
          // clang-format on

          // set upper part to zero and 1 on diagonal (reflectors)
          // clang-format off
          lapack::laset(lapack::MatrixType::Upper,
              tile_v.size().rows(), tile_v.size().cols(),
              Type(0), // off diag
              Type(1), // on  diag
              tile_v.ptr(), tile_v.ld());
          // clang-format on
        });
        hpx::dataflow(setup_V0_func, mat_a.read(Ai_start), v0(LocalTileIndex{0, 0}));

        print(v0, std::string("V0_") + std::to_string(j_panel));
      }
    }

    /*
     * T is owned by the first block of the reflector panel, and it has to
     * broadcast it to the entire column
     */

    // broadcast T
    if (is_reflector_rank_col) {
      if (rank_v0.row() == rank.row()) {
        // TODO Avoid useless communication
        auto send_bcast_f = unwrapping([](auto&& tile_t, auto&& comm_wrapper) {
          broadcast::send(comm_wrapper().colCommunicator(), make_data(tile_t));
        });

        hpx::dataflow(send_bcast_f, t.read(LocalTileIndex{0, 0}), serial_comm());
      }
      else {
        auto recv_bcast_f = unwrapping([rank_v0](auto&& tile_t, auto&& comm_wrapper) {
          broadcast::receive_from(rank_v0.row(), comm_wrapper().colCommunicator(), make_data(tile_t));
        });

        hpx::dataflow(recv_bcast_f, t(LocalTileIndex{0, 0}), serial_comm());
      }

      print(t, std::string("T") + std::to_string(j_panel));
    }

    /*
     * V is computed on the first column, than everyone has to know the ones corresponding to
     * its row and column.
     *
     * So, every rank receives the former from his row (from the rank the on in the v0 column)
     * and then the column can receive the letter from the diagonal one
     */

    // communicate V row-wise
    MatrixT<Type> mat_v(workspace_col_localsize, dist.blockSize());
    FutureConstPanel<Type> v(Ai_size.rows());

    if (is_reflector_rank_col) {
      // TODO Avoid useless communication
      auto send_bcast_f = unwrapping([](auto&& tile_v, auto&& comm_wrapper) {
        trace("Vrows -> sending");
        print_tile(tile_v);
        broadcast::send(comm_wrapper().rowCommunicator(), tile_v);
      });

      for (const LocalTileIndex& index_v_loc : iterate_range2d(Ai_start, Ai_size)) {
        const SizeType index_v = dist.template globalTileFromLocalTile<Coord::Row>(index_v_loc.row());

        const bool is_component0 = (index_v == Ai_start_global.row());

        FutureConstTile<Type> tile_v;
        if (is_component0)
          tile_v = v0.read(LocalTileIndex{0, 0});
        else
          tile_v = mat_a.read(index_v_loc);

        hpx::dataflow(send_bcast_f, tile_v, serial_comm());

        const SizeType i_component = index_v_loc.row() - Ai_start.row();
        v[i_component] = tile_v;
      }
    }
    else {
      auto recv_bcast_f = unwrapping([rank_v0](auto&& tile_v, auto&& comm_wrapper) {
        trace("Vrows -> receiving");
        broadcast::receive_from(rank_v0.col(), comm_wrapper().rowCommunicator(), tile_v);
        print_tile(tile_v);
      });

      for (const LocalTileIndex& index_v_loc : iterate_range2d(LocalTileIndex{0, 0}, Ai_size)) {
        hpx::dataflow(recv_bcast_f, mat_v(index_v_loc), serial_comm());

        const SizeType i_component = index_v_loc.row();
        v[i_component] = mat_v.read(index_v_loc);
      }
    }

    DLAF_ASSERT_HEAVY(Ai_size.rows() == v.size(), Ai_size.rows(), v.size());

    // communicate Vcols col-wise
    MatrixT<Type> mat_v_tmp(workspace_row_localsize, dist.blockSize());
    FutureConstPanel<Type> v_tmp(At_size.cols());

    for (SizeType index_v_loc = 0; index_v_loc < At_size.cols(); ++index_v_loc) {
      const auto index_v =
          dist.template globalTileFromLocalTile<Coord::Col>(index_v_loc + At_start.col());
      const SizeType owner_rank_row = dist.template rankGlobalTile<Coord::Row>(index_v);

      if (owner_rank_row == rank.row()) {
        const SizeType index_tmp =
            dist.template localTileFromGlobalTile<Coord::Row>(index_v) - At_start.row();
        FutureConstTile<Type> tile_v = v[index_tmp];

        auto send_bcast_f = unwrapping([=](auto&& tile_v, auto&& comm_wrapper) {
          trace("Vcols -> sending", index_tmp, index_v_loc, index_v);
          print_tile(tile_v);
          broadcast::send(comm_wrapper().colCommunicator(), tile_v);
        });

        hpx::dataflow(send_bcast_f, tile_v, serial_comm());

        v_tmp[index_v_loc] = tile_v;
      }
      else {
        LocalTileIndex index_tile_v{0, index_v_loc};

        auto recv_bcast_f = unwrapping([=](auto&& tile_v, auto&& comm_wrapper) {
          trace("Vcols -> receiving", index_v_loc, index_v);
          broadcast::receive_from(owner_rank_row, comm_wrapper().colCommunicator(), tile_v);
          print_tile(tile_v);
        });

        hpx::dataflow(recv_bcast_f, mat_v_tmp(index_tile_v), serial_comm());

        v_tmp[index_v_loc] = mat_v_tmp.read(index_tile_v);
      }
    }

    DLAF_ASSERT_HEAVY(At_size.cols() == v_tmp.size(), At_size.cols(), v_tmp.size());

    print(mat_v_tmp, "Vcols");

    // 3 UPDATE TRAILING MATRIX
    trace(">>> UPDATE TRAILING MATRIX", j_panel);
    trace(">>> At", At_size, At_start);

    // 3A COMPUTE W
    MatrixT<Type> w(workspace_col_localsize, dist.blockSize());

    // TRMM W = V . T
    if (is_reflector_rank_col)
      compute_w(w, v, t);

    // W bcast row-wise
    for (const LocalTileIndex index_w_loc : iterate_range2d(w.distribution().localNrTiles())) {
      if (is_reflector_rank_col) {
        auto bcast_all_func = unwrapping([](auto&& tile_w, auto&& comm_wrapper) {
          broadcast::send(comm_wrapper().rowCommunicator(), make_data(tile_w));
        });

        hpx::dataflow(std::move(bcast_all_func), w.read(index_w_loc), serial_comm());
      }
      else {
        auto bcast_all_func = unwrapping([rank_v0](auto&& tile_w, auto&& comm_wrapper) {
          auto comm_grid = comm_wrapper();
          broadcast::receive_from(rank_v0.col(), comm_grid.rowCommunicator(), make_data(tile_w));
        });

        hpx::dataflow(std::move(bcast_all_func), w(index_w_loc), serial_comm());
      }
    }

    // W* bcast col-wise
    MatrixT<Type> w_tmp(workspace_row_localsize, dist.blockSize());
    set_to_zero(w_tmp);  // TODO superflous? if it is not used here, it will not be used later?

    for (const LocalTileIndex index_x_loc : iterate_range2d(w_tmp.distribution().localNrTiles())) {
      const SizeType index_x_row =
          dist.template globalTileFromLocalTile<Coord::Col>(index_x_loc.col() + At_start.col());
      const IndexT_MPI rank_row_owner = dist.template rankGlobalTile<Coord::Row>(index_x_row);

      if (rank_row_owner == rank.row()) {
        auto bcast_send_w_colwise_func = unwrapping([=](auto&& tile_w, auto&& comm_wrapper) {
          broadcast::send(comm_wrapper().colCommunicator(), make_data(tile_w));
        });

        const SizeType index_x_row_loc =
            dist.template localTileFromGlobalTile<Coord::Row>(index_x_row) - At_start.row();

        FutureConstTile<Type> tile_w = w.read(LocalTileIndex{index_x_row_loc, 0});

        hpx::dataflow(std::move(bcast_send_w_colwise_func), std::move(tile_w), serial_comm());
      }
      else {
        auto bcast_recv_w_colwise_func = unwrapping([=](auto&& tile_w, auto&& comm_wrapper) {
          broadcast::receive_from(rank_row_owner, comm_wrapper().colCommunicator(), make_data(tile_w));
        });

        FutureTile<Type> tile_w = w_tmp(index_x_loc);

        hpx::dataflow(std::move(bcast_recv_w_colwise_func), std::move(tile_w), serial_comm());
      }
    }

    print(w, "W");
    print(w_tmp, "W_conj");

    // 3B COMPUTE X
    /*
     * Since At is hermitian, just the lower part is referenced.
     * When the tile is not part of the main diagonal, the same tile has to be used for two computations
     * that will contribute to two different rows of X: the ones indexed with row and col.
     * This is achieved by storing the two results in two different workspaces for X, X and X_conj respectively.
     */
    MatrixT<Type> x({(At_size.rows() != 0 ? At_size.rows() : 1) * nb, nb}, dist.blockSize());
    MatrixT<Type> x_tmp({nb, (At_size.cols() != 0 ? At_size.cols() : 1) * nb}, dist.blockSize());

    // TODO maybe it may be enough doing it if (At_size.isEmpty())
    set_to_zero(x);
    set_to_zero(x_tmp);

    // HEMM X = At . W
    compute_x(x, x_tmp, At_start, mat_a, w, w_tmp);

    print(x, "Xpre-rows");
    print(x_tmp, "Xpre-cols");

    /*
     * The result for X has to be reduced using both Xrows and Xcols.
     */

    // reducing X col-wise
    for (SizeType index_xcol = 0; index_xcol < At_size.cols(); ++index_xcol) {
      const SizeType index_tile_k =
          dist.template globalTileFromLocalTile<Coord::Col>(index_xcol + At_start.col());

      const IndexT_MPI rank_owner_row = dist.template rankGlobalTile<Coord::Row>(index_tile_k);

      if (rank_owner_row == rank.row()) {
        const auto index_x_row =
            dist.template localTileFromGlobalTile<Coord::Row>(index_tile_k) - At_start.row();

        FutureTile<Type> tile_x = x(LocalTileIndex{index_x_row, 0});
        auto reduce_x_func = unwrapping([=](auto&& tile_x, auto&& comm_wrapper) {
          auto comm_grid = comm_wrapper();

          trace("reducing root X col-wise", rank_owner_row, index_tile_k);
          print_tile(tile_x);

          reduce(rank_owner_row, comm_grid.colCommunicator(), MPI_SUM, make_data(tile_x),
                 make_data(tile_x));

          trace("REDUCED COL", index_tile_k);
          print_tile(tile_x);
        });

        hpx::dataflow(reduce_x_func, std::move(tile_x), serial_comm());
      }
      else {
        FutureConstTile<Type> tile_x = x_tmp.read(LocalTileIndex{0, index_xcol});

        auto reduce_x_func = unwrapping([=](auto&& tile_x_conj, auto&& comm_wrapper) {
          auto comm_grid = comm_wrapper();

          trace("reducing send X col-wise", rank_owner_row, index_tile_k, "x", index_xcol);
          print_tile(tile_x_conj);

          Type fake;
          reduce(rank_owner_row, comm_grid.colCommunicator(), MPI_SUM, make_data(tile_x_conj),
                 make_data(&fake, 0));
        });

        hpx::dataflow(reduce_x_func, std::move(tile_x), serial_comm());
      }
    }

    print(x, "Xint");

    /*
     * TODO on rank not in V column, the first data is not referenced/used
     * for the reducers, it happens in-place
     * it can be splitted for read only management of the tiles on senders' side
     */
    for (SizeType index_x = 0; index_x < At_size.rows(); ++index_x) {
      auto reduce_x_func = unwrapping([rank_v0](auto&& tile_x, auto&& comm_wrapper) {
        reduce(rank_v0.col(), comm_wrapper().rowCommunicator(), MPI_SUM, make_data(tile_x),
            make_data(tile_x));
      });

      hpx::dataflow(reduce_x_func, x(LocalTileIndex{index_x, 0}), serial_comm());
    }

    // 3C COMPUTE W2
    /*
     * W2 can be computed by the first column only, which is the only one that has the X result
     */
    if (is_reflector_rank_col) {
      print(x, "Xpre");

      MatrixT<Type> w2 = std::move(t);
      set_to_zero(w2);  // superflous? I don't think so, because if any tile does not participate, it has
                        // to not contribute with any value

      compute_w2(w2, w, x, serial_comm);

      print(w2, "W2");

      // 3D UPDATE X
      update_x(x, w2, v);

      print(x, "X");
    }

    /*
     * Then the X has to be communicated to everyone, since it will be used by all ranks
     * in the last step for updating the trailing matrix.
     *
     * Each cell of the lower part of At, in order to be updated, requires both the X from the
     * row and from the column (because it will use the transposed X).
     *
     * So, as first step each row i will get the X(i) tile, then each column j will get the X(j)
     */

    // broadcast X rowwise
    auto bcast_x_rowwise_func = unwrapping([=](auto&& tile_x, auto&& comm_wrapper) {
      if (is_reflector_rank_col)
        broadcast::send(comm_wrapper().rowCommunicator(), make_data(tile_x));
      else
        broadcast::receive_from(rank_v0.col(), comm_wrapper().rowCommunicator(), make_data(tile_x));
    });

    for (const auto& index_tile_x : iterate_range2d(x.distribution().localNrTiles()))
      hpx::dataflow(bcast_x_rowwise_func, x(index_tile_x), serial_comm());

    // broadcast X colwise
    for (SizeType index_xconj_col = 0; index_xconj_col < At_size.cols(); ++index_xconj_col) {
      const auto index_tile_k =
          dist.template globalTileFromLocalTile<Coord::Col>(index_xconj_col + At_start.col());

      const SizeType rank_owner = dist.template rankGlobalTile<Coord::Row>(index_tile_k);

      if (rank_owner == rank.row()) {
        const SizeType index_x_row =
            dist.template localTileFromGlobalTile<Coord::Row>(index_tile_k) - At_start.row();

        auto bcast_xupd_colwise_func = unwrapping([=](auto&& tile_x, auto&& comm_wrapper) {
          trace("sending tile_x");
          print_tile(tile_x);
          broadcast::send(comm_wrapper().colCommunicator(), make_data(tile_x));
        });

        hpx::dataflow(bcast_xupd_colwise_func, x(LocalTileIndex{index_x_row, 0}), serial_comm());
      }
      else {
        auto bcast_xupd_colwise_func = unwrapping([=](auto&& tile_x, auto&& comm_wrapper) {
          broadcast::receive_from(rank_owner, comm_wrapper().colCommunicator(), make_data(tile_x));
          trace("received tile_x");
          print_tile(tile_x);
        });

        hpx::dataflow(bcast_xupd_colwise_func, x_tmp(LocalTileIndex{0, index_xconj_col}), serial_comm());
      }
    }

    print(x, "Xrows");
    print(x_tmp, "Xcols");

    // 3E UPDATE
    trace("At", At_start, "size:", At_size);

    trace("Vrows");
    for (const auto& tile_v_fut : v)
      print_tile(tile_v_fut.get());

    trace("Vcols");
    for (const auto& tile_vcols_fut : v_tmp)
      print_tile(tile_vcols_fut.get());

    // HER2K At = At - X . V* + V . X*
    update_a(At_start, mat_a, x, x_tmp, v, v_tmp);
  }
}

namespace {

template <class Type>
void print(ConstMatrixT<Type>& matrix, std::string prefix) {
  using common::iterate_range2d;

  const auto& distribution = matrix.distribution();

  std::ostringstream ss;
  ss << prefix << " = np.zeros((" << distribution.size() << "))" << std::endl;

  for (const auto& index_tile : iterate_range2d(distribution.localNrTiles())) {
    const auto& tile = matrix.read(index_tile).get();

    for (const auto& index_el : iterate_range2d(tile.size())) {
      GlobalElementIndex index_g{
          distribution.template globalElementFromLocalTileAndTileElement<Coord::Row>(index_tile.row(),
                                                                                     index_el.row()),
          distribution.template globalElementFromLocalTileAndTileElement<Coord::Col>(index_tile.col(),
                                                                                     index_el.col()),
      };
      ss << prefix << "[" << index_g.row() << "," << index_g.col() << "] = " << tile(index_el)
         << std::endl;
    }
  }

  trace(ss.str());
}

template <class Type>
void print_tile(const ConstTileT<Type>& tile) {
  std::ostringstream ss;
  for (SizeType i_loc = 0; i_loc < tile.size().rows(); ++i_loc) {
    for (SizeType j_loc = 0; j_loc < tile.size().cols(); ++j_loc)
      ss << tile({i_loc, j_loc}) << ", ";
    ss << std::endl;
  }
  trace(ss.str());
}
}

}
}
}
