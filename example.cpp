#include <cmath>
#include <iostream>

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include "dlaf/common/index2d.h"
#include "dlaf/common/range2d.h"
#include "dlaf/lapack_tile.h"
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

using MemoryViewType = dlaf::memory::MemoryView<Type, dlaf::Device::CPU>;

//#define TRACE_ON

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

struct ReflectorParams {
  Type norm;
  Type x0;
  Type y;
  Type tau;
  Type factor;
};

void print(ConstMatrixType& matrix, std::string prefix = "");
void print_tile(const ConstTileType& tile);

auto setup_v_func = hpx::util::unwrapping([](const ConstTileType& tile_v) -> ConstTileType {
  MemoryViewType mem_view(dlaf::util::size_t::mul(tile_v.size().rows(), tile_v.size().cols()));
  TileType tile_tmp(tile_v.size(), std::move(mem_view), tile_v.size().rows());

  // clang-format off
  lapack::lacpy(lapack::MatrixType::Lower,
      tile_v.size().rows(), tile_v.size().cols(),
      tile_v.ptr(), tile_v.ld(),
      tile_tmp.ptr(), tile_tmp.ld());
  // clang-format on

  // set upper part to zero and 1 on diagonal (reflectors)
  // clang-format off
  lapack::laset(lapack::MatrixType::Upper,
      tile_tmp.size().rows(), tile_tmp.size().cols(),
      Type(0), // off diag
      Type(1), // on  diag
      tile_tmp.ptr(), tile_tmp.ld());
  // clang-format on

  return ConstTileType(std::move(tile_tmp));
});

int miniapp(hpx::program_options::variables_map& vm) {
  const SizeType n = vm["matrix-rows"].as<SizeType>();
  const SizeType nb = vm["block-size"].as<SizeType>();

  LocalElementSize matrix_size(n, n);
  TileElementSize block_size(nb, nb);

  MatrixType A{matrix_size, block_size};
  dlaf::matrix::util::set_random_hermitian(A);

  const auto& distribution = A.distribution();

  print(A, "A = ");
  trace(A);
  trace(A.distribution().localNrTiles());

  using dlaf::common::iterate_range2d;
  using hpx::util::unwrapping;

  // for each panel
  const LocalTileSize A_size = distribution.localNrTiles();
  for (SizeType j_panel = 0; j_panel < A_size.cols() && (j_panel + 1) < A_size.rows(); ++j_panel) {
    const LocalTileIndex Ai_start{j_panel + 1, j_panel};
    const LocalTileSize Ai_size{distribution.nrTiles().rows() - Ai_start.row(), 1};

    const LocalTileIndex At_start{Ai_start.row(), Ai_start.col() + 1};
    const LocalTileSize At_size{Ai_size.rows(), A.nrTiles().cols() - (Ai_start.col() + 1)};

    trace(">>> COMPUTING panel");
    trace(">>> Ai ", Ai_size, Ai_start);

    MatrixType T(LocalElementSize{nb, nb}, distribution.blockSize());
    dlaf::matrix::util::set(T, [](auto&&) { return 0; });

    // for each column in the panel, compute reflector and update panel
    // if reflector would be just the first 1, skip the last column
    const SizeType last_reflector = (nb - 1) - (Ai_size.rows() == 1 ? 1 : 0);
    for (SizeType j_reflector = 0; j_reflector <= last_reflector; ++j_reflector) {
      const TileElementIndex index_el_x0{j_reflector, j_reflector};

      trace(">>> COMPUTING local reflector ", index_el_x0);

      // compute norm + identify x0 component
      trace("COMPUTING NORM");

      hpx::future<std::pair<Type, Type>> fut_x0_and_partial_norm =
          hpx::make_ready_future<std::pair<Type, Type>>();

      for (const LocalTileIndex& index_tile_x : iterate_range2d(Ai_start, Ai_size)) {
        const bool has_first_component = (index_tile_x.row() == Ai_start.row());

        if (has_first_component) {
          auto compute_x0_and_partial_norm_func =
              unwrapping([index_el_x0](auto&& tile_x, std::pair<Type, Type>&& x0_and_norm) {
                x0_and_norm.first = tile_x(index_el_x0);

                const Type* x_ptr = tile_x.ptr(index_el_x0);
                x0_and_norm.second =
                    blas::dot(tile_x.size().rows() - index_el_x0.row(), x_ptr, 1, x_ptr, 1);

                trace("x = ", *x_ptr);
                trace("x0 = ", x0_and_norm.first);

                tile_x(index_el_x0) = 1;  // TODO FIX THIS and remove RW from tile_x

                return std::move(x0_and_norm);
              });

          fut_x0_and_partial_norm =
              hpx::dataflow(compute_x0_and_partial_norm_func, A(index_tile_x), fut_x0_and_partial_norm);
        }
        else {
          auto compute_partial_norm_func =
              unwrapping([index_el_x0](auto&& tile_x, std::pair<Type, Type>&& x0_and_norm) {
                const Type* x_ptr = tile_x.ptr({0, index_el_x0.col()});
                x0_and_norm.second += blas::dot(tile_x.size().rows(), x_ptr, 1, x_ptr, 1);

                trace("x = ", *x_ptr);

                return std::move(x0_and_norm);
              });

          fut_x0_and_partial_norm =
              hpx::dataflow(compute_partial_norm_func, A.read(index_tile_x), fut_x0_and_partial_norm);
        }
      }

      hpx::shared_future<ReflectorParams> reflector_params;
      {
        auto compute_parameters_func =
            unwrapping([](const std::pair<Type, Type>& x0_and_norm, ReflectorParams&& params) {
              params.x0 = x0_and_norm.first;
              params.norm = std::sqrt(x0_and_norm.second);

              // compute first component of the reflector
              params.y = std::signbit(params.x0) ? params.norm : -params.norm;

              // compute tau
              params.tau = (params.y - params.x0) / params.y;

              // compute factor
              params.factor = 1 / (params.x0 - params.y);

              trace("|x| = ", params.norm);
              trace("x0  = ", params.x0);
              trace("y   = ", params.y);
              trace("tau = ", params.tau);

              return std::move(params);
            });

        hpx::future<ReflectorParams> rw_reflector_params = hpx::make_ready_future<ReflectorParams>();

        reflector_params =
            hpx::dataflow(compute_parameters_func, fut_x0_and_partial_norm, rw_reflector_params);
      }

      // compute V (reflector components)
      trace("COMPUTING REFLECTOR COMPONENT");

      dlaf::GlobalElementIndex
          index_reflector{distribution.template globalElementFromLocalTileAndTileElement<
                              Coord::Row>(Ai_start.row(), index_el_x0.row()),
                          distribution.template globalElementFromLocalTileAndTileElement<
                              Coord::Col>(Ai_start.col(), index_el_x0.col())};

      LocalTileIndex V_second_component{distribution.template nextLocalTileFromGlobalElement<Coord::Row>(
                                            index_reflector.row() + 1),
                                        Ai_start.col()};
      LocalTileSize V_size{distribution.nrTiles().rows() - V_second_component.row(), 1};

      for (const LocalTileIndex& index_tile_v : iterate_range2d(V_second_component, V_size)) {
        const bool has_first_component = (index_tile_v.row() == Ai_start.row());
        // if it contains the first component, skip it
        const SizeType first_tile_element = has_first_component ? index_el_x0.row() + 1 : 0;

        auto compute_reflector_components_func =
            unwrapping([index_el_x0, first_tile_element](const ReflectorParams& params, auto&& tile_v) {
              Type* v = tile_v.ptr({first_tile_element, index_el_x0.col()});
              blas::scal(tile_v.size().rows() - first_tile_element, params.factor, v, 1);
            });

        hpx::dataflow(compute_reflector_components_func, reflector_params, A(index_tile_v));
      }

      // is there a remaining panel?
      if (index_el_x0.col() < nb - 1) {
        // compute W
        const LocalElementSize W_size{1, nb};
        MatrixType W(W_size, distribution.blockSize());

        // for each tile in the panel, consider just the trailing panel
        // i.e. all rows (height = reflector), just columns to the right of the current reflector
        for (const LocalTileIndex& index_tile_a : iterate_range2d(Ai_start, Ai_size)) {
          const bool has_first_component = (index_tile_a.row() == Ai_start.row());

          auto compute_W_func =
              unwrapping([has_first_component, index_el_x0](auto&& tile_a, auto&& tile_w) {
                const SizeType first_element_in_tile = has_first_component ? index_el_x0.row() : 0;
                const TileElementSize A_size{tile_a.size().rows() - first_element_in_tile,
                                             tile_a.size().cols() - (index_el_x0.col() + 1)};
                const TileElementIndex A_start{first_element_in_tile, index_el_x0.col() + 1};
                const TileElementIndex V_start{first_element_in_tile, index_el_x0.col()};
                const TileElementIndex W_start{0, index_el_x0.col() + 1};

                // W += 1 . A* . V
                const Type beta = has_first_component ? 0 : 1;
                // clang-format off
                blas::gemv(blas::Layout::ColMajor,
                    blas::Op::ConjTrans,
                    A_size.rows(), A_size.cols(),
                    Type(1),
                    tile_a.ptr(A_start), tile_a.ld(),
                    tile_a.ptr(V_start), 1,
                    beta,
                    tile_w.ptr(W_start), tile_w.ld());
                // clang-format on
              });

          hpx::dataflow(compute_W_func, A.read(index_tile_a), W(LocalTileIndex{0, 0}));
        }
        print(W, "W = ");

        // update trailing panel
        for (const LocalTileIndex& index_tile_a : iterate_range2d(Ai_start, Ai_size)) {
          const bool has_first_component = (index_tile_a.row() == Ai_start.row());

          auto apply_reflector_func = unwrapping([index_el_x0,
                                                  has_first_component](const ReflectorParams& params,
                                                                       auto&& w, auto&& tile_a) {
            const SizeType first_element_in_tile = has_first_component ? index_el_x0.row() : 0;

            const TileElementSize V_size{tile_a.size().rows() - first_element_in_tile, 1};
            const TileElementSize W_size{1, tile_a.size().cols() - (index_el_x0.col() + 1)};
            const TileElementSize A_size{V_size.rows(), W_size.cols()};

            const TileElementIndex A_start{first_element_in_tile, index_el_x0.col() + 1};
            const TileElementIndex V_start{first_element_in_tile, index_el_x0.col()};
            const TileElementIndex W_start{0, index_el_x0.col() + 1};

            // Pt = Pt - tau * v * w*
            const Type alpha = -params.tau;
            // clang-format off
            blas::ger(blas::Layout::ColMajor,
                A_size.rows(), A_size.cols(),
                alpha,
                tile_a.ptr(V_start), 1,
                w.ptr(W_start), w.ld(),
                tile_a.ptr(A_start), tile_a.ld());
            // clang-format on
          });

          hpx::dataflow(apply_reflector_func, reflector_params, W(LocalTileIndex{0, 0}),
                        A(index_tile_a));
        }
      }

      // put in place the previously computed result
      // A(Ai_start).get()(index_el_x0) = y; // TODO fix this

      print(A, "A = ");

      // TODO compute T-factor component for this reflector
      // T(0:j, j) = T(0:j, 0:j) . -tau(j) . V(j:, 0:j)* . V(j:, j)
      const TileElementSize T_size{index_el_x0.row(), 1};
      const TileElementIndex T_start{0, index_el_x0.col()};
      for (const auto& index_tile_v : iterate_range2d(Ai_start, Ai_size)) {
        trace("* COMPUTING T ", index_tile_v);

        const bool has_first_component = (index_tile_v.row() == Ai_start.row());
        // skip the first component, becuase it should be 1, but it is not

        auto gemv_func = unwrapping([T_start, T_size, has_first_component,
                                     index_el_x0](const ReflectorParams& params, auto&& tile_v,
                                                  auto&& tile_t) {
          const Type tau = params.tau;

          const SizeType first_element_in_tile = has_first_component ? index_el_x0.row() + 1 : 0;

          // T(0:j, j) = -tau . V(j:, 0:j)* . V(j:, j)
          // [j x 1] = [(n-j) x j]* . [(n-j) x 1]
          const TileElementSize V_size{tile_v.size().rows() - first_element_in_tile, index_el_x0.col()};
          const TileElementIndex Va_start{first_element_in_tile, 0};
          const TileElementIndex Vb_start{first_element_in_tile, index_el_x0.col()};

          // set tau on the diagonal
          if (has_first_component) {
            trace("t on diagonal ", tau);
            tile_t(index_el_x0) = tau;

            // compute first component with implicit one
            for (const auto& index_el_t : iterate_range2d(T_start, T_size)) {
              const auto index_el_va = dlaf::common::internal::transposed(index_el_t);
              tile_t(index_el_t) = -tau * tile_v(index_el_va);

              trace(tile_t(index_el_t), -tau, tile_v(index_el_va));
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

        hpx::dataflow(gemv_func, reflector_params, A.read(index_tile_v), T(LocalTileIndex{0, 0}));
      }

      // std::cout << "TRMV" << std::endl;
      // for (SizeType i_loc = 0; i_loc < tile_t.size().rows(); ++i_loc)
      //  std::cout << "t[" << i_loc << "] " << tile_t({i_loc, index_el_x0.col()}) << std::endl;

      // std::cout << tile_t(T_start) << " " << tile_t({0, 0}) << std::endl;

      auto trmv_func = unwrapping([T_start, index_el_x0](auto&& tile_t) {
        // t = T . t
        // clang-format off
        blas::trmv(blas::Layout::ColMajor,
            blas::Uplo::Upper, blas::Op::NoTrans, blas::Diag::NonUnit,
            index_el_x0.row(),
            tile_t.ptr(), tile_t.ld(),
            tile_t.ptr(T_start), 1);
        // clang-format on
      });

      hpx::dataflow(trmv_func, T(LocalTileIndex{0, 0}));

      // for (SizeType i_loc = 0; i_loc < tile_t.size().rows(); ++i_loc)
      //  std::cout << "t[" << i_loc << "] " << tile_t({i_loc, index_el_x0.col()}) << std::endl;

      // std::cout << "T(partial) = ";
      // print(T);
    }

    print(T, "T = ");

    // TODO UPDATE TRAILING MATRIX
    trace(">>> UPDATE TRAILING MATRIX");
    trace(">>> At ", At_size, At_start);

    MatrixType W({Ai_size.rows() * nb, nb}, distribution.blockSize());
    // TODO TRMM W = V . T
    for (SizeType i_t = At_start.row(); i_t < distribution.nrTiles().rows(); ++i_t) {
      const LocalTileIndex index_tile_v{i_t, j_panel};
      const LocalTileIndex index_tile_w{i_t - At_start.row(), 0};

      trace("COMPUTING W", index_tile_w, "with V", index_tile_v);

      const bool is_diagonal_tile = index_tile_v.row() == At_start.row();

      auto copy_v_func = unwrapping([is_diagonal_tile](auto&& tile_w, auto&& tile_v) {
        // clang-format off
        lapack::lacpy(is_diagonal_tile ? lapack::MatrixType::Lower : lapack::MatrixType::General,
            tile_v.size().rows(), tile_v.size().cols(),
            tile_v.ptr(), tile_v.ld(),
            tile_w.ptr(), tile_w.ld());
        // clang-format on

        if (is_diagonal_tile) {  // is this the first one? (diagonal)
          trace("setting V on W");
          // set upper part to zero and 1 on diagonal (reflectors)
          // clang-format off
          lapack::laset(lapack::MatrixType::Upper,
              tile_w.size().rows(), tile_w.size().cols(),
              Type(0), // off diag
              Type(1), // on  diag
              tile_w.ptr(), tile_w.ld());
          // clang-format on
        }
      });

      auto trmm_func = unwrapping([](auto&& tile_t, auto&& tile_w) {
        // W = V . T
        // clang-format off
        blas::trmm(blas::Layout::ColMajor,
            blas::Side::Right, blas::Uplo::Upper, blas::Op::NoTrans, blas::Diag::NonUnit,
            tile_w.size().rows(), tile_w.size().cols(),
            Type(1),
            tile_t.ptr(), tile_t.ld(),
            tile_w.ptr(), tile_w.ld());
        // clang-format on
      });

      hpx::dataflow(copy_v_func, W(index_tile_w), A.read(index_tile_v));
      hpx::dataflow(trmm_func, T.read(LocalTileIndex{0, 0}), W(index_tile_w));
    }

    print(W, "W = ");

    // TODO HEMM X = At . W
    MatrixType X({At_size.rows() * nb, W.size().cols()}, distribution.blockSize());
    dlaf::matrix::util::set(X, [](auto&&) { return 0; });

    for (SizeType i_t = At_start.row(); i_t < distribution.nrTiles().rows(); ++i_t) {
      for (SizeType j_t = At_start.col(); j_t <= i_t; ++j_t) {
        const LocalTileIndex index_tile_at{i_t, j_t};

        trace("COMPUTING X", index_tile_at);

        const bool is_diagonal_tile = (i_t == j_t);

        if (is_diagonal_tile) {
          // HEMM
          const LocalTileIndex index_tile_x{index_tile_at.row() - At_start.row(), 0};
          const LocalTileIndex index_tile_w = index_tile_x;

          auto hemm_func = unwrapping([](auto&& tile_a, auto&& tile_w, auto&& tile_x) {
            // clang-format off
            blas::hemm(blas::Layout::ColMajor,
                blas::Side::Left, blas::Uplo::Lower,
                tile_x.size().rows(), tile_a.size().cols(),
                Type(1),
                tile_a.ptr(), tile_a.ld(),
                tile_w.ptr(), tile_w.ld(),
                Type(1),
                tile_x.ptr(), tile_x.ld());
            // clang-format on
          });

          hpx::dataflow(hemm_func, A.read(index_tile_at), W.read(index_tile_w), X(index_tile_x));
        }
        else {
          // A  . W
          {
            const LocalTileIndex index_tile_x{index_tile_at.row() - At_start.row(), 0};
            const LocalTileIndex index_tile_w{index_tile_at.col() - At_start.col(), 0};

            trace("GEMM(1)", index_tile_x, index_tile_at, index_tile_w);

            auto gemm_a_func = unwrapping([](auto&& tile_a, auto&& tile_w, auto&& tile_x) {
              // clang-format off
              blas::gemm(blas::Layout::ColMajor,
                  blas::Op::NoTrans, blas::Op::NoTrans,
                  tile_a.size().rows(), tile_w.size().cols(), tile_a.size().cols(),
                  Type(1),
                  tile_a.ptr(), tile_a.ld(),
                  tile_w.ptr(), tile_w.ld(),
                  Type(1),
                  tile_x.ptr(), tile_x.ld());
              // clang-format on
            });

            hpx::dataflow(gemm_a_func, A.read(index_tile_at), W.read(index_tile_w), X(index_tile_x));
          }

          // A* . W
          {
            const LocalTileIndex index_tile_x{index_tile_at.col() - At_start.col(), 0};
            const LocalTileIndex index_tile_w{index_tile_at.row() - At_start.row(), 0};

            trace("GEMM(2)", index_tile_x, index_tile_at, index_tile_w);

            auto gemm_b_func = unwrapping([](auto&& tile_a, auto&& tile_w, auto&& tile_x) {
              // clang-format off
              blas::gemm(blas::Layout::ColMajor,
                  blas::Op::ConjTrans, blas::Op::NoTrans,
                  tile_a.size().rows(), tile_w.size().cols(), tile_a.size().cols(),
                  Type(1),
                  tile_a.ptr(), tile_a.ld(),
                  tile_w.ptr(), tile_w.ld(),
                  Type(1),
                  tile_x.ptr(), tile_x.ld());
              // clang-format on
            });

            hpx::dataflow(gemm_b_func, A.read(index_tile_at), W.read(index_tile_w), X(index_tile_x));
          }
        }
      }
    }

    print(X, "X = ");

    // TODO GEMM W2 = W* . X
    for (const auto& index_tile : iterate_range2d(W.nrTiles())) {
      const Type beta = (index_tile.row() == 0) ? 0 : 1;
      auto gemm_func = unwrapping([beta](auto&& tile_w, auto&& tile_x, auto&& tile_w2) {
        // clang-format off
        blas::gemm(blas::Layout::ColMajor,
            blas::Op::ConjTrans, blas::Op::NoTrans,
            tile_w2.size().rows(), tile_w2.size().cols(), tile_w.size().cols(),
            Type(1),
            tile_w.ptr(), tile_w.ld(),
            tile_x.ptr(), tile_x.ld(),
            beta,
            tile_w2.ptr(), tile_w2.ld());
        // clang-format on
      });

      // re-use T as W2
      hpx::dataflow(gemm_func, W.read(index_tile), X.read(index_tile), T(LocalTileIndex{0, 0}));
    }

    print(T, "W2 = ");

    // TODO GEMM X = X - 0.5 . V . W2
    for (const auto& index_tile_x : iterate_range2d(W.nrTiles())) {
      const LocalTileIndex index_tile_v{Ai_start.row() + index_tile_x.row(), Ai_start.col()};

      trace("UPDATING X", index_tile_x, "V", index_tile_v);

      hpx::shared_future<ConstTileType> fut_tile_v = A.read(index_tile_v);

      const bool is_diagonal_tile = (Ai_start.row() == index_tile_v.row());
      if (is_diagonal_tile)
        fut_tile_v = fut_tile_v.then(setup_v_func);

      auto gemm_func = unwrapping([](auto&& tile_v, auto&& tile_w2, auto&& tile_x) {
        // clang-format off
        blas::gemm(blas::Layout::ColMajor,
            blas::Op::NoTrans, blas::Op::NoTrans,
            tile_x.size().rows(), tile_x.size().cols(), tile_v.size().cols(),
            Type(-0.5),
            tile_v.ptr(), tile_v.ld(),
            tile_w2.ptr(), tile_w2.ld(),
            Type(1),
            tile_x.ptr(), tile_x.ld());
        // clang-format on
      });

      // W2 is stored in T
      hpx::dataflow(gemm_func, fut_tile_v, T(LocalTileIndex{0, 0}), X(index_tile_x));
    }

    print(X, "X = ");

    // TODO HER2K At = At - X . V* + V . X*
    trace("At", At_start, "size:", At_size);
    for (SizeType i = At_start.row(); i < A.nrTiles().cols(); ++i) {
      for (SizeType j = At_start.col(); j <= i; ++j) {
        const LocalTileIndex index_tile_at{i, j};

        trace("HER2K At", index_tile_at);

        const bool is_diagonal_tile = (index_tile_at.row() == index_tile_at.col());

        if (is_diagonal_tile) {
          const LocalTileIndex index_tile_v{index_tile_at.row(), Ai_start.col()};
          const LocalTileIndex index_tile_x{index_tile_at.row() - At_start.row(), 0};

          hpx::shared_future<ConstTileType> fut_tile_v = A.read(index_tile_v);

          const bool is_first_reflector_tile = (Ai_start.row() == index_tile_v.row());
          if (is_first_reflector_tile)
            fut_tile_v = fut_tile_v.then(setup_v_func);

          auto her2k_func = unwrapping([](auto&& tile_v, auto&& tile_x, auto&& tile_at) {
            // clang-format off
            blas::her2k(blas::Layout::ColMajor,
                blas::Uplo::Lower, blas::Op::NoTrans,
                tile_at.size().rows(), tile_v.size().cols(),
                Type(-1),
                tile_v.ptr(), tile_v.ld(),
                tile_x.ptr(), tile_x.ld(),
                Type(1),
                tile_at.ptr(), tile_at.ld());
            // clang-format on
          });

          hpx::dataflow(her2k_func, fut_tile_v, X.read(index_tile_x), A(index_tile_at));
        }
        else {
          trace("double gemm");

          // GEMM A: X . V*
          {
            const LocalTileIndex index_tile_x{index_tile_at.row() - At_start.row(), 0};
            const LocalTileIndex index_tile_v{index_tile_at.col(), Ai_start.col()};
            hpx::shared_future<ConstTileType> fut_tile_v = A.read(index_tile_v);

            const bool is_first_reflector_tile = (Ai_start.row() == index_tile_v.row());
            if (is_first_reflector_tile)
              fut_tile_v = fut_tile_v.then(setup_v_func);

            auto gemm_a_func = unwrapping([](auto&& tile_x, auto&& tile_v, auto&& tile_at) {
              // clang-format off
              blas::gemm(blas::Layout::ColMajor,
                  blas::Op::NoTrans, blas::Op::ConjTrans,
                  tile_at.size().rows(), tile_at.size().cols(), tile_x.size().rows(),
                  Type(-1),
                  tile_x.ptr(), tile_x.ld(),
                  tile_v.ptr(), tile_v.ld(),
                  Type(1),
                  tile_at.ptr(), tile_at.ld());
              // clang-format on
            });

            hpx::dataflow(gemm_a_func, X.read(index_tile_x), fut_tile_v, A(index_tile_at));
          }

          {
            const LocalTileIndex index_tile_v{index_tile_at.row(), Ai_start.col()};
            const LocalTileIndex index_tile_x{index_tile_at.col() - At_start.row(), 0};

            hpx::shared_future<ConstTileType> fut_tile_v = A.read(index_tile_v);

            const bool is_first_reflector_tile = (Ai_start.row() == index_tile_v.row());
            if (is_first_reflector_tile)
              fut_tile_v = fut_tile_v.then(setup_v_func);

            auto gemm_b_func = unwrapping([](auto&& tile_v, auto&& tile_x, auto&& tile_at) {
              // clang-format off
              blas::gemm(blas::Layout::ColMajor,
                  blas::Op::NoTrans, blas::Op::ConjTrans,
                  tile_at.size().rows(), tile_at.size().cols(), tile_x.size().rows(),
                  Type(-1),
                  tile_v.ptr(), tile_v.ld(),
                  tile_x.ptr(), tile_x.ld(),
                  Type(1),
                  tile_at.ptr(), tile_at.ld());
              // clang-format on
            });

            hpx::dataflow(gemm_b_func, fut_tile_v, X.read(index_tile_x), A(index_tile_at));
          }
        }
      }
    }
  }

  print(A, "Z = ");

  return hpx::finalize();
}

int main(int argc, char** argv) {
  // options
  using namespace hpx::program_options;
  options_description desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

  // clang-format off
  desc_commandline.add_options()
    ("matrix-rows", value<SizeType>()->default_value(4), "Matrix rows")
    ("block-size",  value<SizeType>()->default_value(2), "Block cyclic distribution size");
  // clang-format on

  auto ret_code = hpx::init(miniapp, desc_commandline, argc, argv);

  std::cout << "finished" << std::endl;

  return ret_code;
}

void print(ConstMatrixType& matrix, std::string prefix) {
  using dlaf::common::iterate_range2d;

  std::ostringstream ss;
  ss << prefix << "np.array([";

  const auto& distribution = matrix.distribution();
  auto matrix_size = distribution.size();
  for (const auto& index_g : iterate_range2d(matrix_size)) {
    dlaf::GlobalTileIndex tile_g = distribution.globalTileIndex(index_g);
    TileElementIndex index_e = distribution.tileElementIndex(index_g);

    const auto& tile = matrix.read(tile_g).get();

    ss << tile(index_e) << ", ";
  }

  bool is_vector = matrix_size.rows() == 1 || matrix_size.cols() == 1;
  if (!is_vector)
    matrix_size.transpose();
  ss << "]).reshape" << matrix_size << (is_vector ? "" : ".T") << std::endl;

  trace(ss.str());
}

void print_tile(const ConstTileType& tile) {
  std::ostringstream ss;
  for (SizeType i_loc = 0; i_loc < tile.size().rows(); ++i_loc) {
    for (SizeType j_loc = 0; j_loc < tile.size().cols(); ++j_loc)
      ss << tile({i_loc, j_loc}) << ", ";
    ss << std::endl;
  }
  trace(ss.str());
}
