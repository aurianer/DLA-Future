#include <cmath>
#include <iostream>

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include "dlaf/common/index2d.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/common/range2d.h"
#include "dlaf/common/vector.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/functions_sync.h"
#include "dlaf/communication/init.h"
#include "dlaf/communication/sync/broadcast.h"
#include "dlaf/communication/sync/reduce.h"
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
using dlaf::GlobalElementSize;
using dlaf::GlobalTileSize;
using dlaf::SizeType;
using dlaf::TileElementSize;
using dlaf::TileElementIndex;
using dlaf::comm::Communicator;
using dlaf::comm::CommunicatorGrid;
using dlaf::comm::Index2D;
using dlaf::comm::IndexT_MPI;

using MatrixType = dlaf::Matrix<Type, dlaf::Device::CPU>;
using ConstMatrixType = dlaf::Matrix<const Type, dlaf::Device::CPU>;
using TileType = dlaf::Tile<Type, dlaf::Device::CPU>;
using ConstTileType = dlaf::Tile<const Type, dlaf::Device::CPU>;

using MemoryViewType = dlaf::memory::MemoryView<Type, dlaf::Device::CPU>;

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

struct ReflectorParams {
  Type norm;
  Type x0;
  Type y;
  Type tau;
  Type factor;
};

void print(ConstMatrixType& matrix, std::string prefix = "");
void print_tile(const ConstTileType& tile);

int miniapp(hpx::program_options::variables_map& vm) {
  const SizeType n = vm["matrix-rows"].as<SizeType>();
  const SizeType nb = vm["block-size"].as<SizeType>();

  const SizeType grid_rows = vm["grid-rows"].as<SizeType>();
  const SizeType grid_cols = vm["grid-cols"].as<SizeType>();

  Communicator world(MPI_COMM_WORLD);
  CommunicatorGrid comm_grid(world, grid_rows, grid_cols, dlaf::common::Ordering::ColumnMajor);

  GlobalElementSize matrix_size(n, n);
  TileElementSize block_size(nb, nb);

  MatrixType A(matrix_size, block_size, comm_grid);
  dlaf::matrix::util::set_random_hermitian(A);

  const auto& dist = A.distribution();
  const auto rank = dist.rankIndex();

  print(A, "A");
  trace(A);
  trace(A.distribution().localNrTiles());

  dlaf::common::Pipeline<CommunicatorGrid> serial_comm(std::move(comm_grid));

  using dlaf::common::iterate_range2d;
  using dlaf::common::make_data;
  using hpx::util::unwrapping;

  using namespace dlaf::comm::sync;

  // for each reflector panel
  for (SizeType j_panel = 0; j_panel < (dist.nrTiles().cols() - 1); ++j_panel) {
    MatrixType T(LocalElementSize{nb, nb}, dist.blockSize());
    dlaf::matrix::util::set(T, [](auto&&) { return 0; });

    MatrixType V0(LocalElementSize{nb, nb}, dist.blockSize());  // used just by the owner

    const SizeType rank_panel_col = dist.rankGlobalTile<Coord::Col>(j_panel);

    const dlaf::GlobalTileIndex Ai_start_global{j_panel + 1, j_panel};
    const dlaf::GlobalTileIndex At_start_global = Ai_start_global + GlobalTileSize{0, 1};

    const auto rank_v0 = dist.rankGlobalTile(Ai_start_global);

    const LocalTileIndex Ai_start{
        dist.nextLocalTileFromGlobalTile<Coord::Row>(Ai_start_global.row()),
        dist.nextLocalTileFromGlobalTile<Coord::Col>(Ai_start_global.col()),
    };
    const LocalTileSize Ai_size{dist.localNrTiles().rows() - Ai_start.row(), 1};

    const SizeType Ai_start_row_el_global = dist.globalElementFromGlobalTileAndTileElement<Coord::Row>(Ai_start_global.row(), 0);
    const SizeType Ai_el_size_rows_global = A.size().rows() - Ai_start_row_el_global;

    const LocalTileIndex At_start{
        Ai_start.row(),
        dist.nextLocalTileFromGlobalTile<Coord::Col>(At_start_global.col()),
    };
    const LocalTileSize At_size{Ai_size.rows(), dist.localNrTiles().cols() - At_start.col()};

    if (rank.col() == rank_panel_col) {  // if this rank is part of the reflector panel
      trace(">>> COMPUTING panel");
      trace(">>> Ai", Ai_size, Ai_start);

      // for each column in the panel, compute reflector and update panel
      // if reflector would be just the first 1, skip the last column
      const SizeType last_reflector = (nb - 1) - (Ai_el_size_rows_global == nb ? 1 : 0);
      trace("LIMIT:", last_reflector);
      for (SizeType j_reflector = 0; j_reflector <= last_reflector; ++j_reflector) {
        const TileElementIndex index_el_x0{j_reflector, j_reflector};

        trace(">>> COMPUTING local reflector", index_el_x0, dist.globalElementFromLocalTileAndTileElement<Coord::Col>(Ai_start.col(), index_el_x0.col()));

        // compute norm + identify x0 component
        trace("COMPUTING NORM");

        hpx::future<std::pair<Type, Type>> fut_x0_and_partial_norm =
            hpx::make_ready_future<std::pair<Type, Type>>(Type(0), Type(0));

        for (const LocalTileIndex& index_tile_x : iterate_range2d(Ai_start, Ai_size)) {
          const SizeType index_tile_v_global =
            dist.globalTileFromLocalTile<Coord::Row>(index_tile_x.row());

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

            fut_x0_and_partial_norm = hpx::dataflow(compute_x0_and_partial_norm_func, A(index_tile_x),
                                                    fut_x0_and_partial_norm);
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

        // reduce norm
        auto reduce_norm_func = unwrapping([rank_v0](auto&& x0_and_norm, auto&& comm_wrapper) {
            const Type local_sum = x0_and_norm.second;
            Type norm = x0_and_norm.second;
            reduce(rank_v0.row(), comm_wrapper().colCommunicator(), MPI_SUM, make_data(&local_sum, 1), make_data(&norm, 1));
            x0_and_norm.second = norm;
            return std::move(x0_and_norm);
        });
        fut_x0_and_partial_norm = hpx::dataflow(reduce_norm_func, fut_x0_and_partial_norm, serial_comm());

        hpx::shared_future<ReflectorParams> reflector_params;
        if (rank_v0 == rank) {
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

                trace("COMPUTE REFLECTOR PARAMS");
                trace("|x| = ", params.norm);
                trace("x0  = ", params.x0);
                trace("y   = ", params.y);
                trace("tau = ", params.tau);

                return std::move(params);
              });

          hpx::future<ReflectorParams> rw_reflector_params = hpx::make_ready_future<ReflectorParams>();

          reflector_params =
              hpx::dataflow(compute_parameters_func, fut_x0_and_partial_norm, rw_reflector_params);

          auto bcast_params_func = unwrapping([](const auto& params, auto&& comm_wrapper) {
              Type data[2] = { params.y, params.factor };
              broadcast::send(comm_wrapper().colCommunicator(), make_data(data, 2));
              trace("sending params", data[0], data[1]);
          });

          hpx::dataflow(bcast_params_func, reflector_params, serial_comm());
        }
        else {
          auto bcast_params_func = unwrapping([rank=rank_v0.row()](auto&& comm_wrapper) {
              trace("waiting params");
              Type data[2];
              broadcast::receive_from(rank, comm_wrapper().colCommunicator(), make_data(data, 2));
              ReflectorParams params;
              params.y = data[0];
              params.factor = data[1];
              trace("received params", data[0], data[1]);
              return params;
          });

          reflector_params = hpx::dataflow(bcast_params_func, serial_comm());
        }

        // compute V (reflector components)
        trace("COMPUTING REFLECTOR COMPONENT");

        for (const LocalTileIndex& index_tile_v : iterate_range2d(Ai_start, Ai_size)) {
          const SizeType index_tile_v_global =
            dist.globalTileFromLocalTile<Coord::Row>(index_tile_v.row());

          const bool has_first_component = (index_tile_v_global == Ai_start_global.row());

          auto compute_reflector_components_func = unwrapping(
              [index_el_x0, has_first_component](const ReflectorParams& params, auto&& tile_v) {
                if (has_first_component)
                  tile_v(index_el_x0) = params.y;

                const SizeType first_tile_element = has_first_component ? index_el_x0.row() + 1 : 0;

                if (first_tile_element > tile_v.size().rows() - 1)
                  return;

                Type* v = tile_v.ptr({first_tile_element, index_el_x0.col()});
                blas::scal(tile_v.size().rows() - first_tile_element, params.factor, v, 1);
                trace("compute tile v");
                print_tile(tile_v);
              });

          hpx::dataflow(compute_reflector_components_func, reflector_params, A(index_tile_v));
        }

        // is there a remaining panel?
        if (index_el_x0.col() < nb - 1) {
          // for each tile in the panel, consider just the trailing panel
          // i.e. all rows (height = reflector), just columns to the right of the current reflector

          // compute W
          const LocalElementSize W_size{1, nb};
          MatrixType W(W_size, dist.blockSize());

          for (const LocalTileIndex& index_tile_a : iterate_range2d(Ai_start, Ai_size)) {
            const SizeType index_tile_a_global =
              dist.globalTileFromLocalTile<Coord::Row>(index_tile_a.row());

            const bool has_first_component = (index_tile_a_global == Ai_start_global.row());

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

            hpx::dataflow(compute_W_func, A.read(index_tile_a), W(LocalTileIndex{0, 0}));
          }
          // TODO all-reduce W
          auto reduce_w_func = unwrapping([rank_v0](auto&& tile_w, auto&& comm_wrapper) {
              auto communicator = comm_wrapper();
              reduce(rank_v0.row(), comm_wrapper().colCommunicator(), MPI_SUM, make_data(tile_w), make_data(tile_w));
              if (rank_v0.row() == communicator.rank().row())
                broadcast::send(communicator.colCommunicator(), make_data(tile_w));
              else
                broadcast::receive_from(rank_v0.row(), communicator.colCommunicator(), make_data(tile_w));
          });
          hpx::dataflow(reduce_w_func, W(LocalTileIndex{0, 0}), serial_comm());
          print(W, "W");

          // update trailing panel
          for (const LocalTileIndex& index_tile_a : iterate_range2d(Ai_start, Ai_size)) {
            const SizeType index_tile_a_global =
              dist.globalTileFromLocalTile<Coord::Row>(index_tile_a.row());

            const bool has_first_component = (index_tile_a_global == Ai_start_global.row());

            auto apply_reflector_func =
                unwrapping([index_el_x0, has_first_component](const ReflectorParams& params,
                                                              auto&& tile_w, auto&& tile_a) {
                  const SizeType first_element_in_tile = has_first_component ? index_el_x0.row() : 0;

                  TileElementIndex Pt_start{first_element_in_tile, index_el_x0.col() + 1};
                  TileElementSize Pt_size{tile_a.size().rows() - Pt_start.row(),
                                          tile_a.size().cols() - Pt_start.col()};

                  TileElementIndex V_start{first_element_in_tile, index_el_x0.col()};
                  const TileElementIndex W_start{0, index_el_x0.col() + 1};

                  const Type tau = - 1 / (params.factor * params.y); // TODO FIXME
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
                          A(index_tile_a));
          }
        }

        print(A, "A");

        // TODO compute T-factor component for this reflector

        // T(0:j, j) = T(0:j, 0:j) . -tau(j) . V(j:, 0:j)* . V(j:, j)
        const TileElementSize T_size{index_el_x0.row(), 1};
        const TileElementIndex T_start{0, index_el_x0.col()};
        for (const auto& index_tile_v : iterate_range2d(Ai_start, Ai_size)) {
          trace("* COMPUTING T", index_tile_v);

          const SizeType index_tile_v_global =
            dist.globalTileFromLocalTile<Coord::Row>(index_tile_v.row());

          const bool has_first_component = (index_tile_v_global == Ai_start_global.row());

          // skip the first component, becuase it should be 1, but it is not

          auto gemv_func =
              unwrapping([T_start, T_size, has_first_component,
                          index_el_x0](const ReflectorParams& params, auto&& tile_v, auto&& tile_t) {
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
                    const auto index_el_va = dlaf::common::internal::transposed(index_el_t);
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

          hpx::dataflow(gemv_func, reflector_params, A.read(index_tile_v), T(LocalTileIndex{0, 0}));
        }

        if (!T_size.isEmpty()) {
          auto reduce_t_func = unwrapping([rank_v0, T_start, T_size](auto&& tile_t, auto&& comm_wrapper) {
            auto&& input_t = make_data(tile_t.ptr(T_start), T_size.rows());
            std::vector<Type> out_data(T_size.rows());
            auto&& output_t = make_data(out_data.data(), T_size.rows());
            reduce(rank_v0.row(), comm_wrapper().colCommunicator(), MPI_SUM, input_t, output_t);
            dlaf::common::copy(output_t, input_t);
            trace("reducing", T_start, T_size.rows(), *tile_t.ptr()); // TODO reduce just the current, otherwise reduce all together
          });

          hpx::dataflow(reduce_t_func, T(LocalTileIndex{0, 0}), serial_comm()); // TODO just reducer needs RW
        }

        if (rank_v0 == rank) {
          auto trmv_func = unwrapping([T_start, T_size](auto&& tile_t) {
              trace("trmv", *tile_t.ptr());
              // t = T . t
              // clang-format off
              blas::trmv(blas::Layout::ColMajor,
                  blas::Uplo::Upper, blas::Op::NoTrans, blas::Diag::NonUnit,
                  T_size.rows(),
                  tile_t.ptr(), tile_t.ld(),
                  tile_t.ptr(T_start), 1);
              // clang-format on
              });

          hpx::dataflow(trmv_func, T(LocalTileIndex{0, 0}));
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
        hpx::dataflow(setup_V0_func, A.read(Ai_start), V0(LocalTileIndex{0, 0}));

        print(V0, "V0");
      }
    }

    // broadcast T
    if (rank_v0 == rank) {  // owner
      // TODO Avoid useless communication
      auto send_bcast_f = unwrapping([](auto&& tile_t, auto&& comm_wrapper) {
        broadcast::send(comm_wrapper().fullCommunicator(), tile_t);
      });
      hpx::dataflow(send_bcast_f, T.read(LocalTileIndex{0, 0}), serial_comm());
    }
    else {
      auto recv_bcast_f = unwrapping([rank_root=comm_grid.rankFullCommunicator(rank_v0)](auto&& tile_t, auto&& comm_wrapper) {
        broadcast::receive_from(rank_root, comm_wrapper().fullCommunicator(), tile_t);
      });
      hpx::dataflow(recv_bcast_f, T(LocalTileIndex{0, 0}), serial_comm());
    }

    print(T, "T");

    // communicate V row-wise
    MatrixType V({Ai_size.rows() * nb, Ai_size.cols() * nb}, dist.blockSize());
    dlaf::common::internal::vector<hpx::shared_future<ConstTileType>> V_futures;
    V_futures.reserve(Ai_size.rows());

    auto get_reflector_row = [Ai_start](const LocalTileIndex& index) {
      return index.row() - Ai_start.row();
    };

    if (rank.col() == rank_panel_col) {  // owner
      // TODO Avoid useless communication
      auto send_bcast_f = unwrapping([](auto&& tile_v, auto&& comm_wrapper) {
        broadcast::send(comm_wrapper().rowCommunicator(), tile_v);
      });

      for (const LocalTileIndex& index_tile_v : iterate_range2d(Ai_start, Ai_size)) {
        const SizeType index_tile_v_global =
            dist.globalTileFromLocalTile<Coord::Row>(index_tile_v.row());

        const bool is_first = (index_tile_v_global == Ai_start_global.row());
        hpx::shared_future<ConstTileType> tile_v =
            is_first ? V0.read(LocalTileIndex{0, 0}) : A.read(index_tile_v);
        hpx::dataflow(send_bcast_f, tile_v, serial_comm());

        V_futures.push_back(tile_v);
      }
    }
    else {
      auto recv_bcast_f = unwrapping([rank_panel_col](auto&& tile_v, auto&& comm_wrapper) {
        broadcast::receive_from(rank_panel_col, comm_wrapper().rowCommunicator(), tile_v);
      });

      for (const LocalTileIndex& index_tile_v : iterate_range2d(LocalTileIndex{0, 0}, Ai_size)) {
        hpx::dataflow(recv_bcast_f, V(index_tile_v), serial_comm());

        V_futures.push_back(V.read(index_tile_v));
      }
    }

    if (rank.col() != rank_panel_col)
      print(V, "V");

    // TODO UPDATE TRAILING MATRIX
    trace(">>> UPDATE TRAILING MATRIX");
    trace(">>> At", At_size, At_start);

    MatrixType W({Ai_size.rows() * nb, nb}, dist.blockSize());
    // TODO TRMM W = V . T
    if (rank.col() == rank_panel_col) {
      for (SizeType i_t = At_start.row(); i_t < dist.localNrTiles().rows(); ++i_t) {
        const LocalTileIndex index_tile_v{i_t, Ai_start.col()};
        const LocalTileIndex index_tile_w{i_t - At_start.row(), 0};

        trace("COMPUTING W", index_tile_w, "with V", index_tile_v);

        auto trmm_func = unwrapping([](auto&& tile_v, auto&& tile_t, auto&& tile_w) {
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
              Type(1),
              tile_t.ptr(), tile_t.ld(),
              tile_w.ptr(), tile_w.ld());
          // clang-format on
        });

        hpx::dataflow(trmm_func, V_futures[get_reflector_row(index_tile_v)],
                      T.read(LocalTileIndex{0, 0}), W(index_tile_w));
      }

      // TODO ALL REDUCE (multiple rows)
      // TODO bcast just if needed (multiple columns/rows)
      auto send_bcast_f = unwrapping([](auto&& tile_w, auto&& comm_wrapper) {
        broadcast::send(comm_wrapper().rowCommunicator(), tile_w);
      });

      for (const dlaf::GlobalTileIndex& index_tile_w : iterate_range2d(W.nrTiles()))
        hpx::dataflow(send_bcast_f, W.read(index_tile_w), serial_comm());
    }
    else {
      auto recv_bcast_f = unwrapping([rank_panel_col](auto&& tile_w, auto&& comm_wrapper) {
        broadcast::receive_from(rank_panel_col, comm_wrapper().rowCommunicator(), tile_w);
      });

      for (const dlaf::GlobalTileIndex& index_tile_w : iterate_range2d(W.nrTiles()))
        hpx::dataflow(recv_bcast_f, W(index_tile_w), serial_comm());
    }

    print(W, "W");

    // TODO HEMM X = At . W
    MatrixType X({At_size.rows() * nb, W.size().cols()}, dist.blockSize());
    dlaf::matrix::util::set(X, [](auto&&) { return 0; });

    for (SizeType i_t = At_start.row(); i_t < dist.localNrTiles().rows(); ++i_t) {
      const auto limit = dist.nextLocalTileFromGlobalTile<Coord::Col>(
          dist.globalTileFromLocalTile<Coord::Row>(i_t) + 1);
      for (SizeType j_t = At_start.col(); j_t < limit; ++j_t) {
        const LocalTileIndex index_tile_at{i_t, j_t};

        trace("COMPUTING X", index_tile_at);

        const auto index_tile_at_g = dist.globalTileIndex(index_tile_at);
        const bool is_diagonal_tile = (index_tile_at_g.row() == index_tile_at_g.col());

        if (is_diagonal_tile) {
          // HEMM
          const LocalTileIndex index_tile_x{index_tile_at.row() - At_start.row(), 0};
          const LocalTileIndex index_tile_w = index_tile_x;

          auto hemm_func = unwrapping([](auto&& tile_a, auto&& tile_w, auto&& tile_x) {
            trace("HEMM");
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

    // TODO ALL-REDUCE X
    auto reduce_x_func = unwrapping([rank, rank_panel_col](auto&& tile_x, auto&& comm_wrapper) {
      reduce(rank_panel_col, comm_wrapper().rowCommunicator(), MPI_SUM, make_data(tile_x),
             make_data(tile_x));
      if (rank.col() == rank_panel_col)
        broadcast::send(comm_wrapper().rowCommunicator(), make_data(tile_x));
      else
        broadcast::receive_from(rank_panel_col, comm_wrapper().rowCommunicator(), make_data(tile_x));
    });

    for (const auto& index_tile_x : iterate_range2d(X.nrTiles()))
      hpx::dataflow(reduce_x_func, X(index_tile_x), serial_comm());

    print(X, "X");

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

    // TODO all-reduce, everyone in the panel should have it

    print(T, "W2");

    // TODO GEMM X = X - 0.5 . V . W2
    for (const auto& index_tile_x : iterate_range2d(X.nrTiles())) {
      const LocalTileIndex index_tile_v{Ai_start.row() + index_tile_x.row(), Ai_start.col()};

      trace("UPDATING X", index_tile_x, "V", index_tile_v);

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

      auto tile_v = V_futures[get_reflector_row(index_tile_v)];
      auto tile_w2 = T(LocalTileIndex{0, 0});  // W2 is stored in T

      hpx::dataflow(gemm_func, tile_v, tile_w2, X(index_tile_x));
    }

    print(X, "X");

    // TODO HER2K At = At - X . V* + V . X*
    trace("At", At_start, "size:", At_size);
    for (SizeType i = At_start.row(); i < dist.localNrTiles().rows(); ++i) {
      const auto limit =
          dist.nextLocalTileFromGlobalTile<Coord::Col>(dist.globalTileFromLocalTile<Coord::Row>(i) + 1);
      for (SizeType j = At_start.col(); j < limit; ++j) {
        const LocalTileIndex index_tile_at{i, j};

        trace("HER2K At", index_tile_at);

        const auto index_tile_at_g = dist.globalTileIndex(index_tile_at);
        const bool is_diagonal_tile = (index_tile_at_g.row() == index_tile_at_g.col());

        if (is_diagonal_tile) {
          const LocalTileIndex index_tile_v{index_tile_at.row(), Ai_start.col()};
          const LocalTileIndex index_tile_x{index_tile_at.row() - At_start.row(), 0};

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

          hpx::dataflow(her2k_func, V_futures[get_reflector_row(index_tile_v)], X.read(index_tile_x),
                        A(index_tile_at));
        }
        else {
          trace("double gemm");

          const SizeType index_tile_at_local_row =
              dist.localTileFromGlobalTile<Coord::Row>(index_tile_at_g.col());

          // GEMM A: X . V*
          {
            const LocalTileIndex index_tile_x{index_tile_at.row() - At_start.row(), 0};
            const LocalTileIndex index_tile_v{index_tile_at_local_row, Ai_start.col()};

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

            hpx::dataflow(gemm_a_func, X.read(index_tile_x), V_futures[get_reflector_row(index_tile_v)],
                          A(index_tile_at));
          }

          {
            const LocalTileIndex index_tile_v{index_tile_at.row(), Ai_start.col()};
            const LocalTileIndex index_tile_x{index_tile_at_local_row - At_start.row(), 0};

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

            hpx::dataflow(gemm_b_func, V_futures[get_reflector_row(index_tile_v)], X.read(index_tile_x),
                          A(index_tile_at));
          }
        }
      }
    }
  }

  print(A, "Z");

  return hpx::finalize();
}

int main(int argc, char** argv) {
  dlaf::comm::mpi_init mpi_initter(argc, argv, dlaf::comm::mpi_thread_level::serialized);

  // options
  using namespace hpx::program_options;
  options_description desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

  // clang-format off
  desc_commandline.add_options()
    ("matrix-rows", value<SizeType>()->default_value(4), "Matrix rows")
    ("block-size",  value<SizeType>()->default_value(2), "Block cyclic distribution size")
    ("grid-rows", value<int>()->default_value(1), "Number of row processes in the 2D communicator")
    ("grid-cols", value<int>()->default_value(1), "Number of column processes in the 2D communicator");
  // clang-format on

  auto ret_code = hpx::init(miniapp, desc_commandline, argc, argv);

  std::cout << "finished" << std::endl;

  return ret_code;
}

void print(ConstMatrixType& matrix, std::string prefix) {
  using dlaf::common::iterate_range2d;

  const auto& distribution = matrix.distribution();

  std::ostringstream ss;
  ss << prefix << " = np.zeros((" << distribution.size() << "))" << std::endl;

  for (const auto& index_tile : iterate_range2d(distribution.localNrTiles())) {
    const auto& tile = matrix.read(index_tile).get();

    for (const auto& index_el : iterate_range2d(tile.size())) {
      dlaf::GlobalElementIndex index_g{
          distribution.globalElementFromLocalTileAndTileElement<Coord::Row>(index_tile.row(),
                                                                            index_el.row()),
          distribution.globalElementFromLocalTileAndTileElement<Coord::Col>(index_tile.col(),
                                                                            index_el.col()),
      };
      ss << prefix << "[" << index_g.row() << "," << index_g.col() << "] = " << tile(index_el)
         << std::endl;
    }
  }

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
