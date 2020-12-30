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

#include <hpx/include/parallel_executors.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/include/util.hpp>
#include <hpx/local/future.hpp>

#include <blas.hh>

#include "dlaf/solver/backtransformation/api.h"

#include "dlaf/blas_tile.h"
#include "dlaf/common/index2d.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/common/vector.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/executor.h"
#include "dlaf/communication/functions_sync.h"
#include "dlaf/communication/init.h"
#include "dlaf/lapack_tile.h"
#include "dlaf/matrix/copy.h"
//#include "dlaf/matrix/copy_tile.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/layout_info.h"
#include "dlaf/util_matrix.h"

#include "dlaf/matrix_output.h"

namespace dlaf {
namespace solver {
namespace internal {

  using namespace dlaf::matrix;
  //  using namespace dlaf::tile;

  template <class T>
  void print_mat(dlaf::Matrix<T, dlaf::Device::CPU>& matrix) {
    using dlaf::common::iterate_range2d;

    const auto& distribution = matrix.distribution();

    //std::cout << matrix << std::endl;

    for (const auto& index : iterate_range2d(distribution.localNrTiles())) {
      const auto index_global = distribution.globalTileIndex(index);
      std::cout << index_global << '\t' << matrix.read(index).get()({0, 0})  <<  " rank " << matrix.rankIndex() << std::endl;
    }
    std::cout << "finished" << std::endl;
  }
  
// Implementation based on:
// 1. Part of algorithm 6 "LAPACK Algorithm for the eigenvector back-transformation", page 15, PhD thesis "GPU
// Accelerated Implementations of a Generalized Eigenvalue Solver for Hermitian Matrices with Systematic
// Energy and Time to Solution Analysis" presented by Raffaele Solcà (2016)
// 2. Report "Gep + back-transformation", Alberto Invernizzi (2020)
// 3. Report "Reduction to band + back-transformation", Raffaele Solcà (2020)

template <class T>
struct BackTransformation<Backend::MC, Device::CPU, T> {
  static void call_FC(Matrix<T, Device::CPU>& mat_c, Matrix<const T, Device::CPU>& mat_v,
		      Matrix<T, Device::CPU>& mat_t);
  static void call_FC(comm::CommunicatorGrid grid, Matrix<T, Device::CPU>& mat_c, Matrix<const T, Device::CPU>& mat_v, Matrix<T, Device::CPU>& mat_t);
 };

 
 template <class T>
   void BackTransformation<Backend::MC, Device::CPU, T>::call_FC(Matrix<T, Device::CPU>& mat_c, Matrix<const T, Device::CPU>& mat_v, Matrix<T, Device::CPU>& mat_t)
   {
     constexpr auto Left = blas::Side::Left;
     constexpr auto Right = blas::Side::Right;
     constexpr auto Upper = blas::Uplo::Upper;
     constexpr auto Lower = blas::Uplo::Lower;
     constexpr auto NoTrans = blas::Op::NoTrans;
     constexpr auto ConjTrans = blas::Op::ConjTrans;
     constexpr auto NonUnit = blas::Diag::NonUnit;

     using hpx::threads::executors::pool_executor;
     using hpx::threads::thread_priority_high;
     using hpx::threads::thread_priority_default;
     using hpx::util::unwrapping;
     
     // Set up executor on the default queue with high priority.
     pool_executor executor_hp("default", thread_priority_high);
     // Set up executor on the default queue with default priority.
     pool_executor executor_normal("default", thread_priority_default);

     SizeType m = mat_c.nrTiles().rows();
     SizeType n = mat_c.nrTiles().cols();
     SizeType mb = mat_c.blockSize().rows();
     SizeType nb = mat_c.blockSize().cols();

     TileElementSize size(mb, nb);

     // CHECK!!
     auto dist_w = mat_c.distribution();
     auto layout_w = tileLayout(dist_w.localSize(), size);
     Matrix<T, Device::CPU> mat_w(std::move(dist_w), layout_w);
     copy(mat_v, mat_w);

     auto dist_w2 = mat_c.distribution();
     auto layout_w2 = tileLayout(dist_w2.localSize(), size);
     Matrix<T, Device::CPU> mat_w2(std::move(dist_w2), layout_w2);
     matrix::util::set(mat_w2, [](auto&&){return 0;});

     // n-1 reflectors
     for (SizeType i = 0; i < (m - 1); ++i) {

       for (SizeType k = i; k < m; ++k) {
	 auto ki = LocalTileIndex{k, i};
	 auto ii = LocalTileIndex{i, i};
	 // W = V T
	 hpx::dataflow(executor_hp, hpx::util::unwrapping(tile::trmm<T, Device::CPU>), Right, Upper, ConjTrans,
		       NonUnit, 1.0, mat_t.read(ii), std::move(mat_w(ki)));
       }

       for (SizeType k = i; k < m; ++k) {
	 auto ik = LocalTileIndex{i, k};
	 for (SizeType j = i; j < m; ++j) {
	   auto ji = LocalTileIndex{j, i};
	   auto jk = LocalTileIndex{j, k};
	   // W2 = WH C
	   hpx::dataflow(executor_normal, hpx::util::unwrapping(tile::gemm<T, Device::CPU>), ConjTrans,
			 NoTrans, 1.0, std::move(mat_w(ji)), mat_c.read(jk), 1.0, std::move(mat_w2(ik)));
	 }
       }

       for (SizeType k = i; k < m; ++k) {
	 auto ki = LocalTileIndex{k, i};
	 for (SizeType j = i; j < m; ++j) {
	   auto ij = LocalTileIndex{i, j};
	   auto kj = LocalTileIndex{k, j};
	   // C = C - V W2
	   hpx::dataflow(executor_normal, hpx::util::unwrapping(tile::gemm<T, Device::CPU>), NoTrans,
			 NoTrans, -1.0, mat_v.read(ki), mat_w2.read(ij), 1.0, std::move(mat_c(kj)));
	 }
       }
     }
   }

 
 template <class T>
   void BackTransformation<Backend::MC, Device::CPU, T>::call_FC(comm::CommunicatorGrid grid, Matrix<T, Device::CPU>& mat_c, Matrix<const T, Device::CPU>& mat_v, Matrix<T, Device::CPU>& mat_t)
   {
     constexpr auto Left = blas::Side::Left;
     constexpr auto Right = blas::Side::Right;
     constexpr auto Upper = blas::Uplo::Upper;
     constexpr auto Lower = blas::Uplo::Lower;
     constexpr auto NoTrans = blas::Op::NoTrans;
     constexpr auto ConjTrans = blas::Op::ConjTrans;
     constexpr auto NonUnit = blas::Diag::NonUnit;

     using comm::IndexT_MPI;
     using comm::internal::mpi_pool_exists;
     using common::internal::vector;
     using dlaf::comm::Communicator;
     using dlaf::comm::CommunicatorGrid;
     using dlaf::common::make_data;

     using TileType = typename Matrix<T, Device::CPU>::TileType;     
     using ConstTileType = typename Matrix<const T, Device::CPU>::ConstTileType;     

     using hpx::threads::executors::pool_executor;
     using hpx::threads::thread_priority_high;
     using hpx::threads::thread_priority_default;
     using hpx::util::unwrapping;

     pool_executor executor_hp("default", thread_priority_high);
     pool_executor executor_normal("default", thread_priority_default);
     // Set up MPI
     auto executor_mpi = (mpi_pool_exists()) ? pool_executor("mpi", thread_priority_high) : executor_hp;
     common::Pipeline<comm::CommunicatorGrid> serial_comm(std::move(grid));

     SizeType m = mat_c.nrTiles().rows();
     SizeType n = mat_c.nrTiles().cols();
     SizeType mb = mat_c.blockSize().rows();
     SizeType nb = mat_c.blockSize().cols();

     TileElementSize size(mb, nb);

     auto distrib = mat_c.distribution();
     auto local_rows = distrib.localNrTiles().rows();
     auto local_cols = distrib.localNrTiles().cols();
     

     // CHECK!!
     // Distribution distributionC(szC, blockSizeC, comm_grid.size(), comm_grid.rank(), src_rank_index);
     auto dist_w = mat_v.distribution();
//     auto layout_w = tileLayout(dist_w.localSize(), size);
//     Matrix<T, Device::CPU> mat_w(std::move(dist_w), layout_w);
     Matrix<T, Device::CPU> mat_w(std::move(dist_w));
     copy(mat_v, mat_w);

     auto dist_w2 = mat_c.distribution();
//     auto layout_w2 = tileLayout(dist_w2.localSize(), size);
//     Matrix<T, Device::CPU> mat_w2(std::move(dist_w2), layout_w2);
     Matrix<T, Device::CPU> mat_w2(std::move(dist_w2));
     matrix::util::set(mat_w2, [](auto&&){return 0;});

     auto dist_w2_partial = mat_c.distribution();
//     auto layout_w2 = tileLayout(dist_w2.localSize(), size);
//     Matrix<T, Device::CPU> mat_w2(std::move(dist_w2), layout_w2);
     Matrix<T, Device::CPU> mat_w2_partial(std::move(dist_w2_partial));
     matrix::util::set(mat_w2_partial, [](auto&&){return 0;});


//     const Distribution dist_block(LocalElementSize{nb, nb}, distrib.blockSize());
//     Matrix<T, Device::CPU> mat_w2_local(dist_block);
//     auto dist_w2_local = mat_c.distribution();
////     auto layout_w2 = tileLayout(dist_w2.localSize(), size);
////     Matrix<T, Device::CPU> mat_w2(std::move(dist_w2), layout_w2);
//     Matrix<T, Device::CPU> mat_w2_local(std::move(dist_w2_local));
//     matrix::util::set(mat_w2_local, [](auto&&){return 0;});

     // n-1 reflectors
     for (SizeType i = 0; i < (m - 1); ++i) {
     //for (SizeType i = 0; i < 1; ++i) {

	 Matrix<T, Device::CPU> mat_w2_local({1, nb}, mat_v.blockSize());
	 matrix::util::set(mat_w2_local, [](auto&&){return 0;});
	 //     auto tile_w2 = mat_w2_local(LocalTileIndex{0,0});
	 //     std::cout << tile_w2 << std::endl;
	 //       std::cout << "w2 local " << mat_w2_local << std::endl;

	 const IndexT_MPI rank_i_col = distrib.template rankGlobalTile<Coord::Col>(i); 
	 const IndexT_MPI rank_i_row = distrib.template rankGlobalTile<Coord::Row>(i); 
	 
	 const SizeType local_i_row = distrib.template localTileFromGlobalTile<Coord::Row>(i);
	 const SizeType local_i_col = distrib.template localTileFromGlobalTile<Coord::Col>(i);
	 auto ii = LocalTileIndex{local_i_row, local_i_col};
	 hpx::shared_future<ConstTileType> matt_ii_tile; 
       
	 // Broadcast Tii column-wise
	 if (mat_t.rankIndex().col() == rank_i_col) {
	   if (mat_t.rankIndex().row() == rank_i_row) {
	     matt_ii_tile = mat_t.read(ii);
	     comm::send_tile(executor_mpi, serial_comm, Coord::Col, mat_t.read(ii));
	   }
	   else {
	     matt_ii_tile = comm::recv_tile<T>(executor_mpi, serial_comm, Coord::Col, mat_t.tileSize(GlobalTileIndex(i, i)), rank_i_row);
	   }
	 }

	 // Compute W = V T
	 for (SizeType k_local = distrib.template nextLocalTileFromGlobalTile<Coord::Row>(i); k_local < local_rows; ++k_local) {
	   auto k = distrib.template globalTileFromLocalTile<Coord::Row>(k_local);
	   const IndexT_MPI rank_k_row = distrib.template rankGlobalTile<Coord::Row>(k); 	 
	   auto ki = LocalTileIndex{k_local, local_i_col};

	   if (mat_w.rankIndex().col() == rank_i_col) {
	     if (mat_w.rankIndex().row() == rank_k_row) {
	       // W = V T
	       hpx::dataflow(executor_hp, hpx::util::unwrapping(tile::trmm<T, Device::CPU>), Right, Upper, ConjTrans,
			     NonUnit, 1.0, matt_ii_tile, std::move(mat_w(ki)));
	     }
	   }



	   hpx::shared_future<ConstTileType> wki_tile;
	   if (mat_w.rankIndex().row() == rank_k_row) {
	     if (mat_w.rankIndex().col() == rank_i_col) {
	       
	       //	       if (j == i) {
		 // std::cout << "send " << k << " " << j  << " (" << i << ") " << " l = " << m-i << " k_local " << k_local << " j_local " << j_local << " index " << k-i <<  " rank_i_col " << rank_i_col << " rank_k_row " << rank_k_row << " rank_j_col " << rank_j_col << std::endl;
		 //panel_matwki[k-i] = mat_w.read(kj);
		 // Send Wji
		 comm::send_tile(executor_mpi, serial_comm, Coord::Row, mat_w.read(ki));
		 wki_tile = mat_w.read(ki);

		 //hpx::shared_future<TileType> tile_w2 = mat_w2_local(kj);
		 //hpx::future<TileType> tile_w2 = mat_w2_local(LocalTileIndex{0,0});
		 //
		 //// Compute Wji Ckj = W2kj, local on W2ki
		 //hpx::dataflow(executor_normal, hpx::util::unwrapping(tile::gemm<T, Device::CPU>), ConjTrans, NoTrans, 1.0, wki_tile, std::move(mat_c.read(kj)), 0.0, std::move(tile_w2));
		 std::cout << "send " << k << " (" << i << ") Wki " << wki_tile.get()({0,0}) <<  std::endl;
	       }
	       else {
		 //std::cout << "receive " << k << " " << j << " (" << i << ") " << " k_local " << k_local << " j_local " << j_local << " index " << k-i <<  " rank_i_col " << rank_i_col << " rank_k_row " << rank_k_row << " rank_j_col " << rank_j_col << " Wki " << wki_tile.get()({0,0}) << std::endl;
		 // Receive Wji
		 wki_tile = comm::recv_tile<T>(executor_mpi, serial_comm, Coord::Row, mat_w.tileSize(GlobalTileIndex(k, i)), rank_i_col);
		   
		 //hpx::future<TileType> tile_w2 = mat_w2_local(LocalTileIndex{0,0});
		 //
		 //// Compute Wji Ckj = W2kj, local on W2ki
		 //hpx::dataflow(executor_normal, hpx::util::unwrapping(tile::gemm<T, Device::CPU>), ConjTrans, NoTrans, 1.0, wki_tile, std::move(mat_c.read(kj)), 1.0, std::move(tile_w2));
		 std::cout << "receive " << k  << " (" << i << ") Wki " << wki_tile.get()({0,0}) << std::endl;
	       }
	     //}
	   }


	   
	   //vector<hpx::shared_future<ConstTileType>> panel_matwki(m-i);
	   // 
	   for (SizeType j_local = distrib.template nextLocalTileFromGlobalTile<Coord::Col>(i); j_local < local_cols; ++j_local) {
	     auto j = distrib.template globalTileFromLocalTile<Coord::Col>(j_local);
	     const IndexT_MPI rank_j_col = distrib.template rankGlobalTile<Coord::Col>(j);

	     //	     std::cout <<  "Loop k " << k << " j " << j  << " vs i " << i << std::endl;
	     auto kj = LocalTileIndex{k_local, j_local};
	     //hpx::shared_future<ConstTileType> wki_tile;
	 

	     hpx::future<TileType> tile_w2 = mat_w2_local(LocalTileIndex{0,0});

	     if (mat_w.rankIndex().row() == rank_k_row) {
	       if (mat_w.rankIndex().col() == rank_j_col) {

		 //if (j == i) {
		   // std::cout << "send " << k << " " << j  << " (" << i << ") " << " l = " << m-i << " k_local " << k_local << " j_local " << j_local << " index " << k-i <<  " rank_i_col " << rank_i_col << " rank_k_row " << rank_k_row << " rank_j_col " << rank_j_col << std::endl;

		   //panel_matwki[k-i] = mat_w.read(kj);
		   // Send Wji
		   //comm::send_tile(executor_mpi, serial_comm, Coord::Row, mat_w.read(kj));
		   //wki_tile = mat_w.read(kj);

		   //hpx::shared_future<TileType> tile_w2 = mat_w2_local(kj);
		   //hpx::future<TileType> tile_w2 = mat_w2_local(LocalTileIndex{0,0});
	 	   //
		   //// Compute Wji Ckj = W2kj, local on W2ki
		   //hpx::dataflow(executor_normal, hpx::util::unwrapping(tile::gemm<T, Device::CPU>), ConjTrans, NoTrans, 1.0, wki_tile, std::move(mat_c.read(kj)), 0.0, std::move(tile_w2));
		   //std::cout << "send " << k << " " << j << " (" << i << ") Wki " << wki_tile.get()({0,0}) << " Ckj " << mat_c.read(kj).get()({0,0}) << " W2 " << mat_w2(LocalTileIndex{0,0}).get()({0,0}) << " comm " << grid.size() <<  std::endl;
		   // }
		 //else {
		   //std::cout << "receive " << k << " " << j << " (" << i << ") " << " k_local " << k_local << " j_local " << j_local << " index " << k-i <<  " rank_i_col " << rank_i_col << " rank_k_row " << rank_k_row << " rank_j_col " << rank_j_col << " Wki " << wki_tile.get()({0,0}) << std::endl;
		   // Receive Wji
		   //		   wki_tile = comm::recv_tile<T>(executor_mpi, serial_comm, Coord::Row, mat_w.tileSize(GlobalTileIndex(k, j)), rank_i_col);
		   
		   //hpx::future<TileType> tile_w2 = mat_w2_local(LocalTileIndex{0,0});
		   //
		   //// Compute Wji Ckj = W2kj, local on W2ki
		   hpx::dataflow(executor_normal, hpx::util::unwrapping(tile::gemm<T, Device::CPU>), ConjTrans, NoTrans, 1.0, wki_tile, std::move(mat_c.read(kj)), 1.0, std::move(tile_w2));
		   std::cout << "receive " << k << " " << j << " (" << i << ") Wki " << wki_tile.get()({0,0}) << " Ckj " << mat_c.read(kj).get()({0,0}) << " W2 " << mat_w2(LocalTileIndex{0,0}).get()({0,0}) << " rank i col " << rank_i_col << std::endl;
		   // }
	       }
	     }
	   } // end loop on j_local (cols)

	   //print_mat(mat_w2_local);
        
//	   hpx::future<TileType> tile_w2 = mat_w2_local(LocalTileIndex{0,0});
//	   auto all_reduce_w_func = hpx::util::unwrapping([](auto&& tile_w2, auto&& comm_wrapper) {
//	       dlaf::comm::sync::all_reduce(comm_wrapper.ref().colCommunicator(), MPI_SUM, make_data(tile_w2), make_data(tile_w2));
//	     
//	     });
//	   hpx::dataflow(all_reduce_w_func, tile_w2, serial_comm());
	 
	 } // end loop on k_local (rows)


	 // NEED TO CHECK THIS SOMEHOW!!!
	 //       std::cout << mat_w2_local(LocalTileIndex{0,0}).get()({0,0}) << std::endl;


//	 // Broadcast Vki row-wise
//	 for (SizeType k_local = 0; k_local < local_rows; ++k_local) { 
//	   auto k = distrib.template globalTileFromLocalTile<Coord::Row>(k_local);
//	   const IndexT_MPI rank_k_row = distrib.template rankGlobalTile<Coord::Row>(k);
//
//	 
//	   vector<hpx::shared_future<ConstTileType>> panel_matvki(m-i);
//
//	   for (SizeType j_local = 0; j_local < local_cols; ++j_local) {
//	     auto j = distrib.template globalTileFromLocalTile<Coord::Col>(j_local);
//	     const IndexT_MPI rank_j_col = distrib.template rankGlobalTile<Coord::Col>(j);
//
//	     auto kj = LocalTileIndex{k_local, j_local};
//		 
//	     if (mat_v.rankIndex().col() == rank_j_col) {
//	       if (mat_v.rankIndex().row() == rank_k_row) {
//
//		 if (i == j) {
//		   panel_matvki[j] = mat_v.read(kj);
//		   // Send Wji
//		   comm::send_tile(executor_mpi, serial_comm, Coord::Row, panel_matvki[j]);
//		   //std::cout << "send " << k << " " << j << " Vkj " << panel_matvki[j].get()({0,0}) << std::endl;
//		   //hpx::future<TileType> tile_w2 = mat_w2_local(LocalTileIndex{0,0});
//		   // C = C - V W2
//		   //hpx::dataflow(executor_normal, hpx::util::unwrapping(tile::gemm<T, Device::CPU>), NoTrans, NoTrans, -1.0, panel_matvki[j], std::move(mat_v.read(kj)), 1.0, std::move(mat_c(kj)));		 
//		   std::cout << "send " << k << " " << j << " Vkj " << panel_matvki[j].get()({0,0}) << " Ckj " << mat_c.read(kj).get()({0,0}) << std::endl;
//		 }
//		 else {
//		   std::cout << "receive " << k << " " << j << std::endl;
//		   // Receive Wji
//		   panel_matvki[j] = comm::recv_tile<T>(executor_mpi, serial_comm, Coord::Row, mat_v.tileSize(GlobalTileIndex(k, j)), rank_i_col);
//
//		   //hpx::future<TileType> tile_w2 = mat_w2_local(LocalTileIndex{0,0});
//		   // C = C - V W2
//		   // std::move(tile_w2)
//		   //		   hpx::dataflow(executor_normal, hpx::util::unwrapping(tile::gemm<T, Device::CPU>), NoTrans, NoTrans, -1.0, panel_matvki[j], std::move(mat_v.read(kj)), 1.0, std::move(mat_c(kj)));		 
//		   std::cout << "receive " << k << " " << j << " Vkj " << panel_matvki[j].get()({0,0}) << " Ckj" << mat_c.read(kj).get()({0,0}) << std::endl;
//		 }
//	       }
//	     }
//	   } // end loop on j_local (cols)
//
//	 } // end loop ono k_local (rows)



	 //Last step
       
	 //       for (SizeType k = i; k < m; ++k) {
	 //	 auto ki = LocalTileIndex{k, i};
	 //	 for (SizeType j = i; j < m; ++j) {
	 //	   auto ij = LocalTileIndex{i, j};
	 //	   auto kj = LocalTileIndex{k, j};
	 //	   // C = C - V W2
	 //	   hpx::dataflow(executor_normal, hpx::util::unwrapping(tile::gemm<T, Device::CPU>), NoTrans,
	 //			 NoTrans, -1.0, mat_v.read(ki), mat_w2.read(ij), 1.0, std::move(mat_c(kj)));
	 //	 }
	 //       }
       }
   }

 
 /// ---- ETI
#define DLAF_SOLVER_BACKTRANSFORMATION_MC_ETI(KWORD, DATATYPE)		\
 KWORD template struct BackTransformation<Backend::MC, Device::CPU, DATATYPE>;

 DLAF_SOLVER_BACKTRANSFORMATION_MC_ETI(extern, float)
 DLAF_SOLVER_BACKTRANSFORMATION_MC_ETI(extern, double)
 DLAF_SOLVER_BACKTRANSFORMATION_MC_ETI(extern, std::complex<float>)
 DLAF_SOLVER_BACKTRANSFORMATION_MC_ETI(extern, std::complex<double>)

}
}
}



//hpx::dataflow(executor_normal, hpx::util::unwrapping(tile::gemm<T, Device::CPU>), ConjTrans, NoTrans, 1.0, panel_matwki[j], std::move(mat_c.read(kj)), 0.0, std::move(mat_w2_local(kj)));

//		 auto tmpw = panel_matwki[j];
//		 auto tmpc = mat_c(kj); 
//		 auto tmpt = tile_w2.get();
//		 std::cout << "Mao: send kj " << k << " " << j <<  " panel W " << tmpw.get()({0,0}) << " - Ckj " << tmpc.get()({0,0}) << " - W2 " << tmpt({0,0}) << std::endl;
		 


 
//	 hpx::future<TileType> tile_w2 = mat_w2(ki);
//	 vector<hpx::future<ConstTileType>> panel_colj(m-i);

//	 auto dist_w2_local = mat_v.distribution();
//	 auto layout_w2_local = tileLayout(dist_w2.localSize(), size);
//	 Matrix<T, Device::CPU> mat_w2_local(std::move(dist_w2_local), layout_w2_local);
////	 //	 Matrix<T, Device::CPU> mat_w2_local(std::move(dist_w2_local));
////	 matrix::util::set(mat_w2_local, [](auto&&){return 0;});
////	 hpx::future<TileType> tile_w2 = mat_w2(LocalTileIndex{0,0});

//	 for (SizeType j = i; j < m; ++j) {

       //	 auto reduce_w_func = hpx::util::unwrapping([rank_i_row](auto&& tile_w2, auto&& comm_wrapper) {
//	     dlaf::comm::sync::reduce(rank_i_row, comm_wrapper.ref().colCommunicator(), MPI_SUM, make_data(tile_w2), make_data(tile_w2));
//	     
//	 });

//       hpx::future<TileType> tile_w2 = mat_w2(LocalTileIndex{0,0});
//       hpx::dataflow(all_reduce_w_func, tile_w2, serial_comm());
       
       //       auto reduce_w_func = hpx::util::unwrapping([rank_i_row](auto&& tile_w2, auto&& comm_wrapper) {
//	   dlaf::comm::sync::reduce(rank_i_row, comm_wrapper.ref().colCommunicator(), MPI_SUM, make_data(tile_w2), make_data(tile_w2));
//	     
//	 });
//	 
//	 for (SizeType j = i; j < m; ++j) {
//	   const IndexT_MPI rank_j_col = distrib.template rankGlobalTile<Coord::Col>(j);
//	   const SizeType local_j_col = distrib.template localTileFromGlobalTile<Coord::Col>(j);
//	   auto ki = LocalTileIndex{local_k_row, local_i_col};
//	   hpx::future<TileType> tile_w2 = mat_w2(ki);
//
//	   if (mat_w.rankIndex().row() == rank_k_row) {
//	     if (rank_j_col == rank_i_col) {
//	       using dlaf::common::make_data;
//	       
//	       hpx::dataflow(executor_normal, hpx::util::unwrapping(dlaf::comm::sync::reduce<T, Device::CPU>), rank_i_col, grid.rowCommunicator(), MPI_SUM, make_data(tile_w2), make_data(tile_w2));
//	     }
//	   }
//	 }
