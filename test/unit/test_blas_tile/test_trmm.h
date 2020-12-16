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

#include <functional>
#include <sstream>
#include <tuple>
#include "gtest/gtest.h"
#include "dlaf/blas_tile.h"
#include "dlaf/memory/memory_view.h"
#include "dlaf/matrix/tile.h"
#include "dlaf/util_blas.h"
#include "dlaf_test/matrix/util_tile.h"
#include "dlaf_test/matrix/util_tile_blas.h"
#include "dlaf_test/util_types.h"

namespace dlaf {
namespace test {
  
using namespace dlaf::matrix;
using namespace dlaf::matrix::test;
using namespace testing;

using dlaf::util::size_t::mul;

template <class ElementIndex, class T, class CT = const T>
void testTrmm(blas::Side side, blas::Uplo uplo, blas::Op op, blas::Diag diag, SizeType m, SizeType n,
              SizeType extra_lda, SizeType extra_ldb) {
  TileElementSize size_a = side == blas::Side::Left ? TileElementSize(m, m) : TileElementSize(n, n);
  TileElementSize size_b(m, n);

  SizeType lda = std::max<SizeType>(1, size_a.rows()) + extra_lda;
  SizeType ldb = std::max<SizeType>(1, size_b.rows()) + extra_ldb;

  std::stringstream s;
  s << "TRMM: " << side << ", " << uplo << ", " << op << ", " << diag << ", m = " << m << ", n = " << n
    << ", lda = " << lda << ", ldb = " << ldb;
  SCOPED_TRACE(s.str());

  memory::MemoryView<T, Device::CPU> mem_a(mul(lda, size_a.cols()));
  memory::MemoryView<T, Device::CPU> mem_b(mul(ldb, size_b.cols()));

  Tile<T, Device::CPU> a0(size_a, std::move(mem_a), lda);
  Tile<T, Device::CPU> b(size_b, std::move(mem_b), ldb);

  T alpha = TypeUtilities<T>::element(-1.2, .7);

  std::function<T(const TileElementIndex&)> el_op_a, el_b, res_b;

  if (side == blas::Side::Left)
    std::tie(el_op_a, el_b, res_b) =
        getLeftTriangularMMSystem<ElementIndex, T>(uplo, op, diag, alpha, m);
  else
    std::tie(el_op_a, el_b, res_b) =
        getRightTriangularMMSystem<ElementIndex, T>(uplo, op, diag, alpha, n);

  set(a0, el_op_a, op);
  set(b, el_b);

  Tile<CT, Device::CPU> a(std::move(a0));

  tile::trmm(side, uplo, op, diag, alpha, a, b);

  CHECK_TILE_NEAR(res_b, b, 10 * (m + 1) * TypeUtilities<T>::error,
                  10 * (m + 1) * TypeUtilities<T>::error);
}

}
}
