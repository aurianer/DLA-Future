//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

namespace dlaf {
namespace matrix {

template <class T, Device device>
hpx::shared_future<Tile<const T, device>> MatrixView<const T, device>::read(
    const LocalTileIndex& index) noexcept {
  std::size_t i = static_cast<std::size_t>(tileLinearIndex(index));
  return tile_shared_futures_[i];
}

template <class T, Device device>
void MatrixView<const T, device>::done(const LocalTileIndex& index) noexcept {
  std::size_t i = static_cast<std::size_t>(tileLinearIndex(index));
  tile_shared_futures_[i] = {};
}

}
}
