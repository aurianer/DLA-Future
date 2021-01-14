//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

/// @file

#include <type_traits>
#include <utility>

#include <hpx/functional.hpp>
#include <hpx/pack_traversal/unwrap.hpp>
#include <hpx/tuple.hpp>

namespace dlaf {

#define DLAF_MAKE_CALLABLE_OBJECT(fname)          \
  constexpr struct fname##_t {                    \
    template <typename... Ts>                     \
    decltype(auto) operator()(Ts&&... ts) const { \
      return fname(std::forward<Ts>(ts)...);      \
    }                                             \
  } fname##_o {                                   \
  }

namespace internal {

template <class Func, class Tuple, std::size_t... Is>
auto apply_impl(Func func, Tuple&& t, std::index_sequence<Is...>) {
  return hpx::tuple<decltype(func(hpx::get<Is, std::decay_t<Tuple>>(t)))...>(func(hpx::get<Is>(t))...);
}

}

// TODO apply does not work with single argument (useful becase unwrap(a) does not return a tuple)
template <class Func, class Tuple>
auto apply(Func func, Tuple&& t) {
  return internal::apply_impl(std::move(func), std::forward<Tuple>(t),
                              std::make_index_sequence<hpx::tuple_size<std::decay_t<Tuple>>::value>());
}

}
