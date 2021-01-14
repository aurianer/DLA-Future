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

#include <hpx/functional.hpp>
#include <hpx/include/util.hpp>
#include <hpx/local/future.hpp>
#include <hpx/tuple.hpp>

#include "dlaf/common/functional.h"

namespace dlaf {
namespace common {

/// A `promise` like type which is set upon destruction. The type separates the placement of data
/// (T) into the promise from notifying the corresponding `hpx::future`.
///
/// Note: The type is non-copiable.
template <class T>
class PromiseGuard {
public:
  /// Create a wrapper.
  /// @param object	the resource to wrap (the wrapper becomes the owner of the resource),
  /// @param next	the promise that has to be set on destruction.
  PromiseGuard(T object, hpx::lcos::local::promise<T> next)
      : object_(std::move(object)), promise_(std::move(next)) {}

  PromiseGuard(PromiseGuard&&) = default;
  PromiseGuard& operator=(PromiseGuard&&) = default;

  PromiseGuard(const PromiseGuard&) = delete;
  PromiseGuard& operator=(const PromiseGuard&) = delete;

  /// This is where the "magic" happens!
  ///
  /// If the wrapper is still valid, set the promise to unlock the next future.
  ~PromiseGuard() {
    if (promise_.valid())
      promise_.set_value(std::move(object_));
  }

  /// Get a reference to the internal object.
  T& ref() {
    return object_;
  }

  const T& ref() const {
    return object_;
  };

private:
  T object_;                              /// the object owned by the wrapper.
  hpx::lcos::local::promise<T> promise_;  /// the shared state that will unlock the next user.
};

/// Pipeline takes ownership of a given object and manages the access to this resource by serializing
/// calls. Anyone that requires access to the underlying resource will get an hpx::future, which is the
/// way to register to the queue. All requests are serialized and served in the same order they arrive.
///
/// The mechanism for auto-releasing the resource and passing it to the next user works thanks to the
/// internal Wrapper object. This Wrapper contains the real resource, and it will do what is needed to
/// unlock the next user as soon as the Wrapper is destroyed.
template <class T>
class Pipeline {
public:
  /// Create a Pipeline by moving in the resource (it takes the ownership).
  Pipeline(T object) {
    future_ = hpx::make_ready_future(std::move(object));
  }

  /// On destruction it waits that all users have finished using it.
  ~Pipeline() {
    if (future_.valid())
      future_.get();
  }

  /// Enqueue for the resource.
  ///
  /// @return a future that will become ready as soon as the previous user release the resource.
  hpx::future<PromiseGuard<T>> operator()() {
    auto before_last = std::move(future_);

    hpx::lcos::local::promise<T> promise_next;
    future_ = promise_next.get_future();

    return before_last.then(hpx::launch::sync,
                            hpx::util::unwrapping(
                                [promise_next = std::move(promise_next)](T&& object) mutable {
                                  return PromiseGuard<T>{std::move(object), std::move(promise_next)};
                                }));
  }

private:
  hpx::future<T> future_;  ///< This contains always the "tail" of the queue of futures.
};
}

namespace internal {
  struct UnwrapPromiseGuards {
    template <class T>
      decltype(auto) operator()(T&& t) {
        return std::forward<T>(t);
      }

    template <class T>
      T& operator()(dlaf::common::PromiseGuard<T>& u) {
        return u.ref();
      }
  };

  template <class Callable>
    struct unwrap_guards_impl {
      Callable f;

      template <class... Ts>
        auto operator()(Ts&&... ts) {
          // extract all futures
          auto t1 = hpx::tuple<Ts...>(std::forward<Ts>(ts)...);

          // Extract just PromiseGuards resources, move everything else
          auto t2 = apply(UnwrapPromiseGuards{}, t1);

          // TODO still to investigate why unwrapping is needed
          return hpx::invoke_fused(std::move(f), t2);
        }
    };
}

template <class Callable>
auto unwrap_guards(Callable func) {
  return internal::unwrap_guards_impl<Callable>{std::move(func)};
}


}
