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

/// @file

#if defined(DLAF_TRACE_WITH_HPX)
#include <hpx/util/annotated_function.hpp>
#define DLAF_TRACE(task, _, group, ...)  hpx::util::annotated_function(task, group)
#elif defined(DLAF_TRACE_CUSTOM)
#include "dlaf/profiling/profiler.h"
#define DLAF_TRACE(task, name, group, ...) dlaf::profiling::util::time_it(task, name, group)
#else
#define DLAF_TRACE(task, ...) task
#endif
