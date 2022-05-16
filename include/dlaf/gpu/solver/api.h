//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

/// @file

#if defined(DLAF_WITH_HIP)

#include <hipblas.h>
#include <rocblas.h>
#include <rocsolver.h>

// Use of rocblas when the corresponding rocsolver functions are deprecated
#define cusolverDnChegst    rocsolver_chegst
#define cusolverDnCreate    rocblas_create_handle
#define cusolverDnDestroy   rocblas_destroy_handle
#define cusolverDnDsygst    rocsolver_dsygst
#define cusolverDnGetStream rocblas_get_stream
#define cusolverDnHandle_t  rocblas_handle
#define cusolverDnSetStream rocblas_set_stream
#define cusolverDnSsygst    rocsolver_ssygst
#define cusolverDnZhegst    rocsolver_zhegst
#define cusolverStatus_t    rocblas_status

// In ascendent order of error codes value
#define CUSOLVER_STATUS_SUCCESS             rocblas_status_success
#define CUSOLVER_STATUS_INVALID_HANDLE      rocblas_status_invalid_handle
#define CUSOLVER_STATUS_NOT_IMPLEMENTED     rocblas_status_not_implemented
#define CUSOLVER_STATUS_INVALID_POINTER     rocblas_status_invalid_pointer
#define CUSOLVER_STATUS_INVALID_SIZE        rocblas_status_invalid_size
#define CUSOLVER_STATUS_MEMORY_ERROR        rocblas_status_memory_error
#define CUSOLVER_STATUS_INTERNAL_ERROR      rocblas_status_internal_error
#define CUSOLVER_STATUS_PERF_DEGRADED       rocblas_status_perf_degraded
#define CUSOLVER_STATUS_SIZE_QUERY_MISMATCH rocblas_status_size_query_mismatch
#define CUSOLVER_STATUS_SIZE_INCREASED      rocblas_status_size_increased
#define CUSOLVER_STATUS_SIZE_UNCHANGED      rocblas_status_size_unchanged
#define CUSOLVER_STATUS_INVALID_VALUE       rocblas_status_invalid_value
#define CUSOLVER_STATUS_CONTINUE            rocblas_status_continue
#define CUSOLVER_STATUS_CHECK_NUMERICS_FAIL rocblas_status_check_numerics_fail

#elif defined(DLAF_WITH_CUDA)

#include <cusolverDn.h>

#endif
