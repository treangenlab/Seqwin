#pragma once

#include <cstdlib>

#if defined(__GLIBC__)
#include <malloc.h>
#endif

namespace seqwin::internal {

/**
 * @brief Ask the process allocator to return unused heap pages to the OS.
 *
 * On glibc, this calls `malloc_trim(0)`, which scans the allocator's arenas and
 * releases wholly unused pages when possible. The operation is best-effort:
 * live allocations and allocator fragmentation can prevent some pages from
 * being returned.
 *
 * Call this only at coarse phase boundaries, after large temporary containers
 * have been destroyed and before the next memory-intensive phase. Calling it
 * frequently can add allocator and system-call overhead.
 *
 * On platforms that do not use glibc, this function is a no-op.
 */
inline void trim_heap() noexcept
{
#if defined(__GLIBC__)
    (void)::malloc_trim(0);
#endif
}

} // namespace seqwin::internal
