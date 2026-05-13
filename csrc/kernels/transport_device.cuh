#pragma once

#if defined(DEEPEP_USE_ISHMEM) && __has_include(<ishmem.h>)
#include "ishmem_device.cuh"
#else
#include "ibgda_device.cuh"
#endif
