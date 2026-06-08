#include <ishmem.h>
#include <ishmemx.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <string>
#include <sycl/sycl.hpp>
#include <thread>
#include <vector>

namespace {

constexpr int kSentinel = -777777;

void usage(const char* argv0) {
    std::cerr << "Usage: " << argv0 << " --case CASE [--num-elems N]\n"
              << "Cases:\n"
              << "  init_attr_uniqueid\n"
              << "  normal_putmem_blocking\n"
              << "  normal_putmem_nbi_quiet\n"
              << "  normal_putmem_parallel_work_items\n"
              << "  combine_payload_work_group_putmem_atomic_tail\n"
              << "  work_group_sideband_putmem_nbi_atomic_tail_repeat\n"
              << "  work_group_sideband_putmem_nbi_split_atomic_tail_repeat\n"
              << "  ll_int_put_nbi_quiet\n"
              << "  ll_putmem_nbi_atomic_flag\n"
              << "  atomic_add_remote\n"
              << "  atomic_add_remote_many\n"
              << "  atomic_add_all_pes\n"
              << "  ll_ptr_device\n"
              << "  team_split_sync_destroy\n"
              << "  device_barrier_all\n"
              << "  normal_sync_all_device\n"
              << "  ll_barrier_work_group\n"
              << "  quiet_empty\n"
              << "  intranode_no_mapped_ishmem_api\n"
              << "  multi_channel_nbi_quiet_atomic\n"
              << "  multi_channel_nbi_quiet_atomic_split\n"
              << "  dispatch_then_combine_blocking_put\n"
              << "\n  --- Scalar device API hang/stability tests ---\n"
              << "  scalar_putmem_single_task\n"
              << "  scalar_putmem_nbi_quiet_single_task\n"
              << "  scalar_int_put_single_task\n"
              << "  scalar_quiet_after_nbi_single_task\n"
              << "  scalar_fence_single_task\n"
              << "  scalar_barrier_all_device\n"
              << "  scalar_sync_all_device\n"
              << "  scalar_putmem_parallel_for\n"
              << "  scalar_putmem_nd_range_no_wg_api\n"
              << "  scalar_get_single_task\n"
              << "  scalar_putmem_large_single_task\n"
              << "  scalar_p_single_task\n"
              << "  scalar_g_single_task\n"
              << "  scalar_atomic_fetch_add_single_task\n"
              << "  scalar_quiet_no_preceding_put\n"
              << "  scalar_put_multi_pe_single_task\n"
              << "  scalar_nbi_quiet_repeat_single_task\n";
}

std::string parse_case(int argc, char** argv) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], "--case") == 0) {
            return argv[i + 1];
        }
    }
    return {};
}

int parse_num_elems(int argc, char** argv, int default_value) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], "--num-elems") == 0) {
            return std::max(std::atoi(argv[i + 1]), 1);
        }
    }
    return default_value;
}

int env_int_or(const char* name, int default_value) {
    const char* value = std::getenv(name);
    return value == nullptr ? default_value : std::atoi(value);
}

void init_buffers(sycl::queue& queue, int* recv, int* src, int num_elems, int rank, bool zero_recv) {
    queue
        .parallel_for(sycl::range<1>(num_elems),
                      [=](sycl::id<1> id) {
                          const int i = static_cast<int>(id[0]);
                          recv[i] = zero_recv ? 0 : kSentinel;
                          src[i] = (rank + 1) * 100000 + i;
                      })
        .wait_and_throw();
}

int count_errors(sycl::queue& queue, int* recv, int num_elems, int expected_first) {
    std::vector<int> host(num_elems);
    queue.memcpy(host.data(), recv, static_cast<size_t>(num_elems) * sizeof(int)).wait_and_throw();
    int errors = 0;
    for (int i = 0; i < num_elems; ++i) {
        const int expected = expected_first == kSentinel ? kSentinel : expected_first + i;
        if (host[i] != expected) {
            if (errors < 8) {
                std::cout << "  mismatch index=" << i << " expected=" << expected << " got=" << host[i] << "\n";
            }
            ++errors;
        }
    }
    return errors;
}

int count_constant_errors(sycl::queue& queue, int* recv, int num_elems, int expected) {
    std::vector<int> host(num_elems);
    queue.memcpy(host.data(), recv, static_cast<size_t>(num_elems) * sizeof(int)).wait_and_throw();
    int errors = 0;
    for (int i = 0; i < num_elems; ++i) {
        if (host[i] != expected) {
            if (errors < 8) {
                std::cout << "  mismatch index=" << i << " expected=" << expected << " got=" << host[i] << "\n";
            }
            ++errors;
        }
    }
    return errors;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string test_case = parse_case(argc, argv);
    if (test_case.empty()) {
        usage(argv[0]);
        return 2;
    }

    int num_elems = parse_num_elems(argc, argv, 64);
    if (test_case == "atomic_add_remote" || test_case == "atomic_add_all_pes") {
        num_elems = parse_num_elems(argc, argv, 1);
    } else if (test_case == "atomic_add_remote_many") {
        num_elems = parse_num_elems(argc, argv, 64);
    } else if (test_case == "multi_channel_nbi_quiet_atomic" ||
               test_case == "multi_channel_nbi_quiet_atomic_split" ||
               test_case == "dispatch_then_combine_blocking_put") {
        num_elems = parse_num_elems(argc, argv, 2048);
    }

    if (test_case == "init_attr_uniqueid") {
        ishmemx_uniqueid_t unique_id{};
        if (ishmemx_get_uniqueid(&unique_id) != 0) {
            std::cerr << "ishmemx_get_uniqueid failed\n";
            return 2;
        }
        ishmemx_attr_t attr{};
        attr.runtime = ISHMEMX_RUNTIME_MPI;
        attr.initialize_runtime = true;
        attr.gpu = true;
        attr.use_uid = true;
        attr.nranks = env_int_or("PMI_SIZE", env_int_or("WORLD_SIZE", 2));
        attr.rank = env_int_or("PMI_RANK", env_int_or("RANK", 0));
        attr.uid = &unique_id;
        ishmemx_init_attr(&attr);
    } else {
        ishmem_init();
    }
    sycl::queue queue;
    const int rank = ishmem_my_pe();
    const int world = ishmem_n_pes();

    // if (world != 2) {
    //     if (rank == 0) {
    //         std::cerr << "expected 2 PEs, got " << world << "\n";
    //     }
    //     ishmem_finalize();
    //     return 2;
    // }

    std::cout << "world size: " << world << " rank:" << rank << "\n";

    const int peer = (rank + 1) % world;       /* ring next: who I send to */
    const int sender = (rank - 1 + world) % world; /* ring prev: who sends to me */
    const int flag_idx = num_elems;
    const int queue_elems = num_elems * world;
    const int combine_tail_base = queue_elems;
    const int warmup_tail_base = combine_tail_base + world;
    const int alloc_elems = warmup_tail_base + world;
    int* recv = static_cast<int*>(ishmem_align(128, static_cast<size_t>(alloc_elems) * sizeof(int)));
    int* src = static_cast<int*>(ishmem_align(128, static_cast<size_t>(alloc_elems) * sizeof(int)));
    if (recv == nullptr || src == nullptr) {
        std::cerr << "[rank " << rank << "] ishmem_align failed\n";
        ishmem_finalize();
        return 2;
    }

    int errors = 0;
    bool ptr_was_null = false;
    bool ran = true;

    if (test_case == "init_attr_uniqueid") {
        init_buffers(queue, recv, src, num_elems, rank, false);
        ishmem_barrier_all();
        queue
            .submit([&](sycl::handler& h) {
                h.single_task([=]() { ishmem_putmem(recv, src, static_cast<size_t>(num_elems) * sizeof(int), peer); });
            })
            .wait_and_throw();
        ishmem_barrier_all();
        errors = count_errors(queue, recv, num_elems, (sender + 1) * 100000);
    } else if (test_case == "intranode_no_mapped_ishmem_api") {
        init_buffers(queue, recv, src, num_elems, rank, false);
        errors = count_constant_errors(queue, recv, num_elems, kSentinel);
        std::cout << "[rank " << rank << "] intranode path has no mapped NVSHMEM/iSHMEM API in csrc/kernels/intranode.cu\n";
    } else if (test_case == "normal_putmem_blocking") {
        init_buffers(queue, recv, src, num_elems, rank, false);
        queue
            .single_task([=]() {
                recv[flag_idx] = 0;
                src[flag_idx] = 0;
            })
            .wait_and_throw();
        ishmem_barrier_all();
        queue
            .submit([&](sycl::handler& h) {
                h.single_task([=]() { ishmem_putmem(recv, src, static_cast<size_t>(num_elems) * sizeof(int), peer); });
            })
            .wait_and_throw();
        ishmem_barrier_all();
        errors = count_errors(queue, recv, num_elems, (sender + 1) * 100000);
    } else if (test_case == "normal_putmem_nbi_quiet") {
        init_buffers(queue, recv, src, num_elems, rank, false);
        queue
            .single_task([=]() {
                recv[flag_idx] = 0;
                src[flag_idx] = 0;
            })
            .wait_and_throw();
        ishmem_barrier_all();
        queue
            .submit([&](sycl::handler& h) {
                h.single_task([=]() {
                    ishmem_putmem_nbi(recv, src, static_cast<size_t>(num_elems) * sizeof(int), peer);
                    ishmem_quiet();
                });
            })
            .wait_and_throw();
        ishmem_barrier_all();
        errors = count_errors(queue, recv, num_elems, (sender + 1) * 100000);
    } else if (test_case == "normal_putmem_parallel_work_items") {
        init_buffers(queue, recv, src, num_elems, rank, false);
        queue
            .single_task([=]() {
                recv[flag_idx] = 0;
                src[flag_idx] = 0;
            })
            .wait_and_throw();
        ishmem_barrier_all();
        queue
            .parallel_for(sycl::range<1>(num_elems),
                          [=](sycl::id<1> id) {
                              const int i = static_cast<int>(id[0]);
                              ishmem_putmem(recv + i, src + i, sizeof(int), peer);
                          })
            .wait_and_throw();
        ishmem_barrier_all();
        errors = count_errors(queue, recv, num_elems, (sender + 1) * 100000);
    } else if (test_case == "combine_payload_work_group_putmem_atomic_tail") {
        /* Merged-kernel test: work-group payload NBI puts followed by an
         * atomic completion tail, all within a SINGLE nd_range kernel.
         * This matches the production DeepEP combine kernel pattern where
         * WG NBI payload puts and atomic_add tail are in the same kernel.
         *
         * Layout:
         *   recv[0 .. num_elems*world-1] : payload (rank * num_elems slots each)
         *   recv[combine_tail_base + r]  : atomic completion counter for rank r
         *
         * Each work-group i represents source rank i sending payload to this rank.
         * After all WG puts complete (via work-group barrier + device quiet),
         * work-item 0 of each WG issues ishmem_int_atomic_add on the combine
         * tail counter, signalling that source rank i's contribution is complete. */
        int peer = (rank + 1) % world;
        queue
            .parallel_for(sycl::range<1>(alloc_elems),
                          [=](sycl::id<1> id) {
                              const int i = static_cast<int>(id[0]);
                              recv[i] = i >= combine_tail_base ? 0 : kSentinel;
                              src[i] = (rank + 1) * 100000 + i;
                          })
            .wait_and_throw();
        queue
            .single_task([=]() {
                for (int i = 0; i < world; ++i) {
                    recv[combine_tail_base + i] = 0;
                }
            })
            .wait_and_throw();
        std::cout << "[rank " << rank << "] step1: before warmup barrier\n" << std::flush;
        ishmem_barrier_all();

        /* step2: merged WG-put + atomic-tail kernel */
        std::cout << "[rank " << rank << "] step2: before merged WG put + atomic kernel\n" << std::flush;
        queue
            .submit([&](sycl::handler& h) {
                h.parallel_for(sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(world) * 32), sycl::range<1>(32)),
                               [=](sycl::nd_item<1> it) {
                                   auto group = it.get_group();
                                   const int src_rank = static_cast<int>(it.get_group(0));
                                   const int local_id = static_cast<int>(it.get_local_id(0));
                                   const int local_size = static_cast<int>(it.get_local_range(0));
                                   int* dst = recv + rank * num_elems;
                                   /* Phase 1: WG payload put */
                                   if (src_rank == rank) {
                                       for (int i = local_id; i < num_elems; i += local_size) {
                                           dst[i] = src[i];
                                       }
                                   } else {
                                       ishmemx_putmem_nbi_work_group(dst, src, static_cast<size_t>(num_elems) * sizeof(int), src_rank, group);
                                   }
                                   /* WG barrier ensures all work-items finish WG NBI put */
                                   sycl::group_barrier(group);
                                   /* Phase 2: leader issues atomic tail after device quiet */
                                   if (local_id == 0 && src_rank != rank) {
                                       ishmem_quiet();
                                       ishmem_int_atomic_add(recv + combine_tail_base + rank, 1, src_rank);
                                   }
                               });
            })
            .wait_and_throw();
        std::cout << "[rank " << rank << "] step3: merged kernel done\n" << std::flush;
        ishmem_barrier_all();
        std::cout << "[rank " << rank << "] step4: after barrier\n" << std::flush;
        std::vector<int> host(alloc_elems);
        queue.memcpy(host.data(), recv, static_cast<size_t>(alloc_elems) * sizeof(int)).wait_and_throw();
        for (int src_rank = 0; src_rank < world; ++src_rank) {
            for (int i = 0; i < num_elems; ++i) {
                const int index = src_rank * num_elems + i;
                const int expected = (src_rank + 1) * 100000 + i;
                if (host[index] != expected) {
                    if (errors < 8) {
                        std::cout << "  combine queue mismatch src_rank=" << src_rank << " index=" << i << " expected=" << expected
                                  << " got=" << host[index] << "\n";
                    }
                    ++errors;
                }
            }
            /* Each non-self rank does atomic_add(1) to this PE's tail
             * at position combine_tail_base + sender_rank.  So
             * tail[src_rank] == 1 iff src_rank != rank (remote sender). */
            int expected_tail = (src_rank != rank) ? 1 : 0;
            if (host[combine_tail_base + src_rank] != expected_tail) {
                std::cout << "  combine tail mismatch src_rank=" << src_rank
                          << " expected=" << expected_tail
                          << " got=" << host[combine_tail_base + src_rank] << "\n";
                ++errors;
            }
        }
    } else if (test_case == "combine_payload_work_group_putmem_atomic_tail_separate_kernels") {
        queue
            .parallel_for(sycl::range<1>(alloc_elems),
                          [=](sycl::id<1> id) {
                              const int i = static_cast<int>(id[0]);
                              recv[i] = i >= combine_tail_base ? 0 : kSentinel;
                              src[i] = (rank + 1) * 100000 + i;
                          })
            .wait_and_throw();
        queue
            .single_task([=]() {
                for (int i = 0; i < world; ++i) {
                    recv[combine_tail_base + i] = 0;
                    recv[warmup_tail_base + i] = 0;
                }
            })
            .wait_and_throw();
        ishmem_barrier_all();
        queue.submit([&](sycl::handler& h) { h.single_task([=]() { ishmem_int_atomic_add(recv + warmup_tail_base + rank, 1, peer); }); })
            .wait_and_throw();
        ishmem_barrier_all();
        queue
            .submit([&](sycl::handler& h) {
                h.parallel_for(sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(world) * 32), sycl::range<1>(32)),
                               [=](sycl::nd_item<1> it) {
                                   auto group = it.get_group();
                                   const int src_rank = static_cast<int>(it.get_group(0));
                                   const int local_id = static_cast<int>(it.get_local_id(0));
                                   const int local_size = static_cast<int>(it.get_local_range(0));
                                   int* dst = recv + rank * num_elems;
                                   if (src_rank == rank) {
                                       for (int i = local_id; i < num_elems; i += local_size) {
                                           dst[i] = src[i];
                                       }
                                   } else {
                                       ishmemx_putmem_work_group(dst, src, static_cast<size_t>(num_elems) * sizeof(int), src_rank, group);
                                   }
                                   sycl::group_barrier(group);
                               });
            })
            .wait_and_throw();
        queue
            .submit([&](sycl::handler& h) {
                h.single_task([=]() {
                    for (int src_rank = 0; src_rank < world; ++src_rank) {
                        if (src_rank == rank) {
                            recv[combine_tail_base + rank] += 1;
                        } else {
                            ishmem_int_atomic_add(recv + combine_tail_base + rank, 1, src_rank);
                        }
                    }
                });
            })
            .wait_and_throw();
        ishmem_barrier_all();
        std::vector<int> host(alloc_elems);
        queue.memcpy(host.data(), recv, static_cast<size_t>(alloc_elems) * sizeof(int)).wait_and_throw();
        for (int src_rank = 0; src_rank < world; ++src_rank) {
            for (int i = 0; i < num_elems; ++i) {
                const int index = src_rank * num_elems + i;
                const int expected = (src_rank + 1) * 100000 + i;
                if (host[index] != expected) {
                    if (errors < 8) {
                        std::cout << "  combine queue mismatch src_rank=" << src_rank << " index=" << i << " expected=" << expected
                                  << " got=" << host[index] << "\n";
                    }
                    ++errors;
                }
            }
            if (host[combine_tail_base + src_rank] != 1) {
                std::cout << "  combine tail mismatch src_rank=" << src_rank << " expected=1 got=" << host[combine_tail_base + src_rank]
                          << "\n";
                ++errors;
            }
        }
    } else if (test_case == "work_group_sideband_putmem_nbi_atomic_tail_repeat" ||
               test_case == "work_group_sideband_putmem_nbi_split_atomic_tail_repeat") {
        const bool split = test_case == "work_group_sideband_putmem_nbi_split_atomic_tail_repeat";
        constexpr int repeats = 1;
        for (int iter = 0; iter < repeats; ++iter) {
            init_buffers(queue, recv, src, num_elems, rank, false);
            queue
                .single_task([=]() {
                    recv[flag_idx] = 0;
                    src[flag_idx] = 0;
                })
                .wait_and_throw();
            ishmem_barrier_all();
            queue
                .submit([&](sycl::handler& h) {
                    h.parallel_for(sycl::nd_range<1>(sycl::range<1>(32), sycl::range<1>(32)),
                                   [=](sycl::nd_item<1> it) {
                                       auto group = it.get_group();
                                       if (split) {
                                           const int first = num_elems / 2;
                                           ishmemx_putmem_nbi_work_group(
                                               recv, src, static_cast<size_t>(first) * sizeof(int), peer, group);
                                           ishmemx_quiet_work_group(group);
                                           ishmemx_putmem_nbi_work_group(
                                               recv + first, src + first,
                                               static_cast<size_t>(num_elems - first) * sizeof(int), peer, group);
                                       } else {
                                           ishmemx_putmem_nbi_work_group(
                                               recv, src, static_cast<size_t>(num_elems) * sizeof(int), peer, group);
                                       }
                                       sycl::group_barrier(group);
                                       if (group.leader()) {
                                           ishmem_int_atomic_add(recv + flag_idx, 1, peer);
                                       }
                                   });
                })
                .wait_and_throw();
            ishmem_barrier_all();
            errors += count_errors(queue, recv, num_elems, (sender + 1) * 100000);
            std::vector<int> flag_host(1);
            queue.memcpy(flag_host.data(), recv + flag_idx, sizeof(int)).wait_and_throw();
            if (flag_host[0] != 1) {
                std::cout << "  iter=" << iter << " flag mismatch expected=1 got=" << flag_host[0] << "\n";
                ++errors;
            }
        }
    } else if (test_case == "ll_int_put_nbi_quiet") {
        init_buffers(queue, recv, src, num_elems, rank, false);
        queue
            .single_task([=]() {
                recv[flag_idx] = 0;
                src[flag_idx] = 0;
            })
            .wait_and_throw();
        ishmem_barrier_all();
        queue
            .submit([&](sycl::handler& h) {
                h.single_task([=]() {
                    ishmem_int_put_nbi(recv, src, static_cast<size_t>(num_elems), peer);
                    ishmem_quiet();
                });
            })
            .wait_and_throw();
        ishmem_barrier_all();
        errors = count_errors(queue, recv, num_elems, (sender + 1) * 100000);
    } else if (test_case == "ll_putmem_nbi_atomic_flag") {
        init_buffers(queue, recv, src, num_elems, rank, false);
        queue
            .single_task([=]() {
                recv[flag_idx] = 0;
                src[flag_idx] = 0;
            })
            .wait_and_throw();
        ishmem_barrier_all();
        queue
            .submit([&](sycl::handler& h) {
                h.single_task([=]() {
                    ishmem_putmem_nbi(recv, src, static_cast<size_t>(num_elems) * sizeof(int), peer);
                    ishmem_int_atomic_add(recv + flag_idx, 1, peer);
                });
            })
            .wait_and_throw();
        ishmem_barrier_all();
        errors = count_errors(queue, recv, num_elems, (sender + 1) * 100000);
        std::vector<int> flag_host(1);
        queue.memcpy(flag_host.data(), recv + flag_idx, sizeof(int)).wait_and_throw();
        if (flag_host[0] != 1) {
            std::cout << "  flag mismatch expected=1 got=" << flag_host[0] << "\n";
            ++errors;
        }
    } else if (test_case == "atomic_add_remote" || test_case == "atomic_add_remote_many") {
        init_buffers(queue, recv, src, num_elems, rank, true);
        queue.single_task([=]() { recv[flag_idx] = 0; }).wait_and_throw();
        ishmem_barrier_all();
        queue
            .submit([&](sycl::handler& h) {
                h.single_task([=]() {
                    for (int i = 0; i < num_elems; ++i) {
                        ishmem_int_atomic_add(recv + i, rank + 1, peer);
                    }
                });
            })
            .wait_and_throw();
        ishmem_barrier_all();
        errors = count_constant_errors(queue, recv, num_elems, sender + 1);
    } else if (test_case == "atomic_add_all_pes") {
        init_buffers(queue, recv, src, num_elems, rank, true);
        queue.single_task([=]() { recv[flag_idx] = 0; }).wait_and_throw();
        ishmem_barrier_all();
        queue
            .submit([&](sycl::handler& h) {
                h.single_task([=]() {
                    for (int pe = 0; pe < world; ++pe) {
                        for (int i = 0; i < num_elems; ++i) {
                            ishmem_int_atomic_add(recv + i, rank + 1, pe);
                        }
                    }
                });
            })
            .wait_and_throw();
        ishmem_barrier_all();
        errors = count_constant_errors(queue, recv, num_elems, world * (world + 1) / 2);
    } else if (test_case == "ll_ptr_device") {
        init_buffers(queue, recv, src, num_elems, rank, false);
        queue
            .single_task([=]() {
                recv[flag_idx] = 0;
                src[flag_idx] = 0;
            })
            .wait_and_throw();
        ishmem_barrier_all();
        auto* ptr_state = sycl::malloc_host<int>(1, queue);
        *ptr_state = 0;
        queue
            .submit([&](sycl::handler& h) {
                h.single_task([=]() {
                    auto* peer_recv = static_cast<int*>(ishmem_ptr(recv, peer));
                    if (peer_recv == nullptr) {
                        *ptr_state = 1;
                    } else {
                        for (int i = 0; i < num_elems; ++i) {
                            peer_recv[i] = src[i];
                        }
                    }
                });
            })
            .wait_and_throw();
        ishmem_barrier_all();
        ptr_was_null = *ptr_state != 0;
        sycl::free(ptr_state, queue);
        errors = ptr_was_null ? count_constant_errors(queue, recv, num_elems, kSentinel)
                              : count_errors(queue, recv, num_elems, (sender + 1) * 100000);
    } else if (test_case == "team_split_sync_destroy") {
        init_buffers(queue, recv, src, num_elems, rank, false);
        queue
            .single_task([=]() {
                recv[flag_idx] = 0;
                src[flag_idx] = 0;
            })
            .wait_and_throw();
        ishmem_barrier_all();
        ishmem_team_t rdma_like_team = ISHMEM_TEAM_INVALID;
        ishmem_team_config_t* config = nullptr;
        int ret = ishmem_team_split_strided(ISHMEM_TEAM_WORLD, 0, 1, world, config, 0, &rdma_like_team);
        if (ret != 0 || rdma_like_team == ISHMEM_TEAM_INVALID) {
            std::cout << "  team split failed ret=" << ret << "\n";
            ++errors;
        } else {
            queue
                .submit([&](sycl::handler& h) {
                    h.single_task([=]() {
                        ishmem_int_put(recv, src, static_cast<size_t>(num_elems), peer);
                        ishmem_team_sync(rdma_like_team);
                    });
                })
                .wait_and_throw();
            errors += count_errors(queue, recv, num_elems, (sender + 1) * 100000);
            ishmem_team_destroy(rdma_like_team);
        }
    } else if (test_case == "device_barrier_all") {
        init_buffers(queue, recv, src, num_elems, rank, false);
        queue
            .single_task([=]() {
                recv[flag_idx] = 0;
                src[flag_idx] = 0;
            })
            .wait_and_throw();
        ishmem_barrier_all();
        queue
            .submit([&](sycl::handler& h) {
                h.single_task([=]() {
                    ishmem_int_put(recv, src, static_cast<size_t>(num_elems), peer);
                    ishmem_barrier_all();
                });
            })
            .wait_and_throw();
        errors = count_errors(queue, recv, num_elems, (sender + 1) * 100000);
    } else if (test_case == "normal_sync_all_device") {
        init_buffers(queue, recv, src, num_elems, rank, false);
        queue
            .single_task([=]() {
                recv[flag_idx] = 0;
                src[flag_idx] = 0;
            })
            .wait_and_throw();
        ishmem_barrier_all();
        queue
            .submit([&](sycl::handler& h) {
                h.single_task([=]() {
                    ishmem_int_put(recv, src, static_cast<size_t>(num_elems), peer);
                    ishmem_sync_all();
                });
            })
            .wait_and_throw();
        errors = count_errors(queue, recv, num_elems, (sender + 1) * 100000);
    } else if (test_case == "ll_barrier_work_group") {
        init_buffers(queue, recv, src, num_elems, rank, false);
        queue
            .single_task([=]() {
                recv[flag_idx] = 0;
                src[flag_idx] = 0;
            })
            .wait_and_throw();
        ishmem_barrier_all();
        queue
            .submit([&](sycl::handler& h) {
                h.parallel_for(sycl::nd_range<1>(sycl::range<1>(32), sycl::range<1>(32)), [=](sycl::nd_item<1> it) {
                    auto group = it.get_group();
                    if (group.leader()) {
                        ishmem_int_put(recv, src, static_cast<size_t>(num_elems), peer);
                    }
                    ishmemx_barrier_all_work_group(group);
                });
            })
            .wait_and_throw();
        errors = count_errors(queue, recv, num_elems, (sender + 1) * 100000);
    } else if (test_case == "quiet_empty") {
        init_buffers(queue, recv, src, num_elems, rank, false);
        ishmem_barrier_all();
        queue.submit([&](sycl::handler& h) { h.single_task([=]() { ishmem_quiet(); }); }).wait_and_throw();
        ishmem_barrier_all();
        errors = count_constant_errors(queue, recv, num_elems, kSentinel);
    } else if (test_case == "l3_snoop_verify") {
        /* -----------------------------------------------------------------
         * L3 Snoop Verification Test
         * -----------------------------------------------------------------
         * Purpose: Verify that NIC DMA reads from GPU VRAM snoop the L3
         * cache, so that a device-scope fence (EU write buffer → L3) is
         * sufficient for RDMA correctness.
         *
         * Method:
         *   Each iteration writes a unique pattern to a SEPARATE region of
         *   snoop_src (no reuse!) then immediately NBI-puts it to the peer.
         *   Because each iteration has its own source region, the NIC can
         *   DMA-read it at any time without overwrite interference.
         *
         *   If the NIC DMA bypasses L3 and reads stale GDDR6 (zeros), we
         *   will see mismatches.  If L3 is snooped, all data will match.
         *
         *   The total src buffer (NITERS * CHUNK * 4 bytes) is kept small
         *   enough to fit in L3, so natural eviction to GDDR6 is unlikely.
         *   This maximises the window where L3-bypass would be exposed.
         * ----------------------------------------------------------------- */
        constexpr int NITERS = 4096;
        constexpr int CHUNK  = 256;  /* ints per iteration (1 KB) */
        const int total_elems = NITERS * CHUNK;

        int* snoop_recv = static_cast<int*>(
            ishmem_align(128, static_cast<size_t>(total_elems) * sizeof(int)));
        int* snoop_src = static_cast<int*>(
            ishmem_align(128, static_cast<size_t>(total_elems) * sizeof(int)));
        if (snoop_recv == nullptr || snoop_src == nullptr) {
            std::cerr << "[rank " << rank << "] ishmem_align failed for l3_snoop\n";
            ishmem_finalize();
            return 2;
        }

        /* Zero BOTH buffers on device — GDDR6 backing will be zeros */
        queue.parallel_for(sycl::range<1>(total_elems),
                           [=](sycl::id<1> id) {
                               snoop_recv[id[0]] = 0;
                               snoop_src[id[0]]  = 0;
                           })
            .wait_and_throw();
        ishmem_barrier_all();

        /* GPU kernel: write unique pattern per iteration, NBI-put from
         * separate src region.  No src buffer reuse → no overwrite race. */
        queue.submit([&](sycl::handler& h) {
            h.single_task([=]() {
                for (int iter = 0; iter < NITERS; ++iter) {
                    int* iter_src = snoop_src + iter * CHUNK;
                    const int tag = (iter + 1) * 100000;
                    for (int i = 0; i < CHUNK; ++i) {
                        iter_src[i] = tag + i;
                    }
                    /* NBI put from this iteration's unique src region.
                     * Only a device-scope fence (in ring_doorbell) separates
                     * the writes above from the NIC's DMA read of iter_src.
                     * If NIC bypasses L3, it reads zeros (GDDR6 initial). */
                    ishmem_putmem_nbi(snoop_recv + iter * CHUNK, iter_src,
                                     static_cast<size_t>(CHUNK) * sizeof(int), peer);
                }
                ishmem_quiet();
            });
        }).wait_and_throw();

        ishmem_barrier_all();

        /* Verify on host */
        std::vector<int> host(total_elems);
        queue.memcpy(host.data(), snoop_recv,
                     static_cast<size_t>(total_elems) * sizeof(int))
            .wait_and_throw();

        for (int iter = 0; iter < NITERS; ++iter) {
            const int tag = (iter + 1) * 100000;
            for (int i = 0; i < CHUNK; ++i) {
                const int idx = iter * CHUNK + i;
                const int expected = tag + i;
                if (host[idx] != expected) {
                    if (errors < 16) {
                        std::cout << "  L3 snoop MISMATCH iter=" << iter
                                  << " i=" << i << " expected=" << expected
                                  << " got=" << host[idx];
                        if (host[idx] == 0) {
                            std::cout << " (zero — NIC read stale GDDR6, L3 bypassed)";
                        }
                        std::cout << "\n";
                    }
                    ++errors;
                }
            }
        }

        if (errors == 0) {
            std::cout << "[rank " << rank << "] L3 snoop verified: "
                      << NITERS << " iterations x " << CHUNK
                      << " ints — NIC DMA correctly snoops L3\n";
        } else {
            std::cout << "[rank " << rank << "] L3 snoop FAILED: "
                      << errors << "/" << total_elems
                      << " mismatches — NIC DMA may bypass L3\n";
        }

        ishmem_free(snoop_src);
        ishmem_free(snoop_recv);
    } else if (test_case == "multi_channel_nbi_quiet_atomic") {
        // Reproducer for DeepEP internode multi-channel/multi-window flow.
        // Pattern: count exchange (blocking putmem) → barrier →
        //   per iteration: NBI WG put → quiet+atomic → barrier
        // Use large num_elems via --num-elems (default 1024) and 8 iterations.
        const int total_iters = 8;  // 2 channels × 4 windows
        constexpr int wg_size = 32;

        // Buffer layout:
        //   recv[0 .. num_elems-1]: payload data (overwritten each iter)
        //   tail[0 .. world-1]: tail counters per sender rank
        const int tail_base = num_elems;
        const int total_alloc = tail_base + world;
        int* mbuf_recv = static_cast<int*>(ishmem_align(128, static_cast<size_t>(total_alloc) * sizeof(int)));
        int* mbuf_src  = static_cast<int*>(ishmem_align(128, static_cast<size_t>(total_alloc) * sizeof(int)));
        if (!mbuf_recv || !mbuf_src) {
            std::cerr << "[rank " << rank << "] ishmem_align failed (multi_channel)\n";
            errors = 1;
        } else {
            // Init
            queue.parallel_for(sycl::range<1>(total_alloc), [=](sycl::id<1> id) {
                const int i = static_cast<int>(id[0]);
                mbuf_recv[i] = 0;
                mbuf_src[i] = (rank + 1) * 100000 + i;
            }).wait_and_throw();
            ishmem_barrier_all();

            // Warmup: one atomic_add to prime the path (matches DeepEP warmup)
            queue.submit([&](sycl::handler& h) {
                h.single_task([=]() {
                    ishmem_int_atomic_add(mbuf_recv + tail_base + rank, 0, peer);
                });
            }).wait_and_throw();
            ishmem_barrier_all();

            // Reset tails after warmup
            queue.single_task([=]() {
                for (int i = 0; i < world; ++i) mbuf_recv[tail_base + i] = 0;
            }).wait_and_throw();
            ishmem_barrier_all();

            // Count exchange: blocking putmem (mimics DispatchCountExchangeKernel)
            int* count_buf = static_cast<int*>(ishmem_align(128, static_cast<size_t>(world) * sizeof(int)));
            if (count_buf) {
                queue.single_task([=]() { count_buf[rank] = num_elems; }).wait_and_throw();
                ishmem_barrier_all();
                queue.submit([&](sycl::handler& h) {
                    h.single_task([=]() {
                        for (int dst = 0; dst < world; ++dst) {
                            if (dst != rank)
                                ishmem_putmem(count_buf, count_buf, static_cast<size_t>(world) * sizeof(int), dst);
                        }
                    });
                }).wait_and_throw();
                ishmem_barrier_all();
            }

            for (int iter = 0; iter < total_iters; ++iter) {
                // Kernel A: NBI work-group put (payload)
                queue.submit([&](sycl::handler& h) {
                    h.parallel_for(
                        sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(world) * wg_size),
                                          sycl::range<1>(wg_size)),
                        [=](sycl::nd_item<1> it) {
                            auto group = it.get_group();
                            const int dst_rank = static_cast<int>(it.get_group(0));
                            if (dst_rank == rank) {
                                // Self-send: local copy
                                const int local_id = static_cast<int>(it.get_local_id(0));
                                const int local_size = static_cast<int>(it.get_local_range(0));
                                for (int i = local_id; i < num_elems; i += local_size) {
                                    mbuf_recv[i] = mbuf_src[i];
                                }
                            } else {
                                ishmemx_putmem_nbi_work_group(
                                    mbuf_recv, mbuf_src,
                                    static_cast<size_t>(num_elems) * sizeof(int),
                                    dst_rank, group);
                            }
                        });
                }).wait_and_throw();

                // Kernel B: quiet + atomic_add (tail)
                queue.submit([&](sycl::handler& h) {
                    h.single_task([=]() {
                        ishmem_quiet();
                        for (int dst_rank = 0; dst_rank < world; ++dst_rank) {
                            if (dst_rank == rank) {
                                mbuf_recv[tail_base + rank] += 1;
                            } else {
                                ishmem_int_atomic_add(mbuf_recv + tail_base + rank, 1, dst_rank);
                            }
                        }
                    });
                }).wait_and_throw();

                ishmem_barrier_all();

                if (rank == 0) {
                    std::cout << "[rank " << rank << "] iter " << iter << "/" << total_iters << " done\n";
                    std::cout << std::flush;
                }
            }

            // Verify tails (each rank r adds 1 to tail[r] on every PE per iter)
            std::vector<int> host(total_alloc);
            queue.memcpy(host.data(), mbuf_recv, static_cast<size_t>(total_alloc) * sizeof(int)).wait_and_throw();
            for (int r = 0; r < world; ++r) {
                if (host[tail_base + r] != total_iters) {
                    std::cout << "  tail mismatch rank=" << r
                              << " expected=" << total_iters
                              << " got=" << host[tail_base + r] << "\n";
                    ++errors;
                }
            }
            if (count_buf) ishmem_free(count_buf);
            ishmem_free(mbuf_src);
            ishmem_free(mbuf_recv);
        }
    } else if (test_case == "dispatch_then_combine_blocking_put") {
        // Reproduces exact DeepEP flow:
        // Phase 1 (dispatch-like): NBI WG put → quiet + atomic → barrier
        // Phase 2 (combine-like): BLOCKING WG put → atomic → barrier
        // The combine PayloadKernel hangs on rank 1 in DeepEP.
        const int total_iters = 4;
        constexpr int wg_size = 32;

        const int tail_base = num_elems;
        const int total_alloc = tail_base + world;
        int* mbuf_recv = static_cast<int*>(ishmem_align(128, static_cast<size_t>(total_alloc) * sizeof(int)));
        int* mbuf_src  = static_cast<int*>(ishmem_align(128, static_cast<size_t>(total_alloc) * sizeof(int)));
        if (!mbuf_recv || !mbuf_src) {
            std::cerr << "[rank " << rank << "] ishmem_align failed\n";
            errors = 1;
        } else {
            queue.parallel_for(sycl::range<1>(total_alloc), [=](sycl::id<1> id) {
                const int i = static_cast<int>(id[0]);
                mbuf_recv[i] = 0;
                mbuf_src[i] = (rank + 1) * 100000 + i;
            }).wait_and_throw();
            ishmem_barrier_all();

            // Count exchange (blocking putmem, like DispatchCountExchangeKernel)
            int* count_buf = static_cast<int*>(ishmem_align(128, static_cast<size_t>(world) * sizeof(int)));
            if (count_buf) {
                queue.single_task([=]() { count_buf[rank] = num_elems; }).wait_and_throw();
                ishmem_barrier_all();
                queue.submit([&](sycl::handler& h) {
                    h.single_task([=]() {
                        for (int dst = 0; dst < world; ++dst) {
                            if (dst != rank)
                                ishmem_putmem(count_buf, count_buf, static_cast<size_t>(world) * sizeof(int), dst);
                        }
                    });
                }).wait_and_throw();
                ishmem_barrier_all();
            }

            // Phase 1: Dispatch-like (NBI puts + quiet + atomic)
            if (rank == 0) std::cout << "[rank 0] === DISPATCH PHASE ===\n" << std::flush;
            for (int iter = 0; iter < total_iters; ++iter) {
                queue.submit([&](sycl::handler& h) {
                    h.parallel_for(
                        sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(world) * wg_size),
                                          sycl::range<1>(wg_size)),
                        [=](sycl::nd_item<1> it) {
                            auto group = it.get_group();
                            const int dst_rank = static_cast<int>(it.get_group(0));
                            if (dst_rank != rank) {
                                ishmemx_putmem_nbi_work_group(
                                    mbuf_recv, mbuf_src,
                                    static_cast<size_t>(num_elems) * sizeof(int),
                                    dst_rank, group);
                            }
                        });
                }).wait_and_throw();
                queue.submit([&](sycl::handler& h) {
                    h.single_task([=]() {
                        ishmem_quiet();
                        for (int dst = 0; dst < world; ++dst) {
                            if (dst != rank)
                                ishmem_int_atomic_add(mbuf_recv + tail_base + rank, 1, dst);
                        }
                    });
                }).wait_and_throw();
                ishmem_barrier_all();
                if (rank == 0) std::cout << "[rank 0] dispatch iter " << iter << " done\n" << std::flush;
            }

            // Reset tails
            queue.single_task([=]() {
                for (int i = 0; i < world; ++i) mbuf_recv[tail_base + i] = 0;
            }).wait_and_throw();
            ishmem_barrier_all();

            // Phase 2: Combine-like (BLOCKING WG puts + atomic)
            if (rank == 0) std::cout << "[rank 0] === COMBINE PHASE ===\n" << std::flush;
            for (int iter = 0; iter < total_iters; ++iter) {
                // PayloadKernel: BLOCKING putmem_work_group (like DeepEP combine)
                queue.submit([&](sycl::handler& h) {
                    h.parallel_for(
                        sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(world) * wg_size),
                                          sycl::range<1>(wg_size)),
                        [=](sycl::nd_item<1> it) {
                            auto group = it.get_group();
                            const int src_rank = static_cast<int>(it.get_group(0));
                            if (src_rank == rank) {
                                const int local_id = static_cast<int>(it.get_local_id(0));
                                const int local_size = static_cast<int>(it.get_local_range(0));
                                for (int i = local_id; i < num_elems; i += local_size) {
                                    mbuf_recv[i] = mbuf_src[i];
                                }
                            } else {
                                // BLOCKING work-group put (like DeepEP combine)
                                ishmemx_putmem_work_group(
                                    mbuf_recv, mbuf_src,
                                    static_cast<size_t>(num_elems) * sizeof(int),
                                    src_rank, group);
                            }
                        });
                }).wait_and_throw();
                if (rank == 0) std::cout << "[rank 0] combine iter " << iter << " payload done\n" << std::flush;

                // TailKernel: atomic_add (no quiet needed, puts were blocking)
                queue.submit([&](sycl::handler& h) {
                    h.single_task([=]() {
                        for (int dst = 0; dst < world; ++dst) {
                            if (dst != rank)
                                ishmem_int_atomic_add(mbuf_recv + tail_base + rank, 1, dst);
                        }
                    });
                }).wait_and_throw();
                if (rank == 0) std::cout << "[rank 0] combine iter " << iter << " tail done\n" << std::flush;
                ishmem_barrier_all();
            }

            // Verify tails (combine phase only; dispatch tails were reset)
            // Each rank r sends atomic_add(tail[r], 1, dst) to all non-self dsts.
            // On PE X: tail[r] = total_iters for r != X (rank r sent to us), tail[X] = 0 (no self-send)
            std::vector<int> host(total_alloc);
            queue.memcpy(host.data(), mbuf_recv, static_cast<size_t>(total_alloc) * sizeof(int)).wait_and_throw();
            for (int r = 0; r < world; ++r) {
                int expected_tail = (r != rank) ? total_iters : 0;
                if (host[tail_base + r] != expected_tail) {
                    std::cout << "  tail mismatch rank=" << r
                              << " expected=" << expected_tail
                              << " got=" << host[tail_base + r] << "\n";
                    ++errors;
                }
            }
            if (count_buf) ishmem_free(count_buf);
            ishmem_free(mbuf_src);
            ishmem_free(mbuf_recv);
        }
    } else if (test_case == "multi_channel_nbi_quiet_atomic_split") {
        // Same as multi_channel_nbi_quiet_atomic but quiet and atomic_add in
        // SEPARATE kernels (3 kernels per iteration instead of 2).
        const int total_iters = 8;
        constexpr int wg_size = 32;

        const int tail_base = num_elems;
        const int total_alloc = tail_base + world;
        int* mbuf_recv = static_cast<int*>(ishmem_align(128, static_cast<size_t>(total_alloc) * sizeof(int)));
        int* mbuf_src  = static_cast<int*>(ishmem_align(128, static_cast<size_t>(total_alloc) * sizeof(int)));
        if (!mbuf_recv || !mbuf_src) {
            std::cerr << "[rank " << rank << "] ishmem_align failed (multi_channel_split)\n";
            errors = 1;
        } else {
            queue.parallel_for(sycl::range<1>(total_alloc), [=](sycl::id<1> id) {
                const int i = static_cast<int>(id[0]);
                mbuf_recv[i] = 0;
                mbuf_src[i] = (rank + 1) * 100000 + i;
            }).wait_and_throw();
            ishmem_barrier_all();

            // Warmup atomic
            queue.submit([&](sycl::handler& h) {
                h.single_task([=]() {
                    ishmem_int_atomic_add(mbuf_recv + tail_base + rank, 0, peer);
                });
            }).wait_and_throw();
            ishmem_barrier_all();

            queue.single_task([=]() {
                for (int i = 0; i < world; ++i) mbuf_recv[tail_base + i] = 0;
            }).wait_and_throw();
            ishmem_barrier_all();

            // Count exchange: blocking putmem
            int* count_buf2 = static_cast<int*>(ishmem_align(128, static_cast<size_t>(world) * sizeof(int)));
            if (count_buf2) {
                queue.single_task([=]() { count_buf2[rank] = num_elems; }).wait_and_throw();
                ishmem_barrier_all();
                queue.submit([&](sycl::handler& h) {
                    h.single_task([=]() {
                        for (int dst = 0; dst < world; ++dst) {
                            if (dst != rank)
                                ishmem_putmem(count_buf2, count_buf2, static_cast<size_t>(world) * sizeof(int), dst);
                        }
                    });
                }).wait_and_throw();
                ishmem_barrier_all();
            }

            for (int iter = 0; iter < total_iters; ++iter) {
                // Kernel A: NBI work-group put (payload)
                queue.submit([&](sycl::handler& h) {
                    h.parallel_for(
                        sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(world) * wg_size),
                                          sycl::range<1>(wg_size)),
                        [=](sycl::nd_item<1> it) {
                            auto group = it.get_group();
                            const int dst_rank = static_cast<int>(it.get_group(0));
                            if (dst_rank == rank) {
                                const int local_id = static_cast<int>(it.get_local_id(0));
                                const int local_size = static_cast<int>(it.get_local_range(0));
                                for (int i = local_id; i < num_elems; i += local_size) {
                                    mbuf_recv[i] = mbuf_src[i];
                                }
                            } else {
                                ishmemx_putmem_nbi_work_group(
                                    mbuf_recv, mbuf_src,
                                    static_cast<size_t>(num_elems) * sizeof(int),
                                    dst_rank, group);
                            }
                        });
                }).wait_and_throw();

                // Kernel B: quiet only
                queue.submit([&](sycl::handler& h) {
                    h.single_task([=]() { ishmem_quiet(); });
                }).wait_and_throw();

                // Kernel C: atomic_add only (cross-kernel from NBI puts)
                queue.submit([&](sycl::handler& h) {
                    h.single_task([=]() {
                        for (int dst_rank = 0; dst_rank < world; ++dst_rank) {
                            if (dst_rank == rank) {
                                mbuf_recv[tail_base + rank] += 1;
                            } else {
                                ishmem_int_atomic_add(mbuf_recv + tail_base + rank, 1, dst_rank);
                            }
                        }
                    });
                }).wait_and_throw();

                ishmem_barrier_all();

                if (rank == 0) {
                    std::cout << "[rank " << rank << "] iter " << iter << "/" << total_iters << " done\n";
                    std::cout << std::flush;
                }
            }

            std::vector<int> host(total_alloc);
            queue.memcpy(host.data(), mbuf_recv, static_cast<size_t>(total_alloc) * sizeof(int)).wait_and_throw();
            for (int r = 0; r < world; ++r) {
                if (host[tail_base + r] != total_iters) {
                    std::cout << "  tail mismatch rank=" << r
                              << " expected=" << total_iters
                              << " got=" << host[tail_base + r] << "\n";
                    ++errors;
                }
            }

            if (count_buf2) ishmem_free(count_buf2);
            ishmem_free(mbuf_src);
            ishmem_free(mbuf_recv);
        }
    // ========================== Scalar device API hang/stability tests ==========================
    // These tests exercise scalar (non-work-group) iSHMEM device APIs that are known to hang or
    // be unstable with IBGDA direct-doorbell transport. They serve as regression tests: if a
    // future iSHMEM release fixes scalar device APIs, these tests should start passing, enabling
    // the removal of work-group API workarounds in the LL kernels.

    } else if (test_case == "scalar_putmem_single_task") {
        // Simplest scalar put: single work-item puts payload to peer.
        // Known behavior: HANGS with IBGDA direct-doorbell.
        init_buffers(queue, recv, src, num_elems, rank, false);
        ishmem_barrier_all();
        queue
            .submit([&](sycl::handler& h) {
                h.single_task([=]() {
                    ishmem_putmem(recv, src, static_cast<size_t>(num_elems) * sizeof(int), peer);
                });
            })
            .wait_and_throw();
        ishmem_barrier_all();
        errors = count_errors(queue, recv, num_elems, (sender + 1) * 100000);

    } else if (test_case == "scalar_putmem_nbi_quiet_single_task") {
        // NBI put + scalar quiet in single_task.
        // Known behavior: HANGS — ishmem_quiet() never completes.
        init_buffers(queue, recv, src, num_elems, rank, false);
        ishmem_barrier_all();
        queue
            .submit([&](sycl::handler& h) {
                h.single_task([=]() {
                    ishmem_putmem_nbi(recv, src, static_cast<size_t>(num_elems) * sizeof(int), peer);
                    ishmem_quiet();
                });
            })
            .wait_and_throw();
        ishmem_barrier_all();
        errors = count_errors(queue, recv, num_elems, (sender + 1) * 100000);

    } else if (test_case == "scalar_int_put_single_task") {
        // Typed scalar int put (blocking) in single_task.
        // Tests whether ishmem_int_put (typed API) has the same hang as ishmem_putmem.
        init_buffers(queue, recv, src, num_elems, rank, false);
        ishmem_barrier_all();
        queue
            .submit([&](sycl::handler& h) {
                h.single_task([=]() {
                    ishmem_int_put(recv, src, static_cast<size_t>(num_elems), peer);
                });
            })
            .wait_and_throw();
        ishmem_barrier_all();
        errors = count_errors(queue, recv, num_elems, (sender + 1) * 100000);

    } else if (test_case == "scalar_quiet_after_nbi_single_task") {
        // Separate kernels: NBI put in one single_task, quiet in another.
        // Tests whether splitting put and quiet across kernel launches helps.
        init_buffers(queue, recv, src, num_elems, rank, false);
        ishmem_barrier_all();
        queue
            .submit([&](sycl::handler& h) {
                h.single_task([=]() {
                    ishmem_putmem_nbi(recv, src, static_cast<size_t>(num_elems) * sizeof(int), peer);
                });
            })
            .wait_and_throw();
        queue
            .submit([&](sycl::handler& h) {
                h.single_task([=]() {
                    ishmem_quiet();
                });
            })
            .wait_and_throw();
        ishmem_barrier_all();
        errors = count_errors(queue, recv, num_elems, (sender + 1) * 100000);

    } else if (test_case == "scalar_fence_single_task") {
        // Scalar ishmem_fence() in single_task — tests ordering primitive alone.
        init_buffers(queue, recv, src, num_elems, rank, false);
        ishmem_barrier_all();
        queue
            .submit([&](sycl::handler& h) {
                h.single_task([=]() {
                    ishmem_putmem_nbi(recv, src, static_cast<size_t>(num_elems) * sizeof(int), peer);
                    ishmem_fence();
                    // After fence, put a flag to signal ordering
                    ishmem_int_put(recv + flag_idx, src + flag_idx, 1, peer);
                });
            })
            .wait_and_throw();
        ishmem_barrier_all();
        errors = count_errors(queue, recv, num_elems, (sender + 1) * 100000);

    } else if (test_case == "scalar_barrier_all_device") {
        // Scalar ishmem_barrier_all() called from device (single_task).
        // Tests whether device-side barrier_all hangs like other scalar APIs.
        init_buffers(queue, recv, src, num_elems, rank, false);
        ishmem_barrier_all();  // host barrier
        queue
            .submit([&](sycl::handler& h) {
                h.single_task([=]() {
                    ishmem_putmem(recv, src, static_cast<size_t>(num_elems) * sizeof(int), peer);
                    ishmem_barrier_all();
                });
            })
            .wait_and_throw();
        errors = count_errors(queue, recv, num_elems, (sender + 1) * 100000);

    } else if (test_case == "scalar_sync_all_device") {
        // Scalar ishmem_sync_all() called from device (single_task).
        init_buffers(queue, recv, src, num_elems, rank, false);
        ishmem_barrier_all();
        queue
            .submit([&](sycl::handler& h) {
                h.single_task([=]() {
                    ishmem_putmem_nbi(recv, src, static_cast<size_t>(num_elems) * sizeof(int), peer);
                    ishmem_quiet();
                    ishmem_sync_all();
                });
            })
            .wait_and_throw();
        errors = count_errors(queue, recv, num_elems, (sender + 1) * 100000);

    } else if (test_case == "scalar_putmem_parallel_for") {
        // Multiple work-items each doing independent scalar puts.
        // Tests whether concurrent scalar puts from parallel_for(range) hang.
        init_buffers(queue, recv, src, num_elems, rank, false);
        ishmem_barrier_all();
        queue
            .parallel_for(sycl::range<1>(num_elems),
                          [=](sycl::id<1> id) {
                              const int i = static_cast<int>(id[0]);
                              ishmem_int_put(recv + i, src + i, 1, peer);
                          })
            .wait_and_throw();
        ishmem_barrier_all();
        errors = count_errors(queue, recv, num_elems, (sender + 1) * 100000);

    } else if (test_case == "scalar_putmem_nd_range_no_wg_api") {
        // nd_range kernel where each work-item does scalar puts (NOT using WG API).
        // Tests whether scalar APIs work inside nd_range kernels at all.
        const int wg_size = 32;
        const int n_wgs = (num_elems + wg_size - 1) / wg_size;
        init_buffers(queue, recv, src, num_elems, rank, false);
        ishmem_barrier_all();
        queue
            .submit([&](sycl::handler& h) {
                h.parallel_for(
                    sycl::nd_range<1>(sycl::range<1>(n_wgs * wg_size), sycl::range<1>(wg_size)),
                    [=](sycl::nd_item<1> item) {
                        const int i = static_cast<int>(item.get_global_id(0));
                        if (i < num_elems) {
                            ishmem_int_put(recv + i, src + i, 1, peer);
                        }
                    });
            })
            .wait_and_throw();
        ishmem_barrier_all();
        errors = count_errors(queue, recv, num_elems, (sender + 1) * 100000);

    } else if (test_case == "scalar_get_single_task") {
        // Scalar ishmem_int_get in single_task — tests get path.
        init_buffers(queue, recv, src, num_elems, rank, false);
        ishmem_barrier_all();
        // src on peer has (peer+1)*100000 + i, which is (sender+1)*100000 + i for us reading from peer=sender
        // Actually peer = (rank+1)%world, so we GET from peer's src
        queue
            .submit([&](sycl::handler& h) {
                h.single_task([=]() {
                    // Get peer's src into our recv
                    ishmem_int_get(recv, src, static_cast<size_t>(num_elems), peer);
                });
            })
            .wait_and_throw();
        ishmem_barrier_all();
        // peer's src has (peer+1)*100000 + i
        errors = count_constant_errors(queue, recv, num_elems, (peer + 1) * 100000);
        // Check element 0 specifically
        if (errors > 0) {
            std::vector<int> host(num_elems);
            queue.memcpy(host.data(), recv, static_cast<size_t>(num_elems) * sizeof(int)).wait_and_throw();
            std::cout << "  [rank " << rank << "] recv[0]=" << host[0]
                      << " expected=" << (peer + 1) * 100000 << "\n";
        }

    } else if (test_case == "scalar_putmem_large_single_task") {
        // Large payload scalar put in single_task.
        // Tests whether the hang is size-dependent.
        const int large_elems = 4096;  // 16KB payload
        int* large_recv = static_cast<int*>(ishmem_align(128, static_cast<size_t>(large_elems) * sizeof(int)));
        int* large_src = static_cast<int*>(ishmem_align(128, static_cast<size_t>(large_elems) * sizeof(int)));
        if (large_recv && large_src) {
            init_buffers(queue, large_recv, large_src, large_elems, rank, false);
            ishmem_barrier_all();
            queue
                .submit([&](sycl::handler& h) {
                    h.single_task([=]() {
                        ishmem_putmem(large_recv, large_src, static_cast<size_t>(large_elems) * sizeof(int), peer);
                    });
                })
                .wait_and_throw();
            ishmem_barrier_all();
            errors = count_errors(queue, large_recv, large_elems, (sender + 1) * 100000);
            ishmem_free(large_src);
            ishmem_free(large_recv);
        } else {
            std::cout << "[rank " << rank << "] large alloc failed, skipping\n";
        }

    } else if (test_case == "scalar_p_single_task") {
        // Scalar ishmem_int_p (single element put) in single_task.
        // Tests the smallest possible scalar put operation.
        init_buffers(queue, recv, src, num_elems, rank, false);
        ishmem_barrier_all();
        queue
            .submit([&](sycl::handler& h) {
                h.single_task([=]() {
                    for (int i = 0; i < num_elems; ++i) {
                        ishmem_int_p(recv + i, src[i], peer);
                    }
                });
            })
            .wait_and_throw();
        ishmem_barrier_all();
        errors = count_errors(queue, recv, num_elems, (sender + 1) * 100000);

    } else if (test_case == "scalar_g_single_task") {
        // Scalar ishmem_int_g (single element get) in single_task.
        init_buffers(queue, recv, src, num_elems, rank, false);
        ishmem_barrier_all();
        queue
            .submit([&](sycl::handler& h) {
                h.single_task([=]() {
                    for (int i = 0; i < num_elems; ++i) {
                        recv[i] = ishmem_int_g(src + i, peer);
                    }
                });
            })
            .wait_and_throw();
        ishmem_barrier_all();
        errors = count_constant_errors(queue, recv, num_elems, (peer + 1) * 100000);

    } else if (test_case == "scalar_atomic_fetch_add_single_task") {
        // Scalar ishmem_int_atomic_fetch_add in single_task.
        // Tests whether fetch-returning atomics also hang.
        init_buffers(queue, recv, src, num_elems, rank, true);
        ishmem_barrier_all();
        auto* fetch_results = sycl::malloc_device<int>(num_elems, queue);
        queue
            .submit([&](sycl::handler& h) {
                h.single_task([=]() {
                    for (int i = 0; i < num_elems; ++i) {
                        fetch_results[i] = ishmem_int_atomic_fetch_add(recv + i, rank + 1, peer);
                    }
                });
            })
            .wait_and_throw();
        ishmem_barrier_all();
        // recv on each PE should have been incremented by sender+1
        errors = count_constant_errors(queue, recv, num_elems, sender + 1);
        sycl::free(fetch_results, queue);

    } else if (test_case == "scalar_quiet_no_preceding_put") {
        // Scalar ishmem_quiet() with NO preceding put — should be a no-op.
        // Known behavior: PASSES (no actual work to quiet).
        ishmem_barrier_all();
        queue
            .submit([&](sycl::handler& h) {
                h.single_task([=]() {
                    ishmem_quiet();
                });
            })
            .wait_and_throw();
        ishmem_barrier_all();
        errors = 0;

    } else if (test_case == "scalar_put_multi_pe_single_task") {
        // Single work-item puts to ALL other PEs sequentially.
        // Tests whether putting to multiple destinations from one work-item hangs.
        init_buffers(queue, recv, src, num_elems, rank, false);
        ishmem_barrier_all();
        queue
            .submit([&](sycl::handler& h) {
                h.single_task([=]() {
                    for (int dst = 0; dst < world; ++dst) {
                        if (dst == rank) continue;
                        ishmem_putmem(recv, src, static_cast<size_t>(num_elems) * sizeof(int), dst);
                    }
                });
            })
            .wait_and_throw();
        ishmem_barrier_all();
        // Last writer wins (undefined which sender's data we see, but we should have SOME valid data)
        // Just check we didn't hang and data is not sentinel
        std::vector<int> host(num_elems);
        queue.memcpy(host.data(), recv, static_cast<size_t>(num_elems) * sizeof(int)).wait_and_throw();
        for (int i = 0; i < num_elems; ++i) {
            if (host[i] == kSentinel) {
                ++errors;
            }
        }

    } else if (test_case == "scalar_nbi_quiet_repeat_single_task") {
        // Repeated NBI+quiet cycles in a single kernel (simulates iteration loop).
        // Tests stability of scalar quiet over multiple rounds.
        const int iters = 5;
        init_buffers(queue, recv, src, num_elems, rank, false);
        ishmem_barrier_all();
        queue
            .submit([&](sycl::handler& h) {
                h.single_task([=]() {
                    for (int iter = 0; iter < iters; ++iter) {
                        ishmem_putmem_nbi(recv, src, static_cast<size_t>(num_elems) * sizeof(int), peer);
                        ishmem_quiet();
                    }
                });
            })
            .wait_and_throw();
        ishmem_barrier_all();
        errors = count_errors(queue, recv, num_elems, (sender + 1) * 100000);

    } else {
        ran = false;
        if (rank == 0) {
            usage(argv[0]);
        }
    }

    std::cout << "[rank " << rank << "] case=" << test_case << " num_elems=" << num_elems << " errors=" << errors;
    if (test_case == "ll_ptr_device") {
        std::cout << " ptr_was_null=" << (ptr_was_null ? 1 : 0);
    }
    std::cout << (errors == 0 && ran ? " PASS" : " FAIL") << "\n";

    ishmem_free(src);
    ishmem_free(recv);
    ishmem_finalize();
    return (errors == 0 && ran) ? 0 : 1;
}
