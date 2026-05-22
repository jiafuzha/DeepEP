#include <ishmem.h>
#include <ishmemx.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <string>
#include <sycl/sycl.hpp>
#include <vector>

namespace {

constexpr int kSentinel = -777777;

void usage(const char* argv0) {
    std::cerr << "Usage: " << argv0 << " --case CASE [--num-elems N]\n"
              << "Cases:\n"
              << "  normal_putmem_blocking\n"
              << "  normal_putmem_nbi_quiet\n"
              << "  normal_putmem_parallel_work_items\n"
              << "  ll_int_put_nbi_quiet\n"
              << "  ll_putmem_nbi_atomic_flag\n"
              << "  atomic_add_remote\n"
              << "  atomic_add_remote_many\n"
              << "  atomic_add_all_pes\n"
              << "  ll_ptr_device\n"
              << "  normal_sync_all_device\n"
              << "  ll_barrier_work_group\n"
              << "  quiet_empty\n"
              << "  intranode_no_mapped_ishmem_api\n";
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
    }

    ishmem_init();
    sycl::queue queue;
    const int rank = ishmem_my_pe();
    const int world = ishmem_n_pes();

    if (world != 2) {
        if (rank == 0) {
            std::cerr << "expected 2 PEs, got " << world << "\n";
        }
        ishmem_finalize();
        return 2;
    }

    const int peer = 1 - rank;
    const int flag_idx = num_elems;
    const int alloc_elems = num_elems + 1;
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

    if (test_case == "intranode_no_mapped_ishmem_api") {
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
        errors = count_errors(queue, recv, num_elems, (peer + 1) * 100000);
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
        errors = count_errors(queue, recv, num_elems, (peer + 1) * 100000);
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
        errors = count_errors(queue, recv, num_elems, (peer + 1) * 100000);
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
        errors = count_errors(queue, recv, num_elems, (peer + 1) * 100000);
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
        errors = count_errors(queue, recv, num_elems, (peer + 1) * 100000);
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
        errors = count_constant_errors(queue, recv, num_elems, peer + 1);
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
                              : count_errors(queue, recv, num_elems, (peer + 1) * 100000);
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
        errors = count_errors(queue, recv, num_elems, (peer + 1) * 100000);
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
        errors = count_errors(queue, recv, num_elems, (peer + 1) * 100000);
    } else if (test_case == "quiet_empty") {
        init_buffers(queue, recv, src, num_elems, rank, false);
        ishmem_barrier_all();
        queue.submit([&](sycl::handler& h) { h.single_task([=]() { ishmem_quiet(); }); }).wait_and_throw();
        ishmem_barrier_all();
        errors = count_constant_errors(queue, recv, num_elems, kSentinel);
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
