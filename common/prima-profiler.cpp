// prima.cpp: device and model profiling

#include "prima.h"

#include "common.h"
#include "log.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <thread>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#if defined(__linux__)
#include <sys/sysinfo.h>
#endif

#if defined(__APPLE__) && defined(__MACH__)
#include <mach/mach.h>
#include <sys/sysctl.h>
#endif

static double time_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void copy_str(char * dst, size_t n, const char * src) {
    snprintf(dst, n, "%s", src ? src : "");
}

//
// model statistics
//

prima_model_stats prima_model_stats_from_file(const std::string & path, int64_t n_ctx, int64_t n_predict) {
    prima_model_stats ms;

    std::vector<std::string> paths = { path };

    // the KV pairs are read from the first split
    ggml_context * meta = nullptr;
    gguf_context * gctx = gguf_init_from_file(path.c_str(), { /*.no_alloc =*/ true, /*.ctx =*/ &meta });
    if (!gctx) {
        throw std::runtime_error("prima: failed to read " + path);
    }

    const int64_t kid_arch = gguf_find_key(gctx, "general.architecture");
    const std::string arch = kid_arch >= 0 ? gguf_get_val_str(gctx, kid_arch) : "llama";

    auto get_u32 = [&](const std::string & key, int64_t def) -> int64_t {
        const int64_t kid = gguf_find_key(gctx, (arch + "." + key).c_str());
        if (kid < 0) {
            return def;
        }
        switch (gguf_get_kv_type(gctx, kid)) {
            case GGUF_TYPE_UINT32: return gguf_get_val_u32(gctx, kid);
            case GGUF_TYPE_INT32:  return gguf_get_val_i32(gctx, kid);
            case GGUF_TYPE_UINT64: return (int64_t) gguf_get_val_u64(gctx, kid);
            default:               return def;
        }
    };

    ms.n_layer = get_u32("block_count",          0);
    ms.n_embd  = get_u32("embedding_length",     0);
    ms.n_head  = get_u32("attention.head_count", 0);
    ms.n_ff    = get_u32("feed_forward_length",  0);

    const int64_t n_head_kv_hp  = get_u32("attention.head_count_kv", ms.n_head);
    const int64_t n_expert_used = get_u32("expert_used_count", 0);

    {
        const int64_t kid = gguf_find_key(gctx, "split.count");
        const int n_split = kid >= 0 ? (int) gguf_get_val_u16(gctx, kid) : 1;
        if (n_split > 1) {
            char prefix[4096] = {};
            if (!llama_split_prefix(prefix, sizeof(prefix), path.c_str(), 1, n_split)) {
                throw std::runtime_error("prima: invalid split file name " + path);
            }
            for (int i = 2; i <= n_split; ++i) {
                char split_path[4096] = {};
                llama_split_path(split_path, sizeof(split_path), prefix, i, n_split);
                paths.push_back(split_path);
            }
        }
    }

    if (ms.n_layer <= 0 || ms.n_embd <= 0 || ms.n_head <= 0) {
        gguf_free(gctx);
        ggml_free(meta);
        throw std::runtime_error("prima: the model has no block_count, embedding_length or attention.head_count");
    }

    const int64_t n_kv = n_predict > 0 ? std::min(n_ctx, GGML_PAD(n_predict, 256)) : n_ctx;

    int64_t nb_layers  = 0;
    int64_t tok_flops  = 0;
    int     tok_type   = GGML_TYPE_F32;
    bool    has_output = false;

    std::vector<int64_t> layer_flops(PRIMA_N_TYPES, 0);

    auto add = [](int64_t * flops, int type, double n) {
        if (type >= 0 && type < PRIMA_N_TYPES) {
            flops[type] += (int64_t) n;
        }
    };

    for (size_t ip = 0; ip < paths.size(); ++ip) {
        if (ip > 0) {
            gguf_free(gctx);
            ggml_free(meta);
            meta = nullptr;
            gctx = gguf_init_from_file(paths[ip].c_str(), { /*.no_alloc =*/ true, /*.ctx =*/ &meta });
            if (!gctx) {
                throw std::runtime_error("prima: failed to read " + paths[ip]);
            }
        }

        for (int64_t i = 0; i < gguf_get_n_tensors(gctx); ++i) {
            const char * name = gguf_get_tensor_name(gctx, i);
            const ggml_tensor * t = ggml_get_tensor(meta, name);
            if (!t) {
                continue;
            }

            const int     type   = t->type;
            const int64_t nbytes = ggml_nbytes(t);
            const std::string sname = name;

            int il = -1;
            int n  = 0;
            if (sscanf(name, "blk.%d.%n", &il, &n) == 1 && n > 0) {
                const std::string suffix = sname.substr(n);
                nb_layers += nbytes;

                if (suffix == "attn_k.weight") {
                    ms.n_embd_k_gqa = t->ne[1];
                }
                if (suffix == "attn_v.weight") {
                    ms.n_embd_v_gqa = t->ne[1];
                }

                if (ggml_n_dims(t) == 1) {
                    // norms and biases
                    add(layer_flops.data(), type, t->ne[0]);
                    if (suffix.find("norm") != std::string::npos) {
                        add(layer_flops.data(), GGML_TYPE_F32, 4.0*t->ne[0]);
                    }
                } else if (ggml_n_dims(t) == 3 && suffix.find("_exps") != std::string::npos) {
                    // experts, only the used experts run
                    const int64_t n_used = n_expert_used > 0 ? n_expert_used : t->ne[2];
                    add(layer_flops.data(), type, 2.0*t->ne[0]*t->ne[1]*n_used);
                } else {
                    add(layer_flops.data(), type, 2.0*t->ne[0]*t->ne[1]);
                }
            } else if (sname == "token_embd.weight") {
                ms.nb_input = nbytes;
                ms.n_vocab  = t->ne[1];
                tok_flops   = 2*t->ne[0]*t->ne[1];
                tok_type    = type;
            } else if (sname == "output.weight") {
                has_output = true;
                ms.nb_output += nbytes;
                add(ms.output_flops, type, 2.0*t->ne[0]*t->ne[1]);
                add(ms.output_flops, GGML_TYPE_F32, 5.0*t->ne[1]); // softmax
            } else if (sname == "output_norm.weight") {
                ms.nb_output += nbytes;
                add(ms.output_flops, type, t->ne[0]);
                add(ms.output_flops, GGML_TYPE_F32, 4.0*t->ne[0]);
            }
        }
    }

    gguf_free(gctx);
    ggml_free(meta);

    if (!has_output) {
        // tied embeddings, the output head uses the input embeddings
        ms.nb_output += ms.nb_input;
        add(ms.output_flops, tok_type, (double) tok_flops);
    }

    if (ms.n_embd_k_gqa == 0) {
        ms.n_embd_k_gqa = ms.n_embd;
    }
    if (ms.n_embd_v_gqa == 0) {
        ms.n_embd_v_gqa = ms.n_embd;
    }
    if (ms.n_ff == 0) {
        ms.n_ff = 4*ms.n_embd;
    }

    for (int t = 0; t < PRIMA_N_TYPES; ++t) {
        ms.layer_flops[t] = layer_flops[t] / ms.n_layer;
    }
    ms.nb_layer = nb_layers / ms.n_layer;

    // attention with the KV cache (F16), rope, softmax and the other element-wise ops
    const int64_t n_head_kv = std::max<int64_t>(1, n_head_kv_hp);
    const int64_t head_k    = ms.n_embd_k_gqa / n_head_kv;
    const int64_t head_v    = ms.n_embd_v_gqa / n_head_kv;

    ms.layer_flops[GGML_TYPE_F16] += 2*head_k*ms.n_head*n_kv + 2*head_v*ms.n_head*n_kv;
    ms.layer_flops[GGML_TYPE_F32] += 7*ms.n_head*n_kv + (int64_t)(2.5*(ms.n_embd + ms.n_embd_k_gqa)) + 2*ms.n_embd + 9*ms.n_ff;

    return ms;
}

//
// device profiling
//

static const char * device_os() {
#if defined(_WIN32)
    return "Windows";
#elif defined(__ANDROID__)
    return "Android";
#elif defined(__APPLE__) && defined(__MACH__)
    return "macOS";
#elif defined(__linux__)
    // Termux on Android reports Linux to the build
    return access("/data/data/com.termux/files/usr/bin", F_OK) == 0 ? "Android" : "Linux";
#else
    return "Unknown";
#endif
}

#if defined(__linux__)
static std::map<std::string, uint64_t> read_meminfo() {
    std::map<std::string, uint64_t> res;
    std::ifstream f("/proc/meminfo");
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream iss(line);
        std::string key;
        uint64_t kb = 0;
        if (iss >> key >> kb) {
            if (!key.empty() && key.back() == ':') {
                key.pop_back();
            }
            res[key] = kb * 1024;
        }
    }
    return res;
}

static uint64_t read_u64_file(const char * path) {
    std::ifstream f(path);
    std::string s;
    if (!(f >> s)) {
        return 0;
    }
    try {
        return std::stoull(s);
    } catch (...) {
        return 0; // e.g. "max"
    }
}
#endif

// total or available physical memory in bytes
static uint64_t device_physical_memory(bool available) {
    uint64_t mem = 0;

#if defined(_WIN32)
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    mem = available ? status.ullAvailPhys : status.ullTotalPhys;
#elif defined(__linux__)
    auto mi = read_meminfo();
    // keep 20% of the available memory as a margin
    mem = available ? (uint64_t) (mi["MemAvailable"] * 0.8) : mi["MemTotal"];

    // in a container the cgroup (v2) limit can be lower
    const uint64_t cg_max = read_u64_file("/sys/fs/cgroup/memory.max");
    if (cg_max > 0) {
        const uint64_t cg_cur = read_u64_file("/sys/fs/cgroup/memory.current");
        const uint64_t cg_mem = available ? (cg_max > cg_cur ? cg_max - cg_cur : 0) : cg_max;
        mem = std::min(mem, cg_mem);
    }
#elif defined(__APPLE__) && defined(__MACH__)
    uint64_t total = 0;
    size_t len = sizeof(total);
    int mib[2] = { CTL_HW, HW_MEMSIZE };
    if (sysctl(mib, 2, &total, &len, NULL, 0) != 0) {
        return 0;
    }
    if (!available) {
        return total;
    }

    vm_statistics64_data_t vm;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t) &vm, &count) == KERN_SUCCESS) {
        const uint64_t page = (uint64_t) sysconf(_SC_PAGESIZE);

        int is_arm64 = 0;
        size_t sz = sizeof(is_arm64);
        const bool uma = sysctlbyname("hw.optional.arm64", &is_arm64, &sz, NULL, 0) == 0 && is_arm64;

        mem = (uint64_t) (vm.free_count + vm.inactive_count + vm.purgeable_count) * page;
        // some active pages can be compressed (empirical values from prima.cpp)
        mem += (uint64_t) (vm.active_count * (uma ? 0.1 : 0.5) * page);
        if (!uma) {
            mem += (uint64_t) (vm.speculative_count + vm.compressor_page_count) * page;
        }
    }
#else
    GGML_UNUSED(available);
#endif

    return mem;
}

static uint64_t device_swap_memory(bool available) {
    uint64_t mem = 0;

#if defined(__linux__)
    auto mi = read_meminfo();
    mem = available ? mi["SwapFree"] : mi["SwapTotal"];
#elif defined(__APPLE__) && defined(__MACH__)
    int mib[2] = { CTL_VM, VM_SWAPUSAGE };
    struct xsw_usage swap;
    size_t len = sizeof(swap);
    if (sysctl(mib, 2, &swap, &len, NULL, 0) == 0) {
        mem = available ? swap.xsu_avail : swap.xsu_total;
    }
#else
    GGML_UNUSED(available);
#endif

    return mem;
}

// used memory that the OS can swap out or compress to make room for the model
static uint64_t device_swappable_memory() {
#if defined(__APPLE__) && defined(__MACH__)
    vm_statistics64_data_t vm;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t) &vm, &count) == KERN_SUCCESS) {
        return (uint64_t) vm.internal_page_count * (uint64_t) sysconf(_SC_PAGESIZE);
    }
    return 0;
#elif defined(__linux__)
    if (strcmp(device_os(), "Android") != 0) {
        return 0;
    }
    auto mi = read_meminfo();
    return mi["Active(anon)"] + mi["Inactive(anon)"];
#else
    return 0;
#endif
}

static float device_memory_bw(int n_threads) {
    const size_t size = 256ull*1024*1024;
    std::vector<char> data(size, 1);

    std::vector<double> res(n_threads, 0.0);
    std::vector<std::thread> threads;

    for (int i = 0; i < n_threads; ++i) {
        threads.emplace_back([&, i]() {
            const size_t chunk = size / n_threads;
            const char * p = data.data() + i*chunk;
            volatile char sum = 0;
            const double t0 = time_s();
            for (size_t j = 0; j < chunk; j += 64) {
                sum += p[j];
            }
            res[i] = chunk / (time_s() - t0) / 1e9;
            GGML_UNUSED(sum);
        });
    }
    for (auto & t : threads) {
        t.join();
    }

    double bw = 0.0;
    for (double r : res) {
        bw += r;
    }
    return (float) bw;
}

// read speed of the model file without the page cache, in GB/s
static void device_disk_bw(const std::string & path, float & seq_bw, float & rnd_bw) {
    seq_bw = 0.0f;
    rnd_bw = 0.0f;

#if defined(_WIN32)
    GGML_UNUSED(path);
    seq_bw = rnd_bw = 1.0f;
#else
    bool direct = false;
    int fd = -1;
#if defined(O_DIRECT)
    fd = open(path.c_str(), O_RDONLY | O_DIRECT);
    direct = fd >= 0;
#endif
    if (fd < 0) {
        fd = open(path.c_str(), O_RDONLY);
    }
    if (fd < 0) {
        LOG_WRN("%s: failed to open %s, assume 1 GB/s\n", __func__, path.c_str());
        seq_bw = rnd_bw = 1.0f;
        return;
    }
#if defined(__APPLE__)
    fcntl(fd, F_NOCACHE, 1);
    direct = true;
#endif
#if defined(POSIX_FADV_DONTNEED)
    if (!direct) {
        posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    }
#endif

    struct stat st;
    fstat(fd, &st);
    const size_t file_size = (size_t) st.st_size;

    const size_t blk_seq = 1024*1024;
    const size_t blk_rnd = 4096;

    void * buf = nullptr;
    if (posix_memalign(&buf, 4096, blk_seq) != 0) {
        close(fd);
        return;
    }

    // sequential: up to 512 MiB or 1 second
    {
        size_t n_read = 0;
        const double t0 = time_s();
        while (n_read + blk_seq <= std::min(file_size, (size_t) 512*1024*1024) && time_s() - t0 < 1.0) {
            const ssize_t r = pread(fd, buf, blk_seq, (off_t) n_read);
            if (r <= 0) {
                break;
            }
            n_read += (size_t) r;
        }
        const double dt = time_s() - t0;
        seq_bw = dt > 0.0 && n_read > 0 ? (float) (n_read / dt / 1e9) : 0.0f;
    }

    // random 4 KiB reads: up to 0.5 seconds
    if (file_size > blk_rnd) {
        std::mt19937_64 rng(42);
        std::uniform_int_distribution<size_t> dist(0, file_size / blk_rnd - 1);

        size_t n_read = 0;
        const double t0 = time_s();
        for (int i = 0; i < 20000 && time_s() - t0 < 0.5; ++i) {
            const ssize_t r = pread(fd, buf, blk_rnd, (off_t) (dist(rng) * blk_rnd));
            if (r <= 0) {
                break;
            }
            n_read += (size_t) r;
        }
        const double dt = time_s() - t0;
        rnd_bw = dt > 0.0 && n_read > 0 ? (float) (n_read / dt / 1e9) : 0.0f;
    }

    free(buf);
    close(fd);

    if (!direct) {
        LOG_WRN("%s: direct I/O is not available, the disk speed can include the page cache\n", __func__);
    }
#endif
}

// run a graph built by fn on the device and return the time of one run in seconds (0 if not supported)
template <typename F>
static double bench_graph(ggml_backend_dev_t dev, int n_threads, size_t n_tensors, F && build) {
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (!backend) {
        return 0.0;
    }

    if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        auto * reg = ggml_backend_dev_backend_reg(dev);
        auto * set_n_threads = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        if (set_n_threads) {
            set_n_threads(backend, n_threads);
        }
    }

    ggml_init_params ip = {
        /*.mem_size   =*/ ggml_tensor_overhead()*(n_tensors + 8) + ggml_graph_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx_w = ggml_init(ip);
    ggml_context * ctx_g = ggml_init(ip);

    ggml_cgraph * gf = ggml_new_graph(ctx_g);
    build(ctx_w, ctx_g, gf);

    double res = 0.0;

    bool supported = true;
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        supported = supported && ggml_backend_supports_op(backend, ggml_graph_node(gf, i));
    }

    ggml_backend_buffer_t buf = supported ? ggml_backend_alloc_ctx_tensors(ctx_w, backend) : nullptr;
    ggml_gallocr_t galloc = nullptr;

    if (buf) {
        // the values do not change the speed
        ggml_backend_buffer_clear(buf, 0);

        galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (ggml_gallocr_alloc_graph(galloc, gf)) {
            ggml_backend_graph_compute(backend, gf); // warmup

            int n_run = 0;
            const double t0 = time_s();
            do {
                ggml_backend_graph_compute(backend, gf);
                n_run++;
            } while (time_s() - t0 < 0.05 && n_run < 100);
            res = (time_s() - t0) / n_run;
        }
    }

    ggml_gallocr_free(galloc);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx_g);
    ggml_free(ctx_w);
    ggml_backend_free(backend);

    return res;
}

// matmul speed of a [n, n] matrix of type t with a [n, n] F32 matrix, in GFLOPS
static float device_gflops(ggml_backend_dev_t dev, ggml_type type, int64_t n_embd, int n_threads) {
    int64_t n = std::min<int64_t>(n_embd, 4096);
    if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        n /= 8; // keep the test short on the CPU
    }
    const int64_t bs = ggml_blck_size(type);
    n = std::max<int64_t>(bs, n / bs * bs);

    const double t = bench_graph(dev, n_threads, 2, [&](ggml_context * ctx_w, ggml_context * ctx_g, ggml_cgraph * gf) {
        ggml_tensor * a = ggml_new_tensor_2d(ctx_w, type,          n, n);
        ggml_tensor * b = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, n, n);
        ggml_build_forward_expand(gf, ggml_mul_mat(ctx_g, a, b));
    });

    return t > 0.0 ? (float) (2.0*n*n*n / t / 1e9) : 0.0f;
}

// read speed of the device memory, in GB/s
static float device_read_bw(ggml_backend_dev_t dev, int n_threads) {
    const int64_t n = 64ll*1024*1024; // 256 MiB of F32

    const double t = bench_graph(dev, n_threads, 2, [&](ggml_context * ctx_w, ggml_context * ctx_g, ggml_cgraph * gf) {
        ggml_tensor * a = ggml_new_tensor_1d(ctx_w, GGML_TYPE_F32, n);
        ggml_build_forward_expand(gf, ggml_read(ctx_g, a));
    });

    return t > 0.0 ? (float) (n*sizeof(float) / t / 1e9) : 0.0f;
}

// time to copy the K and V of one token of one layer into a F16 cache, in ms
static float device_kv_cpy_ms(ggml_backend_dev_t dev, const prima_model_stats & ms, int n_threads) {
    const double t = bench_graph(dev, n_threads, 4, [&](ggml_context * ctx_w, ggml_context * ctx_g, ggml_cgraph * gf) {
        ggml_tensor * k  = ggml_new_tensor_1d(ctx_w, GGML_TYPE_F32, ms.n_embd_k_gqa);
        ggml_tensor * v  = ggml_new_tensor_1d(ctx_w, GGML_TYPE_F32, ms.n_embd_v_gqa);
        ggml_tensor * kc = ggml_new_tensor_1d(ctx_w, GGML_TYPE_F16, ms.n_embd_k_gqa);
        ggml_tensor * vc = ggml_new_tensor_1d(ctx_w, GGML_TYPE_F16, ms.n_embd_v_gqa);
        ggml_build_forward_expand(gf, ggml_cpy(ctx_g, k, kc));
        ggml_build_forward_expand(gf, ggml_cpy(ctx_g, v, vc));
    });

    return (float) (t * 1e3);
}

prima_device_info prima_profile_device(const common_params & params, const prima_model_stats & ms) {
    prima_device_info di;

    const int n_threads = params.cpuparams.n_threads > 0 ? params.cpuparams.n_threads : common_cpu_get_num_math();

    di.rank = params.rank;
    copy_str(di.os,      sizeof(di.os),      device_os());
    copy_str(di.next_ip, sizeof(di.next_ip), params.next_node_ip.c_str());

#if defined(_WIN32)
    copy_str(di.name, sizeof(di.name), getenv("COMPUTERNAME"));
#else
    gethostname(di.name, sizeof(di.name) - 1);
#endif

    const double gib = 1024.0*1024.0*1024.0;

    di.cpu_cores       = std::thread::hardware_concurrency();
    di.mem_total       = (float) (device_physical_memory(false) / gib);
    di.mem_available   = (float) (device_physical_memory(true)  / gib);
    di.mem_swappable   = (float) (device_swappable_memory()     / gib);
    di.swap_total      = (float) (device_swap_memory(false)     / gib);
    di.swap_available  = (float) (device_swap_memory(true)      / gib);
    di.cpu_read_ram_bw = device_memory_bw(n_threads);

    device_disk_bw(params.model.path, di.disk_read_seq_bw, di.disk_read_rnd_bw);

    // the weight types of the model
    std::vector<ggml_type> types;
    for (int t = 0; t < std::min<int>(PRIMA_N_TYPES, GGML_TYPE_COUNT); ++t) {
        if (ms.layer_flops[t] > 0 || ms.output_flops[t] > 0) {
            types.push_back((ggml_type) t);
        }
    }

    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_dev_t gpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);

    if (cpu) {
        copy_str(di.cpu_name, sizeof(di.cpu_name), ggml_backend_dev_description(cpu));
        di.cpu_kv_cpy_ms = device_kv_cpy_ms(cpu, ms, n_threads);
        for (ggml_type t : types) {
            di.cpu_gflops[t] = device_gflops(cpu, t, ms.n_embd, n_threads);
        }
    }

    if (gpu) {
        const std::string reg = ggml_backend_reg_name(ggml_backend_dev_backend_reg(gpu));

        di.gpu_metal = reg == "MTL" || reg == "Metal";
        di.gpu_cuda  = reg == "CUDA" || reg == "ROCm" || reg == "MUSA";
        di.has_gpu   = di.gpu_metal || di.gpu_cuda;

        if (!di.has_gpu) {
            LOG_WRN("%s: GPU backend %s is not used by the layer assignment, only CUDA (ROCm, MUSA) and Metal are\n", __func__, reg.c_str());
        }

        size_t free = 0;
        size_t total = 0;
        ggml_backend_dev_memory(gpu, &free, &total);

        copy_str(di.gpu_name, sizeof(di.gpu_name), ggml_backend_dev_description(gpu));
        di.gpu_mem_free  = std::min((float) params.gpu_mem, (float) (free / gib));
        di.gpu_mem_total = (float) (total / gib);

        if (di.has_gpu) {
            di.gpu_kv_cpy_ms    = device_kv_cpy_ms(gpu, ms, n_threads);
            di.gpu_read_vram_bw = device_read_bw(gpu, n_threads);
            for (ggml_type t : types) {
                di.gpu_gflops[t] = device_gflops(gpu, t, ms.n_embd, n_threads);
            }
        }
    }

    return di;
}

void prima_compute_buf_size(const prima_model_stats & ms, const prima_device_info & dev, bool is_master, uint32_t n_ctx, uint32_t n_ubatch,
        bool offload, bool has_gpu_layers, int64_t & cpu_buf, int64_t & gpu_buf) {
    const int64_t ub = n_ubatch;
    const int64_t f32 = sizeof(float);

    // the activations of one layer: residual, norm, attention scores (or FFN) and the KQ mask
    const int64_t act = ub*(3*ms.n_embd + std::max(2*ms.n_ff, 2*ms.n_embd + ms.n_head*(int64_t) n_ctx))*f32 + ub*(int64_t) n_ctx*f32;

    // the master also holds the input tokens, the output head activations and the logits
    const int64_t io = is_master ? ub*(ms.n_vocab + 2*ms.n_embd + 2)*f32 : ub*ms.n_embd*f32;

    cpu_buf = io;
    gpu_buf = 0;

    if (dev.has_gpu && has_gpu_layers) {
        gpu_buf = act;
        if (offload) {
            cpu_buf += act;
        }
        // the GPU context and a margin against OOM (values from prima.cpp)
        gpu_buf += dev.gpu_cuda ? 700ll*1024*1024 : 300ll*1024*1024;
    } else {
        cpu_buf += act;
    }
}

void prima_print_devices(const std::vector<prima_device_info> & devs, const prima_model_stats & ms) {
    auto row = [&](const char * name, auto fn) {
        std::string s = string_format("| %-26s ", name);
        for (const auto & d : devs) {
            s += string_format("| %-14s ", fn(d).c_str());
        }
        LOG_INF("%s|\n", s.c_str());
    };

    LOG_INF("\n");
    row("Rank",                    [](const prima_device_info & d) { return string_format("%u", d.rank); });
    row("Device name",             [](const prima_device_info & d) { return std::string(d.name).substr(0, 14); });
    row("Device OS",               [](const prima_device_info & d) { return std::string(d.os); });
    row("Next IP",                 [](const prima_device_info & d) { return std::string(d.next_ip).substr(0, 14); });
    row("CPU cores",               [](const prima_device_info & d) { return string_format("%u", d.cpu_cores); });
    row("Physical memory (GiB)",   [](const prima_device_info & d) { return string_format("%.2f", d.mem_total); });
    row("Available memory (GiB)",  [](const prima_device_info & d) { return string_format("%.2f", d.mem_available); });
    row("Swappable memory (GiB)",  [](const prima_device_info & d) { return string_format("%.2f", d.mem_swappable); });
    row("Swap available (GiB)",    [](const prima_device_info & d) { return string_format("%.2f", d.swap_available); });
    row("RAM read bw (GB/s)",      [](const prima_device_info & d) { return string_format("%.2f", d.cpu_read_ram_bw); });
    row("Disk seq read (GB/s)",    [](const prima_device_info & d) { return string_format("%.2f", d.disk_read_seq_bw); });
    row("Disk rnd read (GB/s)",    [](const prima_device_info & d) { return string_format("%.2f", d.disk_read_rnd_bw); });
    row("CPU KV copy (ms)",        [](const prima_device_info & d) { return string_format("%.3f", d.cpu_kv_cpy_ms); });
    for (int t = 0; t < std::min<int>(PRIMA_N_TYPES, GGML_TYPE_COUNT); ++t) {
        if (ms.layer_flops[t] > 0 || ms.output_flops[t] > 0) {
            const std::string name = string_format("CPU %s x F32 (GFLOPS)", ggml_type_name((ggml_type) t));
            row(name.c_str(), [t](const prima_device_info & d) { return string_format("%.2f", d.cpu_gflops[t]); });
        }
    }
    row("GPU",                     [](const prima_device_info & d) { return std::string(d.gpu_metal ? "Metal" : d.gpu_cuda ? "CUDA" : "none"); });
    row("GPU free memory (GiB)",   [](const prima_device_info & d) { return string_format("%.2f", d.gpu_mem_free); });
    row("GPU KV copy (ms)",        [](const prima_device_info & d) { return string_format("%.3f", d.gpu_kv_cpy_ms); });
    row("VRAM read bw (GB/s)",     [](const prima_device_info & d) { return string_format("%.2f", d.gpu_read_vram_bw); });
    for (int t = 0; t < std::min<int>(PRIMA_N_TYPES, GGML_TYPE_COUNT); ++t) {
        if (ms.layer_flops[t] > 0 || ms.output_flops[t] > 0) {
            const std::string name = string_format("GPU %s x F32 (GFLOPS)", ggml_type_name((ggml_type) t));
            row(name.c_str(), [t](const prima_device_info & d) { return string_format("%.2f", d.gpu_gflops[t]); });
        }
    }
    LOG_INF("\n");
    LOG_INF("model: n_layer = %" PRId64 ", n_vocab = %" PRId64 ", n_embd = %" PRId64 ", input = %.2f MiB, layer = %.2f MiB, output = %.2f MiB\n",
            ms.n_layer, ms.n_vocab, ms.n_embd, ms.nb_input/1024.0/1024.0, ms.nb_layer/1024.0/1024.0, ms.nb_output/1024.0/1024.0);
    LOG_INF("\n");
}
