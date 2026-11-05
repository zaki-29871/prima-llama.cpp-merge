// prima.cpp: automatic layer assignment (HiGHS), device selection and ring setup

#include "prima.h"

#include "common.h"
#include "log.h"

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

#if defined(USE_HIGHS)
#include "interfaces/highs_c_api.h"
#endif

static constexpr double GIB = 1024.0*1024.0*1024.0;
static constexpr double EPS = 1e-9;

template <typename T>
static std::string vec_to_str(const std::vector<T> & vec) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        oss << vec[i] << (i + 1 < vec.size() ? ", " : "");
    }
    oss << "]";
    return oss.str();
}

#if defined(USE_HIGHS)
// time to compute the given flops with the given speeds, in ms
static double calc_ms(const int64_t * flops, const float * gflops) {
    double t = 0.0;
    for (int i = 0; i < PRIMA_N_TYPES; ++i) {
        if (flops[i] > 0) {
            t += flops[i] / (gflops[i]*1e9 + EPS);
        }
    }
    return t * 1000.0;
}

static bool is_os(const prima_device_info & dev, const char * os) {
    return strcmp(dev.os, os) == 0;
}
#endif

// initial windows proportional to the memory budget, sum(w) = n_layer
static std::vector<int> init_windows(const std::vector<float> & mem_budget, int n_layer) {
    const int n_world = (int) mem_budget.size();
    const float total = std::accumulate(mem_budget.begin(), mem_budget.end(), 0.0f);

    std::vector<int> w(n_world, 0);
    for (int m = 0; m < n_world; ++m) {
        w[m] = (int) std::round(mem_budget[m] / (total + EPS) * n_layer);
    }

    // each device needs at least one layer
    for (int m = 0; m < n_world; ++m) {
        if (w[m] == 0) {
            w[m] = 1;
            auto it = std::max_element(w.begin(), w.end());
            if (*it > 1) {
                *it -= 1;
            }
        }
    }

    const int diff = n_layer - std::accumulate(w.begin(), w.end(), 0);
    const auto dev = diff > 0 ? std::max_element(mem_budget.begin(), mem_budget.end()) : std::min_element(mem_budget.begin(), mem_budget.end());
    w[std::distance(mem_budget.begin(), dev)] += diff;

    return w;
}

// one round of the layer assignment, ported from prima.cpp
static bool assign_layers_to_devices(const std::vector<prima_device_info> & devs, const prima_model_stats & ms, const common_params & params,
        std::vector<uint32_t> & n_layer_window, std::vector<uint32_t> & n_gpu_layers) {
    const uint32_t n_world = (uint32_t) devs.size();
    const int      n_layer = (int) ms.n_layer;
    const int64_t  n_kv    = params.n_ctx;

    const bool keep_out_in_metal = params.keep_out_in_metal;

    std::vector<int>   w(n_world, 0);
    std::vector<int>   n(n_world, 0);
    std::vector<float> mem_budget(n_world, 0.0f);

    // bytes of one layer, with its KV cache (F16)
    const int64_t b       = ms.nb_layer;
    const int64_t bo      = ms.nb_output;
    const int64_t b_kv    = 2*(ms.n_embd_k_gqa + ms.n_embd_v_gqa)*n_kv;
    const int64_t b_prime = b + b_kv;

    auto has_gpu = [&](uint32_t m) { return devs[m].has_gpu != 0; };

    auto compute_buf = [&](uint32_t m, bool offload, bool has_gpu_layers, int64_t & c_cpu, int64_t & c_gpu) {
        prima_compute_buf_size(ms, devs[m], m == 0, params.n_ctx, params.n_ubatch, offload, has_gpu_layers, c_cpu, c_gpu);
    };

#if defined(USE_HIGHS)
    const int64_t bi_row = ms.nb_input / std::max<int64_t>(1, ms.n_vocab); // one token of the input embeddings

    // alpha: time of one layer on the CPU, beta: time difference GPU - CPU, xi: communication time
    std::vector<float> alpha(n_world, 0.0f);
    std::vector<float> beta (n_world, 0.0f);
    std::vector<float> xi   (n_world, 0.0f);

    for (uint32_t m = 0; m < n_world; ++m) {
        const auto & dev = devs[m];

        const float t_calc_cpu = calc_ms(ms.layer_flops, dev.cpu_gflops);
        alpha[m] = t_calc_cpu + dev.cpu_kv_cpy_ms;

        if (has_gpu(m)) {
            const float t_calc_gpu = calc_ms(ms.layer_flops, dev.gpu_gflops);
            beta[m] = t_calc_gpu - t_calc_cpu + dev.gpu_kv_cpy_ms - dev.cpu_kv_cpy_ms;
        }

        // the RAM-VRAM transfer time and the communication time are less than 1 ms
        xi[m] = 0.0f;
    }

    for (uint32_t m = 0; m < n_world; ++m) {
        const auto & dev = devs[m];
        if (is_os(dev, "macOS") && dev.gpu_metal) {
            mem_budget[m] = dev.gpu_mem_free + 1e-4f;
        } else if (is_os(dev, "Android")) {
            mem_budget[m] = dev.mem_available + dev.mem_swappable;
        } else {
            mem_budget[m] = dev.mem_available;
        }
    }

    w = init_windows(mem_budget, n_layer);

    // assume all layers can run on the GPU at first
    for (uint32_t m = 0; m < n_world; ++m) {
        n[m] = has_gpu(m) ? w[m] : 0;
    }

    // the page faults on Linux read the file sequentially (readahead), on the other OS randomly
    std::vector<float> disk_speed(n_world, 0.0f);
    for (uint32_t m = 0; m < n_world; ++m) {
        disk_speed[m] = is_os(devs[m], "Linux") ? devs[m].disk_read_seq_bw : devs[m].disk_read_rnd_bw;
        disk_speed[m] = std::max(disk_speed[m], 1e-3f);
    }

    std::vector<int> valid_k;
    if (params.n_cycles > 0) {
        valid_k.push_back(params.n_cycles);
    } else {
        for (int k = 1; k <= n_layer / 2; ++k) {
            if (n_layer % k == 0) {
                valid_k.push_back(k);
            }
        }
    }

    // M1: macOS without Metal and with insufficient memory
    // M2: macOS with Metal and with insufficient memory
    // M3: Linux or Android with insufficient memory
    // M4: devices with sufficient memory or a very slow disk
    std::vector<uint32_t> M1, M2, M3, M4, M1_prev, M2_prev, M3_prev, M4_prev;
    std::vector<bool>    M4_force(n_world, false);
    std::vector<int64_t> c_cpu(n_world, 0), c_gpu(n_world, 0);

    const float min_disk_read_speed = 0.1f; // GB/s

    auto in_set = [](uint32_t m, const std::vector<uint32_t> & M) {
        return std::find(M.begin(), M.end(), m) != M.end();
    };

    auto assign_sets = [&](int k) -> bool {
        M1.clear(); M2.clear(); M3.clear(); M4.clear();

        for (uint32_t m = 0; m < n_world; ++m) {
            const auto & dev = devs[m];
            const bool is_macos = is_os(dev, "macOS");
            const bool is_linux = is_os(dev, "Linux") || is_os(dev, "Android");

            compute_buf(m, w[m] > n[m], n[m] > 0, c_cpu[m], c_gpu[m]);

            const int64_t l_m     = w[m] * k; // layers on device m
            const int64_t l_m_gpu = n[m] * k; // layers on the GPU of device m
            const int64_t b_io    = (bi_row + bo) * int(m == 0);
            const double  budget  = mem_budget[m] * GIB;

            const bool cond1 = l_m*b + b_io + b_kv*l_m + c_cpu[m] > budget;
            const bool cond2 = l_m*b + b_io + b_kv*l_m + c_cpu[m] + c_gpu[m] > budget;
            const bool cond3 = (l_m - l_m_gpu)*b_prime + b_io + c_cpu[m] > budget;

            if (M4_force[m] || disk_speed[m] < min_disk_read_speed) {
                M4.push_back(m);
            } else if (is_macos && !dev.gpu_metal && cond1) {
                M1.push_back(m);
            } else if (is_macos && dev.gpu_metal && cond2) {
                M2.push_back(m);
            } else if (is_linux && cond3) {
                M3.push_back(m);
            } else {
                M4.push_back(m);
            }
        }

        const bool changed = M1 != M1_prev || M2 != M2_prev || M3 != M3_prev || M4 != M4_prev;
        M1_prev = M1; M2_prev = M2; M3_prev = M3; M4_prev = M4;

        return changed;
    };

    std::vector<double> final_solution, rollback_solution;
    int final_k = -1, rollback_k = -1;

    for (int iter = 0; iter < 64; ++iter) {
        const int W = std::accumulate(w.begin(), w.end(), 0);
        if (W <= 0 || n_layer % W != 0) {
            LOG_ERR("%s: L = k * W must hold, but W = %d, L = %d\n", __func__, W, n_layer);
            return false;
        }
        const int cur_k = n_layer / W;

        if (!assign_sets(cur_k) && !final_solution.empty()) {
            break;
        }

        LOG_INF("Set assignment: M1: %s, M2: %s, M3: %s, M4: %s\n",
                vec_to_str(M1).c_str(), vec_to_str(M2).c_str(), vec_to_str(M3).c_str(), vec_to_str(M4).c_str());

        // kappa: the constant terms, the output head and the input/output loading on the master
        double kappa = 0.0;
        for (uint32_t m = 0; m < n_world; ++m) {
            const auto & dev = devs[m];
            if (m == 0) {
                kappa += calc_ms(ms.output_flops, dev.cpu_gflops);
                kappa += bi_row / (disk_speed[m] * 1e9) * 1000.0;
                if (!in_set(m, M4)) {
                    kappa += bo / (disk_speed[m] * 1e9) * 1000.0;
                }
            }
            if (in_set(m, M1) || in_set(m, M3)) {
                kappa += (c_cpu[m] - dev.mem_available*GIB - dev.mem_swappable*GIB*int(is_os(dev, "Android"))) / (disk_speed[m] * 1e9) * 1000.0;
            }
        }

        // cost vectors: a for w[m], b for n[m], c constant
        std::vector<double> vec_a(n_world, 0.0), vec_b(n_world, 0.0), vec_c(n_world, 0.0);
        for (uint32_t m = 0; m < n_world; ++m) {
            const double t_load_prime = b_prime / (disk_speed[m] * 1e9) * 1000.0;
            const double t_load       = b       / (disk_speed[m] * 1e9) * 1000.0;

            if (in_set(m, M1)) {
                vec_a[m] = alpha[m] + t_load_prime;
            } else if (in_set(m, M2)) {
                vec_a[m] = alpha[m] + t_load;
                vec_b[m] = beta[m];
            } else if (in_set(m, M3)) {
                vec_a[m] = alpha[m] + t_load_prime;
                if (has_gpu(m)) {
                    vec_b[m] = beta[m] - t_load_prime;
                }
            } else {
                vec_a[m] = alpha[m];
                if (has_gpu(m)) {
                    vec_b[m] = beta[m];
                }
            }
            vec_c[m] = xi[m];
        }

        // memory vectors: z for RAM, z_gpu for VRAM, in units of n_layer * b_prime
        std::vector<double> vec_z(n_world, 0.0), vec_z_gpu(n_world, 0.0);
        for (uint32_t m = 0; m < n_world; ++m) {
            const auto & dev = devs[m];
            const bool is_macos   = is_os(dev, "macOS");
            const bool is_android = is_os(dev, "Android");

            const double b_cio = (double) ((bi_row + bo) * int(m == 0) + c_cpu[m]);
            const double unit  = (double) n_layer * b_prime;

            if (in_set(m, M1)) {
                vec_z[m] = (dev.mem_available*GIB - b_cio) / unit;
            } else if (in_set(m, M2)) {
                vec_z[m] = (dev.gpu_mem_free*GIB - b_cio - c_gpu[m]) / unit;
            } else if (in_set(m, M3)) {
                vec_z[m] = (dev.mem_available*GIB + dev.mem_swappable*GIB*int(is_android) - b_cio) / unit;
            } else if (is_macos && !dev.gpu_metal) {
                vec_z[m] = -(dev.mem_available*GIB - b_cio) / unit;
            } else if (is_macos && dev.gpu_metal) {
                vec_z[m] = -(dev.gpu_mem_free*GIB - b_cio - c_gpu[m]) / unit;
            } else {
                vec_z[m] = -((dev.mem_available + dev.mem_swappable*int(is_android))*GIB - b_cio) / unit;
            }

            if (has_gpu(m)) {
                vec_z_gpu[m] = (dev.gpu_mem_free*GIB - c_gpu[m]) / unit;
                if (dev.gpu_metal && m == 0 && keep_out_in_metal) {
                    vec_z_gpu[m] -= bo / unit;
                }
                vec_z_gpu[m] = std::max(vec_z_gpu[m], 0.0);
            }
        }

        // solve the MIP for k cycles
        // obj_cap < 0: minimize the latency k * sum(a[m] * w[m] + b[m] * n[m]) + kappa + k * sum(c[m])
        // obj_cap > 0: among the solutions with latency <= obj_cap, minimize the sum of the squared busy times of the devices
        //              x^2 is replaced by its tangents x^2 >= 2*p*x - p^2 at n_tan points, the busy times are scaled by obj_cap
        const int n_tan = 128;

        auto solve = [&](int k, double obj_cap, std::vector<double> & sol, double & objective) -> bool {
            const int Wk = n_layer / k;

            const bool balance = obj_cap > 0.0;

            const HighsInt n_cols = 2*n_world + (balance ? n_world : 0); // w[m], n[m], s[m]
            const HighsInt n_rows = 1 + 3*n_world + (balance ? 1 + n_world*(n_tan + 1) : 0);

            const double offset = k * std::accumulate(vec_c.begin(), vec_c.end(), 0.0) + kappa;

            std::vector<double> cost(2*n_world);
            for (uint32_t m = 0; m < n_world; ++m) {
                cost[m]           = k * vec_a[m];
                cost[m + n_world] = k * vec_b[m];
            }
            // priority of the master
            cost[0] *= 1.0 / params.master_priority;

            std::vector<double> col_lower(n_cols, 0.0), col_upper(n_cols, (double) n_layer);
            std::fill(col_lower.begin(), col_lower.begin() + n_world, 1.0);

            std::vector<double> row_lower(n_rows, -1.0e30), row_upper(n_rows, 1.0e30);
            std::vector<std::vector<double>> A(n_rows, std::vector<double>(n_cols, 0.0));

            int r = 0;

            // sum(w[m]) = W
            row_lower[r] = row_upper[r] = Wk;
            std::fill_n(A[r].begin(), n_world, 1.0);
            r++;

            // n[m] <= w[m]
            for (uint32_t m = 0; m < n_world; ++m, ++r) {
                row_upper[r] = 0.0;
                A[r][m]           = -1.0;
                A[r][m + n_world] =  1.0;
            }

            // RAM
            for (uint32_t m = 0; m < n_world; ++m, ++r) {
                row_upper[r] = -Wk * vec_z[m];
                if (in_set(m, M1) || in_set(m, M2)) {
                    A[r][m] = -1.0;
                } else if (in_set(m, M3)) {
                    A[r][m] = -1.0;
                    if (has_gpu(m)) {
                        A[r][m + n_world] = 1.0;
                    }
                } else {
                    A[r][m] = 1.0;
                    if (!is_os(devs[m], "macOS") && has_gpu(m)) {
                        A[r][m + n_world] = -1.0;
                    }
                }
            }

            // VRAM
            for (uint32_t m = 0; m < n_world; ++m, ++r) {
                row_upper[r] = std::max(Wk * vec_z_gpu[m], 0.0);
                A[r][m + n_world] = 1.0;
            }

            std::vector<double> obj(n_cols, 0.0);

            if (!balance) {
                std::copy(cost.begin(), cost.end(), obj.begin());
            } else {
                // the latency stays within the cap
                row_upper[r] = obj_cap - offset;
                std::copy(cost.begin(), cost.end(), A[r].begin());
                r++;

                // busy time of device m: x[m] = (k * (a[m] * w[m] + b[m] * n[m] + c[m]) + kappa on the master) / obj_cap
                // no device is busy for longer than the latency, so the tangent points cover [0, 1.1]
                for (uint32_t m = 0; m < n_world; ++m) {
                    const HighsInt s_m = 2*n_world + m;

                    obj[s_m]       = 1.0;
                    col_lower[s_m] = 0.0;
                    col_upper[s_m] = 1.0e30;

                    const double x0 = (k * vec_c[m] + (m == 0 ? kappa : 0.0)) / obj_cap;

                    // s[m] >= 2*p*x[m] - p^2
                    for (int j = 0; j <= n_tan; ++j, ++r) {
                        const double p = 1.1 * j / n_tan;
                        row_upper[r] = p*p - 2.0*p*x0;
                        A[r][m]           = 2.0*p * k * vec_a[m] / obj_cap;
                        A[r][m + n_world] = 2.0*p * k * vec_b[m] / obj_cap;
                        A[r][s_m]         = -1.0;
                    }
                }
            }

            std::vector<HighsInt> a_start(n_cols + 1), a_index;
            std::vector<double>   a_value;
            for (HighsInt j = 0; j < n_cols; ++j) {
                a_start[j] = (HighsInt) a_index.size();
                for (HighsInt i = 0; i < n_rows; ++i) {
                    if (A[i][j] != 0.0) {
                        a_index.push_back(i);
                        a_value.push_back(A[i][j]);
                    }
                }
            }
            a_start[n_cols] = (HighsInt) a_index.size();

            std::vector<HighsInt> integrality(n_cols, kHighsVarTypeInteger);
            for (HighsInt j = 2*n_world; j < n_cols; ++j) {
                integrality[j] = kHighsVarTypeContinuous;
            }

            std::vector<double> col_value(n_cols, 0.0), row_value(n_rows, 0.0);
            HighsInt model_status = 0;

            const HighsInt status = Highs_mipCall(n_cols, n_rows, (HighsInt) a_value.size(), kHighsMatrixFormatColwise, kHighsObjSenseMinimize, balance ? 0.0 : offset,
                    obj.data(), col_lower.data(), col_upper.data(), row_lower.data(), row_upper.data(),
                    a_start.data(), a_index.data(), a_value.data(), integrality.data(), col_value.data(), row_value.data(), &model_status);

            if (status == kHighsStatusError || model_status != kHighsModelStatusOptimal) {
                return false;
            }

            sol.assign(2*n_world, 0.0);
            objective = offset;
            for (uint32_t j = 0; j < 2*n_world; ++j) {
                sol[j] = std::round(col_value[j]);
                objective += cost[j] * sol[j];
            }

            return true;
        };

        double best_objective = 1.0e30;
        std::vector<double> best_solution;
        int best_k = -1;

        for (int k : valid_k) {
            if (n_layer % k != 0) {
                continue;
            }

            std::vector<double> col_value;
            double objective = 0.0;
            if (!solve(k, -1.0, col_value, objective)) {
                continue;
            }

            if (objective < best_objective) {
                best_objective = objective;
                best_k         = k;
                best_solution  = col_value;
            }
            LOG_INF("k = %2d, obj = %7.1f, solution: %s | best_k = %2d, best_obj = %7.1f\n", k, objective, vec_to_str(col_value).c_str(), best_k, best_objective);
        }

        // the latency model is linear, so the optimum puts all layers on the cheapest device even when the others are almost as fast
        // trade at most balance_tol of the predicted latency for a more even busy time of the devices
        if (!best_solution.empty() && params.balance_tol > 0.0f) {
            std::vector<double> sol;
            double objective = 0.0;
            if (solve(best_k, best_objective * (1.0 + params.balance_tol), sol, objective)) {
                if (sol != best_solution) {
                    LOG_INF("balance (tol = %.2f): obj = %7.1f, solution: %s\n", params.balance_tol, objective, vec_to_str(sol).c_str());
                }
                best_solution = std::move(sol);
            }
        }

        if (best_solution.empty()) {
            if (rollback_solution.empty()) {
                LOG_WRN("%s: no feasible solution found\n", __func__);
                break;
            }
            LOG_INF("No feasible solution found for this set assignment, rolling back to the previous sets.\n");
            final_solution = rollback_solution;
            final_k        = rollback_k;
            break;
        }

        rollback_solution = best_solution;
        rollback_k        = best_k;

        const int W_best = n_layer / best_k;

        // check the solution
        bool has_free_gpu_memory = false, has_gpu_overload = false, has_cpu_overload = false, has_weak_device = false;
        for (uint32_t m = 0; m < n_world; ++m) {
            const int w_m = (int) best_solution[m];
            const int n_m = (int) best_solution[m + n_world];

            if (w_m == 1 && n_m == 0) {
                has_weak_device = true;
                LOG_INF("Device %u is weak, need to be removed: w_m = %d, n_m = %d\n", m, w_m, n_m);
            }

            if (has_gpu(m)) {
                const int n_gpu_max = (int) std::floor(W_best * vec_z_gpu[m]);
                if (n_m < n_gpu_max) {
                    has_free_gpu_memory = true;
                    LOG_INF("Device %u still has free GPU memory: w_m = %d, n_m = %d, max = %d\n", m, w_m, n_m, n_gpu_max);
                }
                if (w_m > n_m) {
                    has_gpu_overload = true;
                    LOG_INF("Device %u has GPU overload: w_m = %d, n_m = %d\n", m, w_m, n_m);
                }
            } else if (!in_set(m, M4)) {
                has_cpu_overload = true;
                LOG_INF("Device %u has CPU overload\n", m);
            }
        }

        if (!has_weak_device && has_free_gpu_memory && (has_gpu_overload || has_cpu_overload)) {
            // force the device with the slowest disk that is not in M4 yet into M4
            int   worst = -1;
            float worst_speed = std::numeric_limits<float>::max();
            for (uint32_t m = 0; m < n_world; ++m) {
                if (!in_set(m, M4) && disk_speed[m] < worst_speed) {
                    worst_speed = disk_speed[m];
                    worst = (int) m;
                }
            }
            if (worst != -1) {
                M4_force[worst] = true;
                LOG_INF("Forcing device %d (disk speed %.2f GB/s) into M4\n", worst, worst_speed);
                continue;
            }
            LOG_INF("Infeasible solution detected but no device can be forced into M4\n");
        }

        for (uint32_t m = 0; m < n_world; ++m) {
            w[m] = (int) best_solution[m];
            n[m] = (int) best_solution[m + n_world];
        }

        const bool unchanged = final_solution == best_solution;

        final_k        = best_k;
        final_solution = best_solution;

        if (unchanged) {
            break;
        }
    }

    if (!final_solution.empty()) {
        for (uint32_t m = 0; m < n_world; ++m) {
            w[m] = (int) final_solution[m];
            n[m] = (int) final_solution[m + n_world];
        }

        LOG_INF("\n----- Allocation Strategy (by HiGHS) -----\n");
        LOG_INF("  - k = %d\n", final_k);
        LOG_INF("  - W = %d\n", n_layer / final_k);
        for (uint32_t m = 0; m < n_world; ++m) {
            LOG_INF("%s (rank %u): set %s, n_layer_window = %d, n_gpu_layers = %d\n", devs[m].name, devs[m].rank,
                    in_set(m, M1) ? "M1" : in_set(m, M2) ? "M2" : in_set(m, M3) ? "M3" : "M4", w[m], n[m]);
        }
        LOG_INF("\n");

        n_layer_window.assign(w.begin(), w.end());
        n_gpu_layers  .assign(n.begin(), n.end());
        return true;
    }

    LOG_WRN("%s: HiGHS failed, assign the layers by the memory size\n", __func__);
#endif

    // assign the layers by RAM/VRAM size, one cycle
    for (uint32_t m = 0; m < n_world; ++m) {
        mem_budget[m] = has_gpu(m) ? devs[m].gpu_mem_free : devs[m].mem_available;
    }

    w = init_windows(mem_budget, n_layer);

    n_layer_window.assign(w.begin(), w.end());
    n_gpu_layers.assign(n_world, 0);

    for (uint32_t m = 0; m < n_world; ++m) {
        if (!has_gpu(m)) {
            continue;
        }

        int64_t c_cpu = 0, c_gpu = 0;
        compute_buf(m, false, true, c_cpu, c_gpu);

        const int64_t required  = w[m] * b_prime;
        int64_t       available = (int64_t) (devs[m].gpu_mem_free * GIB) - c_gpu;
        if (devs[m].gpu_metal && m == 0 && keep_out_in_metal) {
            available -= bo;
        }

        n_gpu_layers[m] = required <= available ? w[m] : (uint32_t) std::max<int64_t>(0, available / b_prime);
    }

    LOG_INF("%s: n_layer_window = %s, n_gpu_layers = %s\n", __func__, vec_to_str(n_layer_window).c_str(), vec_to_str(n_gpu_layers).c_str());

    return true;
}

bool prima_assign_layers(const std::vector<prima_device_info> & devs, const prima_model_stats & ms, const common_params & params,
        std::vector<uint32_t> & n_layer_window, std::vector<uint32_t> & n_gpu_layers) {
    // the indices of the devices that are still in the ring
    std::vector<uint32_t> idx(devs.size());
    std::iota(idx.begin(), idx.end(), 0);

    std::vector<uint32_t> w, n;

    while (true) {
        std::vector<prima_device_info> cur;
        for (uint32_t i : idx) {
            cur.push_back(devs[i]);
        }

        if (!assign_layers_to_devices(cur, ms, params, w, n)) {
            return false;
        }

        // remove the devices with only one layer, except the master
        std::vector<uint32_t> keep;
        for (size_t j = 0; j < idx.size(); ++j) {
            if (w[j] > 1 || j == 0) {
                keep.push_back((uint32_t) j);
            } else {
                LOG_INF("Remove device %s (rank %u) with only %u layer assigned.\n", devs[idx[j]].name, devs[idx[j]].rank, w[j]);
            }
        }

        if (keep.size() == idx.size()) {
            break;
        }

        std::vector<uint32_t> idx_new;
        for (uint32_t j : keep) {
            idx_new.push_back(idx[j]);
        }
        idx = std::move(idx_new);

        LOG_INF("Reassign the layers to the remaining %zu device(s).\n\n", idx.size());
    }

    n_layer_window.assign(devs.size(), 0);
    n_gpu_layers  .assign(devs.size(), 0);
    for (size_t j = 0; j < idx.size(); ++j) {
        n_layer_window[idx[j]] = w[j];
        n_gpu_layers  [idx[j]] = n[j];
    }

    return true;
}

//
// ring setup
//

namespace {

constexpr uint32_t PRIMA_MAGIC   = 0x414d5250; // "PRMA"
constexpr uint32_t PRIMA_VERSION = 1;

struct msg_startup {
    uint32_t magic;
    uint32_t version;
    uint32_t size_device_info;
    uint32_t n_world;
    uint32_t auto_schedule;
    uint32_t n_ctx;
    uint32_t n_batch;
    uint32_t n_ubatch;
    uint32_t n_parallel;
};

struct msg_layout {
    uint32_t n_layer_window[PRIMA_MAX_NODES];
    uint32_t n_gpu_layers  [PRIMA_MAX_NODES];
    uint32_t has_gpu_layers;
};

struct msg_topo {
    uint32_t n_layer_window[PRIMA_MAX_NODES];
    uint8_t  forwarder     [PRIMA_MAX_NODES];
    char     next_ip       [PRIMA_MAX_NODES][64];
};

struct net_deleter {
    void operator()(llama_prima_net * net) const { llama_prima_net_free(net); }
};

using net_ptr = std::unique_ptr<llama_prima_net, net_deleter>;

template <typename T>
void send_msg(llama_prima_net * net, const T & msg) {
    if (llama_prima_net_send(net, &msg, sizeof(msg)) != 0) {
        throw std::runtime_error("prima: failed to send a setup message");
    }
}

template <typename T>
void recv_msg(llama_prima_net * net, T & msg) {
    if (llama_prima_net_recv(net, &msg, sizeof(msg)) != (int64_t) sizeof(msg)) {
        throw std::runtime_error("prima: received a bad setup message, check that all nodes run the same build and --world");
    }
}

}

// handle the topology message: connect to the next active node, mark forwarders and tell the removed nodes to exit
// returns true if this node stays in the ring
static bool prima_topo_step(llama_prima_net * net, msg_topo & topo, uint32_t n_world, uint32_t rank, uint32_t data_port) {
    auto active = [&](uint32_t r) { return topo.n_layer_window[r] > 0 || topo.forwarder[r]; };
    auto prev   = [&](uint32_t r) { return (r + n_world - 1) % n_world; };

    const uint32_t nxt = (rank + 1) % n_world;

    if (!active(rank)) {
        // pass the exit notice on to the next removed node
        if (!active(nxt)) {
            send_msg(net, topo);
        }
        return false;
    }

    // the nodes up to the next worker
    std::vector<uint32_t> cand;
    for (uint32_t r = nxt; r != rank; r = (r + 1) % n_world) {
        cand.push_back(r);
        if (topo.n_layer_window[r] > 0) {
            break;
        }
    }

    if (cand.empty() || topo.n_layer_window[cand.back()] == 0) {
        // no other worker, only the removed nodes need to know
        if (!active(nxt) && nxt != rank) {
            send_msg(net, topo);
        }
        return true;
    }

    // connect as far as possible, the nodes in between become forwarders
    uint32_t target = nxt;
    for (size_t i = cand.size() - 1; i > 0; --i) {
        const uint32_t c = cand[i];
        if (llama_prima_port_open(topo.next_ip[prev(c)], data_port + c, 1000)) {
            target = c;
            break;
        }
        LOG_INF("%s: rank %u cannot reach rank %u at %s:%u\n", __func__, rank, c, topo.next_ip[prev(c)], data_port + c);
    }

    if (topo.n_layer_window[target] == 0) {
        topo.forwarder[target] = 1;
    }

    if (target != nxt) {
        if (!active(nxt)) {
            send_msg(net, topo);
        }
        if (llama_prima_net_set_next(net, topo.next_ip[prev(target)], data_port + target) != 0) {
            throw std::runtime_error("prima: failed to connect to the next node");
        }
    }

    send_msg(net, topo);

    return true;
}

llama_prima_net * prima_setup(common_params & params) {
    const uint32_t n_world   = params.n_world;
    const uint32_t rank      = params.rank;
    const uint32_t data_port = params.data_port;

    if (n_world < 2 || n_world > PRIMA_MAX_NODES || rank >= n_world) {
        throw std::runtime_error(string_format("prima: invalid world size %u or rank %u", n_world, rank));
    }

    net_ptr net(llama_prima_net_init(data_port + rank, params.next_node_ip.c_str(), data_port + (rank + 1) % n_world));
    if (!net) {
        throw std::runtime_error("prima: failed to set up the ring connection");
    }

    // the master sends its arguments, so that the workers use the same values
    msg_startup su = {};
    if (rank == 0) {
        uint32_t n_ctx = params.n_ctx;
        if (n_ctx == 0) {
            // the context the model was trained with
            gguf_context * gctx = gguf_init_from_file(params.model.path.c_str(), { /*.no_alloc =*/ true, /*.ctx =*/ nullptr });
            if (gctx) {
                const int64_t kid_arch = gguf_find_key(gctx, "general.architecture");
                if (kid_arch >= 0) {
                    const int64_t kid = gguf_find_key(gctx, (std::string(gguf_get_val_str(gctx, kid_arch)) + ".context_length").c_str());
                    if (kid >= 0 && gguf_get_kv_type(gctx, kid) == GGUF_TYPE_UINT32) {
                        n_ctx = gguf_get_val_u32(gctx, kid);
                    }
                }
                gguf_free(gctx);
            }
        }

        bool auto_schedule = true;
        for (uint32_t r = 0; r < n_world; ++r) {
            auto_schedule = auto_schedule && params.n_layer_window[r] == 0;
        }

        su = { PRIMA_MAGIC, PRIMA_VERSION, (uint32_t) sizeof(prima_device_info), n_world, auto_schedule,
               n_ctx, (uint32_t) params.n_batch, (uint32_t) params.n_ubatch, (uint32_t) params.n_parallel };

        LOG_INF("%s: waiting for %u worker(s) ...\n", __func__, n_world - 1);
        send_msg(net.get(), su);

        msg_startup ack = {};
        recv_msg(net.get(), ack);
        if (memcmp(&ack, &su, sizeof(su)) != 0) {
            throw std::runtime_error("prima: the ring returned different startup arguments");
        }
    } else {
        LOG_INF("%s: waiting for the master ...\n", __func__);
        recv_msg(net.get(), su);
        if (su.magic != PRIMA_MAGIC || su.version != PRIMA_VERSION || su.size_device_info != sizeof(prima_device_info)) {
            throw std::runtime_error("prima: the master runs a different build");
        }
        if (su.n_world != n_world) {
            throw std::runtime_error(string_format("prima: the master uses --world %u, this node uses --world %u", su.n_world, n_world));
        }
        send_msg(net.get(), su);
    }

    params.n_ctx      = su.n_ctx;
    params.n_batch    = su.n_batch;
    params.n_ubatch   = su.n_ubatch;
    params.n_parallel = su.n_parallel;

    const bool auto_schedule = su.auto_schedule != 0;

    // profile the devices and gather the profiles on the master
    std::vector<prima_device_info> devs;
    prima_model_stats ms;

    if (auto_schedule) {
        LOG_INF("%s: profiling this device, this can take some seconds ...\n", __func__);

        ms = prima_model_stats_from_file(params.model.path, params.n_ctx, params.n_predict);
        const prima_device_info di = prima_profile_device(params, ms);

        devs.resize(rank + 1);
        if (rank == 0) {
            devs[0] = di;
            if (llama_prima_net_send(net.get(), devs.data(), sizeof(prima_device_info)) != 0) {
                throw std::runtime_error("prima: failed to send the device info");
            }
            devs.resize(n_world);
            if (llama_prima_net_recv(net.get(), devs.data(), n_world*sizeof(prima_device_info)) != (int64_t) (n_world*sizeof(prima_device_info))) {
                throw std::runtime_error("prima: failed to gather the device info");
            }
        } else {
            if (llama_prima_net_recv(net.get(), devs.data(), rank*sizeof(prima_device_info)) != (int64_t) (rank*sizeof(prima_device_info))) {
                throw std::runtime_error("prima: failed to receive the device info");
            }
            devs[rank] = di;
            if (llama_prima_net_send(net.get(), devs.data(), (rank + 1)*sizeof(prima_device_info)) != 0) {
                throw std::runtime_error("prima: failed to send the device info");
            }
        }
    }

    // the master assigns the layers and sends the layout
    msg_layout layout = {};
    if (rank == 0) {
        if (auto_schedule) {
            prima_print_devices(devs, ms);

            std::vector<uint32_t> w, n;
            if (!prima_assign_layers(devs, ms, params, w, n)) {
                throw std::runtime_error("prima: failed to assign the layers");
            }
            std::copy(w.begin(), w.end(), layout.n_layer_window);
            std::copy(n.begin(), n.end(), layout.n_gpu_layers);
            layout.has_gpu_layers = 1;
        } else {
            std::copy(params.n_layer_window, params.n_layer_window + n_world, layout.n_layer_window);
        }

        send_msg(net.get(), layout);

        msg_layout ack = {};
        recv_msg(net.get(), ack);
    } else {
        recv_msg(net.get(), layout);
        send_msg(net.get(), layout);
    }

    // remove the devices without layers, the nodes that cannot reach the next worker keep a forwarder in between
    msg_topo topo = {};
    std::copy(layout.n_layer_window, layout.n_layer_window + n_world, topo.n_layer_window);

    const bool has_removed = std::any_of(layout.n_layer_window, layout.n_layer_window + n_world, [](uint32_t w) { return w == 0; });
    if (has_removed) {
        if (rank == 0) {
            for (uint32_t r = 0; r < n_world; ++r) {
                snprintf(topo.next_ip[r], sizeof(topo.next_ip[r]), "%s", devs[r].next_ip);
            }
            const bool ring = prima_topo_step(net.get(), topo, n_world, rank, data_port);

            // wait for the ring to close, then all workers are connected
            const bool has_workers = std::any_of(layout.n_layer_window + 1, layout.n_layer_window + n_world, [](uint32_t w) { return w > 0; });
            if (ring && has_workers) {
                recv_msg(net.get(), topo);
            }
        } else {
            recv_msg(net.get(), topo);
            if (!prima_topo_step(net.get(), topo, n_world, rank, data_port)) {
                LOG_INF("%s: no layer is assigned to this device, exit\n", __func__);
                net.reset();
                std::exit(0);
            }
            if (topo.n_layer_window[rank] == 0) {
                LOG_INF("%s: no layer is assigned to this device, it forwards the messages to the next node\n", __func__);
                const int32_t ret = llama_prima_net_forward(net.get());
                net.reset();
                std::exit(ret == 0 ? 0 : 1);
            }
        }
    }

    // the new ring has only the workers
    uint32_t new_rank    = 0;
    uint32_t new_n_world = 0;
    uint32_t n_layer_window[PRIMA_MAX_NODES] = {};

    for (uint32_t r = 0; r < n_world; ++r) {
        if (layout.n_layer_window[r] == 0) {
            continue;
        }
        if (r < rank) {
            new_rank++;
        }
        n_layer_window[new_n_world] = layout.n_layer_window[r];
        new_n_world++;
    }

    // n_gpu_layers from the solver is per window
    if (layout.has_gpu_layers && params.n_gpu_layers == -1 && new_n_world > 1) {
        params.n_gpu_layers = (int32_t) layout.n_gpu_layers[rank];
    }

    LOG_INF("%s: rank %u -> %u, world %u -> %u, n_layer_window = %s, n_gpu_layers = %d\n", __func__, rank, new_rank, n_world, new_n_world,
            vec_to_str(std::vector<uint32_t>(n_layer_window, n_layer_window + new_n_world)).c_str(), params.n_gpu_layers);

    params.rank    = (int32_t) new_rank;
    params.n_world = (int32_t) new_n_world;
    std::copy(n_layer_window, n_layer_window + PRIMA_MAX_NODES, params.n_layer_window);

    if (new_n_world == 1) {
        // this node runs alone, all layers in one window
        std::fill(params.n_layer_window, params.n_layer_window + PRIMA_MAX_NODES, 0);
        return nullptr;
    }

    return net.release();
}
