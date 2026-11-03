#pragma once

// prima.cpp: device profiling, automatic layer assignment and ring setup for distributed inference

#include "llama.h"

#include <cstdint>
#include <string>
#include <vector>

struct common_params;

#define PRIMA_MAX_NODES 32
#define PRIMA_N_TYPES   64 // >= GGML_TYPE_COUNT, fixed so that the wire format does not depend on the ggml version

// model statistics, computed by the master from the GGUF file
struct prima_model_stats {
    int64_t n_layer      = 0;
    int64_t n_vocab      = 0;
    int64_t n_embd       = 0;
    int64_t n_embd_k_gqa = 0;
    int64_t n_embd_v_gqa = 0;
    int64_t n_head       = 0;
    int64_t n_ff         = 0;

    // flops per token, by weight type (src0 of the matmul, src1 is F32)
    int64_t layer_flops [PRIMA_N_TYPES] = {}; // average of one layer
    int64_t output_flops[PRIMA_N_TYPES] = {};

    int64_t nb_input  = 0; // bytes of the input embeddings
    int64_t nb_layer  = 0; // average bytes of one layer
    int64_t nb_output = 0; // bytes of the output head
};

// device profile, sent from each node to the master
// plain old data, the nodes exchange it as raw bytes
struct prima_device_info {
    uint32_t rank = 0;
    char     name   [64] = {};
    char     os     [16] = {}; // Linux, macOS, Android or Windows
    char     next_ip[64] = {};

    char     cpu_name[128] = {};
    uint32_t cpu_cores     = 0;

    float    mem_total       = 0.0f; // GiB
    float    mem_available   = 0.0f; // GiB
    float    mem_swappable   = 0.0f; // GiB, used memory that the OS can swap out (Android, macOS)
    float    swap_total      = 0.0f; // GiB
    float    swap_available  = 0.0f; // GiB
    float    cpu_read_ram_bw = 0.0f; // GB/s
    float    cpu_kv_cpy_ms   = 0.0f; // ms to copy the KV of one token of one layer

    float    disk_read_seq_bw = 0.0f; // GB/s
    float    disk_read_rnd_bw = 0.0f; // GB/s

    uint8_t  has_gpu   = 0; // the solver uses a GPU only if it is a CUDA or Metal device
    uint8_t  gpu_metal = 0;
    uint8_t  gpu_cuda  = 0;
    uint8_t  pad0      = 0;
    char     gpu_name[128] = {};
    float    gpu_mem_free  = 0.0f; // GiB
    float    gpu_mem_total = 0.0f; // GiB
    float    gpu_kv_cpy_ms = 0.0f;
    float    gpu_read_vram_bw = 0.0f; // GB/s, measured with GGML_OP_READ

    float    cpu_gflops[PRIMA_N_TYPES] = {}; // matmul speed by weight type
    float    gpu_gflops[PRIMA_N_TYPES] = {};
};

// profiling, see prima-profiler.cpp
prima_model_stats prima_model_stats_from_file(const std::string & path, int64_t n_ctx, int64_t n_predict);

prima_device_info prima_profile_device(const common_params & params, const prima_model_stats & ms);

void prima_print_devices(const std::vector<prima_device_info> & devs, const prima_model_stats & ms);

// compute buffer estimate in bytes
void prima_compute_buf_size(const prima_model_stats & ms, const prima_device_info & dev, bool is_master, uint32_t n_ctx, uint32_t n_ubatch,
        bool offload, bool has_gpu_layers, int64_t & cpu_buf, int64_t & gpu_buf);

// layer assignment, n_layer_window and n_gpu_layers have one entry per device, 0 layers = remove the device
bool prima_assign_layers(const std::vector<prima_device_info> & devs, const prima_model_stats & ms, const common_params & params,
        std::vector<uint32_t> & n_layer_window, std::vector<uint32_t> & n_gpu_layers);

// ring setup before the model is loaded, call on all nodes when n_world > 1
// profiles the devices and assigns the layers if no --n-layer-window is given, removes weak devices and sets up forwarders
// updates params.rank, params.n_world, params.n_layer_window and params.n_gpu_layers
// a removed device or a forwarder does not return
// returns the ring connection for the context (NULL if this node now runs alone)
// throws on error
llama_prima_net * prima_setup(common_params & params);
