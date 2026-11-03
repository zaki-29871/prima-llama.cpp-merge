// prima.cpp device profiler
//
// runs the profiling of the automatic layer assignment on this node only and prints the result
// use it to check a device, or to choose --n-layer-window and -ngl by hand

#include "arg.h"
#include "common.h"
#include "log.h"
#include "prima.h"

#include "llama.h"

static void print_usage(int, char ** argv) {
    LOG("\nexample usage:\n");
    LOG("\n    %s -m model.gguf [-c 4096] [-t 8] [--gpu-mem 8]\n", argv[0]);
    LOG("\n");
}

int main(int argc, char ** argv) {
    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON, print_usage)) {
        return 1;
    }

    llama_backend_init();

    const int64_t n_ctx = params.n_ctx > 0 ? params.n_ctx : 4096;

    try {
        const prima_model_stats ms  = prima_model_stats_from_file(params.model.path, n_ctx, params.n_predict);
        const prima_device_info dev = prima_profile_device(params, ms);

        prima_print_devices({ dev }, ms);
    } catch (const std::exception & e) {
        LOG_ERR("%s: %s\n", __func__, e.what());
        llama_backend_free();
        common_log_flush(common_log_main());
        return 1;
    }

    llama_backend_free();
    common_log_flush(common_log_main());

    return 0;
}
