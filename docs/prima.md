# prima.cpp distributed inference

This fork ports the piped-ring parallelism of [prima.cpp](../prima.cpp) to the current llama.cpp code base.

The model layers are split into windows. The windows run on the nodes of a ring, one cycle after another. In each cycle the windows go in rank order `1, 2, ..., n-1, 0`. Node 0 (the master) also computes the input embeddings and the output head, so the master holds `token_embd`, `output_norm` and `output`. Each node loads only the tensors and the KV cache of its own layers.

## Status

Phase 1 (this port):

- ring transport over ZeroMQ, hidden states of each ubatch go from node to node
- per-node layer subset loading and per-node KV cache
- manual layer windows (`--n-layer-window`), or an even split when no windows are given
- several cycles per token (`sum(windows)` must divide `n_layer`)
- KV cache operations on the master (`llama_memory_seq_rm`, `seq_add`, ...) are forwarded to all nodes, so context shift works
- supported architectures: `llama`, `qwen2`, `qwen3` (this covers the Llama, Qwen 2.5, QwQ and DeepSeek-R1 distill models)
- supported tool: `llama-completion`

Not ported yet: device profiling and automatic layer assignment (HiGHS), device selection and forwarder nodes, weight prefetching, the Metal/CUDA specific changes, `llama-server`, speculative decoding, embeddings and session files.

## Build

Install ZeroMQ, for example `sudo apt install libzmq3-dev` or `brew install zeromq`. The CMake option `LLAMA_PRIMA` (default `ON`) links `libzmq` when it is found (pkg-config first, then `/usr/local` and, on macOS, `/opt/homebrew`). Without `libzmq` the build works, but `--world` > 1 fails at runtime.

```bash
cmake -B build
cmake --build build --config Release --target prima
```

The `prima` target builds the same set of programs as the default target of the prima.cpp Makefile (`llama-server`, `llama-cli`, `llama-completion`, `llama-speculative`, `llama-gguf-split`).

The prima.cpp Makefile options map to CMake as follows:

| Makefile | CMake |
|----------|-------|
| `-lzmq` (always) | `-DLLAMA_PRIMA=ON` (default) |
| `USE_HIGHS=1` | `-DLLAMA_PRIMA_HIGHS=ON`, defines `USE_HIGHS` and links HiGHS (`find_package(highs)` or `libhighs` in `/usr/local`, `/opt/homebrew`) |
| `src/network-utils.o`, `common/profiler.o` | added to `llama` / `llama-common` once the sources are ported |
| `profile-tool` | `tools/profile-tool`, added once it is ported |

## Usage

All nodes use the same model file and the same `-c`, `-b`, `-ub` and `-np` values. Node `r` listens on port `data_port + r` and connects to `data_port + r + 1` on the next node (the last node connects back to node 0).

Example with 2 nodes and the 32 layers of Llama 3 8B split as 16 + 16:

```bash
# node 1 (192.168.1.3), start the workers first
llama-completion -m model.gguf -c 2048 --world 2 --rank 1 -lw 16,16 --next 192.168.1.2
```

```bash
# node 0 (192.168.1.2), the master
llama-completion -m model.gguf -c 2048 --world 2 --rank 0 -lw 16,16 --next 192.168.1.3 -p "Once upon a time" -n 128
```

With `-lw 8,8` the same model runs in 2 cycles: node 1 runs layers 0-7, node 0 runs layers 8-15, node 1 runs layers 16-23 and node 0 runs layers 24-31.

The workers exit when the master exits.

Options:

| option | description |
|--------|-------------|
| `-w, --world N` | number of nodes |
| `-r, --rank N` | rank of this node, 0 is the master |
| `-lw, --n-layer-window A,B,...` | layers per node in each cycle, indexed by rank |
| `-nip, --next IP` | ip address of the next node in the ring |
| `--data-port N` | base data port (default 9000) |
