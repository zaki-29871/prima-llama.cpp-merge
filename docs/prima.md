# prima.cpp distributed inference

This fork ports the piped-ring parallelism of [prima.cpp](../prima.cpp) to the current llama.cpp code base.

The model layers are split into windows. The windows run on the nodes of a ring, one cycle after another. In each cycle the windows go in rank order `1, 2, ..., n-1, 0`. Node 0 (the master) also computes the input embeddings and the output head, so the master holds `token_embd`, `output_norm` and `output`. Each node loads only the tensors and the KV cache of its own layers.

## Status

Ported:

- ring transport over ZeroMQ, hidden states of each ubatch go from node to node
- per-node layer subset loading and per-node KV cache
- manual layer windows (`--n-layer-window`), or an even split when no windows are given and the model is used without `common`
- several cycles per token (`sum(windows)` must divide `n_layer`)
- KV cache operations on the master (`llama_memory_seq_rm`, `seq_add`, ...) are forwarded to all nodes, so context shift works
- device profiling and automatic layer assignment with the HiGHS solver (`common/prima-profiler.cpp`, `common/prima.cpp`)
- device selection: devices that get only one layer are removed, nodes that cannot reach the next worker keep the removed node in between as a forwarder
- weight prefetching (`--prefetch`)
- supported architectures: `llama`, `qwen2`, `qwen3` (this covers the Llama, Qwen 2.5, QwQ and DeepSeek-R1 distill models)
- supported tool: `llama-completion`

Not ported yet: the Metal/CUDA specific changes, `llama-server`, speculative decoding, embeddings and session files.

## Automatic layer assignment

When `--n-layer-window` is not given, the nodes set up the ring before they load the model:

1. the master sends its `-c`, `-b`, `-ub` and `-np` values to all nodes, so the workers use the same values
2. each node profiles its device: CPU and GPU matmul speed for the weight types of the model, KV copy time, RAM, swap, GPU memory and disk read speed of the model file. The master gathers the profiles and prints them
3. the master solves the layer assignment of prima.cpp (an integer linear program over the layer window `w[m]` and the GPU layers `n[m]` of each device, and the number of cycles `k`) with HiGHS. Without HiGHS the layers are assigned by the RAM/VRAM size of each device in one cycle
4. a device that gets only one layer is removed and the solver runs again with the remaining devices
5. if devices are removed, each node connects to its next worker. If the next worker cannot be reached, the removed node in between stays as a forwarder that only passes the messages on. Other removed nodes exit
6. the workers get new ranks `0..n-1` in the ring order, then all nodes load their layers

`n[m]` is the number of GPU layers per window: the last `n[m]` layers of each window of device `m` go to the GPU. If `-ngl` is given, it is used instead (also per window). The output layer goes to the GPU only when the whole last window of the master is on the GPU.

The solver uses a GPU only if it is a CUDA (or ROCm, MUSA) or a Metal device.

## Weight prefetching

With `--prefetch`, after a node computes a window and sends the result, it asks the OS (`posix_madvise(WILLNEED)`) to load the weights of its next window. The OS then reads them from disk while the other nodes compute. This helps when a node has less memory than its layers need and the weights come from the memory-mapped model file. `--force` also reads one byte of each page, so the pages are loaded at once.

Only host weights that are in the memory-mapped file profit from prefetching. Weights in repacked CPU buffers (for example AMX or the CPU repack buffers) are copies in RAM, use `--no-repack` to keep the weights in the mapped file. In the ring the nodes do not read the whole model file at load time (no `MAP_POPULATE`), each node reads only the pages of its own layers.

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
| `src/network-utils.o` | the port check is in `src/llama-prima.cpp` (`llama_prima_port_open`) |
| `common/profiler.o` | `common/prima-profiler.cpp`, the solver and the ring setup are in `common/prima.cpp` |
| `profile-tool` | `tools/profile-tool`, added once it is ported |

For the automatic layer assignment install [HiGHS](https://github.com/ERGO-Code/HiGHS), for example `brew install highs`, or build it from source:

```bash
git clone https://github.com/ERGO-Code/HiGHS.git && cd HiGHS
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
sudo cmake --install build
```

Then configure with `-DLLAMA_PRIMA_HIGHS=ON` (default `OFF`). Set `CMAKE_PREFIX_PATH` if HiGHS is in another place. Without HiGHS the automatic layer assignment splits the layers by the RAM/VRAM size of each device.

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

Example with automatic layer assignment on 3 nodes, the nodes use the same `--data-port`:

```bash
# node 2 (192.168.1.4), then node 1 (192.168.1.3)
llama-completion -m model.gguf --world 3 --rank 2 --next 192.168.1.2
llama-completion -m model.gguf --world 3 --rank 1 --next 192.168.1.4
```

```bash
# node 0 (192.168.1.2), the master
llama-completion -m model.gguf -c 4096 --world 3 --rank 0 --next 192.168.1.3 --prefetch -p "Once upon a time" -n 128
```

Options:

| option | description |
|--------|-------------|
| `-w, --world N` | number of nodes |
| `-r, --rank N` | rank of this node, 0 is the master |
| `-lw, --n-layer-window A,B,...` | layers per node in each cycle, indexed by rank |
| `-nip, --next IP` | ip address of the next node in the ring |
| `--data-port N` | base data port (default 9000), node `r` listens on `data_port + r` |
| `--prefetch` | prefetch the weights of the next window |
| `--force` | with `--prefetch`, load the prefetched pages at once |
| `-k, --n-cycles N` | number of cycles for the solver, 0 = try all (default) |
| `--master-priority F` | weight of the master in the solver, > 1 puts more layers on the master (default 1.01) |
| `-gm, --gpu-mem N` | GPU memory in GiB that the solver may use (default all) |

`--master`, `--signal-port` and `--keep-out-in-cuda` of prima.cpp are accepted, but this port does not use them.
