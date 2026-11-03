#pragma once

// prima.cpp piped-ring parallelism
//
// the layers are split into windows that run on the nodes of a ring
// in each cycle the windows go in rank order 1, 2, ..., n_world - 1, 0
// node 0 (master) also computes the input embeddings and the output head

#include "llama.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

struct ggml_tensor;
struct llama_memory_i;

struct llama_prima_window {
    uint32_t rank;
    int32_t  il0; // first layer
    int32_t  il1; // one past the last layer
};

// throws on an invalid layout
std::vector<llama_prima_window> llama_prima_make_windows(uint32_t n_layer, uint32_t n_world, const uint32_t * n_layer_window);

enum llama_prima_msg_type : uint32_t {
    LLAMA_PRIMA_MSG_NONE = 0,
    LLAMA_PRIMA_MSG_DATA = 1, // hidden state of a ubatch
    LLAMA_PRIMA_MSG_OPS  = 2, // memory operations
    LLAMA_PRIMA_MSG_STOP = 3,
    LLAMA_PRIMA_MSG_CTL  = 4, // setup messages (device info, layer setup, topology)
};

enum llama_prima_op_type : int32_t {
    LLAMA_PRIMA_OP_CLEAR    = 0,
    LLAMA_PRIMA_OP_SEQ_RM   = 1,
    LLAMA_PRIMA_OP_SEQ_CP   = 2,
    LLAMA_PRIMA_OP_SEQ_KEEP = 3,
    LLAMA_PRIMA_OP_SEQ_ADD  = 4,
    LLAMA_PRIMA_OP_SEQ_DIV  = 5,
};

struct llama_prima_op {
    int32_t type;
    int32_t seq_id;
    int32_t seq_id_dst;
    int32_t p0;
    int32_t p1;
    int32_t arg; // delta for seq_add, d for seq_div, data flag for clear
};

void llama_prima_op_apply(llama_memory_i * mem, const llama_prima_op & op);

struct llama_prima_data {
    uint32_t window = 0; // index of the window that consumes this data
    uint32_t n_embd = 0;

    // ubatch layout, in the order of the master ubatch
    std::vector<llama_pos>    pos;
    std::vector<int32_t>      n_seq_id;
    std::vector<llama_seq_id> seq_id; // flattened
    std::vector<int8_t>       output;

    std::vector<float> hidden; // [n_embd, n_tokens]

    uint32_t n_tokens() const { return (uint32_t) n_seq_id.size(); }
};

struct llama_prima_net {
public:
    // listen on listen_port, send to next_ip:next_port
    llama_prima_net(uint32_t listen_port, const std::string & next_ip, uint32_t next_port);
    ~llama_prima_net();

    void send_data(const llama_prima_data & data);
    void send_ops (const std::vector<llama_prima_op> & ops);
    void send_stop();
    void send_ctl (const void * data, size_t size);

    // blocking, returns the message type
    llama_prima_msg_type recv(llama_prima_data & data, std::vector<llama_prima_op> & ops);

    // blocking, throws if the message is not a setup message
    std::vector<uint8_t> recv_ctl();

    // connect to another next node, pending messages on the old connection still go out
    void set_next(const std::string & ip, uint32_t port);

    // forward all messages to the next node until a stop message passes
    void forward();

    // record a memory op on the master, sent before the next ubatch
    void record(const llama_prima_op & op) { pending.push_back(op); }
    void flush_ops();

    // check if a TCP port accepts connections
    static bool port_open(const std::string & ip, uint32_t port, int timeout_ms);

private:
    void send_parts(const std::vector<std::pair<const void *, size_t>> & parts);
    std::vector<std::vector<uint8_t>> recv_parts();

    void * zctx      = nullptr;
    void * sock_send = nullptr;
    void * sock_recv = nullptr;

    std::vector<llama_prima_op> pending;
};

//
// weight prefetching
//

struct llama_prima_segment {
    uintptr_t start;
    uintptr_t end;
};

// page aligned and merged memory ranges of the host tensors
std::vector<llama_prima_segment> llama_prima_make_segments(const std::vector<ggml_tensor *> & tensors);

// ask the OS to load the pages, with force the pages are also touched
void llama_prima_prefetch(const std::vector<llama_prima_segment> & segs, bool force);

// the master registers its memory so that the llama_memory_* API forwards the ops to the ring
void llama_prima_mem_register  (const llama_memory_i * mem, llama_prima_net * net);
void llama_prima_mem_unregister(const llama_memory_i * mem);
void llama_prima_mem_record    (const llama_memory_i * mem, const llama_prima_op & op);
