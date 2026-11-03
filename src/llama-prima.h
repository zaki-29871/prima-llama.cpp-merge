#pragma once

// prima.cpp piped-ring parallelism
//
// the layers are split into windows that run on the nodes of a ring
// in each cycle the windows go in rank order 1, 2, ..., n_world - 1, 0
// node 0 (master) also computes the input embeddings and the output head

#include "llama.h"

#include <cstdint>
#include <string>
#include <vector>

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

class llama_prima_net {
public:
    llama_prima_net(uint32_t n_world, uint32_t rank, const std::string & next_ip, uint32_t data_port);
    ~llama_prima_net();

    void send_data(const llama_prima_data & data);
    void send_ops (const std::vector<llama_prima_op> & ops);
    void send_stop();

    // blocking, returns the message type
    llama_prima_msg_type recv(llama_prima_data & data, std::vector<llama_prima_op> & ops);

    // record a memory op on the master, sent before the next ubatch
    void record(const llama_prima_op & op) { pending.push_back(op); }
    void flush_ops();

    const uint32_t n_world;
    const uint32_t rank;

private:
    void send_parts(const std::vector<std::pair<const void *, size_t>> & parts);

    void * zctx      = nullptr;
    void * sock_send = nullptr;
    void * sock_recv = nullptr;

    std::vector<llama_prima_op> pending;
};

// the master registers its memory so that the llama_memory_* API forwards the ops to the ring
void llama_prima_mem_register  (const llama_memory_i * mem, llama_prima_net * net);
void llama_prima_mem_unregister(const llama_memory_i * mem);
void llama_prima_mem_record    (const llama_memory_i * mem, const llama_prima_op & op);
