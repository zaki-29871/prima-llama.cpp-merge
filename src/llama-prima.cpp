#include "llama-prima.h"

#include "llama-impl.h"
#include "llama-memory.h"

#include <cstring>
#include <mutex>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>

#ifdef LLAMA_USE_ZMQ
#include <zmq.h>
#endif

std::vector<llama_prima_window> llama_prima_make_windows(uint32_t n_layer, uint32_t n_world, const uint32_t * n_layer_window) {
    if (n_world < 1 || n_world > 32) {
        throw std::runtime_error(format("prima: n_world must be in [1, 32], got %u", n_world));
    }

    std::vector<uint32_t> w(n_world, 0);

    bool all_zero = true;
    for (uint32_t r = 0; r < n_world; ++r) {
        w[r] = n_layer_window ? n_layer_window[r] : 0;
        all_zero = all_zero && w[r] == 0;
    }

    if (all_zero) {
        // even split, one cycle
        for (uint32_t r = 0; r < n_world; ++r) {
            w[r] = n_layer / n_world + (r < n_layer % n_world ? 1 : 0);
        }
    }

    uint32_t cycle = 0;
    for (uint32_t r = 0; r < n_world; ++r) {
        if (w[r] == 0) {
            throw std::runtime_error(format("prima: node %u has no layers, each node needs at least one layer", r));
        }
        cycle += w[r];
    }

    if (n_layer % cycle != 0) {
        throw std::runtime_error(format("prima: the sum of the layer windows (%u) must divide the number of layers (%u)", cycle, n_layer));
    }

    std::vector<llama_prima_window> res;

    int32_t il = 0;
    for (uint32_t c = 0; c < n_layer / cycle; ++c) {
        for (uint32_t i = 1; i <= n_world; ++i) {
            const uint32_t r = i % n_world;
            res.push_back({ r, il, il + (int32_t) w[r] });
            il += w[r];
        }
    }

    return res;
}

void llama_prima_op_apply(llama_memory_i * mem, const llama_prima_op & op) {
    if (!mem) {
        return;
    }

    switch (op.type) {
        case LLAMA_PRIMA_OP_CLEAR:    mem->clear(op.arg != 0); break;
        case LLAMA_PRIMA_OP_SEQ_RM:   mem->seq_rm  (op.seq_id, op.p0, op.p1); break;
        case LLAMA_PRIMA_OP_SEQ_CP:   mem->seq_cp  (op.seq_id, op.seq_id_dst, op.p0, op.p1); break;
        case LLAMA_PRIMA_OP_SEQ_KEEP: mem->seq_keep(op.seq_id); break;
        case LLAMA_PRIMA_OP_SEQ_ADD:  mem->seq_add (op.seq_id, op.p0, op.p1, op.arg); break;
        case LLAMA_PRIMA_OP_SEQ_DIV:  mem->seq_div (op.seq_id, op.p0, op.p1, op.arg); break;
        default:
            LLAMA_LOG_ERROR("%s: unknown memory op %d\n", __func__, op.type);
    }
}

//
// wire format: a multipart message, the first part is the header
//

namespace {

struct msg_header {
    uint32_t type;
    uint32_t window;
    uint32_t n_tokens;
    uint32_t n_seq_ids;
    uint32_t n_embd;
    uint32_t n_ops;
};

}

#ifdef LLAMA_USE_ZMQ

static void * zmq_check(void * p, const char * what) {
    if (!p) {
        throw std::runtime_error(format("prima: %s failed: %s", what, zmq_strerror(zmq_errno())));
    }
    return p;
}

static void zmq_check(int rc, const char * what) {
    if (rc != 0) {
        throw std::runtime_error(format("prima: %s failed: %s", what, zmq_strerror(zmq_errno())));
    }
}

llama_prima_net::llama_prima_net(uint32_t n_world, uint32_t rank, const std::string & next_ip, uint32_t data_port) : n_world(n_world), rank(rank) {
    const uint32_t next_rank = (rank + 1) % n_world;

    const std::string recv_endp = format("tcp://*:%u",   data_port + rank);
    const std::string send_endp = format("tcp://%s:%u", next_ip.c_str(), data_port + next_rank);

    zctx      = zmq_check(zmq_ctx_new(), "zmq_ctx_new");
    sock_recv = zmq_check(zmq_socket(zctx, ZMQ_PULL), "zmq_socket");
    sock_send = zmq_check(zmq_socket(zctx, ZMQ_PUSH), "zmq_socket");

    // give pending messages (e.g. stop) some time to go out on exit
    const int linger = 2000;
    zmq_setsockopt(sock_send, ZMQ_LINGER, &linger, sizeof(linger));

    zmq_check(zmq_bind   (sock_recv, recv_endp.c_str()), ("bind " + recv_endp).c_str());
    zmq_check(zmq_connect(sock_send, send_endp.c_str()), ("connect " + send_endp).c_str());

    LLAMA_LOG_INFO("%s: rank %u/%u, listen on %s, next node at %s\n", __func__, rank, n_world, recv_endp.c_str(), send_endp.c_str());
}

llama_prima_net::~llama_prima_net() {
    if (sock_send) {
        zmq_close(sock_send);
    }
    if (sock_recv) {
        zmq_close(sock_recv);
    }
    if (zctx) {
        zmq_ctx_term(zctx);
    }
}

void llama_prima_net::send_parts(const std::vector<std::pair<const void *, size_t>> & parts) {
    for (size_t i = 0; i < parts.size(); ++i) {
        const int flags = i + 1 < parts.size() ? ZMQ_SNDMORE : 0;
        if (zmq_send(sock_send, parts[i].first, parts[i].second, flags) < 0) {
            throw std::runtime_error(format("prima: zmq_send failed: %s", zmq_strerror(zmq_errno())));
        }
    }
}

llama_prima_msg_type llama_prima_net::recv(llama_prima_data & data, std::vector<llama_prima_op> & ops) {
    std::vector<std::vector<uint8_t>> parts;

    int more = 1;
    while (more) {
        zmq_msg_t msg;
        zmq_msg_init(&msg);
        if (zmq_msg_recv(&msg, sock_recv, 0) < 0) {
            zmq_msg_close(&msg);
            throw std::runtime_error(format("prima: zmq_msg_recv failed: %s", zmq_strerror(zmq_errno())));
        }
        const uint8_t * p = (const uint8_t *) zmq_msg_data(&msg);
        parts.emplace_back(p, p + zmq_msg_size(&msg));
        more = zmq_msg_more(&msg);
        zmq_msg_close(&msg);
    }

    if (parts.empty() || parts[0].size() != sizeof(msg_header)) {
        throw std::runtime_error("prima: received a malformed message");
    }

    msg_header hdr;
    memcpy(&hdr, parts[0].data(), sizeof(hdr));

    auto read_part = [&](size_t i, auto & dst, size_t n) {
        using T = typename std::remove_reference_t<decltype(dst)>::value_type;
        if (i >= parts.size() || parts[i].size() != n*sizeof(T)) {
            throw std::runtime_error(format("prima: received a malformed message part %zu", i));
        }
        dst.resize(n);
        if (n > 0) {
            memcpy(dst.data(), parts[i].data(), n*sizeof(T));
        }
    };

    switch (hdr.type) {
        case LLAMA_PRIMA_MSG_DATA:
            {
                data.window = hdr.window;
                data.n_embd = hdr.n_embd;
                read_part(1, data.pos,      hdr.n_tokens);
                read_part(2, data.n_seq_id, hdr.n_tokens);
                read_part(3, data.seq_id,   hdr.n_seq_ids);
                read_part(4, data.output,   hdr.n_tokens);
                read_part(5, data.hidden,   (size_t) hdr.n_tokens*hdr.n_embd);
            } break;
        case LLAMA_PRIMA_MSG_OPS:
            {
                read_part(1, ops, hdr.n_ops);
            } break;
        case LLAMA_PRIMA_MSG_STOP:
            break;
        default:
            throw std::runtime_error(format("prima: received an unknown message type %u", hdr.type));
    }

    return (llama_prima_msg_type) hdr.type;
}

#else

llama_prima_net::llama_prima_net(uint32_t n_world, uint32_t rank, const std::string &, uint32_t) : n_world(n_world), rank(rank) {
    throw std::runtime_error("prima: llama was built without ZeroMQ, rebuild with LLAMA_PRIMA=ON and libzmq installed");
}

llama_prima_net::~llama_prima_net() = default;

void llama_prima_net::send_parts(const std::vector<std::pair<const void *, size_t>> &) {
    throw std::runtime_error("prima: llama was built without ZeroMQ");
}

llama_prima_msg_type llama_prima_net::recv(llama_prima_data &, std::vector<llama_prima_op> &) {
    throw std::runtime_error("prima: llama was built without ZeroMQ");
}

#endif

void llama_prima_net::send_data(const llama_prima_data & data) {
    const uint32_t n_tokens = data.n_tokens();

    GGML_ASSERT(data.pos.size()    == n_tokens);
    GGML_ASSERT(data.output.size() == n_tokens);
    GGML_ASSERT(data.hidden.size() == (size_t) n_tokens*data.n_embd);

    const msg_header hdr = { LLAMA_PRIMA_MSG_DATA, data.window, n_tokens, (uint32_t) data.seq_id.size(), data.n_embd, 0 };

    send_parts({
        { &hdr,                 sizeof(hdr) },
        { data.pos.data(),      data.pos.size()*sizeof(llama_pos) },
        { data.n_seq_id.data(), data.n_seq_id.size()*sizeof(int32_t) },
        { data.seq_id.data(),   data.seq_id.size()*sizeof(llama_seq_id) },
        { data.output.data(),   data.output.size()*sizeof(int8_t) },
        { data.hidden.data(),   data.hidden.size()*sizeof(float) },
    });
}

void llama_prima_net::send_ops(const std::vector<llama_prima_op> & ops) {
    const msg_header hdr = { LLAMA_PRIMA_MSG_OPS, 0, 0, 0, 0, (uint32_t) ops.size() };

    send_parts({
        { &hdr,       sizeof(hdr) },
        { ops.data(), ops.size()*sizeof(llama_prima_op) },
    });
}

void llama_prima_net::send_stop() {
    const msg_header hdr = { LLAMA_PRIMA_MSG_STOP, 0, 0, 0, 0, 0 };

    send_parts({ { &hdr, sizeof(hdr) } });
}

void llama_prima_net::flush_ops() {
    if (pending.empty()) {
        return;
    }

    send_ops(pending);
    pending.clear();
}

//
// memory registry
//

static std::mutex g_prima_mem_mutex;
static std::unordered_map<const llama_memory_i *, llama_prima_net *> g_prima_mem;

void llama_prima_mem_register(const llama_memory_i * mem, llama_prima_net * net) {
    std::lock_guard<std::mutex> lock(g_prima_mem_mutex);
    g_prima_mem[mem] = net;
}

void llama_prima_mem_unregister(const llama_memory_i * mem) {
    std::lock_guard<std::mutex> lock(g_prima_mem_mutex);
    g_prima_mem.erase(mem);
}

void llama_prima_mem_record(const llama_memory_i * mem, const llama_prima_op & op) {
    std::lock_guard<std::mutex> lock(g_prima_mem_mutex);
    if (g_prima_mem.empty()) {
        return;
    }
    auto it = g_prima_mem.find(mem);
    if (it != g_prima_mem.end()) {
        it->second->record(op);
    }
}
