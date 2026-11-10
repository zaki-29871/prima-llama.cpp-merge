#include "llama-prima.h"

#include "llama-impl.h"
#include "llama-memory.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>

#ifdef LLAMA_USE_ZMQ
#include <zmq.h>
#endif

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <poll.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <unistd.h>
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

static void * zmq_new_push(void * zctx, const std::string & endp) {
    void * sock = zmq_check(zmq_socket(zctx, ZMQ_PUSH), "zmq_socket");

    // give pending messages (e.g. stop) some time to go out on exit
    const int linger = 3000;
    zmq_setsockopt(sock, ZMQ_LINGER, &linger, sizeof(linger));

    if (zmq_connect(sock, endp.c_str()) != 0) {
        const std::string err = zmq_strerror(zmq_errno());
        zmq_close(sock);
        throw std::runtime_error(format("prima: connect %s failed: %s", endp.c_str(), err.c_str()));
    }

    return sock;
}

llama_prima_net::llama_prima_net(uint32_t listen_port, const std::string & next_ip, uint32_t next_port) {
    const std::string recv_endp = format("tcp://*:%u",  listen_port);
    const std::string send_endp = format("tcp://%s:%u", next_ip.c_str(), next_port);

    zctx      = zmq_check(zmq_ctx_new(), "zmq_ctx_new");
    sock_recv = zmq_check(zmq_socket(zctx, ZMQ_PULL), "zmq_socket");

    zmq_check(zmq_bind(sock_recv, recv_endp.c_str()), ("bind " + recv_endp).c_str());
    sock_send = zmq_new_push(zctx, send_endp);

    LLAMA_LOG_INFO("%s: listen on %s, next node at %s\n", __func__, recv_endp.c_str(), send_endp.c_str());
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

void llama_prima_net::set_next(const std::string & ip, uint32_t port) {
    const std::string endp = format("tcp://%s:%u", ip.c_str(), port);

    void * sock = zmq_new_push(zctx, endp);

    // the old socket sends its pending messages in the background until the linger time ends
    zmq_close(sock_send);
    sock_send = sock;

    LLAMA_LOG_INFO("%s: next node at %s\n", __func__, endp.c_str());
}

void llama_prima_net::send_parts(const std::vector<std::pair<const void *, size_t>> & parts) {
    for (size_t i = 0; i < parts.size(); ++i) {
        const int flags = i + 1 < parts.size() ? ZMQ_SNDMORE : 0;
        if (zmq_send(sock_send, parts[i].first, parts[i].second, flags) < 0) {
            throw std::runtime_error(format("prima: zmq_send failed: %s", zmq_strerror(zmq_errno())));
        }
    }
}

std::vector<std::vector<uint8_t>> llama_prima_net::recv_parts() {
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

    return parts;
}

void llama_prima_net::forward() {
    while (true) {
        auto parts = recv_parts();

        std::vector<std::pair<const void *, size_t>> out;
        for (const auto & p : parts) {
            out.emplace_back(p.data(), p.size());
        }
        send_parts(out);

        msg_header hdr;
        if (!parts.empty() && parts[0].size() == sizeof(hdr)) {
            memcpy(&hdr, parts[0].data(), sizeof(hdr));
            if (hdr.type == LLAMA_PRIMA_MSG_STOP) {
                return;
            }
        }
    }
}

llama_prima_msg_type llama_prima_net::recv(llama_prima_data & data, std::vector<llama_prima_op> & ops, llama_prima_state * state) {
    const auto parts = recv_parts();

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
        case LLAMA_PRIMA_MSG_STATE:
            {
                if (!state) {
                    throw std::runtime_error("prima: received an unexpected state message");
                }
                state->op     = hdr.window;
                state->seq_id = (llama_seq_id) hdr.n_tokens;
                state->flags  = hdr.n_seq_ids;
                read_part(1, state->ranks, hdr.n_ops);
                if (parts.size() != 2 + (size_t) hdr.n_ops) {
                    throw std::runtime_error("prima: received a malformed state message");
                }
                state->blobs.assign(parts.begin() + 2, parts.end());
            } break;
        case LLAMA_PRIMA_MSG_STOP:
        case LLAMA_PRIMA_MSG_CTL:
            break;
        default:
            throw std::runtime_error(format("prima: received an unknown message type %u", hdr.type));
    }

    return (llama_prima_msg_type) hdr.type;
}

#else

llama_prima_net::llama_prima_net(uint32_t, const std::string &, uint32_t) {
    throw std::runtime_error("prima: llama was built without ZeroMQ, rebuild with LLAMA_PRIMA=ON and libzmq installed");
}

llama_prima_net::~llama_prima_net() = default;

void llama_prima_net::set_next(const std::string &, uint32_t) {
    throw std::runtime_error("prima: llama was built without ZeroMQ");
}

void llama_prima_net::send_parts(const std::vector<std::pair<const void *, size_t>> &) {
    throw std::runtime_error("prima: llama was built without ZeroMQ");
}

std::vector<std::vector<uint8_t>> llama_prima_net::recv_parts() {
    throw std::runtime_error("prima: llama was built without ZeroMQ");
}

void llama_prima_net::forward() {
    throw std::runtime_error("prima: llama was built without ZeroMQ");
}

llama_prima_msg_type llama_prima_net::recv(llama_prima_data &, std::vector<llama_prima_op> &, llama_prima_state *) {
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

void llama_prima_net::send_ctl(const void * data, size_t size) {
    const msg_header hdr = { LLAMA_PRIMA_MSG_CTL, 0, 0, 0, 0, 0 };

    send_parts({
        { &hdr, sizeof(hdr) },
        { data, size },
    });
}

void llama_prima_net::send_state(const llama_prima_state & state) {
    GGML_ASSERT(state.ranks.size() == state.blobs.size());

    const msg_header hdr = { LLAMA_PRIMA_MSG_STATE, state.op, (uint32_t) state.seq_id, state.flags, 0, (uint32_t) state.ranks.size() };

    std::vector<std::pair<const void *, size_t>> parts = {
        { &hdr,               sizeof(hdr) },
        { state.ranks.data(), state.ranks.size()*sizeof(uint32_t) },
    };
    for (const auto & blob : state.blobs) {
        parts.emplace_back(blob.data(), blob.size());
    }

    send_parts(parts);
}

std::vector<uint8_t> llama_prima_net::recv_ctl() {
    auto parts = recv_parts();

    msg_header hdr;
    if (parts.size() != 2 || parts[0].size() != sizeof(hdr)) {
        throw std::runtime_error("prima: received a malformed setup message");
    }
    memcpy(&hdr, parts[0].data(), sizeof(hdr));
    if (hdr.type != LLAMA_PRIMA_MSG_CTL) {
        throw std::runtime_error(format("prima: expected a setup message, got message type %u", hdr.type));
    }

    return std::move(parts[1]);
}

bool llama_prima_net::port_open(const std::string & ip, uint32_t port, int timeout_ms) {
#if defined(_WIN32)
    GGML_UNUSED(ip);
    GGML_UNUSED(port);
    GGML_UNUSED(timeout_ms);
    return true;
#else
    addrinfo hints = {};
    hints.ai_family   = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    addrinfo * res = nullptr;
    if (getaddrinfo(ip.c_str(), std::to_string(port).c_str(), &hints, &res) != 0) {
        return false;
    }

    bool ok = false;
    for (addrinfo * ai = res; ai && !ok; ai = ai->ai_next) {
        const int fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) {
            continue;
        }
        fcntl(fd, F_SETFL, fcntl(fd, F_GETFL, 0) | O_NONBLOCK);

        if (connect(fd, ai->ai_addr, ai->ai_addrlen) == 0) {
            ok = true;
        } else if (errno == EINPROGRESS) {
            pollfd pfd = { fd, POLLOUT, 0 };
            if (poll(&pfd, 1, timeout_ms) == 1) {
                int err = 0;
                socklen_t len = sizeof(err);
                ok = getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &len) == 0 && err == 0;
            }
        }
        close(fd);
    }
    freeaddrinfo(res);

    return ok;
#endif
}

void llama_prima_net::flush_ops() {
    if (pending.empty()) {
        return;
    }

    send_ops(pending);
    pending.clear();
}

//
// weight prefetching
//

std::vector<llama_prima_segment> llama_prima_make_segments(const std::vector<ggml_tensor *> & tensors) {
    std::vector<llama_prima_segment> segs;

#if !defined(_WIN32)
    const uintptr_t page = (uintptr_t) sysconf(_SC_PAGESIZE);

    for (const ggml_tensor * t : tensors) {
        if (!t || !t->data || !t->buffer || !ggml_backend_buffer_is_host(t->buffer)) {
            continue;
        }
        const uintptr_t start = (uintptr_t) t->data;
        const uintptr_t end   = start + ggml_nbytes(t);
        segs.push_back({ start - start % page, (end + page - 1) / page * page });
    }

    std::sort(segs.begin(), segs.end(), [](const llama_prima_segment & a, const llama_prima_segment & b) { return a.start < b.start; });

    std::vector<llama_prima_segment> merged;
    for (const auto & s : segs) {
        if (!merged.empty() && s.start <= merged.back().end) {
            merged.back().end = std::max(merged.back().end, s.end);
        } else {
            merged.push_back(s);
        }
    }
    segs = std::move(merged);
#else
    GGML_UNUSED(tensors);
#endif

    return segs;
}

void llama_prima_prefetch(const std::vector<llama_prima_segment> & segs, bool force) {
#if !defined(_WIN32)
    const size_t page = (size_t) sysconf(_SC_PAGESIZE);

    for (const auto & s : segs) {
        const size_t len = s.end - s.start;
        posix_madvise((void *) s.start, len, POSIX_MADV_WILLNEED);

        if (force) {
            // read one byte per page to load the pages now
            const volatile char * p = (const volatile char *) s.start;
            for (size_t off = 0; off < len; off += page) {
                (void) p[off];
            }
        }
    }
#else
    GGML_UNUSED(segs);
    GGML_UNUSED(force);
#endif
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

//
// public API
//

llama_prima_net * llama_prima_net_init(uint32_t listen_port, const char * next_ip, uint32_t next_port) {
    try {
        return new llama_prima_net(listen_port, next_ip ? next_ip : "127.0.0.1", next_port);
    } catch (const std::exception & e) {
        LLAMA_LOG_ERROR("%s: %s\n", __func__, e.what());
        return nullptr;
    }
}

void llama_prima_net_free(llama_prima_net * net) {
    delete net;
}

int32_t llama_prima_net_send(llama_prima_net * net, const void * data, size_t size) {
    try {
        net->send_ctl(data, size);
        return 0;
    } catch (const std::exception & e) {
        LLAMA_LOG_ERROR("%s: %s\n", __func__, e.what());
        return -1;
    }
}

int64_t llama_prima_net_recv(llama_prima_net * net, void * buf, size_t size) {
    try {
        const auto msg = net->recv_ctl();
        if (msg.size() > size) {
            LLAMA_LOG_ERROR("%s: message too large (%zu > %zu bytes)\n", __func__, msg.size(), size);
            return -1;
        }
        if (!msg.empty()) {
            memcpy(buf, msg.data(), msg.size());
        }
        return (int64_t) msg.size();
    } catch (const std::exception & e) {
        LLAMA_LOG_ERROR("%s: %s\n", __func__, e.what());
        return -1;
    }
}

int32_t llama_prima_net_set_next(llama_prima_net * net, const char * ip, uint32_t port) {
    try {
        net->set_next(ip, port);
        return 0;
    } catch (const std::exception & e) {
        LLAMA_LOG_ERROR("%s: %s\n", __func__, e.what());
        return -1;
    }
}

int32_t llama_prima_net_forward(llama_prima_net * net) {
    try {
        net->forward();
        return 0;
    } catch (const std::exception & e) {
        LLAMA_LOG_ERROR("%s: %s\n", __func__, e.what());
        return -1;
    }
}

bool llama_prima_port_open(const char * ip, uint32_t port, int32_t timeout_ms) {
    return llama_prima_net::port_open(ip, port, timeout_ms);
}
