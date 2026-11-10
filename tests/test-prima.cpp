// unit tests for the prima.cpp layer layout

#include "../src/llama-prima.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond); exit(1); } } while (0)

static bool make_windows_throws(uint32_t n_layer, uint32_t n_world, std::vector<uint32_t> lw) {
    lw.resize(32, 0);
    try {
        llama_prima_make_windows(n_layer, n_world, lw.data());
    } catch (const std::runtime_error &) {
        return true;
    }
    return false;
}

int main() {
    // even split, one cycle, the master runs last and gets the extra layer
    {
        const auto w = llama_prima_make_windows(10, 3, nullptr);
        CHECK(w.size() == 3);
        CHECK(w[0].rank == 1 && w[0].il0 == 0 && w[0].il1 == 3);
        CHECK(w[1].rank == 2 && w[1].il0 == 3 && w[1].il1 == 6);
        CHECK(w[2].rank == 0 && w[2].il0 == 6 && w[2].il1 == 10);
    }

    // two cycles
    {
        std::vector<uint32_t> lw = { 8, 8 };
        lw.resize(32, 0);
        const auto w = llama_prima_make_windows(32, 2, lw.data());
        CHECK(w.size() == 4);
        CHECK(w[0].rank == 1 && w[0].il0 == 0  && w[0].il1 == 8);
        CHECK(w[1].rank == 0 && w[1].il0 == 8  && w[1].il1 == 16);
        CHECK(w[2].rank == 1 && w[2].il0 == 16 && w[2].il1 == 24);
        CHECK(w[3].rank == 0 && w[3].il0 == 24 && w[3].il1 == 32);
    }

    CHECK(make_windows_throws(32, 2, { 5, 5 }));   // 10 does not divide 32
    CHECK(make_windows_throws(32, 3, { 16, 16 })); // node 2 has no layers
    CHECK(make_windows_throws(32, 33, {}));

    printf("test-prima: OK\n");

    return 0;
}
