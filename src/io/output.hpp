#pragma once
#include "../types.hpp"
#include <string>
#include <vector>

namespace io {

class CSVWriter {
public:
    explicit CSVWriter(const std::string& dir);

    // Write one snapshot: x_centres, states, simulation time, nx, dx
    void write_snapshot(const std::vector<State>& q,
                        int    nx,
                        Real   dx,
                        Real   t);

    // Print performance summary to stdout
    static void print_perf(double runtime_ms, int nx, double tend);

private:
    std::string dir_;
    int         snap_idx_ = 0;
};

} // namespace io
