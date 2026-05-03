#include "output.hpp"
#include <cstdio>
#include <cmath>
#include <stdexcept>
#include <sys/stat.h>

namespace io {

CSVWriter::CSVWriter(const std::string& dir) : dir_(dir) {
    // Ensure trailing slash
    if (!dir_.empty() && dir_.back() != '/')
        dir_ += '/';
    // Create directory if not exists (best-effort, POSIX)
    mkdir(dir_.c_str(), 0755);
}

void CSVWriter::write_snapshot(const std::vector<State>& q,
                               int  nx, Real dx, Real t)
{
    char fname[512];
    std::snprintf(fname, sizeof(fname),
                  "%ssnap_%05d_t%.4f.csv", dir_.c_str(), snap_idx_++, (double)t);

    FILE* fp = std::fopen(fname, "w");
    if (!fp)
        throw std::runtime_error(std::string("Cannot open ") + fname);

    std::fprintf(fp, "x,h,u,t\n");
    for (int i = 0; i < nx; ++i) {
        const Real xc = (i + 0.5) * dx;
        const Real u  = q[i].u();
        std::fprintf(fp, "%.8e,%.8e,%.8e,%.8e\n",
                     (double)xc, (double)q[i].h, (double)u, (double)t);
    }
    std::fclose(fp);
}

void CSVWriter::print_perf(double runtime_ms, int nx, double tend) {
    // MCUPS = million cell-updates per second
    // We don't know #steps here, print based on CFL estimate
    std::printf("Runtime: %.2f ms\n", runtime_ms);
    (void)nx; (void)tend;
}

} // namespace io
