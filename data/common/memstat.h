#pragma once
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>

namespace memstat {

inline size_t read_status_kb_(const char* key) {
    std::ifstream in("/proc/self/status");
    if (!in) return 0;
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind(key, 0) == 0) {
            std::istringstream iss(line);
            std::string k, unit;
            size_t kb = 0;
            iss >> k >> kb >> unit; // kB
            return kb;
        }
    }
    return 0;
}
inline size_t rss_bytes()    { return read_status_kb_("VmRSS:");   } // 当前常驻内存
inline size_t hwm_bytes()    { return read_status_kb_("VmHWM:");   } // 历史峰值RSS
inline size_t vmsize_bytes() { return read_status_kb_("VmSize:");  } // 虚拟地址空间
inline size_t vmpeak_bytes() { return read_status_kb_("VmPeak:");  } // 虚拟内存峰值
inline size_t vmdata_bytes() { return read_status_kb_("VmData:");  } // 数据段

inline double toMB(size_t kb) { return double(kb) / 1024.0; }

inline void print_now(const std::string& tag) {
    double rss = toMB(rss_bytes());
    double hwm = toMB(hwm_bytes());
    double vss = toMB(vmsize_bytes());
    double vpk = toMB(vmpeak_bytes());
    double vdt = toMB(vmdata_bytes());
    std::cout << "[内存] " << tag
              << " RSS=" << std::fixed << std::setprecision(2) << rss << " MB"
              << " HWM=" << hwm << " MB"
              << " VSize=" << vss << " MB"
              << " VPeak=" << vpk << " MB"
              << " Data=" << vdt << " MB"
              << std::endl;
}

struct Snapshot {
    std::string tag;
    size_t rss = 0;
    size_t hwm = 0;
    size_t vmsize = 0;
    size_t vmpeak = 0;
    size_t vmdata = 0;
    long long ms_since_start = 0;
};

class Logger {
  public:
    Logger() : t0_(std::chrono::steady_clock::now()) {}

    void snap(const std::string& tag, bool also_print = true) {
        Snapshot s;
        s.tag = tag;
        s.rss = rss_bytes();
        s.hwm = hwm_bytes();
        s.vmsize = vmsize_bytes();
        s.vmpeak = vmpeak_bytes();
        s.vmdata = vmdata_bytes();
        auto now = std::chrono::steady_clock::now();
        s.ms_since_start = std::chrono::duration_cast<std::chrono::milliseconds>(now - t0_).count();
        snaps_.push_back(s);
        if (also_print) {
            std::cout << "[内存] " << tag << " (+"
                      << s.ms_since_start << " ms)"
                      << " RSS="   << std::fixed << std::setprecision(2) << toMB(s.rss)    << " MB"
                      << " HWM="   << toMB(s.hwm)    << " MB"
                      << " VSize=" << toMB(s.vmsize) << " MB"
                      << " VPeak=" << toMB(s.vmpeak) << " MB"
                      << " Data="  << toMB(s.vmdata) << " MB"
                      << std::endl;
        }
    }

    void dump_to(std::ostream& os) const {
        os << "Memory Snapshots (MB):\n";
        os << std::fixed << std::setprecision(2);
        for (auto& s : snaps_) {
            os << "  [" << s.ms_since_start << " ms] " << s.tag
               << "  RSS="   << toMB(s.rss)    << " MB"
               << " HWM="   << toMB(s.hwm)    << " MB"
               << " VSize=" << toMB(s.vmsize) << " MB"
               << " VPeak=" << toMB(s.vmpeak) << " MB"
               << " Data="  << toMB(s.vmdata) << " MB"
               << "\n";
        }
    }

    bool dump_to_file(const std::string& path) const {
        std::ofstream out(path, std::ios::out | std::ios::trunc);
        if (!out) return false;
        dump_to(out);
        return true;
    }

  private:
    std::chrono::steady_clock::time_point t0_;
    std::vector<Snapshot> snaps_;
};

} // namespace memstat