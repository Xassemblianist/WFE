#include "core/config.hpp"

#include <cstdlib>
#include <fstream>

namespace wfe {

namespace {
std::string trim(const std::string& s) {
  size_t a = s.find_first_not_of(" \t\r\n");
  if (a == std::string::npos) return "";
  size_t b = s.find_last_not_of(" \t\r\n");
  return s.substr(a, b - a + 1);
}
} // namespace

bool Config::load(const std::string& path) {
  std::ifstream f(path);
  if (!f) return false;
  std::string line;
  while (std::getline(f, line)) {
    size_t c = line.find_first_of("#;");
    if (c != std::string::npos) line = line.substr(0, c);
    size_t eq = line.find('=');
    if (eq == std::string::npos) continue;
    std::string key = trim(line.substr(0, eq));
    std::string val = trim(line.substr(eq + 1));
    if (!key.empty()) kv_[key] = val;
  }
  return true;
}

real Config::get_real(const std::string& key, real def) const {
  used_.insert(key);
  auto it = kv_.find(key);
  return it == kv_.end() ? def : (real)std::atof(it->second.c_str());
}

int Config::get_int(const std::string& key, int def) const {
  used_.insert(key);
  auto it = kv_.find(key);
  return it == kv_.end() ? def : std::atoi(it->second.c_str());
}

std::string Config::get_str(const std::string& key, const std::string& def) const {
  used_.insert(key);
  auto it = kv_.find(key);
  return it == kv_.end() ? def : it->second;
}

std::vector<std::string> Config::unused() const {
  std::vector<std::string> out;
  for (const auto& [k, v] : kv_) {
    if (used_.count(k)) continue;
    if (k.rfind("proj_", 0) == 0) continue;  // yalniz prep_gfs.py okur
    out.push_back(k);
  }
  return out;
}

} // namespace wfe
