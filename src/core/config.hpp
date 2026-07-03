#pragma once

#include <map>
#include <string>

#include "core/precision.hpp"

namespace wfe {

// Basit "anahtar = deger" formatli config okuyucu. '#' ve ';' yorum baslatir.
class Config {
 public:
  bool load(const std::string& path);
  bool has(const std::string& key) const { return kv_.count(key) > 0; }
  real get_real(const std::string& key, real def) const;
  int get_int(const std::string& key, int def) const;
  std::string get_str(const std::string& key, const std::string& def) const;

 private:
  std::map<std::string, std::string> kv_;
};

} // namespace wfe
