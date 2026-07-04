#pragma once

#include <map>
#include <set>
#include <string>
#include <vector>

#include "core/precision.hpp"

namespace wfe {

// Basit "anahtar = deger" formatli config okuyucu. '#' ve ';' yorum baslatir.
// Okunan anahtarlar izlenir: unused() hic okunmamis anahtarlari dondurur
// (yazim hatalarini yakalar).
class Config {
 public:
  bool load(const std::string& path);
  bool has(const std::string& key) const { return kv_.count(key) > 0; }
  real get_real(const std::string& key, real def) const;
  int get_int(const std::string& key, int def) const;
  std::string get_str(const std::string& key, const std::string& def) const;
  const std::map<std::string, std::string>& raw() const { return kv_; }
  // Hic getter'la okunmamis anahtarlar (prep'e ait proj_* haric).
  std::vector<std::string> unused() const;

 private:
  std::map<std::string, std::string> kv_;
  mutable std::set<std::string> used_;
};

} // namespace wfe
