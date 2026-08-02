#pragma once

#include <bit>
#include <cmath>
#include <compare>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "lunar_path_planner/v3/contracts/base_types.hpp"

namespace lunar::planning::v3 {

struct SafeProjectionCacheKey final {
  std::string map_snapshot_id;
  std::string map_revision;
  std::string layer_manifest_hash;
  ContentRef safety_capability_ref;
  ContentRef algorithm_config_ref;
  std::optional<ContentRef> learned_cost_snapshot_ref;
  std::string frame_id;
  std::uint64_t resolution_ieee754_bits{};
  std::string error_bound_model_id;

  auto operator<=>(const SafeProjectionCacheKey&) const = default;
};

[[nodiscard]] inline SafeProjectionCacheKey MakeSafeProjectionCacheKey(
    std::string map_snapshot_id,
    std::string map_revision,
    std::string layer_manifest_hash,
    ContentRef safety_capability_ref,
    ContentRef algorithm_config_ref,
    std::optional<ContentRef> learned_cost_snapshot_ref,
    std::string frame_id,
    double resolution_m,
    std::string error_bound_model_id) {
  if (!std::isfinite(resolution_m)) {
    throw std::invalid_argument{
        "safe projection cache resolution must be finite"};
  }
  const double normalized_resolution =
      resolution_m == 0.0 ? 0.0 : resolution_m;
  return {
      .map_snapshot_id = std::move(map_snapshot_id),
      .map_revision = std::move(map_revision),
      .layer_manifest_hash = std::move(layer_manifest_hash),
      .safety_capability_ref = std::move(safety_capability_ref),
      .algorithm_config_ref = std::move(algorithm_config_ref),
      .learned_cost_snapshot_ref = std::move(learned_cost_snapshot_ref),
      .frame_id = std::move(frame_id),
      .resolution_ieee754_bits =
          std::bit_cast<std::uint64_t>(normalized_resolution),
      .error_bound_model_id = std::move(error_bound_model_id),
  };
}

template <class Key, class Value>
  requires std::totally_ordered<Key>
class DeterministicCache final {
 public:
  explicit DeterministicCache(std::size_t capacity) : capacity_{capacity} {
    if (capacity_ == 0U) {
      throw std::invalid_argument{
          "deterministic cache capacity must be greater than zero"};
    }
  }

  [[nodiscard]] std::shared_ptr<const Value> Get(const Key& key) const {
    const std::shared_lock lock{mutex_};
    const auto entry = entries_.find(key);
    return entry == entries_.end() ? nullptr : entry->second;
  }

  void Publish(const Key& key, std::shared_ptr<const Value> value) {
    if (value == nullptr) {
      throw std::invalid_argument{
          "deterministic cache publication must not be null"};
    }

    const std::unique_lock lock{mutex_};
    entries_.insert_or_assign(key, std::move(value));
    if (entries_.size() > capacity_) {
      entries_.erase(std::prev(entries_.end()));
    }
  }

  [[nodiscard]] std::vector<Key> Keys() const {
    const std::shared_lock lock{mutex_};
    std::vector<Key> keys;
    keys.reserve(entries_.size());
    for (const auto& [key, value] : entries_) {
      static_cast<void>(value);
      keys.push_back(key);
    }
    return keys;
  }

 private:
  const std::size_t capacity_;
  mutable std::shared_mutex mutex_;
  std::map<Key, std::shared_ptr<const Value>> entries_;
};

}  // namespace lunar::planning::v3
