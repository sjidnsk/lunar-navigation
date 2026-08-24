#pragma once

#include <array>
#include <chrono>
#include <concepts>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>

#include "lunar_pure_planner_core/search_control.hpp"
#include "lunar_pure_planner_core/types/planning_request.hpp"

namespace lunar::pure_planning {

struct GlobalRoute;

namespace shared {

class GoalDistanceField;
class GlobalOccupancyProjection;
class LocalTerrainProjection;
class MapSnapshot;

struct GlobalSnapshotCacheDomain;
struct GlobalProjectionCacheDomain;
struct GlobalRouteCacheDomain;
struct LocalSnapshotCacheDomain;
struct LocalProjectionCacheDomain;
struct GoalFieldCacheDomain;

}  // namespace shared
}  // namespace lunar::pure_planning

namespace lunar::pure_planning::shared {

template <typename Domain, std::size_t SourceCount,
          std::size_t SemanticCount>
struct RevisionCacheKey final {
  static_assert(SourceCount > 0U);

  std::array<std::uint64_t, SourceCount> source_sequences{};
  std::array<std::uint64_t, SemanticCount> semantic_identities{};

  [[nodiscard]] bool cacheable() const noexcept {
    for (const std::uint64_t sequence : source_sequences) {
      if (sequence == 0U) {
        return false;
      }
    }
    return true;
  }

  bool operator==(const RevisionCacheKey&) const = default;
};

template <typename Key>
concept ImmutableActiveCacheKey = std::equality_comparable<Key> &&
    requires(const Key& key) {
      { key.cacheable() } noexcept -> std::same_as<bool>;
    };

template <typename Value>
struct ImmutableCacheBuildResult final {
  std::shared_ptr<const Value> value;
  std::string reason_code;

  [[nodiscard]] bool ok() const noexcept {
    return value != nullptr && reason_code.empty();
  }
};

template <typename Value>
struct ImmutableCacheResult final {
  std::shared_ptr<const Value> value;
  bool cache_hit{};
  std::string reason_code;

  [[nodiscard]] bool ok() const noexcept {
    return value != nullptr && reason_code.empty();
  }
};

namespace detail {

[[nodiscard]] std::string ActivePlannerCacheStopReason(
    const SearchControl& control) noexcept;

}  // namespace detail

template <ImmutableActiveCacheKey Key, typename Value>
class ImmutableActiveCacheSlot final {
 public:
  ImmutableActiveCacheSlot() = default;
  ImmutableActiveCacheSlot(const ImmutableActiveCacheSlot&) = delete;
  ImmutableActiveCacheSlot& operator=(const ImmutableActiveCacheSlot&) = delete;

  template <typename Builder>
  [[nodiscard]] ImmutableCacheResult<Value> GetOrBuild(
      const Key& key, const SearchControl& control, Builder builder) {
    if (const std::string stopped =
            detail::ActivePlannerCacheStopReason(control);
        !stopped.empty()) {
      return Failure(std::move(stopped));
    }
    if (!key.cacheable()) {
      return BuildDirect(control, builder);
    }

    std::optional<std::uint64_t> admitted_epoch;
    for (;;) {
      std::shared_ptr<Flight> flight;
      std::uint64_t build_epoch{};
      bool owns_build{};
      bool identity_was_replaced{};
      {
        const std::scoped_lock lock{mutex_};
        if (admitted_epoch.has_value() &&
            (epoch_ != *admitted_epoch || key_ != key)) {
          identity_was_replaced = true;
        } else if (key_ == key && value_ != nullptr) {
          return {.value = value_, .cache_hit = true};
        } else if (!key_.has_value() || *key_ != key) {
          key_ = key;
          value_.reset();
          flight_.reset();
          ++epoch_;
        }
        if (!identity_was_replaced) {
          build_epoch = epoch_;
          admitted_epoch = build_epoch;
          if (flight_ != nullptr) {
            flight = flight_;
          } else {
            flight = std::make_shared<Flight>();
            flight_ = flight;
            owns_build = true;
          }
        }
      }
      if (identity_was_replaced) {
        return BuildDirect(control, builder);
      }

      if (!owns_build) {
        const auto waited = WaitForBuild(flight, control);
        if (!waited.reason_code.empty() || waited.value != nullptr) {
          return waited;
        }
        continue;
      }

      ImmutableCacheBuildResult<Value> built = InvokeBuilder(control, builder);
      if (const std::string stopped =
              detail::ActivePlannerCacheStopReason(control);
          !stopped.empty()) {
        built = {.reason_code = std::move(stopped)};
      } else if (!built.ok() && built.reason_code.empty()) {
        built.reason_code = "PLANNER_ERROR";
      }

      ImmutableCacheResult<Value> owner_result{
          .value = built.value,
          .cache_hit = false,
          .reason_code = built.reason_code,
      };
      {
        const std::scoped_lock lock{mutex_};
        if (epoch_ == build_epoch && key_ == key && flight_ == flight) {
          flight_.reset();
          if (built.ok()) {
            if (value_ == nullptr) {
              value_ = built.value;
            }
            owner_result.value = value_;
          }
        }
      }
      CompleteFlight(flight, owner_result);
      return owner_result;
    }
  }

 private:
  struct Flight final {
    std::mutex mutex;
    std::condition_variable completed_condition;
    bool completed{};
    ImmutableCacheResult<Value> result;
  };

  template <typename Builder>
  [[nodiscard]] static ImmutableCacheBuildResult<Value> InvokeBuilder(
      const SearchControl& control, Builder& builder) noexcept {
    try {
      return std::invoke(builder, control);
    } catch (...) {
      return {.reason_code = "PLANNER_ERROR"};
    }
  }

  template <typename Builder>
  [[nodiscard]] static ImmutableCacheResult<Value> BuildDirect(
      const SearchControl& control, Builder& builder) {
    ImmutableCacheBuildResult<Value> built = InvokeBuilder(control, builder);
    if (const std::string stopped =
            detail::ActivePlannerCacheStopReason(control);
        !stopped.empty()) {
      return Failure(std::move(stopped));
    }
    if (!built.ok()) {
      return Failure(built.reason_code.empty() ? "PLANNER_ERROR"
                                               : std::move(built.reason_code));
    }
    return {.value = std::move(built.value), .cache_hit = false};
  }

  [[nodiscard]] static ImmutableCacheResult<Value> WaitForBuild(
      const std::shared_ptr<Flight>& flight,
      const SearchControl& control) {
    std::unique_lock lock{flight->mutex};
    while (!flight->completed) {
      if (const std::string stopped =
              detail::ActivePlannerCacheStopReason(control);
          !stopped.empty()) {
        return Failure(std::move(stopped));
      }
      flight->completed_condition.wait_for(lock, std::chrono::milliseconds{1});
    }
    if (flight->result.ok()) {
      ImmutableCacheResult<Value> result = flight->result;
      result.cache_hit = true;
      return result;
    }
    return {};
  }

  static void CompleteFlight(
      const std::shared_ptr<Flight>& flight,
      const ImmutableCacheResult<Value>& result) {
    {
      const std::scoped_lock lock{flight->mutex};
      flight->result = result;
      flight->completed = true;
    }
    flight->completed_condition.notify_all();
  }

  [[nodiscard]] static ImmutableCacheResult<Value> Failure(
      std::string reason_code) {
    return {.reason_code = std::move(reason_code)};
  }

  std::mutex mutex_;
  std::optional<Key> key_;
  std::shared_ptr<const Value> value_;
  std::shared_ptr<Flight> flight_;
  std::uint64_t epoch_{};
};

using GlobalSnapshotCacheKey =
    RevisionCacheKey<GlobalSnapshotCacheDomain, 1U, 0U>;
using GlobalProjectionCacheKey =
    RevisionCacheKey<GlobalProjectionCacheDomain, 1U, 3U>;
using GlobalRouteCacheKey =
    RevisionCacheKey<GlobalRouteCacheDomain, 3U, 5U>;
using LocalSnapshotCacheKey =
    RevisionCacheKey<LocalSnapshotCacheDomain, 1U, 0U>;
using LocalProjectionCacheKey =
    RevisionCacheKey<LocalProjectionCacheDomain, 1U, 1U>;
using GoalFieldCacheKey =
    RevisionCacheKey<GoalFieldCacheDomain, 1U, 4U>;

[[nodiscard]] std::uint64_t StableCapabilityFingerprint(
    const PlatformCapability& capability) noexcept;

[[nodiscard]] GlobalSnapshotCacheKey MakeGlobalSnapshotCacheKey(
    std::uint64_t global_map_sequence) noexcept;
[[nodiscard]] GlobalProjectionCacheKey MakeGlobalProjectionCacheKey(
    std::uint64_t global_map_sequence, std::int32_t occupancy_threshold,
    double inflation_m, std::uint64_t capability_fingerprint) noexcept;
[[nodiscard]] GlobalRouteCacheKey MakeGlobalRouteCacheKey(
    const PlanningRequest& input, double inflation_m,
    std::uint64_t capability_fingerprint) noexcept;
[[nodiscard]] LocalSnapshotCacheKey MakeLocalSnapshotCacheKey(
    std::uint64_t local_map_sequence) noexcept;
[[nodiscard]] LocalProjectionCacheKey MakeLocalProjectionCacheKey(
    std::uint64_t local_map_sequence, double occupancy_threshold) noexcept;
[[nodiscard]] GoalFieldCacheKey MakeGoalFieldCacheKey(
    std::uint64_t local_map_sequence, double occupancy_threshold,
    std::uint64_t capability_fingerprint, const LocalGoalSet& goals,
    const AnytimeSearchConfig& search) noexcept;

class ActivePlannerCache final {
 public:
  using GlobalSnapshotSlot =
      ImmutableActiveCacheSlot<GlobalSnapshotCacheKey, MapSnapshot>;
  using GlobalProjectionSlot = ImmutableActiveCacheSlot<
      GlobalProjectionCacheKey, GlobalOccupancyProjection>;
  using GlobalRouteSlot =
      ImmutableActiveCacheSlot<GlobalRouteCacheKey, GlobalRoute>;
  using LocalSnapshotSlot =
      ImmutableActiveCacheSlot<LocalSnapshotCacheKey, MapSnapshot>;
  using LocalProjectionSlot =
      ImmutableActiveCacheSlot<LocalProjectionCacheKey, LocalTerrainProjection>;
  using GoalFieldSlot =
      ImmutableActiveCacheSlot<GoalFieldCacheKey, GoalDistanceField>;

  [[nodiscard]] GlobalSnapshotSlot& global_snapshot() noexcept {
    return global_snapshot_;
  }
  [[nodiscard]] GlobalProjectionSlot& global_projection() noexcept {
    return global_projection_;
  }
  [[nodiscard]] GlobalRouteSlot& global_route() noexcept {
    return global_route_;
  }
  [[nodiscard]] LocalSnapshotSlot& local_snapshot() noexcept {
    return local_snapshot_;
  }
  [[nodiscard]] LocalProjectionSlot& local_projection() noexcept {
    return local_projection_;
  }
  [[nodiscard]] GoalFieldSlot& goal_field() noexcept { return goal_field_; }

 private:
  GlobalSnapshotSlot global_snapshot_;
  GlobalProjectionSlot global_projection_;
  GlobalRouteSlot global_route_;
  LocalSnapshotSlot local_snapshot_;
  LocalProjectionSlot local_projection_;
  GoalFieldSlot goal_field_;
};

}  // namespace lunar::pure_planning::shared
