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

}  // namespace lunar::pure_planning::shared
