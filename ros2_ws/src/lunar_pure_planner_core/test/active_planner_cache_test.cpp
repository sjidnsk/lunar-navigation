#include "shared/active_planner_cache.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <stop_token>
#include <vector>

#include <gtest/gtest.h>

namespace lunar::pure_planning::shared {
namespace {

using namespace std::chrono_literals;

struct TestDomain final {};
using TestKey = RevisionCacheKey<TestDomain, 2U, 3U>;

struct TestArtifact final {
  int value{};
};

[[nodiscard]] GoalRegion CacheGoal(const char* id, const double x) {
  return GoalRegion{
      .goal_id = id,
      .target = PointGoal{
          .position_m = {.x = x, .y = 1.0, .z = 0.0},
          .tolerance_m = 0.1,
      },
  };
}

[[nodiscard]] TestKey Key(
    const std::uint64_t first_sequence = 1U,
    const std::uint64_t second_sequence = 2U,
    const std::array<std::uint64_t, 3U> semantics = {3U, 4U, 5U}) {
  return {
      .source_sequences = {first_sequence, second_sequence},
      .semantic_identities = semantics,
  };
}

[[nodiscard]] ImmutableCacheBuildResult<TestArtifact> Built(const int value) {
  return {.value = std::make_shared<const TestArtifact>(TestArtifact{value})};
}

TEST(ActivePlannerCache, ColdThenWarmReturnsTheSameImmutableArtifact) {
  ImmutableActiveCacheSlot<TestKey, TestArtifact> cache;
  std::size_t builds{};
  const auto builder = [&](const SearchControl&) {
    ++builds;
    return Built(17);
  };

  const auto cold = cache.GetOrBuild(Key(), {}, builder);
  const auto warm = cache.GetOrBuild(Key(), {}, builder);

  ASSERT_TRUE(cold.ok()) << cold.reason_code;
  ASSERT_TRUE(warm.ok()) << warm.reason_code;
  EXPECT_FALSE(cold.cache_hit);
  EXPECT_TRUE(warm.cache_hit);
  EXPECT_EQ(builds, 1U);
  EXPECT_EQ(cold.value, warm.value);
  EXPECT_EQ(cold.value->value, 17);
}

TEST(ActivePlannerCache, AZeroSourceSequenceAlwaysBypassesTheSlot) {
  for (const TestKey non_cacheable : {Key(0U, 2U), Key(1U, 0U)}) {
    ImmutableActiveCacheSlot<TestKey, TestArtifact> cache;
    std::size_t builds{};
    const auto builder = [&](const SearchControl&) {
      return Built(static_cast<int>(++builds));
    };

    const auto resident = cache.GetOrBuild(Key(), {}, builder);
    const auto first = cache.GetOrBuild(non_cacheable, {}, builder);
    const auto second = cache.GetOrBuild(non_cacheable, {}, builder);
    const auto resident_again = cache.GetOrBuild(Key(), {}, builder);

    ASSERT_TRUE(resident.ok()) << resident.reason_code;
    ASSERT_TRUE(first.ok()) << first.reason_code;
    ASSERT_TRUE(second.ok()) << second.reason_code;
    ASSERT_TRUE(resident_again.ok()) << resident_again.reason_code;
    EXPECT_FALSE(first.cache_hit);
    EXPECT_FALSE(second.cache_hit);
    EXPECT_TRUE(resident_again.cache_hit);
    EXPECT_EQ(builds, 3U);
    EXPECT_NE(first.value, second.value);
    EXPECT_EQ(resident_again.value, resident.value);
    EXPECT_EQ(first.value->value, 2);
    EXPECT_EQ(second.value->value, 3);
  }
}

TEST(ActivePlannerCache, EverySourceAndSemanticIdentityParticipatesInTheKey) {
  const std::vector<TestKey> changed_keys{
      Key(9U, 2U), Key(1U, 9U), Key(1U, 2U, {9U, 4U, 5U}),
      Key(1U, 2U, {3U, 9U, 5U}), Key(1U, 2U, {3U, 4U, 9U}),
  };
  for (const TestKey& changed : changed_keys) {
    ImmutableActiveCacheSlot<TestKey, TestArtifact> cache;
    std::size_t builds{};
    const auto builder = [&](const SearchControl&) {
      return Built(static_cast<int>(++builds));
    };

    const auto original = cache.GetOrBuild(Key(), {}, builder);
    const auto replacement = cache.GetOrBuild(changed, {}, builder);

    ASSERT_TRUE(original.ok()) << original.reason_code;
    ASSERT_TRUE(replacement.ok()) << replacement.reason_code;
    EXPECT_FALSE(original.cache_hit);
    EXPECT_FALSE(replacement.cache_hit);
    EXPECT_EQ(builds, 2U);
    EXPECT_NE(original.value, replacement.value);
  }
}

TEST(ActivePlannerCache, GlobalProjectionKeyIncludesInflationAndCapability) {
  const auto original = MakeGlobalProjectionCacheKey(7U, 50, 0.4, 11U);
  const auto inflation_changed =
      MakeGlobalProjectionCacheKey(7U, 50, 0.5, 11U);
  const auto capability_changed =
      MakeGlobalProjectionCacheKey(7U, 50, 0.4, 12U);

  EXPECT_NE(original, inflation_changed);
  EXPECT_NE(original, capability_changed);
}

TEST(ActivePlannerCache,
     LeggedTraversabilityKeysIncludeRevisionProfileAndCapability) {
  PlanningRequest request;
  request.world.odometry_sequence = 3U;
  request.world.tf_sequence = 5U;
  request.goal_map = CacheGoal("legged-grid", 7.0);
  const auto projection =
      MakeLeggedTraversabilityProjectionCacheKey(11U, 13U, 17U);
  const auto revision_changed =
      MakeLeggedTraversabilityProjectionCacheKey(12U, 13U, 17U);
  const auto profile_changed =
      MakeLeggedTraversabilityProjectionCacheKey(11U, 14U, 17U);
  const auto capability_changed =
      MakeLeggedTraversabilityProjectionCacheKey(11U, 13U, 18U);
  const auto route = MakeLeggedTraversabilityRouteCacheKey(
      request, 11U, 13U, 17U);
  const auto route_revision_changed = MakeLeggedTraversabilityRouteCacheKey(
      request, 12U, 13U, 17U);

  EXPECT_NE(projection, revision_changed);
  EXPECT_NE(projection, profile_changed);
  EXPECT_NE(projection, capability_changed);
  EXPECT_NE(route, route_revision_changed);
}

TEST(ActivePlannerCache, GoalFieldKeyPreservesOrderedGoalSemantics) {
  const LocalGoalSet ordered{
      .goals_odom = {CacheGoal("first", 1.0), CacheGoal("second", 2.0)},
      .exact_final_goal = false,
  };
  const LocalGoalSet reversed{
      .goals_odom = {CacheGoal("second", 2.0), CacheGoal("first", 1.0)},
      .exact_final_goal = false,
  };
  const auto original = MakeGoalFieldCacheKey(9U, 0.5, 17U, ordered, {});
  const auto order_changed =
      MakeGoalFieldCacheKey(9U, 0.5, 17U, reversed, {});

  EXPECT_NE(original, order_changed);
}

TEST(ActivePlannerCache, ConcurrentReadersCoalesceOneBuildAndShareOnePointer) {
  ImmutableActiveCacheSlot<TestKey, TestArtifact> cache;
  std::atomic<std::size_t> builds{};
  std::promise<void> builder_entered;
  std::promise<void> release_builder;
  const std::shared_future<void> release = release_builder.get_future().share();
  const auto builder = [&](const SearchControl&) {
    if (builds.fetch_add(1U) == 0U) {
      builder_entered.set_value();
    }
    release.wait();
    return Built(23);
  };

  std::vector<std::future<ImmutableCacheResult<TestArtifact>>> readers;
  readers.push_back(std::async(std::launch::async, [&] {
    return cache.GetOrBuild(Key(), {}, builder);
  }));
  ASSERT_EQ(builder_entered.get_future().wait_for(1s), std::future_status::ready);
  for (std::size_t index = 1U; index < 8U; ++index) {
    readers.push_back(std::async(std::launch::async, [&] {
      return cache.GetOrBuild(Key(), {}, builder);
    }));
  }
  release_builder.set_value();

  std::shared_ptr<const TestArtifact> shared;
  for (auto& reader : readers) {
    ASSERT_EQ(reader.wait_for(1s), std::future_status::ready);
    const auto result = reader.get();
    ASSERT_TRUE(result.ok()) << result.reason_code;
    if (!shared) {
      shared = result.value;
    }
    EXPECT_EQ(result.value, shared);
  }
  EXPECT_EQ(builds.load(), 1U);
}

TEST(ActivePlannerCache, ASlowOldBuildCannotOverwriteANewerCurrentIdentity) {
  ImmutableActiveCacheSlot<TestKey, TestArtifact> cache;
  std::promise<void> old_builder_entered;
  std::promise<void> release_old_builder;
  const std::shared_future<void> release = release_old_builder.get_future().share();
  const TestKey old_key = Key(1U, 2U);
  const TestKey current_key = Key(2U, 2U);
  auto old = std::async(std::launch::async, [&] {
    return cache.GetOrBuild(old_key, {}, [&](const SearchControl&) {
      old_builder_entered.set_value();
      release.wait();
      return Built(31);
    });
  });
  ASSERT_EQ(old_builder_entered.get_future().wait_for(1s),
            std::future_status::ready);

  const auto current = cache.GetOrBuild(
      current_key, {}, [](const SearchControl&) { return Built(47); });
  release_old_builder.set_value();
  ASSERT_EQ(old.wait_for(1s), std::future_status::ready);
  const auto old_result = old.get();
  std::size_t unexpected_rebuilds{};
  const auto still_current = cache.GetOrBuild(
      current_key, {}, [&](const SearchControl&) {
        ++unexpected_rebuilds;
        return Built(99);
      });

  ASSERT_TRUE(old_result.ok()) << old_result.reason_code;
  ASSERT_TRUE(current.ok()) << current.reason_code;
  ASSERT_TRUE(still_current.ok()) << still_current.reason_code;
  EXPECT_EQ(old_result.value->value, 31);
  EXPECT_EQ(current.value->value, 47);
  EXPECT_TRUE(still_current.cache_hit);
  EXPECT_EQ(still_current.value, current.value);
  EXPECT_EQ(unexpected_rebuilds, 0U);
}

TEST(ActivePlannerCache, CancellationDuringAMissDoesNotPublishTheArtifact) {
  ImmutableActiveCacheSlot<TestKey, TestArtifact> cache;
  std::stop_source stop;
  std::size_t builds{};
  SearchControl canceled{.stop_token = stop.get_token()};
  const auto canceled_result = cache.GetOrBuild(
      Key(), canceled, [&](const SearchControl&) {
        ++builds;
        stop.request_stop();
        return Built(53);
      });
  const auto recovered = cache.GetOrBuild(
      Key(), {}, [&](const SearchControl&) {
        ++builds;
        return Built(59);
      });

  EXPECT_FALSE(canceled_result.ok());
  EXPECT_EQ(canceled_result.reason_code, "REQUEST_CANCELED");
  ASSERT_TRUE(recovered.ok()) << recovered.reason_code;
  EXPECT_FALSE(recovered.cache_hit);
  EXPECT_EQ(recovered.value->value, 59);
  EXPECT_EQ(builds, 2U);
}

TEST(ActivePlannerCache, AFailedBuildDoesNotPoisonTheNextAttempt) {
  ImmutableActiveCacheSlot<TestKey, TestArtifact> cache;
  const auto failed = cache.GetOrBuild(
      Key(), {}, [](const SearchControl&) {
        return ImmutableCacheBuildResult<TestArtifact>{
            .reason_code = "BUILD_FAILED"};
      });
  const auto recovered = cache.GetOrBuild(
      Key(), {}, [](const SearchControl&) { return Built(61); });

  EXPECT_FALSE(failed.ok());
  EXPECT_EQ(failed.reason_code, "BUILD_FAILED");
  ASSERT_TRUE(recovered.ok()) << recovered.reason_code;
  EXPECT_FALSE(recovered.cache_hit);
  EXPECT_EQ(recovered.value->value, 61);
}

TEST(ActivePlannerCache, AWaiterRetriesAfterTheOwnerBuildFails) {
  ImmutableActiveCacheSlot<TestKey, TestArtifact> cache;
  std::promise<void> owner_entered;
  std::promise<void> release_owner;
  const std::shared_future<void> release = release_owner.get_future().share();
  auto owner = std::async(std::launch::async, [&] {
    return cache.GetOrBuild(Key(), {}, [&](const SearchControl&) {
      owner_entered.set_value();
      release.wait();
      return ImmutableCacheBuildResult<TestArtifact>{
          .reason_code = "BUILD_FAILED"};
    });
  });
  ASSERT_EQ(owner_entered.get_future().wait_for(1s), std::future_status::ready);
  std::atomic<std::size_t> retry_builds{};
  auto waiter = std::async(std::launch::async, [&] {
    return cache.GetOrBuild(Key(), {}, [&](const SearchControl&) {
      ++retry_builds;
      return Built(67);
    });
  });
  release_owner.set_value();

  ASSERT_EQ(owner.wait_for(1s), std::future_status::ready);
  ASSERT_EQ(waiter.wait_for(1s), std::future_status::ready);
  const auto owner_result = owner.get();
  const auto waiter_result = waiter.get();
  EXPECT_FALSE(owner_result.ok());
  ASSERT_TRUE(waiter_result.ok()) << waiter_result.reason_code;
  EXPECT_EQ(waiter_result.value->value, 67);
  EXPECT_EQ(retry_builds.load(), 1U);
}

TEST(ActivePlannerCache,
     AnOldWaiterCannotRestoreItsKeyAfterANewerIdentityWasPublished) {
  ImmutableActiveCacheSlot<TestKey, TestArtifact> cache;
  const TestKey old_key = Key(1U, 2U);
  const TestKey current_key = Key(2U, 2U);
  std::promise<void> owner_entered;
  std::promise<void> release_owner;
  const std::shared_future<void> release = release_owner.get_future().share();
  auto owner = std::async(std::launch::async, [&] {
    return cache.GetOrBuild(old_key, {}, [&](const SearchControl&) {
      owner_entered.set_value();
      release.wait();
      return ImmutableCacheBuildResult<TestArtifact>{
          .reason_code = "BUILD_FAILED"};
    });
  });
  ASSERT_EQ(owner_entered.get_future().wait_for(1s), std::future_status::ready);

  std::atomic<std::size_t> waiter_clock_reads{};
  std::promise<void> waiter_observed_flight;
  SearchControl waiter_control;
  waiter_control.now = [&] {
    if (waiter_clock_reads.fetch_add(1U) == 1U) {
      waiter_observed_flight.set_value();
    }
    return SteadyClock::now();
  };
  auto waiter = std::async(std::launch::async, [&] {
    return cache.GetOrBuild(
        old_key, waiter_control,
        [](const SearchControl&) { return Built(89); });
  });
  ASSERT_EQ(waiter_observed_flight.get_future().wait_for(1s),
            std::future_status::ready);

  const auto current = cache.GetOrBuild(
      current_key, {}, [](const SearchControl&) { return Built(97); });
  release_owner.set_value();
  ASSERT_EQ(owner.wait_for(1s), std::future_status::ready);
  ASSERT_EQ(waiter.wait_for(1s), std::future_status::ready);
  const auto waiter_result = waiter.get();
  std::size_t unexpected_rebuilds{};
  const auto still_current = cache.GetOrBuild(
      current_key, {}, [&](const SearchControl&) {
        ++unexpected_rebuilds;
        return Built(101);
      });

  ASSERT_TRUE(current.ok()) << current.reason_code;
  ASSERT_TRUE(waiter_result.ok()) << waiter_result.reason_code;
  ASSERT_TRUE(still_current.ok()) << still_current.reason_code;
  EXPECT_EQ(waiter_result.value->value, 89);
  EXPECT_TRUE(still_current.cache_hit);
  EXPECT_EQ(still_current.value, current.value);
  EXPECT_EQ(unexpected_rebuilds, 0U);
}

TEST(ActivePlannerCache, AWaitingCallerCanCancelWithoutBlockingOnTheOwner) {
  ImmutableActiveCacheSlot<TestKey, TestArtifact> cache;
  std::promise<void> owner_entered;
  std::promise<void> release_owner;
  const std::shared_future<void> release = release_owner.get_future().share();
  auto owner = std::async(std::launch::async, [&] {
    return cache.GetOrBuild(Key(), {}, [&](const SearchControl&) {
      owner_entered.set_value();
      release.wait();
      return Built(71);
    });
  });
  ASSERT_EQ(owner_entered.get_future().wait_for(1s), std::future_status::ready);
  std::stop_source stop;
  std::atomic<std::size_t> waiter_clock_reads{};
  std::promise<void> waiter_observed_flight;
  SearchControl waiter_control{.stop_token = stop.get_token()};
  waiter_control.now = [&] {
    if (waiter_clock_reads.fetch_add(1U) == 1U) {
      waiter_observed_flight.set_value();
    }
    return SteadyClock::now();
  };
  auto waiter = std::async(std::launch::async, [&] {
    return cache.GetOrBuild(
        Key(), waiter_control,
        [](const SearchControl&) { return Built(73); });
  });
  ASSERT_EQ(waiter_observed_flight.get_future().wait_for(1s),
            std::future_status::ready);
  stop.request_stop();

  ASSERT_EQ(waiter.wait_for(250ms), std::future_status::ready);
  const auto waiter_result = waiter.get();
  EXPECT_FALSE(waiter_result.ok());
  EXPECT_EQ(waiter_result.reason_code, "REQUEST_CANCELED");
  release_owner.set_value();
  ASSERT_EQ(owner.wait_for(1s), std::future_status::ready);
  ASSERT_TRUE(owner.get().ok());
}

TEST(ActivePlannerCache, ReplacementDoesNotInvalidateAnExistingReader) {
  ImmutableActiveCacheSlot<TestKey, TestArtifact> cache;
  const auto old = cache.GetOrBuild(
      Key(1U, 2U), {}, [](const SearchControl&) { return Built(79); });
  const auto replacement = cache.GetOrBuild(
      Key(2U, 2U), {}, [](const SearchControl&) { return Built(83); });

  ASSERT_TRUE(old.ok()) << old.reason_code;
  ASSERT_TRUE(replacement.ok()) << replacement.reason_code;
  EXPECT_NE(old.value, replacement.value);
  EXPECT_EQ(old.value->value, 79);
  EXPECT_EQ(replacement.value->value, 83);
}

}  // namespace
}  // namespace lunar::pure_planning::shared
