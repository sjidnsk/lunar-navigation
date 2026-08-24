#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <stdexcept>
#include <stop_token>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "lunar_pure_planner_core/search_control.hpp"
#include "lunar_pure_planner_core/types/planning_request.hpp"
#include "shared/anytime_ara_star.hpp"

namespace lunar::pure_planning::shared::anytime {
namespace {

using namespace std::chrono_literals;

std::size_t StablePathIndexForTest(
    const std::vector<std::size_t>& stable_edges) {
  std::uint64_t hash = 1469598103934665603ULL;
  for (const std::size_t edge : stable_edges) {
    std::uint64_t value = static_cast<std::uint64_t>(edge);
    for (std::size_t byte = 0U; byte < sizeof(value); ++byte) {
      hash ^= value & 0xffU;
      hash *= 1099511628211ULL;
      value >>= 8U;
    }
  }
  return static_cast<std::size_t>(hash);
}

SearchCandidate CertifiedCandidate(std::vector<std::size_t> states,
                                   std::vector<std::size_t> stable_edges,
                                   const double cost) {
  return SearchCandidate{
      .states = std::move(states),
      .stable_edge_indices = stable_edges,
      .cost = cost,
      .stable_index = StablePathIndexForTest(stable_edges),
  };
}

AraStarProblem MakeImprovementProblem(std::vector<std::size_t>& expand_calls) {
  AraStarProblem problem;
  problem.state_count = 4U;
  problem.start_state = 0U;
  problem.heuristic = [](const std::size_t state) {
    return state == 2U ? 2.0 : 0.0;
  };
  problem.is_goal = [](const std::size_t state) { return state == 3U; };
  problem.expand = [&expand_calls](const std::size_t state,
                                   const double,
                                   std::vector<GraphEdge>& edges) {
    ++expand_calls.at(state);
    switch (state) {
      case 0U:
        edges = {
            {.target_state = 1U, .cost = 5.0, .stable_index = 10U},
            {.target_state = 2U, .cost = 1.0, .stable_index = 20U},
        };
        break;
      case 1U:
        edges = {{.target_state = 3U, .cost = 10.0, .stable_index = 11U}};
        break;
      case 2U:
        edges = {{.target_state = 1U, .cost = 1.0, .stable_index = 21U}};
        break;
      default:
        break;
    }
  };
  return problem;
}

TEST(AraStarAnytime, ReusesOpenClosedAndInconsAcrossFixedSchedule) {
  std::vector<std::size_t> expand_calls(4U);
  AraStarProblem problem = MakeImprovementProblem(expand_calls);
  problem.config.stop_after_first_solution = false;

  const auto result = SearchAnytimeAraStar(problem);

  ASSERT_EQ(result.status, AraStarStatus::kSolved) << result.reason_code;
  EXPECT_EQ(result.epsilon_history,
            (std::vector<double>{2.5, 2.0, 1.5, 1.0}));
  ASSERT_EQ(result.candidates.size(), 2U);
  EXPECT_DOUBLE_EQ(result.candidates.front().cost, 15.0);
  EXPECT_DOUBLE_EQ(result.candidates.back().cost, 12.0);
  EXPECT_EQ(result.candidates.back().states,
            (std::vector<std::size_t>{0U, 2U, 1U, 3U}));
  EXPECT_EQ(expand_calls, (std::vector<std::size_t>{1U, 1U, 1U, 0U}));
  EXPECT_EQ(result.expanded_states, 4U);
  EXPECT_EQ(result.generated_states, 4U);
  EXPECT_EQ(result.reopened_states, 1U);
  EXPECT_EQ(result.open_peak, 2U);
}

TEST(AraStarAnytime, TimesOutBeforeFindingAnIncumbent) {
  AraStarProblem problem;
  problem.state_count = 2U;
  problem.start_state = 0U;
  problem.heuristic = [](std::size_t) { return 0.0; };
  problem.is_goal = [](std::size_t state) { return state == 1U; };
  problem.expand = [](std::size_t, double, std::vector<GraphEdge>&) {};
  problem.control.deadline = SteadyClock::time_point{0ms};
  problem.control.now = [] { return SteadyClock::time_point{0ms}; };

  const auto result = SearchAnytimeAraStar(problem);

  EXPECT_EQ(result.status, AraStarStatus::kTimedOut);
  EXPECT_TRUE(result.deadline_reached);
  EXPECT_TRUE(result.candidates.empty());
  EXPECT_EQ(result.reason_code, "TIMEOUT");
}

TEST(AraStarAnytime, ReturnsIncumbentWhenDeadlineArrivesDuringImprovement) {
  bool expired = false;
  AraStarProblem problem;
  problem.state_count = 2U;
  problem.start_state = 0U;
  problem.heuristic = [](std::size_t) { return 0.0; };
  problem.is_goal = [](std::size_t state) { return state == 1U; };
  problem.expand = [](const std::size_t state, const double,
                      std::vector<GraphEdge>& edges) {
    if (state == 0U) {
      edges = {{.target_state = 1U, .cost = 1.0, .stable_index = 7U}};
    }
  };
  problem.on_relaxed = [&expired](const std::size_t state, const double) {
    if (state == 1U) {
      expired = true;
    }
  };
  problem.control.deadline = SteadyClock::time_point{};
  problem.control.now = [&expired] {
    return expired ? SteadyClock::time_point{}
                   : SteadyClock::time_point{-1ms};
  };

  const auto result = SearchAnytimeAraStar(problem);

  ASSERT_EQ(result.status, AraStarStatus::kSolved)
      << result.reason_code;
  EXPECT_TRUE(result.deadline_reached);
  ASSERT_EQ(result.candidates.size(), 1U);
  EXPECT_DOUBLE_EQ(result.candidates.front().cost, 1.0);
}

TEST(AraStarAnytime, GivesCancellationPriorityOverDeadline) {
  std::stop_source stop_source;
  stop_source.request_stop();
  AraStarProblem problem;
  problem.state_count = 1U;
  problem.start_state = 0U;
  problem.heuristic = [](std::size_t) { return 0.0; };
  problem.is_goal = [](std::size_t) { return false; };
  problem.expand = [](std::size_t, double, std::vector<GraphEdge>&) {};
  problem.control.stop_token = stop_source.get_token();
  problem.control.deadline = SteadyClock::time_point{0ms};
  problem.control.now = [] { return SteadyClock::time_point{0ms}; };

  const auto result = SearchAnytimeAraStar(problem);

  EXPECT_EQ(result.status, AraStarStatus::kCanceled);
  EXPECT_FALSE(result.deadline_reached);
  EXPECT_EQ(result.reason_code, "REQUEST_CANCELED");
}

TEST(AraStarAnytime, GivesCancellationPriorityWhenNowSetsStopAtInitialDeadline) {
  std::stop_source stop_source;
  std::size_t now_calls = 0U;
  AraStarProblem problem;
  problem.state_count = 1U;
  problem.start_state = 0U;
  problem.heuristic = [](std::size_t) { return 0.0; };
  problem.is_goal = [](std::size_t) { return false; };
  problem.expand = [](std::size_t, double, std::vector<GraphEdge>&) {};
  problem.control.stop_token = stop_source.get_token();
  problem.control.deadline = SteadyClock::time_point{0ms};
  problem.control.now = [&] {
    ++now_calls;
    stop_source.request_stop();
    return SteadyClock::time_point{0ms};
  };

  const auto result = SearchAnytimeAraStar(problem);

  EXPECT_EQ(result.status, AraStarStatus::kCanceled);
  EXPECT_EQ(result.reason_code, "REQUEST_CANCELED");
  EXPECT_EQ(now_calls, 1U);
}

TEST(AraStarAnytime, CancelsWhileRekeyingLiveAndInconsistentStates) {
  std::stop_source stop_source;
  std::vector<std::size_t> expand_calls(4U);
  std::size_t state_one_key_reads = 0U;
  AraStarProblem problem = MakeImprovementProblem(expand_calls);
  problem.guidance = [&](const std::size_t state) {
    if (state == 1U && ++state_one_key_reads == 2U) {
      stop_source.request_stop();
    }
    return 0.0;
  };
  problem.control.stop_token = stop_source.get_token();
  problem.config.stop_after_first_solution = false;

  const auto result = SearchAnytimeAraStar(problem);

  EXPECT_EQ(result.status, AraStarStatus::kCanceled);
  EXPECT_EQ(result.reason_code, "REQUEST_CANCELED");
  EXPECT_EQ(result.expanded_states, 3U);
  EXPECT_EQ(result.epsilon_history, (std::vector<double>{2.5}));
  EXPECT_EQ(state_one_key_reads, 2U);
}

TEST(AraStarAnytime, RejectsInvalidEdgesAndHeuristicsWhenUsed) {
  AraStarProblem invalid_edge;
  invalid_edge.state_count = 2U;
  invalid_edge.start_state = 0U;
  invalid_edge.heuristic = [](std::size_t) { return 0.0; };
  invalid_edge.is_goal = [](std::size_t state) { return state == 1U; };
  invalid_edge.expand = [](std::size_t, double,
                           std::vector<GraphEdge>& edges) {
    edges = {{.target_state = 1U, .cost = 0.0, .stable_index = 1U}};
  };

  const auto edge_result = SearchAnytimeAraStar(invalid_edge);
  EXPECT_EQ(edge_result.status, AraStarStatus::kInvalidProblem);
  EXPECT_EQ(edge_result.reason_code, "SEARCH_EDGE_INVALID");

  AraStarProblem invalid_heuristic = invalid_edge;
  invalid_heuristic.expand = [](std::size_t, double,
                                std::vector<GraphEdge>& edges) {
    edges = {{.target_state = 1U, .cost = 1.0, .stable_index = 1U}};
  };
  invalid_heuristic.heuristic = [](std::size_t state) {
    return state == 1U ? -1.0 : 0.0;
  };
  invalid_heuristic.config.stop_after_first_solution = false;

  const auto heuristic_result = SearchAnytimeAraStar(invalid_heuristic);
  EXPECT_EQ(heuristic_result.status, AraStarStatus::kInvalidProblem);
  EXPECT_EQ(heuristic_result.reason_code, "SEARCH_HEURISTIC_INVALID");
}

TEST(AraStarAnytime, RejectsNonFiniteEdgesBeforeOrderingThem) {
  for (const double invalid_cost :
       {std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity()}) {
    AraStarProblem problem;
    problem.state_count = 2U;
    problem.start_state = 0U;
    problem.heuristic = [](std::size_t) { return 0.0; };
    problem.is_goal = [](std::size_t state) { return state == 1U; };
    problem.expand = [invalid_cost](std::size_t, double,
                                    std::vector<GraphEdge>& edges) {
      edges = {
          {.target_state = 1U, .cost = invalid_cost, .stable_index = 2U},
          {.target_state = 1U, .cost = 1.0, .stable_index = 1U},
      };
    };

    const auto result = SearchAnytimeAraStar(problem);

    EXPECT_EQ(result.status, AraStarStatus::kInvalidProblem);
    EXPECT_EQ(result.reason_code, "SEARCH_EDGE_INVALID");
  }
}

TEST(AraStarAnytime, ChecksCancellationAndDeadlineAfterTheLastExpansion) {
  std::stop_source stop_source;
  AraStarProblem canceled;
  canceled.state_count = 1U;
  canceled.start_state = 0U;
  canceled.heuristic = [](std::size_t) { return 0.0; };
  canceled.is_goal = [](std::size_t) { return false; };
  canceled.control.stop_token = stop_source.get_token();
  canceled.expand = [&stop_source](std::size_t, double,
                                   std::vector<GraphEdge>&) {
    stop_source.request_stop();
  };

  const auto canceled_result = SearchAnytimeAraStar(canceled);

  EXPECT_EQ(canceled_result.status, AraStarStatus::kCanceled);
  EXPECT_EQ(canceled_result.reason_code, "REQUEST_CANCELED");
  EXPECT_EQ(canceled_result.expanded_states, 1U);

  bool expired = false;
  AraStarProblem timed_out;
  timed_out.state_count = 1U;
  timed_out.start_state = 0U;
  timed_out.heuristic = [](std::size_t) { return 0.0; };
  timed_out.is_goal = [](std::size_t) { return false; };
  timed_out.control.deadline = SteadyClock::time_point{0ms};
  timed_out.control.now = [&expired] {
    return SteadyClock::time_point{expired ? 0ms : -1ms};
  };
  timed_out.expand = [&expired](std::size_t, double,
                                std::vector<GraphEdge>&) {
    expired = true;
  };

  const auto timeout_result = SearchAnytimeAraStar(timed_out);

  EXPECT_EQ(timeout_result.status, AraStarStatus::kTimedOut);
  EXPECT_TRUE(timeout_result.deadline_reached);
  EXPECT_EQ(timeout_result.expanded_states, 1U);
}

TEST(AraStarAnytime, PreservesIncumbentAndMetricsWhenSearchAllocationsFail) {
  for (const bool throw_bad_alloc : {false, true}) {
    AraStarProblem problem;
    problem.state_count = 3U;
    problem.start_state = 0U;
    problem.heuristic = [](std::size_t) { return 0.0; };
    problem.is_goal = [](std::size_t state) { return state == 1U; };
    problem.expand = [throw_bad_alloc](const std::size_t state, const double,
                                       std::vector<GraphEdge>& edges) {
      if (state == 0U) {
        edges = {
            {.target_state = 1U, .cost = 1.0, .stable_index = 10U},
            {.target_state = 2U, .cost = 0.5, .stable_index = 20U},
        };
        return;
      }
      if (throw_bad_alloc) {
        throw std::bad_alloc();
      }
      throw std::length_error("simulated edge cache exhaustion");
    };
    problem.config.stop_after_first_solution = false;

    const auto result = SearchAnytimeAraStar(problem);

    EXPECT_EQ(result.status, AraStarStatus::kResourceExhausted);
    EXPECT_TRUE(result.reason_code.empty());
    EXPECT_EQ(result.expanded_states, 2U);
    EXPECT_EQ(result.epsilon_history, (std::vector<double>{2.5}));
    ASSERT_EQ(result.candidates.size(), 1U);
    EXPECT_DOUBLE_EQ(result.candidates.front().cost, 1.0);
    EXPECT_EQ(result.candidates.front().states,
              (std::vector<std::size_t>{0U, 1U}));
  }
}

TEST(AraStarAnytime, ReconstructsAPathWithTheLargestStableEdgeIndex) {
  AraStarProblem problem;
  problem.state_count = 2U;
  problem.start_state = 0U;
  problem.heuristic = [](std::size_t) { return 0.0; };
  problem.is_goal = [](std::size_t state) { return state == 1U; };
  problem.expand = [](std::size_t, double, std::vector<GraphEdge>& edges) {
    edges = {{.target_state = 1U,
              .cost = 1.0,
              .stable_index = std::numeric_limits<std::size_t>::max()}};
  };

  const auto result = SearchAnytimeAraStar(problem);

  ASSERT_EQ(result.status, AraStarStatus::kSolved) << result.reason_code;
  ASSERT_EQ(result.candidates.size(), 1U);
  EXPECT_EQ(result.candidates.front().stable_edge_indices,
            (std::vector<std::size_t>{std::numeric_limits<std::size_t>::max()}));
}

TEST(AraStarAnytime, UsesGuidanceOnlyToBreakEqualPriorityTies) {
  const auto make_problem = [] {
    AraStarProblem problem;
    problem.state_count = 4U;
    problem.start_state = 0U;
    problem.heuristic = [](std::size_t) { return 0.0; };
    problem.is_goal = [](const std::size_t state) { return state == 3U; };
    problem.expand = [](const std::size_t state, const double,
                        std::vector<GraphEdge>& edges) {
      if (state == 0U) {
        edges = {
            {.target_state = 1U, .cost = 1.0, .stable_index = 1U},
            {.target_state = 2U, .cost = 1.0, .stable_index = 2U},
        };
      } else if (state == 1U || state == 2U) {
        edges = {{.target_state = 3U, .cost = 1.0,
                  .stable_index = 10U + state}};
      }
    };
    return problem;
  };

  AraStarProblem baseline = make_problem();
  const auto baseline_result = SearchAnytimeAraStar(baseline);
  ASSERT_EQ(baseline_result.status, AraStarStatus::kSolved)
      << baseline_result.reason_code;
  ASSERT_FALSE(baseline_result.candidates.empty());
  EXPECT_EQ(baseline_result.candidates.front().states,
            (std::vector<std::size_t>{0U, 1U, 3U}));

  AraStarProblem guided = make_problem();
  guided.guidance = [](const std::size_t state) {
    return state == 2U ? 0.0 : 1.0;
  };
  const auto guided_result = SearchAnytimeAraStar(guided);
  ASSERT_EQ(guided_result.status, AraStarStatus::kSolved)
      << guided_result.reason_code;
  ASSERT_FALSE(guided_result.candidates.empty());
  EXPECT_EQ(guided_result.candidates.front().states,
            (std::vector<std::size_t>{0U, 2U, 3U}));
}

TEST(AraStarAnytime, AnchorKeyBeatsGuidanceForUnequalPriority) {
  AraStarProblem problem;
  problem.state_count = 4U;
  problem.start_state = 0U;
  problem.heuristic = [](std::size_t) { return 0.0; };
  problem.guidance = [](const std::size_t state) {
    return state == 2U || state == 3U ? 0.0 : 10.0;
  };
  problem.is_goal = [](const std::size_t state) { return state == 3U; };
  problem.expand = [](const std::size_t state, const double,
                      std::vector<GraphEdge>& edges) {
    if (state == 0U) {
      edges = {
          {.target_state = 1U, .cost = 1.0, .stable_index = 1U},
          {.target_state = 2U, .cost = 2.0, .stable_index = 2U},
      };
    } else if (state == 1U || state == 2U) {
      edges = {{.target_state = 3U, .cost = 1.0,
                .stable_index = 10U + state}};
    }
  };

  const auto result = SearchAnytimeAraStar(problem);

  ASSERT_EQ(result.status, AraStarStatus::kSolved) << result.reason_code;
  ASSERT_FALSE(result.candidates.empty());
  EXPECT_EQ(result.candidates.front().states,
            (std::vector<std::size_t>{0U, 1U, 3U}));
}

TEST(AraStarAnytime, DiscardsLazyStaleOpenEntries) {
  AraStarProblem problem;
  problem.state_count = 4U;
  problem.start_state = 0U;
  problem.heuristic = [](std::size_t) { return 0.0; };
  problem.is_goal = [](const std::size_t state) { return state == 3U; };
  problem.expand = [](const std::size_t state, const double,
                      std::vector<GraphEdge>& edges) {
    if (state == 0U) {
      edges = {
          {.target_state = 1U, .cost = 5.0, .stable_index = 1U},
          {.target_state = 2U, .cost = 1.0, .stable_index = 2U},
      };
    } else if (state == 2U) {
      edges = {{.target_state = 1U, .cost = 1.0, .stable_index = 3U}};
    } else if (state == 1U) {
      edges = {{.target_state = 3U, .cost = 10.0, .stable_index = 4U}};
    }
  };

  const auto result = SearchAnytimeAraStar(problem);

  ASSERT_EQ(result.status, AraStarStatus::kSolved) << result.reason_code;
  ASSERT_FALSE(result.candidates.empty());
  EXPECT_DOUBLE_EQ(result.candidates.back().cost, 12.0);
  EXPECT_EQ(result.expanded_states, 3U);
  EXPECT_EQ(result.generated_states, 4U);
  EXPECT_EQ(result.reopened_states, 0U);
  EXPECT_EQ(result.open_peak, 2U);
}

TEST(AraStarAnytime, SkipsInactiveLabelsWithoutLosingTheActivePath) {
  std::vector<std::size_t> expand_calls(4U);
  AraStarProblem problem;
  problem.state_count = 4U;
  problem.start_state = 0U;
  problem.heuristic = [](std::size_t) { return 0.0; };
  problem.is_goal = [](const std::size_t state) { return state == 3U; };
  problem.state_expandable = [](const std::size_t state) {
    return state != 1U;
  };
  problem.expand = [&expand_calls](const std::size_t state, const double,
                                   std::vector<GraphEdge>& edges) {
    ++expand_calls.at(state);
    if (state == 0U) {
      edges = {
          {.target_state = 1U, .cost = 1.0, .stable_index = 1U},
          {.target_state = 2U, .cost = 2.0, .stable_index = 2U},
      };
    } else if (state == 1U || state == 2U) {
      edges = {{.target_state = 3U, .cost = 1.0,
                .stable_index = 10U + state}};
    }
  };

  const auto result = SearchAnytimeAraStar(problem);

  ASSERT_EQ(result.status, AraStarStatus::kSolved) << result.reason_code;
  ASSERT_FALSE(result.candidates.empty());
  EXPECT_EQ(result.candidates.back().states,
            (std::vector<std::size_t>{0U, 2U, 3U}));
  EXPECT_EQ(expand_calls, (std::vector<std::size_t>{1U, 0U, 1U, 0U}));
  EXPECT_EQ(result.expanded_states, 2U);
}

TEST(AraStarAnytime, PassesSourceCostAndNotifiesEveryRelaxationInOrder) {
  std::vector<std::string> events;
  std::vector<std::pair<std::size_t, double>> expansion_costs;
  std::vector<std::pair<std::size_t, double>> relaxed_costs;
  AraStarProblem problem;
  problem.state_count = 3U;
  problem.start_state = 0U;
  problem.heuristic = [](std::size_t) { return 0.0; };
  problem.is_goal = [](const std::size_t state) { return state == 2U; };
  problem.expand = [&](const std::size_t state, const double source_g,
                       std::vector<GraphEdge>& edges) {
    events.push_back("expand" + std::to_string(state));
    expansion_costs.emplace_back(state, source_g);
    if (state == 0U) {
      edges = {
          {.target_state = 2U, .cost = 5.0, .stable_index = 20U},
          {.target_state = 1U, .cost = 2.0, .stable_index = 10U},
      };
    } else if (state == 1U) {
      edges = {{.target_state = 2U, .cost = 1.0, .stable_index = 11U}};
    }
  };
  problem.on_relaxed = [&](const std::size_t state, const double new_g) {
    events.push_back("relax" + std::to_string(state));
    relaxed_costs.emplace_back(state, new_g);
  };
  problem.config.stop_after_first_solution = false;

  const auto result = SearchAnytimeAraStar(problem);

  ASSERT_EQ(result.status, AraStarStatus::kSolved) << result.reason_code;
  EXPECT_EQ(events, (std::vector<std::string>{
                        "expand0", "relax1", "relax2", "expand1",
                        "relax2"}));
  EXPECT_EQ(expansion_costs,
            (std::vector<std::pair<std::size_t, double>>{{0U, 0.0},
                                                         {1U, 2.0}}));
  EXPECT_EQ(relaxed_costs,
            (std::vector<std::pair<std::size_t, double>>{{1U, 2.0},
                                                         {2U, 5.0},
                                                         {2U, 3.0}}));
  ASSERT_FALSE(result.candidates.empty());
  EXPECT_DOUBLE_EQ(result.candidates.back().cost, 3.0);
}

TEST(AraStarAnytime, ReturnsImmediatelyAfterTheFirstSolutionByDefault) {
  std::vector<std::size_t> expand_calls(4U);
  AraStarProblem problem;
  problem.state_count = 4U;
  problem.start_state = 0U;
  problem.heuristic = [](std::size_t) { return 0.0; };
  problem.is_goal = [](const std::size_t state) { return state == 3U; };
  problem.expand = [&expand_calls](const std::size_t state, const double,
                                   std::vector<GraphEdge>& edges) {
    ++expand_calls.at(state);
    if (state == 0U) {
      edges = {
          {.target_state = 1U, .cost = 1.0, .stable_index = 1U},
          {.target_state = 2U, .cost = 2.0, .stable_index = 2U},
      };
    } else if (state == 1U) {
      edges = {{.target_state = 3U, .cost = 10.0, .stable_index = 11U}};
    } else if (state == 2U) {
      edges = {{.target_state = 3U, .cost = 1.0, .stable_index = 12U}};
    }
  };
  const auto result = SearchAnytimeAraStar(problem);

  ASSERT_EQ(result.status, AraStarStatus::kSolved) << result.reason_code;
  ASSERT_EQ(result.candidates.size(), 1U);
  EXPECT_DOUBLE_EQ(result.candidates.front().cost, 11.0);
  EXPECT_EQ(result.candidates.front().states,
            (std::vector<std::size_t>{0U, 1U, 3U}));
  EXPECT_EQ(expand_calls, (std::vector<std::size_t>{1U, 1U, 0U, 0U}));
  EXPECT_EQ(result.epsilon_history, (std::vector<double>{2.5}));
}

TEST(AraStarAnytime, DoesNotAllocateTheNominalStateCount) {
  AraStarProblem problem;
  problem.state_count = std::numeric_limits<std::size_t>::max();
  problem.start_state = 0U;
  problem.heuristic = [](std::size_t) { return 0.0; };
  problem.is_goal = [](const std::size_t state) { return state == 1U; };
  problem.expand = [](const std::size_t state, const double,
                      std::vector<GraphEdge>& edges) {
    if (state == 0U) {
      edges = {{.target_state = 1U, .cost = 1.0, .stable_index = 7U}};
    }
  };

  const auto result = SearchAnytimeAraStar(problem);

  ASSERT_EQ(result.status, AraStarStatus::kSolved) << result.reason_code;
  ASSERT_EQ(result.candidates.size(), 1U);
  EXPECT_EQ(result.candidates.front().states,
            (std::vector<std::size_t>{0U, 1U}));
}

TEST(AraStarAnytime, ReturnsCertifiedOpaqueInitialCandidateWithoutExpansion) {
  std::size_t expand_calls = 0U;
  AraStarProblem problem;
  problem.state_count = 1U;
  problem.start_state = 0U;
  problem.heuristic = [](std::size_t) { return 0.0; };
  problem.is_goal = [](const std::size_t state) {
    return state == std::numeric_limits<std::size_t>::max();
  };
  problem.expand = [&expand_calls](std::size_t, double,
                                   std::vector<GraphEdge>&) {
    ++expand_calls;
  };
  problem.config.stop_after_first_solution = true;
  problem.certified_initial_candidate = CertifiedCandidate(
      {0U, std::numeric_limits<std::size_t>::max()}, {91U}, 7.0);

  const auto result = SearchAnytimeAraStar(problem);

  ASSERT_EQ(result.status, AraStarStatus::kSolved) << result.reason_code;
  ASSERT_EQ(result.candidates.size(), 1U);
  EXPECT_EQ(result.candidates.front(), *problem.certified_initial_candidate);
  EXPECT_EQ(result.expanded_states, 0U);
  EXPECT_EQ(result.generated_states, 0U);
  EXPECT_EQ(expand_calls, 0U);
  EXPECT_EQ(result.epsilon_history, (std::vector<double>{2.5}));
}

TEST(AraStarAnytime, UsesCertifiedInitialCandidateAsAnImprovementBound) {
  std::size_t expand_calls = 0U;
  AraStarProblem problem;
  problem.state_count = 2U;
  problem.start_state = 0U;
  problem.heuristic = [](std::size_t) { return 0.0; };
  problem.is_goal = [](const std::size_t state) {
    return state == 1U ||
           state == std::numeric_limits<std::size_t>::max();
  };
  problem.expand = [&expand_calls](const std::size_t state, const double,
                                   std::vector<GraphEdge>& edges) {
    ++expand_calls;
    if (state == 0U) {
      edges = {{.target_state = 1U, .cost = 3.0, .stable_index = 10U}};
    }
  };
  problem.certified_initial_candidate =
      CertifiedCandidate({0U, std::numeric_limits<std::size_t>::max()},
                         {3U}, 5.0);
  problem.config.stop_after_first_solution = false;

  const auto result = SearchAnytimeAraStar(problem);

  ASSERT_EQ(result.status, AraStarStatus::kSolved) << result.reason_code;
  ASSERT_EQ(result.candidates.size(), 2U);
  EXPECT_EQ(result.candidates.front(), *problem.certified_initial_candidate);
  EXPECT_DOUBLE_EQ(result.candidates.back().cost, 3.0);
  EXPECT_EQ(result.candidates.back().states,
            (std::vector<std::size_t>{0U, 1U}));
  EXPECT_EQ(result.expanded_states, 1U);
  EXPECT_EQ(expand_calls, 1U);
}

TEST(AraStarAnytime, BreaksEqualCostIncumbentsByStablePathIndex) {
  AraStarProblem problem;
  problem.state_count = 2U;
  problem.start_state = 0U;
  problem.heuristic = [](std::size_t) { return 0.0; };
  problem.is_goal = [](const std::size_t state) {
    return state == 1U || state == 99U;
  };
  problem.expand = [](const std::size_t state, const double,
                      std::vector<GraphEdge>& edges) {
    if (state == 0U) {
      edges = {{.target_state = 1U, .cost = 5.0, .stable_index = 10U}};
    }
  };
  problem.certified_initial_candidate =
      CertifiedCandidate({0U, 99U}, {3U}, 5.0);
  problem.config.stop_after_first_solution = false;
  ASSERT_LT(StablePathIndexForTest({10U}),
            problem.certified_initial_candidate->stable_index);

  const auto result = SearchAnytimeAraStar(problem);

  ASSERT_EQ(result.status, AraStarStatus::kSolved) << result.reason_code;
  ASSERT_EQ(result.candidates.size(), 2U);
  EXPECT_EQ(result.candidates.back().stable_index,
            StablePathIndexForTest({10U}));
  EXPECT_EQ(result.candidates.back().states,
            (std::vector<std::size_t>{0U, 1U}));
}

TEST(AraStarAnytime, KeepsLowerStableInitialCandidateAtEqualCost) {
  AraStarProblem problem;
  problem.state_count = 2U;
  problem.start_state = 0U;
  problem.heuristic = [](std::size_t) { return 0.0; };
  problem.is_goal = [](const std::size_t state) {
    return state == 1U || state == 99U;
  };
  problem.expand = [](const std::size_t state, const double,
                      std::vector<GraphEdge>& edges) {
    if (state == 0U) {
      edges = {{.target_state = 1U, .cost = 5.0, .stable_index = 3U}};
    }
  };
  problem.certified_initial_candidate =
      CertifiedCandidate({0U, 99U}, {10U}, 5.0);
  ASSERT_LT(problem.certified_initial_candidate->stable_index,
            StablePathIndexForTest({3U}));

  const auto result = SearchAnytimeAraStar(problem);

  ASSERT_EQ(result.status, AraStarStatus::kSolved) << result.reason_code;
  ASSERT_EQ(result.candidates.size(), 1U);
  EXPECT_EQ(result.candidates.front(), *problem.certified_initial_candidate);
}

TEST(AraStarAnytime, RejectsMalformedCertifiedInitialCandidates) {
  const auto make_problem = [] {
    AraStarProblem problem;
    problem.state_count = 2U;
    problem.start_state = 0U;
    problem.heuristic = [](std::size_t) { return 0.0; };
    problem.is_goal = [](const std::size_t state) { return state == 1U; };
    problem.expand = [](std::size_t, double, std::vector<GraphEdge>&) {};
    return problem;
  };
  const auto expect_invalid = [&](SearchCandidate candidate) {
    AraStarProblem problem = make_problem();
    problem.certified_initial_candidate = std::move(candidate);
    const auto result = SearchAnytimeAraStar(problem);
    EXPECT_EQ(result.status, AraStarStatus::kInvalidProblem);
    EXPECT_EQ(result.reason_code, "SEARCH_INITIAL_CANDIDATE_INVALID");
    EXPECT_TRUE(result.candidates.empty());
    EXPECT_EQ(result.expanded_states, 0U);
  };

  expect_invalid(CertifiedCandidate({}, {}, 0.0));
  expect_invalid(CertifiedCandidate({1U}, {}, 0.0));
  expect_invalid(CertifiedCandidate({0U, 1U}, {}, 1.0));
  expect_invalid(CertifiedCandidate({0U, 1U}, {7U}, -1.0));
  expect_invalid(CertifiedCandidate(
      {0U, 1U}, {7U}, std::numeric_limits<double>::infinity()));
  expect_invalid(CertifiedCandidate(
      {0U, 1U}, {7U}, std::numeric_limits<double>::quiet_NaN()));
  auto wrong_stable_index = CertifiedCandidate({0U, 1U}, {7U}, 1.0);
  ++wrong_stable_index.stable_index;
  expect_invalid(std::move(wrong_stable_index));
  expect_invalid(CertifiedCandidate({0U, 2U}, {7U}, 1.0));
}

TEST(AraStarAnytime, GivesCancellationPriorityOverInitialCandidate) {
  std::stop_source stop_source;
  stop_source.request_stop();
  AraStarProblem problem;
  problem.state_count = 1U;
  problem.start_state = 0U;
  problem.heuristic = [](std::size_t) { return 0.0; };
  problem.is_goal = [](std::size_t) { return true; };
  problem.expand = [](std::size_t, double, std::vector<GraphEdge>&) {};
  problem.control.stop_token = stop_source.get_token();
  problem.config.stop_after_first_solution = true;
  problem.certified_initial_candidate = CertifiedCandidate({}, {}, -1.0);

  const auto result = SearchAnytimeAraStar(problem);

  EXPECT_EQ(result.status, AraStarStatus::kCanceled);
  EXPECT_EQ(result.reason_code, "REQUEST_CANCELED");
}

TEST(AraStarAnytime, GivesCancellationPriorityAfterInitialGoalValidation) {
  std::stop_source stop_source;
  AraStarProblem problem;
  problem.state_count = 1U;
  problem.start_state = 0U;
  problem.heuristic = [](std::size_t) { return 0.0; };
  problem.is_goal = [&stop_source](std::size_t) {
    stop_source.request_stop();
    return false;
  };
  problem.expand = [](std::size_t, double, std::vector<GraphEdge>&) {};
  problem.control.stop_token = stop_source.get_token();
  problem.certified_initial_candidate =
      CertifiedCandidate({0U}, {}, 0.0);

  const auto result = SearchAnytimeAraStar(problem);

  EXPECT_EQ(result.status, AraStarStatus::kCanceled);
  EXPECT_EQ(result.reason_code, "REQUEST_CANCELED");
}

}  // namespace
}  // namespace lunar::pure_planning::shared::anytime
