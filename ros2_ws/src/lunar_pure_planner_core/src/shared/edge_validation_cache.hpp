#pragma once

#include <cstddef>
#include <functional>
#include <unordered_map>
#include <utility>

namespace lunar::pure_planning::shared {

template <class Key, class Value, class Hash>
class EdgeValidationCache final {
 public:
  template <class EvaluateFn>
  const Value& GetOrEvaluate(const Key& key, EvaluateFn&& evaluate) {
    if (const auto found = values_.find(key); found != values_.end()) {
      return found->second;
    }
    Value value = std::invoke(std::forward<EvaluateFn>(evaluate));
    return values_.emplace(key, std::move(value)).first->second;
  }

  [[nodiscard]] std::size_t evaluation_count() const noexcept {
    return values_.size();
  }

 private:
  std::unordered_map<Key, Value, Hash> values_;
};

}  // namespace lunar::pure_planning::shared
