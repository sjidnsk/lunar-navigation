#pragma once

#include <string>
#include <variant>

namespace lunar::planning::v3 {

enum class ErrorCode {
  kInvalidArgument,
  kSchemaMismatch,
  kStaleInput,
  kInconsistentSnapshot,
  kMissingRegistryObject,
  kResourceLimit,
  kNumericalFailure,
  kNoKnownSafeRoute,
};

struct Error final {
  ErrorCode code{ErrorCode::kInvalidArgument};
  std::string field_path;
  std::string message;
};

template <class T>
using Result = std::variant<T, Error>;

template <class T>
[[nodiscard]] bool IsOk(const Result<T>& value) noexcept {
  return std::holds_alternative<T>(value);
}

enum class ControllerStatus {
  kReady,
  kExecuting,
  kHolding,
  kCommitted,
  kInFlight,
  kFault,
};

}  // namespace lunar::planning::v3
