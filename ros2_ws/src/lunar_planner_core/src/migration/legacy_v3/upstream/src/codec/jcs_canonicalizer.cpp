#include "lunar_path_planner/v3/codec/jcs_canonicalizer.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <double-conversion/double-to-string.h>
#include <nlohmann/json.hpp>

namespace lunar::planning::v3 {
namespace {

constexpr std::uint32_t kUnicodeSurrogateFirst = 0xd800U;
constexpr std::uint32_t kUnicodeSurrogateLast = 0xdfffU;
constexpr std::uint32_t kUnicodeMaximum = 0x10ffffU;
constexpr std::uint32_t kUtf16SupplementaryOffset = 0x10000U;
constexpr std::uint32_t kUtf16HighSurrogate = 0xd800U;
constexpr std::uint32_t kUtf16LowSurrogate = 0xdc00U;
constexpr int kBinary64SignificandBits = 53;

struct DecodedCodePoint final {
  std::uint32_t value{};
  std::size_t encoded_size{};
};

struct ObjectEntry final {
  std::string_view name;
  const nlohmann::json* value{};
  std::vector<std::uint16_t> utf16_name;
};

[[nodiscard]] Error InvalidInput(
    std::string field_path, std::string message) {
  return Error{
      .code = ErrorCode::kInvalidArgument,
      .field_path = std::move(field_path),
      .message = std::move(message),
  };
}

[[nodiscard]] std::optional<DecodedCodePoint> DecodeUtf8CodePoint(
    const std::string_view input, const std::size_t offset) {
  if (offset >= input.size()) {
    return std::nullopt;
  }

  const auto first =
      static_cast<std::uint8_t>(input[offset]);
  if (first <= 0x7fU) {
    return DecodedCodePoint{
        .value = first,
        .encoded_size = 1U,
    };
  }

  std::size_t encoded_size{};
  std::uint32_t minimum_value{};
  std::uint32_t code_point{};
  if (first >= 0xc2U && first <= 0xdfU) {
    encoded_size = 2U;
    minimum_value = 0x80U;
    code_point = static_cast<std::uint32_t>(first & 0x1fU);
  } else if (first >= 0xe0U && first <= 0xefU) {
    encoded_size = 3U;
    minimum_value = 0x800U;
    code_point = static_cast<std::uint32_t>(first & 0x0fU);
  } else if (first >= 0xf0U && first <= 0xf4U) {
    encoded_size = 4U;
    minimum_value = 0x10000U;
    code_point = static_cast<std::uint32_t>(first & 0x07U);
  } else {
    return std::nullopt;
  }

  if (encoded_size > input.size() - offset) {
    return std::nullopt;
  }
  for (std::size_t index = 1U; index < encoded_size; ++index) {
    const auto next =
        static_cast<std::uint8_t>(input[offset + index]);
    if ((next & 0xc0U) != 0x80U) {
      return std::nullopt;
    }
    code_point =
        (code_point << 6U) |
        static_cast<std::uint32_t>(next & 0x3fU);
  }

  if (code_point < minimum_value ||
      code_point > kUnicodeMaximum ||
      (code_point >= kUnicodeSurrogateFirst &&
       code_point <= kUnicodeSurrogateLast)) {
    return std::nullopt;
  }
  return DecodedCodePoint{
      .value = code_point,
      .encoded_size = encoded_size,
  };
}

[[nodiscard]] bool AppendUtf16Units(
    const std::string_view input,
    std::vector<std::uint16_t>* const output) {
  std::size_t offset{};
  while (offset < input.size()) {
    const auto code_point = DecodeUtf8CodePoint(input, offset);
    if (!code_point.has_value()) {
      return false;
    }

    if (code_point->value < kUtf16SupplementaryOffset) {
      output->push_back(
          static_cast<std::uint16_t>(code_point->value));
    } else {
      const std::uint32_t supplementary =
          code_point->value - kUtf16SupplementaryOffset;
      output->push_back(static_cast<std::uint16_t>(
          kUtf16HighSurrogate + (supplementary >> 10U)));
      output->push_back(static_cast<std::uint16_t>(
          kUtf16LowSurrogate + (supplementary & 0x3ffU)));
    }
    offset += code_point->encoded_size;
  }
  return true;
}

[[nodiscard]] bool IsExactlyRepresentableAsBinary64(
    const std::uint64_t magnitude) noexcept {
  if (magnitude == 0U) {
    return true;
  }
  const int significant_width = std::bit_width(magnitude);
  if (significant_width <= kBinary64SignificandBits) {
    return true;
  }
  return std::countr_zero(magnitude) >=
         significant_width - kBinary64SignificandBits;
}

[[nodiscard]] std::uint64_t UnsignedMagnitude(
    const std::int64_t value) noexcept {
  if (value >= 0) {
    return static_cast<std::uint64_t>(value);
  }
  return static_cast<std::uint64_t>(-(value + 1)) + 1U;
}

[[nodiscard]] std::optional<Error> AppendQuotedString(
    const std::string_view input,
    const std::string_view field_path,
    std::string* const output) {
  constexpr char kHexDigits[] = "0123456789abcdef";
  output->push_back('"');

  std::size_t offset{};
  while (offset < input.size()) {
    const auto code_point = DecodeUtf8CodePoint(input, offset);
    if (!code_point.has_value()) {
      return InvalidInput(
          std::string{field_path},
          "string is not valid Unicode encoded as UTF-8");
    }

    const std::uint32_t value = code_point->value;
    if (value <= 0x1fU) {
      switch (value) {
        case 0x08U:
          output->append("\\b");
          break;
        case 0x09U:
          output->append("\\t");
          break;
        case 0x0aU:
          output->append("\\n");
          break;
        case 0x0cU:
          output->append("\\f");
          break;
        case 0x0dU:
          output->append("\\r");
          break;
        default:
          output->append("\\u00");
          output->push_back(
              kHexDigits[(value >> 4U) & 0x0fU]);
          output->push_back(kHexDigits[value & 0x0fU]);
          break;
      }
    } else if (value == 0x22U) {
      output->append("\\\"");
    } else if (value == 0x5cU) {
      output->append("\\\\");
    } else {
      output->append(input.substr(offset, code_point->encoded_size));
    }
    offset += code_point->encoded_size;
  }

  output->push_back('"');
  return std::nullopt;
}

[[nodiscard]] std::optional<Error> AppendNumber(
    const double value, const std::string_view field_path,
    std::string* const output) {
  if (!std::isfinite(value)) {
    return InvalidInput(
        std::string{field_path},
        "JCS does not permit NaN or infinity");
  }

  constexpr int kFlags =
      double_conversion::DoubleToStringConverter::UNIQUE_ZERO |
      double_conversion::DoubleToStringConverter::
          EMIT_POSITIVE_EXPONENT_SIGN;
  const double_conversion::DoubleToStringConverter converter{
      kFlags,
      nullptr,
      nullptr,
      'e',
      -6,
      21,
      6,
      0,
  };
  char buffer[64]{};
  double_conversion::StringBuilder builder{
      buffer, static_cast<int>(sizeof(buffer))};
  if (!converter.ToShortest(value, &builder)) {
    return InvalidInput(
        std::string{field_path},
        "number cannot be represented by the JCS serializer");
  }
  const int length = builder.position();
  builder.Finalize();
  output->append(buffer, static_cast<std::size_t>(length));
  return std::nullopt;
}

[[nodiscard]] std::string ArrayElementPath(
    const std::string_view parent, const std::size_t index) {
  return std::string{parent} + "[" + std::to_string(index) + "]";
}

[[nodiscard]] std::string ObjectValuePath(
    const std::string_view parent, const std::string_view name) {
  std::string path{parent};
  path.push_back('.');
  path.append(name);
  return path;
}

[[nodiscard]] std::optional<Error> AppendValue(
    const nlohmann::json& value, const std::string_view field_path,
    std::string* const output) {
  using JsonType = nlohmann::json::value_t;
  switch (value.type()) {
    case JsonType::null:
      output->append("null");
      return std::nullopt;
    case JsonType::boolean:
      output->append(value.get<bool>() ? "true" : "false");
      return std::nullopt;
    case JsonType::string:
      return AppendQuotedString(
          value.get_ref<const nlohmann::json::string_t&>(),
          field_path, output);
    case JsonType::number_integer: {
      const auto integer =
          value.get<nlohmann::json::number_integer_t>();
      if (!IsExactlyRepresentableAsBinary64(
              UnsignedMagnitude(
                  static_cast<std::int64_t>(integer)))) {
        return InvalidInput(
            std::string{field_path},
            "integer is not exactly representable as IEEE-754 binary64");
      }
      return AppendNumber(static_cast<double>(integer), field_path, output);
    }
    case JsonType::number_unsigned: {
      const auto integer =
          value.get<nlohmann::json::number_unsigned_t>();
      if (!IsExactlyRepresentableAsBinary64(
              static_cast<std::uint64_t>(integer))) {
        return InvalidInput(
            std::string{field_path},
            "integer is not exactly representable as IEEE-754 binary64");
      }
      return AppendNumber(static_cast<double>(integer), field_path, output);
    }
    case JsonType::number_float:
      return AppendNumber(
          value.get<nlohmann::json::number_float_t>(),
          field_path, output);
    case JsonType::array: {
      output->push_back('[');
      bool first = true;
      std::size_t index{};
      for (const auto& element : value) {
        if (!first) {
          output->push_back(',');
        }
        first = false;
        const std::string element_path =
            ArrayElementPath(field_path, index);
        if (auto error =
                AppendValue(element, element_path, output);
            error.has_value()) {
          return error;
        }
        ++index;
      }
      output->push_back(']');
      return std::nullopt;
    }
    case JsonType::object: {
      std::vector<ObjectEntry> entries;
      entries.reserve(value.size());
      for (auto iterator = value.cbegin();
           iterator != value.cend(); ++iterator) {
        ObjectEntry entry{
            .name = iterator.key(),
            .value = &iterator.value(),
            .utf16_name = {},
        };
        if (!AppendUtf16Units(entry.name, &entry.utf16_name)) {
          return InvalidInput(
              std::string{field_path},
              "object property name is not valid Unicode encoded as UTF-8");
        }
        entries.push_back(std::move(entry));
      }

      std::sort(
          entries.begin(), entries.end(),
          [](const ObjectEntry& left, const ObjectEntry& right) {
            return std::lexicographical_compare(
                left.utf16_name.begin(), left.utf16_name.end(),
                right.utf16_name.begin(), right.utf16_name.end());
          });

      output->push_back('{');
      bool first = true;
      for (const auto& entry : entries) {
        if (!first) {
          output->push_back(',');
        }
        first = false;
        if (auto error =
                AppendQuotedString(entry.name, field_path, output);
            error.has_value()) {
          return error;
        }
        output->push_back(':');
        const std::string value_path =
            ObjectValuePath(field_path, entry.name);
        if (auto error =
                AppendValue(*entry.value, value_path, output);
            error.has_value()) {
          return error;
        }
      }
      output->push_back('}');
      return std::nullopt;
    }
    case JsonType::binary:
    case JsonType::discarded:
      return InvalidInput(
          std::string{field_path},
          "value is not part of the I-JSON data model");
  }

  return InvalidInput(
      std::string{field_path},
      "unsupported JSON value type");
}

}  // namespace

Result<std::string> JcsCanonicalizer::Canonicalize(
    const nlohmann::json& value) {
  std::string canonical;
  if (auto error = AppendValue(value, "$", &canonical);
      error.has_value()) {
    return std::move(*error);
  }
  return canonical;
}

}  // namespace lunar::planning::v3
