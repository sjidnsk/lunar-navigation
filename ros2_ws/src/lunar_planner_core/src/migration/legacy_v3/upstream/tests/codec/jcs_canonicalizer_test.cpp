#include <bit>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "lunar_path_planner/v3/codec/jcs_canonicalizer.hpp"

namespace lpp = lunar::planning::v3;

namespace {

struct NumberVector final {
  std::uint64_t ieee754_bits;
  std::string_view expected;
};

}  // namespace

TEST(JcsCanonicalizer, MatchesRfc8785AppendixBNumberVectors) {
  constexpr NumberVector kVectors[] = {
      {0x0000000000000000ULL, "0"},
      {0x8000000000000000ULL, "0"},
      {0x0000000000000001ULL, "5e-324"},
      {0x8000000000000001ULL, "-5e-324"},
      {0x7fefffffffffffffULL, "1.7976931348623157e+308"},
      {0xffefffffffffffffULL, "-1.7976931348623157e+308"},
      {0x4340000000000000ULL, "9007199254740992"},
      {0xc340000000000000ULL, "-9007199254740992"},
      {0x4430000000000000ULL, "295147905179352830000"},
      {0x44b52d02c7e14af5ULL, "9.999999999999997e+22"},
      {0x44b52d02c7e14af6ULL, "1e+23"},
      {0x44b52d02c7e14af7ULL, "1.0000000000000001e+23"},
      {0x444b1ae4d6e2ef4eULL, "999999999999999700000"},
      {0x444b1ae4d6e2ef4fULL, "999999999999999900000"},
      {0x444b1ae4d6e2ef50ULL, "1e+21"},
      {0x3eb0c6f7a0b5ed8cULL, "9.999999999999997e-7"},
      {0x3eb0c6f7a0b5ed8dULL, "0.000001"},
      {0x41b3de4355555553ULL, "333333333.3333332"},
      {0x41b3de4355555554ULL, "333333333.33333325"},
      {0x41b3de4355555555ULL, "333333333.3333333"},
      {0x41b3de4355555556ULL, "333333333.3333334"},
      {0x41b3de4355555557ULL, "333333333.33333343"},
      {0xbecbf647612f3696ULL, "-0.0000033333333333333333"},
      {0x43143ff3c1cb0959ULL, "1424953923781206.2"},
  };

  for (const auto& vector : kVectors) {
    SCOPED_TRACE(testing::Message()
                 << "IEEE-754 bits: 0x" << std::hex
                 << vector.ieee754_bits);
    const nlohmann::json value =
        std::bit_cast<double>(vector.ieee754_bits);
    const auto result = lpp::JcsCanonicalizer::Canonicalize(value);
    ASSERT_TRUE(lpp::IsOk(result));
    EXPECT_EQ(std::get<std::string>(result), vector.expected);
  }
}

TEST(JcsCanonicalizer, SortsPropertyNamesByUtf16CodeUnits) {
  const std::string emoji = "\xF0\x9F\x98\x80";
  const std::string private_use = "\xEE\x80\x80";

  nlohmann::json value = nlohmann::json::object();
  value[private_use] = 2;
  value[emoji] = 1;

  const auto result = lpp::JcsCanonicalizer::Canonicalize(value);

  ASSERT_TRUE(lpp::IsOk(result));
  EXPECT_EQ(std::get<std::string>(result),
            "{\"" + emoji + "\":1,\"" + private_use + "\":2}");
}

TEST(JcsCanonicalizer, UsesRfc8785StringEscaping) {
  std::string value;
  value.push_back('\0');
  value.push_back('\b');
  value.push_back('\t');
  value.push_back('\n');
  value.push_back('\f');
  value.push_back('\r');
  value.push_back('\x1f');
  value.push_back('"');
  value.push_back('\\');
  value.push_back('/');

  const nlohmann::json json_value = value;
  const auto result =
      lpp::JcsCanonicalizer::Canonicalize(json_value);

  ASSERT_TRUE(lpp::IsOk(result));
  EXPECT_EQ(std::get<std::string>(result),
            "\"\\u0000\\b\\t\\n\\f\\r\\u001f\\\"\\\\/\"");
}

TEST(JcsCanonicalizer, PreservesUnicodeWithoutNormalization) {
  const std::string precomposed = "\xC3\xA9";
  const std::string decomposed = "e\xCC\x81";
  const nlohmann::json value =
      nlohmann::json::array({precomposed, decomposed});

  const auto result = lpp::JcsCanonicalizer::Canonicalize(value);

  ASSERT_TRUE(lpp::IsOk(result));
  EXPECT_EQ(std::get<std::string>(result),
            "[\"" + precomposed + "\",\"" + decomposed + "\"]");
}

TEST(JcsCanonicalizer, RejectsNonFiniteNumbers) {
  for (const double value :
       {std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity()}) {
    const nlohmann::json json_value = value;
    const auto result =
        lpp::JcsCanonicalizer::Canonicalize(json_value);
    EXPECT_FALSE(lpp::IsOk(result));
  }
}

TEST(JcsCanonicalizer, RejectsInvalidUnicodeInValuesAndPropertyNames) {
  const std::vector<std::string> invalid_utf8 = {
      std::string{"\x80", 1},
      std::string{"\xC0\xAF", 2},
      std::string{"\xED\xA0\x80", 3},
      std::string{"\xF4\x90\x80\x80", 4},
      std::string{"\xE2\x82", 2},
  };

  for (const auto& text : invalid_utf8) {
    SCOPED_TRACE(testing::PrintToString(text));
    const nlohmann::json string_value = text;
    EXPECT_FALSE(lpp::IsOk(
        lpp::JcsCanonicalizer::Canonicalize(string_value)));

    nlohmann::json object = nlohmann::json::object();
    object[text] = true;
    EXPECT_FALSE(
        lpp::IsOk(lpp::JcsCanonicalizer::Canonicalize(object)));
  }
}

TEST(JcsCanonicalizer, RejectsIntegerNotExactlyRepresentableAsBinary64) {
  const nlohmann::json value =
      std::int64_t{9'007'199'254'740'993LL};

  const auto result = lpp::JcsCanonicalizer::Canonicalize(value);

  EXPECT_FALSE(lpp::IsOk(result));
}

TEST(JcsCanonicalizer, RejectsNonJsonBinaryValue) {
  const nlohmann::json value =
      nlohmann::json::binary({0x00U, 0xffU});

  const auto result = lpp::JcsCanonicalizer::Canonicalize(value);

  EXPECT_FALSE(lpp::IsOk(result));
}
