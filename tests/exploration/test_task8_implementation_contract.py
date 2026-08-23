from __future__ import annotations

import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "ros2_ws/src/lunar_pure_exploration_core"
CMAKE = PACKAGE / "CMakeLists.txt"
CANDIDATE_HEADER = PACKAGE / "include/lunar_pure_exploration_core/candidate_generator.hpp"
CANDIDATE_CPP = PACKAGE / "src/candidate_generator.cpp"
STATE_HEADER = PACKAGE / "include/lunar_pure_exploration_core/exploration_state_machine.hpp"
STATE_CPP = PACKAGE / "src/exploration_state_machine.cpp"
PROGRESS_HEADER = PACKAGE / "include/lunar_pure_exploration_core/progress_monitor.hpp"
PROGRESS_CPP = PACKAGE / "src/progress_monitor.cpp"


def _braced_body(source: str, signature: str) -> str:
    start = source.index(signature)
    brace = source.index("{", start)
    depth = 0
    for offset in range(brace, len(source)):
        if source[offset] == "{":
            depth += 1
        elif source[offset] == "}":
            depth -= 1
            if depth == 0:
                return source[brace + 1 : offset]
    raise AssertionError(f"unterminated braced declaration: {signature}")


def _cmake_call_body(source: str, signature: str) -> str:
    start = source.index(signature)
    open_paren = source.index("(", start)
    depth = 0
    for offset in range(open_paren, len(source)):
        if source[offset] == "(":
            depth += 1
        elif source[offset] == ")":
            depth -= 1
            if depth == 0:
                return source[open_paren + 1 : offset]
    raise AssertionError(f"unterminated CMake call: {signature}")


def _compact(source: str) -> str:
    return re.sub(r"\s+", " ", source).strip()


def _blank_non_newlines(chars: list[str], start: int, end: int) -> None:
    for offset in range(start, end):
        if chars[offset] not in "\r\n":
            chars[offset] = " "


def _cpp_code_only(source: str) -> str:
    """Apply phase-2 line splicing, then remove C++ comments and literals."""
    source = re.sub(r"\\(?:\r\n|\n)", "", source)
    output = list(source)
    offset = 0
    raw_literal = re.compile(r'(?:u8|u|U|L)?R"([^ ()\\\t\r\n]{0,16})\(')
    while offset < len(source):
        raw_match = raw_literal.match(source, offset)
        if raw_match is not None:
            terminator = f'){raw_match.group(1)}"'
            end = source.find(terminator, raw_match.end())
            assert end >= 0, "unterminated C++ raw string literal"
            end += len(terminator)
            _blank_non_newlines(output, offset, end)
            offset = end
            continue

        if source.startswith("//", offset):
            end = source.find("\n", offset + 2)
            end = len(source) if end < 0 else end
            _blank_non_newlines(output, offset, end)
            offset = end
            continue
        if source.startswith("/*", offset):
            end = source.find("*/", offset + 2)
            assert end >= 0, "unterminated C++ block comment"
            end += 2
            _blank_non_newlines(output, offset, end)
            offset = end
            continue

        if source[offset] in {'"', "'"}:
            quote = source[offset]
            end = offset + 1
            while end < len(source):
                if source[end] == "\\":
                    end += 2
                    continue
                if source[end] == quote:
                    end += 1
                    break
                end += 1
            else:
                raise AssertionError("unterminated C++ quoted literal")
            _blank_non_newlines(output, offset, end)
            offset = end
            continue

        offset += 1
    return "".join(output)


def _cmake_without_comments(source: str) -> str:
    """Remove CMake comments and literals without changing line boundaries."""
    output = list(source)
    offset = 0
    bracket_open = re.compile(r"\[(=*)\[")
    while offset < len(source):
        if source[offset] == '"':
            start = offset
            offset += 1
            while offset < len(source):
                if source[offset] == "\\":
                    offset += 2
                    continue
                if source[offset] == '"':
                    offset += 1
                    break
                offset += 1
            else:
                raise AssertionError("unterminated CMake quoted argument")
            _blank_non_newlines(output, start, offset)
            continue

        bracket_match = bracket_open.match(source, offset)
        if bracket_match is not None:
            start = offset
            terminator = f"]{bracket_match.group(1)}]"
            end = source.find(terminator, bracket_match.end())
            assert end >= 0, "unterminated CMake bracket argument"
            offset = end + len(terminator)
            _blank_non_newlines(output, start, offset)
            continue

        if source[offset] == "#":
            bracket_comment = bracket_open.match(source, offset + 1)
            if bracket_comment is not None:
                terminator = f"]{bracket_comment.group(1)}]"
                end = source.find(terminator, bracket_comment.end())
                assert end >= 0, "unterminated CMake bracket comment"
                end += len(terminator)
            else:
                end = source.find("\n", offset + 1)
                end = len(source) if end < 0 else end
            _blank_non_newlines(output, offset, end)
            offset = end
            continue

        if source[offset] == "\\":
            offset += 2
            continue
        offset += 1
    return "".join(output)


def _assert_no_conditional_compilation(source: str, context: str) -> None:
    conditional = re.compile(
        r"^[ \t]*#[ \t]*(?:if|ifdef|ifndef|elif|else|endif)\b",
        flags=re.MULTILINE,
    )
    assert conditional.search(source) is None, (
        f"conditional compilation is not allowed in reviewed {context}"
    )


def _top_level_statements(source: str) -> list[str]:
    statements: list[str] = []
    start = 0
    depths = {"(": 0, "[": 0, "{": 0}
    opening = {"(": ")", "[": "]", "{": "}"}
    closing = {closing: opening for opening, closing in opening.items()}
    for offset, character in enumerate(source):
        if character in opening:
            depths[character] += 1
        elif character in closing:
            opener = closing[character]
            assert depths[opener] > 0, "unbalanced declaration delimiter"
            depths[opener] -= 1
        elif character == ";" and not any(depths.values()):
            statements.append(source[start : offset + 1])
            start = offset + 1
    assert not any(depths.values()), "unbalanced declaration delimiter"
    assert source[start:].strip() == "", "unterminated declaration"
    return statements


def _data_member_signatures(source: str) -> list[tuple[str, str]]:
    signatures: list[tuple[str, str]] = []
    for statement in _top_level_statements(source):
        match = re.fullmatch(
            r"(?P<type>.+\S)\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
            r"(?:\s*(?:\{.*\}|=[^;]*))?;",
            _compact(statement),
            flags=re.DOTALL,
        )
        assert match is not None, f"unparsed data member: {_compact(statement)}"
        signatures.append((_compact(match.group("type")), match.group("name")))
    return signatures


def _code_identifiers(source: str) -> list[str]:
    code_only = re.sub(
        r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|//[^\n]*|/\*.*?\*/',
        " ",
        source,
        flags=re.DOTALL,
    )
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*", code_only)


def _validate_cmake(cmake: str) -> None:
    code = _cmake_without_comments(cmake)
    library = _cmake_call_body(code, "add_library(${PROJECT_NAME} SHARED")
    library_arguments = re.findall(r'"(?:\\.|[^"\\])*"|[^\s()]+', library)
    library_arguments = [argument.strip('"') for argument in library_arguments]
    for source in ("src/exploration_state_machine.cpp", "src/progress_monitor.cpp"):
        assert library_arguments.count(source) == 1, (
            f"core library must link {source} exactly once"
        )

    registrations = [
        (match.group(1), match.group(2))
        for match in re.finditer(
            r"ament_add_gtest\(\s*([A-Za-z0-9_]+)\s+([^\s)]+)", code
        )
    ]
    expected = {
        "test_exploration_state_machine": "test/test_exploration_state_machine.cpp",
        "test_progress_monitor": "test/test_progress_monitor.cpp",
    }
    for target, source in expected.items():
        target_registrations = [item for item in registrations if item[0] == target]
        source_registrations = [item for item in registrations if item[1] == source]
        assert target_registrations == [(target, source)], (
            f"{target} must be registered exactly once with {source}"
        )
        assert source_registrations == [(target, source)], (
            f"{source} must be registered exactly once as {target}"
        )


def _validate_candidate_ownership(candidate_header: str, candidate_cpp: str) -> None:
    candidate_header = _cpp_code_only(candidate_header)
    candidate_cpp = _cpp_code_only(candidate_cpp)
    _assert_no_conditional_compilation(candidate_header, "candidate header")
    _assert_no_conditional_compilation(candidate_cpp, "candidate implementation")
    view = _braced_body(candidate_header, "struct CandidateView")
    owner = "std::shared_ptr<const std::vector<std::int64_t>> frontier_canonical_key{};"
    assert _compact(view).count(_compact(owner)) == 1
    assert "std::vector" not in view.replace(owner, "")

    generate = _braced_body(candidate_cpp, "std::vector<CandidateView> CandidateGenerator::Generate(")
    owner_declaration = (
        "std::shared_ptr<const std::vector<std::int64_t>> frontier_canonical_key;"
    )
    owner_creation = "std::make_shared<const std::vector<std::int64_t>>("
    assert generate.count(owner_declaration) == 1
    assert generate.count(owner_creation) == 1
    assert "frontier.canonical_key" in _braced_body(
        generate, "if (!frontier_canonical_key)"
    )
    assert ".frontier_canonical_key = frontier_canonical_key," in generate
    assert generate.index(owner_declaration) < generate.index("for (const auto& edge : representatives)")


def _validate_active_goal(state_header: str, state_cpp: str) -> None:
    state_header = _cpp_code_only(state_header)
    state_cpp = _cpp_code_only(state_cpp)
    _assert_no_conditional_compilation(state_header, "state header")
    _assert_no_conditional_compilation(state_cpp, "state implementation")
    active_goal = _braced_body(state_header, "class ActiveGoal")
    compact_header = _compact(active_goal)
    assert "ActiveGoal(ActiveGoal&& other) noexcept;" in compact_header
    assert "ActiveGoal& operator=(ActiveGoal&&) = delete;" in compact_header
    assert "ActiveGoal(const ActiveGoal&) = delete;" in compact_header
    assert "ActiveGoal& operator=(const ActiveGoal&) = delete;" in compact_header
    assert re.search(r"\bActiveGoal\s*\(\s*\)", active_goal) is None
    assert "private:" in active_goal
    assert "bool owns_payload_{true};" in compact_header

    move_signature = "ActiveGoal::ActiveGoal(ActiveGoal&& other) noexcept"
    move_start = state_cpp.index(move_signature)
    move_open_brace = state_cpp.index("{", move_start)
    move_source = state_cpp[move_start:move_open_brace] + _braced_body(
        state_cpp, move_signature
    )
    move = _compact(move_source)
    assert "owns_payload_(other.owns_payload_)" in move
    assert "other.owns_payload_ = false;" in move
    assert "= default" not in move

    commit_source = _braced_body(
        state_cpp, "void ExplorationStateMachine::CommitGoal("
    )
    commit = _compact(commit_source)
    assert "if (!goal.owns_payload_)" in commit
    assert "throw std::invalid_argument(" in commit
    assert commit.index("if (!goal.owns_payload_)") < commit.index("active_goal_.emplace(std::move(goal));")


def _validate_progress_contract(progress_header: str, progress_cpp: str) -> None:
    progress_header_code = _cpp_code_only(progress_header)
    progress_cpp_code = _cpp_code_only(progress_cpp)
    _assert_no_conditional_compilation(progress_header_code, "progress header")
    _assert_no_conditional_compilation(progress_cpp_code, "progress implementation")
    parameters = _braced_body(progress_header_code, "struct ProgressParameters")
    assert _data_member_signatures(parameters) == [
        ("std::size_t", "maximum_executable_path_points"),
        ("std::chrono::steady_clock::duration", "timeout"),
        ("double", "minimum_progress_m"),
    ]

    identifiers = [
        identifier.lower()
        for source in (progress_header, progress_cpp)
        for identifier in _code_identifiers(source)
    ]
    forbidden = {
        "system_clock",
        "rclcpp",
        "ros",
        "gps",
        "stamp",
        "revision",
        "version",
        "freshness",
        "retry",
        "replan",
    }
    assert not [
        identifier
        for identifier in identifiers
        if any(fragment in identifier for fragment in forbidden)
    ]


def _validate_new_code_boundary(state_header: str, state_cpp: str) -> None:
    identifiers = [
        identifier.lower()
        for source in (state_header, state_cpp)
        for identifier in _code_identifiers(source)
    ]
    forbidden = {
        "system_clock",
        "rclcpp",
        "ros",
        "gps",
        "stamp",
        "revision",
        "version",
        "freshness",
    }
    assert not [
        identifier
        for identifier in identifiers
        if any(fragment in identifier for fragment in forbidden)
    ]


def _current_sources() -> tuple[str, str, str, str, str, str, str]:
    return tuple(
        path.read_text(encoding="utf-8")
        for path in (
            CMAKE,
            CANDIDATE_HEADER,
            CANDIDATE_CPP,
            STATE_HEADER,
            STATE_CPP,
            PROGRESS_HEADER,
            PROGRESS_CPP,
        )
    )


def _validate_all(
    cmake: str,
    candidate_header: str,
    candidate_cpp: str,
    state_header: str,
    state_cpp: str,
    progress_header: str,
    progress_cpp: str,
) -> None:
    _validate_cmake(cmake)
    _validate_candidate_ownership(candidate_header, candidate_cpp)
    _validate_active_goal(state_header, state_cpp)
    _validate_progress_contract(progress_header, progress_cpp)
    _validate_new_code_boundary(state_header, state_cpp)


def test_task8_implementation_contract() -> None:
    _validate_all(*_current_sources())


@pytest.mark.parametrize(
    "removed",
    ["src/exploration_state_machine.cpp", "src/progress_monitor.cpp"],
)
def test_library_source_mutations_are_rejected(removed: str) -> None:
    cmake, *rest = _current_sources()
    mutant = cmake.replace(removed, f"src/mutated_{Path(removed).name}", 1)
    with pytest.raises(AssertionError):
        _validate_all(mutant, *rest)


@pytest.mark.parametrize(
    "target",
    ["test_exploration_state_machine", "test_progress_monitor"],
)
def test_gtest_registration_mutations_are_rejected(target: str) -> None:
    cmake, *rest = _current_sources()
    mutant = cmake.replace(target, f"mutated_{target}", 1)
    with pytest.raises(AssertionError):
        _validate_all(mutant, *rest)


@pytest.mark.parametrize(
    "mutant",
    [
        lambda cmake: cmake.replace(
            "  src/progress_monitor.cpp",
            "  # src/progress_monitor.cpp",
            1,
        ),
        lambda cmake: cmake.replace(
            "  ament_add_gtest(test_progress_monitor test/test_progress_monitor.cpp)",
            "  # ament_add_gtest(test_progress_monitor test/test_progress_monitor.cpp)",
            1,
        ),
    ],
    ids=["commented-library-source", "commented-gtest-registration"],
)
def test_cmake_comment_bypass_mutations_are_rejected(mutant) -> None:
    cmake, *rest = _current_sources()
    with pytest.raises(AssertionError):
        _validate_all(mutant(cmake), *rest)


@pytest.mark.parametrize(
    "literal",
    [
        'set(dummy "{registration}")',
        'set(dummy [=[{registration}]=])',
    ],
    ids=["quoted-string", "bracket-argument"],
)
def test_cmake_literal_registration_bypass_mutations_are_rejected(
    literal: str,
) -> None:
    cmake, *rest = _current_sources()
    registration = (
        "ament_add_gtest(test_progress_monitor test/test_progress_monitor.cpp)"
    )
    mutant = cmake.replace(registration, literal.format(registration=registration), 1)
    assert mutant != cmake
    with pytest.raises(AssertionError):
        _validate_all(mutant, *rest)


def test_duplicate_library_source_mutation_is_rejected() -> None:
    cmake, *rest = _current_sources()
    mutant = cmake.replace(
        "  src/progress_monitor.cpp",
        "  src/progress_monitor.cpp\n  src/progress_monitor.cpp",
        1,
    )
    with pytest.raises(AssertionError):
        _validate_all(mutant, *rest)


def test_duplicate_gtest_target_mutation_is_rejected() -> None:
    cmake, *rest = _current_sources()
    registration = (
        "  ament_add_gtest(test_progress_monitor test/test_progress_monitor.cpp)"
    )
    mutant = cmake.replace(registration, f"{registration}\n{registration}", 1)
    with pytest.raises(AssertionError):
        _validate_all(mutant, *rest)


def test_duplicate_gtest_source_mutation_is_rejected() -> None:
    cmake, *rest = _current_sources()
    registration = (
        "  ament_add_gtest(test_progress_monitor test/test_progress_monitor.cpp)"
    )
    mutant = cmake.replace(
        registration,
        (
            f"{registration}\n"
            "  ament_add_gtest(test_progress_monitor_alias "
            "test/test_progress_monitor.cpp)"
        ),
        1,
    )
    with pytest.raises(AssertionError):
        _validate_all(mutant, *rest)


def test_per_view_key_vector_mutation_is_rejected() -> None:
    cmake, candidate_header, *rest = _current_sources()
    mutant = candidate_header.replace(
        "std::shared_ptr<const std::vector<std::int64_t>> frontier_canonical_key{};",
        "std::vector<std::int64_t> frontier_canonical_key;",
        1,
    )
    with pytest.raises(AssertionError):
        _validate_all(cmake, mutant, *rest)


def test_per_view_owner_creation_mutation_is_rejected() -> None:
    cmake, candidate_header, candidate_cpp, *rest = _current_sources()
    mutant = candidate_cpp.replace(
        "for (const double yaw_offset : parameters_.yaw_offsets_rad) {",
        (
            "std::shared_ptr<const std::vector<std::int64_t>> frontier_canonical_key;\n"
            "        for (const double yaw_offset : parameters_.yaw_offsets_rad) {"
        ),
        1,
    ).replace(
        "std::shared_ptr<const std::vector<std::int64_t>> frontier_canonical_key;\n"
        "    const std::size_t count",
        "const std::size_t count",
        1,
    )
    with pytest.raises(AssertionError):
        _validate_all(cmake, candidate_header, mutant, *rest)


def test_candidate_owner_assignment_comment_bypass_mutation_is_rejected() -> None:
    cmake, candidate_header, candidate_cpp, *rest = _current_sources()
    mutant = candidate_cpp.replace(
        ".frontier_canonical_key = frontier_canonical_key,",
        (
            ".frontier_canonical_key = nullptr,\n"
            "              // .frontier_canonical_key = frontier_canonical_key,"
        ),
        1,
    )
    with pytest.raises(AssertionError):
        _validate_all(cmake, candidate_header, mutant, *rest)


def test_candidate_owner_assignment_inactive_branch_mutation_is_rejected() -> None:
    cmake, candidate_header, candidate_cpp, *rest = _current_sources()
    assignment = ".frontier_canonical_key = frontier_canonical_key,"
    mutant = candidate_cpp.replace(
        assignment,
        (
            f"#if 0\n              {assignment}\n"
            "#else\n"
            "              .frontier_canonical_key = nullptr,\n"
            "#endif"
        ),
        1,
    )
    assert mutant != candidate_cpp
    with pytest.raises(AssertionError):
        _validate_all(cmake, candidate_header, mutant, *rest)


def test_active_goal_move_and_consumption_mutations_are_rejected() -> None:
    cmake, candidate_header, candidate_cpp, state_header, state_cpp, *rest = _current_sources()
    header_mutant = state_header.replace(
        "ActiveGoal& operator=(ActiveGoal&&) = delete;",
        "ActiveGoal& operator=(ActiveGoal&&) = default;",
        1,
    )
    with pytest.raises(AssertionError):
        _validate_all(cmake, candidate_header, candidate_cpp, header_mutant, state_cpp, *rest)
    source_mutant = state_cpp.replace("other.owns_payload_ = false;", "other.owns_payload_ = true;", 1)
    with pytest.raises(AssertionError):
        _validate_all(cmake, candidate_header, candidate_cpp, state_header, source_mutant, *rest)
    commit_mutant = state_cpp.replace("if (!goal.owns_payload_)", "if (goal.owns_payload_)", 1)
    with pytest.raises(AssertionError):
        _validate_all(cmake, candidate_header, candidate_cpp, state_header, commit_mutant, *rest)


@pytest.mark.parametrize(
    ("needle", "replacement"),
    [
        (
            "other.owns_payload_ = false;",
            (
                "other.owns_payload_ = true;\n"
                "  // other.owns_payload_ = false;"
            ),
        ),
        (
            "if (!goal.owns_payload_)",
            (
                "if (goal.owns_payload_)\n"
                "  // if (!goal.owns_payload_)"
            ),
        ),
    ],
    ids=["move-source-invalidation", "commit-consumption-guard"],
)
def test_active_goal_comment_bypass_mutations_are_rejected(
    needle: str, replacement: str
) -> None:
    (
        cmake,
        candidate_header,
        candidate_cpp,
        state_header,
        state_cpp,
        *rest,
    ) = _current_sources()
    mutant = state_cpp.replace(needle, replacement, 1)
    with pytest.raises(AssertionError):
        _validate_all(
            cmake,
            candidate_header,
            candidate_cpp,
            state_header,
            mutant,
            *rest,
        )


def test_active_goal_inactive_invalidation_mutation_is_rejected() -> None:
    (
        cmake,
        candidate_header,
        candidate_cpp,
        state_header,
        state_cpp,
        *rest,
    ) = _current_sources()
    mutant = state_cpp.replace(
        "  other.owns_payload_ = false;",
        (
            "#if 0\n"
            "  other.owns_payload_ = false;\n"
            "#else\n"
            "  other.owns_payload_ = true;\n"
            "#endif"
        ),
        1,
    )
    assert mutant != state_cpp
    with pytest.raises(AssertionError):
        _validate_all(
            cmake,
            candidate_header,
            candidate_cpp,
            state_header,
            mutant,
            *rest,
        )


def test_active_goal_inactive_commit_guard_mutation_is_rejected() -> None:
    (
        cmake,
        candidate_header,
        candidate_cpp,
        state_header,
        state_cpp,
        *rest,
    ) = _current_sources()
    guard = (
        "  if (!goal.owns_payload_) {\n"
        "    throw std::invalid_argument(\"active goal payload was already consumed\");\n"
        "  }"
    )
    mutant = state_cpp.replace(guard, f"#if 0\n{guard}\n#endif", 1)
    assert mutant != state_cpp
    with pytest.raises(AssertionError):
        _validate_all(
            cmake,
            candidate_header,
            candidate_cpp,
            state_header,
            mutant,
            *rest,
        )


def test_active_goal_outer_inactive_function_mutation_is_rejected() -> None:
    (
        cmake,
        candidate_header,
        candidate_cpp,
        state_header,
        state_cpp,
        *rest,
    ) = _current_sources()
    signature = "ActiveGoal::ActiveGoal(ActiveGoal&& other) noexcept"
    start = state_cpp.index(signature)
    open_brace = state_cpp.index("{", start)
    body = _braced_body(state_cpp, signature)
    end = open_brace + len(body) + 2
    valid_function = state_cpp[start:end]
    broken_function = valid_function.replace(
        "other.owns_payload_ = false;", "other.owns_payload_ = true;", 1
    )
    mutant = (
        state_cpp[:start]
        + f"#if 0\n{valid_function}\n#else\n{broken_function}\n#endif"
        + state_cpp[end:]
    )
    assert mutant != state_cpp
    with pytest.raises(AssertionError):
        _validate_all(
            cmake,
            candidate_header,
            candidate_cpp,
            state_header,
            mutant,
            *rest,
        )


@pytest.mark.parametrize(
    "splice_newline", ["\n", "\r\n"], ids=["lf", "crlf"]
)
def test_active_goal_line_spliced_outer_inactive_function_mutation_is_rejected(
    splice_newline: str,
) -> None:
    (
        cmake,
        candidate_header,
        candidate_cpp,
        state_header,
        state_cpp,
        *rest,
    ) = _current_sources()
    signature = "ActiveGoal::ActiveGoal(ActiveGoal&& other) noexcept"
    start = state_cpp.index(signature)
    open_brace = state_cpp.index("{", start)
    body = _braced_body(state_cpp, signature)
    end = open_brace + len(body) + 2
    valid_function = state_cpp[start:end]
    broken_function = valid_function.replace(
        "other.owns_payload_ = false;", "other.owns_payload_ = true;", 1
    )
    mutant = (
        state_cpp[:start]
        + f"#\\{splice_newline}if 0\n"
        + valid_function
        + f"\n#\\{splice_newline}else\n"
        + broken_function
        + f"\n#\\{splice_newline}endif"
        + state_cpp[end:]
    )
    assert mutant != state_cpp
    with pytest.raises(AssertionError):
        _validate_all(
            cmake,
            candidate_header,
            candidate_cpp,
            state_header,
            mutant,
            *rest,
        )


def test_progress_parameter_and_authority_mutations_are_rejected() -> None:
    cmake, candidate_header, candidate_cpp, state_header, state_cpp, progress_header, progress_cpp = _current_sources()
    parameter_mutant = progress_header.replace(
        "std::size_t maximum_executable_path_points;",
        "std::size_t maximum_executable_path_points;\n  std::uint8_t retry_count;",
        1,
    )
    with pytest.raises(AssertionError):
        _validate_all(cmake, candidate_header, candidate_cpp, state_header, state_cpp, parameter_mutant, progress_cpp)
    authority_mutant = progress_cpp + "\nstd::chrono::system_clock::time_point retry_stamp;\n"
    with pytest.raises(AssertionError):
        _validate_all(cmake, candidate_header, candidate_cpp, state_header, state_cpp, progress_header, authority_mutant)


def test_progress_synonym_authority_member_mutation_is_rejected() -> None:
    (
        cmake,
        candidate_header,
        candidate_cpp,
        state_header,
        state_cpp,
        progress_header,
        progress_cpp,
    ) = _current_sources()
    mutant = progress_header.replace(
        "  double minimum_progress_m{0.2};",
        (
            "  double minimum_progress_m{0.2};\n"
            "  unsigned attempts_remaining{2U};"
        ),
        1,
    )
    assert mutant != progress_header
    with pytest.raises(AssertionError):
        _validate_all(
            cmake,
            candidate_header,
            candidate_cpp,
            state_header,
            state_cpp,
            mutant,
            progress_cpp,
        )


def test_new_code_time_and_revision_mutations_are_rejected() -> None:
    cmake, candidate_header, candidate_cpp, state_header, state_cpp, progress_header, progress_cpp = _current_sources()
    state_mutant = state_cpp + "\nconst auto map_revision = 1;\n"
    with pytest.raises(AssertionError):
        _validate_all(cmake, candidate_header, candidate_cpp, state_header, state_mutant, progress_header, progress_cpp)
