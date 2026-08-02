from __future__ import annotations

import subprocess

from tools.create_source_inventory import create_inventory


def test_inventory_records_each_repository_and_sha256(tmp_path):
    repository = tmp_path / "root"
    repository.mkdir()
    subprocess.run(["git", "init"], cwd=repository, check=True)
    subprocess.run(
        ["git", "remote", "add", "origin", "https://example.invalid/legacy.git"],
        cwd=repository,
        check=True,
    )
    (repository / "README.md").write_text("fixture\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repository, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-m",
            "fixture",
        ],
        cwd=repository,
        check=True,
    )

    output = create_inventory(
        repositories={"legacy_root": repository},
        selected_files=("README.md",),
    )

    assert output["schema_version"] == "lunar-migration-source-inventory/v1"
    assert len(output["repositories"]["legacy_root"]["commit"]) == 40
    assert len(output["files"][0]["sha256"]) == 64
    assert "absolute_path" not in output["files"][0]
