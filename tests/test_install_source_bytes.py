"""VCS dependencies keep reviewed bytes despite host Git line-ending settings."""

from unittest import mock

from modiff import install


def test_dependency_environment_overrides_line_endings_without_changing_parent(monkeypatch):
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "http.sslVerify")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "true")
    before = install.os.environ.copy()
    environment = install._dependency_environment()
    assert environment["GIT_CONFIG_COUNT"] == "3"
    assert environment["GIT_CONFIG_KEY_0"] == "http.sslVerify"
    assert environment["GIT_CONFIG_VALUE_0"] == "true"
    assert environment["GIT_CONFIG_KEY_1"] == "core.autocrlf"
    assert environment["GIT_CONFIG_VALUE_1"] == "false"
    assert environment["GIT_CONFIG_KEY_2"] == "core.eol"
    assert environment["GIT_CONFIG_VALUE_2"] == "lf"
    assert install.os.environ == before


def test_reviewed_vcs_install_ignores_previously_cached_translated_wheels(tmp_path):
    with mock.patch.object(install, "_run") as run:
        install._install_reviewed_diffusers("uv", tmp_path / "python")
    command = run.call_args.args[0]
    assert command[:5] == ["uv", "pip", "install", "--python", str(tmp_path / "python")]
    assert "--no-cache" in command
    assert "--no-deps" in command
    assert "--reinstall-package" in command
    assert command[-1] == (
        "diffusers @ git+https://github.com/huggingface/diffusers.git@"
        "2f7e0154a9db246e95c9ede43edba7db5b130805"
    )
    environment = run.call_args.kwargs["env"]
    assert environment["GIT_CONFIG_VALUE_" + str(int(environment["GIT_CONFIG_COUNT"]) - 2)] == "false"
