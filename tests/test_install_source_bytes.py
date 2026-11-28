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


def test_dependency_environment_treats_an_empty_inherited_git_count_as_unset(monkeypatch):
    monkeypatch.setenv("GIT_CONFIG_COUNT", "")

    environment = install._dependency_environment()

    assert environment["GIT_CONFIG_COUNT"] == "2"
    assert environment["GIT_CONFIG_KEY_0"] == "core.autocrlf"
    assert environment["GIT_CONFIG_VALUE_0"] == "false"
    assert environment["GIT_CONFIG_KEY_1"] == "core.eol"
    assert environment["GIT_CONFIG_VALUE_1"] == "lf"
    assert install.os.environ["GIT_CONFIG_COUNT"] == ""


def test_reviewed_published_wheel_install_uses_the_dependency_lock(tmp_path):
    with mock.patch.object(install, "_run") as run:
        install._install_reviewed_diffusers("uv", tmp_path / "python")
    command = run.call_args.args[0]
    assert command[:5] == ["uv", "pip", "install", "--python", str(tmp_path / "python")]
    assert "--no-deps" in command
    assert command[command.index("--reinstall-package") + 1] == "diffusers"
    assert "--no-cache" not in command
    assert "git+" not in command[-1]
    assert command[-1].startswith("https://files.pythonhosted.org/")
    assert command[-1].endswith(
        "diffusers-0.43.0-py3-none-any.whl#sha256=8afb8ff1b09d693854de0412d8bdc94fc9e6e6e4efefb2386c616f64a4dc9ac6"
    )
    assert run.call_args.kwargs == {}
