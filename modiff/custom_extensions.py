"""Explicit, content-bound local extension approvals. This is not a sandbox.

Staging and inspection only read files/metadata. Imports happen on enable or at
startup for an unchanged approved package. Dependencies are never installed here.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import tomllib
from types import ModuleType

from packaging.requirements import Requirement, InvalidRequirement

MAX_FILES = 256
MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_TOTAL_BYTES = 16 * 1024 * 1024
SOURCE_SUFFIXES = {".py", ".json", ".toml", ".txt", ".md", ".yaml", ".yml", ".js", ".jsx", ".css", ".svg"}
MODULE_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
REVISION = re.compile(r"^[a-f0-9]{40}$")


class ExtensionError(ValueError):
    pass


def module_name(value):
    if not isinstance(value, str) or not MODULE_NAME.fullmatch(value):
        raise ExtensionError(
            "Module name must start with a letter and contain up to 64 letters, numbers or underscores."
        )
    return value


def immutable_revision(value):
    if not isinstance(value, str) or not REVISION.fullmatch(value):
        raise ExtensionError("Remote extensions require an exact lowercase 40-character commit revision.")
    return value


def _linked(path):
    return path.is_symlink() or bool(getattr(path, "is_junction", lambda: False)())


def json_object(content, label):
    try:
        value = json.loads(content)
    except (ValueError, RecursionError) as error:
        raise ExtensionError(f"{label} must contain bounded valid JSON.") from error
    if not isinstance(value, dict):
        raise ExtensionError(f"{label} must contain a JSON object.")
    return value


def source_files(root, *, hub=False):
    """Read a bounded code package; never copy weights, environments or credentials."""
    if _linked(root) or not root.is_dir():
        raise ExtensionError("Extension source must be a regular directory, not a link.")
    files, total, count = {}, 0, 0
    pending = [(root, 0)]
    while pending:
        parent, depth = pending.pop()
        for path in sorted(parent.iterdir()):
            count += 1
            if count > 16384 or depth > 16:
                raise ExtensionError("Extension directory exceeds the bounded scan limit.")
            if path.name.startswith(".") or path.name == "__pycache__":
                continue
            if _linked(path):
                # Hub snapshots use file links into this repository's blobs.
                blob_root = root.parent.parent / "blobs"
                if not hub or path.is_dir() or _linked(blob_root):
                    raise ExtensionError("Extension sources cannot contain links.")
                try:
                    path.resolve(strict=True).relative_to(blob_root.resolve(strict=True))
                except (ValueError, OSError) as error:
                    raise ExtensionError("Hub source link escaped its repository blobs.") from error
            if path.is_dir():
                pending.append((path, depth + 1))
            elif path.suffix.lower() in SOURCE_SUFFIXES:
                if not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
                    raise ExtensionError("Extension file is not regular or exceeds 2 MiB.")
                with path.open("rb") as reader:
                    content = reader.read(MAX_FILE_BYTES + 1)
                total += len(content)
                if len(content) > MAX_FILE_BYTES or total > MAX_TOTAL_BYTES or len(files) >= MAX_FILES:
                    raise ExtensionError("Extension source exceeds its file/byte limit.")
                files[path.relative_to(root).as_posix()] = content
    return files


def dependencies(files):
    declared = []
    if "requirements.txt" in files:
        declared.extend(
            line.strip()
            for line in files["requirements.txt"].decode().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    if "pyproject.toml" in files:
        project = tomllib.loads(files["pyproject.toml"].decode()).get("project", {})
        values = project.get("dependencies", []) if isinstance(project, dict) else None
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            raise ExtensionError("Project dependencies must be a list of requirement strings.")
        declared.extend(values)
    if "modular_config.json" in files:
        requirements = json_object(files["modular_config.json"], "modular_config.json").get("requirements") or {}
        if isinstance(requirements, dict):
            declared.extend(f"{name}{specifier}" for name, specifier in requirements.items())
        elif isinstance(requirements, list):
            declared.extend(requirements)
        else:
            raise ExtensionError("Modular requirements must be a package/specifier mapping or list.")
    if len(declared) > MAX_FILES:
        raise ExtensionError("An extension may declare at most 256 dependencies.")
    result = []
    if any(not isinstance(text, str) for text in declared):
        raise ExtensionError("Dependency declarations must be strings.")
    for text in dict.fromkeys(declared):
        if not isinstance(text, str) or len(text) > 512:
            raise ExtensionError("Dependency declarations must be bounded requirement strings.")
        try:
            requirement = Requirement(text)
            if requirement.url:
                raise InvalidRequirement("Direct URL dependencies require manual review.")
            installed = importlib.metadata.version(requirement.name)
            applies = requirement.marker is None or requirement.marker.evaluate()
            status = "satisfied" if not applies or installed in requirement.specifier else "incompatible"
        except importlib.metadata.PackageNotFoundError:
            installed = None
            status = "satisfied" if requirement.marker and not requirement.marker.evaluate() else "missing"
        except InvalidRequirement:
            installed, status = None, "manual_review"
        result.append({"requirement": text, "installed": installed, "status": status})
    return result


def code_identity(files, deps):
    digest = hashlib.sha256(b"modiff-extension-v1\0")
    for key, content in sorted(files.items()):
        digest.update(key.encode() + b"\0" + hashlib.sha256(content).digest())
    digest.update(json.dumps(deps, sort_keys=True).encode())
    return "sha256:" + digest.hexdigest()


def runtime_role(files):
    """Operator-reviewed resource declaration; never model qualification."""
    config = json_object(files.get("modiff_extension.json", b"{}"), "modiff_extension.json")
    if not isinstance(config, dict) or set(config) - {"runtimeRole"}:
        raise ExtensionError("modiff_extension.json only accepts runtimeRole.")
    role = config.get("runtimeRole", "manual")
    if not isinstance(role, str) or role not in {"manual", "data", "connected_components"}:
        raise ExtensionError("runtimeRole must be manual, data or connected_components.")
    return role


def _preview(files):
    """Literal Python metadata or validated Mellon/MoDiff JSON; no eval/import."""
    sidecar = next(
        (name for name in ("modiff_pipeline_config.json", "mellon_pipeline_config.json") if name in files), None
    )
    if sidecar and "modular_config.json" in files:
        from modules.ModularDiffusers.pipeline_schema import MoDiffPipelineConfig
        from modules.ModularDiffusers.dynamic_node import _custom_node_contract

        config = MoDiffPipelineConfig.from_json_bytes(
            files[sidecar], source_label=sidecar, allow_omitted_custom_model_inputs=True
        )
        contract = _custom_node_contract(config)
        for name in [*contract["input_names"], *contract["model_input_names"]]:
            if name not in contract["params"]:
                raise ExtensionError(f"Modular sidecar input {name} needs a declared field.")
            contract["params"][name]["isInput"] = True
        return {
            "kind": "modular",
            "nodes": {
                "Block": {
                    "type": "custom",
                    "label": contract.get("label") or config.label or "Custom Modular Block",
                    "category": "Custom",
                    "description": "Operator-enabled Modular Diffusers block.",
                    "params": contract["params"],
                }
            },
            "contract": contract,
            "diagnostics": [],
        }
    if "__init__.py" not in files or "main.py" not in files:
        raise ExtensionError("Provide __init__.py and main.py, or modular_config.json and a Mellon/MoDiff sidecar.")
    nodes, diagnostics = {}, []
    for filename, content in files.items():
        if not filename.endswith(".py"):
            continue
        tree = ast.parse(content, filename=filename)
        if filename != "main.py":
            continue
        constants = {}
        for statement in tree.body:
            if isinstance(statement, ast.Assign):
                try:
                    value = ast.literal_eval(statement.value)
                    for target in statement.targets:
                        if isinstance(target, ast.Name):
                            constants[target.id] = value
                except (ValueError, TypeError):
                    pass
            if not isinstance(statement, ast.ClassDef) or not any(
                isinstance(base, ast.Name) and base.id == "NodeBase" for base in statement.bases
            ):
                continue
            node = {
                "type": "custom",
                "label": statement.name,
                "category": "Custom",
                "description": ast.get_docstring(statement) or "",
                "params": {},
            }
            for field in statement.body:
                if not isinstance(field, ast.Assign):
                    continue
                for target in field.targets:
                    if not isinstance(target, ast.Name) or target.id not in node:
                        continue
                    try:
                        node[target.id] = (
                            constants[field.value.id]
                            if isinstance(field.value, ast.Name)
                            else ast.literal_eval(field.value)
                        )
                    except (KeyError, ValueError, TypeError):
                        diagnostics.append(
                            f"{statement.name}.{target.id} requires import; shown after explicit enable."
                        )
            nodes[statement.name] = node
    if len(nodes) > 256:
        raise ExtensionError("An extension may declare at most 256 nodes.")
    for action, node in nodes.items():
        if not isinstance(node["label"], str) or not node["label"]:
            node["label"] = action
        if not isinstance(node["params"], dict) or len(node["params"]) > 256:
            raise ExtensionError("Node params must be an object with at most 256 fields.")
        for key, field in node["params"].items():
            if (
                not isinstance(key, str)
                or key in {"__proto__", "constructor", "prototype"}
                or not isinstance(field, dict)
            ):
                raise ExtensionError("Node params must map safe field names to objects.")
        json.dumps(node, allow_nan=False)
    return {"kind": "python", "nodes": nodes, "diagnostics": diagnostics}


class _FreshSourceLoader(importlib.machinery.SourceFileLoader):
    def __init__(self, fullname, path, content):
        super().__init__(fullname, path)
        self.content = content

    def get_code(self, fullname):
        # Python's timestamp/size pyc cache can serve stale code after a fast edit.
        return compile(self.content, self.path, "exec", dont_inherit=True)


class _PackageFinder(importlib.abc.MetaPathFinder):
    def __init__(self, prefix, root, files):
        self.prefix, self.root, self.files = prefix, root, files

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self.prefix and not fullname.startswith(self.prefix + "."):
            return None
        relative = fullname[len(self.prefix) :].lstrip(".").split(".") if fullname != self.prefix else []
        base = self.root.joinpath(*relative)
        package_key = "/".join([*relative, "__init__.py"])
        source_key = "/".join(relative) + ".py"
        selected = package_key if package_key in self.files else source_key
        filename = self.root / selected
        if selected in self.files:
            return importlib.util.spec_from_file_location(
                fullname,
                filename,
                loader=_FreshSourceLoader(fullname, str(filename), self.files[selected]),
                submodule_search_locations=[str(base)] if selected == package_key else None,
            )
        prefix = "/".join(relative) + "/"
        if relative and any(name.startswith(prefix) for name in self.files):
            spec = importlib.machinery.ModuleSpec(fullname, loader=None, is_package=True)
            spec.submodule_search_locations = []
            return spec
        raise ModuleNotFoundError(
            f"{fullname} is absent from the approved source snapshot. Review and reload changed code."
        )


class ExtensionStore:
    def __init__(self, root=None):
        self.root = Path(root or "custom").absolute()
        if _linked(self.root):
            raise ExtensionError("The custom module root must not be a link.")

    def path(self, name):
        name = module_name(name)
        for path in (self.root / name, self.root / ".disabled" / name):
            if path.exists() or path.is_symlink():
                if _linked(path) or _linked(path.parent):
                    raise ExtensionError("Installed module directories must not be links.")
                return path
        return self.root / name

    def _state(self):
        file = self.root / ".extensions.json"
        if not file.exists():
            return {}
        if _linked(file) or file.stat().st_size > MAX_FILE_BYTES:
            raise ExtensionError("Invalid extension approval file.")
        state = json_object(file.read_text(), "extension approval file")
        if any(not isinstance(record, dict) for record in state.values()):
            raise ExtensionError("Invalid extension approval file.")
        return state

    def _save(self, name, record):
        self.root.mkdir(parents=True, exist_ok=True)
        state = self._state()
        state[name] = record
        fd, temporary = tempfile.mkstemp(prefix=".extensions-", dir=self.root)
        try:
            with os.fdopen(fd, "w") as writer:
                json.dump(state, writer, sort_keys=True)
            os.replace(temporary, self.root / ".extensions.json")
        finally:
            Path(temporary).unlink(missing_ok=True)

    def inspect(self, name):
        path = self.path(name)
        files = source_files(path)
        deps = dependencies(files)
        code_hash = code_identity(files, deps)
        record = self._state().get(name, {})
        enabled = record.get("enabled") is True and record.get("codeHash") == code_hash
        preview = _preview(files)
        return {
            "name": name,
            "moduleKey": f"custom.{name}",
            "source": "custom",
            "kind": record.get("kind", "local"),
            "revision": record.get("revision"),
            "runtimeRole": runtime_role(files),
            "enabled": enabled,
            "status": "enabled" if enabled else "changed" if record.get("enabled") else "disabled",
            "path": str(path),
            "codeHash": code_hash,
            "approvedHash": record.get("codeHash"),
            "files": [
                {"name": key, "bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
                for key, value in sorted(files.items())
            ],
            "dependencies": deps,
            "preview": preview,
            "diagnostic": record.get("diagnostic"),
            "nodes": sorted(preview["nodes"]),
            "nodeCount": len(preview["nodes"]) if enabled else 0,
            "hasInit": "__init__.py" in files,
            "hasMain": "main.py" in files,
            "hasGit": record.get("kind") == "git",
            "canUpdate": False,
            "canDisable": bool(record.get("enabled")),
            "canEnable": not enabled,
        }

    def list(self):
        names = set()
        for directory in (self.root, self.root / ".disabled"):
            if directory.is_dir() and not _linked(directory):
                names.update(p.name for p in directory.iterdir() if p.is_dir() and MODULE_NAME.fullmatch(p.name))
        result = []
        for name in sorted(names):
            try:
                result.append(self.inspect(name))
            except (OSError, ValueError, SyntaxError) as error:
                result.append(
                    {
                        "name": name,
                        "moduleKey": f"custom.{name}",
                        "source": "custom",
                        "enabled": False,
                        "status": "error",
                        "path": str(self.root / name),
                        "codeHash": None,
                        "dependencies": [],
                        "files": [],
                        "preview": None,
                        "diagnostic": f"{type(error).__name__}: {error}"[:2048],
                        "nodes": [],
                        "nodeCount": 0,
                        "hasGit": False,
                        "canUpdate": False,
                        "canDisable": True,
                        "canEnable": False,
                    }
                )
        return result

    def stage(self, *, kind, source, name, revision=None):
        name = module_name(name)
        if self.path(name).exists():
            raise ExtensionError("Module already exists. Edit/reload it, or stage a revision under a new name.")
        if not isinstance(source, str) or not source.strip() or len(source) > 4096:
            raise ExtensionError("A bounded source string is required.")
        if not isinstance(kind, str) or kind not in {"local", "git", "hub"}:
            raise ExtensionError("Source kind must be local, git or hub.")
        if kind != "local":
            immutable_revision(revision)
        with tempfile.TemporaryDirectory(prefix="modiff-extension-") as temporary:
            if kind == "git":
                from urllib.parse import urlsplit

                url = urlsplit(source)
                if (
                    url.scheme != "https"
                    or not url.hostname
                    or url.username
                    or url.password
                    or url.query
                    or url.fragment
                ):
                    raise ExtensionError("Git sources require an HTTPS URL without credentials, query or fragment.")
                checkout = Path(temporary) / "checkout"

                def git(*args):
                    try:
                        result = subprocess.run(
                            ["git", "-c", f"core.hooksPath={temporary}/empty-hooks", *args],
                            capture_output=True,
                            timeout=120,
                            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
                        )
                    except (OSError, subprocess.TimeoutExpired) as error:
                        raise ExtensionError(
                            "Git staging could not finish. Verify local Git access and retry."
                        ) from error
                    if result.returncode:
                        raise ExtensionError("Git staging failed; verify the URL, commit and local Git access.")
                    return result.stdout

                git("init", str(checkout))
                git("-C", str(checkout), "remote", "add", "origin", source)
                git("-C", str(checkout), "config", "remote.origin.promisor", "true")
                git("-C", str(checkout), "config", "remote.origin.partialclonefilter", "blob:none")
                git("-C", str(checkout), "fetch", "--filter=blob:none", "--depth=1", "origin", revision)
                if git("-C", str(checkout), "rev-parse", "FETCH_HEAD").decode().strip() != revision:
                    raise ExtensionError("Git returned a different revision.")
                files, total = {}, 0
                entries = git("-C", str(checkout), "ls-tree", "-rz", "--full-tree", "FETCH_HEAD").split(b"\0")
                if len(entries) > 16384:
                    raise ExtensionError("Git source exceeds the bounded tree scan limit.")
                for entry in entries:
                    if not entry:
                        continue
                    header, raw_path = entry.split(b"\t", 1)
                    mode, object_type, object_id = header.decode().split(" ")
                    relative = raw_path.decode("utf-8")
                    parts = relative.split("/")
                    if any(part.startswith(".") or part == "__pycache__" for part in parts):
                        continue
                    if mode == "120000" or object_type != "blob":
                        raise ExtensionError(
                            "Git source links and submodules must be replaced with reviewed regular source files."
                        )
                    if Path(relative).suffix.lower() not in SOURCE_SUFFIXES:
                        continue
                    if (
                        len(parts) > 16
                        or any(part in {"", ".."} for part in parts)
                        or "\\" in relative
                        or ":" in relative
                    ):
                        raise ExtensionError("Invalid Git source path.")
                    size = int(git("-C", str(checkout), "cat-file", "-s", object_id))
                    if size > MAX_FILE_BYTES or total + size > MAX_TOTAL_BYTES or len(files) >= MAX_FILES:
                        raise ExtensionError("Git source exceeds its file/byte limit.")
                    files[relative] = git("-C", str(checkout), "cat-file", "blob", object_id)
                    total += size
            elif kind == "hub":
                from huggingface_hub import snapshot_download
                from huggingface_hub.utils import validate_repo_id

                validate_repo_id(source)
                snapshot = Path(
                    snapshot_download(
                        source, revision=revision, allow_patterns=["*.py", "*.json", "*.toml", "*.txt", "*.md"]
                    )
                )
                if (
                    snapshot.name != revision
                    or snapshot.parent.name != "snapshots"
                    or any(_linked(path) for path in (snapshot, snapshot.parent, snapshot.parent.parent))
                ):
                    raise ExtensionError("Hub returned a different snapshot revision.")
                files = source_files(snapshot, hub=True)
            else:
                files = source_files(Path(source).expanduser().absolute())
            _preview(files)
            dependencies(files)
            runtime_role(files)
            staged = Path(temporary) / "package"
            staged.mkdir()
            for relative, content in files.items():
                target = staged / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
            self.root.mkdir(parents=True, exist_ok=True)
            # Copy into a same-filesystem temporary directory before atomic publication.
            import shutil

            with tempfile.TemporaryDirectory(prefix=".stage-", dir=self.root) as publish:
                shutil.copytree(staged, Path(publish) / name)
                os.rename(Path(publish) / name, self.root / name)
            self._save(name, {"kind": kind, "revision": revision, "enabled": False})
        return self.inspect(name)

    def require_enabled(self, name):
        item = self.inspect(name)
        if not item["enabled"]:
            raise ExtensionError(
                "Custom module is disabled or its code/dependencies changed. Review and enable/reload it in Custom nodes."
            )
        return item

    def asset(self, name, file):
        if (
            not isinstance(file, str)
            or not file
            or "\\" in file
            or any(part in {".", "..", ""} for part in file.split("/"))
        ):
            raise ExtensionError("Invalid custom asset path.")
        files = source_files(self.path(name))
        record = self._state().get(name, {})
        if record.get("enabled") is not True or record.get("codeHash") != code_identity(files, dependencies(files)):
            raise ExtensionError("Custom browser assets require approval for the current source.")
        try:
            return files["web/" + file]
        except KeyError as error:
            raise FileNotFoundError("Custom asset not found in the approved package.") from error

    def unload(self, name):
        prefix = f"custom.{module_name(name)}"
        for key in list(sys.modules):
            if key == prefix or key.startswith(prefix + "."):
                sys.modules.pop(key, None)
        sys.meta_path[:] = [
            finder
            for finder in sys.meta_path
            if not isinstance(finder, _PackageFinder)
            or not (finder.prefix == prefix or finder.prefix.startswith(prefix + "."))
        ]
        parent = sys.modules.get("custom")
        if parent is not None:
            parent.__dict__.pop(name, None)
        importlib.invalidate_caches()

    def _load(self, item):
        name = item["name"]
        files = source_files(Path(item["path"]))
        if code_identity(files, dependencies(files)) != item["codeHash"]:
            raise ExtensionError("Source changed before import. Review and reload after editing is complete.")
        self.unload(name)
        # Never execute a root custom/__init__.py while discovering packages.
        if "custom" not in sys.modules:
            namespace = ModuleType("custom")
            namespace.__path__ = []
            sys.modules["custom"] = namespace
        if item["preview"]["kind"] == "modular":
            from modiff.custom_modular_extension import load_modular_extension

            return load_modular_extension(item, files)
        sys.meta_path.insert(0, _PackageFinder(item["moduleKey"], Path(item["path"]), files))
        package = importlib.import_module(item["moduleKey"])
        main = importlib.import_module(f"{item['moduleKey']}.main")
        from modules import parse_node_class
        from modiff.NodeBase import NodeBase

        registry = dict(getattr(package, "MODULE_MAP", {}))
        for stem in getattr(package, "MODULE_PARSE", ["main"]):
            if not isinstance(stem, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", stem):
                raise ExtensionError("MODULE_PARSE must list simple Python file stems.")
            tree = ast.parse(files.get(f"{stem}.py", b""), filename=f"{stem}.py")
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.ClassDef)
                    and node.name not in registry
                    and any(isinstance(base, ast.Name) and base.id == "NodeBase" for base in node.bases)
                ):
                    registry[node.name] = parse_node_class(node, package)
        if not registry:
            raise ExtensionError("No NodeBase classes were registered; export them through main.py and __init__.py.")
        for action in registry:
            klass = getattr(main, action, None)
            if not isinstance(klass, type) or not issubclass(klass, NodeBase):
                raise ExtensionError(f"{action} must be a NodeBase subclass exported from main.py.")
        json.dumps(registry, allow_nan=False)
        return registry

    def enable(self, name, *, code_hash, consent):
        if consent is not True:
            raise ExtensionError("Explicit code execution consent is required.")
        item = self.inspect(name)
        if code_hash != item["codeHash"]:
            raise ExtensionError("Source or dependencies changed after inspection. Review the current code hash.")
        if any(dep["status"] != "satisfied" for dep in item["dependencies"]):
            raise ExtensionError(
                "Resolve declared dependencies manually, then inspect again. No packages were installed."
            )
        record = {**self._state().get(name, {}), "enabled": False, "diagnostic": None}
        self._save(name, record)
        try:
            registry = self._load(item)
            if self.inspect(name)["codeHash"] != code_hash:
                raise ExtensionError("Source changed during import. Review and reload after editing is complete.")
        except Exception as error:
            self.unload(name)
            record["diagnostic"] = f"{type(error).__name__}: {error}"[:2048]
            self._save(name, record)
            raise ExtensionError(record["diagnostic"]) from error
        self._save(name, {**record, "enabled": True, "codeHash": code_hash})
        return registry

    def disable(self, name):
        self._save(module_name(name), {**self._state().get(name, {}), "enabled": False})
        self.unload(name)

    def load_enabled(self, registry):
        for item in self.list():
            if item["enabled"]:
                try:
                    registry[item["moduleKey"]] = self.enable(item["name"], code_hash=item["codeHash"], consent=True)
                except ExtensionError:
                    # Stored diagnostics remain discoverable; one failed extension cannot break startup.
                    continue
