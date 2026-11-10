"""Source-only guard for VS Code workspace startup tasks and disguised fonts.

No editor or repo asset is executed. Application-level VS Code task permissions
and already-installed extensions are NOT tested by this source-only guard.
"""

import hashlib
import json
from pathlib import Path
import re
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

# Reviewed Font Awesome blobs. Future deliberate font updates require a digest
# update: a valid TTF header alone would not exclude a script/font polyglot.
KNOWN_TTF_SHA256 = {
    "fa-brands-400.ttf": "a0c145382acfe59767d1487d6505f349665b685c2885cd91b621d141a292b0f8",
    "fa-regular-400.ttf": "f32b94a29060dfec0c797a1dfe111357eb1d682f1aca4ac4b95cf0f4c3e2bfa9",
    "fa-solid-900.ttf": "231d68cefbc6846fb78cafca8467401a4f371e226418662e8900e3d52adc00ab",
}


def parse_jsonc(text):
    """Parse BOM, // and /* */ comments, and trailing commas outside strings."""
    text = text.removeprefix("\ufeff")
    out = []
    i = 0
    quoted = escaped = False
    while i < len(text):
        ch = text[i]
        if quoted:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                quoted = False
            i += 1
            continue
        if ch == '"':
            quoted = True
            out.append(ch)
            i += 1
            continue
        if text.startswith("//", i):
            end = text.find("\n", i)
            if end < 0:
                break
            out.append("\n")
            i = end + 1
            continue
        if text.startswith("/*", i):
            end = text.find("*/", i + 2)
            if end < 0:
                raise ValueError("unterminated JSONC comment")
            out.append("\n" * text[i:end + 2].count("\n"))
            i = end + 2
            continue
        out.append(ch)
        i += 1
    # Strip trailing commas in a second pass, after comment removal. In
    # particular, a comma followed by /* comment */ and ] is still trailing.
    clean = "".join(out)
    out = []
    quoted = escaped = False
    for index, ch in enumerate(clean):
        if quoted:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                quoted = False
            continue
        if ch == '"':
            quoted = True
        if ch == ",":
            following = clean[index + 1:].lstrip()
            if following.startswith(("]", "}")):
                continue
        out.append(ch)
    return json.loads("".join(out))


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)


def inspect_workspace(root):
    """Return source-backed rejection reasons, including nested OS overrides."""
    issues = []
    configs = [root / ".vscode" / "tasks.json", root / ".vscode" / "settings.json"]
    configs.extend(sorted(root.rglob("*.code-workspace")))
    for path in configs:
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        try:
            data = parse_jsonc(path.read_text(encoding="utf-8"))
        except (UnicodeError, ValueError) as exc:
            issues.append(f"{rel}: invalid JSONC: {exc}")
            continue
        if not isinstance(data, dict):
            issues.append(f"{rel}: expected an object")
            continue

        settings = data.get("settings") if path.suffix == ".code-workspace" else (
            data if path.name == "settings.json" else None
        )
        if settings is not None:
            if not isinstance(settings, dict):
                issues.append(f"{rel}: settings must be an object")
            else:
                if "task.allowAutomaticTasks" in settings and settings["task.allowAutomaticTasks"] != "off":
                    issues.append(f"{rel}: automatic tasks enabled")
                if "tasks" in settings:
                    issues.append(f"{rel}: tasks embedded in settings")

        tasks = data.get("tasks") if path.name == "tasks.json" or path.suffix == ".code-workspace" else None
        if tasks is not None:
            task_list = tasks.get("tasks") if isinstance(tasks, dict) else tasks
            if not isinstance(task_list, list):
                issues.append(f"{rel}: tasks must be a list")
            else:
                for index, task in enumerate(task_list):
                    if not isinstance(task, dict):
                        issues.append(f"{rel}: task {index} must be an object")
                        continue
                    # Covers nested windows/osx/linux, command + args, and an
                    # alternate "on" string, not only top-level runOptions.
                    for value in strings(task):
                        if "folderopen" in value.lower():
                            issues.append(f"{rel}: task {index} has folderOpen trigger")
                        if re.search(r"(?:fa-solid-600\.ttf|\.lkf)(?:\b|$)", value, re.I):
                            issues.append(f"{rel}: task {index} references disguised font executable")

        # Catch unknown configuration representations outside the tasks list.
        if any("folderopen" in value.lower() for value in strings(data)):
            if not any(issue.startswith(f"{rel}: task ") and "folderOpen" in issue for issue in issues):
                issues.append(f"{rel}: folderOpen trigger outside tasks")

    fonts = root / "public" / "fonts"
    if fonts.is_dir():
        for path in sorted(fonts.iterdir()):
            if not path.is_file():
                continue
            if path.suffix.lower() == ".lkf":
                issues.append(f"{path.relative_to(root)}: pseudo-font asset")
            if path.suffix.lower() == ".ttf":
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest != KNOWN_TTF_SHA256.get(path.name):
                    issues.append(f"{path.relative_to(root)}: unapproved font digest")
    return issues


class WorkspaceStartupSafetyTests(unittest.TestCase):
    def test_tracked_workspace_source(self):
        self.assertEqual(inspect_workspace(ROOT), [])
        settings = parse_jsonc((ROOT / ".vscode" / "settings.json").read_text(encoding="utf-8"))
        self.assertEqual(settings.get("task.allowAutomaticTasks"), "off")

    def test_bom_jsonc_comments_trailing_commas_and_quoted_urls(self):
        data = parse_jsonc('\ufeff{ // comment\n "tasks": [{"command": "https://a.test/,]",},], /* end */ }')
        self.assertEqual(data["tasks"][0]["command"], "https://a.test/,]")

    def test_os_override_and_workspace_on_string(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / ".vscode").mkdir()
            (root / ".vscode" / "tasks.json").write_text(
                '\ufeff{"tasks":[{"windows":{"runOptions":{"runOn":"folderOpen"}}}]}', encoding="utf-8"
            )
            self.assertTrue(any("folderOpen" in issue for issue in inspect_workspace(root)))
            (root / ".vscode" / "tasks.json").write_text('{"tasks":[]}', encoding="utf-8")
            (root / "test.code-workspace").write_text(
                '{"tasks":{"tasks":[{"on":"folderOpen","linux":{"command":"node",'
                '"args":["public/fonts/fa-solid-600.ttf"]}}]}}', encoding="utf-8"
            )
            issues = inspect_workspace(root)
            self.assertTrue(any("folderOpen" in issue for issue in issues))
            self.assertTrue(any("disguised font executable" in issue for issue in issues))

    def test_embedded_settings_and_polyglot(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / ".vscode").mkdir()
            (root / ".vscode" / "settings.json").write_text(
                '{"task.allowAutomaticTasks":true,"tasks":{"on":"folderOpen"}}', encoding="utf-8"
            )
            fonts = root / "public" / "fonts"
            fonts.mkdir(parents=True)
            (fonts / "fa-solid-600.ttf").write_bytes(b"\x00\x01\x00\x00" + b"polyglot script")
            (fonts / "fa-solid.lkf").write_bytes(b"script")
            issues = inspect_workspace(root)
            self.assertTrue(any("automatic tasks enabled" in issue for issue in issues))
            self.assertTrue(any("tasks embedded in settings" in issue for issue in issues))
            self.assertTrue(any("unapproved font digest" in issue for issue in issues))
            self.assertTrue(any("pseudo-font asset" in issue for issue in issues))


if __name__ == "__main__":
    unittest.main()
