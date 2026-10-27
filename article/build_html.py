#!/usr/bin/env python3
"""Build a self-contained Chinese HTML article and a Zhihu Markdown variant.

The builder intentionally performs no network access and writes nothing until all
source documents, referenced resources, and KaTeX assets have been validated.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import io
import json
import mimetypes
import os
import re
import sys
import tempfile
import unicodedata
import urllib.parse
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from string import Template
from typing import Any, Sequence

try:
    from markdown_it import MarkdownIt
    from markdown_it.token import Token
except ImportError as exc:  # pragma: no cover - exercised only without dependencies
    MarkdownIt = None  # type: ignore[assignment]
    Token = Any  # type: ignore[misc,assignment]
    _MARKDOWN_IMPORT_ERROR: ImportError | None = exc
else:
    _MARKDOWN_IMPORT_ERROR = None

try:
    from pygments import highlight as pygments_highlight
    from pygments.formatters import HtmlFormatter
    from pygments.lexers import TextLexer, get_lexer_by_name
    from pygments.util import ClassNotFound
except ImportError as exc:  # pragma: no cover - exercised only without dependencies
    _PYGMENTS_IMPORT_ERROR: ImportError | None = exc
else:
    _PYGMENTS_IMPORT_ERROR = None


SCHEMA_VERSION = 1
ATTACHMENT_SUFFIXES = {".csv", ".json", ".jsonl", ".log", ".txt", ".tsv"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".bmp", ".ico", ".avif"}
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
DEPENDENCY_HELP = (
    "缺少纯 Python 构建依赖。请在隔离环境安装（不要改全局环境）：\n"
    "  python3 -m venv /tmp/article-build-venv\n"
    "  /tmp/article-build-venv/bin/python -m pip install markdown-it-py Pygments\n"
    "然后使用 /tmp/article-build-venv/bin/python 运行本脚本。"
)
KATEX_HELP = (
    "请在 index.json 中设置 \"katex\": {\"directory\": \"/path/to/katex/dist\"}，"
    "或将发行版放到 article/assets/katex。也可在隔离临时目录安装：\n"
    "  npm install --prefix /tmp/article-katex katex\n"
    "对应 directory 为 /tmp/article-katex/node_modules/katex/dist。"
    "构建器不会联网下载 KaTeX。"
)


class BuildError(Exception):
    """A user-facing build or validation error."""


@dataclass
class Resource:
    source: Path
    archive_path: str
    kind: str
    media_type: str
    data: bytes = field(repr=False)
    source_label: str = ""

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()

    @property
    def data_uri(self) -> str:
        encoded = base64.b64encode(self.data).decode("ascii")
        return f"data:{self.media_type};base64,{encoded}"


@dataclass
class Chapter:
    path: Path
    chapter_id: str
    tokens: list[Token]
    html: str = ""


@dataclass
class BuildResult:
    title: str
    html: str
    zhihu: str
    formulas: list[dict[str, Any]]
    resources: list[Resource]
    source_documents: list[dict[str, Any]]
    katex_directory: Path | None
    katex_version: str | None

    @property
    def image_count(self) -> int:
        return sum(resource.kind == "image" for resource in self.resources)

    @property
    def attachment_count(self) -> int:
        return sum(resource.kind == "attachment" for resource in self.resources)

    def formula_log_bytes(self) -> bytes:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "replacement": "readable-plain-text",
            "note": "知乎正文中的公式由同源 TeX 自动转为可读文本；source 保留原始公式，未手工改写数字。",
            "formulas": self.formulas,
        }
        return (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


@dataclass
class LoadedConfig:
    path: Path
    data: dict[str, Any]
    raw: bytes
    repo_root: Path

    @property
    def article_dir(self) -> Path:
        return self.path.parent

    @property
    def assets_dir(self) -> Path:
        return self.article_dir / "assets"

    @property
    def evidence_dir(self) -> Path:
        return self.repo_root / "evidence"

    @property
    def manifest_dir(self) -> Path:
        return self.repo_root / "manifest"


# ---------------------------------------------------------------------------
# Generic helpers and configuration validation


def _require_dependencies() -> None:
    if _MARKDOWN_IMPORT_ERROR is not None or _PYGMENTS_IMPORT_ERROR is not None:
        missing = []
        if _MARKDOWN_IMPORT_ERROR is not None:
            missing.append("markdown-it-py")
        if _PYGMENTS_IMPORT_ERROR is not None:
            missing.append("Pygments")
        raise BuildError(f"{'、'.join(missing)} 未安装。\n{DEPENDENCY_HELP}")


def _read_bytes(path: Path, description: str) -> bytes:
    if not path.exists():
        raise BuildError(f"{description}不存在：{path}")
    if not path.is_file():
        raise BuildError(f"{description}不是文件：{path}")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise BuildError(f"无法读取{description} {path}：{exc}") from exc


def _read_text(path: Path, description: str) -> str:
    raw = _read_bytes(path, description)
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BuildError(f"{description}必须是 UTF-8：{path}（{exc}）") from exc


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _relative_label(path: Path, repo_root: Path) -> str:
    if _is_within(path, repo_root):
        return path.relative_to(repo_root).as_posix()
    return str(path)


def _load_config(config_path: Path) -> LoadedConfig:
    path = config_path.expanduser().resolve()
    raw = _read_bytes(path, "配置文件")
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BuildError(f"配置文件不是有效 UTF-8 JSON：{path}（{exc}）") from exc
    if not isinstance(data, dict):
        raise BuildError("index.json 顶层必须是 JSON object")
    if not isinstance(data.get("title"), str) or not data["title"].strip():
        raise BuildError("index.json 缺少非空字符串 title")
    if "description" in data and not isinstance(data["description"], str):
        raise BuildError("index.json 的 description 必须是字符串")
    chapters = data.get("chapters")
    if not isinstance(chapters, list) or not chapters:
        raise BuildError("index.json 的 chapters 必须是非空数组")
    if "attachments" in data and not isinstance(data["attachments"], list):
        raise BuildError("index.json 的 attachments 必须是数组")
    toc_depth = data.get("toc_depth", 3)
    if not isinstance(toc_depth, int) or isinstance(toc_depth, bool) or not 1 <= toc_depth <= 6:
        raise BuildError("toc_depth 必须是 1 到 6 的整数")
    return LoadedConfig(path=path, data=data, raw=raw, repo_root=path.parent.parent.resolve())


def _chapter_path(config: LoadedConfig, raw_path: str) -> Path:
    if not raw_path or urllib.parse.urlsplit(raw_path).scheme:
        raise BuildError(f"无效章节路径：{raw_path!r}")
    candidate = Path(raw_path)
    if candidate.is_absolute():
        raise BuildError(f"章节必须使用相对 article/index.json 的路径：{raw_path}")
    resolved = (config.article_dir / candidate).resolve()
    if not _is_within(resolved, config.article_dir):
        raise BuildError(f"章节不能位于 article 目录之外：{raw_path}")
    _read_bytes(resolved, "Markdown 章节")
    return resolved


def _clean_archive_path(value: str, description: str = "归档路径") -> str:
    if not value or "\\" in value or value.startswith("/"):
        raise BuildError(f"{description}必须是非空的 POSIX 相对路径：{value!r}")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise BuildError(f"{description}包含非法路径段：{value!r}")
    return PurePosixPath(*parts).as_posix()


def _slugify(value: str, fallback: str = "section") -> str:
    normalized = unicodedata.normalize("NFKC", value).lower()
    output: list[str] = []
    previous_dash = False
    for char in normalized:
        category = unicodedata.category(char)
        if char.isalnum() or category.startswith(("L", "N")):
            output.append(char)
            previous_dash = False
        elif char.isspace() or char in "-_":
            if output and not previous_dash:
                output.append("-")
                previous_dash = True
    slug = "".join(output).strip("-")
    return slug[:100] or fallback


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _media_type(path: Path) -> str:
    if path.suffix.lower() == ".svg":
        return "image/svg+xml"
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"


# ---------------------------------------------------------------------------
# Source resolution and resource collection


class ResourceCollector:
    def __init__(self, config: LoadedConfig):
        self.config = config
        self.resources: list[Resource] = []
        self._by_source: dict[Path, Resource] = {}
        self._by_archive: dict[str, Resource] = {}

    def _resolve(self, raw_path: str, base_dir: Path, kind: str) -> Path:
        split = urllib.parse.urlsplit(raw_path)
        if split.scheme or split.netloc:
            raise BuildError(f"本地{kind}路径不能包含 URL scheme：{raw_path}")
        decoded = urllib.parse.unquote(split.path)
        candidate = Path(decoded)
        if not decoded or candidate.is_absolute():
            raise BuildError(f"{kind}必须使用非空相对路径：{raw_path}")

        candidates: list[Path] = []
        if candidate.parts and candidate.parts[0] in {"article", "evidence", "manifest"}:
            candidates.append((self.config.repo_root / candidate).resolve())
        else:
            candidates.append((base_dir / candidate).resolve())
            config_relative = (self.config.article_dir / candidate).resolve()
            if config_relative not in candidates:
                candidates.append(config_relative)

        allowed_roots = (self.config.assets_dir.resolve(), self.config.evidence_dir.resolve(), self.config.manifest_dir.resolve())
        for path in candidates:
            if path.exists():
                if not any(_is_within(path, root) for root in allowed_roots):
                    roots = "、".join(str(root) for root in allowed_roots)
                    raise BuildError(f"{kind}只能来自 article/assets、evidence 或 manifest：{raw_path}；允许根目录：{roots}")
                if not path.is_file():
                    raise BuildError(f"{kind}源路径不是文件：{path}")
                return path

        attempted = "、".join(str(path) for path in candidates)
        raise BuildError(f"{kind}源文件不存在：{raw_path}；已尝试：{attempted}")

    def _default_archive_path(self, source: Path) -> str:
        assets = self.config.assets_dir.resolve()
        evidence = self.config.evidence_dir.resolve()
        if _is_within(source, assets):
            return f"assets/{source.relative_to(assets).as_posix()}"
        if _is_within(source, evidence):
            return f"evidence/{source.relative_to(evidence).as_posix()}"
        manifest = self.config.manifest_dir.resolve()
        if _is_within(source, manifest):
            return f"manifest/{source.relative_to(manifest).as_posix()}"
        raise BuildError(f"资源不在允许的目录中：{source}")

    def add(self, raw_path: str, base_dir: Path, kind: str, name: str | None = None) -> Resource:
        if kind not in {"image", "attachment"}:
            raise ValueError(f"Unknown resource kind: {kind}")
        source = self._resolve(raw_path, base_dir, "图片" if kind == "image" else "附件")
        suffix = source.suffix.lower()
        if kind == "attachment" and suffix not in ATTACHMENT_SUFFIXES:
            allowed = "、".join(sorted(ATTACHMENT_SUFFIXES))
            raise BuildError(f"不支持的附件类型 {suffix or '(无扩展名)'}：{source}；仅支持 {allowed}")
        if kind == "image" and suffix not in IMAGE_SUFFIXES:
            allowed = "、".join(sorted(IMAGE_SUFFIXES))
            raise BuildError(f"不支持的图片类型 {suffix or '(无扩展名)'}：{source}；仅支持 {allowed}")

        archive_path = self._default_archive_path(source)
        if name is not None:
            archive_path = f"attachments/{_clean_archive_path(name, '附件 name')}"
        archive_path = _clean_archive_path(archive_path)

        existing = self._by_source.get(source)
        if existing is not None:
            if existing.archive_path != archive_path or existing.kind != kind:
                raise BuildError(
                    f"同一源文件被声明为不一致的资源：{source}（{existing.kind}/{existing.archive_path} 与 {kind}/{archive_path}）"
                )
            return existing
        collision = self._by_archive.get(archive_path)
        if collision is not None and collision.source != source:
            raise BuildError(f"归档路径冲突：{archive_path} 同时指向 {collision.source} 和 {source}")

        resource = Resource(
            source=source,
            archive_path=archive_path,
            kind=kind,
            media_type=_media_type(source),
            data=_read_bytes(source, "图片" if kind == "image" else "附件"),
            source_label=_relative_label(source, self.config.repo_root),
        )
        self.resources.append(resource)
        self._by_source[source] = resource
        self._by_archive[archive_path] = resource
        return resource


# ---------------------------------------------------------------------------
# Markdown math rules


def _find_unescaped(source: str, delimiter: str, start: int) -> int:
    position = start
    while True:
        position = source.find(delimiter, position)
        if position < 0:
            return -1
        backslashes = 0
        cursor = position - 1
        while cursor >= 0 and source[cursor] == "\\":
            backslashes += 1
            cursor -= 1
        if backslashes % 2 == 0:
            return position
        position += len(delimiter)


def _math_block(state: Any, start_line: int, end_line: int, silent: bool) -> bool:
    source = state.src
    start = state.bMarks[start_line] + state.tShift[start_line]
    line_end = state.eMarks[start_line]
    if source[start : start + 2] != "$$":
        return False

    first_line = source[start + 2 : line_end]
    close = _find_unescaped(first_line, "$$", 0)
    if close >= 0:
        if first_line[close + 2 :].strip():
            return False
        content = first_line[:close]
        next_line = start_line + 1
    else:
        lines = [first_line]
        next_line = start_line + 1
        found = False
        while next_line < end_line:
            line_start = state.bMarks[next_line] + state.tShift[next_line]
            line_stop = state.eMarks[next_line]
            line = source[line_start:line_stop]
            close = _find_unescaped(line, "$$", 0)
            if close >= 0:
                if line[close + 2 :].strip():
                    return False
                lines.append(line[:close])
                found = True
                next_line += 1
                break
            lines.append(line)
            next_line += 1
        if not found:
            if silent:
                return False
            raise BuildError(f"未闭合的 $$ 块级公式，起始行 {start_line + 1}")
        content = "\n".join(lines).strip("\n")

    if silent:
        return True
    token = state.push("math_block", "math", 0)
    token.block = True
    token.map = [start_line, next_line]
    token.markup = "$$"
    token.content = content
    token.meta = {"display": True}
    state.line = next_line
    return True


def _math_inline(state: Any, silent: bool) -> bool:
    source = state.src
    start = state.pos
    if start >= len(source) or source[start] != "$":
        return False
    delimiter = "$$" if source.startswith("$$", start) else "$"
    content_start = start + len(delimiter)
    if content_start >= len(source):
        return False
    if delimiter == "$" and source[content_start].isspace():
        return False

    position = content_start
    while position < len(source):
        close = _find_unescaped(source, delimiter, position)
        if close < 0:
            return False
        content = source[content_start:close]
        if delimiter == "$$" or not content[:1].isspace() and not content[-1:].isspace():
            single_line_valid = delimiter == "$$" or (bool(content) and "\n" not in content)
            if single_line_valid and content:
                if not silent:
                    token = state.push("math_inline", "math", 0)
                    token.markup = delimiter
                    token.content = content
                    token.meta = {"display": delimiter == "$$"}
                state.pos = close + len(delimiter)
                return True
        position = close + len(delimiter)
    return False


# ---------------------------------------------------------------------------
# TeX to readable plain text for Zhihu


_TEX_COMMANDS = {
    "geq": " ≥ ",
    "ge": " ≥ ",
    "leq": " ≤ ",
    "le": " ≤ ",
    "neq": " ≠ ",
    "ne": " ≠ ",
    "approx": " ≈ ",
    "sim": " ~ ",
    "equiv": " ≡ ",
    "cdot": " · ",
    "times": " × ",
    "div": " ÷ ",
    "pm": " ± ",
    "mp": " ∓ ",
    "infty": " ∞ ",
    "in": " ∈ ",
    "notin": " ∉ ",
    "subseteq": " ⊆ ",
    "supseteq": " ⊇ ",
    "subset": " ⊂ ",
    "supset": " ⊃ ",
    "cup": " ∪ ",
    "cap": " ∩ ",
    "sum": "∑",
    "prod": "∏",
    "int": "∫",
    "partial": "∂",
    "nabla": "∇",
    "Rightarrow": " ⇒ ",
    "Longrightarrow": " ⇒ ",
    "rightarrow": " → ",
    "longrightarrow": " → ",
    "to": " → ",
    "Leftarrow": " ⇐ ",
    "leftarrow": " ← ",
    "leftrightarrow": " ↔ ",
    "mapsto": " ↦ ",
    "neg": "¬",
    "lnot": "¬",
    "land": " ∧ ",
    "wedge": " ∧ ",
    "lor": " ∨ ",
    "vee": " ∨ ",
    "forall": "∀",
    "exists": "∃",
    "lceil": "⌈",
    "rceil": "⌉",
    "lfloor": "⌊",
    "rfloor": "⌋",
    "min": "min",
    "max": "max",
    "argmin": "arg min",
    "argmax": "arg max",
    "log": "log",
    "ln": "ln",
    "exp": "exp",
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "ldots": "…",
    "cdots": "…",
    "dots": "…",
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "delta": "δ",
    "epsilon": "ε",
    "varepsilon": "ε",
    "theta": "θ",
    "lambda": "λ",
    "mu": "μ",
    "nu": "ν",
    "pi": "π",
    "rho": "ρ",
    "sigma": "σ",
    "tau": "τ",
    "phi": "φ",
    "varphi": "φ",
    "omega": "ω",
    "Gamma": "Γ",
    "Delta": "Δ",
    "Theta": "Θ",
    "Lambda": "Λ",
    "Pi": "Π",
    "Sigma": "Σ",
    "Phi": "Φ",
    "Omega": "Ω",
}


def _braced_argument(text: str, start: int) -> tuple[str, int] | None:
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    index = start
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:index], index + 1
        index += 1
    return None


def _replace_balanced_command(text: str, pattern: str, arity: int, replacement) -> str:
    command = re.compile(pattern)
    position = 0
    while True:
        match = command.search(text, position)
        if match is None:
            return text
        cursor = match.end()
        arguments = []
        for _ in range(arity):
            cursor += len(re.match(r"\s*", text[cursor:]).group(0))
            parsed = _braced_argument(text, cursor)
            if parsed is None:
                break
            argument, cursor = parsed
            arguments.append(argument)
        if len(arguments) != arity:
            position = match.end()
            continue
        rendered = replacement(*arguments)
        text = text[:match.start()] + rendered + text[cursor:]
        position = match.start()


def tex_to_readable(source: str) -> str:
    """Convert TeX to readable text without changing numeric literals.

    This is deliberately a textual fallback, not a second TeX implementation.
    The original expression is always retained in formula-conversions JSON.
    """

    text = source.strip()
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _replace_balanced_command(text, r"\\(?:d|t)?frac\b", 2, lambda numerator, denominator: f"(({numerator})/({denominator}))")
    text = _replace_balanced_command(text, r"\\sqrt\b", 1, lambda argument: f"sqrt({argument})")
    for command in ("text", "textrm", "textit", "textbf", "mathrm", "mathit", "mathbf", "mathsf", "mathtt", "operatorname", "boldsymbol", "bm"):
        text = re.sub(rf"\\{command}\s*\{{([^{{}}]*)\}}", r"\1", text)
    text = re.sub(r"\\begin\s*\{([^{}]*)\}", r"[\1] ", text)
    text = re.sub(r"\\end\s*\{[^{}]*\}", "", text)
    text = text.replace("\\\\", " ; ")
    text = re.sub(r"\\(?:quad|qquad|,|;|:|!)", " ", text)
    text = re.sub(r"\\[ \t\n]", " ", text)
    text = re.sub(r"\\(?:left|right|displaystyle|limits|nolimits)\b", "", text)
    text = re.sub(r"\\([A-Za-z]+)", lambda match: _TEX_COMMANDS.get(match.group(1), match.group(1)), text)
    replacements = {
        "\\_": "_",
        "\\{": "{",
        "\\}": "}",
        "\\%": "%",
        "\\#": "#",
        "\\&": "&",
        "\\$": "＄",
        "~": " ",
        "&": " ",
        "{": "",
        "}": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = text.replace("\n", " ; ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?，。；：！？])", r"\1", text)
    return text.strip() or "空公式"


# ---------------------------------------------------------------------------
# HTML rendering and token preprocessing


def _plain_inline(tokens: Sequence[Token] | None) -> str:
    if not tokens:
        return ""
    parts: list[str] = []
    for token in tokens:
        if token.type in {"text", "code_inline", "text_special", "html_inline"}:
            parts.append(token.content)
        elif token.type in {"math_inline", "math_block"}:
            parts.append(tex_to_readable(token.content))
        elif token.type == "image":
            parts.append(_plain_inline(token.children) or token.content)
        elif token.type in {"softbreak", "hardbreak"}:
            parts.append(" ")
    return "".join(parts).strip()


def _highlight_code(code: str, language: str) -> str:
    try:
        lexer = get_lexer_by_name(language) if language else TextLexer()
    except ClassNotFound:
        lexer = TextLexer()
    return pygments_highlight(code, lexer, HtmlFormatter(nowrap=True)).rstrip("\n")


def _make_markdown_parser() -> Any:
    md = MarkdownIt("commonmark", {"html": False, "linkify": False, "typographer": False})
    md.enable("table")
    md.enable("strikethrough")
    md.block.ruler.before(
        "fence",
        "math_block",
        _math_block,
        {"alt": ["paragraph", "reference", "blockquote", "list"]},
    )
    md.inline.ruler.after("escape", "math_inline", _math_inline)

    def render_math(renderer: Any, tokens: Sequence[Token], index: int, options: Any, env: Any) -> str:
        token = tokens[index]
        display = token.type == "math_block" or bool(token.meta.get("display"))
        tag = "div" if token.type == "math_block" else "span"
        classes = "math math-display" if display else "math math-inline"
        formula_id = html.escape(str(token.meta.get("formula_id", "")), quote=True)
        source = html.escape(token.content, quote=False)
        return (
            f'<{tag} id="{formula_id}" class="{classes}" data-display="{str(display).lower()}">'
            f"{source}</{tag}>\n"
            if tag == "div"
            else f'<{tag} id="{formula_id}" class="{classes}" data-display="{str(display).lower()}">{source}</{tag}>'
        )

    def render_fence(renderer: Any, tokens: Sequence[Token], index: int, options: Any, env: Any) -> str:
        token = tokens[index]
        language = token.info.strip().split(maxsplit=1)[0] if token.info.strip() else ""
        safe_language = html.escape(language, quote=True)
        class_attribute = f' class="language-{safe_language}"' if language else ""
        highlighted = _highlight_code(token.content, language)
        return f'<pre class="codehilite"><code{class_attribute}>{highlighted}</code></pre>\n'

    md.add_render_rule("math_inline", render_math)
    md.add_render_rule("math_block", render_math)
    md.add_render_rule("fence", render_fence)
    return md


def _process_image(token: Token, chapter: Chapter, collector: ResourceCollector) -> None:
    source = token.attrGet("src") or ""
    if source.startswith("data:"):
        token.meta["package_src"] = source
        return
    split = urllib.parse.urlsplit(source)
    if split.scheme or split.netloc:
        raise BuildError(f"HTML 必须自包含，不能引用远程图片：{chapter.path}: {source}；请先下载到 article/assets 或 evidence")
    resource = collector.add(source, chapter.path.parent, "image")
    token.meta["package_src"] = resource.archive_path
    token.attrSet("src", resource.data_uri)
    token.attrSet("loading", "lazy")


def _process_link(token: Token, chapter: Chapter, collector: ResourceCollector) -> None:
    href = token.attrGet("href") or ""
    split = urllib.parse.urlsplit(href)
    if split.scheme or split.netloc or href.startswith(("#", "//")):
        return
    suffix = Path(urllib.parse.unquote(split.path)).suffix.lower()
    if suffix in ATTACHMENT_SUFFIXES:
        resource = collector.add(href, chapter.path.parent, "attachment")
        token.attrSet("href", resource.archive_path)


def _process_chapter_tokens(
    chapter: Chapter,
    collector: ResourceCollector,
    formulas: list[dict[str, Any]],
    heading_ids: set[str],
    toc_depth: int,
) -> list[dict[str, Any]]:
    toc: list[dict[str, Any]] = []
    table_depth = 0
    tokens = chapter.tokens

    def register_formula(math_token: Token, line: int | None, in_table: bool) -> None:
        formula_id = f"formula-{len(formulas) + 1:04d}"
        display = math_token.type == "math_block" or bool(math_token.meta.get("display"))
        formula = {
            "id": formula_id,
            "chapter": chapter.path.name,
            "chapter_id": chapter.chapter_id,
            "line": line,
            "kind": "display" if display else "inline",
            "in_table": in_table,
            "replacement": "readable-plain-text",
            "source": math_token.content,
            "text": tex_to_readable(math_token.content),
        }
        math_token.meta["formula_id"] = formula_id
        formulas.append(formula)

    for index, token in enumerate(tokens):
        if token.type == "table_open":
            table_depth += 1
        elif token.type == "table_close":
            table_depth -= 1
        elif token.type == "heading_open":
            level = int(token.tag[1])
            inline = tokens[index + 1] if index + 1 < len(tokens) and tokens[index + 1].type == "inline" else None
            label = _plain_inline(inline.children if inline else None) or "未命名章节"
            base_id = f"{chapter.chapter_id}-{_slugify(label)}"
            heading_id = base_id
            suffix = 2
            while heading_id in heading_ids:
                heading_id = f"{base_id}-{suffix}"
                suffix += 1
            heading_ids.add(heading_id)
            token.attrSet("id", heading_id)
            if level <= toc_depth:
                toc.append({"id": heading_id, "level": level, "label": label})

        if token.type == "math_block":
            register_formula(token, token.map[0] + 1 if token.map else None, False)
        elif token.type == "inline" and token.children:
            line = token.map[0] + 1 if token.map else None
            for child in token.children:
                if child.type == "image":
                    _process_image(child, chapter, collector)
                elif child.type == "link_open":
                    _process_link(child, chapter, collector)
                elif child.type == "math_inline":
                    register_formula(child, line, table_depth > 0)
    return toc


# ---------------------------------------------------------------------------
# Zhihu Markdown serializer


def _escape_markdown(text: str, in_table: bool = False) -> str:
    text = text.replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"([\\`*_\[\]#$])", r"\\\1", text)
    if in_table:
        text = text.replace("|", "\\|").replace("\n", "<br>")
    return text


def _markdown_destination(destination: str, title: str | None = None) -> str:
    if any(char in destination for char in " ()"):
        destination = f"<{destination.replace('<', '%3C').replace('>', '%3E')}>"
    suffix = f' "{title.replace(chr(34), chr(92) + chr(34))}"' if title else ""
    return destination + suffix


def _chinese_number(number: int) -> str:
    digits = "零一二三四五六七八九"
    if number < 10:
        return digits[number]
    if number < 20:
        return "十" + (digits[number % 10] if number % 10 else "")
    if number < 100:
        return digits[number // 10] + "十" + (digits[number % 10] if number % 10 else "")
    return str(number)


class ZhihuSerializer:
    def __init__(self) -> None:
        self.major_heading = 0

    def render(self, chapters: Sequence[Chapter]) -> str:
        parts = [self._render_blocks(chapter.tokens) for chapter in chapters]
        return "\n\n".join(part.strip() for part in parts if part.strip()) + "\n"

    @staticmethod
    def _find_close(tokens: Sequence[Token], start: int, open_type: str, close_type: str) -> int:
        depth = 0
        for index in range(start, len(tokens)):
            if tokens[index].type == open_type:
                depth += 1
            elif tokens[index].type == close_type:
                depth -= 1
                if depth == 0:
                    return index
        raise BuildError(f"内部 Markdown token 不匹配：{open_type}/{close_type}")

    def _heading(self, text: str, level: int) -> str:
        clean = text.strip()
        if level == 1:
            self.major_heading += 1
            if re.match(r"^[一二三四五六七八九十百]+、", clean):
                return clean
            return f"{_chinese_number(self.major_heading)}、{clean}"
        # Keep an existing numeric heading prefix verbatim. Unnumbered headings
        # use Markdown's conventional repeated 1. marker rather than renumbering.
        if re.match(r"^\d+[.)、]\s*", clean):
            return clean
        return f"1. {clean}"

    def _render_inline(self, tokens: Sequence[Token] | None, in_table: bool = False) -> str:
        if not tokens:
            return ""
        output: list[str] = []
        links: list[tuple[str, str | None]] = []
        for token in tokens:
            kind = token.type
            if kind in {"text", "text_special", "html_inline"}:
                output.append(_escape_markdown(token.content, in_table))
            elif kind == "code_inline":
                delimiter = "`" * (max((len(match.group(0)) for match in re.finditer(r"`+", token.content)), default=0) + 1)
                content = token.content.replace("|", "\\|") if in_table else token.content
                output.append(f"{delimiter}{content}{delimiter}")
            elif kind in {"softbreak", "hardbreak"}:
                output.append("<br>" if in_table or kind == "hardbreak" else " ")
            elif kind in {"strong_open", "strong_close"}:
                output.append("**")
            elif kind in {"em_open", "em_close"}:
                output.append("*")
            elif kind in {"s_open", "s_close"}:
                output.append("~~")
            elif kind == "link_open":
                links.append((token.attrGet("href") or "", token.attrGet("title")))
                output.append("[")
            elif kind == "link_close":
                href, title = links.pop() if links else ("", None)
                output.append(f"]({_markdown_destination(href, title)})")
            elif kind == "image":
                alt = _plain_inline(token.children) or token.content
                source = token.meta.get("package_src") or token.attrGet("src") or ""
                output.append(f"![{_escape_markdown(alt, in_table)}]({_markdown_destination(source, token.attrGet('title'))})")
            elif kind == "math_inline":
                readable = _escape_markdown(tex_to_readable(token.content), in_table)
                if token.meta.get("display"):
                    output.append(f"公式：{readable}")
                else:
                    output.append(readable)
        return "".join(output)

    def _render_blocks(self, tokens: Sequence[Token], in_table: bool = False) -> str:
        blocks: list[str] = []
        index = 0
        while index < len(tokens):
            block, index = self._render_block(tokens, index, in_table)
            if block.strip():
                blocks.append(block.strip("\n"))
        return "\n\n".join(blocks)

    def _render_block(self, tokens: Sequence[Token], index: int, in_table: bool) -> tuple[str, int]:
        token = tokens[index]
        kind = token.type
        if kind == "inline":
            return self._render_inline(token.children, in_table), index + 1
        if kind == "heading_open":
            close = self._find_close(tokens, index, "heading_open", "heading_close")
            text = self._render_blocks(tokens[index + 1 : close]).strip()
            return self._heading(text, int(token.tag[1])), close + 1
        if kind == "paragraph_open":
            close = self._find_close(tokens, index, "paragraph_open", "paragraph_close")
            return self._render_blocks(tokens[index + 1 : close], in_table), close + 1
        if kind == "fence":
            longest = max((len(match.group(0)) for match in re.finditer(r"`+", token.content)), default=0)
            fence = "`" * max(3, longest + 1)
            info = token.info.strip()
            return f"{fence}{info}\n{token.content}{fence}", index + 1
        if kind == "code_block":
            return f"```\n{token.content}```", index + 1
        if kind == "math_block":
            return f"公式：{_escape_markdown(tex_to_readable(token.content), in_table)}", index + 1
        if kind == "blockquote_open":
            close = self._find_close(tokens, index, "blockquote_open", "blockquote_close")
            content = self._render_blocks(tokens[index + 1 : close])
            quoted = "\n".join(">" if not line else f"> {line}" for line in content.splitlines())
            return quoted, close + 1
        if kind in {"bullet_list_open", "ordered_list_open"}:
            close_type = "bullet_list_close" if kind == "bullet_list_open" else "ordered_list_close"
            close = self._find_close(tokens, index, kind, close_type)
            return self._render_list(tokens[index:close + 1], ordered=kind == "ordered_list_open"), close + 1
        if kind == "table_open":
            close = self._find_close(tokens, index, "table_open", "table_close")
            return self._render_table(tokens[index:close + 1]), close + 1
        if kind == "hr":
            return "---", index + 1
        if kind in {"html_block", "code_inline"}:
            return _escape_markdown(token.content, in_table), index + 1
        if token.nesting == 1:
            close_type = kind[:-5] + "close"
            close = self._find_close(tokens, index, kind, close_type)
            return self._render_blocks(tokens[index + 1 : close], in_table), close + 1
        return _escape_markdown(token.content, in_table) if token.content else "", index + 1

    def _render_list(self, tokens: Sequence[Token], ordered: bool) -> str:
        open_token = tokens[0]
        start = 1
        if ordered and open_token.attrGet("start"):
            try:
                start = int(open_token.attrGet("start"))
            except (TypeError, ValueError):
                start = 1
        items: list[str] = []
        index = 1
        number = start
        while index < len(tokens) - 1:
            if tokens[index].type != "list_item_open":
                index += 1
                continue
            close = self._find_close(tokens, index, "list_item_open", "list_item_close")
            content = self._render_blocks(tokens[index + 1 : close])
            marker = f"{number}. " if ordered else "- "
            lines = content.splitlines() or [""]
            indented = [marker + lines[0]] + [(" " * len(marker) + line if line else "") for line in lines[1:]]
            items.append("\n".join(indented))
            number += 1
            index = close + 1
        return "\n".join(items)

    def _render_table(self, tokens: Sequence[Token]) -> str:
        rows: list[tuple[bool, list[str], list[str]]] = []
        index = 1
        while index < len(tokens) - 1:
            if tokens[index].type != "tr_open":
                index += 1
                continue
            row_close = self._find_close(tokens, index, "tr_open", "tr_close")
            cells: list[str] = []
            aligns: list[str] = []
            header = False
            cursor = index + 1
            while cursor < row_close:
                if tokens[cursor].type not in {"th_open", "td_open"}:
                    cursor += 1
                    continue
                cell_open = tokens[cursor]
                close_type = "th_close" if cell_open.type == "th_open" else "td_close"
                cell_close = self._find_close(tokens, cursor, cell_open.type, close_type)
                header = header or cell_open.type == "th_open"
                content = self._render_blocks(tokens[cursor + 1 : cell_close], in_table=True)
                cells.append(content.replace("\n\n", "<br>").replace("\n", "<br>").strip())
                style = cell_open.attrGet("style") or ""
                aligns.append("center" if "center" in style else "right" if "right" in style else "left")
                cursor = cell_close + 1
            rows.append((header, cells, aligns))
            index = row_close + 1
        if not rows:
            return ""
        column_count = max(len(row[1]) for row in rows)
        header_cells = rows[0][1] + [""] * (column_count - len(rows[0][1]))
        aligns = rows[0][2] + ["left"] * (column_count - len(rows[0][2]))
        separators = {"left": "---", "center": ":---:", "right": "---:"}
        output = [
            "| " + " | ".join(header_cells) + " |",
            "| " + " | ".join(separators[align] for align in aligns) + " |",
        ]
        for _, cells, _ in rows[1:]:
            padded = cells + [""] * (column_count - len(cells))
            output.append("| " + " | ".join(padded) + " |")
        return "\n".join(output)


# ---------------------------------------------------------------------------
# KaTeX and page templates


def _read_template(name: str) -> str:
    return _read_text(TEMPLATE_DIR / name, f"构建模板 {name}")


def _explicit_katex_directory(config: LoadedConfig) -> Path | None:
    setting = config.data.get("katex")
    if setting is None:
        return None
    if isinstance(setting, str):
        raw = setting
    elif isinstance(setting, dict) and set(setting) <= {"directory"} and isinstance(setting.get("directory"), str):
        raw = setting["directory"]
    else:
        raise BuildError('katex 必须是路径字符串或仅含 directory 字符串的 object，例如 {"directory": "/path/to/katex/dist"}')
    path = Path(os.path.expandvars(raw)).expanduser()
    if not path.is_absolute():
        path = config.article_dir / path
    return path.resolve()


def _katex_candidates(config: LoadedConfig) -> list[Path]:
    candidates = [config.assets_dir / "katex", config.assets_dir / "katex" / "dist"]
    for variable in ("KATEX_DIST", "KATEX_HOME"):
        value = os.environ.get(variable)
        if value:
            path = Path(value).expanduser()
            candidates.append(path)
            if variable == "KATEX_HOME":
                candidates.append(path / "dist")
    candidates.extend(
        [
            config.repo_root / "node_modules" / "katex" / "dist",
            config.article_dir / "node_modules" / "katex" / "dist",
        ]
    )
    home = Path.home()
    candidates.extend(sorted(home.glob(".vscode-server/bin/*/node_modules/katex/dist"), reverse=True))
    candidates.extend(sorted(home.glob(".vscode-server/cli/servers/*/node_modules/katex/dist"), reverse=True))
    unique: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in unique:
            unique.append(resolved)
    return unique


def _embed_katex_fonts(css: str, css_path: Path, distribution: Path) -> str:
    pattern = re.compile(r"url\(\s*(?:\"([^\"]+)\"|'([^']+)'|([^)'\"\s]+))\s*\)")
    missing: list[str] = []

    def replace(match: re.Match[str]) -> str:
        raw_url = next(group for group in match.groups() if group is not None)
        if raw_url.startswith("data:"):
            return match.group(0)
        split = urllib.parse.urlsplit(raw_url)
        if split.scheme or split.netloc or raw_url.startswith("//"):
            raise BuildError(f"KaTeX CSS 含外部资源，无法离线打包：{raw_url}（{css_path}）")
        font_path = (css_path.parent / urllib.parse.unquote(split.path)).resolve()
        if not _is_within(font_path, distribution):
            raise BuildError(f"KaTeX 字体路径越出发行目录：{raw_url}（{css_path}）")
        if not font_path.is_file():
            missing.append(str(font_path))
            return match.group(0)
        media_types = {".woff2": "font/woff2", ".woff": "font/woff", ".ttf": "font/ttf", ".otf": "font/otf", ".svg": "image/svg+xml"}
        media_type = media_types.get(font_path.suffix.lower())
        if media_type is None:
            raise BuildError(f"KaTeX CSS 引用了不支持的资源类型：{font_path}")
        encoded = base64.b64encode(_read_bytes(font_path, "KaTeX 字体")).decode("ascii")
        return f'url("data:{media_type};base64,{encoded}")'

    embedded = pattern.sub(replace, css)
    if missing:
        files = "\n  ".join(missing)
        raise BuildError(f"KaTeX CSS 引用的字体缺失：\n  {files}\n{KATEX_HELP}")
    return embedded


def _load_katex(config: LoadedConfig) -> tuple[str, str, Path, str | None]:
    explicit = _explicit_katex_directory(config)
    candidates = [explicit] if explicit is not None else _katex_candidates(config)
    attempts: list[str] = []
    for directory in candidates:
        assert directory is not None
        js_path = directory / "katex.min.js"
        css_path = directory / "katex.min.css"
        if not js_path.is_file() or not css_path.is_file():
            attempts.append(f"{directory}（缺 katex.min.js 或 katex.min.css）")
            if explicit is not None:
                break
            continue
        try:
            js = _read_text(js_path, "KaTeX JavaScript").strip()
            css = _read_text(css_path, "KaTeX CSS").strip()
            if not js or not css:
                raise BuildError(f"KaTeX JS/CSS 不能为空：{directory}")
            css = _embed_katex_fonts(css, css_path, directory.resolve())
        except BuildError:
            if explicit is not None:
                raise
            raise
        version: str | None = None
        package_path = directory.parent / "package.json"
        if package_path.is_file():
            try:
                package_data = json.loads(package_path.read_text(encoding="utf-8"))
                if isinstance(package_data.get("version"), str):
                    version = package_data["version"]
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                version = None
        # Keep script data from prematurely closing the host script element.
        js = re.sub(r"</script", r"<\/script", js, flags=re.IGNORECASE)
        js = re.sub(r"//# sourceMappingURL=.*", "", js)
        css = re.sub(r"/\*# sourceMappingURL=.*?\*/", "", css, flags=re.DOTALL)
        return css, js, directory, version

    detail = "\n  ".join(attempts) if attempts else "未找到候选目录"
    raise BuildError(f"文档含 LaTeX 公式，但 KaTeX 资源不可用：\n  {detail}\n{KATEX_HELP}")


def _render_toc(entries: Sequence[dict[str, Any]]) -> str:
    if not entries:
        return '<p class="toc-empty">本文没有章节标题。</p>'
    items = []
    for entry in entries:
        level = min(max(int(entry["level"]), 1), 6)
        items.append(
            f'<li class="toc-level-{level}"><a href="#{html.escape(entry["id"], quote=True)}">'
            f'{html.escape(entry["label"])}</a></li>'
        )
    return '<ul class="toc-list">' + "".join(items) + "</ul>"


def _safe_inline_js(script: str) -> str:
    return re.sub(r"</script", r"<\/script", script, flags=re.IGNORECASE)


# ---------------------------------------------------------------------------
# Build, manifest, ZIP, and CLI


def build_article(config_path: str | os.PathLike[str]) -> BuildResult:
    """Validate all inputs and render both formats entirely in memory."""

    _require_dependencies()
    config = _load_config(Path(config_path))
    collector = ResourceCollector(config)

    for position, attachment in enumerate(config.data.get("attachments", []), start=1):
        if isinstance(attachment, str):
            raw_path, name = attachment, None
        elif isinstance(attachment, dict) and isinstance(attachment.get("path"), str):
            if set(attachment) - {"path", "name"} or ("name" in attachment and not isinstance(attachment["name"], str)):
                raise BuildError(f"attachments[{position - 1}] 只能包含 path 和可选字符串 name")
            raw_path, name = attachment["path"], attachment.get("name")
        else:
            raise BuildError(f"attachments[{position - 1}] 必须是路径字符串或含 path 的 object")
        collector.add(raw_path, config.article_dir, "attachment", name)

    md = _make_markdown_parser()
    chapters: list[Chapter] = []
    formulas: list[dict[str, Any]] = []
    toc: list[dict[str, Any]] = []
    heading_ids: set[str] = set()
    chapter_ids: set[str] = set()
    source_documents: list[dict[str, Any]] = [
        {
            "path": _relative_label(config.path, config.repo_root),
            "role": "config",
            "bytes": len(config.raw),
            "sha256": _sha256(config.raw),
        }
    ]

    for position, entry in enumerate(config.data["chapters"], start=1):
        custom_id: str | None = None
        if isinstance(entry, str):
            raw_path = entry
        elif isinstance(entry, dict) and isinstance(entry.get("path"), str):
            if set(entry) - {"path", "id"}:
                raise BuildError(f"chapters[{position - 1}] 只能包含 path 和可选 id")
            raw_path = entry["path"]
            if "id" in entry:
                if not isinstance(entry["id"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", entry["id"]):
                    raise BuildError(f"chapters[{position - 1}].id 必须匹配 [A-Za-z0-9][A-Za-z0-9_-]*")
                custom_id = entry["id"]
        else:
            raise BuildError(f"chapters[{position - 1}] 必须是路径字符串或含 path 的 object")
        path = _chapter_path(config, raw_path)
        raw = _read_bytes(path, "Markdown 章节")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise BuildError(f"Markdown 章节必须是 UTF-8：{path}（{exc}）") from exc
        chapter_id = custom_id or f"chapter-{position:02d}-{_slugify(path.stem)}"
        if chapter_id in chapter_ids:
            raise BuildError(f"章节 id 重复：{chapter_id}")
        chapter_ids.add(chapter_id)
        chapter = Chapter(path=path, chapter_id=chapter_id, tokens=[])
        try:
            chapter.tokens = md.parse(text)
        except BuildError as exc:
            raise BuildError(f"解析章节失败 {path}：{exc}") from exc
        source_documents.append(
            {
                "path": _relative_label(path, config.repo_root),
                "role": "chapter",
                "bytes": len(raw),
                "sha256": _sha256(raw),
            }
        )
        toc.extend(
            _process_chapter_tokens(
                chapter,
                collector,
                formulas,
                heading_ids,
                int(config.data.get("toc_depth", 3)),
            )
        )
        chapter.html = md.renderer.render(chapter.tokens, md.options, {"chapter": str(path)})
        chapters.append(chapter)

    katex_css = ""
    katex_js = ""
    katex_directory: Path | None = None
    katex_version: str | None = None
    if formulas:
        katex_css, katex_js, katex_directory, katex_version = _load_katex(config)

    pygments_css = HtmlFormatter(style="default").get_style_defs(".codehilite")
    inline_css = "\n".join([_read_template("style.css"), pygments_css, katex_css])
    boot_js = _safe_inline_js(_read_template("katex-boot.js")) if formulas else ""
    page = Template(_read_template("page.html")).substitute(
        title=html.escape(config.data["title"]),
        description=html.escape(config.data.get("description", ""), quote=True),
        inline_css=inline_css,
        toc=_render_toc(toc),
        content="\n".join(
            f'<section id="{html.escape(chapter.chapter_id, quote=True)}" class="chapter">\n{chapter.html}</section>'
            for chapter in chapters
        ),
        katex_js=katex_js,
        katex_boot_js=boot_js,
        katex_version=html.escape(katex_version or "not-used", quote=True),
    )
    zhihu = ZhihuSerializer().render(chapters)
    return BuildResult(
        title=config.data["title"],
        html=page,
        zhihu=zhihu,
        formulas=formulas,
        resources=collector.resources,
        source_documents=source_documents,
        katex_directory=katex_directory,
        katex_version=katex_version,
    )


def create_manifest(result: BuildResult, generated: dict[str, bytes]) -> bytes:
    files: list[dict[str, Any]] = []
    occupied: dict[str, str] = {}
    for name, data in generated.items():
        archive_path = _clean_archive_path(name, "生成文件名")
        occupied[archive_path] = "generated"
        files.append({"archive_path": archive_path, "kind": "generated", "bytes": len(data), "sha256": _sha256(data)})
    for resource in result.resources:
        if resource.archive_path in occupied:
            raise BuildError(f"ZIP 内生成文件与资源路径冲突：{resource.archive_path}")
        occupied[resource.archive_path] = resource.source_label
        files.append(
            {
                "archive_path": resource.archive_path,
                "kind": resource.kind,
                "source": resource.source_label,
                "media_type": resource.media_type,
                "bytes": len(resource.data),
                "sha256": resource.sha256,
            }
        )
    files.sort(key=lambda item: item["archive_path"])
    payload = {
        "schema_version": SCHEMA_VERSION,
        "title": result.title,
        "source_documents": result.source_documents,
        "files": files,
        "manifest_note": "sha256 对 ZIP 中对应文件字节计算；manifest.json 自身不列入哈希清单。",
    }
    return (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def create_zip(result: BuildResult, generated: dict[str, bytes], manifest: bytes) -> bytes:
    entries: dict[str, bytes] = dict(generated)
    for resource in result.resources:
        if resource.archive_path in entries:
            raise BuildError(f"ZIP 路径冲突：{resource.archive_path}")
        entries[resource.archive_path] = resource.data
    if "manifest.json" in entries:
        raise BuildError("生成文件或资源不能使用保留路径 manifest.json")
    entries["manifest.json"] = manifest
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(entries):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, entries[name])
    return buffer.getvalue()


def _atomic_write(path: Path, data: bytes) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
    except OSError as exc:
        raise BuildError(f"无法写入输出 {path}：{exc}") from exc


def _default_zhihu_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}.zhihu.md")


def _default_formula_log_path(zhihu: Path) -> Path:
    return zhihu.with_suffix(".formulas.json")


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="从 article/index.json 构建离线 HTML、知乎 Markdown、资源 manifest 和 ZIP")
    parser.add_argument("--config", required=True, help="article/index.json 路径")
    parser.add_argument("--output", help="单文件 HTML 输出路径；--check-only 时省略")
    parser.add_argument("--zhihu-output", help="知乎 Markdown 输出路径，默认 <HTML 名>.zhihu.md")
    parser.add_argument("--formula-log", help="公式转换记录 JSON，默认 <知乎文件名>.formulas.json")
    parser.add_argument("--manifest-output", help="额外写出 ZIP 同款 manifest.json")
    parser.add_argument("--zip", dest="zip_output", help="发布资源 ZIP 输出路径")
    parser.add_argument("--check-only", action="store_true", help="完整预检和内存渲染，不写任何文件")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _argument_parser()
    args = parser.parse_args(argv)
    if not args.check_only and not args.output:
        parser.error("非 --check-only 构建必须提供 --output")
    try:
        result = build_article(args.config)
        if args.check_only:
            katex = str(result.katex_directory) if result.katex_directory else "未使用"
            print(
                f"检查通过：{len(result.source_documents) - 1} 个章节，{result.image_count} 张图片，"
                f"{result.attachment_count} 个附件，{len(result.formulas)} 个公式；KaTeX：{katex}；未写入文件。"
            )
            return 0

        output = Path(args.output).expanduser().resolve()
        zhihu = Path(args.zhihu_output).expanduser().resolve() if args.zhihu_output else _default_zhihu_path(output)
        formula_log = Path(args.formula_log).expanduser().resolve() if args.formula_log else _default_formula_log_path(zhihu)
        zip_output = Path(args.zip_output).expanduser().resolve() if args.zip_output else None
        manifest_output = Path(args.manifest_output).expanduser().resolve() if args.manifest_output else None
        destinations = [output, zhihu, formula_log] + ([zip_output] if zip_output else []) + ([manifest_output] if manifest_output else [])
        if len(set(destinations)) != len(destinations):
            raise BuildError("HTML、知乎 Markdown、公式记录、manifest 和 ZIP 输出路径必须互不相同")

        generated = {
            output.name: result.html.encode("utf-8"),
            zhihu.name: result.zhihu.encode("utf-8"),
            formula_log.name: result.formula_log_bytes(),
        }
        manifest = create_manifest(result, generated)
        zip_data = create_zip(result, generated, manifest) if zip_output else None
        writes: list[tuple[Path, bytes]] = [(output, generated[output.name]), (zhihu, generated[zhihu.name]), (formula_log, generated[formula_log.name])]
        if manifest_output:
            writes.append((manifest_output, manifest))
        if zip_output and zip_data is not None:
            writes.append((zip_output, zip_data))
        for path, data in writes:
            _atomic_write(path, data)
            print(f"已生成：{path}")
        if result.katex_directory:
            version = f" {result.katex_version}" if result.katex_version else ""
            print(f"KaTeX{version}：{result.katex_directory}")
        return 0
    except BuildError as exc:
        print(f"构建错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
