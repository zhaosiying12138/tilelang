"""Synthetic-only tests for article/build_html.py.

No test reads or writes the article's real chapters, evidence, or index.json.
All documents, images, attachments, and KaTeX fixtures live in TemporaryDirectory.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


ARTICLE_DIR = Path(__file__).resolve().parents[1]
SCRIPT = ARTICLE_DIR / "build_html.py"
SPEC = importlib.util.spec_from_file_location("article_build_html", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


class SyntheticArticleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="article-builder-test-")
        self.root = Path(self.temporary.name)
        self.article = self.root / "article"
        self.assets = self.article / "assets"
        self.evidence = self.root / "evidence" / "G1"
        self.assets.mkdir(parents=True)
        self.evidence.mkdir(parents=True)
        self.katex = self.root / "katex" / "dist"
        (self.katex / "fonts").mkdir(parents=True)
        (self.katex / "katex.min.js").write_text(
            "window.katex={render:function(source,element){element.textContent=source;}};",
            encoding="utf-8",
        )
        (self.katex / "katex.min.css").write_text(
            "@font-face{font-family:Synthetic;src:url(fonts/synthetic.woff2) format('woff2');}"
            ".katex{font-family:Synthetic;}",
            encoding="utf-8",
        )
        (self.katex / "fonts" / "synthetic.woff2").write_bytes(b"synthetic-woff2")
        (self.katex.parent / "package.json").write_text('{"version":"0.0.0-synthetic"}', encoding="utf-8")
        self.config_path = self.article / "index.json"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_chapter(self, text: str, name: str = "chapter.md") -> Path:
        path = self.article / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def write_config(
        self,
        chapters: list[object] | None = None,
        attachments: list[object] | None = None,
        katex: object = None,
    ) -> Path:
        if chapters is None:
            chapters = [{"path": "chapter.md", "id": "demo"}]
        if katex is None:
            katex = {"directory": str(self.katex)}
        payload = {
            "title": "合成测试文章",
            "description": "仅用于临时目录测试",
            "toc_depth": 3,
            "chapters": chapters,
            "attachments": attachments or [],
            "katex": katex,
        }
        self.config_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return self.config_path

    def test_tex_fallback_handles_nested_groups_ceils_and_control_spaces(self) -> None:
        cases = {
            r"\min\ makespan": "min makespan",
            r"\lceil \frac{ops}{throughput_{dtype}} \rceil": "⌈ ((ops)/(throughput_dtype)) ⌉",
            r"\frac{a}{\frac{b}{c_{idx}}}": "((a)/(((b)/(c_idx))))",
            r"\sqrt{\frac{a}{b_{idx}}}": "sqrt(((a)/(b_idx)))",
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                self.assertEqual(builder.tex_to_readable(source), expected)

    def test_manifest_directory_is_allowed_evidence_root(self) -> None:
        self.write_chapter("[硬件清单](../manifest/hardware.json)")
        manifest = self.root / "manifest"
        manifest.mkdir()
        (manifest / "hardware.json").write_text('{"sm_count": 82}', encoding="utf-8")
        self.write_config()
        result = builder.build_article(self.config_path)
        self.assertTrue(any(resource.archive_path == "manifest/hardware.json" for resource in result.resources))

    def test_html_zhihu_math_images_manifest_and_zip(self) -> None:
        chapter = self.write_chapter(
            """# 总览

数值 12.340 与 $x_1 \\geq 0$。行内代码 `$code$` 不变。

## 方法

```python
def answer():
    return 42
```

```text
$not_math$
```

$$
\\frac{a}{b} \\leq 2
$$

| 名称 | 关系 |
| --- | --- |
| 合成 | $x \\cdot y$ |

### 2. 已有编号

![资源图片](assets/pixel.png)

![证据图片](../evidence/G1/evidence.png)

[原始日志](../evidence/G1/run.log)
"""
        )
        (self.assets / "pixel.png").write_bytes(PNG_BYTES)
        (self.evidence / "evidence.png").write_bytes(PNG_BYTES + b"evidence")
        (self.evidence / "data.csv").write_text("name,value\nsynthetic,12.340\n", encoding="utf-8")
        (self.evidence / "meta.json").write_text('{"synthetic": true}', encoding="utf-8")
        (self.evidence / "run.log").write_text("synthetic log\n", encoding="utf-8")
        self.write_config(
            attachments=[
                {"path": "../evidence/G1/data.csv", "name": "results/data.csv"},
                "evidence/G1/meta.json",
            ]
        )

        result = builder.build_article(self.config_path)

        self.assertIn('<nav id="table-of-contents"', result.html)
        self.assertIn('href="#demo-', result.html)
        self.assertIn("<table>", result.html)
        self.assertIn('class="math math-inline"', result.html)
        self.assertIn('class="math math-display"', result.html)
        self.assertIn('data-display="true"', result.html)
        self.assertIn('<span class="k">def</span>', result.html)
        self.assertIn("window.katex", result.html)
        self.assertIn("data:font/woff2;base64,", result.html)
        self.assertNotIn("url(fonts/", result.html)
        self.assertNotIn('<script src="', result.html)
        self.assertNotIn('<link rel="stylesheet"', result.html)
        self.assertIn("12.340", result.html)
        self.assertIn("$not_math$", result.html)
        self.assertIn("$code$", result.html)
        self.assertEqual(result.html.count('class="math '), 3)
        for resource in result.resources:
            if resource.kind == "image":
                self.assertIn(resource.data_uri, result.html)

        self.assertIn("一、总览", result.zhihu)
        self.assertIn("1. 方法", result.zhihu)
        self.assertIn("2. 已有编号", result.zhihu)
        self.assertNotIn("1. 2. 已有编号", result.zhihu)
        self.assertNotIn("###", result.zhihu)
        self.assertIn("12.340", result.zhihu)
        self.assertIn("return 42", result.zhihu)
        self.assertIn("x\\_1 ≥ 0", result.zhihu)
        self.assertIn("((a)/(b)) ≤ 2", result.zhihu)
        self.assertIn("x · y", result.zhihu)
        table_lines = [line for line in result.zhihu.splitlines() if line.startswith("|")]
        self.assertGreaterEqual(len(table_lines), 3)
        for line in table_lines:
            self.assertNotIn("$", line)
            self.assertNotIn("\\cdot", line)
            self.assertNotIn("\\geq", line)
        self.assertIn("](evidence/G1/run.log)", result.zhihu)
        self.assertIn("](assets/pixel.png)", result.zhihu)
        self.assertIn("](evidence/G1/evidence.png)", result.zhihu)

        self.assertEqual(len(result.formulas), 3)
        self.assertEqual([formula["kind"] for formula in result.formulas], ["inline", "display", "inline"])
        table_formulas = [formula for formula in result.formulas if formula["in_table"]]
        self.assertEqual(len(table_formulas), 1)
        self.assertEqual(table_formulas[0]["source"], "x \\cdot y")
        self.assertEqual(table_formulas[0]["text"], "x · y")
        log = json.loads(result.formula_log_bytes())
        self.assertEqual(log["replacement"], "readable-plain-text")
        self.assertEqual(len(log["formulas"]), 3)

        generated = {
            "article.html": result.html.encode("utf-8"),
            "article.zhihu.md": result.zhihu.encode("utf-8"),
            "article.formulas.json": result.formula_log_bytes(),
        }
        manifest_bytes = builder.create_manifest(result, generated)
        manifest = json.loads(manifest_bytes)
        archive_bytes = builder.create_zip(result, generated, manifest_bytes)
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            self.assertEqual(archive.read("manifest.json"), manifest_bytes)
            names = set(archive.namelist())
            self.assertIn("assets/pixel.png", names)
            self.assertIn("evidence/G1/evidence.png", names)
            self.assertIn("attachments/results/data.csv", names)
            self.assertIn("evidence/G1/meta.json", names)
            self.assertIn("evidence/G1/run.log", names)
            for entry in manifest["files"]:
                payload = archive.read(entry["archive_path"])
                self.assertEqual(hashlib.sha256(payload).hexdigest(), entry["sha256"])
                self.assertEqual(len(payload), entry["bytes"])
        chapter_entry = next(item for item in manifest["source_documents"] if item["role"] == "chapter")
        self.assertEqual(chapter_entry["sha256"], hashlib.sha256(chapter.read_bytes()).hexdigest())

    def test_package_reference_collects_original_png_without_embedding_it(self) -> None:
        self.write_chapter(
            """# 截图证据包

![裁图](../evidence/G1/cropped.png)

[全屏原图][package:original]；[未声明 PNG 链接](../evidence/G1/unlisted.png)。

[package:original]: ../evidence/G1/original.png
"""
        )
        cropped_bytes = PNG_BYTES + b"cropped"
        original_bytes = PNG_BYTES + b"original"
        (self.evidence / "cropped.png").write_bytes(cropped_bytes)
        (self.evidence / "original.png").write_bytes(original_bytes)
        (self.evidence / "unlisted.png").write_bytes(PNG_BYTES + b"unlisted")
        self.write_config()

        result = builder.build_article(self.config_path)
        resources = {resource.archive_path: resource for resource in result.resources}

        self.assertEqual(set(resources), {"evidence/G1/cropped.png", "evidence/G1/original.png"})
        self.assertEqual(resources["evidence/G1/cropped.png"].kind, "image")
        self.assertEqual(resources["evidence/G1/original.png"].kind, "attachment")
        self.assertIn(resources["evidence/G1/cropped.png"].data_uri, result.html)
        self.assertNotIn(resources["evidence/G1/original.png"].data_uri, result.html)
        self.assertIn('<a href="evidence/G1/original.png">全屏原图</a>', result.html)
        self.assertIn('href="../evidence/G1/unlisted.png"', result.html)
        self.assertNotIn("package:original", result.html)
        self.assertIn("](evidence/G1/original.png)", result.zhihu)
        self.assertIn("](../evidence/G1/unlisted.png)", result.zhihu)
        self.assertNotIn("package:original", result.zhihu)

        manifest_bytes = builder.create_manifest(result, {})
        manifest = json.loads(manifest_bytes)
        original_entry = next(
            entry for entry in manifest["files"] if entry["archive_path"] == "evidence/G1/original.png"
        )
        self.assertEqual(original_entry["kind"], "attachment")
        archive_bytes = builder.create_zip(result, {}, manifest_bytes)
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            self.assertEqual(archive.read("evidence/G1/cropped.png"), cropped_bytes)
            self.assertEqual(archive.read("evidence/G1/original.png"), original_bytes)
            self.assertNotIn("evidence/G1/unlisted.png", archive.namelist())

    def test_missing_package_reference_image_is_an_error(self) -> None:
        self.write_chapter("# 缺原图\n\n[package:original]: ../evidence/G1/missing.png\n")
        self.write_config()
        with self.assertRaisesRegex(builder.BuildError, "附件源文件不存在"):
            builder.build_article(self.config_path)

    def test_check_only_cli_writes_nothing_and_missing_chapter_fails(self) -> None:
        self.write_chapter("# 预检\n\n$x$\n")
        self.write_config()
        output = self.root / "out" / "article.html"
        zip_output = self.root / "out" / "article.zip"
        command = [
            sys.executable,
            str(SCRIPT),
            "--config",
            str(self.config_path),
            "--output",
            str(output),
            "--zip",
            str(zip_output),
            "--check-only",
        ]
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        checked = subprocess.run(command, text=True, capture_output=True, env=environment, check=False)
        self.assertEqual(checked.returncode, 0, checked.stderr)
        self.assertIn("检查通过", checked.stdout)
        self.assertIn("未写入文件", checked.stdout)
        self.assertFalse(output.exists())
        self.assertFalse(zip_output.exists())
        self.assertFalse(output.parent.exists())

        (self.article / "chapter.md").unlink()
        missing = subprocess.run(command, text=True, capture_output=True, env=environment, check=False)
        self.assertEqual(missing.returncode, 1)
        self.assertIn("构建错误", missing.stderr)
        self.assertIn("Markdown 章节不存在", missing.stderr)
        self.assertFalse(output.exists())

    def test_full_cli_writes_html_zhihu_formula_log_manifest_and_zip(self) -> None:
        self.write_chapter("# 命令行\n\n$x + 1$\n")
        self.write_config()
        output = self.root / "dist" / "article.html"
        zhihu = self.root / "dist" / "zhihu.md"
        formulas = self.root / "dist" / "formulas.json"
        manifest = self.root / "dist" / "manifest.json"
        zip_output = self.root / "dist" / "article.zip"
        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--config",
                str(self.config_path),
                "--output",
                str(output),
                "--zhihu-output",
                str(zhihu),
                "--formula-log",
                str(formulas),
                "--manifest-output",
                str(manifest),
                "--zip",
                str(zip_output),
            ],
            text=True,
            capture_output=True,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        for path in (output, zhihu, formulas, manifest, zip_output):
            self.assertTrue(path.is_file(), path)
        self.assertIn("KaTeX 0.0.0-synthetic", completed.stdout)
        with zipfile.ZipFile(zip_output) as archive:
            self.assertEqual(archive.read("manifest.json"), manifest.read_bytes())
            self.assertEqual(archive.read("article.html"), output.read_bytes())
            self.assertEqual(archive.read("zhihu.md"), zhihu.read_bytes())
            self.assertEqual(archive.read("formulas.json"), formulas.read_bytes())

    def test_missing_image_is_an_error(self) -> None:
        self.write_chapter("# 缺图\n\n![不存在](assets/missing.png)\n")
        self.write_config()
        with self.assertRaisesRegex(builder.BuildError, "图片源文件不存在"):
            builder.build_article(self.config_path)

    def test_missing_declared_attachment_is_an_error(self) -> None:
        self.write_chapter("# 缺附件\n")
        self.write_config(attachments=["evidence/G1/missing.csv"])
        with self.assertRaisesRegex(builder.BuildError, "附件源文件不存在"):
            builder.build_article(self.config_path)

    def test_missing_katex_distribution_is_an_error(self) -> None:
        self.write_chapter("# 缺公式资源\n\n$x$\n")
        self.write_config(katex={"directory": str(self.root / "no-katex")})
        with self.assertRaisesRegex(builder.BuildError, "KaTeX 资源不可用"):
            builder.build_article(self.config_path)

    def test_missing_katex_font_is_an_error(self) -> None:
        self.write_chapter("# 缺字体\n\n$x$\n")
        (self.katex / "katex.min.css").write_text(
            "@font-face{src:url(fonts/missing.woff2)}", encoding="utf-8"
        )
        self.write_config()
        with self.assertRaisesRegex(builder.BuildError, "字体缺失"):
            builder.build_article(self.config_path)

    def test_remote_image_is_rejected_for_self_contained_html(self) -> None:
        self.write_chapter("# 远程图片\n\n![外部](https://example.invalid/pixel.png)\n")
        self.write_config()
        with self.assertRaisesRegex(builder.BuildError, "不能引用远程图片"):
            builder.build_article(self.config_path)


if __name__ == "__main__":
    unittest.main()
