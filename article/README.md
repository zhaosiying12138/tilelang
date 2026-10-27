# 文章构建工具

`article/build_html.py` 从一份 `article/index.json` 和同一组 Markdown 章节生成：

1. **单文件中文 HTML**：目录、标题锚点、代码高亮、表格、KaTeX 公式、base64 本地图片、内嵌 CSS/JavaScript/KaTeX 字体；运行时不依赖 CDN 或网络。
2. **知乎 Markdown**：仅使用“一、”和“1.”两种标题形式；表格内不保留 LaTeX；公式自动转成可读文本。
3. **公式转换记录 JSON**：逐条保留原始 TeX、转换文本、章节、行号、公式类型及是否位于表格中。
4. **SHA256 manifest 与 ZIP**：收集文章图片及 CSV/JSON/log 等证据附件，同时打包 HTML、知乎 Markdown 和公式记录；也可用非显示声明把全屏原图等 PNG 作为附件收入 ZIP，而不把原图重复 base64 嵌入 HTML。

本工具只转换已有内容，**不会生成、补齐或修改性能数字**。原始 Markdown、机制文档和 evidence 不会被改写。正文配置由 `article/index.json` 驱动；下面的配置仅是 schema 示例。

## 1. 依赖

需要 Python 3.10+，以及纯 Python 包：

- `markdown-it-py`：Markdown 解析和表格支持；
- `Pygments`：静态代码高亮，高亮 CSS 也内嵌到 HTML。

如果缺包，脚本会退出并显示安装说明。不要安装到全局环境；可使用临时虚拟环境：

```bash
python3 -m venv /tmp/article-build-venv
/tmp/article-build-venv/bin/python -m pip install markdown-it-py Pygments
/tmp/article-build-venv/bin/python article/build_html.py --help
```

构建器不联网下载 Python 包、KaTeX、图片或字体。

## 2. `article/index.json` 配置

最小配置只需要 `title` 和 `chapters`：

```json
{
  "title": "CUDA Z3 调度机制",
  "description": "可选的 HTML 摘要",
  "toc_depth": 3,
  "chapters": [
    "mechanisms/01-z3-scheduler.md",
    {"path": "mechanisms/02-latency-model.md", "id": "latency-model"}
  ],
  "attachments": [
    "../evidence/G1/solver-tests.log",
    {"path": "evidence/G4/guided-pruning-funnel.csv", "name": "results/guided-pruning-funnel.csv"}
  ],
  "katex": {
    "directory": "/absolute/path/to/katex/dist"
  }
}
```

### 字段

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `title` | 是 | 非空字符串，用于 HTML 标题和 manifest。 |
| `description` | 否 | HTML 摘要和正文顶部说明。 |
| `toc_depth` | 否 | 目录收录的标题深度，1–6，默认 3。 |
| `chapters` | 是 | 非空有序数组；字符串是路径，object 可含 `path` 和 `id`。 |
| `attachments` | 否 | 额外附件数组；字符串是路径，object 可含 `path` 和 `name`。除文本证据外，也可显式加入不嵌入 HTML 的图片附件。 |
| `katex` | 含公式时建议配置 | 路径字符串，或只含 `directory` 的 object；省略时自动发现本机资源。 |

- 章节路径相对于 `article/index.json`，必须留在 `article` 内，并按数组顺序拼接。
- `id` 用于章节和标题锚点，须匹配 `[A-Za-z0-9][A-Za-z0-9_-]*`；省略时根据序号和文件名生成。重复 ID 会报错。
- `attachments[].name` 是 ZIP 内 `attachments/` 下的可选 POSIX 相对路径；不允许绝对路径、`..` 或反斜杠。

### 图片与附件路径解析

内容资源只允许来自 `article/assets` 或仓库的 `evidence`、`manifest`，以防误打包任意文件：

- Markdown 图片先相对于当前章节解析，再相对于 `article` 解析。
- 配置中的附件相对于 `article` 解析。
- 以 `evidence/...` 或 `article/...` 开头的资源路径相对于仓库根目录。
- `../evidence/...` 这类从 `article` 出发的相对路径也可使用。
- Markdown 中链接的本地 `.csv`、`.json`、`.jsonl`、`.log`、`.txt`、`.tsv` 会自动作为附件收集；配置 `attachments` 可加入正文未直接链接的文件。
- 支持 PNG、JPEG、GIF、SVG、WebP、BMP、ICO、AVIF 图片。正文中的 Markdown 图片使用 base64 data URI 嵌入 HTML；知乎 Markdown 使用 ZIP 内相对路径。
- 普通的 PNG/JPEG 等**文字链接**不会仅凭链接就自动收集为附件，也不会把目标图片嵌入 HTML。需要随 ZIP 交付全屏原图时，必须使用下面的 `package:` 声明或配置 `attachments`。
- 远程图片会报错，因为远程图片不符合单文件离线目标。普通 HTTP/HTTPS 文字链接可保留。

### 非显示附件声明（包括全屏原图）

章节可使用 CommonMark 链接引用定义声明一个不显示的附件资源，标签以 `package:` 开头：

```markdown
[全屏原图][package:stock-original]

[package:stock-original]: ../../evidence/G7/screenshots/stock-original.png
```

引用定义本身不会出现在 HTML 或知乎正文中。上例会显示“全屏原图”链接，并把目标 PNG 作为 `kind=attachment` 收进 ZIP 和 manifest，链接改为 ZIP 内路径；PNG 字节**不会**重复 base64 嵌入 HTML。若不需要可见链接，只保留 `[package:...]: path` 定义也会收集附件。裁图仍可用普通 `![裁图](...)` 显示并嵌入，原图则通过该声明随包交付。

`package:` 声明支持上述文本附件及图片格式，路径规则、缺失文件检查和远程路径限制与其他资源一致。配置 `attachments` 是等效的全局声明方式；不要仅写一个普通 `.png` 文字链接就声称原图已经入包。

**任何章节、已声明附件、`package:` 资源、被引用图片、自动收集附件或模板缺失都会终止构建。** 所有输入先在内存中验证和渲染，不会因缺文件而生成占位内容或留下正常外观的半成品。

## 3. KaTeX 资源

公式语法为 `$inline$` 和 `$$display$$`。 fenced code 与行内代码中的美元符号不会被当成公式。

含公式时必须具有完整的 KaTeX 发行目录，其中至少包括：

```text
katex.min.js
katex.min.css
fonts/...
```

可通过 `katex.directory` 显式指定。省略时依次检查：

1. `article/assets/katex` 和 `article/assets/katex/dist`；
2. `KATEX_DIST`，以及 `KATEX_HOME`/`KATEX_HOME/dist`；
3. 仓库或 `article` 下的 `node_modules/katex/dist`；
4. 本机 VS Code Server 的 `node_modules/katex/dist`。

CSS 中引用的 WOFF2/WOFF/TTF/OTF/SVG 字体会逐个读取并转成 base64。JS、CSS 或任一引用字体缺失时，构建会给出具体路径和配置建议，**不会静默退化成无渲染公式**。没有公式的文档不强制要求 KaTeX。

如需在临时目录准备发行版，可手工执行：

```bash
npm install --prefix /tmp/article-katex katex
```

然后把 `katex.directory` 设为 `/tmp/article-katex/node_modules/katex/dist`。这一步不是构建器自动行为，也不应安装到全局环境。

## 4. CLI

### 完整预检，不写文件

```bash
python3 article/build_html.py \
  --config article/index.json \
  --check-only
```

`--check-only` 会解析全部章节、检查资源、渲染 HTML/知乎文本并验证 KaTeX 和字体，但不创建目录或输出文件；此时可省略 `--output`。即使同时给出 `--output` 或 `--zip`，也只检查。

### 构建 HTML、知乎 Markdown 和 ZIP

```bash
python3 article/build_html.py \
  --config article/index.json \
  --output /tmp/article-output/article.html \
  --zip /tmp/article-output/article.zip \
  --manifest-output /tmp/article-output/manifest.json
```

默认同时生成：

- `/tmp/article-output/article.html`；
- `/tmp/article-output/article.zhihu.md`；
- `/tmp/article-output/article.zhihu.formulas.json`；
- 可选的 `manifest.json` 和 ZIP。

可用 `--zhihu-output` 和 `--formula-log` 分别覆盖后两个路径。所有输出路径必须互不相同。文件采用同目录临时文件加原子替换写入；ZIP 在写盘前也已完成内存组装。

### ZIP 内容

```text
article.html
article.zhihu.md
article.zhihu.formulas.json
assets/...                 # 来自 article/assets 的默认资源路径
evidence/...               # 来自 evidence 的默认资源路径
manifest/...               # 来自 manifest 的默认资源路径
attachments/...            # 使用 attachments[].name 的附件
manifest.json
```

知乎 Markdown 中的图片和附件链接对应上述 ZIP 路径，因此发布准备时应保留整个 ZIP 的目录结构，而不是只复制 `.md` 文件。`package:` 声明的原图等图片附件仍使用其默认 `evidence/...` 或 `assets/...` 路径；它们是 ZIP 附件，不是单文件 HTML 中的内嵌图片。

`manifest.json` 包含：

- `source_documents`：配置和每个源章节的字节数、SHA256；
- `files`：ZIP 内生成文件和资源的归档路径、类型、字节数、SHA256；资源另含源路径和 media type。

`manifest.json` 自身不列入哈希清单。`--manifest-output` 写出的内容与 ZIP 内清单相同；不指定时，清单仍包含在 ZIP 内。

## 5. 知乎转换规则

- 一级标题显示为“一、标题”“二、标题”等。
- 二至六级标题都压平为第二级：未编号标题使用 Markdown 常用的 `1. 标题` 形式；源标题已经有 `2.`、`3)` 等数字前缀时原样保留，避免生成 `1. 2. ...`，也不手工重编号。
- 所有公式由同源 TeX 自动转成可读文本；复杂公式采用文本方案，不生成公式截图。常见上下标、分式、根式、求和、关系符号、希腊字母和运算符会转成 Unicode 或可读函数形式。
- 表格单元格中的公式同样转成文本，不输出 `$...$` 或原始 LaTeX 命令。单元格内换行使用 `<br>`。
- 公式记录 JSON 中的 `source` 是未经手工改写的原始 TeX，`text` 是知乎正文使用的文本，`in_table` 标记表格公式。该 JSON 是发布前核对复杂公式的必要记录。
- 代码块、代码中的数字和正文数字不由工具手工调整；Markdown 特殊字符只做必要转义。
- 原始 HTML 被禁用，避免源文档引入外部脚本或破坏单文件页面。

## 6. 测试

```bash
cd /home/zhaosiying/codebase/tilelang-cuda-z3
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s article/tests -v
```

测试只使用 `tempfile.TemporaryDirectory` 中的 synthetic minimal 配置、Markdown、图片、CSV/JSON/log 和 KaTeX fixture，不读取正文、不写入 evidence，也不用于生成正文。覆盖：

- HTML 目录/锚点/表格/代码高亮/公式/base64 图片；
- 知乎两级标题、表内公式降级、数字保留和公式记录；
- `article/assets` 与 evidence 图片、显式及自动附件；
- 非显示 `package:` 原图声明、可见链接改写、原图不嵌入 HTML、普通 PNG 链接不隐式收集，以及缺失声明文件报错；
- manifest 逐项 SHA256 和 ZIP 内容；
- 完整 CLI、`--check-only` 零写入；
- 缺章节、缺图片、缺附件、缺 KaTeX、缺字体和远程图片错误。

测试是 CPU-only，不构建 TileLang、不调用 GPU、不访问网络。

## 7. 发布前检查

1. **来源检查**：确认 `index.json` 章节顺序正确；所有数字和结论来自源文及真实 evidence。构建器不会补占位性能数据。
2. **预检**：运行 `--check-only`，确保章节、图片、附件、KaTeX 和全部字体均通过。
3. **离线 HTML 检查**：断网或用浏览器禁用网络后打开 HTML；检查目录跳转、锚点、代码、表格、图片、字体和公式，并确认没有“公式渲染失败”提示。
4. **知乎检查**：检查标题只有规定两级、表格没有 LaTeX、复杂公式语义与 `*.formulas.json` 一致；保留 ZIP 中的图片相对路径。
5. **证据检查**：核对 CSV/JSON/log 的源路径与 SHA256，确认没有把 synthetic 测试数据或临时结果当成真实性能证据。
6. **包完整性检查**：解压 ZIP，按 `manifest.json` 逐项复核 SHA256，并检查清单中的源文档哈希是否为本次审阅版本。
7. **发布边界**：本工具不执行发布。不要自动发布到 Awesome-HTML 或任何公网；公开前仍需人工审阅内容、证据和平台格式。
