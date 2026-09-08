//! ModelScope 论文检索 CLI
//!
//! 请求格式（逆向自 modelscope.cn 论文频道前端）：
//!   PUT https://www.modelscope.cn/api/v1/dolphin/papers
//!   body: {"Query":"...","PageSize":30,"PageNumber":1,"Sort":"default","Criterion":[]}
//!
//! 示例：
//!   paper-cli "evolution strategies"
//!   paper-cli "generative modeling" -n 10 --pages 2 -f drifting -l 200
//!   paper-cli "spiking neural network" --json -o hits.json

use anyhow::{bail, Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::io::Write;

const ENDPOINT: &str = "https://www.modelscope.cn/api/v1/dolphin/papers";

#[derive(Parser)]
#[command(
    name = "paper-cli",
    version,
    about = "ModelScope 论文检索 CLI（请求格式：PUT /api/v1/dolphin/papers）"
)]
struct Args {
    /// 检索关键词
    query: String,
    /// 每页数量
    #[arg(short = 'n', long, default_value_t = 20)]
    num: u32,
    /// 起始页码（从 1 开始）
    #[arg(short = 'p', long, default_value_t = 1)]
    page: u32,
    /// 连续抓取的页数
    #[arg(long, default_value_t = 1)]
    pages: u32,
    /// 排序（透传给服务端，如 default）
    #[arg(short = 's', long, default_value = "default")]
    sort: String,
    /// 客户端二次过滤：标题/摘要包含该词（大小写不敏感）
    #[arg(short = 'f', long)]
    filter: Option<String>,
    /// 摘要截断长度（0 = 不截断）
    #[arg(short = 'l', long, default_value_t = 140)]
    abs_len: usize,
    /// 以 JSON 输出（过滤后的记录，UTF-8）
    #[arg(long, default_value_t = false)]
    json: bool,
    /// 输出到文件而非 stdout（UTF-8）
    #[arg(short = 'o', long)]
    out: Option<String>,
}

#[derive(Serialize)]
struct ReqBody<'a> {
    #[serde(rename = "Query")]
    query: &'a str,
    #[serde(rename = "PageSize")]
    page_size: u32,
    #[serde(rename = "PageNumber")]
    page_number: u32,
    #[serde(rename = "Sort")]
    sort: &'a str,
    #[serde(rename = "Criterion")]
    criterion: Vec<serde_json::Value>,
}

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase")]
struct Resp {
    #[serde(default)]
    code: i32,
    #[serde(default)]
    message: Option<String>,
    data: Option<RespData>,
}

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase")]
struct RespData {
    #[serde(default)]
    papers: Vec<Paper>,
    #[serde(default)]
    total_count: Option<u64>,
}

#[derive(Deserialize, Serialize, Clone)]
#[serde(rename_all = "PascalCase")]
struct Paper {
    #[serde(default)]
    id: Option<i64>,
    #[serde(default)]
    title: Option<String>,
    #[serde(default)]
    recommended_title: Option<String>,
    #[serde(default)]
    abstract_cn: Option<String>,
    #[serde(default)]
    abstract_en: Option<String>,
    #[serde(default)]
    authors: Option<String>,
    #[serde(default)]
    publish_date: Option<String>,
    #[serde(default)]
    arxiv_id: Option<String>,
    #[serde(default)]
    arxiv_url: Option<String>,
    #[serde(default)]
    pdf_url: Option<String>,
    #[serde(default)]
    code_link: Option<String>,
    #[serde(default)]
    star: Option<f64>,
}

impl Paper {
    fn title(&self) -> &str {
        self.title
            .as_deref()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| self.recommended_title.as_deref().unwrap_or("(无标题)"))
    }

    fn abstract_text(&self) -> Option<&str> {
        self.abstract_cn
            .as_deref()
            .filter(|s| !s.is_empty())
            .or_else(|| self.abstract_en.as_deref().filter(|s| !s.is_empty()))
    }

    fn link(&self) -> Option<&str> {
        self.arxiv_url
            .as_deref()
            .filter(|s| !s.is_empty())
            .or_else(|| self.pdf_url.as_deref().filter(|s| !s.is_empty()))
    }
}

fn fetch_page(client: &reqwest::blocking::Client, args: &Args, page: u32) -> Result<RespData> {
    let body = ReqBody {
        query: &args.query,
        page_size: args.num,
        page_number: page,
        sort: &args.sort,
        criterion: Vec::new(),
    };
    let resp = client
        .put(ENDPOINT)
        .json(&body)
        .send()
        .context("请求 ModelScope 失败")?;
    let status = resp.status();
    let resp: Resp = resp.json().context("解析响应 JSON 失败")?;
    if !status.is_success() || (resp.code != 0 && resp.code != 200) {
        bail!(
            "服务端返回异常：HTTP {}，Code={:?}，Message={:?}",
            status,
            resp.code,
            resp.message
        );
    }
    resp.data.context("响应缺少 Data 字段")
}

fn matches_filter(p: &Paper, kw: &str) -> bool {
    let kw_lower = kw.to_lowercase();
    let hay = || -> String {
        let mut s = String::new();
        s.push_str(p.title());
        if let Some(a) = p.abstract_text() {
            s.push(' ');
            s.push_str(a);
        }
        s.to_lowercase()
    };
    hay().contains(&kw_lower)
}

fn truncate(s: &str, max: usize) -> String {
    if max == 0 || s.chars().count() <= max {
        s.replace('\n', " ")
    } else {
        let t: String = s.chars().take(max).collect();
        format!("{}…", t.replace('\n', " "))
    }
}

fn fmt_date(d: &Option<String>) -> String {
    d.as_deref()
        .unwrap_or("")
        .split('T')
        .next()
        .unwrap_or("")
        .to_string()
}

fn render_text(papers: &[Paper], args: &Args) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "检索 \"{}\"（页 {}..{}，每页 {}）——命中 {} 条\n\n",
        args.query,
        args.page,
        args.page + args.pages - 1,
        args.num,
        papers.len()
    ));
    for (i, p) in papers.iter().enumerate() {
        s.push_str(&format!(
            "[{}] {}（{}，★{:.1}）\n",
            i + 1,
            p.title(),
            fmt_date(&p.publish_date),
            p.star.unwrap_or(0.0)
        ));
        if let Some(a) = p.authors.as_deref() {
            if !a.is_empty() {
                s.push_str(&format!("    作者: {}\n", truncate(a, 90)));
            }
        }
        if let Some(link) = p.link() {
            s.push_str(&format!("    链接: {link}\n"));
        }
        if let Some(code) = p.code_link.as_deref() {
            if !code.is_empty() {
                s.push_str(&format!("    代码: {code}\n"));
            }
        }
        if let Some(ab) = p.abstract_text() {
            s.push_str(&format!("    摘要: {}\n", truncate(ab, args.abs_len)));
        }
        s.push('\n');
    }
    s
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.num == 0 || args.num > 100 {
        bail!("-n/--num 取值 1..=100");
    }
    if args.pages == 0 {
        bail!("--pages 至少为 1");
    }

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .user_agent("paper-cli/0.1 (SDT-ES research tooling)")
        .build()
        .context("构建 HTTP 客户端失败")?;

    let mut papers: Vec<Paper> = Vec::new();
    let mut total: Option<u64> = None;
    for k in 0..args.pages {
        let page = args.page + k;
        let data = fetch_page(&client, &args, page)?;
        if total.is_none() {
            total = data.total_count;
        }
        papers.extend(data.papers);
    }

    if let Some(kw) = &args.filter {
        papers.retain(|p| matches_filter(p, kw));
    }

    let payload = if args.json {
        serde_json::to_string_pretty(&papers).context("序列化 JSON 失败")?
    } else {
        render_text(&papers, &args)
    };

    match &args.out {
        Some(path) => {
            std::fs::write(path, payload.as_bytes())
                .with_context(|| format!("写入文件失败: {path}"))?;
            println!(
                "已写入 {path}（{} 条，服务端 TotalCount={:?}）",
                papers.len(),
                total
            );
        }
        None => {
            let stdout = std::io::stdout();
            let mut lock = stdout.lock();
            lock.write_all(payload.as_bytes())?;
            lock.flush()?;
        }
    }
    Ok(())
}
