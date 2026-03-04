#!/usr/bin/env python3

import os
import arxiv
import datetime
from pathlib import Path
from openai import OpenAI
import time
import logging
import sys
import json
import requests
import fitz  # PyMuPDF
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# 配置
KIMI_API_KEY = os.getenv("KIMI_API_KEY")
assert KIMI_API_KEY, "KIMI_API_KEY 环境变量未设置"
os.environ["OPENAI_API_BASE"] = "https://api.moonshot.cn/v1"

# PAPERS_DIR = Path("./papers")  # No longer needed for PDF downloads
# PAPERS_DIR.mkdir(exist_ok=True)
# logger.info(f"论文将保存在: {PAPERS_DIR.absolute()}")

DOWNLOADS_DIR = Path("./pdf_downloads")
DOWNLOADS_DIR.mkdir(exist_ok=True)

CATEGORIES = [
    "cs.AI",
    "cs.CL",
    "cs.SE",
]

KEYWORDS = [
    "code generation",
    "code repair",
    "code translation",
    "unit test",
    "coder",
    "speculative decoding",
    "HDL",
    "verilog",
    "code",
    "RTL",
    "kernel"
]
MAX_PAPERS = 500  # 设置为1以便快速测试


def get_recent_papers(categories, max_results=MAX_PAPERS, keywords=KEYWORDS):
    """获取最近7天内发布且包含关键词的指定类别论文（直接在arxiv query中检索关键词）"""
    today = datetime.datetime.now()
    seven_days_ago = today - datetime.timedelta(days=10)
    start_date = seven_days_ago.strftime("%Y%m%d")
    end_date = today.strftime("%Y%m%d")

    # 类别部分
    category_query = " OR ".join([f"cat:{cat}" for cat in categories])
    date_range = f"submittedDate:[{start_date}000000 TO {end_date}235959]"

    # 关键词部分，支持title/abstract字段
    keyword_query = " OR ".join([f'ti:"{kw}" OR abs:"{kw}"' for kw in keywords])

    # 综合query
    query = f"(({category_query}) AND {date_range} AND ({keyword_query}))"
    logger.info(f"正在搜索论文，查询条件: {query}")

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    results = list(search.results())
    logger.info(f"找到{len(results)}篇符合条件的论文")
    return results


def analyze_paper(paper):
    try:
        client = OpenAI(
            api_key=KIMI_API_KEY,
            base_url="https://api.moonshot.cn/v1",
        )

        prompt = f"""\
        论文标题: {paper.title}
        
        论文摘要:
        {paper.summary}
        
        请分析这篇研究论文的摘要，判断其是否与大模型代码生成相关，即是否应用大模型的推理能力搭建agent/训练模型实现提升代码生成的正确性/性能
        判断请严格，如果不是LLM相关，或者不是LLM for 代码生成，一定要标记为 relevant=false
        特别地，如果论文涉及以下内容，请将其标记为 "is_highlight":
        1. LLM for 硬件代码生成 (HDL, Verilog, RTL 等)
        2. LLM for GPU kernel 生成 (CUDA, Triton, 高性能算子等)
        注意，这些工作的核心都是 LLM for ***，即利用上LLM的能力，没利用上就是不相关
        请严格按照以下 JSON 格式返回结果：
        {{
            "relevant": true/false,
            "is_highlight": true/false,
            "highlight_reason": "如果是 highlight，说明原因（如 'HDL硬件代码生成'），否则为空",
            "one_sentence_summary": "一句话介绍论文讲的故事（中文）",
            "translation": "摘要翻译（中文）"
        }}
        
        如果 relevant 为 false，其他字段可以为空。
        """
        logger.info(f"正在分析论文: {paper.title}")

        system_msg = "你是一位专门总结和分析学术论文的研究助手。请务必返回合法的 JSON 格式。"
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ]

        response = client.chat.completions.create(
            model="moonshot-v1-8k",
            messages=messages,
            temperature=0.3,
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            # 尝试修复可能的 markdown code block
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
                result = json.loads(content)
            else:
                logger.error(f"JSON解析失败: {content}")
                return None

        if not result.get("relevant", False):
            logger.info(f"论文不相关: {paper.title}")
            return None

        logger.info(f"论文分析完成 (相关): {paper.title}")
        return result
    except Exception as e:
        logger.error(f"分析论文失败 {paper.title}: {str(e)}")
        return None


def download_pdf(paper):
    """下载论文 PDF 并返回本地路径"""
    try:
        pdf_url = paper.pdf_url
        file_name = f"{paper.entry_id.split('/')[-1]}.pdf"
        file_path = DOWNLOADS_DIR / file_name
        
        if file_path.exists():
            return file_path
            
        logger.info(f"正在下载 PDF: {paper.title}")
        response = requests.get(pdf_url, timeout=30)
        if response.status_code == 200:
            with open(file_path, "wb") as f:
                f.write(response.content)
            return file_path
        else:
            logger.error(f"下载 PDF 失败 ({response.status_code}): {paper.title}")
            return None
    except Exception as e:
        logger.error(f"下载 PDF 出错 {paper.title}: {str(e)}")
        return None


def extract_pdf_text(pdf_path):
    """从 PDF 中提取文本"""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        logger.error(f"提取 PDF 文本失败: {str(e)}")
        return ""


def generate_detailed_intro(paper, pdf_text):
    """基于 PDF 内容生成更详细的介绍"""
    try:
        client = OpenAI(
            api_key=KIMI_API_KEY,
            base_url="https://api.moonshot.cn/v1",
        )
        
        prompt = f"""\
        你在撰写论文精读博客。请为这篇论文写一段详细的中文介绍
        
        要求：
        1. 本文背景、动机 
        2. 通俗介绍本文的方法、流程
        3. 实验效果
        
        论文标题: {paper.title}
        
        论文内容片段:
        {pdf_text}
        """
        
        messages = [
            {"role": "system", "content": "你是一位专业的学术论文分析专家。"},
            {"role": "user", "content": prompt}
        ]
        
        response = client.chat.completions.create(
            model="moonshot-v1-32k",  # 使用长上下文模型
            messages=messages,
            temperature=0.3,
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"生成详细介绍失败: {str(e)}")
        return "（生成详细介绍失败）"


def write_to_conclusion(papers_analyses):
    """将分析结果写入archive目录下以yyyy-mm命名的文件夹下，文件名为当天日期如0522.md，并在上一级readme.md中插入列表链接。"""
    today = datetime.datetime.now()
    month_folder = today.strftime("%Y-%m")
    archive_dir = Path("../archive") / month_folder
    archive_dir.mkdir(parents=True, exist_ok=True)
    md_filename = today.strftime("%m%d") + ".md"
    conclusion_file = archive_dir / md_filename
    date_str = today.strftime("%Y-%m%d")
    start_date = (today - datetime.timedelta(days=7)).strftime("%Y-%m%d")
    end_date = date_str
    
    # 准备链接列表用于更新README
    links = []

    # 分离 highlights 和普通论文
    highlights = [(p, a) for p, a in papers_analyses if a.get("is_highlight", False)]
    regular_papers = [(p, a) for p, a in papers_analyses if not a.get("is_highlight", False)]

    # 使用 writelines 或多次 write，这里为了结构清晰，先整理内容
    with open(conclusion_file, "w", encoding="utf-8") as f:
        # 1. 写入 Highlights (本期看点)
        if highlights:
            f.write("## 本期看点\n\n")
            for paper, analysis in highlights:
                f.write(f"### {paper.title}\n\n")
                f.write(f"> **看点**: {analysis.get('highlight_reason', '核心技术突破')}\n\n")
                f.write(f"{analysis.get('detailed_intro', analysis.get('one_sentence_summary', '暂无详细介绍'))}\n\n")
                f.write(f"**PDF 链接**: {paper.pdf_url}\n\n")
            f.write("---\n\n")

        # 2. 写入 Overview
        f.write("## Overview\n\n")
        if highlights:
            f.write("#### 🌟 重点关注\n")
            for paper, analysis in highlights:
                f.write(f"- **{paper.title}**\n")
                f.write(f"    - {analysis.get('one_sentence_summary', '暂无总结')}\n")
            f.write("\n")
            
        f.write("#### 📚 常规更新\n")
        for paper, analysis in regular_papers:
            f.write(f"- **{paper.title}**\n")
            f.write(f"    - {analysis.get('one_sentence_summary', '暂无总结')}\n")
        f.write("\n---\n\n")

        # 3. 写入详细内容
        f.write("## 详细列表\n\n")
        for paper, analysis in papers_analyses:
            author_names = [author.name for author in paper.authors]
            is_hl = analysis.get("is_highlight", False)
            title_prefix = "🌟 " if is_hl else ""
            f.write(f"### {title_prefix}{paper.title}\n\n")
            f.write(f"**作者**: {', '.join(author_names)}\n")
            f.write(f"**日期**: {paper.published.strftime('%Y-%m-%d')}\n")
            f.write(f"**链接**: {paper.entry_id}\n\n")
            
            f.write(f"#### 一句话总结\n{analysis.get('one_sentence_summary', '')}\n\n")
            f.write(f"#### 摘要翻译\n{analysis.get('translation', '')}\n\n")
            f.write("---\n\n")
            
            # --- 收集链接 Logic ---
            anchor = (
                paper.title.replace(" ", "-")
                .replace("#", "")
                .replace(".", "")
                .replace("?", "")
                .replace("!", "")
                .replace(":", "")
                .replace("，", "")
                .replace("。", "")
                .replace("：", "")
                .replace("、", "")
                .replace("/", "")
                .replace("（", "")
                .replace("）", "")
                .replace("(", "")
                .replace(")", "")
                .replace("[", "")
                .replace("]", "")
                .replace("【", "")
                .replace("】", "")
                .replace('"', "")
                .replace("'", "")
                .replace(",", "")
                .replace(";", "")
                .replace("；", "")
                .replace("—", "")
                .replace("–", "")
                .replace("·", "")
                .replace("…", "")
                .replace("=", "")
                .replace("+", "")
                .replace("*", "")
                .replace("&", "")
                .replace("^", "")
                .replace("%", "")
                .replace("$", "")
                .replace("@", "")
                .replace("~", "")
                .replace("`", "")
                .replace("|", "")
                .replace("<", "")
                .replace(">", "")
                .replace("{", "")
                .replace("}", "")
                .replace("\\", "")
                .replace(" ", "-")
            )
            link = f"- [{title_prefix}{paper.title}](archive/{month_folder}/{md_filename}#{anchor})"
            links.append(link)
            
    logger.info(f"分析结果已写入 {conclusion_file}")

    # 在上一级readme.md中插入/追加列表
    readme_path = Path("../README.md")
    if links:
        # 年-月标题
        year_month = today.strftime("%Y-%m")
        ym_title = f"\n\n## {year_month} \n\n"
        # 读取现有readme内容
        if readme_path.exists():
            with open(readme_path, "r", encoding="utf-8") as readme:
                readme_content = readme.read()
        else:
            readme_content = ""
        # 去除第一行大标题 # Awesome-LLM4Code
        readme_content = readme_content.split("\n")[1:]
        readme_content = "\n".join(readme_content)
        # 准备新内容
        list_title = f"### {start_date[5:]}-{end_date[5:]}\n\n"
        new_content = list_title
        for link in links:
            new_content += link + "\n"

        # 如果存在年月标题，则插入到该标题下面
        if ym_title.strip() in readme_content:
            # 找到年月标题的位置
            title_pos = readme_content.find(ym_title.strip())
            readme_content = (
                readme_content[: title_pos + len(ym_title)]
                + new_content
                + readme_content[title_pos + len(ym_title) :]
            )
        else:
            # 如果没有年月标题则插入到开头
            readme_content = ym_title + new_content + readme_content

        # 写入更新后的内容
        with open(readme_path, "w", encoding="utf-8") as readme:
            readme.write("# Awesome-LLM4Code\n\n" + readme_content)


def filter_papers_by_keywords(papers, keywords=KEYWORDS):
    """只保留标题或摘要中包含指定关键词的论文（全部小写比对）"""
    filtered = []
    for paper in papers:
        title = getattr(paper, "title", "").lower()
        summary = getattr(paper, "summary", "").lower()
        if any(kw in title or kw in summary for kw in keywords):
            filtered.append(paper)
    return filtered


import concurrent.futures

def main():
    logger.info("开始ArXiv论文跟踪")
    papers = get_recent_papers(CATEGORIES, MAX_PAPERS, KEYWORDS)
    logger.info(f"最终筛选后剩余{len(papers)}篇论文")
    if not papers:
        logger.info("所选时间段没有找到论文。退出。")
        return

    # 处理每篇论文 (多线程)
    papers_analyses = []
    MAX_WORKERS = 10  # 根据API限制调整
    
    logger.info(f"开始利用多线程分析论文，线程数: {MAX_WORKERS}")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务
        future_to_paper = {executor.submit(analyze_paper, paper): paper for paper in papers}
        
        # 获取结果
        for future in concurrent.futures.as_completed(future_to_paper):
            paper = future_to_paper[future]
            try:
                result = future.result()
                if result:
                    # 如果是 highlight，进一步处理
                    if result.get("is_highlight", False):
                        logger.info(f"处理高亮论文: {paper.title}")
                        pdf_path = download_pdf(paper)
                        if pdf_path:
                            text = extract_pdf_text(pdf_path)
                            if text:
                                detailed_intro = generate_detailed_intro(paper, text)
                                result["detailed_intro"] = detailed_intro
                            else:
                                logger.warning(f"无法从 PDF 提取文本: {paper.title}")
                        else:
                            logger.warning(f"无法下载 PDF: {paper.title}")
                    
                    papers_analyses.append((paper, result))
            except Exception as exc:
                logger.error(f"{paper.title} generated an exception: {exc}")

    # 将分析结果写入conclusion.md（包含所有历史记录）
    if papers_analyses:
        # 按发布日期排序（可选，保持一致性）
        papers_analyses.sort(key=lambda x: x[0].published, reverse=True)
        write_to_conclusion(papers_analyses)
    else:
        logger.info("没有找到相关论文(经LLM筛选)。")
    logger.info("ArXiv论文追踪和分析完成")


if __name__ == "__main__":
    main()
