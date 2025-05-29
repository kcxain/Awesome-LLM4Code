#!/usr/bin/env python3

import os
import arxiv
import datetime
from pathlib import Path
from openai import OpenAI
import time
import logging
import sys
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                   handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# 配置
KIMI_API_KEY = os.getenv("KIMI_API_KEY")
assert KIMI_API_KEY, "KIMI_API_KEY 环境变量未设置"
os.environ["OPENAI_API_BASE"] = "https://api.moonshot.cn/v1"

PAPERS_DIR = Path("./papers")
CATEGORIES = [
    "cs.SE",
    "cs.PL",
    "cs.LG",
    "cs.AI",
    "cs.CL",
]

KEYWORDS = [
    "code generation",
    "code repair",
    "code translation",
    "unit test",
    "coder",
    "speculative decoding", 
    "program",
    "programer",
    "HDL",
    "verilog"
]
MAX_PAPERS = 20  # 设置为1以便快速测试


# 如果不存在论文目录则创建
PAPERS_DIR.mkdir(exist_ok=True)
logger.info(f"论文将保存在: {PAPERS_DIR.absolute()}")

def get_recent_papers(categories, max_results=MAX_PAPERS, keywords=KEYWORDS):
    """获取最近7天内发布且包含关键词的指定类别论文（直接在arxiv query中检索关键词）"""
    today = datetime.datetime.now()
    seven_days_ago = today - datetime.timedelta(days=7)
    start_date = seven_days_ago.strftime('%Y%m%d')
    end_date = today.strftime('%Y%m%d')

    # 类别部分
    category_query = " OR ".join([f"cat:{cat}" for cat in categories])
    date_range = f"submittedDate:[{start_date}000000 TO {end_date}235959]"

    # 关键词部分，支持title/abstract字段
    keyword_query = " OR ".join([f"ti:\"{kw}\" OR abs:\"{kw}\"" for kw in keywords])

    # 综合query
    query = f"(({category_query}) AND {date_range} AND ({keyword_query}))"
    logger.info(f"正在搜索论文，查询条件: {query}")

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    results = list(search.results())
    logger.info(f"找到{len(results)}篇符合条件的论文")
    return results

def download_paper(paper, output_dir):
    """将论文PDF下载到指定目录"""
    pdf_path = output_dir / f"{paper.get_short_id().replace('/', '_')}.pdf"
    
    # 如果已下载则跳过
    if pdf_path.exists():
        logger.info(f"论文已下载: {pdf_path}")
        return pdf_path
    
    try:
        logger.info(f"正在下载: {paper.title}")
        paper.download_pdf(filename=str(pdf_path))
        logger.info(f"已下载到 {pdf_path}")
        return pdf_path
    except Exception as e:
        logger.error(f"下载论文失败 {paper.title}: {str(e)}")
        return None

def analyze_paper(pdf_path, paper):
    try:
        
        prompt = f"""\
        论文标题: {paper.title}
        
        请分析这篇研究论文并提供：
        1. 摘要翻译
        2. 主要贡献和创新点，解决的什么问题
        3. 研究方法，具体采用的技术，工具，数据集
        4. 实验结果，包括数据集，实验设置，实验结果，实验结论
        5. 方法可以用在其它什么领域（如 代码生成，代码修复，Verilog 代码生成，思维链）？
        
        请使用中文回答，并以纯文本，分自然段格式输出
        """
        logger.info(f"正在分析论文: {paper.title}")
        client = OpenAI(
            api_key=KIMI_API_KEY,
            base_url="https://api.moonshot.cn/v1",
        )
        # moonshot.pdf 是一个示例文件, 我们支持文本文件和图片文件，对于图片文件，我们提供了 OCR 的能力
        # 上传文件时，我们可以直接使用 openai 库的文件上传 API，使用标准库 pathlib 中的 Path 构造文件
        # 对象，并将其传入 file 参数即可，同时将 purpose 参数设置为 file-extract；注意，目前文件上传
        # 接口仅支持 file-extract 一种 purpose 值。
        file_object = client.files.create(file=pdf_path, purpose="file-extract")
        
        # 获取结果
        # file_content = client.files.retrieve_content(file_id=file_object.id)
        # 注意，某些旧版本示例中的 retrieve_content API 在最新版本标记了 warning, 可以用下面这行代替
        # （如果使用旧版本的 SDK，可以继续延用 retrieve_content API）
        file_content = client.files.content(file_id=file_object.id).text
        # 计算token数并在需要时截断
        system_msg = "你是一位专门总结和分析学术论文的研究助手。请使用中文回复。"
        messages = [{"role": "system", "content": system_msg}]
        
        # 估算token数 (粗略估计:中文字符2个token,英文单词1个token)
        def estimate_tokens(text):
            chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
            other_chars = len(text) - chinese_chars
            return chinese_chars * 2 + other_chars
            
        file_tokens = estimate_tokens(file_content)
        prompt_tokens = estimate_tokens(prompt)
        system_tokens = estimate_tokens(system_msg)
        
        total_tokens = file_tokens + prompt_tokens + system_tokens
        
        if total_tokens > 24576:
            # 截断file_content以适应token限制
            max_file_tokens = 24576 - prompt_tokens - system_tokens
            truncated_content = file_content[:int(max_file_tokens/2)]  # 粗略截断
            messages.append({"role": "system", "content": truncated_content})
        else:
            messages.append({"role": "system", "content": file_content})
            
        messages.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(
            model="moonshot-v1-32k",
            messages=messages,
            temperature = 0.3
        )
        
        analysis = response.choices[0].message.content
        logger.info(f"论文分析完成: {paper.title}")
        return analysis
    except Exception as e:
        logger.error(f"分析论文失败 {paper.title}: {str(e)}")
        return f"**论文分析出错**: {str(e)}"

def write_to_conclusion(papers_analyses):
    """将分析结果写入archive目录下以yyyy-mm命名的文件夹下，文件名为当天日期如0522.md，并在上一级readme.md中插入列表链接。"""
    today = datetime.datetime.now()
    month_folder = today.strftime('%Y-%m')
    archive_dir = Path("../archive") / month_folder
    archive_dir.mkdir(parents=True, exist_ok=True)
    md_filename = today.strftime('%m%d') + ".md"
    conclusion_file = archive_dir / md_filename
    date_str = today.strftime('%Y-%m%d')
    start_date = (today - datetime.timedelta(days=7)).strftime('%Y-%m%d')
    end_date = date_str
    # 生成markdown内容并收集标题
    links = []
    with open(conclusion_file, 'a', encoding='utf-8') as f:
        for paper, analysis in papers_analyses:
            author_names = [author.name for author in paper.authors]
            f.write(f"### {paper.title}\n\n")
            f.write(f"**作者**: {', '.join(author_names)}\n\n")
            f.write(f"**日期**: {paper.published.strftime('%Y-%m-%d')}\n\n")
            f.write(f"**链接**: {paper.entry_id}\n\n")
            f.write(f"{analysis}\n\n")
            f.write("---\n\n")
            # 收集链接
            anchor = paper.title.replace(' ', '-').replace('#', '').replace('.', '').replace('?', '').replace('!', '').replace(':', '').replace('，', '').replace('。', '').replace('：', '').replace('、', '').replace('/', '').replace('（', '').replace('）', '').replace('(', '').replace(')', '').replace('[', '').replace(']', '').replace('【', '').replace('】', '').replace('"', '').replace("'", '').replace(',', '').replace(';', '').replace('；', '').replace('—', '').replace('–', '').replace('·', '').replace('…', '').replace('=', '').replace('+', '').replace('*', '').replace('&', '').replace('^', '').replace('%', '').replace('$', '').replace('@', '').replace('~', '').replace('`', '').replace('|', '').replace('<', '').replace('>', '').replace('{', '').replace('}', '').replace('^', '').replace('=', '').replace('\\', '').replace(' ', '-')
            link = f"- [{paper.title}](archive/{month_folder}/{md_filename}#{anchor})"
            links.append(link)
    logger.info(f"分析结果已写入 {conclusion_file}")

    # 在上一级readme.md中插入/追加列表
    readme_path = Path("../README.md")
    if links:
        # 年-月标题
        year_month = today.strftime('%Y-%m')
        ym_title = f"\n\n## {year_month} \n\n"
        # 读取现有readme内容
        if readme_path.exists():
            with open(readme_path, 'r', encoding='utf-8') as readme:
                readme_content = readme.read()
        else:
            readme_content = ''
            
        # 准备新内容
        list_title = f"### {start_date[5:]}-{end_date[5:]}\n\n"
        new_content = list_title
        for link in links:
            new_content += link + "\n"
            
        # 如果没有年-月标题则插入到开头
        if ym_title.strip() not in readme_content:
            new_content = ym_title + new_content
            
        # 将新内容插入到文件开头
        with open(readme_path, 'w', encoding='utf-8') as readme:
            readme.write(new_content + readme_content)

def delete_pdf(pdf_path):
    """删除PDF文件"""
    try:
        if pdf_path.exists():
            pdf_path.unlink()
            logger.info(f"已删除PDF文件: {pdf_path}")
        else:
            logger.info(f"PDF文件不存在，无需删除: {pdf_path}")
    except Exception as e:
        logger.error(f"删除PDF文件失败 {pdf_path}: {str(e)}")

def filter_papers_by_keywords(papers, keywords=KEYWORDS):
    """只保留标题或摘要中包含指定关键词的论文（全部小写比对）"""
    filtered = []
    for paper in papers:
        title = getattr(paper, 'title', '').lower()
        summary = getattr(paper, 'summary', '').lower()
        if any(kw in title or kw in summary for kw in keywords):
            filtered.append(paper)
    return filtered

def main():
    logger.info("开始ArXiv论文跟踪")
    papers = get_recent_papers(CATEGORIES, MAX_PAPERS, KEYWORDS)
    logger.info(f"最终筛选后剩余{len(papers)}篇论文")
    if not papers:
        logger.info("所选时间段没有找到论文。退出。")
        return
    
    # 处理每篇论文
    papers_analyses = []
    for i, paper in enumerate(papers, 1):
        logger.info(f"正在处理论文 {i}/{len(papers)}: {paper.title}")
        # 下载论文
        pdf_path = download_paper(paper, PAPERS_DIR)
        if pdf_path:
            # 休眠以避免达到API速率限制
            time.sleep(2)
            
            # 分析论文
            analysis = analyze_paper(pdf_path, paper)
            papers_analyses.append((paper, analysis))
            
            # 分析完成后删除PDF文件
            delete_pdf(pdf_path)
    
    # 将分析结果写入conclusion.md（包含所有历史记录）
    write_to_conclusion(papers_analyses)
    
    
    logger.info("ArXiv论文追踪和分析完成")

if __name__ == "__main__":
    main()
