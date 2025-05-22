#!/usr/bin/env python3

import os
import arxiv
import datetime
from pathlib import Path
import openai
import time
import logging
import sys
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
import calendar

# 加载环境变量
load_dotenv()

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                   handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# 配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

PAPERS_DIR = Path("./papers")
CONCLUSION_FILE = Path("./conclusion.md")
CATEGORIES = [
    "cs.SE",
    "cs.PL",
    "cs.LG",
    "cs.AI",
    "cs.CL",
]

KEYWORDS = [
    " code ",
    " coder ",
    "speculative decoding", 
    " program ",
    " programer ",
    "HDL",
    "verilog"
]
MAX_PAPERS = 1  # 设置为1以便快速测试

# 配置OpenAI API用于DeepSeek
openai.api_key = DEEPSEEK_API_KEY
openai.api_base = "https://api.deepseek.com"

# 如果不存在论文目录则创建
PAPERS_DIR.mkdir(exist_ok=True)
logger.info(f"论文将保存在: {PAPERS_DIR.absolute()}")
logger.info(f"分析结果将写入: {CONCLUSION_FILE.absolute()}")

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

def analyze_paper_with_deepseek(pdf_path, paper):
    """使用DeepSeek API分析论文（使用OpenAI 0.28.0兼容格式）"""
    try:
        # 从Author对象中提取作者名
        author_names = [author.name for author in paper.authors]
        
        prompt = f"""
        论文标题: {paper.title}
        作者: {', '.join(author_names)}
        类别: {', '.join(paper.categories)}
        发布时间: {paper.published}
        
        请分析这篇研究论文并提供：
        1. 简明摘要
        2. 主要贡献和创新点，解决的什么问题
        3. 研究方法，具体采用的技术，工具，数据集
        4. 实验结果，包括数据集，实验设置，实验结果，实验结论
        5. 局限性或未来工作方向
        6. 论文的方法可以用在其它什么领域？（如 Verilog 代码生成，思维链）？
        
        请使用中文回答，并以纯文本，分自然段格式输出。
        """
        
        logger.info(f"正在分析论文: {paper.title}")
        response = openai.ChatCompletion.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一位专门总结和分析学术论文的研究助手。请使用中文回复。"},
                {"role": "user", "content": prompt},
            ]
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
    date_str = today.strftime('%Y-%m-%d')
    start_date = (today - datetime.timedelta(days=7)).strftime('%Y-%m-%d')
    end_date = date_str
    # 生成markdown内容并收集标题
    links = []
    with open(conclusion_file, 'a', encoding='utf-8') as f:
        f.write(f"\n\n## LLM4Code: {start_date} - {end_date}\n\n")
        for paper, analysis in papers_analyses:
            author_names = [author.name for author in paper.authors]
            f.write(f"### {paper.title}\n")
            f.write(f"**作者**: {', '.join(author_names)}\n")
            f.write(f"**发布日期**: {paper.published.strftime('%Y-%m-%d')}\n")
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
        ym_title = f"\n\n## {year_month} \n"
        # 读取现有readme内容
        if readme_path.exists():
            with open(readme_path, 'r', encoding='utf-8') as readme:
                readme_content = readme.read()
        else:
            readme_content = ''
        # 如果没有年-月标题则插入
        if ym_title.strip() not in readme_content:
            with open(readme_path, 'a', encoding='utf-8') as readme:
                readme.write(ym_title)
        # 插入本次分析列表
        list_title = f"\n\n### {start_date} - {end_date}\n"
        with open(readme_path, 'a', encoding='utf-8') as readme:
            readme.write(list_title)
            for link in links:
                readme.write(link + "\n")

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
            analysis = analyze_paper_with_deepseek(pdf_path, paper)
            papers_analyses.append((paper, analysis))
            
            # 分析完成后删除PDF文件
            delete_pdf(pdf_path)
    
    # 将分析结果写入conclusion.md（包含所有历史记录）
    write_to_conclusion(papers_analyses)
    
    
    logger.info("ArXiv论文追踪和分析完成")
    logger.info(f"结果已保存至 {CONCLUSION_FILE.absolute()}")

if __name__ == "__main__":
    main()
