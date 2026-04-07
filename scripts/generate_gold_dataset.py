#!/usr/bin/env python3
"""
Generate the 100-pair gold standard Q&A dataset.

Distribution: 40 simple + 30 multi-hop + 30 comparative = 100 total.
Answers are placeholder text pointing to EDGAR data; they get filled
in after ingestion by a human review pass.

Output: data/gold_dataset.json
"""
from __future__ import annotations

import json
from pathlib import Path

import structlog

log = structlog.get_logger()

SIMPLE_QUESTIONS = [
    ("AAPL", "What was Apple's total net sales revenue in their most recent 10-K fiscal year?"),
    ("AAPL", "What was Apple's net income in their most recent fiscal year?"),
    ("AAPL", "What was Apple's earnings per diluted share in the most recent fiscal year?"),
    ("AAPL", "What was Apple's gross margin percentage in the most recent fiscal year?"),
    ("AAPL", "What were Apple's research and development expenses?"),
    ("AAPL", "What percentage of Apple's revenue came from iPhone sales?"),
    ("AAPL", "What was Apple's Services segment revenue?"),
    ("AAPL", "How much did Apple return to shareholders through buybacks and dividends?"),
    ("MSFT", "What was Microsoft's total revenue in their most recent fiscal year?"),
    ("MSFT", "What was Microsoft's Azure cloud revenue growth rate?"),
    ("MSFT", "What was Microsoft's operating income in the most recent fiscal year?"),
    ("MSFT", "What was Microsoft's net income?"),
    ("MSFT", "What were Microsoft's capital expenditures?"),
    ("MSFT", "What was Microsoft's Intelligent Cloud segment revenue?"),
    ("MSFT", "What was Microsoft's More Personal Computing segment revenue?"),
    ("MSFT", "What was Microsoft's Productivity and Business Processes segment revenue?"),
    ("AMZN", "What was Amazon's total net sales in the most recent fiscal year?"),
    ("AMZN", "What was Amazon Web Services revenue?"),
    ("AMZN", "What was Amazon's operating income?"),
    ("AMZN", "What was Amazon's North America segment revenue?"),
    ("AMZN", "What was Amazon's International segment operating income or loss?"),
    ("AMZN", "What was Amazon's net income?"),
    ("AMZN", "What was Amazon's free cash flow?"),
    ("AMZN", "What percentage of total revenue did AWS represent?"),
    ("GOOGL", "What was Alphabet's total revenue in the most recent fiscal year?"),
    ("GOOGL", "What was Google Search advertising revenue?"),
    ("GOOGL", "What was Google Cloud revenue?"),
    ("GOOGL", "What was Alphabet's net income?"),
    ("GOOGL", "What was YouTube advertising revenue?"),
    ("GOOGL", "What was Alphabet's operating margin?"),
    ("GOOGL", "What was Alphabet's headcount at fiscal year end?"),
    ("GOOGL", "What were Alphabet's capital expenditures?"),
    ("META", "What was Meta's total revenue in the most recent fiscal year?"),
    ("META", "What was Meta's advertising revenue?"),
    ("META", "What was Meta's net income?"),
    ("META", "What was Meta's operating margin?"),
    ("META", "What was Meta's Reality Labs operating loss?"),
    ("META", "How many daily active users did Meta's Family of Apps have?"),
    ("META", "What was Meta's free cash flow?"),
    ("META", "What were Meta's research and development expenses?"),
]

MULTIHOP_QUESTIONS = [
    ("Which of the five companies (AAPL, MSFT, AMZN, GOOGL, META) had the highest YoY revenue growth in their most recent 10-K? Show the calculation.", ["AAPL", "MSFT", "AMZN", "GOOGL", "META"], 4),
    ("What was Apple's compound annual growth rate (CAGR) for net income over the last two reported fiscal years?", ["AAPL"], 3),
    ("How did Microsoft's Azure revenue growth compare to its overall company revenue growth rate?", ["MSFT"], 3),
    ("What percentage of Amazon's total net sales did AWS account for, and how did this ratio change year-over-year?", ["AMZN"], 3),
    ("Calculate the ratio of Meta's Reality Labs losses to its total advertising revenue. What does this imply for profitability?", ["META"], 4),
    ("Which company had the highest R&D expense as a percentage of revenue among AAPL, MSFT, GOOGL? Show the calculation for each.", ["AAPL", "MSFT", "GOOGL"], 4),
    ("What was Alphabet's effective tax rate, and how does it compare to Microsoft's effective tax rate?", ["GOOGL", "MSFT"], 3),
    ("For Amazon, what is the ratio of AWS operating income to Amazon North America operating income?", ["AMZN"], 3),
    ("If Apple maintains its current gross margin percentage, how much gross profit would it generate at $400 billion in revenue?", ["AAPL"], 3),
    ("Which two of the five companies had the smallest gap between gross margin and operating margin?", ["AAPL", "MSFT", "AMZN", "GOOGL", "META"], 5),
    ("What is the combined cloud revenue of Microsoft Azure, Amazon AWS, and Google Cloud? Which is largest?", ["MSFT", "AMZN", "GOOGL"], 3),
    ("Calculate the total shareholder returns (buybacks + dividends) for all five companies combined.", ["AAPL", "MSFT", "AMZN", "GOOGL", "META"], 4),
    ("What is Apple's revenue per employee compared to Microsoft's revenue per employee?", ["AAPL", "MSFT"], 3),
    ("How much did Meta reduce its headcount relative to revenue growth between the two most recent fiscal years?", ["META"], 4),
    ("Among AMZN, GOOGL, and META, which had the highest operating leverage (operating income growth vs revenue growth)?", ["AMZN", "GOOGL", "META"], 5),
    ("What is the ratio of capex to revenue for each of the five companies? Which is the most capital-intensive?", ["AAPL", "MSFT", "AMZN", "GOOGL", "META"], 4),
    ("If Amazon's AWS segment grew at its current rate, when would it reach $150 billion in annual revenue?", ["AMZN"], 3),
    ("Compare Apple's iPhone revenue decline to Google's YouTube revenue growth in the same period.", ["AAPL", "GOOGL"], 4),
    ("What percentage of Alphabet's total revenue comes from advertising, and how has this percentage changed year-over-year?", ["GOOGL"], 3),
    ("Calculate the total net income for all five companies combined. Which company contributed the largest share?", ["AAPL", "MSFT", "AMZN", "GOOGL", "META"], 4),
    ("How does Meta's cost per daily active user compare to Alphabet's implied cost per search user?", ["META", "GOOGL"], 5),
    ("What is the combined R&D investment of all five companies? How does this compare to Apple's total revenue?", ["AAPL", "MSFT", "AMZN", "GOOGL", "META"], 3),
    ("Microsoft's gaming revenue (Xbox) plus Apple's Wearables revenue: which is larger and by how much?", ["MSFT", "AAPL"], 3),
    ("Which risk factor appeared in the most 10-K filings across the five companies?", ["AAPL", "MSFT", "AMZN", "GOOGL", "META"], 4),
    ("If Meta eliminated all Reality Labs losses, what would its effective net income be as a percentage of revenue?", ["META"], 3),
    ("Which company had the best free cash flow conversion rate (FCF / net income)?", ["AAPL", "MSFT", "AMZN", "GOOGL", "META"], 4),
    ("How does Amazon's operating margin in AWS compare to Microsoft's operating margin in Intelligent Cloud?", ["AMZN", "MSFT"], 4),
    ("What would Apple's total revenue be if Services grew 20% and Products stayed flat?", ["AAPL"], 3),
    ("Compare the percentage of revenue reinvested (R&D + capex) across all five companies.", ["AAPL", "MSFT", "AMZN", "GOOGL", "META"], 5),
    ("Which company had the highest EPS growth rate between the two most recent fiscal years?", ["AAPL", "MSFT", "AMZN", "GOOGL", "META"], 4),
]

COMPARATIVE_QUESTIONS = [
    "Compare Amazon and Microsoft's cloud revenue segments. Which is growing faster?",
    "How do Apple and Microsoft's gross margins compare, and what drives the difference?",
    "Compare the operating margins of all five companies. Which is most efficient?",
    "How does Google's advertising revenue compare to Meta's advertising revenue?",
    "Compare Amazon's e-commerce vs. AWS revenue split to Microsoft's commercial cloud vs. other revenue split.",
    "Which of the five companies has the highest net income margin? Show the calculation.",
    "Compare the risk factors disclosed by Apple and Microsoft around AI competition.",
    "How do the capital allocation strategies (buybacks vs. dividends vs. capex) differ across the five companies?",
    "Compare Apple's international vs. domestic revenue split to Google's.",
    "Which company has the most diversified revenue stream? Justify with segment data.",
    "How does Meta's advertising ARPU compare to Alphabet's ARPU?",
    "Compare employee count and revenue-per-employee across all five companies.",
    "Which company had the largest absolute increase in operating income year-over-year?",
    "How do the tax rates of Amazon and Apple compare? What factors explain the difference?",
    "Compare the debt levels (long-term debt) of Microsoft and Apple as a percentage of revenue.",
    "Which company has the strongest cash position relative to its market operations?",
    "Compare the geographic revenue concentration (US vs. International) for Apple vs. Meta.",
    "How do Microsoft and Google's cloud growth rates compare for their most recent fiscal year?",
    "Which company has the highest sales and marketing expense as a percentage of revenue?",
    "Compare the earnings per share growth of Apple, Microsoft, and Alphabet over the reported periods.",
    "How does Amazon's logistics and fulfillment cost structure compare to Meta's infrastructure costs?",
    "Which company has the most aggressive share buyback program relative to market cap?",
    "Compare the gross profit per employee for Apple vs. Microsoft.",
    "How do the operating expense structures of Google and Meta compare as percentages of revenue?",
    "Which company's risk factors most prominently feature regulatory/antitrust concerns?",
    "Compare the growth rates of Microsoft Teams vs. the overall Productivity segment.",
    "How does Apple's hardware gross margin compare to its services gross margin?",
    "Compare Amazon's AWS operating margin to Google Cloud's operating margin.",
    "Which of the five companies mentions AI most prominently in their risk factors and business overview?",
    "Compare the sustainability and ESG commitments disclosed across all five 10-K filings.",
]


def build_dataset() -> list[dict]:
    dataset = []
    pair_id = 0

    for ticker, question in SIMPLE_QUESTIONS:
        dataset.append({
            "id": f"gold_{pair_id:03d}",
            "question": question,
            "answer": f"[Answer requires EDGAR filing data for {ticker}. Run after ingestion.]",
            "supporting_chunks": [],
            "question_type": "simple",
            "difficulty": 2,
            "tickers": [ticker],
        })
        pair_id += 1

    for question, tickers, difficulty in MULTIHOP_QUESTIONS:
        dataset.append({
            "id": f"gold_{pair_id:03d}",
            "question": question,
            "answer": "[Answer requires EDGAR filing data. Run after ingestion.]",
            "supporting_chunks": [],
            "question_type": "multi_hop",
            "difficulty": difficulty,
            "tickers": tickers,
        })
        pair_id += 1

    for question in COMPARATIVE_QUESTIONS:
        dataset.append({
            "id": f"gold_{pair_id:03d}",
            "question": question,
            "answer": "[Answer requires EDGAR filing data. Run after ingestion.]",
            "supporting_chunks": [],
            "question_type": "comparative",
            "difficulty": 3,
            "tickers": ["AAPL", "MSFT", "AMZN", "GOOGL", "META"],
        })
        pair_id += 1

    return dataset


if __name__ == "__main__":
    Path("data").mkdir(exist_ok=True)
    dataset = build_dataset()
    output_path = Path("data/gold_dataset.json")
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)
    log.info("gold_dataset.saved", path=str(output_path), count=len(dataset))
    print(f"Generated {len(dataset)} Q&A pairs -> {output_path}")
    by_type: dict[str, int] = {}
    for item in dataset:
        qt = item["question_type"]
        by_type[qt] = by_type.get(qt, 0) + 1
    print("Distribution:", by_type)
