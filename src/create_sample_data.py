
from pathlib import Path
from html import escape

from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
)

# Project data folder
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    "DocumentTitle",
    parent=styles["Title"],
    textColor=colors.darkblue,
    spaceAfter=20,
)

heading_style = ParagraphStyle(
    "SectionHeading",
    parent=styles["Heading2"],
    textColor=colors.darkblue,
    spaceBefore=12,
    spaceAfter=10,
)

body_style = ParagraphStyle(
    "DocumentBody",
    parent=styles["BodyText"],
    leading=16,
    spaceAfter=10,
)

# All figures below are fictional.
documents = {
    "q3_earnings_transcript.pdf": [
        (
            "Q3 Earnings Overview",
            """
            Fictional Company: NovaTech Industries Ltd.<br/>
            Financial Year: 2025-2026<br/><br/>
            Q3 Revenue: Rs. 125 crore<br/>
            Q3 Net Profit: Rs. 18 crore<br/>
            Q3 Operating Expenses: Rs. 82 crore<br/>
            Q3 EBITDA: Rs. 28 crore<br/><br/>
            Revenue increased by 12 percent compared
            with Q3 of the previous financial year.
            """
        ),
        (
            "Management Discussion",
            """
            Management reported increased demand for
            software products and cloud services.<br/><br/>
            Q3 revenue in the previous year was
            Rs. 111.61 crore.<br/>
            The company plans to increase investment
            in research and development.<br/><br/>
            Management identified competition and
            operating costs as major business risks.
            """
        ),
    ],

    "annual_report_2025.pdf": [
        (
            "Annual Financial Highlights",
            """
            NovaTech Industries Ltd.<br/>
            Illustrative Annual Report 2025<br/><br/>
            Total Annual Revenue: Rs. 450 crore<br/>
            Net Profit: Rs. 62 crore<br/>
            Total Assets: Rs. 380 crore<br/>
            Total Liabilities: Rs. 145 crore<br/>
            Shareholders' Equity: Rs. 235 crore<br/><br/>
            Annual revenue increased by 15 percent
            over the previous financial year.
            """
        ),
        (
            "Business Risks and Outlook",
            """
            Major risks include cybersecurity threats,
            changing customer requirements and
            increasing competition.<br/><br/>
            The company plans to expand its cloud
            services and invest in employee training.<br/><br/>
            Research and Development expenditure:
            Rs. 24 crore during 2025.
            """
        ),
    ],

    "10k_filing.pdf": [
        (
            "Annual Filing - Business Overview",
            """
            NovaTech Industries Ltd.<br/>
            Fictional 10-K Style Sample Filing<br/>
            Reporting Year: 2025<br/><br/>
            The company provides software development,
            cloud computing and IT consulting services.<br/><br/>
            Employees: 2,500<br/>
            Main operating regions: India and Asia-Pacific.
            """
        ),
        (
            "Financial Data and Risk Factors",
            """
            Total Revenue: Rs. 450 crore<br/>
            Cost of Revenue: Rs. 180 crore<br/>
            Gross Profit: Rs. 270 crore<br/>
            Operating Expenses: Rs. 165 crore<br/>
            Net Profit: Rs. 62 crore<br/><br/>
            Risk factors include data security,
            regulatory changes and market competition.
            """
        ),
    ],

    "financial_statements.pdf": [
        (
            "Balance Sheet",
            """
            NovaTech Industries Ltd.<br/>
            Sample Balance Sheet - 31 March 2025<br/><br/>
            Cash and Cash Equivalents: Rs. 45 crore<br/>
            Accounts Receivable: Rs. 35 crore<br/>
            Property and Equipment: Rs. 200 crore<br/>
            Other Assets: Rs. 100 crore<br/><br/>
            Total Assets: Rs. 380 crore<br/>
            Total Liabilities: Rs. 145 crore<br/>
            Shareholders' Equity: Rs. 235 crore
            """
        ),
        (
            "Income Statement and Cash Flow",
            """
            Annual Revenue: Rs. 450 crore<br/>
            Cost of Revenue: Rs. 180 crore<br/>
            Gross Profit: Rs. 270 crore<br/>
            Operating Expenses: Rs. 165 crore<br/>
            Net Profit: Rs. 62 crore<br/><br/>
            Operating Cash Flow: Rs. 75 crore<br/>
            Investing Cash Flow: Rs. -30 crore<br/>
            Financing Cash Flow: Rs. -12 crore<br/>
            Net Cash Increase: Rs. 33 crore
            """
        ),
    ],
}


def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 9)
    canvas.drawRightString(
        7.5 * inch,
        0.4 * inch,
        f"Page {doc.page}"
    )
    canvas.restoreState()


def create_pdf(filename, sections):
    file_path = DATA_DIR / filename

    pdf = SimpleDocTemplate(str(file_path))
    story = []

    for index, (heading, content) in enumerate(sections):
        if index > 0:
            story.append(PageBreak())

        story.append(Paragraph(
            escape(filename.replace(".pdf", "").replace("_", " ").title()),
            title_style
        ))

        story.append(Paragraph(heading, heading_style))
        story.append(Spacer(1, 10))

        story.append(Paragraph(content, body_style))

        story.append(Spacer(1, 20))
        story.append(Paragraph(
            "Note: This is fictional sample data created "
            "for educational RAG testing.",
            styles["Italic"]
        ))

    pdf.build(
        story,
        onFirstPage=add_page_number,
        onLaterPages=add_page_number
    )

    print(f"Created: {file_path}")


if __name__ == "__main__":
    for filename, sections in documents.items():
        create_pdf(filename, sections)

    print("\nAll four sample PDFs created successfully!")
