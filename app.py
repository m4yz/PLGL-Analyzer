import io
import re
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="IT OPEX Variance Analyzer", page_icon="📊", layout="wide")

st.title("📊 IT OPEX Variance Analyzer")
st.caption("PL → GL → Opex Budget | 2 Properties | Monthly variance root-cause analysis")


# -----------------------------
# Helpers
# -----------------------------
def money(v):
    """Indonesian Rupiah display with thousands separators."""
    if pd.isna(v):
        return "-"
    return f"Rp {float(v):,.0f}".replace(",", ".")


def normalize_account(x):
    """Normalize Excel/SAP account values without corrupting numeric float accounts."""
    if pd.isna(x):
        return ""
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        if float(x).is_integer():
            return str(int(x))
    s = str(x).strip().upper()
    # Handle text values such as "758069.0" or "P758069".
    s = re.sub(r"^P(?=\\d)", "", s)
    if re.fullmatch(r"\\d+\\.0+", s):
        s = s.split(".")[0]
    return re.sub(r"[^0-9]", "", s)


def read_excel_file(uploaded):
    data = uploaded.getvalue()
    return pd.ExcelFile(io.BytesIO(data))


def classify_property(filename, workbook):
    text = (filename + " " + " ".join(workbook.sheet_names)).upper()
    if "SVHI" in text or "PAN PACIFIC" in text or "PPJKT" in text:
        return "PPJKT"
    if "SVSSI" in text or "PARKROYAL SERVICED" in text or "PRSJKT" in text:
        return "PRSJKT"
    return "UNKNOWN"


def parse_pl(uploaded):
    xls = read_excel_file(uploaded)
    sheet = "PL IT" if "PL IT" in xls.sheet_names else xls.sheet_names[0]
    raw = pd.read_excel(xls, sheet_name=sheet, header=None)

    # Actual / Budget / Variance / Description are fixed positions in the supplied PL format.
    rows = []
    for i in range(5, len(raw)):
        desc = raw.iloc[i, 11] if raw.shape[1] > 11 else None
        if pd.isna(desc):
            continue
        desc = str(desc).strip()
        if not re.match(r"^P?\d{5,}", desc):
            continue

        m = re.match(r"^(P?\d+)\s*-\s*(.*)$", desc)
        if not m:
            continue

        account = normalize_account(m.group(1))
        description = m.group(2).strip()

        def num(col):
            try:
                return pd.to_numeric(raw.iloc[i, col], errors="coerce")
            except Exception:
                return np.nan

        rows.append({
            "account": account,
            "pl_account": "P" + account if account else "",
            "description": description,
            "actual": num(1),
            "budget": num(3),
            "variance": num(5),
            "ytd_actual": num(12),
            "ytd_budget": num(14),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"Could not find PL account rows in {uploaded.name}")
    df["actual"] = pd.to_numeric(df["actual"], errors="coerce").fillna(0)
    df["budget"] = pd.to_numeric(df["budget"], errors="coerce").fillna(0)
    df["variance"] = pd.to_numeric(df["variance"], errors="coerce").fillna(df["actual"] - df["budget"])
    df["ytd_actual"] = pd.to_numeric(df["ytd_actual"], errors="coerce").fillna(0)
    df["ytd_budget"] = pd.to_numeric(df["ytd_budget"], errors="coerce").fillna(0)
    df["ytd_variance"] = df["ytd_actual"] - df["ytd_budget"]
    return df


def parse_gl(uploaded):
    xls = read_excel_file(uploaded)
    sheet = "GL IT" if "GL IT" in xls.sheet_names else xls.sheet_names[0]
    raw = pd.read_excel(xls, sheet_name=sheet, header=None)

    header_row = None
    for i in range(min(15, len(raw))):
        vals = [str(v).strip().lower() for v in raw.iloc[i].tolist()]
        if "account" in vals and any("amount" in v for v in vals):
            header_row = i
            break

    if header_row is None:
        # PPJKT export has the header after metadata rows; fallback to likely row 6.
        for i in range(min(20, len(raw))):
            joined = " | ".join(str(v).lower() for v in raw.iloc[i].tolist())
            if "posting date" in joined and "text" in joined:
                header_row = i
                break

    if header_row is None:
        raise ValueError(f"Could not find GL header in {uploaded.name}")

    df = pd.read_excel(xls, sheet_name=sheet, header=header_row)
    df.columns = [str(c).strip() for c in df.columns]

    def find_col(patterns):
        normalized = {}
        for c in df.columns:
            key = re.sub(r"[^a-z0-9]+", " ", str(c).lower()).strip()
            normalized[c] = key
        for pattern in patterns:
            p = re.sub(r"[^a-z0-9]+", " ", str(pattern).lower()).strip()
            # Prefer an exact normalized header match.
            for c, key in normalized.items():
                if key == p:
                    return c
            # Then allow a contained phrase match.
            for c, key in normalized.items():
                if p in key:
                    return c
        return None

    account_col = find_col(["account"])
    amount_col = find_col(["amount in local currency", "amount"])
    date_col = find_col(["posting date"])
    assignment_col = find_col(["assignment"])
    text_col = find_col(["text"])
    doc_col = find_col(["document number"])
    ref_col = find_col(["reference"])
    cost_col = find_col(["cost center"])
    profit_col = find_col(["profit center"])

    if not account_col or not amount_col:
        raise ValueError(f"GL Account/Amount columns not found in {uploaded.name}")

    out = pd.DataFrame({
        "account": df[account_col].map(normalize_account),
        "amount": pd.to_numeric(df[amount_col], errors="coerce").fillna(0),
        "posting_date": pd.to_datetime(df[date_col], errors="coerce") if date_col else pd.NaT,
        "assignment": df[assignment_col].astype(str) if assignment_col else "",
        "text": df[text_col].astype(str) if text_col else "",
        "document": df[doc_col].astype(str) if doc_col else "",
        "reference": df[ref_col].astype(str) if ref_col else "",
        "cost_center": df[cost_col].astype(str) if cost_col else "",
        "profit_center": df[profit_col].astype(str) if profit_col else "",
    })
    out = out[out["account"] != ""].copy()

    # Defensive schema: Assignment is required by the drill-down UI.
    # If an export ever omits it, keep the column blank rather than crashing.
    required_cols = {
        "assignment": "",
        "document": "",
        "text": "",
        "cost_center": "",
        "profit_center": "",
        "reference": "",
    }
    for col, default in required_cols.items():
        if col not in out.columns:
            out[col] = default

    return out


def parse_budget(uploaded):
    xls = read_excel_file(uploaded)
    frames = []

    for sheet in xls.sheet_names:
        raw = pd.read_excel(xls, sheet_name=sheet)
        raw.columns = [str(c).strip() for c in raw.columns]
        required = {"Property Code", "Budget Item", "Account Code"}
        if not required.issubset(set(raw.columns)):
            continue

        cols = {
            "property": "Property Code",
            "category": "Budget Category",
            "budget_item": "Budget Item",
            "tagging": "Tagging",
            "owner": "Business Owner",
            "account_code": "Account Code",
            "application": "Application (if applicable)",
            "budget_sgd": "Budget Amount (SGD)",
            "budget_usd": "Amount (USD)",
            "remarks": "Remarks",
        }
        out = pd.DataFrame()
        for new, old in cols.items():
            out[new] = raw[old] if old in raw.columns else ""

        out["account"] = out["account_code"].map(normalize_account)
        out["budget_sgd"] = pd.to_numeric(out["budget_sgd"], errors="coerce").fillna(0)
        out["budget_usd"] = pd.to_numeric(out["budget_usd"], errors="coerce").fillna(0)
        out["property"] = out["property"].astype(str).str.upper().str.strip()
        frames.append(out)

    if not frames:
        raise ValueError(f"Could not find Opex Budget tables in {uploaded.name}")
    return pd.concat(frames, ignore_index=True)


def property_from_file(uploaded):
    xls = read_excel_file(uploaded)
    return classify_property(uploaded.name, xls)


def mapping_table(pl, budget, prop):
    b = budget[budget["property"] == prop].copy()
    agg = (
        b[b["account"] != ""]
        .groupby("account", as_index=False)
        .agg(
            budget_items=("budget_item", lambda x: " | ".join(pd.Series(x).dropna().astype(str).unique())),
            budget_categories=("category", lambda x: " | ".join(pd.Series(x).dropna().astype(str).unique())),
            budget_sgd=("budget_sgd", "sum"),
            budget_usd=("budget_usd", "sum"),
            budget_rows=("account", "size"),
        )
    )
    m = pl.merge(agg, on="account", how="left")
    m["mapping_status"] = np.select(
        [
            m["budget_rows"].isna(),
            m["budget_rows"] > 1,
        ],
        [
            "🔴 No Opex Budget Mapping",
            "🟡 Multiple Budget Items",
        ],
        default="🟢 Mapped",
    )
    return m


def analyze_gl(pl_account, gl):
    g = gl[gl["account"] == pl_account].copy()
    if g.empty:
        return g
    return g.sort_values("amount", ascending=False)



# -----------------------------
# PDF Summary Analysis
# -----------------------------
def pdf_text(value):
    """Safe text for ReportLab Paragraph."""
    if pd.isna(value):
        return "-"
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def build_pdf_summary(
    prop, view, usd_idr,
    total_actual, total_budget, total_variance,
    data, drivers, selected, row, g, bmatch,
    actual_col, budget_col, variance_col
):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

    def rp(v):
        if pd.isna(v):
            return "-"
        return f"Rp {float(v):,.0f}".replace(",", ".")

    def usd(v):
        if pd.isna(v):
            return "-"
        return f"{float(v):,.2f}"

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=1.0 * cm,
        rightMargin=1.0 * cm,
        topMargin=1.0 * cm,
        bottomMargin=1.0 * cm,
    )

    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "PDFTitle", parent=styles["Title"],
        fontSize=17, leading=21, alignment=TA_CENTER, spaceAfter=4
    )
    subtitle = ParagraphStyle(
        "PDFSubtitle", parent=styles["Normal"],
        fontSize=8.5, leading=11, alignment=TA_CENTER,
        textColor=colors.HexColor("#666666"), spaceAfter=10
    )
    heading = ParagraphStyle(
        "PDFHeading", parent=styles["Heading2"],
        fontSize=11.5, leading=14, textColor=colors.HexColor("#1F4E78"),
        spaceBefore=9, spaceAfter=5
    )
    body = ParagraphStyle(
        "PDFBody", parent=styles["Normal"], fontSize=8.2, leading=10.5
    )
    tiny = ParagraphStyle(
        "PDFTiny", parent=styles["Normal"], fontSize=6.2, leading=7.5
    )

    story = []

    story.append(Paragraph("IT OPEX VARIANCE ANALYSIS SUMMARY", title))
    story.append(Paragraph(
        f"Property: <b>{pdf_text(prop)}</b> &nbsp; | &nbsp; "
        f"View: <b>{pdf_text(view)}</b> &nbsp; | &nbsp; "
        f"Generated: {datetime.now().strftime('%d-%b-%Y %H:%M')}",
        subtitle
    ))

    # 1. Executive Summary
    story.append(Paragraph("1. Executive Summary", heading))
    above_count = int((data[variance_col] > 0).sum())
    below_count = int((data[variance_col] < 0).sum())

    kpi_rows = [
        ["Actual", "PL Budget", "Net Variance", "Above", "Below"],
        [rp(total_actual), rp(total_budget), rp(total_variance), str(above_count), str(below_count)],
    ]
    kpi = Table(kpi_rows, colWidths=[3.6*cm, 3.6*cm, 3.6*cm, 3.0*cm, 3.0*cm])
    kpi.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E78")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#F4F7FA")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C7CED6")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(kpi)
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "Variance status is based on PL Actual versus PL Budget. "
        "Opex Budget is supporting reference information only.",
        body
    ))

    # 2. Top Variance Contributors
    story.append(Paragraph("2. Top Variance Contributors", heading))
    top = drivers.head(10).copy()
    rows = [["Status", "PL", "Description", "Actual", "PL Budget", "Variance"]]

    for _, r in top.iterrows():
        status = "ABOVE" if r[variance_col] > 0 else ("BELOW" if r[variance_col] < 0 else "ON")
        rows.append([
            status,
            pdf_text(r.get("pl_account", "")),
            Paragraph(pdf_text(r.get("description", "")), tiny),
            rp(r[actual_col]),
            rp(r[budget_col]),
            rp(r[variance_col]),
        ])

    top_table = Table(
        rows,
        colWidths=[1.45*cm, 1.55*cm, 5.15*cm, 3.05*cm, 3.05*cm, 3.05*cm],
        repeatRows=1
    )
    top_style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9E2F3")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 6.1),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#C7CED6")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (3, 1), (-1, -1), "RIGHT"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    for i, (_, r) in enumerate(top.iterrows(), start=1):
        if r[variance_col] > 0:
            top_style.append(("BACKGROUND", (0, i), (0, i), colors.HexColor("#FCE4D6")))
        elif r[variance_col] < 0:
            top_style.append(("BACKGROUND", (0, i), (0, i), colors.HexColor("#E2F0D9")))
    top_table.setStyle(TableStyle(top_style))
    story.append(top_table)

    # 3. Selected Account Investigation
    story.append(Paragraph("3. Selected Account Investigation", heading))
    selected_status = (
        "ABOVE BUDGET" if row[variance_col] > 0
        else ("BELOW BUDGET" if row[variance_col] < 0 else "ON BUDGET")
    )

    account_rows = [
        ["PL Account", f"P{selected}"],
        ["Description", pdf_text(row.get("description", ""))],
        ["Status", selected_status],
        ["Actual", rp(row[actual_col])],
        ["PL Budget", rp(row[budget_col])],
        ["Variance", rp(row[variance_col])],
        ["GL Transactions", str(len(g))],
        ["GL Total", rp(g["amount"].sum()) if not g.empty else "Rp 0"],
    ]
    account_table = Table(account_rows, colWidths=[4.2*cm, 13.1*cm])
    account_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F1F3F5")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#C7CED6")),
        ("FONTSIZE", (0, 0), (-1, -1), 7.8),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(account_table)

    # 4. GL Transaction Contributors
    story.append(Paragraph("4. GL Transaction Contributors", heading))
    if g.empty:
        story.append(Paragraph("No matching GL transactions found for this account.", body))
    else:
        gl_pdf = g.copy()
        gl_pdf["_abs_amount"] = gl_pdf["amount"].abs()
        gl_pdf = gl_pdf.sort_values("_abs_amount", ascending=False).head(12)

        rows = [["Posting Date", "Assignment", "Document", "Amount", "Text / Description"]]
        for _, r in gl_pdf.iterrows():
            posting = "-"
            if pd.notna(r.get("posting_date")):
                posting = pd.to_datetime(r["posting_date"]).strftime("%d-%b-%Y")
            rows.append([
                posting,
                Paragraph(pdf_text(r.get("assignment", "")), tiny),
                pdf_text(r.get("document", "")),
                rp(r.get("amount", 0)),
                Paragraph(pdf_text(r.get("text", "")), tiny),
            ])

        gl_table = Table(
            rows,
            colWidths=[2.2*cm, 3.0*cm, 2.1*cm, 3.0*cm, 7.0*cm],
            repeatRows=1
        )
        gl_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9E2F3")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 6.0),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#C7CED6")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ALIGN", (3, 1), (3, -1), "RIGHT"),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(gl_table)

    # 5. Opex Budget Reference
    story.append(Paragraph("5. Opex Budget Reference", heading))
    if bmatch.empty:
        story.append(Paragraph(
            f"No Opex Budget reference is mapped to P{selected}. "
            "This is a mapping/reference item and does not determine the variance status.",
            body
        ))
    else:
        budget_pdf = bmatch.copy()
        budget_pdf["budget_idr"] = (
            pd.to_numeric(budget_pdf["budget_usd"], errors="coerce").fillna(0) * float(usd_idr)
        )

        story.append(Paragraph(
            f"Exchange rate used: <b>1 USD = {rp(usd_idr)}</b>. "
            "Budget IDR = Budget USD × exchange rate.",
            body
        ))
        story.append(Spacer(1, 4))

        rows = [["Category", "Budget Item", "Application", "Budget USD", "Budget IDR", "Remarks"]]
        for _, r in budget_pdf.iterrows():
            rows.append([
                Paragraph(pdf_text(r.get("category", "")), tiny),
                Paragraph(pdf_text(r.get("budget_item", "")), tiny),
                Paragraph(pdf_text(r.get("application", "")), tiny),
                usd(r.get("budget_usd", 0)),
                rp(r.get("budget_idr", 0)),
                Paragraph(pdf_text(r.get("remarks", "")), tiny),
            ])

        budget_table = Table(
            rows,
            colWidths=[2.5*cm, 5.2*cm, 2.4*cm, 2.5*cm, 2.9*cm, 2.6*cm],
            repeatRows=1
        )
        budget_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#FFF2CC")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 5.9),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#C7CED6")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ALIGN", (3, 1), (4, -1), "RIGHT"),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(budget_table)

    # 6. Key Observations
    story.append(Paragraph("6. Key Observations", heading))
    observations = []

    if not drivers.empty:
        largest = drivers.iloc[0]
        observations.append(
            f"Largest absolute variance: <b>{pdf_text(largest.get('pl_account', ''))} — "
            f"{pdf_text(largest.get('description', ''))}</b> at "
            f"<b>{rp(largest[variance_col])}</b>."
        )

    observations.append(
        f"Selected account P{selected} is <b>{selected_status}</b> with variance "
        f"<b>{rp(row[variance_col])}</b>."
    )

    if not g.empty:
        biggest = g.loc[g["amount"].abs().idxmax()]
        observations.append(
            f"Largest GL transaction: <b>{rp(biggest['amount'])}</b>; "
            f"Assignment: <b>{pdf_text(biggest.get('assignment', ''))}</b>; "
            f"Text: {pdf_text(biggest.get('text', ''))}."
        )

    if bmatch.empty:
        observations.append(
            "No Opex Budget reference is mapped to the selected account; review mapping if a budget reference is expected."
        )
    else:
        observations.append(
            f"The selected account has <b>{len(bmatch)}</b> Opex Budget reference item(s); "
            "review GL Assignment and Text / Description against the budget items."
        )

    for item in observations:
        story.append(Paragraph("• " + item, body))
        story.append(Spacer(1, 2))

    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Note: ABOVE/BELOW BUDGET is determined only by PL Actual versus PL Budget. "
        "Opex Budget is supporting reference information.",
        tiny
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# -----------------------------
# Upload
# -----------------------------
st.sidebar.header("📁 Upload Monthly Files")

pp_files = st.sidebar.file_uploader(
    "PPJKT — PL + GL",
    type=["xlsx", "xls"],
    accept_multiple_files=True,
    key="pp",
)
pr_files = st.sidebar.file_uploader(
    "PRSJKT — PL + GL",
    type=["xlsx", "xls"],
    accept_multiple_files=True,
    key="pr",
)
budget_file = st.sidebar.file_uploader(
    "Opex Budget — 2 Property",
    type=["xlsx", "xls"],
    key="budget",
)

if not pp_files or not pr_files or not budget_file:
    st.info("👈 Upload 2 files for PPJKT, 2 files for PRSJKT, and the Opex Budget file to start.")
    st.markdown(
        """
        ### Workflow
        **PL variance → GL contributors → transaction drill-down → Opex Budget mapping check**

        The analyzer intentionally does **not** compare PL budget amounts directly to the Opex Budget,
        because the supplied Opex Budget is annual SGD/USD while the PL is monthly/YTD IDR.
        The Opex Budget USD can be converted to IDR using the exchange rate entered in the sidebar.
        """
    )
    st.stop()

# Parse
def process_all(pp_bytes, pr_bytes, budget_bytes, pp_names, pr_names, budget_name):
    def wrap(name, data):
        class Upload:
            pass
        u = Upload()
        u.name = name
        u.getvalue = lambda: data
        return u

    pp_pl = pp_gl = pr_pl = pr_gl = None

    for name, data in zip(pp_names, pp_bytes):
        u = wrap(name, data)
        wb = read_excel_file(u)
        if "PL IT" in wb.sheet_names:
            pp_pl = parse_pl(u)
        if "GL IT" in wb.sheet_names:
            pp_gl = parse_gl(u)

    for name, data in zip(pr_names, pr_bytes):
        u = wrap(name, data)
        wb = read_excel_file(u)
        if "PL IT" in wb.sheet_names:
            pr_pl = parse_pl(u)
        if "GL IT" in wb.sheet_names:
            pr_gl = parse_gl(u)

    budget = parse_budget(wrap(budget_name, budget_bytes))
    return pp_pl, pp_gl, pr_pl, pr_gl, budget

with st.spinner("Reading PL, GL and Opex Budget..."):
    pp_pl, pp_gl, pr_pl, pr_gl, budget = process_all(
        [f.getvalue() for f in pp_files],
        [f.getvalue() for f in pr_files],
        budget_file.getvalue(),
        [f.name for f in pp_files],
        [f.name for f in pr_files],
        budget_file.name,
    )

datasets = {
    "PPJKT": (pp_pl, pp_gl),
    "PRSJKT": (pr_pl, pr_gl),
}

# -----------------------------
# Controls
# -----------------------------
st.sidebar.divider()
prop = st.sidebar.selectbox("Property", ["PPJKT", "PRSJKT"])
view = st.sidebar.radio("View", ["Current Month", "YTD"])
usd_idr = st.sidebar.number_input(
    "USD → IDR Exchange Rate",
    min_value=1_000.0,
    value=17_770.0,
    step=10.0,
    help="Enter how many IDR for 1 USD. Opex Budget IDR is calculated from Budget USD × this rate.",
)
st.sidebar.caption(f"1 USD = {money(usd_idr)}")

threshold = st.sidebar.number_input("Minimum absolute variance (Rp)", min_value=0, value=1_000_000, step=500_000)

pl, gl = datasets[prop]
data = mapping_table(pl, budget, prop)

actual_col = "actual" if view == "Current Month" else "ytd_actual"
budget_col = "budget" if view == "Current Month" else "ytd_budget"
variance_col = "variance" if view == "Current Month" else "ytd_variance"

# -----------------------------
# KPI
# -----------------------------
total_actual = data[actual_col].sum()
total_budget = data[budget_col].sum()
total_variance = data[variance_col].sum()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Actual", money(total_actual))
c2.metric("PL Budget", money(total_budget))
c3.metric("Variance", money(total_variance))
c4.metric("GL Transactions", f"{len(gl):,}")

st.divider()

# -----------------------------
# Variance contributors
# -----------------------------
st.subheader("🔎 Top Variance Contributors")
st.caption("Variance is driven by PL Actual vs PL Budget. Opex Budget is reference only. PL account Pxxxx is matched to GL account xxxx.")

drivers = data[abs(data[variance_col]) >= threshold].copy()
drivers["abs_variance"] = drivers[variance_col].abs()
drivers = drivers.sort_values("abs_variance", ascending=False)

display_cols = [
    "pl_account", "description", actual_col, budget_col, variance_col,
    "budget_rows"
]
show = drivers[display_cols].copy()
show["Status"] = np.where(
    show[variance_col] > 0, "🔴 ABOVE BUDGET",
    np.where(show[variance_col] < 0, "🟢 BELOW BUDGET", "⚪ ON BUDGET")
)
show["budget_rows"] = pd.to_numeric(show["budget_rows"], errors="coerce")
show = show.rename(columns={
    "pl_account": "PL Account",
    "description": "Description",
    actual_col: "Actual",
    budget_col: "PL Budget",
    variance_col: "Variance",
    "budget_rows": "Opex Budget Ref.",
})

def variance_style(v):
    if pd.isna(v):
        return ""
    if v > 0:
        return "background-color: #ffd6d6; color: #9b0000; font-weight: 700"
    if v < 0:
        return "background-color: #dff2df; color: #146414; font-weight: 700"
    return ""

cols_order = ["Status", "PL Account", "Description", "Actual", "PL Budget", "Variance", "Opex Budget Ref."]
show = show[cols_order]

styled = (
    show.style
    .format({
        "Actual": lambda v: money(v),
        "PL Budget": lambda v: money(v),
        "Variance": lambda v: money(v),
    })
    .map(variance_style, subset=["Variance"])
)
st.dataframe(styled, use_container_width=True, hide_index=True)

st.caption(
    "🔴 ABOVE BUDGET = Actual > PL Budget. "
    "🟢 BELOW BUDGET = Actual < PL Budget. "
    "Opex Budget is reference information only and does not determine the variance status."
)


# -----------------------------
# Drilldown
# -----------------------------
st.divider()
st.subheader("🧾 GL Drill-down")

accounts = drivers["account"].tolist()
if not accounts:
    st.success("No variance above the selected threshold.")
    st.stop()

selected = st.selectbox(
    "Select a variance account",
    accounts,
    format_func=lambda a: f"P{a} — {data.loc[data.account.eq(a), 'description'].iloc[0]}",
)

row = data[data["account"] == selected].iloc[0]
g = analyze_gl(selected, gl)

# Ensure the GL frame always has the fields required by the drill-down.
for _col in ["assignment", "document", "text", "cost_center", "profit_center"]:
    if _col not in gl.columns:
        gl[_col] = ""

d1, d2, d3, d4 = st.columns(4)
d1.metric("PL Variance", money(row[variance_col]))
d2.metric("GL Total", money(g["amount"].sum()) if not g.empty else "Rp 0")
d3.metric("GL Transactions", f"{len(g):,}")
d4.metric("Largest GL", money(g["amount"].max()) if not g.empty else "Rp 0")

if g.empty:
    st.warning("No matching GL transactions found for this account.")
else:
    gl_show = g[[
        "posting_date", "assignment", "document",
        "amount", "text", "cost_center", "profit_center"
    ]].copy()
    gl_show = gl_show.rename(columns={
        "posting_date": "Posting Date",
        "assignment": "Assignment",
        "document": "Document",
        "amount": "Amount",
        "text": "Text / Description",
        "cost_center": "Cost Center",
        "profit_center": "Profit Center",
    })
    st.caption(
        "**Assignment** = label/category used on the GL posting (often the quickest clue to who/what the charge relates to). "
        "**Text / Description** = the transaction narrative/details. "
        "Use both together with Amount when investigating the variance."
    )
    gl_styled = gl_show.style.format({
        "Posting Date": lambda v: "" if pd.isna(v) else v.strftime("%d-%b-%Y"),
        "Amount": lambda v: money(v),
    })
    st.dataframe(gl_styled, use_container_width=True, hide_index=True)

# -----------------------------
# Budget mapping
# -----------------------------
st.divider()
st.subheader("💰 Opex Budget Reference")

bmatch = budget[
    (budget["property"] == prop) & (budget["account"] == selected)
].copy()

if bmatch.empty:
    st.info(
        f"No Opex Budget reference is mapped to P{selected}. "
        "This does not make the PL variance wrong; it is only a reference for review."
    )
else:
    st.write(f"**P{selected} — {row['description']}**")
    bmatch["budget_idr"] = bmatch["budget_usd"] * usd_idr
    st.caption(
        f"Budget IDR is calculated from **Budget USD × exchange rate**. "
        f"Current input: **1 USD = {money(usd_idr)}**."
    )
    # Format the Opex reference as display text so Indonesian digit separators
    # are guaranteed to appear in Streamlit (e.g. Rp 444.250.000).
    budget_show = bmatch[
        ["category", "budget_item", "tagging", "owner", "account_code",
         "application", "budget_sgd", "budget_usd", "budget_idr", "remarks"]
    ].copy()
    budget_show = budget_show.rename(columns={
        "category": "category",
        "budget_item": "budget_item",
        "tagging": "tagging",
        "owner": "owner",
        "account_code": "account_code",
        "application": "application",
        "budget_sgd": "Budget SGD",
        "budget_usd": "Budget USD",
        "budget_idr": "Budget IDR",
        "remarks": "remarks",
    })
    budget_styled = budget_show.style.format({
        "Budget SGD": lambda v: "-" if pd.isna(v) else f"{float(v):,.2f}".replace(",", "X").replace(".", ",").replace("X", "."),
        "Budget USD": lambda v: "-" if pd.isna(v) else f"{float(v):,.2f}".replace(",", "X").replace(".", ",").replace("X", "."),
        "Budget IDR": lambda v: money(v),
    })
    st.dataframe(
        budget_styled,
        use_container_width=True,
        hide_index=True,
    )

# -----------------------------
# Opex reference coverage
# -----------------------------
st.divider()
st.subheader("📋 Opex Budget Reference Coverage")

mapped_count = int((data["budget_rows"].fillna(0) > 0).sum())
unmapped_count = int((data["budget_rows"].fillna(0) == 0).sum())

m1, m2 = st.columns(2)
m1.metric("PL Accounts with Opex Reference", mapped_count)
m2.metric("PL Accounts without Opex Reference", unmapped_count)

st.caption(
    "Coverage is informational only. An account without an Opex Budget reference "
    "is not automatically treated as a variance error."
)

st.caption(
    "The Opex Budget IDR reference uses Budget USD × the USD→IDR rate entered in the sidebar. "
    "V1 focuses on the business question: why did Actual exceed PL Budget? "
    "The next enhancement can compare transaction descriptions and budget items "
    "to suggest possible Finance mapping issues."
)

# -----------------------------
# PDF Summary Analysis
# -----------------------------
st.divider()
st.subheader("📄 PDF Summary Analysis")
st.caption(
    "Generate a management-ready PDF based on the current Property, View, "
    "selected variance account and USD → IDR exchange rate."
)

if st.button("📄 Generate PDF Summary Analysis", use_container_width=True):
    try:
        with st.spinner("Generating PDF Summary Analysis..."):
            st.session_state["pdf_summary_bytes"] = build_pdf_summary(
                prop=prop,
                view=view,
                usd_idr=usd_idr,
                total_actual=total_actual,
                total_budget=total_budget,
                total_variance=total_variance,
                data=data,
                drivers=drivers,
                selected=selected,
                row=row,
                g=g,
                bmatch=bmatch,
                actual_col=actual_col,
                budget_col=budget_col,
                variance_col=variance_col,
            )
        st.success("PDF Summary Analysis is ready.")
    except ImportError:
        st.error(
            "ReportLab is not installed. Add reportlab>=4.0 to requirements.txt and redeploy."
        )
    except Exception as e:
        st.error(f"PDF generation failed: {type(e).__name__}: {e}")

if st.session_state.get("pdf_summary_bytes"):
    st.download_button(
        "📥 Download PDF Summary Analysis",
        data=st.session_state["pdf_summary_bytes"],
        file_name=f"IT_OPEX_Variance_Analysis_{prop}_{view.replace(' ', '_')}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )
