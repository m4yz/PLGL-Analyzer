import io
import re
from pathlib import Path
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
)

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
# PDF Summary
# -----------------------------
def pdf_money(v):
    if pd.isna(v):
        return "-"
    return f"Rp {float(v):,.0f}".replace(",", ".")


def pdf_text(v):
    if pd.isna(v):
        return "-"
    return str(v).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_pdf_summary(
    prop, view, usd_idr, total_actual, total_budget, total_variance,
    drivers, data, selected, row, g, bmatch, actual_col, budget_col, variance_col
):
    """Create an executive PDF summary from the currently selected analysis."""
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=1.3 * cm,
        leftMargin=1.3 * cm,
        topMargin=1.2 * cm,
        bottomMargin=1.2 * cm,
        title=f"IT OPEX Variance Analysis - {prop}",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=19,
        leading=23,
        spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        textColor=colors.HexColor("#666666"),
        fontSize=9,
        spaceAfter=14,
    )
    heading_style = ParagraphStyle(
        "ReportHeading",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#1F4E78"),
        spaceBefore=10,
        spaceAfter=6,
    )
    small_style = ParagraphStyle(
        "Small",
        parent=styles["Normal"],
        fontSize=7.5,
        leading=9,
    )
    normal_style = ParagraphStyle(
        "NormalReport",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
    )

    story = []

    # Header
    story.append(Paragraph("IT OPEX VARIANCE ANALYSIS SUMMARY", title_style))
    story.append(
        Paragraph(
            f"Property: <b>{pdf_text(prop)}</b> &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"View: <b>{pdf_text(view)}</b> &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"Generated: {datetime.now().strftime('%d-%b-%Y %H:%M')}",
            subtitle_style,
        )
    )

    # KPI Summary
    story.append(Paragraph("1. Executive Summary", heading_style))

    above_count = int((data[variance_col] > 0).sum())
    below_count = int((data[variance_col] < 0).sum())
    on_count = int((data[variance_col] == 0).sum())

    kpi_data = [
        ["Actual", "PL Budget", "Net Variance", "Above Budget", "Below Budget"],
        [
            pdf_money(total_actual),
            pdf_money(total_budget),
            pdf_money(total_variance),
            str(above_count),
            str(below_count),
        ],
    ]
    kpi_table = Table(kpi_data, colWidths=[3.35 * cm] * 5)
    kpi_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E78")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#F4F7FA")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B7C3D0")),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 8))

    summary_note = (
        f"The selected dataset contains <b>{above_count}</b> accounts above budget, "
        f"<b>{below_count}</b> accounts below budget and <b>{on_count}</b> on budget. "
        "Variance status is based on PL Actual versus PL Budget. "
        "Opex Budget information is used as a reference for mapping and investigation."
    )
    story.append(Paragraph(summary_note, normal_style))

    # Top Variance Contributors
    story.append(Paragraph("2. Top Variance Contributors", heading_style))
    top = drivers.copy()
    top["Status"] = np.where(
        top[variance_col] > 0, "ABOVE BUDGET",
        np.where(top[variance_col] < 0, "BELOW BUDGET", "ON BUDGET")
    )
    top = top.head(10)

    top_rows = [["Status", "PL Account", "Description", "Actual", "PL Budget", "Variance"]]
    for _, r in top.iterrows():
        top_rows.append([
            pdf_text(r["Status"]),
            pdf_text(r["pl_account"]),
            Paragraph(pdf_text(r["description"]), small_style),
            pdf_money(r[actual_col]),
            pdf_money(r[budget_col]),
            pdf_money(r[variance_col]),
        ])

    top_table = Table(
        top_rows,
        colWidths=[2.0 * cm, 1.8 * cm, 5.1 * cm, 2.45 * cm, 2.45 * cm, 2.45 * cm],
        repeatRows=1,
    )
    top_style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9E2F3")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C7CED6")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (3, 1), (-1, -1), "RIGHT"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    for i, (_, r) in enumerate(top.iterrows(), start=1):
        if r[variance_col] > 0:
            top_style.append(("BACKGROUND", (0, i), (0, i), colors.HexColor("#FCE4D6")))
        elif r[variance_col] < 0:
            top_style.append(("BACKGROUND", (0, i), (0, i), colors.HexColor("#E2F0D9")))
    top_table.setStyle(TableStyle(top_style))
    story.append(top_table)

    # Selected account analysis
    story.append(Paragraph("3. Selected Account Investigation", heading_style))
    status = (
        "ABOVE BUDGET" if row[variance_col] > 0
        else "BELOW BUDGET" if row[variance_col] < 0
        else "ON BUDGET"
    )

    account_rows = [
        ["PL Account", f"P{selected}"],
        ["Description", pdf_text(row["description"])],
        ["Status", status],
        ["Actual", pdf_money(row[actual_col])],
        ["PL Budget", pdf_money(row[budget_col])],
        ["Variance", pdf_money(row[variance_col])],
        ["GL Transactions", str(len(g))],
        ["GL Total", pdf_money(g["amount"].sum()) if not g.empty else "Rp 0"],
    ]
    account_table = Table(account_rows, colWidths=[4.0 * cm, 13.0 * cm])
    account_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F1F3F5")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C7CED6")),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(account_table)
    story.append(Spacer(1, 7))

    # GL detail
    story.append(Paragraph("4. GL Transaction Contributors", heading_style))
    if g.empty:
        story.append(Paragraph("No matching GL transactions were found for this account.", normal_style))
    else:
        gl_pdf = g.copy()
        gl_pdf["abs_amount"] = gl_pdf["amount"].abs()
        gl_pdf = gl_pdf.sort_values("abs_amount", ascending=False).head(12)

        gl_rows = [["Posting Date", "Assignment", "Document", "Amount", "Text / Description"]]
        for _, r in gl_pdf.iterrows():
            date_value = ""
            if pd.notna(r.get("posting_date")):
                date_value = pd.to_datetime(r["posting_date"]).strftime("%d-%b-%Y")
            gl_rows.append([
                date_value,
                Paragraph(pdf_text(r.get("assignment", "")), small_style),
                pdf_text(r.get("document", "")),
                pdf_money(r.get("amount", 0)),
                Paragraph(pdf_text(r.get("text", "")), small_style),
            ])

        gl_table = Table(
            gl_rows,
            colWidths=[2.2 * cm, 3.2 * cm, 2.2 * cm, 2.6 * cm, 7.8 * cm],
            repeatRows=1,
        )
        gl_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9E2F3")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C7CED6")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ALIGN", (3, 1), (3, -1), "RIGHT"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(gl_table)

    # Opex reference
    story.append(Paragraph("5. Opex Budget Reference", heading_style))
    if bmatch is None or bmatch.empty:
        story.append(
            Paragraph(
                f"No Opex Budget reference is mapped to P{selected}. "
                "This does not automatically mean the PL variance is incorrect.",
                normal_style,
            )
        )
    else:
        b = bmatch.copy()
        b["budget_idr"] = pd.to_numeric(b["budget_usd"], errors="coerce").fillna(0) * usd_idr

        story.append(
            Paragraph(
                f"USD to IDR rate used: <b>1 USD = {pdf_money(usd_idr)}</b>. "
                "Budget IDR is calculated as Budget USD multiplied by the input exchange rate.",
                normal_style,
            )
        )
        story.append(Spacer(1, 5))

        budget_rows = [["Category", "Budget Item", "Application", "Budget USD", "Budget IDR", "Remarks"]]
        for _, r in b.iterrows():
            budget_rows.append([
                Paragraph(pdf_text(r.get("category", "")), small_style),
                Paragraph(pdf_text(r.get("budget_item", "")), small_style),
                Paragraph(pdf_text(r.get("application", "")), small_style),
                pdf_text(f"{float(r.get('budget_usd', 0)):,.2f}"),
                pdf_money(r.get("budget_idr", 0)),
                Paragraph(pdf_text(r.get("remarks", "")), small_style),
            ])

        budget_table = Table(
            budget_rows,
            colWidths=[3.0 * cm, 6.2 * cm, 2.4 * cm, 2.2 * cm, 2.8 * cm, 2.5 * cm],
            repeatRows=1,
        )
        budget_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#FFF2CC")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C7CED6")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ALIGN", (3, 1), (4, -1), "RIGHT"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(budget_table)

    # Key observations
    story.append(Paragraph("6. Key Observations", heading_style))
    observations = []

    if not drivers.empty:
        largest = drivers.iloc[0]
        observations.append(
            f"Largest absolute variance: <b>{largest['pl_account']} - {pdf_text(largest['description'])}</b> "
            f"at <b>{pdf_money(largest[variance_col])}</b>."
        )

    observations.append(
        f"Selected account status: <b>{status}</b>, with variance of "
        f"<b>{pdf_money(row[variance_col])}</b>."
    )

    if not g.empty:
        biggest = g.loc[g["amount"].abs().idxmax()]
        observations.append(
            f"Largest GL transaction for the selected account: "
            f"<b>{pdf_money(biggest['amount'])}</b> - "
            f"{pdf_text(biggest.get('assignment', ''))} - "
            f"{pdf_text(biggest.get('text', ''))}."
        )

    if bmatch is not None and not bmatch.empty:
        observations.append(
            f"The selected account has <b>{len(bmatch)}</b> Opex Budget reference item(s). "
            "This reference should be reviewed together with GL Assignment and Text / Description."
        )
    else:
        observations.append(
            "The selected account has no Opex Budget reference mapping and should be reviewed if mapping is expected."
        )

    for obs in observations:
        story.append(Paragraph(f"• {obs}", normal_style))
        story.append(Spacer(1, 3))

    story.append(Spacer(1, 10))
    story.append(
        Paragraph(
            "Note: Variance status is determined by PL Actual versus PL Budget. "
            "Opex Budget data is a supporting reference and does not independently determine Above/Below Budget status.",
            small_style,
        )
    )

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
# PDF Summary Download
# -----------------------------
st.divider()
st.subheader("📄 PDF Summary Analysis")
st.caption(
    "Generate an executive summary containing KPI, top variance contributors, "
    "selected account investigation, GL contributors, Opex Budget reference and key observations."
)

pdf_bytes = build_pdf_summary(
    prop=prop,
    view=view,
    usd_idr=usd_idr,
    total_actual=total_actual,
    total_budget=total_budget,
    total_variance=total_variance,
    drivers=drivers,
    data=data,
    selected=selected,
    row=row,
    g=g,
    bmatch=bmatch,
    actual_col=actual_col,
    budget_col=budget_col,
    variance_col=variance_col,
)

st.download_button(
    label="📥 Download PDF Summary Analysis",
    data=pdf_bytes,
    file_name=f"IT_OPEX_Variance_Analysis_{prop}_{view.replace(' ', '_')}.pdf",
    mime="application/pdf",
    use_container_width=True,
)
