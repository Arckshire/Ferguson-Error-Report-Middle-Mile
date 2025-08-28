import streamlit as st
import pandas as pd
import xlsxwriter
import numpy as np
from io import BytesIO, StringIO
from pathlib import Path
from datetime import datetime

# =========================
# App Metadata / Settings
# =========================
st.set_page_config(page_title="Ferguson Weekly Report Builder", layout="wide")

# -------------------------
# Defaults: embedded refer (fallback)
# -------------------------
DEFAULT_REFER_CSV = """ErrorText,Theme
# Put exact error text in first column and your theme in second column
# Example rows (delete or modify as you wish):
P44(The vendor dispatch service has returned an error.)VENDOR( 08222025 is not in avaliable pickup dates List 082525082625082725082825.  Please contact dispatch 800-942-9909 to expedite your pick up request.),Appointment Window / Scheduling
"""

REFER_STORE_PATH = Path("refer_store.csv")

# =========================
# Helper functions
# =========================

def load_current_refer_df() -> tuple[pd.DataFrame, str]:
    """Load the current main refer (priority: disk -> embedded default)."""
    if REFER_STORE_PATH.exists():
        df = pd.read_csv(REFER_STORE_PATH).iloc[:, :2]
        df.columns = ["ErrorText", "Theme"]
        ts = datetime.fromtimestamp(REFER_STORE_PATH.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        return df, f"On disk (last updated: {ts})"
    df = pd.read_csv(StringIO(DEFAULT_REFER_CSV), comment="#").iloc[:, :2]
    df.columns = ["ErrorText", "Theme"]
    return df, "Embedded default"


def persist_refer_df(df: pd.DataFrame) -> None:
    df = df.iloc[:, :2].copy()
    df.columns = ["ErrorText", "Theme"]
    df.to_csv(REFER_STORE_PATH, index=False)


def validate_new_refer(current_refer: pd.DataFrame, new_refer: pd.DataFrame) -> tuple[int, int]:
    """Require that the new refer contains ALL existing mappings with identical themes.
    Return (missing_count, mismatched_count)."""
    cur = current_refer.iloc[:, :2].copy()
    cur.columns = ["ErrorText", "Theme_old"]
    new = new_refer.iloc[:, :2].copy()
    new.columns = ["ErrorText", "Theme_new"]
    left = cur.merge(new, on="ErrorText", how="left")
    missing = left["Theme_new"].isna().sum()
    mismatched = (left["Theme_new"].notna() & (left["Theme_old"].astype(str) != left["Theme_new"].astype(str))).sum()
    return int(missing), int(mismatched)


def classify_theme_freeform(msg: str) -> str:
    """Rule-based classifier for non-exact cases (best-effort)."""
    if not isinstance(msg, str) or not msg.strip():
        return ""
    s = msg.lower()

    # Appointment / scheduling patterns
    if ("not in available pickup dates" in s) or ("available pickup dates" in s) or ("appointment" in s) or ("pickup window" in s) or ("delivery window" in s):
        return "Appointment Window / Scheduling"

    # Phone / contact
    if ("phone" in s) or ("contact number" in s) or ("call" in s and "contact" in s):
        return "Origin/Destination Phone Number Missing"

    # Address / location details
    if any(k in s for k in ["address", "zipcode", "zip code", "postal", "city", "state", "location invalid", "not serviceable"]):
        return "Origin/Destination Details"

    # Auth / permission
    if any(k in s for k in ["unauthorized", "forbidden", "authentication", "authorization", "invalid api key", "token"]):
        return "Authentication/Authorization"

    # Technical / server
    if any(k in s for k in ["timeout", "timed out", "gateway timeout", "service unavailable", "internal server error", "bad gateway", " 500"]):
        return "Technical/Server Error"
    if "vendor dispatch service" in s:
        return "Technical/Server Error"

    # Capacity
    if any(k in s for k in ["no capacity", "capacity full", "over capacity", "not available for pickup"]):
        return "Carrier Capacity"

    # Data validation
    if any(k in s for k in ["missing", "required", "invalid", "format", "cannot be blank", "must provide"]):
        return "Data Validation"

    # Reference / ID
    if any(k in s for k in ["not found", "does not exist", "unknown", "reference", " id ", "po ", "bol "]):
        return "Reference/ID Issue"

    # Freight details
    if any(k in s for k in ["weight", "dimension", "length", "width", "height", "nmfc", "class "]):
        return "Freight Details"

    # Duplicate / already scheduled
    if any(k in s for k in ["duplicate", "already exists", "already scheduled"]):
        return "Duplicate / Already Scheduled"

    # Coverage / service area
    if any(k in s for k in ["not in service area", "lane not serviced", "no service in"]):
        return "Coverage / Service Area"

    return "Other / Needs Review"


def build_sheet1_dispatch(current_df: pd.DataFrame, last_df: pd.DataFrame) -> pd.DataFrame:
    # Keep only columns: scac, RequestCount, SuccessPercent
    cur = current_df[["scac", "RequestCount", "SuccessPercent"]].copy()
    last = last_df[["scac", "SuccessPercent"]].copy().rename(columns={"SuccessPercent": "SuccessPercent_last_week"})

    out = cur.merge(last, on="scac", how="left")
    out["Change (Week over Week)"] = np.where(out["SuccessPercent_last_week"].notna(),
                                              out["SuccessPercent"] - out["SuccessPercent_last_week"],
                                              "")
    out = out.drop(columns=["SuccessPercent_last_week"])
    out["p44 Benchmark - Low"] = ""
    out["p44 Benchmark - High"] = ""
    out["p44 Benchmark - Best in Class"] = ""
    out = out[[
        "scac", "RequestCount", "SuccessPercent", "Change (Week over Week)",
        "p44 Benchmark - Low", "p44 Benchmark - High", "p44 Benchmark - Best in Class"
    ]]
    return out


def build_sheet2_empty() -> pd.DataFrame:
    return pd.DataFrame(columns=["(leave empty, you'll paste here)"])


def build_sheet3_server_errors(server_df: pd.DataFrame, refer_df: pd.DataFrame) -> pd.DataFrame:
    # Expect first three columns A,B,C where C is a count
    base_cols = server_df.columns.tolist()[:3]
    df = server_df[base_cols].copy()

    # Ensure counts numeric
    if len(base_cols) < 3:
        while len(df.columns) < 3:
            df[f"col_{len(df.columns)+1}"] = ""
        base_cols = df.columns.tolist()[:3]
    count_col = base_cols[2]
    df[count_col] = pd.to_numeric(df[count_col], errors="coerce").fillna(0)

    # % of Total Errors
    total_errors = df[count_col].sum()
    df["% of Total Errors"] = np.where(total_errors > 0, df[count_col] / total_errors, 0.0)

    # Exact match from refer (A:B -> ErrorText, Theme). Lookup key is column B of server errors
    refer_map = dict(zip(refer_df.iloc[:, 0].astype(str).str.strip(), refer_df.iloc[:, 1].astype(str).str.strip()))

    text_col = base_cols[1] if len(base_cols) > 1 else base_cols[0]
    keys = df[text_col].astype(str).str.strip()

    exact_theme = keys.map(refer_map).replace({np.nan: ""})
    df["Theme"] = exact_theme
    df["MatchType"] = np.where(df["Theme"].astype(str).str.len() > 0, "Exact", "NotExact")

    # Smart fill for NotExact
    mask = df["MatchType"] == "NotExact"
    df.loc[mask, "Theme"] = df.loc[mask, text_col].astype(str).map(classify_theme_freeform)

    # Needs Review flag for NotExact
    df["Needs Review"] = np.where(df["MatchType"] == "NotExact", "Yes", "")

    # Order
    df = df[base_cols + ["% of Total Errors", "Theme", "MatchType", "Needs Review"]]

    return df


def to_excel(sheets: dict[str, pd.DataFrame]) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for sheet_name, sheet_df in sheets.items():
            sheet_df.to_excel(writer, sheet_name=sheet_name[:31], index=False)

        # Formatting for Server_Tender_Errors if present
        if "Server_Tender_Errors" in sheets:
            wb = writer.book
            ws = writer.sheets.get("Server_Tender_Errors")
            if ws is not None:
                bold = wb.add_format({'bold': True})
                ws.set_row(0, None, bold)
                # percentage format
                pct_col_idx = list(sheets["Server_Tender_Errors"].columns).index("% of Total Errors")
                pct_fmt = wb.add_format({'num_format': '0.00%'})
                ws.set_column(pct_col_idx, pct_col_idx, 14, pct_fmt)
                # highlight Needs Review rows
                review_col_idx = list(sheets["Server_Tender_Errors"].columns).index("Needs Review")
                ws.conditional_format(1, 0, len(sheets["Server_Tender_Errors"]), len(sheets["Server_Tender_Errors"].columns)-1, {
                    'type': 'formula',
                    'criteria': f'=${chr(65+review_col_idx)}2="Yes"',
                    'format': wb.add_format({'bg_color': '#FFF2CC'})
                })
    return output.getvalue()

# =========================
# UI
# =========================
st.title("Ferguson Weekly Report Builder")
st.caption("Build the 3-sheet weekly workbook + Refer sheet, with Exact/NotExact tagging and smart themes.")

# Sidebar: Refer control (load current, optionally validate & replace)
st.sidebar.header("Refer Sheet Control (Theme Mapping)")
current_refer_df, refer_source = load_current_refer_df()
st.sidebar.markdown(f"**Current refer source:** {refer_source}")
if REFER_STORE_PATH.exists():
    ts = datetime.fromtimestamp(REFER_STORE_PATH.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    st.sidebar.info(f"Main refer last updated: {ts}")
else:
    st.sidebar.info("Main refer not yet saved on disk — using embedded default.")

refer_upload = st.sidebar.file_uploader("Optional: Upload a NEW refer CSV (two columns: ErrorText, Theme)", type=["csv"], key="refer_upload")

if refer_upload is not None:
    try:
        uploaded_refer = pd.read_csv(refer_upload).iloc[:, :2]
        uploaded_refer.columns = ["ErrorText", "Theme"]
        missing, mismatched = validate_new_refer(current_refer_df, uploaded_refer)
        if missing == 0 and mismatched == 0:
            st.sidebar.success("Refer validation passed: new file contains all current mappings (you may have added more — that's fine).")
            if st.sidebar.button("Make uploaded refer the MAIN refer (persist)"):
                persist_refer_df(uploaded_refer)
                current_refer_df = uploaded_refer
                st.sidebar.success("Uploaded refer saved as main. It will be used by default next time.")
        else:
            st.sidebar.error(f"Refer validation failed: missing={missing}, mismatched={mismatched}. The new refer must include all current rows with identical themes.")
    except Exception as e:
        st.sidebar.error(f"Failed to read/validate uploaded refer: {e}")

# Download current refer
st.sidebar.download_button(
    label="Download CURRENT refer.csv",
    data=current_refer_df.to_csv(index=False).encode("utf-8"),
    file_name="refer.csv",
    mime="text/csv",
)

# Main inputs
st.subheader("1) Upload CSVs")
col1, col2, col3 = st.columns(3)
with col1:
    current_week_file = st.file_uploader("Current Week: Dispatch Success Rate by Carrier SCAC", type=["csv"], key="cur")
with col2:
    last_week_file = st.file_uploader("Last Week: Dispatch Success Rate by Carrier SCAC", type=["csv"], key="last")
with col3:
    server_errors_file = st.file_uploader("Server Tender Response Errors", type=["csv"], key="serr")

run = st.button("Build Workbook")

if run:
    # Basic validations
    if not (current_week_file and last_week_file and server_errors_file):
        st.error("Please upload all three CSVs to proceed.")
        st.stop()

    # Read inputs
    cur = pd.read_csv(current_week_file)
    last = pd.read_csv(last_week_file)
    serr = pd.read_csv(server_errors_file)

    # Build sheets
    try:
        sheet1 = build_sheet1_dispatch(cur, last)
    except Exception as e:
        st.error(f"Failed to build Sheet 1 (Dispatch Success Rate): {e}")
        st.stop()

    sheet2 = build_sheet2_empty()

    try:
        sheet3 = build_sheet3_server_errors(serr, current_refer_df)
    except Exception as e:
        st.error(f"Failed to build Sheet 3 (Server Tender Errors): {e}")
        st.stop()

    # Compose workbook — ONLY THREE operational sheets + Refer at the end
    sheets = {
        "Dispatch_Success_Rate_By_Carrie": sheet1,
        "Shipment_Level_Errors": sheet2,
        "Server_Tender_Errors": sheet3,
        "Refer": current_refer_df,
    }

    xlsx_bytes = to_excel(sheets)

    st.success("Workbook built!")
    st.download_button(
        label="Download Excel Workbook",
        data=xlsx_bytes,
        file_name=f"Ferguson_Weekly_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    # Show quick previews
    with st.expander("Preview: Dispatch_Success_Rate_By_Carrie"):
        st.dataframe(sheet1, use_container_width=True)
    with st.expander("Preview: Server_Tender_Errors"):
        st.dataframe(sheet3, use_container_width=True)

else:
    st.markdown(
        """
        **How it works**
        1. Upload **current week** and **last week** Dispatch Success CSVs, plus **Server Tender Errors** CSV.
        2. Use the **Refer** mapping from disk (if saved before) or the **embedded default**. Optionally upload a NEW refer and, if it passes validation (must include all current mappings with identical themes), set it as the main refer.
        3. Click **Build Workbook** to download an Excel with 4 sheets:
           - **Dispatch_Success_Rate_By_Carrie** (WoW change; benchmarks blank)
           - **Shipment_Level_Errors** (empty placeholder)
           - **Server_Tender_Errors** (% of total, Theme, MatchType, Needs Review)
           - **Refer** (the mapping used in this run)
        4. If you add new rows to your Refer, upload it and click **Make uploaded refer the MAIN refer** (only accepted if it contains all prior mappings exactly). Next run will use the latest saved refer.
        """
    )

