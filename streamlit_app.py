# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from pathlib import Path
from datetime import datetime
from zipfile import ZipFile, ZIP_DEFLATED

# =========================
# App Meta
# =========================
st.set_page_config(page_title="Ferguson Weekly Report Builder", layout="wide")

REFER_STORE_PATH = Path("refer_store.csv")

# =========================
# Helpers
# =========================
def persist_refer_df(df: pd.DataFrame) -> None:
    """Persist the 2-column refer mapping to disk."""
    df = df.iloc[:, :2].copy()
    df.columns = ["ErrorText", "Theme"]
    df.to_csv(REFER_STORE_PATH, index=False)

def load_current_refer_df():
    """
    Load the current main refer from disk.
    Returns (df or None, status_text).
    """
    if REFER_STORE_PATH.exists():
        df = pd.read_csv(REFER_STORE_PATH).iloc[:, :2]
        df.columns = ["ErrorText", "Theme"]
        ts = datetime.fromtimestamp(REFER_STORE_PATH.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        return df, f"On disk (last updated: {ts})"
    return None, "Not set — please upload a Refer CSV to initialize."

def _norm_series(s: pd.Series) -> pd.Series:
    # Normalize to reduce spurious mismatches for validation
    return (
        s.astype(str)
         .str.replace(r"\s+", " ", regex=True)  # collapse multi-space
         .str.replace("\u00A0", " ")            # NBSP to space
         .str.strip()
    )

def validate_new_refer(current_refer: pd.DataFrame, new_refer: pd.DataFrame):
    """
    New refer must contain ALL existing ErrorText rows with IDENTICAL themes.
    Returns (missing_count, mismatched_count, diff_df) for debugging.
    """
    cur = current_refer.iloc[:, :2].copy()
    cur.columns = ["ErrorText", "Theme_old"]
    new = new_refer.iloc[:, :2].copy()
    new.columns = ["ErrorText", "Theme_new"]

    cur["ErrorText"] = _norm_series(cur["ErrorText"]); cur["Theme_old"] = _norm_series(cur["Theme_old"])
    new["ErrorText"] = _norm_series(new["ErrorText"]); new["Theme_new"] = _norm_series(new["Theme_new"])

    left = cur.merge(new, on="ErrorText", how="left")
    missing_mask = left["Theme_new"].isna()
    mismatched_mask = (~missing_mask) & (left["Theme_old"].astype(str) != left["Theme_new"].astype(str))

    diff_df = left.loc[missing_mask | mismatched_mask, :].copy()
    return int(missing_mask.sum()), int(mismatched_mask.sum()), diff_df

def classify_theme_freeform(msg: str) -> str:
    """Rule-based classifier for NotExact gaps (best-effort)."""
    if not isinstance(msg, str) or not msg.strip():
        return ""
    s = msg.lower()
    if ("not in available pickup dates" in s) or ("available pickup dates" in s) or ("appointment" in s) or ("pickup window" in s) or ("delivery window" in s):
        return "Appointment Window / Scheduling"
    if ("phone" in s) or ("contact number" in s) or ("call" in s and "contact" in s):
        return "Origin/Destination Phone Number Missing"
    if any(k in s for k in ["address", "zipcode", "zip code", "postal", "city", "state", "location invalid", "not serviceable"]):
        return "Origin/Destination Details"
    if any(k in s for k in ["unauthorized", "forbidden", "authentication", "authorization", "invalid api key", "token"]):
        return "Authentication/Authorization"
    if any(k in s for k in ["timeout", "timed out", "gateway timeout", "service unavailable", "internal server error", "bad gateway", " 500"]) or "vendor dispatch service" in s:
        return "Technical/Server Error"
    if any(k in s for k in ["no capacity", "capacity full", "over capacity", "not available for pickup"]):
        return "Carrier Capacity"
    if any(k in s for k in ["missing", "required", "invalid", "format", "cannot be blank", "must provide"]):
        return "Data Validation"
    if any(k in s for k in ["not found", "does not exist", "unknown", "reference", " id ", "po ", "bol "]):
        return "Reference/ID Issue"
    if any(k in s for k in ["weight", "dimension", "length", "width", "height", "nmfc", "class "]):
        return "Freight Details"
    if any(k in s for k in ["duplicate", "already exists", "already scheduled"]):
        return "Duplicate / Already Scheduled"
    if any(k in s for k in ["not in service area", "lane not serviced", "no service in"]):
        return "Coverage / Service Area"
    return "Other / Needs Review"

def build_sheet1_dispatch(current_df: pd.DataFrame, last_df: pd.DataFrame) -> pd.DataFrame:
    cur = current_df[["scac", "RequestCount", "SuccessPercent"]].copy()
    last = last_df[["scac", "SuccessPercent"]].copy().rename(columns={"SuccessPercent": "SuccessPercent_last_week"})
    out = cur.merge(last, on="scac", how="left")
    out["Change (Week over Week)"] = np.where(out["SuccessPercent_last_week"].notna(),
                                              out["SuccessPercent"] - out["SuccessPercent_last_week"], "")
    out = out.drop(columns=["SuccessPercent_last_week"])
    out["p44 Benchmark - Low"] = ""
    out["p44 Benchmark - High"] = ""
    out["p44 Benchmark - Best in Class"] = ""
    return out[[
        "scac", "RequestCount", "SuccessPercent", "Change (Week over Week)",
        "p44 Benchmark - Low", "p44 Benchmark - High", "p44 Benchmark - Best in Class"
    ]]

def build_sheet2_empty() -> pd.DataFrame:
    return pd.DataFrame(columns=["(leave empty, you'll paste here)"])

def build_sheet3_server_errors(server_df: pd.DataFrame, refer_df: pd.DataFrame) -> pd.DataFrame:
    base_cols = server_df.columns.tolist()[:3]
    df = server_df[base_cols].copy()
    if len(base_cols) < 3:
        while len(df.columns) < 3:
            df[f"col_{len(df.columns)+1}"] = ""
        base_cols = df.columns.tolist()[:3]
    count_col = base_cols[2]
    df[count_col] = pd.to_numeric(df[count_col], errors="coerce").fillna(0)

    total_errors = df[count_col].sum()
    df["% of Total Errors"] = np.where(total_errors > 0, df[count_col] / total_errors, 0.0)

    # Exact Theme lookup (server errors: use column B)
    refer_map = dict(zip(refer_df.iloc[:, 0].astype(str).str.strip(), refer_df.iloc[:, 1].astype(str).str.strip()))
    text_col = base_cols[1] if len(base_cols) > 1 else base_cols[0]
    keys = df[text_col].astype(str).str.strip()

    exact_theme = keys.map(refer_map).replace({np.nan: ""})
    df["Theme"] = exact_theme
    df["MatchType"] = np.where(df["Theme"].astype(str).str.len() > 0, "Exact", "NotExact")

    # Smart classification for NotExact
    mask = df["MatchType"] == "NotExact"
    df.loc[mask, "Theme"] = df.loc[mask, text_col].astype(str).map(classify_theme_freeform)

    df["Needs Review"] = np.where(df["MatchType"] == "NotExact", "Yes", "")

    return df[base_cols + ["% of Total Errors", "Theme", "MatchType", "Needs Review"]]

def to_excel_openpyxl(sheets: dict[str, pd.DataFrame]) -> bytes:
    """Write Excel using openpyxl ONLY (no xlsxwriter dependency)."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, sheet_df in sheets.items():
            sheet_df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    return output.getvalue()

def to_zip_csvs(sheets: dict[str, pd.DataFrame], include_refer: bool = False) -> bytes:
    """Create a ZIP of CSVs. By default, excludes Refer to avoid leakage."""
    buf = BytesIO()
    with ZipFile(buf, mode="w", compression=ZIP_DEFLATED) as z:
        for name, df in sheets.items():
            if name == "Refer" and not include_refer:
                continue
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            z.writestr(f"{name}.csv", csv_bytes)
    buf.seek(0)
    return buf.getvalue()

# =========================
# UI
# =========================
st.title("Ferguson Weekly Report Builder")
st.caption("Download CSVs (zip) or create one Excel workbook. Uses openpyxl only — no xlsxwriter needed.")

# Sidebar: Refer control (NO download of refer)
st.sidebar.header("Refer Sheet Control (Theme Mapping)")
current_refer_df, refer_status = load_current_refer_df()
st.sidebar.markdown(f"**Refer status:** {refer_status}")

refer_upload = st.sidebar.file_uploader(
    "Upload Refer CSV (two columns: ErrorText, Theme)",
    type=["csv"],
    key="refer_upload"
)

# First-time initialization: if no refer exists, require an upload and auto-persist it
if current_refer_df is None:
    if refer_upload is None:
        st.warning("No Refer is set yet. Please upload your Refer CSV in the sidebar to initialize the app.")
        st.stop()
    try:
        uploaded_refer = pd.read_csv(refer_upload).iloc[:, :2]
        uploaded_refer.columns = ["ErrorText", "Theme"]
        persist_refer_df(uploaded_refer)
        current_refer_df, refer_status = load_current_refer_df()
        st.success("Refer initialized and saved as main.")
    except Exception as e:
        st.error(f"Failed to read/persist uploaded Refer: {e}")
        st.stop()
else:
    # If a refer already exists, allow replacing it with a superset that matches all current rows/themes.
    if refer_upload is not None:
        try:
            uploaded_refer = pd.read_csv(refer_upload).iloc[:, :2]
            uploaded_refer.columns = ["ErrorText", "Theme"]
            missing, mismatched, diff = validate_new_refer(current_refer_df, uploaded_refer)
            if missing == 0 and mismatched == 0:
                st.sidebar.success("New Refer contains all current mappings. Click below to persist it.")
                if st.sidebar.button("Persist NEW Refer as Main"):
                    persist_refer_df(uploaded_refer)
                    current_refer_df, refer_status = load_current_refer_df()
                    st.sidebar.success("New Refer saved as main.")
            else:
                st.sidebar.error(
                    f"Refer validation failed: missing={missing}, mismatched={mismatched}. "
                    "The new refer must include all current rows with identical themes."
                )
                if not diff.empty:
                    st.sidebar.download_button(
                        "Download validation_diff.csv",
                        diff.to_csv(index=False).encode("utf-8"),
                        file_name="validation_diff.csv",
                        mime="text/csv",
                    )
        except Exception as e:
            st.sidebar.error(f"Failed to read/validate uploaded Refer: {e}")

# Main inputs
st.subheader("1) Upload CSVs")
col1, col2, col3 = st.columns(3)
with col1:
    current_week_file = st.file_uploader("Current Week: Dispatch Success Rate by Carrier SCAC", type=["csv"], key="cur")
with col2:
    last_week_file = st.file_uploader("Last Week: Dispatch Success Rate by Carrier SCAC", type=["csv"], key="last")
with col3:
    server_errors_file = st.file_uploader("Server Tender Response Errors", type=["csv"], key="serr")

output_mode = st.radio("2) Choose output", ["ZIP of CSVs (recommended)", "One Excel workbook (openpyxl)"], index=0)
include_refer_in_excel = st.checkbox("Include Refer sheet in Excel", value=False)

run = st.button("Build Output")

if run:
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

    sheets = {
        "Dispatch_Success_Rate_By_Carrie": sheet1,
        "Shipment_Level_Errors": sheet2,
        "Server_Tender_Errors": sheet3,
        "Refer": current_refer_df,  # only included if explicitly requested
    }

    if output_mode == "ZIP of CSVs (recommended)":
        zip_bytes = to_zip_csvs(sheets, include_refer=False)  # keep Refer private by default
        st.success("ZIP created.")
        st.download_button(
            label="Download ZIP of CSVs",
            data=zip_bytes,
            file_name=f"Ferguson_Weekly_Report_CSVs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            mime="application/zip",
        )
    else:
        try:
            # Compose workbook with/without Refer
            sheets_for_excel = {
                "Dispatch_Success_Rate_By_Carrie": sheet1,
                "Shipment_Level_Errors": sheet2,
                "Server_Tender_Errors": sheet3,
            }
            if include_refer_in_excel:
                sheets_for_excel["Refer"] = current_refer_df

            xlsx_bytes = to_excel_openpyxl(sheets_for_excel)
            st.success("Workbook built!")
            st.download_button(
                label="Download Excel Workbook",
                data=xlsx_bytes,
                file_name=f"Ferguson_Weekly_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except Exception as e:
            st.error(f"Excel build failed with openpyxl: {e}")
            zip_bytes = to_zip_csvs(sheets, include_refer=False)
            st.info("As a fallback, here's a ZIP of the CSVs.")
            st.download_button(
                label="Download ZIP of CSVs",
                data=zip_bytes,
                file_name=f"Ferguson_Weekly_Report_CSVs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip",
            )

    # Quick previews
    with st.expander("Preview: Dispatch_Success_Rate_By_Carrie"):
        st.dataframe(sheet1, use_container_width=True)
    with st.expander("Preview: Server_Tender_Errors"):
        st.dataframe(sheet3, use_container_width=True)

# =========================
# Optional MERGE utility (upload CSVs -> one Excel)
# =========================
st.divider()
st.subheader("Optional: Merge previously-downloaded CSVs into a single Excel")
mcol1, mcol2, mcol3, mcol4 = st.columns(4)
with mcol1:
    csv1 = st.file_uploader("Dispatch_Success_Rate_By_Carrie.csv", type=["csv"], key="m1")
with mcol2:
    csv2 = st.file_uploader("Shipment_Level_Errors.csv", type=["csv"], key="m2")
with mcol3:
    csv3 = st.file_uploader("Server_Tender_Errors.csv", type=["csv"], key="m3")
with mcol4:
    refer_csv_optional = st.file_uploader("Optional Refer.csv", type=["csv"], key="m4")

merge_include_refer = st.checkbox("Include Refer CSV in merged Excel", value=False, key="merge_ref")
if st.button("Create Excel from CSVs"):
    if not (csv1 and csv2 and csv3):
        st.error("Please upload the three CSVs.")
        st.stop()
    try:
        s1 = pd.read_csv(csv1)
        s2 = pd.read_csv(csv2)
        s3 = pd.read_csv(csv3)
        sheets_pkg = {
            "Dispatch_Success_Rate_By_Carrie": s1,
            "Shipment_Level_Errors": s2,
            "Server_Tender_Errors": s3,
        }
        if merge_include_refer and refer_csv_optional:
            ref_df = pd.read_csv(refer_csv_optional).iloc[:, :2]
            ref_df.columns = ["ErrorText", "Theme"]
            sheets_pkg["Refer"] = ref_df
        xlsx_bytes2 = to_excel_openpyxl(sheets_pkg)
        st.success("Merged Excel created.")
        st.download_button(
            label="Download Merged Excel",
            data=xlsx_bytes2,
            file_name=f"Ferguson_Weekly_Report_MERGED_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except Exception as e:
        st.error(f"Merge failed: {e}")
