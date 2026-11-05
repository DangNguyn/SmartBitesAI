from pathlib import Path
import copy
import hashlib
import io
import json
import warnings
import base64

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import PowerTransformer

# Nạp các mô hình bổ sung nếu môi trường đã cài đặt
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None

try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None

warnings.filterwarnings("ignore", category=UserWarning)

# =========================================================
# SMARTBITES AI — DEMO DASHBOARD
# =========================================================
st.set_page_config(
    page_title="SmartBites AI",
    page_icon="🍜",
    layout="wide",
    initial_sidebar_state="expanded",
)

BACKGROUND = "#F3E9DF"       # Kem ấm
WHITE = "#FFFCF8"            # Trắng ấm
NAVY = "#1C3829"              # Xanh lá đậm chuẩn logo SmartBites
NAVY_DARK = "#12261C"         # Xanh lá đậm khi hover
TEAL = "#5D856C"              # Xanh sage chuẩn chữ AI
TEAL_LIGHT = "#E2E9DF"        # Xanh sage nhạt
SLATE = "#9C938A"             # Xám trung tính
AMBER = "#C58D6D"             # Nâu cam
RED = "#A94F3D"               # Đỏ đất cảnh báo
BLACK = "#302B27"             # Nâu đen
TEXT_MUTED = "#737865"        # Xám nâu chuẩn subtitle logo
BORDER = "#E8DCD0"             # Viền kem
ACTIVE_MENU = "#F2C6A8"        # Cam đào nhạt

CHART_PALETTE = [
    "#1C3829", "#5D856C", "#EFD0B8", "#563426", "#B7C6AE",
    "#C58D6D", "#9DAD99", "#D8B9A2", "#705044", "#D6DCCF",
]

DATA_DIR = Path("data")
DEFAULT_DATA = DATA_DIR / "train_final.csv"
HISTORY_DATA = DATA_DIR / "history.csv"
TEST_DATA = DATA_DIR / "test.csv"
IF_NO_TEST_FILE = Path("test.csv")

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800;900&display=swap');

    :root {{
        --sb-navy: {NAVY};
        --sb-navy-dark: {NAVY_DARK};
        --sb-teal: {TEAL};
        --sb-bg: {BACKGROUND};
        --sb-border: {BORDER};
        --sb-muted: {TEXT_MUTED};
    }}
    .stApp {{
        background:
            radial-gradient(circle at 92% 2%, rgba(28,56,41,.07), transparent 24rem),
            {BACKGROUND};
        color: {BLACK};
    }}
    .main .block-container {{
        max-width: 1780px;
        padding: 1.7rem 2.2rem 3.5rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
        background: #FFFCF8;
        border: 1px solid #E8DCD0;
        border-radius: 30px;
        box-shadow: 0 12px 36px rgba(86,52,38,.07);
    }}
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp p, .stApp label {{
        color: {BLACK};
    }}
    h1, h2, h3 {{ letter-spacing: -.035em; }}
    h2 {{ margin-top: .6rem !important; font-size: 1.9rem !important; font-weight: 850 !important; }}
    h3 {{ font-size: 1.42rem !important; font-weight: 800 !important; }}
    .chart-section-title {{ text-align:left; color:{NAVY} !important; font-size:1.22rem; font-weight:850; line-height:1.35; min-height:3.1rem; display:flex; align-items:center; justify-content:flex-start; padding:0 .35rem .35rem; }}
    [data-testid="stDataFrame"] {{ color:#000000 !important; }}
    [data-testid="stDataFrame"] * {{ color:#000000 !important; }}
    .stApp h2, .stApp h3 {{ color: {NAVY} !important; }}
    hr {{ border: none; height: 1px; background: {BORDER}; margin: 1.6rem 0; }}
    [data-testid="stSidebar"] {{
        background: #DCE6D8 !important;
        border-right: 1px solid #C7D5C2;
        box-shadow: 5px 0 24px rgba(28,56,41,.08);
    }}
    .sidebar-subtitle {{
        text-align: center;
        color: {TEXT_MUTED} !important;
        font-size: .88rem;
        line-height: 1.4;
        margin-bottom: .6rem;
    }}
    .sidebar-section-title {{
        color: #1C3829 !important;
        font-size: .92rem;
        font-weight: 900;
        font-family: 'Montserrat', sans-serif !important;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin: 1.25rem 15px .7rem;
    }}
    [data-testid="stSidebar"] .stButton > button {{
        width: 100%;
        min-height: 48px;
        border-radius: 12px;
        border: 1px solid transparent;
        background: transparent;
        display: flex !important;
        text-align: left !important;
        justify-content: flex-start !important;
        padding-left: 15px;
        padding-right: 12px;
        gap: 12px;
        color: #81766E !important;
        font-weight: 600;
        font-size: 1.02rem;
        transition: background .18s ease, border-color .18s ease;
    }}
    [data-testid="stSidebar"] .stButton > button > div {{
        width: 100%;
        justify-content: flex-start !important;
        text-align: left !important;
    }}
    [data-testid="stSidebar"] .stButton > button p {{
        margin: 0;
        text-align: left !important;
    }}
    [data-testid="stSidebar"] .stButton > button:hover {{
        background: #F2C6A8;
        border-color: #E8B694;
        color: {NAVY};
    }}
    [data-testid="stSidebar"] .stButton > button[kind="primary"] {{
        background: {ACTIVE_MENU};
        color: #111111 !important;
        border-color: #D3DFCD;
        font-weight: 800;
        box-shadow: 0 3px 8px rgba(28,56,41,.07);
    }}
    div[data-testid="stMetric"] {{
        background: {WHITE};
        border: 1px solid #E8DCD0;
        border-left: 1px solid #E8DCD0;
        border-radius: 22px;
        padding: 16px 18px;
        min-height: 108px;
        box-shadow: 0 6px 22px rgba(86,52,38,.055);
        transition: transform .18s ease, box-shadow .18s ease;
    }}
    div[data-testid="stMetric"]:hover {{
        transform: translateY(-2px);
        box-shadow: 0 9px 24px rgba(18,59,93,.09);
    }}
    div[data-testid="stMetricLabel"] {{
        color: #81766E !important;
        font-size: .95rem !important;
        font-weight: 550 !important;
    }}
    div[data-testid="stMetricValue"] {{
        color: #1C3829 !important;
        font-size: clamp(1.15rem, 1.5vw, 1.7rem) !important;
        font-weight: 850 !important;
        letter-spacing: -.035em;
        overflow-wrap: anywhere;
    }}
    div[data-testid="stMetricDelta"] {{ font-size: .78rem !important; }}
    .overview-kpi-card {{
        min-height: 108px; padding: 16px 18px; display:flex; align-items:center; gap:14px; background: #FFFCF8;
        border: 1px solid #E8DCD0; border-radius: 22px;
        box-shadow: 0 6px 22px rgba(86,52,38,.055); overflow: hidden;
    }}
    .overview-kpi-top {{ display:flex; align-items:center; justify-content:flex-start; flex:0 0 auto; }}
    .overview-kpi-icon {{ width:44px; height:44px; flex:0 0 44px; border-radius:50%; display:flex; align-items:center; justify-content:center; }}
    .overview-kpi-icon svg {{ width:21px; height:21px; }}
    .overview-kpi-content {{ min-width:0; }}
    .overview-kpi-value {{ color:#111827; font-size:1.45rem; line-height:1.18; font-weight:850; letter-spacing:-.035em; overflow-wrap:anywhere; }}
    .overview-kpi-label {{ margin-top:4px; color:#87909D; font-size:.86rem; font-weight:550; }}
    .pricing-kpi-card {{
        min-height:108px; height:100%; padding:17px 20px; display:flex;
        align-items:center; justify-content:space-between; gap:14px;
        background:{WHITE}; border:1px solid {BORDER}; border-radius:22px;
        box-shadow:0 6px 22px rgba(86,52,38,.055); overflow:hidden;
        transition:transform .18s ease, box-shadow .18s ease;
    }}
    .pricing-kpi-card:hover {{ transform:translateY(-2px); box-shadow:0 9px 24px rgba(86,52,38,.10); }}
    .pricing-kpi-content {{ flex:1; min-width:0; text-align:center; }}
    .pricing-kpi-label {{ color:{TEXT_MUTED}; font-size:.9rem; font-weight:650; line-height:1.3; text-align:center; min-height:2.6rem; display:flex; align-items:center; justify-content:center; }}
    .pricing-kpi-value {{ color:{NAVY}; font-size:clamp(1.2rem,1.65vw,1.75rem); font-weight:900; line-height:1.25; margin-top:4px; overflow-wrap:anywhere; text-align:center; }}
    .pricing-kpi-delta {{ display:inline-block; margin-top:7px; padding:4px 9px; border-radius:999px; background:#F8E8E3; color:#A94F3D; font-size:.76rem; font-weight:750; }}
    .pricing-kpi-icon {{ width:48px; height:48px; flex:0 0 48px; display:flex; align-items:center; justify-content:center; border-radius:15px; background:color-mix(in srgb, var(--kpi-accent) 12%, white); color:var(--kpi-accent); font-size:1.55rem; }}
    .main .stButton > button {{
        background: {NAVY};
        color: {WHITE} !important;
        border: 1px solid {NAVY};
        border-radius: 11px;
        font-weight: 750;
        padding: .58rem 1rem;
        box-shadow: 0 3px 8px rgba(18,59,93,.10);
    }}
    .main .stButton > button:hover {{
        background: {NAVY_DARK};
        border-color: {NAVY_DARK};
        color: {WHITE} !important;
    }}
    [data-testid="stDataFrame"] {{
        background: {WHITE};
        border: 1px solid {BORDER};
        border-radius: 13px;
        overflow: hidden;
    }}
    [data-testid="stPlotlyChart"] {{
        background: {WHITE};
        border: 1px solid #E8DCD0;
        border-radius: 24px;
        padding: .6rem .6rem .25rem;
        box-shadow: 0 6px 22px rgba(86,52,38,.05);
        margin-bottom: .8rem;
        overflow: hidden;
    }}
    [data-testid="stPlotlyChart"] > div,
    [data-testid="stPlotlyChart"] iframe {{
        border-radius: 19px !important;
        overflow: hidden !important;
    }}
    [data-testid="stExpander"] {{
        background: {WHITE};
        border: 1px solid #E8DCD0;
        border-radius: 16px;
    }}
    [data-testid="stAlert"] {{ border-radius: 12px; }}
    [data-testid="stSelectbox"] > div > div,
    [data-testid="stMultiSelect"] > div > div {{
        border-radius: 10px;
    }}
    [data-testid="stFileUploader"] {{
        background: rgba(255,252,248,.85);
        border: 1px dashed #CDB8A7;
        border-radius: 14px;
        padding: .5rem;
    }}
    hr {{ border-color: {BORDER}; margin: 1.35rem 0; }}
    [data-testid="stCaptionContainer"] p {{ color: {TEXT_MUTED}; }}
    @media (max-width: 900px) {{
        .main .block-container {{ padding: 1rem 1rem 2rem; }}
    }}

    [data-testid="stDataFrame"] thead th,
    [data-testid="stDataFrame"] th,
    [data-testid="stDataFrame"] th span,
    [data-testid="stDataFrame"] th div {{
        color: #111111 !important;
        font-weight: 800 !important;
        opacity: 1 !important;
        background-color: #F5EFE8 !important;
    }}
    [data-testid="stDataFrame"] thead {{
        background-color: #F5EFE8 !important;
    }}

    [data-testid="stSlider"] {{
        padding-top: 10px !important;
        padding-bottom: 18px !important;
    }}
    [data-testid="stSlider"] [data-baseweb="slider"] {{
        height: 48px !important;
        min-height: 48px !important;
        align-items: center !important;
    }}
    [data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {{
        width: 52px !important;
        height: 52px !important;
        min-width: 52px !important;
        min-height: 52px !important;
        border-width: 5px !important;
        box-shadow: 0 4px 14px rgba(169, 79, 61, .42) !important;
        margin-top: 0 !important;
    }}
    [data-testid="stSlider"] [data-baseweb="slider"] > div:first-child {{
        height: 28px !important;
        border-radius: 999px !important;
    }}
    [data-testid="stSlider"] [data-baseweb="slider"] div[role="presentation"] {{
        height: 28px !important;
        border-radius: 999px !important;
    }}

    [data-testid="stCheckbox"] {{
        padding: 16px 0 !important;
        min-height: 64px !important;
    }}
    [data-testid="stCheckbox"] label {{
        gap: 18px !important;
        align-items: center !important;
        font-size: 1.65rem !important;
        font-weight: 700 !important;
        line-height: 1.5 !important;
        cursor: pointer !important;
    }}
    [data-testid="stCheckbox"] label [data-baseweb="checkbox"] {{
        flex: 0 0 48px !important;
        width: 48px !important;
        height: 48px !important;
        min-width: 48px !important;
        min-height: 48px !important;
    }}
    [data-testid="stCheckbox"] label [data-baseweb="checkbox"] > div {{
        width: 48px !important;
        height: 48px !important;
        min-width: 48px !important;
        min-height: 48px !important;
        border-radius: 11px !important;
        border-width: 3px !important;
    }}
    [data-testid="stCheckbox"] label [data-baseweb="checkbox"] svg {{
        width: 28px !important;
        height: 28px !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Data handling ----------
REQUIRED = {
    "week", "center_id", "meal_id", "checkout_price", "base_price", "num_orders"
}
TEST_REQUIRED = {
    "week", "center_id", "meal_id", "checkout_price", "base_price"
}
FEATURE_CANDIDATES = [
    "week", "center_id", "meal_id", "checkout_price", "base_price",
    "emailer_for_promotion", "homepage_featured", "city_code", "region_code",
    "op_area", "center_type_encoded", "category_encoded", "cuisine_encoded",
    "discount_amount", "discount_percent", "price_ratio", "total_promo",
    "year_num", "week_of_year", "quarter", "is_festive_season",
    "is_summer_season", "promo_x_discount", "price_rel_to_cat_avg",
    "promo_x_op_area",
]

CATEGORY_TRANSLATIONS = {
    "Beverages": "Đồ uống",
    "Biryani": "Cơm Biryani",
    "Desert": "Tráng miệng",
    "Dessert": "Tráng miệng",
    "Extras": "Món ăn kèm",
    "Fish": "Cá",
    "Other Snacks": "Đồ ăn nhẹ",
    "Pasta": "Mì Ý",
    "Pizza": "Pizza",
    "Rice Bowl": "Cơm",
    "Salad": "Salad",
    "Sandwich": "Bánh mì kẹp",
    "Seafood": "Hải sản",
    "Soup": "Súp",
    "Starters": "Món khai vị",
}

def clean_input_frame(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    df.columns = [str(c).strip() for c in df.columns]
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError("Thiếu cột bắt buộc trong file lịch sử: " + ", ".join(sorted(missing)))
    for col in REQUIRED:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in FEATURE_CANDIDATES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=list(REQUIRED))
    df = df[df["week"] >= 1].copy()
    df["week"] = df["week"].astype(int)
    df["num_orders"] = df["num_orders"].clip(lower=0)
    df = df.drop_duplicates(subset=["week", "center_id", "meal_id"], keep="last")
    df = df.sort_values(["week", "center_id", "meal_id"]).reset_index(drop=True)

    if df["checkout_price"].median() < 10000:
        df["checkout_price"] = df["checkout_price"] * 300
        df["base_price"] = df["base_price"] * 300

    df["discount_amount"] = df["base_price"] - df["checkout_price"]
    df["discount_percent"] = np.where(
        df["base_price"].ne(0),
        df["discount_amount"] / df["base_price"] * 100,
        0,
    )
    df["price_ratio"] = np.where(
        df["base_price"].ne(0), df["checkout_price"] / df["base_price"], 1
    )
    if "emailer_for_promotion" not in df:
        df["emailer_for_promotion"] = 0
    if "homepage_featured" not in df:
        df["homepage_featured"] = 0
    df["emailer_for_promotion"] = df["emailer_for_promotion"].fillna(0)
    df["homepage_featured"] = df["homepage_featured"].fillna(0)
    df["total_promo"] = (
        df["emailer_for_promotion"].fillna(0) + df["homepage_featured"].fillna(0)
    )
    if "promo_x_discount" not in df:
        df["promo_x_discount"] = df["total_promo"] * df["discount_percent"]
    if "promo_x_op_area" not in df:
        df["promo_x_op_area"] = df["total_promo"] * df.get("op_area", 0)
    if "price_rel_to_cat_avg" not in df:
        df["price_rel_to_cat_avg"] = df["price_ratio"]
    if "year_num" not in df:
        df["year_num"] = ((df["week"] - 1) // 52) + 1
    if "week_of_year" not in df:
        df["week_of_year"] = ((df["week"] - 1) % 52) + 1
    if "quarter" not in df:
        df["quarter"] = ((df["week_of_year"] - 1) // 13) + 1
    if "is_festive_season" not in df:
        df["is_festive_season"] = 0
    if "is_summer_season" not in df:
        df["is_summer_season"] = 0

    static_cols = [
        c for c in [
            "city_code", "region_code", "op_area", "center_type_encoded",
            "category_encoded", "cuisine_encoded", "is_festive_season",
            "is_summer_season",
        ] if c in df.columns
    ]
    if static_cols:
        df[static_cols] = (
            df.groupby(["center_id", "meal_id"], sort=False)[static_cols]
            .transform(lambda s: s.ffill().bfill())
            .fillna(0)
        )
    return df

def clean_test_frame(frame: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    df.columns = [str(c).strip() for c in df.columns]
    missing = TEST_REQUIRED - set(df.columns)
    if missing:
        raise ValueError("Thiếu cột bắt buộc trong file test: " + ", ".join(sorted(missing)))
    for col in TEST_REQUIRED:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in FEATURE_CANDIDATES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=list(TEST_REQUIRED))
    df["week"] = df["week"].astype(int)

    if df["checkout_price"].median() < 10000:
        df["checkout_price"] = df["checkout_price"] * 300
        df["base_price"] = df["base_price"] * 300

    df["discount_amount"] = df["base_price"] - df["checkout_price"]
    df["discount_percent"] = np.where(
        df["base_price"].ne(0),
        df["discount_amount"] / df["base_price"] * 100,
        0,
    )
    df["price_ratio"] = np.where(
        df["base_price"].ne(0), df["checkout_price"] / df["base_price"], 1
    )
    if "emailer_for_promotion" not in df:
        df["emailer_for_promotion"] = 0
    if "homepage_featured" not in df:
        df["homepage_featured"] = 0
    df["emailer_for_promotion"] = df["emailer_for_promotion"].fillna(0)
    df["homepage_featured"] = df["homepage_featured"].fillna(0)
    df["total_promo"] = (
        df["emailer_for_promotion"].fillna(0) + df["homepage_featured"].fillna(0)
    )
    if "promo_x_discount" not in df:
        df["promo_x_discount"] = df["total_promo"] * df["discount_percent"]
    if "promo_x_op_area" not in df:
        df["promo_x_op_area"] = df["total_promo"] * df.get("op_area", 0)
    if "price_rel_to_cat_avg" not in df:
        df["price_rel_to_cat_avg"] = df["price_ratio"]
    if "year_num" not in df:
        df["year_num"] = ((df["week"] - 1) // 52) + 1
    if "week_of_year" not in df:
        df["week_of_year"] = ((df["week"] - 1) % 52) + 1
    if "quarter" not in df:
        df["quarter"] = ((df["week_of_year"] - 1) // 13) + 1

    meta_cols = [c for c in ["city_code", "region_code", "op_area", "center_type_encoded", "category_encoded", "cuisine_encoded"] if c in history.columns]
    if meta_cols:
        history_meta = history[["center_id", "meal_id"] + meta_cols].drop_duplicates(subset=["center_id", "meal_id"])
        df = df.merge(history_meta, on=["center_id", "meal_id"], how="left")
        for c in meta_cols:
            df[c] = df[c].fillna(0)

    df = df.sort_values(["week", "center_id", "meal_id"]).reset_index(drop=True)
    return df

def load_history() -> pd.DataFrame:
    if HISTORY_DATA.exists():
        source = HISTORY_DATA
    elif DEFAULT_DATA.exists():
        source = DEFAULT_DATA
    else:
        st.error("Không tìm thấy dữ liệu huấn luyện (tuần 1-145). Hãy đặt train_final.csv trong thư mục data/.")
        st.stop()
    try:
        return clean_input_frame(pd.read_csv(source))
    except Exception as exc:
        st.error(f"Không thể đọc dữ liệu huấn luyện: {exc}")
        st.stop()

def load_test_data(history: pd.DataFrame) -> pd.DataFrame:
    if TEST_DATA.exists():
        source = TEST_DATA
    elif IF_NO_TEST_FILE.exists():
        source = IF_NO_TEST_FILE
    else:
        return pd.DataFrame()
    try:
        return clean_test_frame(pd.read_csv(source), history)
    except Exception as exc:
        st.warning(f"Không thể đọc tệp test.csv: {exc}")
        return pd.DataFrame()

def format_vnd(value) -> str:
    value = float(value)
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:,.2f} tỷ ₫"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:,.1f} triệu ₫"
    return f"{value:,.0f} ₫"

def overview_kpi_card(label: str, value: str, _series, accent: str, icon: str) -> None:
    st.markdown(
        f"""
        <div class="overview-kpi-card">
          <div class="overview-kpi-top">
            <div class="overview-kpi-icon" style="background:{accent}22;color:{accent};">{icon}</div>
          </div>
          <div class="overview-kpi-content">
            <div class="overview-kpi-value">{value}</div>
            <div class="overview-kpi-label">{label}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def pricing_kpi_card(label: str, value: str, icon: str, accent: str = "#1C3829", delta: str = "") -> None:
    if delta:
        delta_html = f'<div class="pricing-kpi-delta">{delta}</div>'
    else:
        delta_html = '<div class="pricing-kpi-delta" style="visibility: hidden; opacity: 0;">&nbsp;</div>'

    st.markdown(
        f"""
        <div class="pricing-kpi-card">
          <div class="pricing-kpi-content">
            <div class="pricing-kpi-label">{label}</div>
            <div class="pricing-kpi-value">{value}</div>
            {delta_html}
          </div>
          <div class="pricing-kpi-icon" style="--kpi-accent:{accent};">{icon}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

DISH_NAMES = {
    "Beverages": ["Trà đào cam sả", "Nước ép dưa hấu", "Trà tắc mật ong", "Sinh tố xoài", "Cà phê sữa đá", "Nước chanh dây", "Trà vải hoa nhài", "Sữa bắp non", "Nước ép ổi", "Trà gừng ấm"],
    "Biryani": ["Cơm gà Biryani", "Cơm bò Biryani", "Cơm Biryani rau củ", "Cơm Biryani tôm"],
    "Desert": ["Bánh flan caramel", "Chè khúc bạch", "Bánh mousse chanh dây", "Sữa chua nếp cẩm"],
    "Dessert": ["Bánh flan caramel", "Chè khúc bạch", "Bánh mousse chanh dây", "Sữa chua nếp cẩm"],
    "Extras": ["Khoai tây chiên", "Bắp xào bơ", "Há cảo hấp", "Chả giò giòn", "Phô mai que"],
    "Fish": ["Cá basa sốt cà", "Cá hồi áp chảo", "Cá thu kho tiêu", "Cá diêu hồng chiên"],
    "Other Snacks": ["Gỏi cuốn tôm thịt", "Bánh tráng trộn", "Nem chua rán", "Bắp rang bơ", "Khoai lang lắc"],
    "Pasta": ["Mì Ý sốt bò bằm", "Mì Ý sốt kem nấm", "Mì Ý hải sản", "Mì Ý sốt pesto", "Mì Ý sốt cà chua", "Mì Ý gà nướng", "Mì Ý carbonara", "Mì Ý sốt phô mai"],
    "Pizza": ["Pizza Margherita", "Pizza hải sản", "Pizza bò phô mai", "Pizza gà BBQ", "Pizza xúc xích Ý", "Pizza nấm phô mai", "Pizza Hawaiian", "Pizza pepperoni", "Pizza rau củ nướng", "Pizza thịt nguội"],
    "Rice Bowl": ["Cơm gà nướng mật ong", "Cơm sườn nướng", "Cơm bò lúc lắc", "Cơm cá kho tộ", "Cơm thịt kho trứng", "Cơm gà xối mỡ", "Cơm heo quay", "Cơm tôm rim", "Cơm bò sốt tiêu", "Cơm đậu hũ nấm"],
    "Salad": ["Salad ức gà", "Salad cá ngừ", "Salad bơ trứng", "Salad tôm rau mầm", "Salad trái cây"],
    "Sandwich": ["Bánh mì gà xé", "Bánh mì thịt nướng", "Bánh mì bò phô mai", "Bánh mì trứng ốp la", "Bánh mì cá ngừ", "Bánh mì xíu mại", "Bánh mì chả lụa", "Bánh mì gà BBQ"],
    "Seafood": ["Tôm rim mặn ngọt", "Mực xào sa tế", "Tôm hấp sả", "Nghêu sốt bơ tỏi", "Bạch tuộc nướng"],
    "Soup": ["Súp cua trứng cút", "Súp bí đỏ kem tươi", "Canh chua cá", "Súp gà nấm", "Súp hải sản"],
    "Starters": ["Chả giò tôm thịt", "Gỏi gà bắp cải", "Đậu hũ chiên sả", "Cánh gà chiên nước mắm", "Bò cuộn lá lốt"],
}

def meal_label(meal_id) -> str:
    try:
        mid = int(meal_id)
    except (TypeError, ValueError):
        return f"Món ăn {meal_id}"
    category = MEAL_CATEGORY_MAP.get(mid, "Chưa phân loại") if "MEAL_CATEGORY_MAP" in globals() else "Chưa phân loại"
    category_name = category_label(category) if "category_label" in globals() else str(category)
    category_ids = sorted(int(k) for k, v in MEAL_CATEGORY_MAP.items() if str(v) == str(category)) if "MEAL_CATEGORY_MAP" in globals() else [mid]
    try:
        ordinal = category_ids.index(mid)
    except ValueError:
        ordinal = mid
    names = DISH_NAMES.get(str(category), [])
    if names and ordinal < len(names):
        dish_name = names[ordinal]
    elif names:
        dish_name = f"{names[ordinal % len(names)]} phiên bản {ordinal + 1}"
    else:
        dish_name = f"Món đặc trưng {mid}"
    return f"{category_name} — {dish_name}"

def center_label(center_id) -> str:
    return f"Bếp trung tâm {int(center_id)}"

def category_label(value) -> str:
    return CATEGORY_TRANSLATIONS.get(str(value), str(value).replace("_", " ").title())

def add_lag_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.sort_values(["center_id", "meal_id", "week"]).copy()
    grouped = out.groupby(["center_id", "meal_id"], sort=False)["num_orders"]
    out["lag_1"] = grouped.shift(1)
    out["lag_2"] = grouped.shift(2)
    out["lag_3"] = grouped.shift(3)
    out["rolling_mean_3"] = (
        out.groupby(["center_id", "meal_id"], sort=False)["num_orders"]
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    )
    out["rolling_mean_5"] = (
        out.groupby(["center_id", "meal_id"], sort=False)["num_orders"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    )
    out["rolling_std_3"] = (
        out.groupby(["center_id", "meal_id"], sort=False)["num_orders"]
        .transform(lambda s: s.shift(1).rolling(3, min_periods=2).std())
    )
    return out

MODEL_FEATURES = FEATURE_CANDIDATES + [
    "lag_1", "lag_2", "lag_3", "rolling_mean_3", "rolling_mean_5", "rolling_std_3"
]
MODEL_FEATURES = list(dict.fromkeys(MODEL_FEATURES))

def make_feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    X = frame.copy()
    for col in MODEL_FEATURES:
        if col not in X.columns:
            X[col] = 0
    X = X[MODEL_FEATURES].replace([np.inf, -np.inf], np.nan)
    return X.apply(pd.to_numeric, errors="coerce").fillna(0)

# ---------- ML Pipeline Class & Candidate Models Generator ----------

class YeoJohnsonPipeline:
    def __init__(self, model, transformer):
        self.model = model
        self.transformer = transformer

    def predict(self, X):
        pred_yj = self.model.predict(X)
        pred_orig = self.transformer.inverse_transform(np.array(pred_yj).reshape(-1, 1)).ravel()
        return np.maximum(0, pred_orig)

def get_candidate_models():
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.07, max_depth=5, random_state=42),
    }

    if XGBRegressor is not None:
        models["XGBoost"] = XGBRegressor(n_estimators=100, learning_rate=0.07, max_depth=5, random_state=42, verbosity=0)
    else:
        models["XGBoost"] = HistGradientBoostingRegressor(max_iter=100, learning_rate=0.07, random_state=42)

    if LGBMRegressor is not None:
        models["LightGBM"] = LGBMRegressor(n_estimators=100, learning_rate=0.07, max_depth=5, random_state=42, verbose=-1)
    else:
        models["LightGBM"] = HistGradientBoostingRegressor(max_iter=100, learning_rate=0.07, max_leaf_nodes=31, random_state=43)

    if CatBoostRegressor is not None:
        models["CatBoost"] = CatBoostRegressor(iterations=100, learning_rate=0.07, depth=5, random_seed=42, verbose=0)
    else:
        models["CatBoost"] = HistGradientBoostingRegressor(max_iter=100, learning_rate=0.07, max_leaf_nodes=25, random_state=44)

    return models

@st.cache_resource(show_spinner="AI đang huấn luyện và đánh giá 5 mô hình từ dữ liệu tuần 1-145...")
def fit_model_cached(data_hash: str, frame: pd.DataFrame):
    featured = add_lag_features(frame)
    usable = featured.dropna(subset=["lag_1"]).copy()
    if len(usable) < 100:
        raise ValueError("Chưa đủ dữ liệu lịch sử để huấn luyện mô hình.")

    max_week = int(frame["week"].max())
    validation_start = max_week - 1
    train_rows = usable[usable["week"] < validation_start]
    valid_rows = usable[usable["week"] >= validation_start]
    if len(train_rows) < 100 or valid_rows.empty:
        train_rows, valid_rows = usable.iloc[:-max(1, len(usable)//5)], usable.iloc[-max(1, len(usable)//5):]

    X_train = make_feature_matrix(train_rows)
    X_valid = make_feature_matrix(valid_rows)
    y_valid_actual = valid_rows["num_orders"].values

    pt_train = PowerTransformer(method="yeo-johnson")
    y_train_yj = pt_train.fit_transform(train_rows[["num_orders"]].values).ravel()

    candidates = get_candidate_models()
    eval_results = {}

    for name, model_inst in candidates.items():
        model_inst.fit(X_train, y_train_yj)
        y_valid_pred_yj = model_inst.predict(X_valid)
        y_valid_pred_orig = pt_train.inverse_transform(np.array(y_valid_pred_yj).reshape(-1, 1)).ravel()
        y_valid_pred_orig = np.maximum(0, y_valid_pred_orig)

        mae = float(mean_absolute_error(y_valid_actual, y_valid_pred_orig))
        rmse = float(np.sqrt(mean_squared_error(y_valid_actual, y_valid_pred_orig)))

        eval_results[name] = {
            "mae": mae,
            "rmse": rmse,
            "pred_orig": y_valid_pred_orig,
            "model": model_inst
        }

    best_name = min(eval_results, key=lambda k: eval_results[k]["mae"])
    best_eval = eval_results[best_name]

    denominator = np.abs(y_valid_actual).sum()
    wape = float(np.abs(y_valid_actual - best_eval["pred_orig"]).sum() / denominator * 100) if denominator else 0.0

    pt_full = PowerTransformer(method="yeo-johnson")
    y_full_yj = pt_full.fit_transform(usable[["num_orders"]].values).ravel()
    X_full = make_feature_matrix(usable)

    final_model = copy.deepcopy(candidates[best_name])
    final_model.fit(X_full, y_full_yj)

    final_pipeline = YeoJohnsonPipeline(final_model, pt_full)

    metrics = {
        "mae": best_eval["mae"],
        "rmse": best_eval["rmse"],
        "wape": wape,
        "best_model_name": best_name,
        "validation_weeks": sorted(valid_rows["week"].astype(int).unique().tolist()),
        "training_rows": int(len(usable))
    }
    return final_pipeline, metrics

def predict_test_set(pipeline: YeoJohnsonPipeline, history: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    if test_df.empty:
        return pd.DataFrame()

    test_sorted = test_df.sort_values(["week", "center_id", "meal_id"]).copy()
    test_weeks = sorted(test_sorted["week"].unique())

    history_sorted = history.sort_values(["center_id", "meal_id", "week"])
    recent_history = {}
    for (c, m), grp in history_sorted.groupby(["center_id", "meal_id"], sort=False):
        recent_history[(c, m)] = grp["num_orders"].values[-5:].tolist()

    all_predictions = []

    for w in test_weeks:
        w_df = test_sorted[test_sorted["week"] == w].copy()

        l1, l2, l3, rm3, rm5, rstd3 = [], [], [], [], [], []
        for _, row in w_df.iterrows():
            key = (int(row["center_id"]), int(row["meal_id"]))
            vals = recent_history.get(key, [0])
            l1.append(vals[-1] if len(vals) >= 1 else 0)
            l2.append(vals[-2] if len(vals) >= 2 else 0)
            l3.append(vals[-3] if len(vals) >= 3 else 0)
            rm3.append(float(np.mean(vals[-3:])) if vals else 0.0)
            rm5.append(float(np.mean(vals[-5:])) if vals else 0.0)
            rstd3.append(float(np.std(vals[-3:], ddof=1)) if len(vals[-3:]) >= 2 else 0.0)

        w_df["lag_1"] = l1
        w_df["lag_2"] = l2
        w_df["lag_3"] = l3
        w_df["rolling_mean_3"] = rm3
        w_df["rolling_mean_5"] = rm5
        w_df["rolling_std_3"] = rstd3

        X_w = make_feature_matrix(w_df)
        preds = pipeline.predict(X_w)
        preds_int = np.rint(preds).astype(int)
        w_df["forecast_orders"] = preds_int
        w_df["forecast_revenue"] = w_df["checkout_price"] * preds_int

        all_predictions.append(w_df)

        for idx, row in w_df.iterrows():
            key = (int(row["center_id"]), int(row["meal_id"]))
            val = float(preds_int[len(all_predictions[-1]) - len(w_df) + (idx - w_df.index[0])])
            if key not in recent_history:
                recent_history[key] = []
            recent_history[key].append(val)
            if len(recent_history[key]) > 5:
                recent_history[key].pop(0)

    result_df = pd.concat(all_predictions, ignore_index=True)
    return result_df

# ---------- Load and filter ----------
history = load_history()
test_df = load_test_data(history)

def infer_meal_category(frame: pd.DataFrame) -> pd.Series:
    category_columns = [c for c in frame.columns if c.startswith("category_") and c != "category_encoded"]
    if not category_columns:
        if "category" in frame.columns:
            return frame["category"].fillna("Chưa phân loại").astype(str)
        return pd.Series("Chưa phân loại", index=frame.index, dtype="object")

    flags = frame[category_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
    values_array = flags.to_numpy()
    names = [c[len("category_"):] for c in category_columns]
    selected = values_array.argmax(axis=1)
    has_category = values_array.max(axis=1) > 0
    categories = np.array(names, dtype=object)[selected]
    categories[~has_category] = "Chưa phân loại"
    return pd.Series(categories, index=frame.index, dtype="object")

if "category" not in history.columns:
    history["category"] = infer_meal_category(history)

MEAL_CATEGORY_MAP = (
    history[["meal_id", "category"]]
    .dropna(subset=["meal_id"])
    .drop_duplicates(subset=["meal_id"])
    .set_index("meal_id")["category"]
    .to_dict()
)
latest_week = int(history["week"].max())
data_fingerprint = hashlib.md5(
    pd.util.hash_pandas_object(history[sorted(history.columns)], index=True).values.tobytes()
).hexdigest()

# =========================================================
# VECTOR SVG LOGO (Logo Nắp đậy Món Ăn & Mạng AI)
# =========================================================
def get_exact_logo_b64_url(width=135):
    svg_raw = f"""<svg viewBox="0 0 200 200" width="{width}" height="{width}" xmlns="http://www.w3.org/2000/svg" style="display: block; margin: 0 auto;">
<g fill="none" stroke="#1C3829" stroke-linecap="round" stroke-linejoin="round">
  <path d="M 30 130 L 170 130 A 6 6 0 0 1 170 142 L 30 142 A 6 6 0 0 1 30 130 Z" stroke-width="4.5" fill="#FFFCF8"/>
  <path d="M 90 55 C 90 48 110 48 110 55 Z" stroke-width="4" fill="#1C3829"/>
  <path d="M 35 130 A 65 65 0 0 1 100 58" stroke-width="5"/>
  <path d="M 45 120 A 52 52 0 0 1 90 70" stroke-width="2.5" stroke-dasharray="6 4"/>
  <g stroke-width="2.2">
    <line x1="100" y1="58" x2="125" y2="68"/>
    <line x1="125" y1="68" x2="150" y2="88"/>
    <line x1="150" y1="88" x2="165" y2="130"/>
    <line x1="100" y1="58" x2="115" y2="90"/>
    <line x1="125" y1="68" x2="115" y2="90"/>
    <line x1="125" y1="68" x2="140" y2="105"/>
    <line x1="150" y1="88" x2="140" y2="105"/>
    <line x1="150" y1="88" x2="160" y2="110"/>
    <line x1="115" y1="90" x2="105" y2="120"/>
    <line x1="115" y1="90" x2="135" y2="125"/>
    <line x1="140" y1="105" x2="135" y2="125"/>
    <line x1="140" y1="105" x2="165" y2="130"/>
    <line x1="105" y1="120" x2="100" y2="130"/>
    <line x1="135" y1="125" x2="130" y2="130"/>
    <line x1="135" y1="125" x2="155" y2="130"/>
  </g>
  <g fill="#1C3829" stroke="none">
    <circle cx="100" cy="58" r="4.5"/>
    <circle cx="125" cy="68" r="5"/>
    <circle cx="150" cy="88" r="5.5"/>
    <circle cx="165" cy="130" r="5"/>
    <circle cx="115" cy="90" r="5"/>
    <circle cx="140" cy="105" r="5.5"/>
    <circle cx="105" cy="120" r="4.5"/>
    <circle cx="135" cy="125" r="5"/>
  </g>
</g>
</svg>"""
    b64_str = base64.b64encode(svg_raw.encode("utf-8")).decode("utf-8")
    return f"data:image/svg+xml;base64,{b64_str}"

# 1. SIDEBAR LOGO
st.sidebar.markdown(
    f"""
    <div style="text-align: center; padding: 0.8rem 0 0.5rem 0;">
      <img src="{get_exact_logo_b64_url(135)}" width="135" height="135" style="display: block; margin: 0 auto; object-fit: contain;" />
      <div style="font-family: 'Montserrat', sans-serif; font-size: 1.85rem; font-weight: 900; color: #1C3829; letter-spacing: -0.5px; margin-top: -1.2rem; line-height: 1.1;">
        SmartBites <span style="color: #5D856C; font-weight: 800;">AI</span>
      </div>
      <div class="sidebar-subtitle" style="margin-top: 0.35rem;">
        Quản trị chuỗi cung ứng & giá bán F&B
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown('<div style="height:1px;background:#C7D5C2;margin:0 8px 12px;"></div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-section-title">Chức năng</div>', unsafe_allow_html=True)
pages = ["Tổng quan kinh doanh", "Dự báo nhu cầu & vận hành", "Giá bán & khuyến mãi", "Cập nhật dữ liệu"]
page_icons = [":material/format_list_bulleted:", ":material/monitoring:", ":material/price_check:", ":material/description:"]
if "selected_page" not in st.session_state:
    st.session_state.selected_page = pages[0]
elif st.session_state.selected_page not in pages:
    old_page = str(st.session_state.selected_page)
    if "Dự báo" in old_page:
        st.session_state.selected_page = pages[1]
    elif "Giá bán" in old_page:
        st.session_state.selected_page = pages[2]
    elif "Cập nhật" in old_page:
        st.session_state.selected_page = pages[3]
    else:
        st.session_state.selected_page = pages[0]
for label, icon in zip(pages, page_icons):
    if st.sidebar.button(label, key=f"nav_{label}", icon=icon, type="primary" if st.session_state.selected_page == label else "secondary", use_container_width=True):
        st.session_state.selected_page = label
        st.rerun()
page = st.session_state.selected_page

st.sidebar.markdown('<div class="sidebar-section-title">Bộ lọc dữ liệu</div>', unsafe_allow_html=True)
center_ids = sorted(history["center_id"].dropna().unique().tolist())
meal_ids = sorted(history["meal_id"].dropna().unique().tolist())
region_ids = sorted(history["region_code"].dropna().unique().tolist()) if "region_code" in history else []
selected_region_label = st.sidebar.selectbox("Khu vực", ["Tất cả"] + [f"Khu vực {int(x)}" for x in region_ids])
selected_center_label = st.sidebar.selectbox("Bếp trung tâm", ["Tất cả"] + [center_label(x) for x in center_ids])
category_col = "category" if "category" in history.columns else None
category_values = sorted(history[category_col].dropna().unique().tolist()) if category_col else []
selected_category_label = st.sidebar.selectbox("Nhóm món", ["Tất cả"] + [category_label(x) for x in category_values])

filtered = history.copy()
if selected_region_label != "Tất cả" and "region_code" in filtered:
    selected_region = int(selected_region_label.replace("Khu vực ", ""))
    filtered = filtered[filtered["region_code"] == selected_region]
if selected_center_label != "Tất cả":
    selected_center = int(selected_center_label.replace("Bếp trung tâm ", ""))
    filtered = filtered[filtered["center_id"] == selected_center]
if selected_category_label != "Tất cả" and category_col:
    filtered = filtered[filtered[category_col] == selected_category_label]

# 2. MAIN HEADER BANNER
st.markdown(
    f"""
    <div style="
        position:relative; overflow:hidden; padding:1.4rem 1.7rem;
        border:1px solid #E8DCD0; border-radius:24px;
        background:linear-gradient(115deg,#FFFCF8 0%,#F8F0E8 62%,#E5EBDD 100%);
        box-shadow:0 9px 30px rgba(86,52,38,.07); margin-bottom:1rem;">
      <div style="position:relative;z-index:1;display:flex;align-items:center;gap:20px;">
        <!-- Brand Logo Container -->
        <div style="flex:0 0 auto; display:flex; align-items:center; justify-content:center;">
          <img src="{get_exact_logo_b64_url(78)}" width="78" height="78" style="display: block; object-fit: contain;" />
        </div>
        <div>
          <div style="font-size:.72rem;font-weight:850;letter-spacing:1.8px;color:{TEAL};text-transform:uppercase;margin-bottom:.15rem;">
            AI-POWERED F&B INTELLIGENCE
          </div>
          <div style="font-family:'Montserrat',sans-serif;font-size:clamp(1.65rem,2.7vw,2.25rem);line-height:1.12;font-weight:900;letter-spacing:-1px;color:{NAVY};">
            SmartBites <span style="color:{TEAL};">AI Dashboard</span>
          </div>
          <div style="font-size:.96rem;line-height:1.5;color:#737865;margin-top:.35rem;font-weight:500;">
            Quản trị chuỗi cung ứng & giá bán F&B
          </div>
        </div>
      </div>
      <div style="position:absolute;right:-30px;top:-58px;width:190px;height:190px;border:26px solid rgba(28,56,41,.08);border-radius:50%;"></div>
      <div style="position:absolute;right:45px;bottom:-80px;width:150px;height:150px;border:20px solid rgba(18,59,93,.045);border-radius:50%;"></div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.divider()

# ---------- Overview ----------
if page == pages[0]:
    st.subheader("Tổng quan kinh doanh")
    st.write("Theo dõi doanh thu, nhu cầu và hiệu quả hoạt động trên toàn chuỗi trong giai đoạn huấn luyện.")
    if filtered.empty:
        st.warning("Không có dữ liệu phù hợp với bộ lọc hiện tại.")
        st.stop()
    filtered = filtered.copy()
    filtered["revenue"] = filtered["checkout_price"] * filtered["num_orders"]
    total_revenue = filtered["revenue"].sum()
    total_orders = filtered["num_orders"].sum()
    discount = ((filtered["base_price"] - filtered["checkout_price"]) / filtered["base_price"].replace(0, np.nan)).mean()
    active_centers = filtered["center_id"].nunique()
    filtered["discount_rate"] = (filtered["base_price"] - filtered["checkout_price"]) / filtered["base_price"].replace(0, np.nan)
    weekly = filtered.groupby("week", as_index=False).agg(
        revenue=("revenue", "sum"),
        orders=("num_orders", "sum"),
        discount=("discount_rate", "mean"),
        centers=("center_id", "nunique"),
    )
    kpi_cols = st.columns(4, gap="medium")
    with kpi_cols[0]:
        overview_kpi_card("Doanh thu thuần", format_vnd(total_revenue), weekly["revenue"], "#1C3829", '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M12 3v18M17 7.5c0-1.7-2.2-3-5-3s-5 1.3-5 3 2.2 3 5 3 5 1.3 5 3-2.2 3-5 3-5-1.3-5-3"/></svg>')
    with kpi_cols[1]:
        overview_kpi_card("Tổng số đơn hàng", f"{total_orders:,.0f}", weekly["orders"], "#C58D6D", '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.1" stroke-linecap="round"><path d="M9 6h11M9 12h11M9 18h11"/><circle cx="4" cy="6" r="1"/><circle cx="4" cy="12" r="1"/><circle cx="4" cy="18" r="1"/></svg>')
    with kpi_cols[2]:
        overview_kpi_card("Mức giảm giá bình quân", f"{(discount if pd.notna(discount) else 0):.1%}", weekly["discount"].fillna(0), "#563426", '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.1" stroke-linecap="round"><path d="M7 17 17 7M8 7h.01M16 17h.01"/><circle cx="8" cy="7" r="2"/><circle cx="16" cy="17" r="2"/></svg>')
    with kpi_cols[3]:
        overview_kpi_card("Bếp đang hoạt động", f"{active_centers}", weekly["centers"], "#5D856C", '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.1" stroke-linejoin="round"><path d="M4 21V5h16v16M2 21h20M8 9h2m4 0h2M8 13h2m4 0h2M10 21v-4h4v4"/></svg>')

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=weekly["week"], y=weekly["revenue"], name="Doanh thu (VNĐ)",
        mode="lines", line=dict(color=NAVY, width=3.4, shape="spline"),
        hovertemplate="Tuần %{x}<br>Doanh thu: %{y:,.0f} ₫<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=weekly["week"], y=weekly["orders"], name="Số đơn hàng",
        mode="lines", yaxis="y2", line=dict(color="#C58D6D", width=2.7, shape="spline", dash="dot"),
        hovertemplate="Tuần %{x}<br>Số đơn: %{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Xu hướng doanh thu và đơn hàng lịch sử", x=0.015, xanchor="left",
                   font=dict(size=18, color=NAVY)),
        hovermode="x unified",
        xaxis_title="Tuần",
        yaxis=dict(title="Doanh thu (VNĐ)", tickformat="~s", gridcolor="#E8DCD0", zeroline=False),
        yaxis2=dict(title="Số đơn hàng", overlaying="y", side="right", showgrid=False, tickformat="~s"),
        plot_bgcolor=WHITE, paper_bgcolor=WHITE,
        font=dict(family="Arial", color="#334155", size=12),
        legend=dict(title="", orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=18, r=18, t=75, b=18),
    )
    fig.update_xaxes(showgrid=False, linecolor="#DCCFC2", zeroline=False)
    fig.update_traces(hoverlabel=dict(bgcolor=WHITE, font_color=NAVY, bordercolor=BORDER))
    st.plotly_chart(fig, use_container_width=True)

    best_week_row = weekly.loc[weekly["revenue"].idxmax()]
    best_week_num = int(best_week_row["week"])
    best_week_rev = float(best_week_row["revenue"])
    best_week_orders = int(best_week_row["orders"])
    st.markdown(
        f"""
        <div style="
            margin-top:1rem; padding:1.15rem 1.35rem; border-radius:18px;
            background:linear-gradient(120deg,#FFFCF8 0%,#F3E9DF 55%,#E8F0E4 100%);
            border:1px solid #E8DCD0; box-shadow:0 6px 20px rgba(86,52,38,.06);">
          <div style="font-size:1.1rem;font-weight:800;letter-spacing:1.2px;color:{TEAL};text-transform:uppercase;margin-bottom:.35rem;">
            Tuần doanh thu cao nhất
          </div>
          <div style="display:flex;flex-wrap:wrap;align-items:baseline;gap:14px 28px;">
            <div style="font-size:1.55rem;font-weight:850;color:{NAVY};">Tuần {best_week_num}</div>
            <div style="font-size:1.25rem;font-weight:800;color:#1C3829;">{format_vnd(best_week_rev)}</div>
            <div style="font-size:1rem;color:#64748B;font-weight:600;">{best_week_orders:,.0f} đơn hàng</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander(f"Xem chi tiết lịch sử tuần {best_week_num}", expanded=False):
        week_slice = filtered[filtered["week"] == best_week_num].copy()
        week_slice["revenue"] = week_slice["checkout_price"] * week_slice["num_orders"]
        st.markdown(f"**Top 5 món bán chạy nhất — Tuần {best_week_num}**")
        top_meals_week = (
            week_slice.groupby("meal_id", as_index=False)
            .agg(total_orders=("num_orders", "sum"), revenue=("revenue", "sum"))
            .sort_values("total_orders", ascending=False)
            .head(5)
        )
        top_meals_week["Món ăn"] = top_meals_week["meal_id"].map(meal_label)
        top_meals_week["Nhóm món"] = top_meals_week["meal_id"].map(MEAL_CATEGORY_MAP).fillna("Chưa phân loại").map(category_label)
        meal_detail = top_meals_week[["Nhóm món", "Món ăn", "total_orders", "revenue"]].rename(
            columns={"total_orders": "Số đơn", "revenue": "Doanh thu (VNĐ)"}
        )
        st.dataframe(
            meal_detail.style.set_properties(**{"color": "#111111", "font-weight": "600"}),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Số đơn": st.column_config.NumberColumn(format="%,.0f"),
                "Doanh thu (VNĐ)": st.column_config.NumberColumn(format="%,.0f ₫"),
            },
        )
        st.markdown(f"**Top 5 bếp doanh thu mạnh nhất — Tuần {best_week_num}**")
        top_centers_week = (
            week_slice.groupby("center_id", as_index=False)
            .agg(total_orders=("num_orders", "sum"), revenue=("revenue", "sum"))
            .sort_values("revenue", ascending=False)
            .head(5)
        )
        top_centers_week["Tên bếp"] = top_centers_week["center_id"].map(center_label)
        center_detail = top_centers_week[["Tên bếp", "total_orders", "revenue"]].rename(
            columns={"total_orders": "Số đơn", "revenue": "Doanh thu (VNĐ)"}
        )
        st.dataframe(
            center_detail.style.set_properties(**{"color": "#111111", "font-weight": "600"}),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Số đơn": st.column_config.NumberColumn(format="%,.0f"),
                "Doanh thu (VNĐ)": st.column_config.NumberColumn(format="%,.0f ₫"),
            },
        )

    st.divider()

    st.subheader("Phân tích các lĩnh vực kinh doanh")
    st.caption("Khám phá nhóm món tạo doanh thu, món ăn bán chạy, tác động của giảm giá và các bếp có doanh thu cao nhất.")

    row1_left, row1_right = st.columns(2, gap="large")
    with row1_left:
        st.markdown("<div class='chart-section-title'>Top 5 nhóm món theo doanh thu</div>", unsafe_allow_html=True)
        if category_col:
            top = filtered.assign(revenue=filtered["checkout_price"] * filtered["num_orders"]).groupby(category_col)["revenue"].sum().nlargest(5).reset_index()
            top["Nhóm món"] = top[category_col].map(category_label)
            chart = px.bar(
                top, x="Nhóm món", y="revenue", text_auto=".2s",
                color="Nhóm món", color_discrete_sequence=CHART_PALETTE,
            )
            y_title = "Doanh thu (VNĐ)"
            title_text = "Doanh thu theo nhóm món"
        else:
            top = filtered.groupby("meal_id", as_index=False)["num_orders"].sum().nlargest(5, "num_orders")
            top["Nhóm món"] = top["meal_id"].map(meal_label)
            chart = px.bar(
                top, x="Nhóm món", y="num_orders", text_auto=".2s",
                color="Nhóm món", color_discrete_sequence=CHART_PALETTE,
            )
            y_title = "Số đơn"
            title_text = "Số đơn theo món"
        chart.update_traces(
            marker_line_color=WHITE, marker_line_width=1.2,
            textposition="outside", cliponaxis=False,
            hovertemplate="%{x}<br>Giá trị: %{y:,.0f}<extra></extra>",
        )
        chart.update_layout(
            title=dict(text=title_text, x=.5, xanchor="center", font=dict(size=18, color=NAVY)),
            plot_bgcolor=WHITE, paper_bgcolor=WHITE,
            font=dict(family="Arial", color=BLACK, size=11),
            xaxis_title="", yaxis_title=y_title,
            showlegend=False, margin=dict(l=12, r=18, t=55, b=25),
        )
        chart.update_xaxes(showgrid=False, linecolor=BORDER)
        chart.update_yaxes(gridcolor=BORDER, zeroline=False)
        st.plotly_chart(chart, use_container_width=True)

    with row1_right:
        st.markdown("<div class='chart-section-title'>Top 10 món ăn theo số đơn</div>", unsafe_allow_html=True)
        top_meals = filtered.groupby("meal_id", as_index=False).agg(
            total_orders=("num_orders", "sum"),
            revenue=("checkout_price", lambda x: 0),
        )
        revenue_by_meal = filtered.assign(_rev=filtered["checkout_price"] * filtered["num_orders"]).groupby("meal_id")["_rev"].sum()
        top_meals["revenue"] = top_meals["meal_id"].map(revenue_by_meal)
        top_meals["Tên món / nhóm món"] = top_meals["meal_id"].map(meal_label)
        top_meals = top_meals.nlargest(10, "total_orders")
        top_meals["Nhãn món hiển thị"] = top_meals["Tên món / nhóm món"].astype(str).str.replace(" — ", "<br>", n=1, regex=False)
        dish_chart = px.bar(
            top_meals.sort_values("total_orders"), x="total_orders",
            y="Nhãn món hiển thị", orientation="h", text_auto=".2s",
            color="Tên món / nhóm món", color_discrete_sequence=CHART_PALETTE,
        )
        dish_chart.update_traces(
            marker_line_color=WHITE, marker_line_width=1.0,
            textposition="outside", cliponaxis=False,
            hovertemplate="%{y}<br>Số đơn: %{x:,.0f}<extra></extra>",
        )
        dish_chart.update_layout(
            title=dict(text="10 món có số đơn cao nhất", x=.5, xanchor="center", font=dict(size=18, color=NAVY)),
            plot_bgcolor=WHITE, paper_bgcolor=WHITE,
            font=dict(family="Arial", color=BLACK, size=11),
            xaxis_title="Tổng số đơn", yaxis_title="", showlegend=False,
            height=520,
            margin=dict(l=150, r=48, t=58, b=88),
        )
        dish_chart.update_xaxes(gridcolor=BORDER, zeroline=False, automargin=False)
        dish_chart.update_yaxes(showgrid=False, automargin=False, tickfont=dict(size=10), ticklabeloverflow="allow", ticklabelstandoff=5)
        st.plotly_chart(dish_chart, use_container_width=True)

    row2_left, row2_right = st.columns(2, gap="large")
    with row2_left:
        st.markdown("<div class='chart-section-title'>Mức giảm giá và số đơn</div>", unsafe_allow_html=True)
        scatter = filtered.sample(min(2500, len(filtered)), random_state=42).copy()
        scatter["discount_pct"] = (scatter["base_price"] - scatter["checkout_price"]) / scatter["base_price"].replace(0, np.nan)
        chart = px.scatter(scatter, x="discount_pct", y="num_orders", opacity=.52)
        chart.update_traces(
            marker=dict(color="#5D856C", size=7, line=dict(color=WHITE, width=.7)),
            hovertemplate="Giảm giá: %{x:.1%}<br>Số đơn: %{y:,.0f}<extra></extra>",
        )
        chart.update_layout(
            title=dict(text="Mối quan hệ giữa giảm giá và số đơn", x=.5, xanchor="center", font=dict(size=18, color=NAVY)),
            plot_bgcolor=WHITE, paper_bgcolor=WHITE,
            font=dict(family="Arial", color=BLACK, size=11),
            xaxis_title="Tỷ lệ giảm giá", yaxis_title="Số đơn hàng",
            xaxis_tickformat=".0%", margin=dict(l=55, r=25, t=50, b=88),
        )
        chart.update_xaxes(gridcolor=BORDER, zeroline=False, automargin=False)
        chart.update_yaxes(gridcolor=BORDER, zeroline=False)
        st.plotly_chart(chart, use_container_width=True)

    with row2_right:
        st.markdown("<div class='chart-section-title'>Top 10 bếp theo doanh thu</div>", unsafe_allow_html=True)
        center_revenue = (
            filtered.assign(_revenue=filtered["checkout_price"] * filtered["num_orders"])
            .groupby("center_id", as_index=False)["_revenue"].sum()
            .nlargest(10, "_revenue")
            .sort_values("_revenue")
        )
        center_revenue["Tên bếp đầy đủ"] = center_revenue["center_id"].map(center_label)
        center_revenue["Tên bếp"] = center_revenue["Tên bếp đầy đủ"].astype(str).str.replace("Bếp trung tâm ", "Bếp ", regex=False)
        center_chart = px.bar(
            center_revenue, x="_revenue", y="Tên bếp", orientation="h",
            text_auto=".2s", color="Tên bếp",
            color_discrete_sequence=CHART_PALETTE,
        )
        center_chart.update_traces(
            marker_line_color=WHITE, marker_line_width=1.0,
            textposition="outside", cliponaxis=False,
            hovertemplate="%{y}<br>Doanh thu: %{x:,.0f} ₫<extra></extra>",
        )
        center_chart.update_layout(
            title=dict(text="10 bếp có doanh thu cao nhất", x=.5, xanchor="center", font=dict(size=18, color=NAVY)),
            plot_bgcolor=WHITE, paper_bgcolor=WHITE,
            font=dict(family="Arial", color=BLACK, size=11),
            xaxis_title="Doanh thu (VNĐ)", yaxis_title="", showlegend=False,
            height=520,
            margin=dict(l=88, r=48, t=58, b=88),
        )
        center_chart.update_xaxes(gridcolor=BORDER, zeroline=False, automargin=False)
        center_chart.update_yaxes(showgrid=False, automargin=False, tickfont=dict(size=11), ticklabeloverflow="allow", ticklabelstandoff=5)
        st.plotly_chart(center_chart, use_container_width=True)

    st.divider()

    st.subheader("Hiệu quả hoạt động của bếp trung tâm")
    st.caption("Tổng hợp doanh thu, số đơn hàng và mật độ đơn hàng theo diện tích hoạt động của từng bếp.")
    centers = filtered.groupby("center_id", as_index=False).agg(total_orders=("num_orders","sum"), revenue=("checkout_price", lambda x: 0), op_area=("op_area","first") if "op_area" in filtered else ("num_orders","count"))
    rev_center = filtered.assign(_revenue=filtered["checkout_price"] * filtered["num_orders"]).groupby("center_id")["_revenue"].sum()
    centers["revenue"] = centers["center_id"].map(rev_center)
    centers["orders_per_area"] = centers["total_orders"] / centers["op_area"].replace(0, np.nan)
    centers["Tên bếp"] = centers["center_id"].map(center_label)
    display = centers[["Tên bếp","total_orders","revenue","orders_per_area"]].rename(columns={"total_orders":"Tổng đơn hàng","revenue":"Doanh thu (VNĐ)","orders_per_area":"Đơn hàng / diện tích"})
    display_styled = display.style.set_properties(**{"color": "#111111", "font-weight": "700"})
    st.dataframe(display_styled, use_container_width=True, hide_index=True, column_config={"Doanh thu (VNĐ)": st.column_config.NumberColumn(format="%,.0f ₫"), "Tổng đơn hàng": st.column_config.NumberColumn(format="%,.0f"), "Đơn hàng / diện tích": st.column_config.NumberColumn(format="%.1f")})

# ---------- Forecast and operations ----------
elif page == pages[1]:
    try:
        pipeline, metrics = fit_model_cached(data_fingerprint, history)
    except Exception as exc:
        st.error(f"Lỗi huấn luyện mô hình AI: {exc}")
        st.stop()

    if test_df.empty:
        st.warning("Chưa tìm thấy tệp test.csv trong thư mục data/. Hãy tải file test lên ở trang Cập nhật dữ liệu.")
        st.stop()

    test_predictions = predict_test_set(pipeline, history, test_df)

    forecast_weeks = sorted(
        week for week in test_predictions["week"].astype(int).unique().tolist()
        if 146 <= week <= 155
    )
    if not forecast_weeks:
        st.error("Không tìm thấy dữ liệu dự báo tuần 146–155 trong tệp test.csv.")
        st.stop()

    if "forecast_week_selector" not in st.session_state:
        st.session_state.forecast_week_selector = 146 if 146 in forecast_weeks else forecast_weeks[0]
    elif st.session_state.forecast_week_selector not in forecast_weeks:
        st.session_state.forecast_week_selector = forecast_weeks[0]

    title_col, week_col = st.columns([5, 1])
    with week_col:
        selected_week = st.selectbox(
            "Tuần dự báo",
            forecast_weeks,
            format_func=lambda week: f"Tuần {week}",
            key="forecast_week_selector",
        )
    with title_col:
        st.subheader(f"Dự báo nhu cầu & năng lực vận hành — Tuần {selected_week}")

    sub_preds = test_predictions[test_predictions["week"] == selected_week].copy()
    if sub_preds.empty:
        st.error(f"Không tìm thấy dữ liệu tuần {selected_week} trong tệp test.csv!")
        st.stop()

    total_forecast = int(sub_preds["forecast_orders"].sum())
    historical_last = int(sub_preds["lag_1"].sum())
    delta_pct = (total_forecast - historical_last) / historical_last * 100 if historical_last else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Tổng đơn dự báo tuần {selected_week}", f"{total_forecast:,.0f}", f"{delta_pct:+.1f}% so với tuần {selected_week - 1}")
    c2.metric("Món ăn dự báo", f"{sub_preds['meal_id'].nunique():,}")
    c3.metric("Bếp trung tâm", f"{sub_preds['center_id'].nunique():,}")
    c4.metric("WAPE kiểm định", f"{metrics['wape']:.1f}%", help=f"Mô hình được chọn: {metrics['best_model_name']}")

    with st.expander("Chất lượng mô hình AI và quy trình chọn lựa"):
        st.write(f"- Mô hình tối ưu được chọn: **{metrics['best_model_name']}**")
        st.write(f"- MAE kiểm định: **{metrics['mae']:,.1f} đơn**")
        st.write(f"- RMSE kiểm định: **{metrics['rmse']:,.1f} đơn**")
        st.write(f"- WAPE kiểm định: **{metrics['wape']:.1f}%**")
        st.write(f"- Số dòng dữ liệu huấn luyện (tuần 1–{selected_week - 1}): **{metrics['training_rows']:,}**")

    weekly_actual = history.groupby("week", as_index=False)["num_orders"].sum().rename(columns={"num_orders": "Số đơn", "week": "Tuần"})
    weekly_actual["Loại"] = "Thực tế (Tuần 1-145)"

    forecast_path = (
        test_predictions[test_predictions["week"] <= selected_week]
        .groupby("week", as_index=False)["forecast_orders"].sum()
        .rename(columns={"week": "Tuần", "forecast_orders": "Số đơn"})
    )
    forecast_path["Loại"] = "Dự báo AI"
    trend = pd.concat([weekly_actual, forecast_path], ignore_index=True)

    fig = go.Figure()
    act = trend[trend["Loại"].astype(str).str.startswith("Thực tế")]
    fct = trend[trend["Loại"] == "Dự báo AI"].sort_values("Tuần")

    bridge = fct.copy()
    if not act.empty and not fct.empty:
        last_act = act.sort_values("Tuần").iloc[-1]
        bridge = pd.concat([
            pd.DataFrame([{"Tuần": int(last_act["Tuần"]), "Số đơn": float(last_act["Số đơn"])}]),
            fct[["Tuần", "Số đơn"]],
        ], ignore_index=True)

    fig.add_trace(go.Scatter(
        x=act["Tuần"], y=act["Số đơn"], name="Thực tế (Tuần 1-145)",
        mode="lines", line=dict(color=NAVY, width=2.8),
        hovertemplate="Tuần %{x}<br>Thực tế: %{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=bridge["Tuần"], y=bridge["Số đơn"], name=f"Dự báo AI (đến tuần {selected_week})",
        mode="lines", line=dict(color=TEAL, width=3, dash="dot"),
        hovertemplate="Tuần %{x}<br>Dự báo: %{y:,.0f}<extra></extra>",
    ))
    point = fct[fct["Tuần"] == selected_week]
    if not point.empty:
        fig.add_trace(go.Scatter(
            x=point["Tuần"], y=point["Số đơn"],
            name=f"Tuần {selected_week}",
            mode="markers",
            marker=dict(color=TEAL, size=16, symbol="diamond", line=dict(color=WHITE, width=2.5)),
            hovertemplate="Tuần %{x}<br>Dự báo: %{y:,.0f}<extra></extra>",
            showlegend=False,
        ))
    fig.add_vline(x=145.5, line_dash="dash", line_color=SLATE,
                  annotation_text="Bắt đầu dự báo", annotation_position="top")
    fig.update_layout(
        title=dict(text=f"Xu hướng nhu cầu thực tế và dự báo AI đến tuần {selected_week}", x=.015,
                   font=dict(size=18, color=NAVY)),
        plot_bgcolor=WHITE, paper_bgcolor=WHITE,
        font=dict(family="Arial", color="#334155", size=12),
        xaxis_title="Tuần", yaxis_title="Số đơn hàng", hovermode="x unified",
        legend=dict(title="", orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=18, r=18, t=75, b=18),
    )
    fig.update_xaxes(showgrid=False, linecolor="#DCCFC2")
    fig.update_yaxes(gridcolor="#E8DCD0", zeroline=False, tickformat="~s")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader(f"Dự báo nhu cầu theo món ăn — Tuần {selected_week}")
    by_meal = sub_preds.groupby("meal_id", as_index=False).agg(forecast_orders=("forecast_orders","sum"), previous_orders=("lag_1","sum"))
    by_meal["Món ăn"] = by_meal["meal_id"].map(meal_label)
    by_meal["Nhóm món"] = by_meal["meal_id"].map(MEAL_CATEGORY_MAP).fillna("Chưa phân loại").map(category_label)
    by_meal["Thay đổi (%)"] = np.where(by_meal["previous_orders"] > 0, (by_meal["forecast_orders"]-by_meal["previous_orders"])/by_meal["previous_orders"]*100, np.nan)
    by_meal = by_meal.sort_values("forecast_orders", ascending=False)
    forecast_table = by_meal[["Nhóm món", "Món ăn", "previous_orders", "forecast_orders", "Thay đổi (%)"]].rename(columns={"previous_orders":f"Đơn tuần {selected_week - 1}","forecast_orders":f"Đơn dự báo tuần {selected_week}"})
    forecast_col = f"Đơn dự báo tuần {selected_week}"
    forecast_styled = forecast_table.style.set_properties(**{"color": "#000000"})
    forecast_styled = forecast_styled.set_properties(subset=[forecast_col], **{"background-color": "#E7F0E5", "color": "#000000", "font-weight": "700"})
    st.dataframe(forecast_styled, use_container_width=True, hide_index=True, column_config={f"Đơn tuần {selected_week - 1}":st.column_config.NumberColumn(format="%,.0f"),forecast_col:st.column_config.NumberColumn(format="%,.0f"),"Thay đổi (%)":st.column_config.NumberColumn(format="%.1f%%")})

    st.markdown(f"### 🌱 Các món ăn nổi bật — Tuần {selected_week}")
    st.caption("Top 6 món được dự báo bán ổn định (khối lượng cao, biến động không giảm mạnh), sắp xếp theo số đơn dự báo.")
    stable_meals = by_meal.copy()
    stable_meals["Thay đổi (%)"] = stable_meals["Thay đổi (%)"].fillna(0)
    stable_pool = stable_meals[stable_meals["Thay đổi (%)"] >= -10].copy()
    if stable_pool.empty:
        stable_pool = stable_meals.copy()
    top_stable = stable_pool.nlargest(6, "forecast_orders").reset_index(drop=True)

    if not top_stable.empty:
        for row_start in range(0, len(top_stable), 3):
            meal_cols = st.columns(3, gap="medium")
            for col_idx, (_, row) in enumerate(top_stable.iloc[row_start:row_start + 3].iterrows()):
                rank = row_start + col_idx + 1
                change = row["Thay đổi (%)"]
                change_txt = f"{change:+.1f}%" if pd.notna(change) else "—"
                with meal_cols[col_idx]:
                    st.markdown(
                        f"<div class='sb-opportunity-card sb-meal-card'>"
                        f"<div class='sb-meal-rank'>TOP {rank}</div>"
                        f"<div class='sb-meal-name'>{row['Món ăn']}</div>"
                        f"<div class='sb-meal-category'>{row['Nhóm món']}</div>"
                        f"<div class='sb-meal-orders'>{int(row['forecast_orders']):,.0f}<span> đơn dự báo</span></div>"
                        f"<div class='sb-meal-meta'>Tuần trước: <b>{int(row['previous_orders']):,.0f}</b> đơn"
                        f"<br>Biến động: <b>{change_txt}</b></div>"
                        f"<div class='sb-meal-tip'>💡 Ưu tiên tồn kho nguyên liệu và slot sản xuất.</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
    else:
        st.info("Chưa đủ dữ liệu để đề xuất món bán ổn định cho tuần này.")

    st.subheader(f"🚨 Trung tâm cảnh báo vận hành — Tuần {selected_week}")

    by_center = sub_preds.groupby("center_id", as_index=False).agg(
        forecast_orders=("forecast_orders", "sum"),
        last_orders=("lag_1", "sum"),
        op_area=("op_area", "first") if "op_area" in sub_preds
        else ("forecast_orders", "count"),
    )
    by_center["Tên bếp"] = by_center["center_id"].map(center_label)
    by_center["orders_per_area"] = (
        by_center["forecast_orders"] /
        pd.to_numeric(by_center["op_area"], errors="coerce").replace(0, np.nan)
    )
    by_center["growth_pct"] = np.where(
        by_center["last_orders"] > 0,
        (by_center["forecast_orders"] - by_center["last_orders"])
        / by_center["last_orders"] * 100,
        np.nan,
    )

    valid_density = by_center["orders_per_area"].dropna()
    if valid_density.empty:
        density_threshold = np.nan
        median_density = np.nan
        by_center["Mức cảnh báo"] = "⚪ Thiếu dữ liệu"
    else:
        density_threshold = (
            float(valid_density.quantile(0.9))
            if len(valid_density) > 2 else float(valid_density.max())
        )
        median_density = float(valid_density.median())
        by_center["Mức cảnh báo"] = np.select(
            [
                by_center["orders_per_area"].notna()
                & (by_center["orders_per_area"] >= density_threshold),
                by_center["orders_per_area"].notna()
                & (by_center["orders_per_area"] >= median_density),
            ],
            ["🔴 Quá tải", "🟡 Cần theo dõi"],
            default="🟢 Hoạt động ổn định",
        )

    severity_order = {
        "🔴 Quá tải": 0,
        "🟡 Cần theo dõi": 1,
        "🟢 Hoạt động ổn định": 2,
        "⚪ Thiếu dữ liệu": 3,
    }
    by_center["_severity"] = by_center["Mức cảnh báo"].map(severity_order).fillna(9)
    risk = by_center.sort_values(
        ["_severity", "orders_per_area"],
        ascending=[True, False], na_position="last",
    ).copy()
    critical = risk[risk["Mức cảnh báo"] == "🔴 Quá tải"]
    warning = risk[risk["Mức cảnh báo"] == "🟡 Cần theo dõi"]

    st.markdown("""
    <style>
    @keyframes sb-critical-pulse {
        0%, 100% { box-shadow: 0 0 0 0 rgba(194,65,12,.16); border-color:#C2410C; }
        50% { box-shadow: 0 0 0 5px rgba(234,88,12,.14); border-color:#EA580C; }
    }
    .sb-critical-card {
        background:#FFF7ED; color:#9A3412;
        border:2px solid #EA580C; border-left:9px solid #C2410C;
        border-radius:14px; padding:17px 19px; margin:10px 0;
        animation:sb-critical-pulse 1.8s ease-in-out infinite;
    }
    .sb-critical-card * { color:#9A3412 !important; }
    .sb-critical-name { font-size:1.25rem; font-weight:900; line-height:1.35; }
    .sb-critical-value { font-size:1.85rem; font-weight:900; margin:5px 0; }
    .sb-warning-card {
        background:#FFF7DB; color:#854D0E; border:1px solid #EAB308;
        border-left:7px solid #D97706; border-radius:12px;
        padding:13px 16px; margin:8px 0;
    }
    .sb-opportunity-card {
        background:#E8F5E9; color:#14532D; border:1px solid #86C995;
        border-left:7px solid #15803D; border-radius:12px;
        padding:14px 17px; margin:8px 0;
    }
    .sb-safe-card {
        background:#FFFCF8; color:#1C3829; border:1px solid #D8E2D5;
        border-left:6px solid #5D856C; border-radius:11px;
        padding:12px 15px; margin:7px 0;
    }
    .sb-meal-card { min-height:250px; height:100%; margin:0 0 16px; padding:17px 18px; }
    .sb-meal-rank { display:inline-block; padding:4px 9px; border-radius:999px; background:#D7EBDD; color:#14532D; font-size:.75rem; font-weight:900; letter-spacing:.5px; }
    .sb-meal-name { color:#14532D; font-size:1.08rem; line-height:1.35; font-weight:900; margin-top:11px; min-height:2.8rem; }
    .sb-meal-category { color:#52715A; font-size:.82rem; margin-top:3px; }
    .sb-meal-orders { color:#14532D; font-size:1.65rem; font-weight:950; margin-top:12px; }
    .sb-meal-orders span { font-size:.82rem; font-weight:650; }
    .sb-meal-meta { color:#42634A; font-size:.82rem; line-height:1.6; margin-top:5px; }
    .sb-meal-tip { color:#31583A; font-size:.79rem; line-height:1.4; font-weight:700; margin-top:9px; }
    .sb-alert-grid-card { min-height:190px; height:100%; border-radius:14px; padding:15px; margin-bottom:16px; border:1px solid; box-shadow:0 5px 16px rgba(28,56,41,.06); }
    .sb-alert-grid-card.critical { background:#FEE2E2; color:#7F1D1D; border-color:#FCA5A5; border-left:7px solid #B91C1C; }
    .sb-alert-grid-card.warning { background:#FFF7DB; color:#854D0E; border-color:#FDE68A; border-left:7px solid #D97706; }
    .sb-alert-grid-card.safe { background:#F0F6EF; color:#1C3829; border-color:#D8E2D5; border-left:7px solid #5D856C; }
    .sb-alert-grid-card.unknown { background:#F1F5F9; color:#334155; border-color:#CBD5E1; border-left:7px solid #94A3B8; }
    .sb-alert-grid-card * { color:inherit !important; }
    .sb-alert-grid-name { font-size:1.02rem; font-weight:900; line-height:1.35; min-height:2.7rem; }
    .sb-alert-grid-status { font-size:.78rem; font-weight:850; margin-top:7px; }
    .sb-alert-grid-orders { font-size:1.5rem; font-weight:950; margin-top:9px; }
    .sb-alert-grid-meta { font-size:.78rem; line-height:1.5; margin-top:4px; opacity:.9; }
    @media (prefers-reduced-motion: reduce) {
        .sb-critical-card { animation:none; }
    }
    </style>
    """, unsafe_allow_html=True)

    k1, k2 = st.columns(2)
    k1.metric("Bếp cảnh báo đỏ — quá tải", f"{len(critical)}")
    k2.metric("Bếp cảnh báo vàng", f"{len(warning)}")

    if len(critical):
        st.markdown(
            f"<div class='sb-critical-card'>"
            f"<div class='sb-critical-name'>🚨 CẢNH BÁO ĐỎ — {len(critical)} BẾP CÓ NGUY CƠ QUÁ TẢI</div>"
            f"<div>Danh sách bếp quá tải được ưu tiên hiển thị bên dưới. Cần kiểm tra kế hoạch sản xuất và điều phối.</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.success("🟢 Không có bếp nào nằm trong nhóm cảnh báo đỏ theo ngưỡng mật độ đơn hàng.")

    if len(critical):
        st.markdown("### 🧭 Danh sách bếp cảnh báo đỏ (quá tải)")
        for row_start in range(0, len(critical), 4):
            alert_cols = st.columns(4, gap="medium")
            for col_idx, (_, row) in enumerate(critical.iloc[row_start:row_start + 4].iterrows()):
                density_value = row["orders_per_area"]
                density_text = f"{density_value:,.1f}" if pd.notna(density_value) else "—"
                with alert_cols[col_idx]:
                    st.markdown(
                        f"<div class='sb-alert-grid-card critical'>"
                        f"<div class='sb-alert-grid-name'>{row['Tên bếp']}</div>"
                        f"<div class='sb-alert-grid-status'>🔴 QUÁ TẢI</div>"
                        f"<div class='sb-alert-grid-orders'>{row['forecast_orders']:,.0f}</div>"
                        f"<div class='sb-alert-grid-meta'>Đơn dự báo · {density_text} đơn/diện tích"
                        f"<br>Tuần {selected_week} · Kiểm tra tải, cân nhắc điều phối đơn hoặc bổ sung nhân sự.</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

    st.markdown("### 📊 Giám sát chi tiết tất cả bếp trung tâm")
    display_cols = ["Tên bếp", "forecast_orders", "last_orders", "Mức cảnh báo"]
    display = risk[display_cols].rename(columns={
        "forecast_orders": f"Đơn dự báo tuần {selected_week}",
        "last_orders": f"Đơn tuần {selected_week - 1}",
    }).copy()
    display["Diện tích hoạt động"] = risk["op_area"].values
    display["Đơn dự báo/diện tích"] = risk["orders_per_area"].values

    def highlight_center_status(row):
        status = str(row.get("Mức cảnh báo", ""))
        if status.startswith("🔴"):
            return ["background-color:#FECACA;color:#7F1D1D;font-weight:900"] * len(row)
        if status.startswith("🟡"):
            return ["background-color:#FEF3C7;color:#78350F;font-weight:750"] * len(row)
        if status.startswith("🟢"):
            return ["background-color:#E8F5E9;color:#14532D"] * len(row)
        return ["background-color:#F1F5F9;color:#334155"] * len(row)

    styled_risk = display.style.apply(highlight_center_status, axis=1)
    number_formats = {
        f"Đơn dự báo tuần {selected_week}": "{:,.0f}",
        f"Đơn tuần {selected_week - 1}": "{:,.0f}",
    }
    number_formats.update({
        "Diện tích hoạt động": "{:,.1f}",
        "Đơn dự báo/diện tích": "{:,.1f}",
    })
    st.dataframe(
        styled_risk.format(number_formats, na_rep="—"),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        f"Ngưỡng đỏ: từ {density_threshold:,.1f} đơn/đơn vị diện tích (phân vị 90%); "
        f"ngưỡng vàng: từ {median_density:,.1f} đến dưới ngưỡng đỏ; xanh: thấp hơn ngưỡng vàng. "
        "Bếp màu đỏ được hệ thống xếp vào nhóm quá tải tương đối để ưu tiên kiểm tra."
        if pd.notna(density_threshold) and pd.notna(median_density)
        else "Không đủ dữ liệu diện tích để phân loại rủi ro cho các bếp."
    )

    export_cols = [c for c in ["id", "week", "center_id", "meal_id", "checkout_price", "base_price", "forecast_orders", "forecast_revenue"] if c in test_predictions.columns]
    csv_bytes = test_predictions[export_cols].to_csv(index=False).encode("utf-8-sig")
    st.download_button("Tải kết quả dự báo", data=csv_bytes, file_name="smartbites_test_predictions.csv", mime="text/csv")

# ---------- Pricing simulation ----------
elif page == pages[2]:
    st.subheader("Phân tích giá bán & khuyến mãi")
    st.write("Thay đổi giá bán và giả định khuyến mãi để mô phỏng số đơn dự báo.")
    try:
        pipeline, metrics = fit_model_cached(data_fingerprint, history)
    except Exception as exc:
        st.error(f"Chưa thể tạo mô phỏng dựa trên mô hình: {exc}")
        st.stop()

    if test_df.empty:
        st.warning("Chưa có tệp test.csv để mô phỏng giá.")
        st.stop()

    test_predictions = predict_test_set(pipeline, history, test_df)
    meal_146 = test_predictions[test_predictions["week"] == 146].copy()
    available_meals = sorted(meal_146["meal_id"].unique().tolist())
    selected_meal = st.selectbox("Chọn món ăn cần thiết lập giá", available_meals, format_func=meal_label)

    meal_future = meal_146[meal_146["meal_id"] == selected_meal].copy()
    if meal_future.empty:
        st.warning("Không có dữ liệu cho món được chọn.")
        st.stop()

    typical = meal_future.iloc[0]
    source_price = float(meal_future["checkout_price"].median())
    source_base = float(meal_future["base_price"].median())

    a, b, c = st.columns(3, gap="medium")
    baseline_orders = int(meal_future["forecast_orders"].sum())
    with a:
        pricing_kpi_card("Giá bán gốc", f"{source_price:,.0f} ₫", "₫", "#1C3829")
    with b:
        pricing_kpi_card("Giá bán hiện tại", f"{source_base:,.0f} ₫", "🏷️", "#5D856C")
    with c:
        pricing_kpi_card("Đơn AI dự báo ban đầu", f"{baseline_orders:,.0f}", "📦", "#C58D6D")
    st.divider()

    st.subheader("Mô phỏng kịch bản giá bán mới")
    min_price = max(1000.0, source_price * .5)
    max_price = max(source_price * 1.5, min_price + 1000.0)
    step = max(500.0, float(np.round((max_price - min_price) / 100, -2)))
    scenario_price = st.slider(
        "Giá bán mô phỏng mới (VNĐ)",
        min_value=float(min_price),
        max_value=float(max_price),
        value=float(source_price),
        step=float(step),
        format="%.0f ₫"
    )

    col1, col2 = st.columns(2)
    with col1:
        homepage = st.checkbox("Ưu tiên hiển thị trên trang chính", value=bool(int(typical.get("homepage_featured",0) or 0)))
    with col2:
        email = st.checkbox("Khuyến mãi qua email", value=bool(int(typical.get("emailer_for_promotion",0) or 0)))

    scenario = meal_future.copy()
    scenario["checkout_price"] = scenario_price
    scenario["homepage_featured"] = int(homepage)
    scenario["emailer_for_promotion"] = int(email)
    scenario["discount_amount"] = scenario["base_price"] - scenario["checkout_price"]
    scenario["discount_percent"] = np.where(scenario["base_price"].ne(0), scenario["discount_amount"] / scenario["base_price"] * 100, 0)
    scenario["price_ratio"] = np.where(scenario["base_price"].ne(0), scenario["checkout_price"] / scenario["base_price"], 1)
    scenario["total_promo"] = int(homepage) + int(email)
    scenario["promo_x_discount"] = scenario["total_promo"] * scenario["discount_percent"]
    scenario["promo_x_op_area"] = scenario["total_promo"] * scenario.get("op_area", 0)

    scenario_X = make_feature_matrix(scenario)
    scenario_orders_arr = np.rint(pipeline.predict(scenario_X)).astype(int)
    scenario_orders = int(scenario_orders_arr.sum())

    baseline_revenue = float((meal_future["checkout_price"] * meal_future["forecast_orders"]).sum())
    scenario_revenue = float((scenario_price * scenario_orders_arr).sum())
    delta = scenario_orders - baseline_orders
    delta_pct = delta / baseline_orders * 100 if baseline_orders else 0

    st.divider()
    st.subheader("Kết quả AI mô phỏng theo kịch bản giá mới (Tuần 146)")
    x, y, z = st.columns(3, gap="medium")
    with x:
        pricing_kpi_card("Số đơn dự báo theo kịch bản", f"{scenario_orders:,.0f}", "🛒", "#1C3829", f"{delta:+,} đơn ({delta_pct:+.1f}%)")
    with y:
        pricing_kpi_card("Doanh thu dự báo", format_vnd(scenario_revenue), "💰", "#5D856C")

    revenue_delta = scenario_revenue - baseline_revenue
    revenue_delta_pct = (revenue_delta / baseline_revenue * 100) if baseline_revenue else 0
    with z:
        pricing_kpi_card(
            "Doanh thu chênh lệch",
            format_vnd(revenue_delta),
            "📈" if revenue_delta >= 0 else "📉",
            "#C58D6D" if revenue_delta >= 0 else "#A94F3D",
            delta=f"{revenue_delta_pct:+.1f}% so với gốc"
        )

    comparison = pd.DataFrame({
        "Chỉ số": ["Giá bán", "Số đơn dự báo", "Doanh thu dự báo"],
        "Giá gốc hiện tại": [f"{source_price:,.0f} ₫", f"{baseline_orders:,.0f}", format_vnd(baseline_revenue)],
        "Kịch bản giá mới": [f"{scenario_price:,.0f} ₫", f"{scenario_orders:,.0f}", format_vnd(scenario_revenue)],
    })
    st.dataframe(comparison, use_container_width=True, hide_index=True)

# ---------- Data update ----------
else:
    st.subheader("Lịch sử dữ liệu")
    st.write("Tải lên tệp CSV chứa các đơn cần dự báo nhu cầu.")

    active_forecast_week = st.session_state.get("forecast_week_selector", None)
    if active_forecast_week is None or not isinstance(active_forecast_week, (int, np.integer)):
        if not test_df.empty and "week" in test_df.columns:
            active_forecast_week = int(test_df["week"].min())
        else:
            active_forecast_week = int(latest_week) + 1
    history_end_week = max(1, int(active_forecast_week) - 1)

    a, b, c = st.columns(3)
    a.metric("Tuần lịch sử huấn luyện", f"1 - {history_end_week}")
    b.metric("Tuần đang dự báo", f"Tuần {int(active_forecast_week)}")
    c.metric("Dòng dữ liệu dự báo", f"{len(test_df):,}" if not test_df.empty else "Chưa có")

    st.divider()
    st.subheader("Cập nhật tệp dữ liệu để dự báo")
    st.info(
        "Tệp CSV tối thiểu gồm các cột: **week**, **center_id**, **meal_id**, **checkout_price**, **base_price**. "
    )
    uploaded_file = st.file_uploader(
        "Chọn CSV tệp dữ liệu để dự báo",
        type=["csv"],
        key="upload_forecast_data",
    )
    if uploaded_file is not None:
        try:
            incoming = clean_test_frame(pd.read_csv(uploaded_file), history)
            st.write("Xem trước tệp tải lên:")
            st.dataframe(incoming.head(10), use_container_width=True)
            if st.button("Lưu tệp & Chạy AI dự báo", type="primary"):
                incoming.to_csv(TEST_DATA, index=False)
                st.success("Đã cập nhật tệp dữ liệu dự báo thành công! AI sẽ dùng dữ liệu mới khi bạn quay lại trang Dự báo.")
                st.rerun()
        except Exception as exc:
            st.error(f"Lỗi đọc tệp dữ liệu: {exc}")

st.divider()