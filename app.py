from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parent
PAGES = ROOT / "app" / "pages"

st.set_page_config(
    page_title="FORESIGHT | Retail Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

css_path = ROOT / "assets" / "style.css"
st.markdown(
    f"<style>{css_path.read_text()}</style>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        "<div class='brand-mark'><span>◈</span>"
        "<div><strong>FORESIGHT</strong>"
        "<small>RETAIL INTELLIGENCE</small></div></div>",
        unsafe_allow_html=True,
    )

    st.caption(
        "AI-powered demand forecasting and inventory intelligence"
    )

    st.markdown(
        "<div class='sidebar-rule'></div>",
        unsafe_allow_html=True,
    )

    st.caption("DATASET")
    st.markdown(
        "**Online Retail II**  \nHistorical transaction data"
    )

    st.caption("INVENTORY NOTE")
    st.markdown(
        "Stock values are estimated from demand. "
        "No stock snapshots are present in the source dataset."
    )

    st.markdown(
        "<div class='sidebar-footer'>"
        "FORESIGHT v1.0<br>"
        "Current generated artifacts"
        "</div>",
        unsafe_allow_html=True,
    )

pages = {
    "Workspace": [
        st.Page(
            PAGES / "0_Overview.py",
            title="Overview",
            icon="🏠",
            default=True,
        ),
        st.Page(
            PAGES / "1_Sales_Analytics.py",
            title="Sales Analytics",
            icon="📊",
        ),
        st.Page(
            PAGES / "2_Forecast.py",
            title="Demand Forecast",
            icon="🔮",
        ),
        st.Page(
            PAGES / "3_Inventory.py",
            title="Inventory Intelligence",
            icon="📦",
        ),
        st.Page(
            PAGES / "4_Risk_Dashboard.py",
            title="Risk Dashboard",
            icon="⚠️",
        ),
        st.Page(
            PAGES / "5_Product_Details.py",
            title="Product Explorer",
            icon="🔍",
        ),
        st.Page(
            PAGES / "7_Model_Performance.py",
            title="Model Performance",
            icon="📈",
        ),
        st.Page(
            PAGES / "6_Executive_Summary.py",
            title="Executive Summary",
            icon="💼",
        ),
    ]
}

st.navigation(pages).run()