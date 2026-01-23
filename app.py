# app.py — Home (stand-alone): no imports from app_utils, fixes sidebar "app" → "Home",
# pretty gradient title + three centered boxed CTAs whose labels are INSIDE the boxes.

import streamlit as st

st.set_page_config(page_title="Rate My", layout="wide")

# ---------- Inline CSS + Sidebar label rename script ----------
st.markdown(
    """
    <style>
    .block-container{max-width:1140px;}

    /* Brand header */
    .brand{ display:flex; align-items:center; justify-content:center; gap:16px; margin:1.0rem 0 .25rem; }
    .logo{ width:56px; height:52px; flex:0 0 auto; }
    .brand h1{
      font-size:56px;margin:0;line-height:1;font-weight:900;letter-spacing:.3px;
      background:linear-gradient(90deg,#e74c3c 0%,#f39c12 50%,#2ecc71 100%);
      -webkit-background-clip:text;background-clip:text;color:transparent;
    }

    /* CTA boxes row */
    .cta{ padding:.25rem; filter:drop-shadow(0 10px 18px rgba(0,0,0,.35)); }
    .cta .stButton>button{
      width:100%; padding:18px 22px; border-radius:14px; font-weight:800; font-size:1.05rem;
      border:1px solid rgba(255,255,255,.14);
      background:linear-gradient(90deg,#e85d58, #f39c12, #2ecc71);
      color:#0e1015; box-shadow:0 1px 0 rgba(255,255,255,.06) inset;
      transition:transform .08s ease, box-shadow .16s ease, filter .12s ease;
    }
    .cta.dark .stButton>button{
      background:#171a1f; color:#e6e8eb; border-color:#2e3339;
    }
    .cta .stButton>button:hover{ transform:translateY(-1px); filter:saturate(1.06) brightness(1.05); }
    .cta.dark .stButton>button:hover{ border-color:#3a3f46; }

    /* nicer horizontal rule */
    .hr-lite{height:1px;background:linear-gradient(90deg,transparent,rgba(255,255,255,.08),transparent);border:0;margin:18px 0;}
    </style>

    <script>
    (function(){
      // Robustly rename the first sidebar nav item from "app" to "Home"
      function renameFirst(){
        try{
          const nav = document.querySelector('[data-testid="stSidebarNav"]');
          if(!nav) return;
          const first = nav.querySelector('ul li:first-child a p');
          if(first && first.textContent.trim().toLowerCase()==='app'){ first.textContent = 'Home'; }
        }catch(e){}
      }
      const obs = new MutationObserver(renameFirst);
      obs.observe(document.body,{childList:true,subtree:true});
      renameFirst();
    })();
    </script>
    """,
    unsafe_allow_html=True,
)

def inline_logo_svg() -> str:
    return """<svg class="logo" viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg" aria-label="Rate My">
  <defs>
    <linearGradient id="g" x1="0" x2="1" y1="1" y2="0">
      <stop offset="0%"  stop-color="#e74c3c"/>
      <stop offset="50%" stop-color="#f39c12"/>
      <stop offset="100%" stop-color="#2ecc71"/>
    </linearGradient>
    <radialGradient id="glow" cx="50%" cy="50%" r="60%">
      <stop offset="0%" stop-color="#ffffff" stop-opacity="0.12"/>
      <stop offset="100%" stop-color="#ffffff" stop-opacity="0"/>
    </radialGradient>
  </defs>
  <circle cx="60" cy="60" r="50" fill="none" stroke="url(#g)" stroke-width="7" stroke-linecap="round"/>
  <circle cx="60" cy="60" r="46" fill="url(#glow)"/>
  <rect x="34" y="66" width="11" height="20" rx="5" fill="#eef1f5"/>
  <rect x="52" y="52" width="11" height="34" rx="5" fill="#eef1f5"/>
  <rect x="70" y="38" width="11" height="48" rx="5" fill="#eef1f5"/>
  <path d="M32 70 Q44 61 56 50 T86 34" fill="none" stroke="url(#g)" stroke-width="4.5" stroke-linecap="round" stroke-linejoin="round"/>
</svg>"""

def brand_header(title: str):
    st.markdown(f'<div class="brand">{inline_logo_svg()}<h1>{title}</h1></div>', unsafe_allow_html=True)

# ---------- Header ----------
brand_header("Rate My")
st.markdown(
    "<div style='text-align:center;color:#9aa0a6;font-size:1.15rem;margin:.1rem 0 1.2rem;'>"
    "Pick a stock or your entire portfolio — we’ll rate it with clear, friendly explanations and charts."
    "</div>",
    unsafe_allow_html=True,
)

# ---------- Centered boxed CTAs (labels are inside the boxes) ----------
c1, c2, c3 = st.columns([1,1,1], gap="large")

with c1:
    st.markdown('<div class="cta">', unsafe_allow_html=True)
    if st.button("📈  Rate My Stock", use_container_width=True, key="cta_stock"):
        st.switch_page("pages/1_Rate_My_Stock.py")
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="cta dark">', unsafe_allow_html=True)
    if st.button("💼  Rate My Portfolio", use_container_width=True, key="cta_portfolio"):
        st.switch_page("pages/2_Rate_My_Portfolio.py")
    st.markdown('</div>', unsafe_allow_html=True)

with c3:
    st.markdown('<div class="cta dark">', unsafe_allow_html=True)
    if st.button("📊  Portfolio Tracker", use_container_width=True, key="cta_tracker"):
        st.switch_page("pages/3_Portfolio_Tracker.py")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Footer hint ----------
st.markdown("<div class='hr-lite'></div>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align:center;color:#9aa0a6;'>Your portfolios are saved to a local SQLite DB when signed in.</div>",
    unsafe_allow_html=True,
)
