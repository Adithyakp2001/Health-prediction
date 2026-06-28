%%writefile app.py
import streamlit as st
import pandas as pd
from datetime import date

from database import init_db, get_session, create_patient, get_all_patients, \
    get_patient_by_id, get_patient_by_email, update_patient, delete_patient
from ml_engine import predict_health
from validators import validate_patient_inputs

st.set_page_config(page_title="Health Prediction", page_icon="🏥",
                   layout="wide", initial_sidebar_state="expanded")
init_db()

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp { background: #f0f4f8; }

    /* ── Sidebar ── */
    div[data-testid="stSidebarContent"] {
        background: linear-gradient(180deg, #0f2942 0%, #0d4f6e 100%);
        padding-top: 1rem;
    }
    div[data-testid="stSidebarContent"] * { color: #e8f4f8 !important; }
    div[data-testid="stSidebarContent"] hr { border-color: rgba(255,255,255,0.15) !important; }

    /* ── Header ── */
    .main-header {
        background: linear-gradient(135deg, #0f2942 0%, #0d6e6e 60%, #0a8f7a 100%);
        padding: 2rem 2.5rem; border-radius: 16px; margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(15,41,66,0.25);
    }
    .main-header h1 { margin:0; font-size:1.9rem; font-weight:700; color:white !important; letter-spacing:-0.5px; }
    .main-header p  { margin:0.4rem 0 0; font-size:0.9rem; color:rgba(255,255,255,0.75) !important; }
    .header-badges  { display:flex; gap:0.6rem; flex-wrap:wrap; margin-top:1rem; }
    .badge {
        background:rgba(255,255,255,0.15); border:1px solid rgba(255,255,255,0.25);
        border-radius:20px; padding:0.25rem 0.75rem;
        font-size:0.75rem; color:white !important; font-weight:500;
    }

    /* ── Fix form labels ── */
    .stTextInput label, .stDateInput label,
    .stNumberInput label, .stSelectbox label,
    .stRadio label, .stCheckbox label {
        color: #0f2942 !important; font-weight:600 !important; font-size:0.85rem !important;
    }
    p, .stMarkdown p { color: #334155; }
    .stCaption, .stCaption p { color:#64748b !important; font-size:0.78rem !important; }

    /* ── Metrics ── */
    div[data-testid="metric-container"] {
        background:white; border:1px solid #e2e8f0; border-radius:12px;
        padding:1.2rem 1.5rem !important; box-shadow:0 1px 4px rgba(0,0,0,0.06);
    }
    [data-testid="stMetricLabel"] p {
        color:#64748b !important; font-size:0.75rem !important;
        font-weight:700 !important; text-transform:uppercase; letter-spacing:0.05em;
    }
    [data-testid="stMetricValue"] { color:#0f2942 !important; font-size:1.7rem !important; font-weight:700 !important; }

    /* ── Cards ── */
    .card {
        background:white; border-radius:14px; border:1px solid #e2e8f0;
        padding:1.5rem; box-shadow:0 1px 6px rgba(0,0,0,0.05); margin-bottom:1rem;
    }
    .card-title {
        font-size:0.7rem; font-weight:700; text-transform:uppercase;
        letter-spacing:0.08em; color:#64748b; margin-bottom:0.75rem;
    }

    /* ── Risk report box ── */
    .risk-box {
        background:#f8fafc;
        border:1px solid #cbd5e1;
        border-left:5px solid #64748b;
        border-radius:12px;
        padding:1.5rem 1.75rem;
        white-space:pre-wrap;
        font-family:'JetBrains Mono', monospace;
        font-size:0.78rem;
        color:#1e293b !important;
        line-height:1.85;
        margin-top:0.75rem;
        letter-spacing:0.01em;
    }
    .risk-low  { border-left-color:#10b981 !important; background:#f0fdf4 !important; border-color:#86efac !important; color:#052e16 !important; }
    .risk-mild { border-left-color:#f59e0b !important; background:#fffbeb !important; border-color:#fde68a !important; color:#451a03 !important; }
    .risk-high { border-left-color:#ef4444 !important; background:#fef2f2 !important; border-color:#fca5a5 !important; color:#450a0a !important; }

    /* ── Biomarker pills ── */
    .biomarker-row { display:flex; gap:0.75rem; flex-wrap:wrap; margin:0.75rem 0; }
    .bio-pill {
        border-radius:10px; padding:0.65rem 1.1rem;
        display:flex; flex-direction:column; align-items:center; min-width:130px;
        box-shadow:0 1px 4px rgba(0,0,0,0.06);
    }
    .bio-pill .blabel  { font-size:0.62rem; font-weight:700; opacity:0.7; text-transform:uppercase; letter-spacing:0.06em; }
    .bio-pill .bvalue  { font-size:1.15rem; font-weight:700; margin:0.2rem 0; }
    .bio-pill .bstatus { font-size:0.68rem; font-weight:600; }
    .bio-normal { background:#f0fdf4; color:#166534; border:1px solid #86efac; }
    .bio-pre    { background:#fffbeb; color:#92400e; border:1px solid #fde68a; }
    .bio-danger { background:#fef2f2; color:#991b1b; border:1px solid #fca5a5; }

    /* ── Section & analysis titles ── */
    .section-title {
        font-size:1.05rem; font-weight:700; color:#0f2942;
        margin:1.5rem 0 0.75rem; display:flex; align-items:center; gap:0.5rem;
    }
    .analysis-title {
        font-size:0.82rem; font-weight:700; color:#0f2942;
        margin:1rem 0 0.3rem; display:flex; align-items:center; gap:0.4rem;
        text-transform:uppercase; letter-spacing:0.05em;
    }

    /* ── Inputs ── */
    .stTextInput input, .stDateInput input, .stNumberInput input {
        border-radius:8px !important; border:1.5px solid #e2e8f0 !important;
        background:#f8fafc !important; color:#0f2942 !important; font-weight:500 !important;
    }
    .stTextInput input:focus, .stDateInput input:focus, .stNumberInput input:focus {
        border-color:#0d6e6e !important; box-shadow:0 0 0 3px rgba(13,110,110,0.1) !important;
    }

    /* ── Buttons ── */
    .stButton > button, .stFormSubmitButton > button {
        background:linear-gradient(135deg, #0f2942, #0d6e6e) !important;
        color:white !important; border:none !important; border-radius:10px !important;
        font-weight:600 !important; padding:0.6rem 1.5rem !important;
        transition:opacity 0.2s !important; font-size:0.9rem !important;
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover { opacity:0.88 !important; }

    [data-testid="stDataFrame"] { border-radius:12px; overflow:hidden; border:1px solid #e2e8f0; }
    .stAlert { border-radius:10px !important; }

    /* ── Patient info grid ── */
    .info-grid { display:grid; grid-template-columns:1fr 1fr; gap:0.75rem; }
    .info-item { background:#f8fafc; border-radius:8px; padding:0.65rem 0.9rem; border:1px solid #e2e8f0; }
    .info-item .info-label { font-size:0.62rem; font-weight:700; text-transform:uppercase; letter-spacing:0.07em; color:#64748b; }
    .info-item .info-value { font-size:0.92rem; font-weight:600; color:#0f2942; margin-top:0.15rem; }

    h3 { color:#0f2942 !important; font-weight:700 !important; }
    h4 { color:#1e3a5f !important; font-weight:600 !important; }
    #MainMenu, footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏥 Health Prediction System</h1>
    <p>AI/ML-Powered Diabetes Risk Assessment · Patient Blood Test Analysis</p>
    <div class="header-badges">
        <span class="badge">🤖 Random Forest · 100 Trees</span>
        <span class="badge">📊 Kaggle Diabetes Dataset 2024</span>
        <span class="badge">⚕️ Glucose · HbA1c · Cholesterol</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📋 Navigation")
    page = st.radio("", ["📊 Dashboard", "➕ Add Patient", "✏️ Edit / Update", "🗑️ Delete Patient"],
                    label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**⚙️ Tech Stack**")
    st.markdown("- Python 3.x\n- Streamlit\n- SQLAlchemy + SQLite\n- scikit-learn")
    st.markdown("---")
    st.markdown("**📁 Dataset**")
    st.markdown("Kaggle · rabieelkharoua\nDiabetes Health Dataset 2024")
    st.markdown("---")
    st.caption("Health Prediction System v2.0")


# ── Helpers ───────────────────────────────────────────────────────────────────
def calc_age(dob):
    today = date.today()
    return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))

def run_prediction(age, glucose, haemoglobin, cholesterol):
    return predict_health(float(age), float(glucose), float(haemoglobin), float(cholesterol))

def glucose_status(g):
    if g < 70:     return ("Low",          "bio-danger")
    elif g <= 99:  return ("Normal",        "bio-normal")
    elif g <= 125: return ("Pre-Diabetic",  "bio-pre")
    else:          return ("Diabetic",      "bio-danger")

def hba1c_status(h):
    if h < 5.7:    return ("Normal",        "bio-normal")
    elif h <= 6.4: return ("Pre-Diabetic",  "bio-pre")
    else:          return ("Diabetic",      "bio-danger")

def chol_status(c):
    if c < 200:    return ("Desirable",     "bio-normal")
    elif c <= 239: return ("Borderline",    "bio-pre")
    else:          return ("High Risk",     "bio-danger")

def risk_class(remarks):
    r = remarks.upper()
    if "HIGH RISK" in r and "MODERATE" not in r: return "risk-high"
    if "MODERATE" in r or "MILD" in r:           return "risk-mild"
    return "risk-low"

def render_biomarkers(glucose, haemoglobin, cholesterol):
    gs, gc = glucose_status(glucose)
    hs, hc = hba1c_status(haemoglobin)
    cs, cc = chol_status(cholesterol)
    st.markdown(f"""
    <div class="biomarker-row">
        <div class="bio-pill {gc}">
            <span class="blabel">Glucose</span>
            <span class="bvalue">{glucose} mg/dL</span>
            <span class="bstatus">● {gs}</span>
        </div>
        <div class="bio-pill {hc}">
            <span class="blabel">HbA1c</span>
            <span class="bvalue">{haemoglobin}%</span>
            <span class="bstatus">● {hs}</span>
        </div>
        <div class="bio-pill {cc}">
            <span class="blabel">Cholesterol</span>
            <span class="bvalue">{cholesterol} mg/dL</span>
            <span class="bstatus">● {cs}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_patient_info(p):
    st.markdown(f"""
    <div class="info-grid">
        <div class="info-item">
            <div class="info-label">Full Name</div>
            <div class="info-value">{p.full_name}</div>
        </div>
        <div class="info-item">
            <div class="info-label">Age</div>
            <div class="info-value">{p.age} yrs</div>
        </div>
        <div class="info-item">
            <div class="info-label">Date of Birth</div>
            <div class="info-value">{p.date_of_birth}</div>
        </div>
        <div class="info-item">
            <div class="info-label">Email</div>
            <div class="info-value">{p.email}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_analysis(remarks, glucose, haemoglobin, cholesterol):
    rc = risk_class(remarks)
    st.markdown('<div class="analysis-title">🩸 Biomarker Summary</div>', unsafe_allow_html=True)
    render_biomarkers(glucose, haemoglobin, cholesterol)
    st.markdown('<div class="analysis-title">📋 Clinical Assessment Report</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="risk-box {rc}">{remarks}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    db = get_session()
    patients = get_all_patients(db)
    db.close()

    if not patients:
        st.info("No patient records yet. Use ➕ Add Patient to get started.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Patients",  len(patients))
        c2.metric("Avg Glucose",     f"{sum(p.glucose for p in patients)/len(patients):.1f} mg/dL")
        c3.metric("Avg HbA1c",       f"{sum(p.haemoglobin for p in patients)/len(patients):.1f} %")
        c4.metric("Avg Cholesterol", f"{sum(p.cholesterol for p in patients)/len(patients):.1f} mg/dL")

        st.markdown('<div class="section-title">📋 All Patient Records</div>', unsafe_allow_html=True)
        df = pd.DataFrame([{
            "ID": p.id, "Full Name": p.full_name, "Age": p.age,
            "DOB": str(p.date_of_birth), "Email": p.email,
            "Glucose (mg/dL)": p.glucose,
            "HbA1c (%)": p.haemoglobin,
            "Cholesterol (mg/dL)": p.cholesterol,
        } for p in patients])
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown('<div class="section-title">🔍 Patient Detail & Analysis</div>', unsafe_allow_html=True)
        opts = {f"#{p.id} — {p.full_name}": p.id for p in patients}
        sel_id = opts[st.selectbox("Select a patient to view", list(opts.keys()))]
        db = get_session()
        p = get_patient_by_id(db, sel_id)
        db.close()
        if p:
            col1, col2 = st.columns([1, 1.6])
            with col1:
                st.markdown('<div class="card"><div class="card-title">👤 Patient Info</div>', unsafe_allow_html=True)
                render_patient_info(p)
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="card"><div class="card-title">🩸 Biomarkers & Clinical Report</div>', unsafe_allow_html=True)
                render_analysis(p.remarks or "No analysis available.", p.glucose, p.haemoglobin, p.cholesterol)
                st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ADD PATIENT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "➕ Add Patient":
    st.markdown('<div class="section-title">➕ Register New Patient</div>', unsafe_allow_html=True)

    with st.form("add_form", clear_on_submit=True):
        st.markdown("#### 👤 Personal Information")
        c1, c2 = st.columns(2)
        full_name = c1.text_input("Full Name *", placeholder="e.g. John Smith")
        email     = c2.text_input("Email Address *", placeholder="e.g. patient@email.com")
        dob       = st.date_input("Date of Birth *", value=date(1990, 1, 1),
                                   min_value=date(1900, 1, 1), max_value=date.today())
        st.markdown("#### 🩸 Blood Test Results")
        st.caption("Reference ranges based on WHO / ADA clinical guidelines")
        b1, b2, b3 = st.columns(3)
        glucose     = b1.number_input("Glucose (mg/dL) *", min_value=50.0, max_value=500.0,
                                       value=90.0, step=0.1, format="%.1f",
                                       help="Fasting glucose · Normal: 70–99 mg/dL")
        haemoglobin = b2.number_input("HbA1c (%) *", min_value=3.0, max_value=15.0,
                                       value=5.5, step=0.1, format="%.1f",
                                       help="Normal <5.7% | Pre-DM 5.7–6.4% | DM ≥6.5%")
        cholesterol = b3.number_input("Cholesterol (mg/dL) *", min_value=100.0, max_value=400.0,
                                       value=185.0, step=0.1, format="%.1f",
                                       help="Desirable <200 | Borderline 200–239 | High ≥240")
        submitted = st.form_submit_button("🔬 Analyse & Save Patient", use_container_width=True)

    if submitted:
        is_valid, errors = validate_patient_inputs(full_name, dob, email, glucose, haemoglobin, cholesterol)
        if not is_valid:
            for e in errors: st.error(e)
        else:
            db = get_session()
            existing = get_patient_by_email(db, email)
            if existing:
                st.error(f"A patient with email {email} already exists (ID #{existing.id}).")
                db.close()
            else:
                age = calc_age(dob)
                with st.spinner("🤖 Running clinical ML analysis..."):
                    remarks = run_prediction(age, glucose, haemoglobin, cholesterol)
                patient = create_patient(db, full_name, dob, email, age,
                                          float(glucose), float(haemoglobin), float(cholesterol), remarks)
                db.close()
                st.success(f"✅ {patient.full_name} registered successfully — ID #{patient.id} · Age {age}")
                st.markdown('<div class="card">', unsafe_allow_html=True)
                render_analysis(remarks, glucose, haemoglobin, cholesterol)
                st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EDIT / UPDATE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "✏️ Edit / Update":
    st.markdown('<div class="section-title">✏️ Edit Patient Record</div>', unsafe_allow_html=True)
    db = get_session()
    patients = get_all_patients(db)
    db.close()

    if not patients:
        st.info("No records to edit.")
    else:
        opts = {f"#{p.id} — {p.full_name} ({p.email})": p.id for p in patients}
        sel_id = opts[st.selectbox("Select patient to edit", list(opts.keys()))]
        db = get_session()
        p = get_patient_by_id(db, sel_id)
        db.close()

        if p:
            with st.form("edit_form"):
                st.markdown("#### 👤 Personal Information")
                c1, c2 = st.columns(2)
                full_name = c1.text_input("Full Name *", value=p.full_name)
                email     = c2.text_input("Email *", value=p.email)
                dob       = st.date_input("Date of Birth *", value=p.date_of_birth,
                                           min_value=date(1900, 1, 1), max_value=date.today())
                st.markdown("#### 🩸 Blood Test Results")
                b1, b2, b3 = st.columns(3)
                glucose     = b1.number_input("Glucose (mg/dL) *", value=float(p.glucose),
                                               min_value=50.0, max_value=500.0, step=0.1, format="%.1f")
                haemoglobin = b2.number_input("HbA1c (%) *", value=float(p.haemoglobin),
                                               min_value=3.0, max_value=15.0, step=0.1, format="%.1f")
                cholesterol = b3.number_input("Cholesterol (mg/dL) *", value=float(p.cholesterol),
                                               min_value=100.0, max_value=400.0, step=0.1, format="%.1f")
                update_btn = st.form_submit_button("💾 Update & Re-Analyse", use_container_width=True)

            if update_btn:
                is_valid, errors = validate_patient_inputs(full_name, dob, email, glucose, haemoglobin, cholesterol)
                if not is_valid:
                    for e in errors: st.error(e)
                else:
                    db = get_session()
                    ex = get_patient_by_email(db, email)
                    if ex and ex.id != sel_id:
                        st.error(f"Email {email} already belongs to another patient.")
                        db.close()
                    else:
                        age = calc_age(dob)
                        with st.spinner("🤖 Running clinical ML analysis..."):
                            remarks = run_prediction(age, glucose, haemoglobin, cholesterol)
                        update_patient(db, sel_id, full_name=full_name, date_of_birth=dob,
                                        email=email, age=age, glucose=float(glucose),
                                        haemoglobin=float(haemoglobin),
                                        cholesterol=float(cholesterol), remarks=remarks)
                        db.close()
                        st.success(f"✅ Patient #{sel_id} updated successfully!")
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        render_analysis(remarks, glucose, haemoglobin, cholesterol)
                        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DELETE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗑️ Delete Patient":
    st.markdown('<div class="section-title">🗑️ Delete Patient Record</div>', unsafe_allow_html=True)
    db = get_session()
    patients = get_all_patients(db)
    db.close()

    if not patients:
        st.info("No records to delete.")
    else:
        opts = {f"#{p.id} — {p.full_name} ({p.email})": p.id for p in patients}
        sel_id = opts[st.selectbox("Select patient to delete", list(opts.keys()))]
        db = get_session()
        p = get_patient_by_id(db, sel_id)
        db.close()

        if p:
            st.markdown('<div class="card"><div class="card-title">👤 Patient to be Deleted</div>', unsafe_allow_html=True)
            render_patient_info(p)
            st.markdown('</div>', unsafe_allow_html=True)
            st.warning(f"⚠️ You are about to permanently delete **{p.full_name}** (ID #{p.id}). This cannot be undone.")
            confirm = st.checkbox("I understand and confirm this deletion")
            if st.button("🗑️ Delete Patient", disabled=not confirm):
                db = get_session()
                delete_patient(db, sel_id)
                db.close()
                st.success(f"✅ {p.full_name}'s record has been deleted.")
                st.rerun()
