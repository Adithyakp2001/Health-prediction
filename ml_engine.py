import os, numpy as np, joblib
from datetime import date

MODEL_PATH  = "model/rf_health_model.pkl"
SCALER_PATH = "model/scaler.pkl"
_clf    = None
_scaler = None

def _load_model():
    global _clf, _scaler
    if _clf is None:
        if not os.path.exists(MODEL_PATH):
            import train_model
            train_model.train_and_save()
        _clf    = joblib.load(MODEL_PATH)
        _scaler = joblib.load(SCALER_PATH)

def _glucose_detail(g):
    if g < 70:     return ("Low — Hypoglycaemia",        "⚠️", "Below 70 mg/dL")
    elif g <= 99:  return ("Normal",                     "✅", "70–99 mg/dL")
    elif g <= 125: return ("Pre-Diabetic Range",         "⚠️", "100–125 mg/dL")
    else:          return ("Diabetic Range",             "🔴", "≥126 mg/dL")

def _hba1c_detail(h):
    if h < 5.7:    return ("Normal",                     "✅", "Below 5.7%")
    elif h <= 6.4: return ("Pre-Diabetic",               "⚠️", "5.7–6.4%")
    else:          return ("Diabetic",                   "🔴", "≥6.5%")

def _cholesterol_detail(c):
    if c < 200:    return ("Desirable",                  "✅", "Below 200 mg/dL")
    elif c <= 239: return ("Borderline High",            "⚠️", "200–239 mg/dL")
    else:          return ("High — Cardiovascular Risk", "🔴", "≥240 mg/dL")

def _age_context(age):
    if age < 30:   return "Low baseline metabolic risk. Routine annual screening advised."
    elif age < 45: return "Moderate risk window. Biennial screening recommended."
    elif age < 60: return "Elevated risk group. Annual screening essential."
    else:          return "High-risk age group. Regular check-ups every 6–12 months advised."

def predict_health(age, glucose, haemoglobin, cholesterol):
    _load_model()
    X        = np.array([[age, glucose, haemoglobin, cholesterol]], dtype=float)
    X_scaled = _scaler.transform(X)
    prediction = _clf.predict(X_scaled)[0]
    proba      = _clf.predict_proba(X_scaled)[0]
    risk_pct   = proba[1] * 100

    today = date.today().strftime("%d %B %Y")

    if prediction == 0 and risk_pct < 25:
        risk_level = "LOW RISK"
        risk_label = "No Diabetes Detected"
        risk_icon  = "✅"
        recommendation = (
            "All biomarkers are within acceptable clinical ranges.\n"
            "Maintain a balanced diet, regular physical activity (150 min/week),\n"
            "stay hydrated, and schedule an annual health check-up."
        )
    elif prediction == 0 and risk_pct < 50:
        risk_level = "MILD RISK"
        risk_label = "Borderline Values Detected"
        risk_icon  = "⚠️"
        recommendation = (
            "Some biomarkers are approaching threshold levels.\n"
            "Lifestyle modifications are advised — reduce refined carbohydrates,\n"
            "increase physical activity, and schedule a follow-up blood test in 3–6 months."
        )
    elif prediction == 1 and risk_pct < 75:
        risk_level = "MODERATE-HIGH RISK"
        risk_label = "Diabetes Indicators Present"
        risk_icon  = "🔶"
        recommendation = (
            "Multiple biomarkers indicate elevated diabetes risk.\n"
            "Consult a physician promptly for an HbA1c confirmation test,\n"
            "fasting plasma glucose test, and a full lipid panel review."
        )
    else:
        risk_level = "HIGH RISK"
        risk_label = "Strong Diabetes Indicators"
        risk_icon  = "🔴"
        recommendation = (
            "Critical biomarker levels detected across multiple parameters.\n"
            "Immediate medical consultation is strongly advised.\n"
            "Do not delay — early treatment significantly improves outcomes."
        )

    g_status, g_icon, g_range = _glucose_detail(glucose)
    h_status, h_icon, h_range = _hba1c_detail(haemoglobin)
    c_status, c_icon, c_range = _cholesterol_detail(cholesterol)

    report = (
        f"╔══════════════════════════════════════════════════════════╗\n"
        f"  {risk_icon}  ASSESSMENT RESULT: {risk_level}\n"
        f"     {risk_label}\n"
        f"╚══════════════════════════════════════════════════════════╝\n"
        f"\n"
        f"  ML Confidence    : {risk_pct:.1f}% diabetes risk\n"
        f"  Model            : Random Forest Classifier (100 trees)\n"
        f"  Report Date      : {today}\n"
        f"\n"
        f"──────────────────────────────────────────────────────────\n"
        f"  BIOMARKER ANALYSIS\n"
        f"──────────────────────────────────────────────────────────\n"
        f"  {g_icon} Glucose       : {glucose} mg/dL   →  {g_status} ({g_range})\n"
        f"  {h_icon} HbA1c         : {haemoglobin}%       →  {h_status} ({h_range})\n"
        f"  {c_icon} Cholesterol   : {cholesterol} mg/dL  →  {c_status} ({c_range})\n"
        f"  📅 Age           : {int(age)} yrs       →  {_age_context(int(age))}\n"
        f"\n"
        f"──────────────────────────────────────────────────────────\n"
        f"  CLINICAL RECOMMENDATION\n"
        f"──────────────────────────────────────────────────────────\n"
        f"  {recommendation}\n"
        f"\n"
        f"──────────────────────────────────────────────────────────\n"
        f"  ⚕️  DISCLAIMER\n"
        f"──────────────────────────────────────────────────────────\n"
        f"  This report is generated by an AI/ML system for\n"
        f"  informational purposes only. It does not substitute\n"
        f"  professional medical diagnosis or treatment.\n"
        f"  Always consult a qualified healthcare provider.\n"
    )
    return report
