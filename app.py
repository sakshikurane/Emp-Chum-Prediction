import os
import pickle
import warnings
 
import pandas as pd
from flask import Flask, render_template_string, request
 
warnings.filterwarnings("ignore")
 
app = Flask(__name__)
 
# --------------------------------------------------------------------------
# Load model
# --------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "navie.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
 
FEATURES = list(model.feature_names_in_)   # exact column order the model expects
CLASSES = [int(c) for c in model.classes_]  # [0, 1]  ->  0 = stays, 1 = leaves
 
# --------------------------------------------------------------------------
# Label-encoded categories (alphabetical, like sklearn's LabelEncoder).
# Position in each list = the number the model receives. Rename if yours differ.
# --------------------------------------------------------------------------
EDUCATION = ["Bachelors", "Masters", "PhD"]
CITY      = ["Bangalore", "New Delhi", "Pune"]
GENDER    = ["Female", "Male"]
BENCHED   = ["No", "Yes"]
 
 
def opts(lst):
    return list(enumerate(lst))
 
 
# --------------------------------------------------------------------------
# Form definition
# --------------------------------------------------------------------------
SECTIONS = [
    {"title": "👤 Employee Profile", "fields": [
        {"name": "Education", "label": "Education", "kind": "select", "options": opts(EDUCATION), "default": 0},
        {"name": "Gender", "label": "Gender", "kind": "select", "options": opts(GENDER), "default": 1},
        {"name": "Age", "label": "Age", "kind": "range", "min": 20, "max": 45, "step": 1, "default": 30},
        {"name": "City", "label": "City", "kind": "select", "options": opts(CITY), "default": 0},
    ]},
    {"title": "💼 Job Details", "fields": [
        {"name": "JoiningYear", "label": "Joining Year", "kind": "select",
         "options": [(y, str(y)) for y in range(2012, 2019)], "default": 2015},
        {"name": "PaymentTier", "label": "Payment Tier", "kind": "select",
         "options": [(1, "Tier 1"), (2, "Tier 2"), (3, "Tier 3")], "default": 3},
        {"name": "ExperienceInCurrentDomain", "label": "Experience in Current Domain (years)",
         "kind": "range", "min": 0, "max": 7, "step": 1, "default": 3},
        {"name": "EverBenched", "label": "Ever Benched (kept without project)", "kind": "select",
         "options": opts(BENCHED), "default": 0},
    ]},
]
 
 
# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def risk_info(p_leave: float) -> dict:
    """p_leave in 0-100."""
    if p_leave >= 60:
        return {"level": "High Risk", "verdict": "Likely to Leave", "icon": "🚨", "color": "#f43f5e",
                "msg": "This profile closely matches employees who left. Act early with a retention plan."}
    if p_leave >= 35:
        return {"level": "Medium Risk", "verdict": "Uncertain", "icon": "⚠️", "color": "#f59e0b",
                "msg": "Mixed signals. Stay in touch and address any open concerns."}
    return {"level": "Low Risk", "verdict": "Likely to Stay", "icon": "✅", "color": "#14b8a6",
            "msg": "This profile matches employees who tend to stay. Keep engagement strong."}
 
 
def build_tips(v: dict, p_leave: float) -> list:
    tips = []
    if int(v["EverBenched"]) == 1:
        tips.append("🪑 Past bench time is a strong attrition signal, so assign meaningful projects.")
    if int(v["PaymentTier"]) != 3:
        tips.append("💰 Review compensation: this payment tier shows a higher leave rate.")
    if int(v["JoiningYear"]) >= 2017:
        tips.append("🌱 Recent joiners leave more often, so strengthen onboarding and mentoring.")
    if CITY[int(v["City"])] == "Pune":
        tips.append("📍 This location has shown higher attrition, so check local engagement.")
    if int(v["ExperienceInCurrentDomain"]) <= 1:
        tips.append("🎯 Offer clear growth paths and training early in the domain.")
    if not tips:
        tips.append("🤝 Keep up regular check-ins, recognition and career conversations.")
    return tips[:4]
 
 
# --------------------------------------------------------------------------
# HTML template
# --------------------------------------------------------------------------
TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Employee Attrition Predictor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
  :root{
    --bg1:#06202b; --bg2:#14123a; --card:rgba(255,255,255,.07); --border:rgba(255,255,255,.14);
    --text:#eef6f8; --muted:#9db8c2; --a1:#2dd4bf; --a2:#fb923c; --a3:#a78bfa;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{
    font-family:'Outfit',system-ui,sans-serif;color:var(--text);min-height:100vh;padding:34px 16px 60px;
    background:
      radial-gradient(800px 480px at 0% 0%, rgba(45,212,191,.28), transparent 60%),
      radial-gradient(800px 520px at 100% 5%, rgba(251,146,60,.22), transparent 60%),
      radial-gradient(700px 500px at 50% 115%, rgba(167,139,250,.22), transparent 60%),
      linear-gradient(160deg,var(--bg1),var(--bg2));
    background-attachment:fixed;
  }
  .wrap{max-width:980px;margin:0 auto}
  header{text-align:center;margin-bottom:30px}
  .badge{display:inline-block;padding:6px 16px;border-radius:999px;font-size:12px;font-weight:600;
    letter-spacing:.08em;text-transform:uppercase;color:var(--a1);
    background:rgba(45,212,191,.12);border:1px solid rgba(45,212,191,.35)}
  h1{font-size:clamp(28px,5vw,46px);font-weight:800;margin:14px 0 8px;
    background:linear-gradient(90deg,#fff,var(--a1) 45%,var(--a2));
    -webkit-background-clip:text;background-clip:text;color:transparent}
  header p{color:var(--muted);max-width:560px;margin:0 auto;font-size:15px}
 
  /* result */
  .result{margin-bottom:28px;padding:30px;border-radius:24px;display:flex;gap:34px;align-items:center;
    justify-content:center;flex-wrap:wrap;
    background:linear-gradient(135deg,color-mix(in srgb,var(--rc) 20%,transparent),rgba(255,255,255,.04));
    border:1px solid color-mix(in srgb,var(--rc) 55%,transparent);
    box-shadow:0 0 60px color-mix(in srgb,var(--rc) 22%,transparent);animation:pop .5s ease}
  @keyframes pop{from{opacity:0;transform:translateY(14px) scale(.98)}to{opacity:1;transform:none}}
  .ring{position:relative;width:190px;height:190px}
  .ring svg{transform:rotate(-90deg)}
  .ring .bg{stroke:rgba(255,255,255,.12)}
  .ring .fg{stroke:var(--rc);stroke-linecap:round;stroke-dasharray:502;stroke-dashoffset:502;
    animation:fill 1.2s ease forwards;filter:drop-shadow(0 0 8px var(--rc))}
  @keyframes fill{to{stroke-dashoffset:calc(502 - 502 * var(--pct) / 100)}}
  .ring .center{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center}
  .ring .num{font-size:42px;font-weight:800;line-height:1}
  .ring .of{font-size:12px;color:var(--muted);margin-top:4px}
  .info{max-width:440px}
  .level{display:inline-block;font-size:13px;font-weight:700;padding:4px 14px;border-radius:999px;
    background:var(--rc);color:#06121a;margin-bottom:10px}
  .info h2{font-size:28px;margin-bottom:6px}
  .info p.msg{color:#d3e4ea;font-size:14px;margin-bottom:14px}
  .bars{margin-bottom:14px}
  .bar-row{display:flex;align-items:center;gap:10px;margin:7px 0;font-size:13px}
  .bar-row .n{width:50px;color:var(--muted)}
  .track{flex:1;height:10px;border-radius:99px;background:rgba(255,255,255,.1);overflow:hidden}
  .fill{height:100%;border-radius:99px}
  .bar-row .v{width:48px;text-align:right;font-weight:600}
  .tips{list-style:none;display:grid;gap:8px}
  .tips li{font-size:13px;padding:9px 12px;border-radius:12px;background:rgba(255,255,255,.07);
    border:1px solid var(--border)}
 
  .error{background:rgba(244,63,94,.15);border:1px solid rgba(244,63,94,.5);
    padding:14px 18px;border-radius:14px;margin-bottom:20px}
 
  /* form */
  form{display:grid;gap:20px}
  .grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:20px}
  .card{background:var(--card);border:1px solid var(--border);border-radius:20px;padding:22px;
    backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px)}
  .card h2{font-size:18px;margin-bottom:16px;padding-bottom:10px;border-bottom:1px solid var(--border)}
  .field{margin-bottom:16px}.field:last-child{margin-bottom:0}
  label{display:flex;justify-content:space-between;font-size:13px;font-weight:500;color:var(--muted);margin-bottom:7px}
  label output{color:var(--a2);font-weight:700}
  select{width:100%;padding:11px 12px;border-radius:12px;font:inherit;color:var(--text);
    background:rgba(5,14,24,.6);border:1px solid var(--border);outline:none;transition:.2s}
  select option{background:#10203a}
  select:focus{border-color:var(--a1);box-shadow:0 0 0 3px rgba(45,212,191,.25)}
  input[type=range]{width:100%;-webkit-appearance:none;height:6px;border-radius:99px;outline:none;
    background:linear-gradient(90deg,var(--a1),var(--a2))}
  input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:20px;height:20px;border-radius:50%;
    background:#fff;border:3px solid var(--a2);cursor:pointer;box-shadow:0 2px 8px rgba(0,0,0,.4)}
  input[type=range]::-moz-range-thumb{width:16px;height:16px;border-radius:50%;background:#fff;
    border:3px solid var(--a2);cursor:pointer}
 
  .actions{display:flex;gap:12px;justify-content:center;flex-wrap:wrap;margin-top:6px}
  button,.reset{padding:15px 38px;border-radius:14px;border:0;font:inherit;font-weight:700;font-size:16px;
    cursor:pointer;transition:.2s;text-decoration:none;display:inline-block}
  button{color:#06121a;background:linear-gradient(90deg,var(--a1),var(--a2));
    box-shadow:0 10px 30px rgba(45,212,191,.3)}
  button:hover{transform:translateY(-2px);box-shadow:0 14px 36px rgba(251,146,60,.4)}
  .reset{color:var(--text);background:var(--card);border:1px solid var(--border)}
  .reset:hover{background:rgba(255,255,255,.12)}
  footer{text-align:center;color:var(--muted);font-size:12px;margin-top:34px;max-width:620px;
    margin-left:auto;margin-right:auto}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <span class="badge">Naive Bayes · HR Analytics</span>
    <h1>Employee Attrition Predictor</h1>
    <p>Enter an employee's details to estimate whether they are likely to leave the company.</p>
  </header>
 
  {% if error %}<div class="error">⚠️ {{ error }}</div>{% endif %}
 
  {% if result %}
  <div class="result" style="--rc: {{ result.color }}; --pct: {{ result.p_leave }}">
    <div class="ring">
      <svg width="190" height="190" viewBox="0 0 190 190">
        <circle class="bg" cx="95" cy="95" r="80" fill="none" stroke-width="14"/>
        <circle class="fg" cx="95" cy="95" r="80" fill="none" stroke-width="14"/>
      </svg>
      <div class="center"><span class="num">{{ result.p_leave }}%</span><span class="of">chance of leaving</span></div>
    </div>
    <div class="info">
      <span class="level">{{ result.icon }} {{ result.level }}</span>
      <h2>{{ result.verdict }}</h2>
      <p class="msg">{{ result.msg }}</p>
      <div class="bars">
        <div class="bar-row"><span class="n">Stay</span>
          <div class="track"><div class="fill" style="width:{{ result.p_stay }}%;background:#14b8a6"></div></div>
          <span class="v">{{ result.p_stay }}%</span></div>
        <div class="bar-row"><span class="n">Leave</span>
          <div class="track"><div class="fill" style="width:{{ result.p_leave }}%;background:#f43f5e"></div></div>
          <span class="v">{{ result.p_leave }}%</span></div>
      </div>
      <ul class="tips">{% for t in result.tips %}<li>{{ t }}</li>{% endfor %}</ul>
    </div>
  </div>
  {% endif %}
 
  <form method="POST" action="/">
    <div class="grid2">
    {% for s in sections %}
      <div class="card">
        <h2>{{ s.title }}</h2>
        {% for f in s.fields %}
        <div class="field">
          {% set val = values.get(f.name, f.default) %}
          {% if f.kind == 'select' %}
            <label for="{{ f.name }}">{{ f.label }}</label>
            <select id="{{ f.name }}" name="{{ f.name }}">
              {% for code, text in f.options %}
              <option value="{{ code }}" {{ 'selected' if code|string == val|string }}>{{ text }}</option>
              {% endfor %}
            </select>
          {% else %}
            <label for="{{ f.name }}">{{ f.label }}<output id="o_{{ f.name }}">{{ val }}</output></label>
            <input type="range" id="{{ f.name }}" name="{{ f.name }}" min="{{ f.min }}" max="{{ f.max }}"
                   step="{{ f.step }}" value="{{ val }}"
                   oninput="document.getElementById('o_{{ f.name }}').textContent=this.value">
          {% endif %}
        </div>
        {% endfor %}
      </div>
    {% endfor %}
    </div>
    <div class="actions">
      <button type="submit">🔍 Predict Attrition</button>
      <a class="reset" href="/">↺ Reset</a>
    </div>
  </form>
  <footer>Built with Flask &amp; scikit-learn. This is a statistical estimate to support HR planning,
    so it should not be the sole basis for any decision about an individual.</footer>
</div>
</body>
</html>
"""
 
 
# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result, error, values = None, None, {}
 
    if request.method == "POST":
        values = request.form.to_dict()
        try:
            v = {name: float(values[name]) for name in FEATURES}
            X = pd.DataFrame([v])[FEATURES]
 
            proba = model.predict_proba(X)[0]
            p_leave = round(float(proba[CLASSES.index(1)]) * 100, 1)
            p_stay = round(100 - p_leave, 1)
 
            result = {"p_leave": p_leave, "p_stay": p_stay,
                      "tips": build_tips(v, p_leave), **risk_info(p_leave)}
        except (KeyError, ValueError) as e:
            error = f"Please fill every field with valid values ({e})."
        except Exception as e:  # noqa: BLE001
            error = f"Prediction failed: {e}"
 
    return render_template_string(
        TEMPLATE, sections=SECTIONS, values=values, result=result, error=error
    )
 
 
if __name__ == "__main__":
    app.run(debug=True)
 
