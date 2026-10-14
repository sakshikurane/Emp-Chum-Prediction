import os
import pickle
from flask import Flask, render_template_string, request
import numpy as np

app = Flask(__name__)

# Safely load the trained model with error handling
MODEL_PATH = "navie.pkl"
model = None

if os.path.exists(MODEL_PATH):
  try:
    with open(MODEL_PATH, "rb") as f:
      model = pickle.load(f)
  except Exception as e:
    print(f"Error loading model: {e}")
else:
  print(
      f"Warning: '{MODEL_PATH}' not found in the directory."
      f" {os.getcwd()}"
  )

# HTML template with a sleek, modern UI layout
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Employee Prediction Portal</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #4f46e5;
            --primary-hover: #4338ca;
            --bg-gradient: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            --card-bg: #ffffff;
            --text-main: #1f2937;
            --text-muted: #6b7280;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }

        body {
            background: var(--bg-gradient);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }

        .container {
            background: var(--card-bg);
            width: 100%;
            max-width: 650px;
            padding: 40px;
            border-radius: 16px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        }

        h2 {
            color: var(--text-main);
            font-weight: 600;
            margin-bottom: 8px;
            font-size: 24px;
            text-align: center;
        }

        p.subtitle {
            color: var(--text-muted);
            text-align: center;
            margin-bottom: 30px;
            font-size: 14px;
        }

        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }

        .form-group {
            display: flex;
            flex-direction: column;
        }

        .form-group.full-width {
            grid-column: span 2;
        }

        label {
            font-size: 13px;
            font-weight: 500;
            color: var(--text-main);
            margin-bottom: 6px;
        }

        input, select {
            padding: 10px 14px;
            border: 1px solid #d1d5db;
            border-radius: 8px;
            font-size: 14px;
            outline: none;
            transition: all 0.2s ease;
            background-color: #f9fafb;
        }

        input:focus, select:focus {
            border-color: var(--primary);
            background-color: #fff;
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1);
        }

        button {
            grid-column: span 2;
            background-color: var(--primary);
            color: white;
            border: none;
            padding: 12px;
            font-size: 15px;
            font-weight: 500;
            border-radius: 8px;
            cursor: pointer;
            transition: background-color 0.2s;
            margin-top: 10px;
        }

        button:hover {
            background-color: var(--primary-hover);
        }

        .result-card {
            margin-top: 25px;
            padding: 16px;
            border-radius: 8px;
            background-color: #e0e7ff;
            color: #312e81;
            text-align: center;
            font-weight: 500;
            font-size: 16px;
        }

        .error-card {
            margin-top: 25px;
            padding: 16px;
            border-radius: 8px;
            background-color: #fee2e2;
            color: #991b1b;
            text-align: center;
            font-weight: 500;
            font-size: 15px;
        }

        @media (max-width: 600px) {
            .form-grid {
                grid-template-columns: 1fr;
            }
            .form-group.full-width {
                grid-column: span 1;
            }
            button {
                grid-column: span 1;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>Employee Analytics Portal</h2>
        <p class="subtitle">Enter the employee details below to run the classification model</p>
        
        <form method="POST" action="/">
            <div class="form-grid">
                <div class="form-group">
                    <label for="Education">Education</label>
                    <select name="Education" id="Education" required>
                        <option value="0">Bachelors</option>
                        <option value="1">Masters</option>
                        <option value="2">PHD</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="JoiningYear">Joining Year</label>
                    <input type="number" name="JoiningYear" id="JoiningYear" placeholder="e.g., 2018" required>
                </div>

                <div class="form-group">
                    <label for="City">City</label>
                    <select name="City" id="City" required>
                        <option value="0">Bangalore</option>
                        <option value="1">New Delhi</option>
                        <option value="2">Pune</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="PaymentTier">Payment Tier</label>
                    <select name="PaymentTier" id="PaymentTier" required>
                        <option value="1">Tier 1</option>
                        <option value="2">Tier 2</option>
                        <option value="3">Tier 3</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="Age">Age</label>
                    <input type="number" name="Age" id="Age" placeholder="e.g., 28" required>
                </div>

                <div class="form-group">
                    <label for="Gender">Gender</label>
                    <select name="Gender" id="Gender" required>
                        <option value="0">Male</option>
                        <option value="1">Female</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="EverBenched">Ever Benched</label>
                    <select name="EverBenched" id="EverBenched" required>
                        <option value="0">No</option>
                        <option value="1">Yes</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="ExperienceInCurrentDomain">Experience in Domain (Years)</label>
                    <input type="number" name="ExperienceInCurrentDomain" id="ExperienceInCurrentDomain" placeholder="e.g., 3" required>
                </div>

                <button type="submit">Predict Status</button>
            </div>
        </form>

        {% if error_text %}
            <div class="error-card">
                {{ error_text }}
            </div>
        {% endif %}

        {% if prediction_text %}
            <div class="result-card">
                {{ prediction_text }}
            </div>
        {% endif %}
    </div>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():
  prediction_text = None
  error_text = None

  if request.method == "POST":
    if model is None:
      error_text = (
          "Error: The model file 'naive.pkl' is missing or could not be loaded"
          " on the server."
      )
    else:
      try:
        features = [
            float(request.form["Education"]),
            float(request.form["JoiningYear"]),
            float(request.form["City"]),
            float(request.form["PaymentTier"]),
            float(request.form["Age"]),
            float(request.form["Gender"]),
            float(request.form["EverBenched"]),
            float(request.form["ExperienceInCurrentDomain"]),
        ]

        final_features = np.array([features])
        prediction = model.predict(final_features)

        prediction_text = (
            f"Prediction Result: The estimated category is Class"
            f" {int(prediction[0])}"
        )
      except Exception as e:
        error_text = f"Error in processing input: {str(e)}"

  return render_template_string(
      HTML_TEMPLATE, prediction_text=prediction_text, error_text=error_text
  )


if __name__ == "__main__":
  app.run(host="0.0.0.0", port=5000)
