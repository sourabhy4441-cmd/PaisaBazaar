# from pathlib import Path

# from flask import Flask, render_template, request
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import accuracy_score

# BASE_DIR = Path(__file__).resolve().parent
# DATASET_PATH = BASE_DIR / "dataset-2.csv"

# app = Flask(__name__)

# # Keep the original dataset features and Random Forest workflow.
# print("Loading dataset and training model...")
# if not DATASET_PATH.exists():
#     raise FileNotFoundError(
#         f"Could not find {DATASET_PATH.name}. Keep dataset-2.csv in the same folder as app.py."
#     )

# df = pd.read_csv(DATASET_PATH)
# features = [
#     "Age", "Annual_Income", "Monthly_Inhand_Salary",
#     "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate",
#     "Num_of_Loan", "Delay_from_due_date", "Num_of_Delayed_Payment",
#     "Outstanding_Debt", "Credit_Utilization_Ratio"
# ]
# required_columns = features + ["Credit_Score"]
# missing_columns = [column for column in required_columns if column not in df.columns]
# if missing_columns:
#     raise ValueError(f"Missing required dataset columns: {', '.join(missing_columns)}")

# df_model = df[required_columns].dropna().copy()
# le = LabelEncoder()
# df_model["Credit_Score_Encoded"] = le.fit_transform(df_model["Credit_Score"])
# X = df_model[features]
# y = df_model["Credit_Score_Encoded"]
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )
# model = RandomForestClassifier(n_estimators=50, random_state=42)
# model.fit(X_train, y_train)
# model_accuracy = accuracy_score(y_test, model.predict(X_test))
# print("Model trained successfully!")


# @app.route("/")
# def home():
#     total_customers = len(df)
#     avg_income = round(pd.to_numeric(df["Annual_Income"], errors="coerce").mean(), 2)
#     preview_data = df.head(10).to_html(
#         classes="table table-hover align-middle mb-0",
#         index=False,
#         border=0,
#         escape=True,
#         na_rep="—",
#     )
#     return render_template(
#         "index.html",
#         total_customers=total_customers,
#         avg_income=avg_income if pd.notna(avg_income) else 0,
#         table=preview_data,
#         dataset_columns=len(df.columns),
#         page="home",
#     )


# @app.route("/dashboard")
# def dashboard():
#     # High-level project and model metrics, separate from the dataset preview on Overview.
#     category_counts = df["Credit_Score"].dropna().astype(str).value_counts().to_dict()
#     total_labeled = int(df["Credit_Score"].notna().sum())
#     category_data = [
#         {
#             "name": name,
#             "count": int(count),
#             "percent": round((count / total_labeled) * 100, 1) if total_labeled else 0,
#         }
#         for name, count in sorted(category_counts.items(), key=lambda item: item[1], reverse=True)
#     ]
#     return render_template(
#         "dashboard.html",
#         total_customers=len(df),
#         model_accuracy=round(model_accuracy * 100, 2),
#         model_features=len(features),
#         total_categories=len(category_counts),
#         category_data=category_data,
#         page="dashboard",
#     )


# @app.route("/predict", methods=["GET", "POST"])
# def predict():
#     prediction_text = None
#     error_message = None
#     submitted_values = request.form if request.method == "POST" else {}

#     if request.method == "POST":
#         try:
#             # Keep the original HTML form field names and their feature order.
#             input_data = [[
#                 float(request.form["age"]),
#                 float(request.form["annual_income"]),
#                 float(request.form["monthly_salary"]),
#                 float(request.form["num_bank_accounts"]),
#                 float(request.form["num_credit_cards"]),
#                 float(request.form["interest_rate"]),
#                 float(request.form["num_loans"]),
#                 float(request.form["delay_days"]),
#                 float(request.form["delayed_payments"]),
#                 float(request.form["outstanding_debt"]),
#                 float(request.form["credit_utilization"]),
#             ]]
#             pred_encoded = model.predict(input_data)[0]
#             prediction_text = str(le.inverse_transform([pred_encoded])[0])
#         except (KeyError, TypeError, ValueError) as exc:
#             error_message = f"Please check all fields and enter valid numeric values. Details: {exc}"
#         except Exception as exc:
#             app.logger.exception("Credit score prediction failed")
#             error_message = "We couldn't complete the prediction. Please check your values and try again."

#     return render_template(
#         "predict.html",
#         prediction=prediction_text,
#         error_message=error_message,
#         submitted_values=submitted_values,
#         page="predict",
#     )


# if __name__ == "__main__":
#     app.run(debug=True)












from pathlib import Path

from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset-2.csv"

app = Flask(__name__)

# Keep the original dataset features and Random Forest workflow.
print("Loading dataset and training model...")
if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Could not find {DATASET_PATH.name}. Keep dataset-2.csv in the same folder as app.py."
    )

df = pd.read_csv(DATASET_PATH)
features = [
    "Age", "Annual_Income", "Monthly_Inhand_Salary",
    "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate",
    "Num_of_Loan", "Delay_from_due_date", "Num_of_Delayed_Payment",
    "Outstanding_Debt", "Credit_Utilization_Ratio"
]
required_columns = features + ["Credit_Score"]
missing_columns = [column for column in required_columns if column not in df.columns]
if missing_columns:
    raise ValueError(f"Missing required dataset columns: {', '.join(missing_columns)}")

df_model = df[required_columns].dropna().copy()
le = LabelEncoder()
df_model["Credit_Score_Encoded"] = le.fit_transform(df_model["Credit_Score"])
X = df_model[features]
y = df_model["Credit_Score_Encoded"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)
model_accuracy = accuracy_score(y_test, model.predict(X_test))
print("Model trained successfully!")


@app.route("/")
def home():
    # Marketing-style landing page: keep it distinct from dataset Overview.
    total_customers = len(df)
    avg_income = pd.to_numeric(df["Annual_Income"], errors="coerce").mean()
    return render_template(
        "home.html",
        total_customers=total_customers,
        avg_income=float(avg_income) if pd.notna(avg_income) else 0,
        dataset_columns=len(df.columns),
        page="home",
    )


@app.route("/overview")
def overview():
    # Dataset preview and summary statistics live on their own page.
    total_customers = len(df)
    avg_income = round(pd.to_numeric(df["Annual_Income"], errors="coerce").mean(), 2)
    preview_data = df.head(10).to_html(
        classes="table table-hover align-middle mb-0",
        index=False,
        border=0,
        escape=True,
        na_rep="—",
    )
    return render_template(
        "index.html",
        total_customers=total_customers,
        avg_income=avg_income if pd.notna(avg_income) else 0,
        table=preview_data,
        dataset_columns=len(df.columns),
        page="overview",
    )


@app.route("/dashboard")
def dashboard():
    # High-level project and model metrics, separate from the dataset preview on Overview.
    category_counts = df["Credit_Score"].dropna().astype(str).value_counts().to_dict()
    total_labeled = int(df["Credit_Score"].notna().sum())
    category_data = [
        {
            "name": name,
            "count": int(count),
            "percent": round((count / total_labeled) * 100, 1) if total_labeled else 0,
        }
        for name, count in sorted(category_counts.items(), key=lambda item: item[1], reverse=True)
    ]
    return render_template(
        "dashboard.html",
        total_customers=len(df),
        model_accuracy=round(model_accuracy * 100, 2),
        model_features=len(features),
        total_categories=len(category_counts),
        category_data=category_data,
        page="dashboard",
    )


@app.route("/purchase-planner", methods=["GET", "POST"])
def purchase_planner():
    """Estimate purchase timing using savings inputs only; credit model is not involved."""
    values = request.form if request.method == "POST" else request.args
    result = None
    error_message = None

    # Only calculate when the user submitted the form or supplied all query values.
    should_calculate = request.method == "POST" or all(
        key in request.args for key in ("price", "current_savings", "monthly_savings")
    )
    if should_calculate:
        try:
            product = str(values.get("product", "Other")).strip()[:60] or "Other"
            price = float(values.get("price", ""))
            current_savings = float(values.get("current_savings", "0"))
            monthly_savings = float(values.get("monthly_savings", ""))

            if not all(pd.notna(v) and v != float("inf") and v != float("-inf") for v in (price, current_savings, monthly_savings)):
                raise ValueError("Enter finite numeric amounts.")
            if price <= 0:
                raise ValueError("Target price must be greater than zero.")
            if current_savings < 0:
                raise ValueError("Current savings cannot be negative.")
            if monthly_savings <= 0:
                raise ValueError("Monthly savings must be greater than zero.")

            remaining = max(0.0, price - current_savings)
            months = 0 if remaining == 0 else int(__import__("math").ceil(remaining / monthly_savings))
            progress = min(100.0, round((current_savings / price) * 100, 1))
            result = {
                "product": product,
                "price": price,
                "current_savings": current_savings,
                "remaining": remaining,
                "months": months,
                "days": months * 30,
                "progress": progress,
            }
        except (TypeError, ValueError, OverflowError):
            error_message = "Enter a valid target price, current savings (zero or more), and monthly savings greater than zero."

    return render_template(
        "purchase_planner.html",
        values=values,
        result=result,
        error_message=error_message,
        page="purchase_planner",
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction_text = None
    error_message = None
    submitted_values = request.form if request.method == "POST" else {}

    if request.method == "POST":
        try:
            # Keep the original HTML form field names and their feature order.
            input_data = [[
                float(request.form["age"]),
                float(request.form["annual_income"]),
                float(request.form["monthly_salary"]),
                float(request.form["num_bank_accounts"]),
                float(request.form["num_credit_cards"]),
                float(request.form["interest_rate"]),
                float(request.form["num_loans"]),
                float(request.form["delay_days"]),
                float(request.form["delayed_payments"]),
                float(request.form["outstanding_debt"]),
                float(request.form["credit_utilization"]),
            ]]
            pred_encoded = model.predict(input_data)[0]
            prediction_text = str(le.inverse_transform([pred_encoded])[0])
        except (KeyError, TypeError, ValueError) as exc:
            error_message = f"Please check all fields and enter valid numeric values. Details: {exc}"
        except Exception as exc:
            app.logger.exception("Credit score prediction failed")
            error_message = "We couldn't complete the prediction. Please check your values and try again."

    return render_template(
        "predict.html",
        prediction=prediction_text,
        error_message=error_message,
        submitted_values=submitted_values,
        page="predict",
    )


if __name__ == "__main__":
    app.run(debug=True)
