import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Loan Approval Prediction",
    layout="wide"
)

# ---------------- CSS ----------------
st.markdown("""
<style>
h1, h2 {
    color: #0f172a;
}
.result-box {
    padding: 16px;
    border-radius: 10px;
    font-size: 22px;
    font-weight: bold;
    text-align: center;
    margin-bottom: 15px;
}
.approved {
    background-color: #dcfce7;
    color: #166534;
}
.rejected {
    background-color: #fee2e2;
    color: #7f1d1d;
}
.small-text {
    font-size: 15px;
    color: #475569;
}
/* Increase sidebar width */
section[data-testid="stSidebar"] {
    width: 340px !important;
}

/* Push main content accordingly */
section[data-testid="stSidebar"] + div {
    margin-left: 340px !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("🏦 Loan Approval Prediction System")
st.write("Enter applicant details to check loan eligibility.")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("train_u6lujuX_CVtuZ9i.csv")

df = load_data()

# ---------------- PREPROCESS ----------------
df.drop("Loan_ID", axis=1, inplace=True)

for col in ["Gender", "Married", "Self_Employed"]:
    df[col].fillna(df[col].mode()[0], inplace=True)

for col in ["LoanAmount", "Credit_History"]:
    df[col].fillna(df[col].median(), inplace=True)

categorical_cols = ["Gender", "Married", "Self_Employed", "Property_Area"]

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

X = df[
    ["ApplicantIncome", "LoanAmount", "Credit_History",
     "Self_Employed", "Property_Area"]
]
y = df["Loan_Status"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = SVC(kernel="rbf", gamma="scale")
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))

# ---------------- SIDEBAR INPUTS ----------------
st.sidebar.header("📋 Applicant Details")

income = st.sidebar.slider(
    "Applicant Income",
    int(df["ApplicantIncome"].min()),
    int(df["ApplicantIncome"].max()),
    5000
)

loan_amt = st.sidebar.slider(
    "Loan Amount",
    int(df["LoanAmount"].min()),
    int(df["LoanAmount"].max()),
    150
)

credit = st.sidebar.selectbox("Credit History", ["Good", "Bad"])
employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

submit = st.sidebar.button("Check Loan Eligibility")

# ---------------- MAIN OUTPUT ----------------
if submit:
    credit_val = 1.0 if credit == "Good" else 0.0

    input_df = pd.DataFrame([{
        "ApplicantIncome": income,
        "LoanAmount": loan_amt,
        "Credit_History": credit_val,
        "Self_Employed": employed,
        "Property_Area": area
    }])

    for col in ["Self_Employed", "Property_Area"]:
        input_df[col] = label_encoders[col].transform(input_df[col])

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    # ---------- RESULT ----------
    if prediction == 1:
        st.markdown('<div class="result-box approved">✅ Loan Approved</div>', unsafe_allow_html=True)
        color = "green"
    else:
        st.markdown('<div class="result-box rejected">❌ Loan Rejected</div>', unsafe_allow_html=True)
        color = "red"

    st.markdown(
        f"<p class='small-text'>Model Used: SVM (RBF Kernel) | Accuracy: {accuracy:.2f}</p>",
        unsafe_allow_html=True
    )

    # ---------- GRAPH (DEFINE + USE TOGETHER) ----------
    st.subheader("📊 Applicant Position in Dataset")

    # ✅ DEFINE columns HERE
    col_plot, col_space = st.columns([2, 1])

    # ✅ USE columns HERE
    with col_plot:
        fig, ax = plt.subplots(figsize=(3.8, 3))

        ax.scatter(
            df["ApplicantIncome"],
            df["LoanAmount"],
            alpha=0.25,
            label="Existing Applicants"
        )

        ax.scatter(
            income,
            loan_amt,
            color=color,
            s=80,
            label="Your Application"
        )

        ax.set_xlabel("Applicant Income", fontsize=9)
        ax.set_ylabel("Loan Amount", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=8)

        plt.tight_layout()
        st.pyplot(fig)


    # -------- Explanation --------
    st.subheader("🧠 Decision Explanation")

    if prediction == 1:
        st.write(
            "The loan is approved due to a **good credit history** and a "
            "**reasonable loan amount compared to income**."
        )
    else:
        st.write(
            "The loan is rejected due to **poor credit history** or a "
            "**high loan amount relative to income**."
        )
