import streamlit as st
import sqlite3
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import os
from fpdf import FPDF

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(page_title="Cognitive Disorder Screening", layout="centered")

DB_PATH = "responses.db"
MODEL_DIR = "models"

# ----------------------------
# DATABASE SETUP
# ----------------------------
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

# NOTE: Database schema is technically "Legacy" (has 30 q columns). 
# We will just insert 0 for the missing sleep columns to keep DB compatible 
# without complex migration scripts.
cursor.execute("""
CREATE TABLE IF NOT EXISTS responses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    grade TEXT, age INTEGER, gender TEXT,
    q1 INTEGER, q2 INTEGER, q3 INTEGER, q4 INTEGER, q5 INTEGER,
    q6 INTEGER, q7 INTEGER, q8 INTEGER, q9 INTEGER, q10 INTEGER,
    q11 INTEGER, q12 INTEGER, q13 INTEGER, q14 INTEGER, q15 INTEGER,
    q16 INTEGER, q17 INTEGER, q18 INTEGER, q19 INTEGER, q20 INTEGER,
    q21 INTEGER, q22 INTEGER, q23 INTEGER, q24 INTEGER, q25 INTEGER,
    q26 INTEGER, q27 INTEGER, q28 INTEGER, q29 INTEGER, q30 INTEGER,
    submitted_at TEXT
)
""")
conn.commit()

# ----------------------------
# LOAD MODELS
# ----------------------------
def load_model(name):
    path = os.path.join(MODEL_DIR, name)
    if not os.path.exists(path):
        st.error(f"Model file not found: {path}. Please run train_final.py first.")
        st.stop()
    return pickle.load(open(path, "rb"))

# CHANGE HERE: Load whichever model you want to use for the app (e.g., RF or XGB)
rf_risk = load_model("RF_risk.pkl") 

severity_models = {
    "ADHD": load_model("rf_ADHD_sev.pkl"),
    "ASD": load_model("rf_ASD_sev.pkl"),
    "SPCD": load_model("rf_SPCD_sev.pkl"),
    "DEP": load_model("rf_DEP_sev.pkl"),
    "ANX": load_model("rf_ANX_sev.pkl")
}

RISK_ORDER = ["ADHD", "ASD", "SPCD", "DEP", "ANX"]
SEVERITY_DECODER = {0: "Low", 1: "Medium", 2: "High"}

# ----------------------------
# PDF GENERATION
# ----------------------------
def create_pdf(student_data, results, questions, user_choices):
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 15)
            self.cell(0, 10, 'Cognitive Disorder Screening Report', 0, 1, 'C')
            self.set_font('Arial', 'I', 10)
            self.cell(0, 10, 'Confidential - Academic Use Only', 0, 1, 'C')
            self.line(10, 30, 200, 30)
            self.ln(10)
        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    pdf = PDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Student Profile
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Student Profile", 0, 1)
    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 6, f"Grade: {student_data['grade']}", 0, 1)
    pdf.cell(0, 6, f"Age: {student_data['age']}", 0, 1)
    pdf.cell(0, 6, f"Gender: {student_data['gender']}", 0, 1)
    pdf.cell(0, 6, f"Date: {datetime.now().strftime('%Y-%m-%d')}", 0, 1)
    pdf.ln(5)

    # Results
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Screening Results", 0, 1)
    pdf.set_font("Arial", '', 11)
    if not results:
        pdf.set_text_color(0, 100, 0)
        pdf.cell(0, 8, "No significant cognitive disorder risk detected.", 0, 1)
    else:
        for disorder, severity in results.items():
            pdf.set_text_color(200, 0, 0) if severity == "High" else pdf.set_text_color(0, 0, 0)
            pdf.cell(0, 8, f"- {disorder}: {severity} Severity", 0, 1)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)

    # Table
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Detailed Responses", 0, 1)
    pdf.set_font("Arial", 'B', 10)
    pdf.set_fill_color(230, 230, 230)
    
    # Widths
    w_no = 15; w_q = 135; w_ans = 40

    pdf.cell(w_no, 8, "No.", 1, 0, 'C', 1)
    pdf.cell(w_q, 8, "Question", 1, 0, 'L', 1)
    pdf.cell(w_ans, 8, "Answer", 1, 1, 'C', 1)
    pdf.set_font("Arial", '', 9)

    for i, (q, ans) in enumerate(zip(questions, user_choices), 1):
        x_start = pdf.get_x()
        y_start = pdf.get_y()
        if y_start > 250:
            pdf.add_page()
            x_start = pdf.get_x(); y_start = pdf.get_y()
            pdf.set_font("Arial", 'B', 10)
            pdf.cell(w_no, 8, "No.", 1, 0, 'C', 1)
            pdf.cell(w_q, 8, "Question", 1, 0, 'L', 1)
            pdf.cell(w_ans, 8, "Answer", 1, 1, 'C', 1)
            pdf.set_font("Arial", '', 9)
            y_start = pdf.get_y()

        pdf.set_xy(x_start + w_no, y_start)
        pdf.multi_cell(w_q, 6, q, border=1, align='L')
        y_end = pdf.get_y()
        row_height = y_end - y_start
        
        pdf.set_xy(x_start, y_start)
        pdf.cell(w_no, row_height, str(i), border=1, ln=0, align='C')
        pdf.set_xy(x_start + w_no + w_q, y_start)
        pdf.cell(w_ans, row_height, ans, border=1, ln=1, align='C')

    return pdf.output(dest='S').encode('latin-1')

# ----------------------------
# ADMIN
# ----------------------------
with st.sidebar:
    st.header("Admin / Instructor")
    if st.checkbox("Show Stored Data"):
        df_hist = pd.read_sql_query("SELECT * FROM responses", conn)
        st.dataframe(df_hist)

# ----------------------------
# MAIN UI
# ----------------------------
st.title("Cognitive Disorder Screening Tool")
st.caption("Developed for MMIT by Dr Monika, Sankalp and Team")
st.warning("⚠️ **Academic Use Only • Not a Diagnostic Tool**")

grade = st.selectbox("Grade / Year of Study", ["", "UG", "PG", "Higher"])
age = st.number_input("Age", min_value=18, max_value=60, step=1)
gender = st.selectbox("Gender", ["", "Male", "Female", "Other"])

# UPDATED: REMOVED SLEEP QUESTIONS
questions = [
    "I have difficulty starting tasks that require a lot of thinking.",
    "I lose focus during lectures, meetings, or reading.",
    "I forget deadlines or appointments even when they are important.",
    "I struggle to organize my work or study materials.",
    "I postpone until the last moment, even for important tasks.",
    "I feel mentally restless or unable to slow my thoughts.",
    "I make careless mistakes even when I know the material.",
    "I find it hard to know when it is my turn to speak in conversations.",
    "I struggle to understand jokes, sarcasm, or indirect hints.",
    "I feel unsure how much detail to give when explaining something.",
    "I find group discussions confusing or exhausting.",
    "I prefer clear rules and predictable routines.",
    "I miss social cues like tone of voice or facial expressions.",
    "People tell me I sound blunt, awkward, or unclear when I speak.",
    "I struggle to adjust how I speak depending on who I am talking to.",
    "I find it difficult to stay on topic in conversations.",
    "I misunderstand what others expect from me socially.",
    "I feel little interest or pleasure in doing things.",
    "I feel down, hopeless, or emotionally numb.",
    "I feel tired or low on energy most days.",
    "I feel like I am not good enough or have failed.",
    "I have difficulty concentrating because of low mood.",
    "I feel nervous, anxious, or on edge.",
    "I worry too much about academic or social situations.",
    "I find it hard to relax, even when I have time.",
    "My anxiety interferes with my studies or relationships.",
    "I avoid situations because they make me anxious."
]

options = {"Never": 0, "Rarely": 1, "Sometimes": 2, "Often": 3, "Very Often": 4}
responses = []
user_text_answers = []

with st.form("questionnaire_form"):
    for i, q in enumerate(questions, start=1):
        choice = st.radio(f"Q{i}. {q}", list(options.keys()), horizontal=True, key=f"q{i}")
        user_text_answers.append(choice)
        score = options[choice]
        responses.append(score)
    
    submitted = st.form_submit_button("Submit Screening")

if submitted:
    if not grade or not gender:
        st.error("Please fill in Grade and Gender fields.")
        st.stop()

    submitted_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # DB Compat: Pad with 3 zeros for the removed Sleep questions
    db_responses = responses + [0, 0, 0] 
    values = [grade, age, gender] + db_responses + [submitted_at]
    
    cursor.execute("""
    INSERT INTO responses (
        grade, age, gender,
        q1, q2, q3, q4, q5, q6, q7, q8, q9, q10,
        q11, q12, q13, q14, q15, q16, q17, q18, q19, q20,
        q21, q22, q23, q24, q25, q26, q27, q28, q29, q30,
        submitted_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, values)
    conn.commit()
    st.success("Responses saved successfully.")

    st.divider()
    st.subheader("Results Analysis")
    
    # 27 Feature Names
    feature_names = []
    feature_names += [f"ADHD_Q{i}" for i in range(1, 8)]
    feature_names += [f"ASD_Q{i}" for i in range(1, 7)]
    feature_names += [f"SPCD_Q{i}" for i in range(1, 5)]
    feature_names += [f"DEP_Q{i}" for i in range(1, 6)]
    feature_names += [f"ANX_Q{i}" for i in range(1, 6)]
    
    X_df = pd.DataFrame([responses], columns=feature_names)
    
    # Prediction
    risk_preds_array = rf_risk.predict(X_df)[0] 
    risk_results_map = {RISK_ORDER[i]: risk_preds_array[i] for i in range(len(RISK_ORDER))}
    
    # Hierarchy
    if risk_results_map["ASD"] == 1 and risk_results_map["SPCD"] == 1:
        risk_results_map["SPCD"] = 0

    # Display
    any_risk_found = False
    final_report_results = {} 
    
    col1, col2 = st.columns(2)
    for disorder, is_risk in risk_results_map.items():
        if is_risk == 1:
            any_risk_found = True
            sev_model = severity_models[disorder]
            sev_pred_idx = sev_model.predict(X_df)[0] 
            sev_label = SEVERITY_DECODER.get(sev_pred_idx, "Unknown")
            final_report_results[disorder] = sev_label
            
            color = "red" if sev_label == "High" else "orange" if sev_label == "Medium" else "blue"
            st.markdown(f"### :warning: **{disorder}** Detected")
            st.markdown(f"Severity Level: <span style='color:{color}; font-weight:bold'>{sev_label}</span>", unsafe_allow_html=True)
            st.write("---")

    if not any_risk_found:
        st.success("✅ No significant cognitive disorder risk detected based on provided responses.")
        st.balloons()

    st.write("### 📄 Download Your Record")
    student_profile = {"grade": grade, "age": age, "gender": gender}
    pdf_bytes = create_pdf(student_profile, final_report_results, questions, user_text_answers)
    st.download_button("Download Report (PDF)", pdf_bytes, "Screening_Report.pdf", "application/pdf")