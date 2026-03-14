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

# ----------------------------
# TRANSLATIONS
# ----------------------------
translations = {
    "English": {
        "title": "Cognitive Disorder Screening Tool",
        "caption": "Developed for MMIT by Dr Monika, Sankalp and Team",
        "warning": "⚠️ **Academic Use Only • Not a Diagnostic Tool**",
        "grade_label": "Grade / Year of Study",
        "age_label": "Age",
        "gender_label": "Gender",
        "start_btn": "Start Screening",
        "next_btn": "Next",
        "prev_btn": "Previous",
        "submit_btn": "Submit Screening",
        "restart_btn": "Start New Screening",
        "options": {"Never": 0, "Rarely": 1, "Sometimes": 2, "Often": 3, "Very Often": 4},
        "questions": [
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
    },
    "Hindi": {
        "title": "संज्ञानात्मक विकार स्क्रीनिंग टूल",
        "caption": "डॉ मोनिका, संकल्प और टीम द्वारा MMIT के लिए विकसित",
        "warning": "⚠️ **केवल शैक्षणिक उपयोग के लिए • यह नैदानिक उपकरण नहीं है**",
        "grade_label": "ग्रेड / अध्ययन का वर्ष",
        "age_label": "आयु",
        "gender_label": "लिंग",
        "start_btn": "स्क्रीनिंग शुरू करें",
        "next_btn": "अगला",
        "prev_btn": "पिछला",
        "submit_btn": "स्क्रीनिंग जमा करें",
        "restart_btn": "नई स्क्रीनिंग शुरू करें",
        "options": {"कभी नहीं": 0, "दुर्लभ": 1, "कभी-कभी": 2, "अक्सर": 3, "हमेशा": 4},
        "questions": [
            "मुझे उन कार्यों को शुरू करने में कठिनाई होती है जिनमें बहुत अधिक सोचने की आवश्यकता होती है।",
            "मैं व्याख्यान, बैठकों या पढ़ने के दौरान ध्यान खो देता हूँ।",
            "मैं समय सीमा या नियुक्तियों को भूल जाता हूँ, भले ही वे महत्वपूर्ण हों।",
            "मुझे अपने काम या अध्ययन सामग्री को व्यवस्थित करने में संघर्ष करना पड़ता है।",
            "मैं महत्वपूर्ण कार्यों के लिए भी अंतिम क्षण तक टालमटोल करता हूँ।",
            "मैं मानसिक रूप से बेचैन महसूस करता हूँ या अपने विचारों को धीमा करने में असमर्थ हूँ।",
            "विषय जानने के बावजूद मैं लापरवाह गलतियाँ करता हूँ।",
            "बातचीत में अपनी बारी का इंतज़ार करना मुझे मुश्किल लगता है।",
            "मुझे चुटकुले, कटाक्ष या अप्रत्यक्ष संकेतों को समझने में संघर्ष करना पड़ता है।",
            "किसी चीज़ को समझाते समय मैं अनिश्चित रहता हूँ कि कितनी जानकारी देनी है।",
            "मुझे समूह चर्चाएँ भ्रमित करने वाली या थका देने वाली लगती हैं।",
            "मैं स्पष्ट नियमों और अनुमानित दिनचर्या को प्राथमिकता देता हूँ।",
            "मैं सामाजिक संकेतों जैसे आवाज के लहजे या चेहरे के भावों को नहीं समझ पाता।",
            "लोग मुझसे कहते हैं कि जब मैं बोलता हूँ तो मैं रूखा, अजीब या अस्पष्ट लगता हूँ।",
            "मैं जिससे बात कर रहा हूँ उसके अनुसार अपनी बात करने के तरीके को बदलने में संघर्ष करता हूँ।",
            "मुझे बातचीत में विषय पर टिके रहना मुश्किल लगता है।",
            "मैं गलत समझता हूँ कि दूसरे मुझसे सामाजिक रूप से क्या उम्मीद करते हैं।",
            "मुझे चीजें करने में बहुत कम रुचि या आनंद महसूस होता है।",
            "मैं उदास, निराश या भावनात्मक रूप से सुन्न महसूस करता हूँ।",
            "मैं अधिकांश दिनों में थका हुआ या कम ऊर्जा महसूस करता हूँ।",
            "मुझे लगता है कि मैं काफी अच्छा नहीं हूँ या असफल रहा हूँ।",
            "खराब मूड के कारण मुझे ध्यान केंद्रित करने में कठिनाई होती है।",
            "मैं घबराहट, चिंतित या बेचैन महसूस करता हूँ।",
            "मैं शैक्षणिक या सामाजिक स्थितियों के बारे में बहुत अधिक चिंता करता हूँ।",
            "समय होने पर भी मुझे आराम करना मुश्किल लगता है।",
            "मेरी चिंता मेरी पढ़ाई या रिश्तों में बाधा डालती है।",
            "मैं उन स्थितियों से बचता हूँ जो मुझे चिंतित करती हैं।"
        ]
    },
    "Marathi": {
        "title": "संज्ञानात्मक विकार स्क्रीनिंग साधन",
        "caption": "डॉ. मोनिका, संकल्प आणि टीम द्वारे MMIT साठी विकसित",
        "warning": "⚠️ **केवळ शैक्षणिक वापरासाठी • हे निदानात्मक साधन नाही**",
        "grade_label": "इयत्ता / अभ्यासाचे वर्ष",
        "age_label": "वय",
        "gender_label": "लिंग",
        "start_btn": "स्क्रीनिंग सुरू करा",
        "next_btn": "पुढे",
        "prev_btn": "मागे",
        "submit_btn": "स्क्रीनिंग सबमिट करा",
        "restart_btn": "नवीन स्क्रीनिंग सुरू करा",
        "options": {"कधीच नाही": 0, "कधीतरी": 1, "काही वेळा": 2, "बऱ्याचदा": 3, "नेहमी": 4},
        "questions": [
            "मला खूप विचार करावा लागणारी कामे सुरू करण्यात अडचण येते.",
            "व्याख्याने, मीटिंग किंवा वाचनादरम्यान माझे लक्ष विचलित होते.",
            "महत्त्वाचे असूनही मी डेडलाईन्स किंवा भेटीच्या वेळा विसरतो.",
            "मला माझे काम किंवा अभ्यासाचे साहित्य व्यवस्थित करण्यात अडचण येते.",
            "महत्त्वाच्या कामांसाठीही मी शेवटच्या क्षणापर्यंत टाळाटाळ करतो.",
            "मला मानसिक अस्वस्थता जाणवते किंवा माझे विचार थांबवणे कठीण जाते.",
            "माहिती असूनही मी निष्काळजीपणाने चुका करतो.",
            "संभाषणामध्ये माझी बोलण्याची वेळ कधी आहे हे ओळखणे मला कठीण जाते.",
            "मला विनोद, उपहास किंवा अप्रत्यक्ष संकेत समजून घेताना अडचण येते.",
            "एखादी गोष्ट स्पष्ट करताना किती माहिती द्यायची याबद्दल मला खात्री नसते.",
            "मला गटचर्चा गोंधळात टाकणाऱ्या किंवा थकवणाऱ्या वाटतात.",
            "मला स्पष्ट नियम आणि ठराविक दिनचर्या आवडते.",
            "मी आवाजातील चढ-उतार किंवा चेहऱ्यावरील हावभाव यांसारखे सामाजिक संकेत ओळखू शकत नाही.",
            "लोक मला सांगतात की मी बोलताना स्पष्टवक्ता, अवघडलेला किंवा अस्पष्ट वाटतो.",
            "मी कोणाशी बोलत आहे त्यानुसार बोलण्याची पद्धत बदलण्यात मला अडचण येते.",
            "मला संभाषणात विषयावर टिकून राहणे कठीण जाते.",
            "दुसऱ्यांच्या माझ्याकडून असलेल्या सामाजिक अपेक्षा समजण्यात माझी चूक होते.",
            "मला गोष्टी करण्यात फारसा रस किंवा आनंद वाटत नाही.",
            "मला उदास, निराश किंवा भावशून्य वाटते.",
            "मला बहुतेक दिवस थकवा किंवा कमी ऊर्जा जाणवते.",
            "मला वाटते की मी पुरेसा चांगला नाही किंवा मी अपयशी ठरलो आहे.",
            "खराब मूडमुळे मला लक्ष केंद्रित करण्यात अडचण येते.",
            "मला भीती, चिंता किंवा अस्वस्थता वाटते.",
            "मी शैक्षणिक किंवा सामाजिक परिस्थितीबद्दल खूप चिंता करतो.",
            "वेळ असूनही मला आराम करणे कठीण जाते.",
            "माझ्या चिंतेचा परिणाम माझ्या अभ्यासावर किंवा नातेसंबंधांवर होतो.",
            "ज्या परिस्थितीमुळे मला चिंता वाटते अशा गोष्टी मी टाळतो."
        ]
    }
}

# ----------------------------
# SESSION STATE INITIALIZATION
# ----------------------------
if 'step' not in st.session_state:
    st.session_state.step = 0 # 0 = Profile, 1-27 = Questions, 28 = Results
if 'answers' not in st.session_state:
    st.session_state.answers = [None] * 27
if 'profile' not in st.session_state:
    st.session_state.profile = {"grade": "", "age": 18, "gender": ""}

# Helper functions for navigation
def next_step():
    st.session_state.step += 1

def prev_step():
    st.session_state.step -= 1

def restart_screening():
    st.session_state.step = 0
    st.session_state.answers = [None] * 27
    st.session_state.profile = {"grade": "", "age": 18, "gender": ""}

# ----------------------------
# DATABASE SETUP & MODELS
# ----------------------------
DB_PATH = "responses.db"
MODEL_DIR = "models"

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

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

result_columns = ["res_ADHD", "res_ASD", "res_SPCD", "res_DEP", "res_ANX"]
for col in result_columns:
    try:
        cursor.execute(f"ALTER TABLE responses ADD COLUMN {col} TEXT")
    except sqlite3.OperationalError:
        pass

conn.commit()

def load_model(name):
    path = os.path.join(MODEL_DIR, name)
    if not os.path.exists(path):
        st.error(f"Model file not found: {path}. Please run train_final.py first.")
        st.stop()
    return pickle.load(open(path, "rb"))

# Model Loading (Commented out exception handling if models are missing for testing purposes)
try:
    rf_risk = load_model("RF_risk.pkl") 
    severity_models = {
        "ADHD": load_model("rf_ADHD_sev.pkl"),
        "ASD": load_model("rf_ASD_sev.pkl"),
        "SPCD": load_model("rf_SPCD_sev.pkl"),
        "DEP": load_model("rf_DEP_sev.pkl"),
        "ANX": load_model("rf_ANX_sev.pkl")
    }
except:
    pass # Let it fail gracefully inside load_model if not present

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

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Student Profile", 0, 1)
    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 6, f"Grade: {student_data['grade']}", 0, 1)
    pdf.cell(0, 6, f"Age: {student_data['age']}", 0, 1)
    pdf.cell(0, 6, f"Gender: {student_data['gender']}", 0, 1)
    pdf.cell(0, 6, f"Date: {datetime.now().strftime('%Y-%m-%d')}", 0, 1)
    pdf.ln(5)

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

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Detailed Responses", 0, 1)
    pdf.set_font("Arial", 'B', 10)
    pdf.set_fill_color(230, 230, 230)
    
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
    if 'admin_logged_in' not in st.session_state:
        st.session_state['admin_logged_in'] = False

    if not st.session_state['admin_logged_in']:
        if st.checkbox("Access Admin Panel"):
            with st.form("login_form"):
                user = st.text_input("Username")
                pwd = st.text_input("Password", type="password")
                submit_login = st.form_submit_button("Login")
            
            if submit_login:
                if user == "admin" and pwd == "IAmTheAdmin123":
                    st.session_state['admin_logged_in'] = True
                    st.rerun() 
                else:
                    st.error("❌ Invalid Username or Password")
    else:
        st.success("✅ Admin Access Granted")
        if st.button("Logout"):
            st.session_state['admin_logged_in'] = False
            st.rerun()
            
        st.divider()
        st.subheader("Stored Responses")
        try:
            df_hist = pd.read_sql_query("SELECT * FROM responses", conn)
            st.dataframe(df_hist)
            csv = df_hist.to_csv(index=False).encode('utf-8')
            st.download_button("Download Database as CSV", csv, "responses_backup.csv", "text/csv")
        except Exception as e:
            st.error(f"Error loading database: {e}")

# ----------------------------
# MAIN UI
# ----------------------------

# 1. Header & Language Selection 
lang_choice = st.selectbox("🌐 Select Language / भाषा निवडा / भाषा चुनें", ["English", "Hindi", "Marathi"])
t = translations[lang_choice]
questions = t["questions"]
options_dict = t["options"]
option_labels = list(options_dict.keys())

st.title(t["title"])
st.caption(t["caption"])
st.warning(t["warning"])
st.divider()

# STEP 0: Demographic Data Collection
if st.session_state.step == 0:
    st.subheader("Student Profile")
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.session_state.profile["grade"] = st.selectbox(t["grade_label"], ["", "UG", "PG", "Higher"], 
                                                           index=["", "UG", "PG", "Higher"].index(st.session_state.profile["grade"]) if st.session_state.profile["grade"] else 0)
        with col2:
            st.session_state.profile["age"] = st.number_input(t["age_label"], min_value=18, max_value=60, step=1, 
                                                              value=st.session_state.profile["age"])
        with col3:
            st.session_state.profile["gender"] = st.selectbox(t["gender_label"], ["", "Male", "Female", "Other"],
                                                            index=["", "Male", "Female", "Other"].index(st.session_state.profile["gender"]) if st.session_state.profile["gender"] else 0)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Validation for start button
        is_profile_complete = st.session_state.profile["grade"] != "" and st.session_state.profile["gender"] != ""
        if st.button(t["start_btn"], type="primary", disabled=not is_profile_complete):
            next_step()
            st.rerun()

# STEPS 1 to 27: Question Cards
elif 1 <= st.session_state.step <= 27:
    q_index = st.session_state.step - 1
    total_q = len(questions)
    
    # Render the card
    with st.container(border=True):
        st.markdown(f"**Question {st.session_state.step} of {total_q}**")
        st.progress(st.session_state.step / total_q)
        
        st.markdown(f"### {questions[q_index]}")
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Determine the currently selected index based on saved score
        saved_score = st.session_state.answers[q_index]
        current_index = saved_score if saved_score is not None else None
        
        choice = st.radio(
            label="Options",
            options=option_labels,
            index=current_index,
            key=f"q_{q_index}_{lang_choice}", # Unique key per question/language
            label_visibility="collapsed"
        )
        
        # Save score to session state instantly
        if choice is not None:
            st.session_state.answers[q_index] = options_dict[choice]
            
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Navigation Buttons
        col1, col2, col3 = st.columns([1, 4, 1])
        with col1:
            st.button(t["prev_btn"], on_click=prev_step)
        with col3:
            if st.session_state.step < 27:
                st.button(t["next_btn"], on_click=next_step, disabled=(choice is None), type="primary")
            else:
                st.button(t["submit_btn"], on_click=next_step, disabled=(choice is None), type="primary")

# STEP 28: Processing and Results
elif st.session_state.step == 28:
    st.subheader("Results Analysis")
    with st.spinner('Analyzing responses...'):
        # 1. Prepare Features for Prediction
        feature_names = []
        feature_names += [f"ADHD_Q{i}" for i in range(1, 8)]
        feature_names += [f"ASD_Q{i}" for i in range(1, 7)]
        feature_names += [f"SPCD_Q{i}" for i in range(1, 5)]
        feature_names += [f"DEP_Q{i}" for i in range(1, 6)]
        feature_names += [f"ANX_Q{i}" for i in range(1, 6)]
        
        X_df = pd.DataFrame([st.session_state.answers], columns=feature_names)
        
        # 2. Run Risk Model
        risk_preds_array = rf_risk.predict(X_df)[0] 
        risk_results_map = {RISK_ORDER[i]: risk_preds_array[i] for i in range(len(RISK_ORDER))}
        
        # 3. Apply Hierarchy Rule
        if risk_results_map["ASD"] == 1 and risk_results_map["SPCD"] == 1:
            risk_results_map["SPCD"] = 0

        # 4. Determine Severity & Display Results
        any_risk_found = False
        final_report_results = {} 
        db_result_values = {d: "Negative" for d in RISK_ORDER}

        for disorder, is_risk in risk_results_map.items():
            if is_risk == 1:
                any_risk_found = True
                sev_model = severity_models[disorder]
                sev_pred_idx = sev_model.predict(X_df)[0] 
                sev_label = SEVERITY_DECODER.get(sev_pred_idx, "Unknown")
                
                final_report_results[disorder] = sev_label
                db_result_values[disorder] = sev_label 
                
                color = "red" if sev_label == "High" else "orange" if sev_label == "Medium" else "blue"
                with st.container(border=True):
                    st.markdown(f"### :warning: **{disorder}** Detected")
                    st.markdown(f"Severity Level: <span style='color:{color}; font-weight:bold'>{sev_label}</span>", unsafe_allow_html=True)

        if not any_risk_found:
            st.success("✅ No significant cognitive disorder risk detected.")
            st.balloons()

        # 5. Save to Database
        # Ensuring we don't save duplicate records if user refreshes the result page
        if 'saved_to_db' not in st.session_state:
            submitted_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            db_responses = st.session_state.answers + [0, 0, 0] # Pad for DB schema
            results_to_save = [db_result_values["ADHD"], db_result_values["ASD"], 
                               db_result_values["SPCD"], db_result_values["DEP"], 
                               db_result_values["ANX"]]

            values = [st.session_state.profile["grade"], st.session_state.profile["age"], st.session_state.profile["gender"]] + db_responses + results_to_save + [submitted_at]
            
            cursor.execute("""
            INSERT INTO responses (
                grade, age, gender,
                q1, q2, q3, q4, q5, q6, q7, q8, q9, q10,
                q11, q12, q13, q14, q15, q16, q17, q18, q19, q20,
                q21, q22, q23, q24, q25, q26, q27, q28, q29, q30,
                res_ADHD, res_ASD, res_SPCD, res_DEP, res_ANX,
                submitted_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, values)
            conn.commit()
            st.session_state.saved_to_db = True
            st.toast("Data saved successfully.")

        # 6. PDF Generation
        st.write("### 📄 Download Your Record")
        
        # Convert numeric scores back to text answers for the PDF based on English layout to avoid FPDF Unicode errors
        eng_options = list(translations["English"]["options"].keys())
        user_text_answers_eng = [eng_options[score] for score in st.session_state.answers]
        eng_questions = translations["English"]["questions"]

        pdf_bytes = create_pdf(st.session_state.profile, final_report_results, eng_questions, user_text_answers_eng)
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("Download Report (PDF)", pdf_bytes, "Screening_Report.pdf", "application/pdf", type="primary")
        with col2:
            st.button(t["restart_btn"], on_click=restart_screening)
