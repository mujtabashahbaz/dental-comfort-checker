import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# --------------------------
# Step 1: Set Page Configuration
# --------------------------

st.set_page_config(page_title="Ai Dentify's Dental Comfort Checker", layout="centered")

# --------------------------
# Step 2: Data Preparation
# --------------------------

@st.cache_data
def create_dataset():
    np.random.seed(42)
    num_samples = 1000

    # Generate features
    age = np.random.randint(18, 70, num_samples)
    gender = np.random.choice(['Male', 'Female', 'Other'], num_samples)
    previous_experience = np.random.choice(['Positive', 'Negative', 'Neutral'], num_samples)
    frequency_of_visits = np.random.randint(0, 10, num_samples)
    past_behavior = np.random.choice(['Cooperative', 'Anxious', 'Fearful'], num_samples)

    # Generate questionnaire_score based on some logic
    # For simulation, higher questionnaire_score indicates higher anxiety
    questionnaire_score = np.random.randint(0, 41, num_samples)  # 0 to 40

    # Define anxiety_level based on questionnaire_score
    anxiety_level = []
    for score in questionnaire_score:
        if score <= 13:
            anxiety_level.append('Low')
        elif 14 <= score <= 26:
            anxiety_level.append('Medium')
        else:
            anxiety_level.append('High')

    data = {
        'age': age,
        'gender': gender,
        'previous_experience': previous_experience,
        'questionnaire_score': questionnaire_score,
        'frequency_of_visits': frequency_of_visits,
        'past_behavior': past_behavior,
        'anxiety_level': anxiety_level
    }

    df = pd.DataFrame(data)
    return df

# --------------------------
# Step 3: Model Training
# --------------------------

@st.cache_resource
def train_model():
    # Check if model already exists to avoid retraining
    if os.path.exists('dental_anxiety_model.pkl') and \
       os.path.exists('label_encoders.pkl') and \
       os.path.exists('target_encoder.pkl'):
        model = joblib.load('dental_anxiety_model.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        target_encoder = joblib.load('target_encoder.pkl')
    else:
        # Create dataset
        df = create_dataset()

        # Encode categorical variables
        categorical_features = ['gender', 'previous_experience', 'past_behavior']
        label_encoders = {}
        for feature in categorical_features:
            le = LabelEncoder()
            df[feature] = le.fit_transform(df[feature])
            label_encoders[feature] = le

        # Encode target variable
        target_le = LabelEncoder()
        df['anxiety_level'] = target_le.fit_transform(df['anxiety_level'])

        # Features and target
        X = df.drop('anxiety_level', axis=1)
        y = df['anxiety_level']

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize and train the model
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)

        # Evaluate the model (optional)
        accuracy = clf.score(X_test, y_test)
        # Removed the accuracy display to avoid showing it in the app
        # st.write(f"### 🏆 Model Trained with Accuracy: {accuracy * 100:.2f}%")

        # Save the model and encoders
        joblib.dump(clf, 'dental_anxiety_model.pkl')
        joblib.dump(label_encoders, 'label_encoders.pkl')
        joblib.dump(target_le, 'target_encoder.pkl')

        model = clf
        label_encoders = label_encoders
        target_encoder = target_le

    return model, label_encoders, target_encoder

# Load or train the model
model, label_encoders, target_encoder = train_model()

# --------------------------
# Step 4: Helper Functions
# --------------------------

def preprocess_input(age, gender, previous_experience, questionnaire_score, frequency_of_visits, past_behavior):
    input_dict = {
        'age': age,
        'gender': gender,
        'previous_experience': previous_experience,
        'questionnaire_score': questionnaire_score,
        'frequency_of_visits': frequency_of_visits,
        'past_behavior': past_behavior
    }
    input_df = pd.DataFrame([input_dict])

    # Encode categorical features using the loaded label encoders
    for feature in ['gender', 'previous_experience', 'past_behavior']:
        le = label_encoders.get(feature)
        input_df[feature] = le.transform(input_df[feature])

    return input_df

def suggest_techniques(anxiety_level):
    if anxiety_level == 'Low':
        return "• Provide standard dental procedures with optional background music.\n• Encourage open communication."
    elif anxiety_level == 'Medium':
        return ("• Consider mild sedation options.\n"
                "• Play calming music during procedures.\n"
                "• Offer relaxation techniques such as deep breathing.")
    elif anxiety_level == 'High':
        return ("• Strong sedation options (e.g., IV sedation).\n"
                "• Schedule therapy sessions before procedures.\n"
                "• Create a quiet and soothing environment with music.\n"
                "• Use distraction techniques (e.g., virtual reality).")
    else:
        return "No suggestions available."

# --------------------------
# Step 5: Pre-visit Questionnaire
# --------------------------

def pre_visit_questionnaire():
    st.header("📝 Pre-visit Questionnaire")
    st.markdown("Please answer the following questions to help us assess your dental anxiety level.")

    # Define questionnaire questions and options
    questions = [
        {
            "question": "1. How do you feel about visiting the dentist?",
            "options": ["Not anxious at all", "Slightly anxious", "Moderately anxious", "Very anxious", "Extremely anxious"]
        },
        {
            "question": "2. How often do you think about dental procedures?",
            "options": ["Never", "Rarely", "Sometimes", "Often", "Constantly"]
        },
        {
            "question": "3. How do you react to the sound of dental tools?",
            "options": ["Not bothered", "Slightly bothered", "Moderately bothered", "Very bothered", "Extremely bothered"]
        },
        {
            "question": "4. How would you rate your overall dental health?",
            "options": ["Excellent", "Good", "Fair", "Poor", "Very Poor"]
        },
        {
            "question": "5. Have you ever experienced pain during a dental procedure?",
            "options": ["Never", "Rarely", "Sometimes", "Often", "Always"]
        },
        {
            "question": "6. How confident are you in the dentist's ability?",
            "options": ["Not confident at all", "Slightly confident", "Moderately confident", "Very confident", "Extremely confident"]
        },
        {
            "question": "7. How do you feel about the cost of dental procedures?",
            "options": ["Not concerned", "Slightly concerned", "Moderately concerned", "Very concerned", "Extremely concerned"]
        },
        {
            "question": "8. How comfortable are you with dental injections?",
            "options": ["Very comfortable", "Somewhat comfortable", "Neutral", "Somewhat uncomfortable", "Very uncomfortable"]
        },
        {
            "question": "9. How likely are you to delay dental visits due to anxiety?",
            "options": ["Never", "Rarely", "Sometimes", "Often", "Always"]
        },
        {
            "question": "10. How do you prefer to cope with stress during dental procedures?",
            "options": ["No coping mechanism", "Deep breathing", "Listening to music", "Sedation", "Other"]
        }
    ]

    responses = []
    for idx, q in enumerate(questions):
        response = st.radio(q["question"], q["options"], key=f"question_{idx+1}")
        responses.append(response)

    # Map responses to scores (0 to 4)
    total_score = 0
    for i, response in enumerate(responses):
        score = questions[i]["options"].index(response)
        total_score += score

    # Display the total score
    st.markdown(f"**Total Questionnaire Score:** {total_score} / 40")  # Max score: 4*10=40

    return total_score

# --------------------------
# Step 6: Streamlit App Layout
# --------------------------

st.title("🦷 Ai Dentify's Dental Comfort Checker")
st.markdown("""
- A helpful tool that estimates your dental anxiety level using your history and questionnaire responses.
- It also suggests personalized techniques to manage anxiety, making your dental visit more comfortable.
""")
st.header("👤 Patient Information")

# Input Widgets
age = st.number_input("Age", min_value=0, max_value=120, value=30)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
previous_experience = st.selectbox("Previous Dental Experience", ["Positive", "Negative", "Neutral"])

st.markdown("---")  # Separator

# Pre-visit Questionnaire
questionnaire_score = pre_visit_questionnaire()

st.markdown("---")  # Separator

frequency_of_visits = st.number_input("Number of Dental Visits in the Last Year", min_value=0, max_value=20, value=2)
past_behavior = st.selectbox("Past Behavior During Dental Visits", ["Cooperative", "Anxious", "Fearful"])

# Prediction Button
if st.button("🔍 Predict Anxiety Level"):
    # Preprocess input
    try:
        input_data = preprocess_input(age, gender, previous_experience, questionnaire_score, frequency_of_visits, past_behavior)
    except Exception as e:
        st.error(f"Error in preprocessing input: {e}")
        st.stop()

    # Make prediction
    try:
        prediction_numeric = model.predict(input_data)[0]
        prediction_label = target_encoder.inverse_transform([prediction_numeric])[0]
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        st.stop()

    # Display the prediction
    st.subheader(f"### Predicted Anxiety Level: {prediction_label}")

    # Suggest anxiety management techniques
    techniques = suggest_techniques(prediction_label)
    st.markdown("**Suggested Anxiety Management Techniques:**")
    st.markdown(techniques)

# --------------------------
# Optional: Display Model Details
# --------------------------

with st.expander("ℹ️ About the Model"):
    st.write("""
    - **Algorithm:** Random Forest Classifier
    - **Features Used:**
        - Age
        - Gender
        - Previous Dental Experience
        - Pre-visit Questionnaire Score
        - Frequency of Dental Visits
        - Past Behavior During Dental Visits
    - **Target Variable:** Anxiety Level (Low, Medium, High)
    """)

# --------------------------
# Optional: Footer
# --------------------------

st.markdown("---")
st.markdown("© 2024 Ai Dentify's Dental Comfort Checker. All rights reserved.")
