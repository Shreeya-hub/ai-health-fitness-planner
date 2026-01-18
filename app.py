import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from datetime import datetime
from groq import Groq
import re

st.set_page_config(
    page_title="AI Health & Fitness System",
    page_icon="🏥",
    layout="wide"
)

class WorkoutRecommender:
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)

    def get_recommendations(self, patient):
        prompt = f"""You are a certified fitness trainer.

Patient Details:
Age: {patient['age']}
Gender: {patient['gender']}
BMI: {patient['bmi']}
Medical Condition: {patient['disease']}
Activity Level: {patient['activity']}
Weekly Exercise Hours: {patient['exercise']}

Give 6 workout exercises with short explanations.
Format your response EXACTLY as:
1. Exercise - Description
2. Exercise - Description
3. Exercise - Description
4. Exercise - Description
5. Exercise - Description
6. Exercise - Description"""

        chat_completion = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=500
        )
        
        response_text = chat_completion.choices[0].message.content
        lines = response_text.split("\n")
        workouts = [l.strip() for l in lines if l.strip() and re.match(r'^\d+\.', l.strip())]
        
        return workouts if workouts else ["Unable to generate recommendations"]

DIET_MANAGEMENT = {
    "Low Carb": {
        "Diabetes": "Monitor carbs closely, focus on whole grains, non-starchy vegetables, and low-sugar fruits.",
        "Hypertension": "Reduce sodium, eat potassium-rich foods, limit processed carbs.",
        "Obesity": "Control portions, choose complex carbs, increase protein and fiber.",
        "None": "Balance macros, choose whole grains over refined carbs, stay hydrated."
    },
    "Balanced": {
        "Diabetes": "Eat variety of foods with controlled portions, balance carbs with protein.",
        "Hypertension": "Include fruits, vegetables, whole grains, and lean proteins. Limit salt.",
        "Obesity": "Maintain calorie deficit, eat balanced meals with all food groups.",
        "None": "Eat variety of foods, include whole grains, lean proteins, fruits and vegetables."
    },
    "Low Sodium": {
        "Diabetes": "Use herbs and spices for flavor, limit processed foods, control carbs.",
        "Hypertension": "Essential! Use herbs instead of salt, cook from scratch, read labels.",
        "Obesity": "Reduce bloating, cook fresh meals, avoid packaged foods.",
        "None": "Use herbs and spices, limit processed foods, cook from scratch."
    }
}

RECIPES = {
    "Indian": {
        "Vegan": ["Vegetable stir-fry with minimal oil", "Moong dal soup", "Grilled portobello mushrooms with spices"],
        "Vegetarian": ["Paneer tikka (grilled)", "Vegetable biryani with brown rice", "Roasted vegetables with cottage cheese"],
        "Non-Veg": ["Grilled chicken tandoori", "Fish curry with minimal oil", "Chicken stir-fry with vegetables"]
    },
    "Italian": {
        "Vegan": ["Grilled vegetable skewers", "Lentil minestrone soup", "Zucchini noodles with tomato sauce"],
        "Vegetarian": ["Vegetable lasagna with whole wheat", "Caprese salad", "Grilled vegetables with mozzarella"],
        "Non-Veg": ["Grilled chicken breast with herbs", "Baked salmon with lemon", "Turkey and vegetable wrap"]
    },
    "Chinese": {
        "Vegan": ["Stir-fried mixed vegetables", "Vegetable dumplings (steamed)", "Buddha's delight with tofu"],
        "Vegetarian": ["Mixed vegetable stir-fry with tofu", "Vegetable spring rolls (baked)", "Hot and sour soup"],
        "Non-Veg": ["Grilled chicken with vegetables", "Steamed fish with ginger", "Beef and broccoli (lean cut)"]
    }
}

def get_health_status(value, metric):
    ranges = {
        "Blood Pressure": {"healthy": (90, 120), "borderline": (120, 140)},
        "Cholesterol": {"healthy": (0, 200), "borderline": (200, 239)},
        "Glucose": {"healthy": (70, 100), "borderline": (100, 125)},
        "BMI": {"healthy": (18.5, 24.9), "borderline": (25, 29.9)}
    }
    
    if metric not in ranges:
        return "gray", "Unknown"
    
    r = ranges[metric]
    if r["healthy"][0] <= value <= r["healthy"][1]:
        return "#2ECC71", "🟢 Healthy"
    elif r["borderline"][0] <= value <= r["borderline"][1]:
        return "#F39C12", "🟡 Borderline"
    else:
        return "#E74C3C", "🔴 Needs Attention"

@st.cache_resource
def load_models():
    return (
        joblib.load("diet_model.pkl"),
        joblib.load("scaler.pkl"),
        joblib.load("label_encoders.pkl"),
        joblib.load("diet_encoder.pkl"),
    )

model, scaler, label_encoders, diet_encoder = load_models()

st.image("https://images.unsplash.com/photo-1490645935967-10de6ba17061", use_container_width=True)
st.title("🏥 AI Health & Fitness Recommendation System")
st.caption("ML-based Diet + AI-powered Workout Planner")

patient_id = st.text_input("👤 Enter Patient ID")

tab1, tab2 = st.tabs(["🥗 Diet Recommendation", "💪 Workout Recommendation"])

with tab1:
    st.subheader("🥗 Diet Recommendation & Tracking")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age_d = st.number_input("Age", min_value=0, value=0, key="age_diet")
        weight = st.number_input("Weight (kg)", value=0.0, key="weight_diet")
        bmi_d = st.number_input("BMI", value=0.0, key="bmi_diet")
        disease_d = st.selectbox("Disease Type", ["", "Diabetes", "Hypertension", "Obesity", "None"], key="disease_diet")
        calories = st.number_input("Daily Caloric Intake", value=0, key="calories_diet")
        diet_restriction = st.selectbox("Dietary Restrictions", ["", "None", "Vegan", "Vegetarian"], key="diet_restriction")
    
    with col2:
        gender_d = st.selectbox("Gender", ["", "Male", "Female", "Other"], key="gender_diet")
        height = st.number_input("Height (cm)", value=0.0, key="height_diet")
        severity = st.selectbox("Severity", ["", "Low", "Moderate", "High"], key="severity_diet")
        activity_d = st.selectbox("Physical Activity Level", ["", "Low", "Moderate", "High"], key="activity_diet")
        cholesterol = st.number_input("Cholesterol (mg/dL)", value=0.0, key="cholesterol_diet")
        allergy = st.selectbox("Allergies", ["", "None", "Nuts", "Dairy"], key="allergy_diet")
    
    with col3:
        bp = st.number_input("Blood Pressure (mmHg)", value=0.0, key="bp_diet")
        glucose = st.number_input("Glucose (mg/dL)", value=0.0, key="glucose_diet")
        cuisine = st.selectbox("Preferred Cuisine", ["", "Indian", "Chinese", "Italian"], key="cuisine_diet")
        exercise_d = st.number_input("Weekly Exercise Hours", value=0.0, key="exercise_diet")
        adherence = st.selectbox("Adherence to Diet Plan", ["", "Low", "Medium", "High"], key="adherence_diet")
        imbalance = st.number_input("Nutrient Imbalance Score", value=0.0, key="imbalance_diet")
    
    COLUMN_MAP = {
        "Age": "Age", "Gender": "Gender", "Weight": "Weight_kg", "Height": "Height_cm",
        "BMI": "BMI", "Disease": "Disease_Type", "Severity": "Severity",
        "Activity": "Physical_Activity_Level", "Calories": "Daily_Caloric_Intake",
        "Cholesterol": "Cholesterol_mg/dL", "Blood Pressure": "Blood_Pressure_mmHg",
        "Glucose": "Glucose_mg/dL", "Diet Restriction": "Dietary_Restrictions",
        "Allergy": "Allergies", "Cuisine": "Preferred_Cuisine",
        "Exercise": "Weekly_Exercise_Hours", "Adherence": "Adherence_to_Diet_Plan",
        "Imbalance": "Dietary_Nutrient_Imbalance_Score"
    }
    
    def safe_encode(df, encoders):
        df_encoded = df.copy()
        for col, le in encoders.items():
            if col in df_encoded.columns:
                if df_encoded.loc[0, col] not in le.classes_:
                    df_encoded.loc[0, col] = le.classes_[0]
                df_encoded[col] = le.transform(df_encoded[col])
        return df_encoded
    
    if st.button("🍽️ Get Diet Recommendation", use_container_width=True, key="submit_diet"):
        if not patient_id:
            st.warning("⚠️ Please enter patient ID")
        else:
            raw_data = {
                "Age": age_d, "Gender": gender_d, "Weight": weight, "Height": height,
                "BMI": bmi_d, "Disease": disease_d, "Severity": severity, "Activity": activity_d,
                "Calories": calories, "Cholesterol": cholesterol, "Blood Pressure": bp,
                "Glucose": glucose, "Diet Restriction": diet_restriction, "Allergy": allergy,
                "Cuisine": cuisine, "Exercise": exercise_d, "Adherence": adherence, "Imbalance": imbalance
            }
            
            mapped_data = {COLUMN_MAP[k]: v for k, v in raw_data.items()}
            user_df = pd.DataFrame([mapped_data])
            user_df_encoded = safe_encode(user_df, label_encoders)
            user_scaled = scaler.transform(user_df_encoded)
            
            prediction = model.predict(user_scaled)
            diet_type = diet_encoder.inverse_transform(prediction)[0]
            
            st.success(f"✅ Recommended Diet Plan: **{diet_type}**")
            
            st.subheader("📋 Diet Management Guidelines")
            disease_key = disease_d if disease_d else "None"
            management_tip = DIET_MANAGEMENT.get(diet_type, {}).get(disease_key, "Follow a balanced diet.")
            st.info(f"**For {disease_key}:** {management_tip}")
            
            st.subheader("🍳 Recommended Recipes")
            if cuisine and cuisine in RECIPES:
                diet_pref = "Vegan" if diet_restriction == "Vegan" else ("Vegetarian" if diet_restriction == "Vegetarian" else "Non-Veg")
                recipes = RECIPES[cuisine].get(diet_pref, [])
                
                st.write(f"**{cuisine} - {diet_pref} Options:**")
                for recipe in recipes:
                    st.write(f"• {recipe}")
            
            os.makedirs("patient_data", exist_ok=True)
            file_path = f"patient_data/{patient_id}_diet.csv"
            
            progress_row = {
                "Date": datetime.now().strftime("%Y-%m-%d"),
                "Time": datetime.now().strftime("%H:%M"),
                "Weight": weight,
                "BMI": bmi_d,
                "BP": bp,
                "Cholesterol": cholesterol,
                "Glucose": glucose,
                "Calories": calories
            }
            
            if os.path.exists(file_path):
                history = pd.read_csv(file_path)
                history = pd.concat([history, pd.DataFrame([progress_row])], ignore_index=True)
            else:
                history = pd.DataFrame([progress_row])
            
            history.to_csv(file_path, index=False)
            
            st.subheader("📊 Health Metrics Status")
            
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            bp_color, bp_status = get_health_status(bp, "Blood Pressure")
            chol_color, chol_status = get_health_status(cholesterol, "Cholesterol")
            gluc_color, gluc_status = get_health_status(glucose, "Glucose")
            bmi_color, bmi_status = get_health_status(bmi_d, "BMI")
            
            with metric_col1:
                st.markdown(f"<div style='background-color:{bp_color}20; padding:10px; border-radius:5px; border-left:4px solid {bp_color};'>"
                           f"<b>Blood Pressure</b><br>{bp} mmHg<br>{bp_status}</div>", unsafe_allow_html=True)
            
            with metric_col2:
                st.markdown(f"<div style='background-color:{chol_color}20; padding:10px; border-radius:5px; border-left:4px solid {chol_color};'>"
                           f"<b>Cholesterol</b><br>{cholesterol} mg/dL<br>{chol_status}</div>", unsafe_allow_html=True)
            
            with metric_col3:
                st.markdown(f"<div style='background-color:{gluc_color}20; padding:10px; border-radius:5px; border-left:4px solid {gluc_color};'>"
                           f"<b>Glucose</b><br>{glucose} mg/dL<br>{gluc_status}</div>", unsafe_allow_html=True)
            
            with metric_col4:
                st.markdown(f"<div style='background-color:{bmi_color}20; padding:10px; border-radius:5px; border-left:4px solid {bmi_color};'>"
                           f"<b>BMI</b><br>{bmi_d}<br>{bmi_status}</div>", unsafe_allow_html=True)
            
            st.subheader("📈 Health Progress Tracking")
            
            if len(history) == 1:
                st.info("📝 First record saved! Continue tracking to see progress over time.")
            else:
                st.write(f"**Total Visits:** {len(history)}")
                
                fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                fig.suptitle(f'Patient {patient_id} - Health Metrics Over Time', fontsize=16, fontweight='bold')
                
                visits = range(1, len(history) + 1)
                
                axes[0, 0].plot(visits, history["BP"], marker='o', color='#3498db', linewidth=2)
                axes[0, 0].axhspan(90, 120, alpha=0.2, color='green', label='Healthy')
                axes[0, 0].axhspan(120, 140, alpha=0.2, color='orange', label='Borderline')
                axes[0, 0].set_title('Blood Pressure Trend', fontweight='bold')
                axes[0, 0].set_ylabel('BP (mmHg)')
                axes[0, 0].legend(loc='upper right', fontsize=8)
                axes[0, 0].grid(True, alpha=0.3)
                
                axes[0, 1].plot(visits, history["Cholesterol"], marker='o', color='#e74c3c', linewidth=2)
                axes[0, 1].axhspan(0, 200, alpha=0.2, color='green', label='Healthy')
                axes[0, 1].axhspan(200, 239, alpha=0.2, color='orange', label='Borderline')
                axes[0, 1].set_title('Cholesterol Trend', fontweight='bold')
                axes[0, 1].set_ylabel('Cholesterol (mg/dL)')
                axes[0, 1].legend(loc='upper right', fontsize=8)
                axes[0, 1].grid(True, alpha=0.3)
                
                axes[1, 0].plot(visits, history["Glucose"], marker='o', color='#9b59b6', linewidth=2)
                axes[1, 0].axhspan(70, 100, alpha=0.2, color='green', label='Healthy')
                axes[1, 0].axhspan(100, 125, alpha=0.2, color='orange', label='Borderline')
                axes[1, 0].set_title('Glucose Trend', fontweight='bold')
                axes[1, 0].set_ylabel('Glucose (mg/dL)')
                axes[1, 0].set_xlabel('Visit Number')
                axes[1, 0].legend(loc='upper right', fontsize=8)
                axes[1, 0].grid(True, alpha=0.3)
                
                axes[1, 1].plot(visits, history["BMI"], marker='o', color='#f39c12', linewidth=2)
                axes[1, 1].axhspan(18.5, 24.9, alpha=0.2, color='green', label='Healthy')
                axes[1, 1].axhspan(25, 29.9, alpha=0.2, color='orange', label='Borderline')
                axes[1, 1].set_title('BMI Trend', fontweight='bold')
                axes[1, 1].set_ylabel('BMI')
                axes[1, 1].set_xlabel('Visit Number')
                axes[1, 1].legend(loc='upper right', fontsize=8)
                axes[1, 1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                st.subheader("📋 Visit History")
                st.dataframe(history, use_container_width=True)

with tab2:
    st.subheader("💪 AI Workout Recommendation")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=0, value=0, key="age_workout")
        bmi = st.number_input("BMI", value=0.0, key="bmi_workout")
        disease = st.selectbox("Medical Condition", ["", "None", "Diabetes", "Hypertension", "Obesity", "Heart Disease"], key="disease_workout")

    with col2:
        gender = st.selectbox("Gender", ["", "Male", "Female", "Other"], key="gender_workout")
        activity = st.selectbox("Physical Activity Level", ["", "Low", "Moderate", "High"], key="activity_workout")
        exercise = st.number_input("Weekly Exercise Hours", value=0.0, key="exercise_workout")

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    if st.button("💪 Get Workout Plan", use_container_width=True):
        if not patient_id:
            st.warning("⚠️ Please enter Patient ID")
        else:
            recommender = WorkoutRecommender(GROQ_API_KEY)

            patient_data = {
                "age": age,
                "gender": gender,
                "bmi": bmi,
                "disease": disease,
                "activity": activity,
                "exercise": exercise
            }

            with st.spinner("🤖 Generating AI workout plan..."):
                workouts = recommender.get_recommendations(patient_data)

            st.success("✅ Personalized Workout Plan")
            for w in workouts:
                st.write("•", w)

            os.makedirs("patient_data", exist_ok=True)
            file = f"patient_data/{patient_id}_workout.csv"

            record = {
                "Date": datetime.now().strftime("%Y-%m-%d"),
                "Time": datetime.now().strftime("%H:%M"),
                "BMI": bmi,
                "Weekly Hours": exercise
            }

            if os.path.exists(file):
                df = pd.read_csv(file)
                df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
            else:
                df = pd.DataFrame([record])

            df.to_csv(file, index=False)

            st.subheader("📊 Workout Progress")
            
            if len(df) == 1:
                st.info("📝 First workout record saved! Continue tracking to see progress.")
            else:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(range(1, len(df) + 1), df["Weekly Hours"], marker="o", color="#F18F01", linewidth=2)
                ax.set_ylabel("Exercise Hours", fontsize=12)
                ax.set_xlabel("Visit Number", fontsize=12)
                ax.set_title("Exercise Progress Over Time", fontweight="bold")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                st.subheader("📋 Workout History")
                
                st.info("🟢 **Green BMI** = Healthy (18.5-24.9) | 🔴 **Red BMI** = Needs Improvement | **Weekly Hours** = Track your exercise progress")
                
                def color_bmi(val):
                    if 18.5 <= val <= 24.9:
                        return 'background-color: #2ECC71; color: white'
                    else:
                        return 'background-color: #E74C3C; color: white'
                
                styled_df = df.style.applymap(color_bmi, subset=['BMI'])
                st.dataframe(styled_df, use_container_width=True)
