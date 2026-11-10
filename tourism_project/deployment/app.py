"""
Streamlit App - Wellness Tourism Package Purchase Prediction
--------------------------------------------------------------
This app lets a salesperson input a customer's profile and get a prediction
on whether they are likely to purchase the Wellness Tourism Package.
"""

import streamlit as st
import pandas as pd
import joblib
import os

# ----- Page config (must be first Streamlit call) -----
st.set_page_config(
    page_title="Wellness Tourism — Purchase Prediction",
    layout="wide"
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.joblib")


@st.cache_resource
def load_model(path):
    """Load the trained pipeline once and cache it."""
    return joblib.load(path)


def build_input_dataframe(inputs: dict) -> pd.DataFrame:
    """
    Rebuild the engineered features exactly as done in prep.py,
    so the model receives the same features it was trained on.
    """
    df = pd.DataFrame([inputs])

    # Feature engineering (must match prep.py)
    df['TotalVisiting'] = df['NumberOfPersonVisiting'] + df['NumberOfChildrenVisiting']
    df['IncomePerTrip'] = df['MonthlyIncome'] / (df['NumberOfTrips'] + 1)
    df['PitchEfficiency'] = df['PitchSatisfactionScore'] / (df['DurationOfPitch'] + 1)

    # Reorder columns to match the training data
    expected_order = [
        'ProdTaken', 'Age', 'TypeofContact', 'CityTier', 'DurationOfPitch',
        'Occupation', 'Gender', 'NumberOfPersonVisiting', 'NumberOfFollowups',
        'ProductPitched', 'PreferredPropertyStar', 'MaritalStatus', 'NumberOfTrips',
        'Passport', 'PitchSatisfactionScore', 'OwnCar', 'NumberOfChildrenVisiting',
        'Designation', 'MonthlyIncome', 'TotalVisiting', 'IncomePerTrip', 'PitchEfficiency'
    ]

    # Remove the target variable from the feature list
    feature_order = [c for c in expected_order if c != 'ProdTaken']
    return df[feature_order]


def main():
    # ----- Header -----
    st.title("Wellness Tourism Package — Purchase Prediction")
    st.markdown(
        "Use this tool to estimate the likelihood that a customer will "
        "purchase the **Wellness Tourism Package**."
    )
    st.divider()

    # ----- Load model -----
    if not os.path.exists(MODEL_PATH):
        st.error(
            f"Model not found at `{MODEL_PATH}`. "
            "Please run the training pipeline first."
        )
        st.stop()

    model = load_model(MODEL_PATH)

    # ----- Input form -----
    with st.form("customer_form"):
        st.subheader("Customer Profile")

        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            gender = st.selectbox("Gender", ["Male", "Female"])
            marital_status = st.selectbox(
                "Marital Status", ["Single", "Married", "Divorced", "Unmarried"]
            )
            occupation = st.selectbox(
                "Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"]
            )
            designation = st.selectbox(
                "Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
            )
            monthly_income = st.number_input(
                "Monthly Income", min_value=1000, max_value=200000,
                value=20000, step=500
            )

        with col2:
            city_tier = st.selectbox("City Tier", [1, 2, 3], index=0)
            type_of_contact = st.selectbox(
                "Type of Contact", ["Self Enquiry", "Company Invited"]
            )
            product_pitched = st.selectbox(
                "Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"]
            )
            preferred_star = st.selectbox(
                "Preferred Property Star", [3.0, 4.0, 5.0], index=0
            )
            duration_of_pitch = st.number_input(
                "Duration of Pitch (min)", min_value=1, max_value=150, value=15
            )
            pitch_satisfaction = st.slider(
                "Pitch Satisfaction Score", 1, 5, 3
            )

        with col3:
            num_persons = st.number_input(
                "Number of Persons Visiting", min_value=1, max_value=10, value=2
            )
            num_children = st.number_input(
                "Number of Children Visiting", min_value=0, max_value=5, value=0
            )
            num_trips = st.number_input(
                "Number of Trips per Year", min_value=0, max_value=30, value=3
            )
            num_followups = st.number_input(
                "Number of Follow-ups", min_value=0.0, max_value=10.0,
                value=3.0, step=1.0
            )
            passport = st.selectbox(
                "Has Passport?", [0, 1],
                format_func=lambda x: "Yes" if x else "No"
            )
            own_car = st.selectbox(
                "Owns a Car?", [0, 1],
                format_func=lambda x: "Yes" if x else "No"
            )

        st.divider()
        submit = st.form_submit_button(
            "Predict Purchase", use_container_width=True
        )

    # ----- Prediction -----
    if submit:
        inputs = {
            'Age': age,
            'TypeofContact': type_of_contact,
            'CityTier': city_tier,
            'DurationOfPitch': duration_of_pitch,
            'Occupation': occupation,
            'Gender': gender,
            'NumberOfPersonVisiting': num_persons,
            'NumberOfFollowups': num_followups,
            'ProductPitched': product_pitched,
            'PreferredPropertyStar': preferred_star,
            'MaritalStatus': marital_status,
            'NumberOfTrips': num_trips,
            'Passport': passport,
            'PitchSatisfactionScore': pitch_satisfaction,
            'OwnCar': own_car,
            'NumberOfChildrenVisiting': num_children,
            'Designation': designation,
            'MonthlyIncome': monthly_income,
        }

        try:
            df_input = build_input_dataframe(inputs)
            proba = model.predict_proba(df_input)[0]
            p_purchase = proba[1]
            p_no_purchase = proba[0]
            prediction = int(p_purchase >= 0.5)

            st.divider()
            st.subheader("Prediction Result")

            colA, colB = st.columns(2)

            with colA:
                if prediction == 1:
                    st.success(
                        f"### Likely to PURCHASE\n"
                        f"**Confidence: {p_purchase:.1%}**"
                    )
                else:
                    st.warning(
                        f"### Unlikely to PURCHASE\n"
                        f"**Confidence: {p_no_purchase:.1%}**"
                    )

            with colB:
                st.metric("Purchase Probability", f"{p_purchase:.1%}")
                st.progress(float(p_purchase))

            # ----- Business insights -----
            st.subheader("Business Insights")

            if prediction == 1:
                st.info(
                    "**Recommendation:** Prioritize this customer. "
                    "High probability of purchase — schedule the pitch soon."
                )
            else:
                st.info(
                    "**Recommendation:** Low priority. "
                    "Consider re-targeting later or refining the pitch."
                )

                if p_purchase > 0.3:
                    st.caption(
                        f"Borderline case ({p_purchase:.1%}). "
                        "Worth a follow-up call before dropping this lead."
                    )

            # ----- Feature contribution (input summary) -----
            with st.expander("Inputs Summary"):
                st.dataframe(
                    df_input.T.rename(columns={0: "Value"}),
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.exception(e)


if __name__ == "__main__":
    main()
