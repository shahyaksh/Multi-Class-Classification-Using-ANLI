import streamlit as st
import requests
import json
import os

API_URL = st.secrets.get("API_URL", "https://your-api-url.run.app")
# Page config
st.set_page_config(
    page_title="ANLI NLI Predictor",
    page_icon="�",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        padding: 0.5rem;
        border-radius: 5px;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .entailment {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    .neutral {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
    }
    .contradiction {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("Natural Language Inference Predictor")
st.markdown("""
Determine the relationship between two statements using a fine-tuned **DeBERTa-v3-base** model.

**Model Performance**: 50.3% accuracy on ANLI R2 test set
""")

st.info("""
**Note**: The first request may take 30-60 seconds as the Cloud Run API wakes up from idle state. 
Subsequent requests will be much faster. If the request times out, please try again.
""")

st.divider()

# Input section
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Premise")
    premise = st.text_area(
        "Enter the premise statement:",
        placeholder="e.g., A person is walking a dog in the park",
        height=100,
        label_visibility="collapsed"
    )

with col2:
    st.subheader("Hypothesis")
    hypothesis = st.text_area(
        "Enter the hypothesis statement:",
        placeholder="e.g., A person is outside",
        height=100,
        label_visibility="collapsed"
    )

# Predict button
predict_button = st.button("Predict Relationship", use_container_width=True)

# Examples section
with st.expander("See Examples"):
    st.markdown("""
    **Example 1: Entailment**
    - Premise: "A person is walking a dog in the park"
    - Hypothesis: "A person is outside"
    
    **Example 2: Neutral**
    - Premise: "A person is reading a book"
    - Hypothesis: "The person enjoys reading"
    
    **Example 3: Contradiction**
    - Premise: "The cat is sleeping on the couch"
    - Hypothesis: "The cat is awake"
    """)

# Prediction logic
if predict_button:
    if not premise or not hypothesis:
        st.error("Please enter both premise and hypothesis.")
    else:
        with st.spinner("Analyzing..."):
            try:
                # Call API
                response = requests.post(
                    f"{API_URL}/predict",
                    json={
                        "premise": premise,
                        "hypothesis": hypothesis
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    prediction = result["prediction"]
                    confidence = result["confidence"]
                    probabilities = result["probabilities"]
                    
                    # Display result
                    st.success("Prediction Complete")
                    
                    # Prediction box with color coding
                    box_class = prediction.lower()
                    st.markdown(f"""
                        <div class="prediction-box {box_class}">
                            <h2 style="margin:0; color: #333;">Prediction: {prediction}</h2>
                            <p style="margin:0.5rem 0 0 0; font-size: 1.1rem; color: #666;">
                                Confidence: {confidence:.1%}
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Probability breakdown
                    st.subheader("Probability Breakdown")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Entailment",
                            f"{probabilities['entailment']:.1%}",
                            delta=None
                        )
                    
                    with col2:
                        st.metric(
                            "Neutral",
                            f"{probabilities['neutral']:.1%}",
                            delta=None
                        )
                    
                    with col3:
                        st.metric(
                            "Contradiction",
                            f"{probabilities['contradiction']:.1%}",
                            delta=None
                        )
                    
                    # Visual bar chart
                    st.bar_chart({
                        "Entailment": probabilities['entailment'],
                        "Neutral": probabilities['neutral'],
                        "Contradiction": probabilities['contradiction']
                    })
                    
                else:
                    st.error(f"API Error: {response.status_code}")
                    st.json(response.json())
                    
            except requests.exceptions.Timeout:
                st.error("Request timed out. Please try again.")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to API. Please check your internet connection.")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p><strong>Model:</strong> DeBERTa-v3-base + LoRA (r=16)</p>
    <p><strong>Dataset:</strong> ANLI R2</p>
    <p><strong>API:</strong> <a href="https://deberta-anli-nli-66725207998.us-east1.run.app/docs" target="_blank">View API Docs</a></p>
</div>
""", unsafe_allow_html=True)
