import streamlit as st
import ollama
from emission_utils import process_emission_query  # Import the backend processing function

# Initialize Streamlit UI
st.set_page_config(page_title="Industry Emissions AI Assistant", layout="wide")

st.title("🌍 Industry Emissions Query Assistant")
st.write("Ask about industry emissions data and get AI-powered insights!")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User query input
user_query = st.chat_input("🔍 Ask about industry emissions data...")

if user_query:
    # Step 1: Display user message in chat
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Step 2: Process the query and get emissions data
    with st.spinner("Fetching data... ⏳"):
        formatted_data = process_emission_query(user_query)

    llm_prompt = f"""
            You are an AI assistant presenting industry emissions data **exactly as retrieved**.

            ---
            ### 🔍 STRICT RULES 🔍
            🚨 **DO NOT modify, fabricate, round, or mislabel any numbers.**  
            🚨 **DO NOT introduce new industries. Use only the ones provided.**  
            🚨 **DO NOT duplicate industries. Each industry appears only once.**  
            🚨 **Ensure numbers match their respective industries exactly.**  

            ---
            ### 📊 FORMAT THE OUTPUT AS FOLLOWS 📊

            #### **📌 Industry Emissions on {formatted_data.get("Emission Date", "Unknown Date")}**
            | **Industry** | **Emission Value (Metric Tons)** |
            |-------------|--------------------------------|
            {''.join([f"| {industry} | {value:,} |\n" for industry, value in formatted_data.items() if industry != "Emission Date" and isinstance(value, (int, float))])}

            ---
            ### **🔍 Insights & Interpretation (AFTER the table)**
            ✅ **Top Emitters:**
               - **{max({k: v for k, v in formatted_data.items() if k != "Emission Date" and isinstance(v, (int, float))}, key=formatted_data.get, default="No Data")}** had the highest emissions.

            ✅ **Notable Observations:**
               - **Identify industries with significantly lower emissions.**
               - **Highlight any abnormally high or low values with a possible explanation.**

            ✅ **Final Conclusion:**
               - Provide a meaningful summary **only based on the retrieved data**.
               - Suggest a next step (e.g., comparing another date or industry).

        """

    # Step 4: Stream AI response in real-time using Streamlit's chat UI
    with st.chat_message("assistant"):
        response_placeholder = st.empty()  # Placeholder for response updates
        response_text = ""

        for chunk in ollama.chat(model="deepseek-r1", messages=[{"role": "system", "content": llm_prompt}], stream=True):
            response_text += chunk["message"]["content"]
            response_placeholder.markdown(response_text)  # Update response dynamically

    # Step 5: Store AI response in chat history
    st.session_state.messages.append({"role": "assistant", "content": response_text})
