import streamlit as st
import json
from datetime import datetime
import os
from q_learning_agent import QLearningAgent

# Q-learning agent setup
ACTIONS = ["Rest", "Light Exercise", "Take Medication", "Stretch", "Hydrate"]
agent = QLearningAgent(actions=ACTIONS)

FILENAME = "phoenix_data.json"

user_name = st.text_input("👤 What should I call you? (optional)", key="user_name")

def save_entry(entry):
    with open(FILENAME, "r") as f:
        data = json.load(f)
    data.append(entry)
    with open(FILENAME, "w") as f:
        json.dump(data, f, indent=2)
    st.success("✅ Entry saved!")

timestamp = datetime.now().isoformat()

# ========== SECTION 2 - BODY ==========
with st.expander("🏋️ BODY"):
    pain_level = st.slider("Current pain level (0 = no pain, 10 = worst pain)", 0, 10)
    workout_log = st.text_area("🏃 PT exercises/workouts (sets x reps)", height=100)
    body_journal = st.text_area("📝 How does your body feel today?", height=100)

    # Q-learning suggestion
    suggested_action = agent.choose_action(pain_level)
    st.info(f"🤖 Suggested action for pain level {pain_level}: **{suggested_action}**")

    feedback = st.radio("Did this suggestion help?", ["Yes", "No"], key=f"feedback_{pain_level}")
    if st.button("Save Body Entry"):
        # Reward: +1 for Yes, -1 for No
        reward = 1 if feedback == "Yes" else -1
        # For simplicity, next_state = current pain_level (could be improved)
        agent.update(pain_level, suggested_action, reward, pain_level)
        entry = {
            "user": user_name,
            "timestamp": timestamp,
            "pain_level": pain_level,
            "workout_log": workout_log,
            "body_journal": body_journal,
            "suggested_action": suggested_action,
            "feedback": feedback
        }
        save_entry(entry) 