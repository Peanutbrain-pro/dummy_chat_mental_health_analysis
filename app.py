# from streamlit_autorefresh import st_autorefresh
# import joblib

from datetime import datetime as dt
from bertclassifier import load_mentalbert, predict_mental_state
import pandas as pd
import streamlit as st
import json
import os
import plotly.express as px

DB_FILE = "chat_db.json"

print("Loading BERT model for classification")
tokenizer, model = load_mentalbert()
print("BERT model loaded")


def load_messages():
    print("Messages loading")
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            return json.load(f)
    return []


def save_message(user, text):
    print("Mesage saved : ", user, text)
    messages = load_messages()

    timestamp = dt.now().strftime("%Y-%m-%d %H:%M:%S")
    messages.append({"user": user, "content": text, "datetime": timestamp})
    with open(DB_FILE, "w") as f:
        json.dump(messages, f)


def clear_history():
    print("history cleared")
    with open(DB_FILE, "w") as f:
        json.dump([], f)
    st.rerun()


st.title("Dummy Chat App")

with st.sidebar:
    st.header("Settings")
    current_user = st.selectbox("User :", ["User A", "User B", "Parent"])

    st.divider()

    if st.button("Delete All History", type="primary"):
        clear_history()
        st.rerun()


messages = load_messages()


if current_user == "Parent":
    st.title("Parental Analytics Dashboard")

    if not messages:
        st.info("No chat data available yet.")
    else:
        # --- 1. Data Preparation ---
        df = pd.DataFrame(messages)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["date"] = df["datetime"].dt.date
        df["hour"] = df["datetime"].dt.hour

        # Generate predictions
        df[["Category", "Confidence"]] = (
            df["content"].apply(predict_mental_state).apply(pd.Series)
        )

        # CRITICAL PRIVACY: Drop content immediately
        df = df.drop(columns=["content"])

        # Universal Color Map
        color_map = {
            "anxiety": "#28a745",  # Green
            "depression": "#ffc107",  # Yellow
            "mental_disorder": "#fd7e14",  # Orange
            "normal": "#6f42c1",  # Purple
            "suicidewatch": "#dc3545",  # Red
        }

        # --- 2. Time Filtering ---
        st.subheader("⏱️ Filter Data")
        time_filter = st.selectbox(
            "Select Time Range", ["Last 7 Days", "Last 30 Days", "All Time"], index=0
        )

        # Apply the filter
        now = pd.Timestamp.now()
        if time_filter == "Last 7 Days":
            df = df[df["datetime"] >= (now - pd.Timedelta(days=7))]
        elif time_filter == "Last 30 Days":
            df = df[df["datetime"] >= (now - pd.Timedelta(days=30))]

        # Stop rendering if the filter leaves us with no data
        if df.empty:
            st.warning(f"No activity detected in the {time_filter}.")
        else:
            # --- 3. GENERAL SYSTEM OVERVIEW ---
            st.header("🌍 General Chat Trends")

            # High-level KPIs
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Messages", len(df))
            col2.metric("Active Users", df["user"].nunique())
            flagged_msgs = len(df[df["Category"] != "normal"])
            col3.metric("Total Flagged Messages", flagged_msgs)

            # Stacked Bar Chart for Time (Will not break if only 1 day of data exists)
            st.subheader(f"General Mood Over Time ({time_filter})")
            daily_general = (
                df.groupby(["date", "Category"]).size().reset_index(name="Count")
            )

            fig_time = px.bar(
                daily_general,
                x="date",
                y="Count",
                color="Category",
                color_discrete_map=color_map,
                barmode="stack",
            )
            fig_time.update_xaxes(tickformat="%b %d", type="category")
            fig_time.update_layout(margin=dict(t=10, b=10, l=0, r=0))
            st.plotly_chart(fig_time, use_container_width=True)

            st.divider()

            # --- 4. INDIVIDUAL USER INSIGHTS ---
            st.header("👤 User-Specific Insights")

            target_user = st.selectbox("Select a User to Analyze:", df["user"].unique())
            user_df = df[df["user"] == target_user]

            # User specific KPIs
            u_col1, u_col2 = st.columns(2)
            u_col1.metric(f"Messages by {target_user}", len(user_df))
            dominant_mood = (
                user_df["Category"].mode()[0].capitalize()
                if not user_df.empty
                else "N/A"
            )
            u_col2.metric(f"{target_user}'s Dominant Mood", dominant_mood)

            # User Charts Layout
            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                st.markdown(f"**{target_user}'s Mood Profile**")
                user_mood_counts = user_df["Category"].value_counts().reset_index()
                user_mood_counts.columns = ["Category", "Count"]

                fig_pie = px.pie(
                    user_mood_counts,
                    values="Count",
                    names="Category",
                    color="Category",
                    color_discrete_map=color_map,
                    hole=0.4,
                )
                fig_pie.update_layout(margin=dict(t=0, b=0, l=0, r=0))
                st.plotly_chart(fig_pie, use_container_width=True)

            with chart_col2:
                st.markdown(f"**{target_user}'s Activity by Hour**")
                hourly_counts = user_df.groupby("hour").size().reset_index(name="Count")

                fig_hour = px.bar(
                    hourly_counts,
                    x="hour",
                    y="Count",
                    labels={"hour": "Hour of Day (24h)"},
                )
                fig_hour.update_layout(
                    xaxis=dict(tickmode="linear", tick0=0, dtick=1, range=[-0.5, 23.5]),
                    margin=dict(t=0, b=0, l=0, r=0),
                )
                fig_hour.update_traces(marker_color="#4C78A8")
                st.plotly_chart(fig_hour, use_container_width=True)


# Normal chat and Personal Insights
else:
    # We use tabs to keep the chat interface clean while offering rich insights
    tab_chat, tab_insights = st.tabs(["💬 Chat Window", "🧠 My Wellbeing Insights"])

    # ==========================================
    # TAB 1: THE STANDARD CHAT
    # ==========================================
    with tab_chat:
        for msg in messages:
            with st.chat_message(msg["user"]):
                st.write(f"**{msg['user']}:** {msg['content']}")

        if prompt := st.chat_input("Type your message here..."):
            save_message(current_user, prompt)
            st.rerun()

    # ==========================================
    # TAB 2: PERSONAL SELF-REFLECTION & SUPPORT
    # ==========================================
    with tab_insights:
        st.header("Your Personal Wellbeing Insights")

        user_msgs = [m for m in messages if m["user"] == current_user]

        if not user_msgs:
            st.info("Send a few messages to generate your personal insights!")
        else:
            df_user = pd.DataFrame(user_msgs)
            df_user["datetime"] = pd.to_datetime(df_user["datetime"])
            df_user["date"] = df_user["datetime"].dt.date
            df_user["hour"] = df_user["datetime"].dt.hour

            df_user[["Category", "Confidence"]] = (
                df_user["content"].apply(predict_mental_state).apply(pd.Series)
            )

            color_map = {
                "normal": "#28a745",  # Green
                "anxiety": "#fd7e14",  # Orange
                "depression": "#6f42c1",  # Purple
                "mental_disorder": "#6c757d",  # Grey
                "suicide_watch": "#dc3545",  # Red
            }

            # --- 1. Time Filtering ---
            st.subheader("⏱️ Time Range")
            time_filter = st.selectbox(
                "Filter your insights:",
                ["Today", "Last 7 Days", "Last 30 Days", "All Time"],
                index=1,
                key="user_time_filter",
            )

            now = pd.Timestamp.now()
            if time_filter == "Today":
                df_user = df_user[df_user["datetime"].dt.date == now.date()]
            elif time_filter == "Last 7 Days":
                df_user = df_user[df_user["datetime"] >= (now - pd.Timedelta(days=7))]
            elif time_filter == "Last 30 Days":
                df_user = df_user[df_user["datetime"] >= (now - pd.Timedelta(days=30))]

            if df_user.empty:
                st.warning(f"No messages found for {time_filter}.")
            else:
                # --- 2. Actionable Support Engine ---
                st.markdown("### 💡 Recommended For You Right Now")

                # Check what moods are present in the filtered timeframe
                recent_moods = df_user["Category"].unique()

                if "suicide_watch" in recent_moods:
                    st.error("""
                    **You are not alone, and help is available right now.**
                    We noticed some messages indicating you might be in a very dark place. Please reach out to someone who can help.
                    * **Call or Text 9152987821** available 24/7.
                    * Step away from the screen, drink a glass of cold water, and focus on your breathing for just one minute.
                    """)
                elif "anxiety" in recent_moods:
                    st.warning("""
                    **It looks like you've been dealing with elevated anxiety.**
                    When your mind is racing, try the **5-4-3-2-1 Grounding Technique** to bring yourself back to the present:
                    1. Acknowledge **5** things you can see around you.
                    2. Acknowledge **4** things you can physically touch.
                    3. Acknowledge **3** things you can hear.
                    4. Acknowledge **2** things you can smell.
                    5. Acknowledge **1** thing you can taste.
                    """)
                elif "depression" in recent_moods:
                    st.info("""
                    **It seems like things have been feeling heavy lately.**
                    When dealing with depressive moods, try the "No Zero Days" approach. 
                    You don't need to fix everything today. Just aim to do *one* tiny positive thing. Drink a glass of water, step outside for 2 minutes, or text a friend. Give yourself permission to rest without guilt.
                    """)
                else:
                    st.success("""
                    **You're maintaining a great baseline!**
                    Your recent conversations look balanced. This is a great time to build healthy habits—like journaling or a quick walk—to act as a buffer for when things do get stressful.
                    """)

                st.divider()

                # --- 3. Personal KPIs & Charts ---
                st.markdown(f"### 📊 Your Data ({time_filter})")
                col1, col2 = st.columns(2)
                col1.metric("Messages Analyzed", len(df_user))
                dominant = df_user["Category"].mode()[0].capitalize()
                col2.metric("Dominant State", dominant)

                chart_col1, chart_col2 = st.columns(2)

                with chart_col1:
                    daily_mood = (
                        df_user.groupby(["date", "Category"])
                        .size()
                        .reset_index(name="Count")
                    )
                    fig_my_trend = px.bar(
                        daily_mood,
                        x="date",
                        y="Count",
                        color="Category",
                        color_discrete_map=color_map,
                        barmode="stack",
                        title="Your Mood Trend",
                    )
                    fig_my_trend.update_xaxes(tickformat="%b %d", type="category")
                    fig_my_trend.update_layout(margin=dict(t=30, b=0, l=0, r=0))
                    st.plotly_chart(fig_my_trend, use_container_width=True)

                with chart_col2:
                    mood_counts = df_user["Category"].value_counts().reset_index()
                    mood_counts.columns = ["Category", "Count"]
                    fig_my_pie = px.pie(
                        mood_counts,
                        values="Count",
                        names="Category",
                        color="Category",
                        color_discrete_map=color_map,
                        hole=0.4,
                        title="Mood Distribution",
                    )
                    fig_my_pie.update_layout(margin=dict(t=30, b=0, l=0, r=0))
                    st.plotly_chart(fig_my_pie, use_container_width=True)

                st.divider()

                # --- 4. Reflective Moments ---
                st.subheader("📝 Moments to Reflect On")

                flagged_df = (
                    df_user[df_user["Category"] != "normal"]
                    .sort_values(by="datetime", ascending=False)
                    .head(5)
                )

                if flagged_df.empty:
                    st.write("No elevated mood markers detected in this timeframe.")
                else:
                    for _, row in flagged_df.iterrows():
                        time_str = row["datetime"].strftime("%A, %b %d at %I:%M %p")
                        cat = row["Category"]

                        if cat == "suicide_watch":
                            st.error(
                                f'**{time_str}** | 🔴 {cat.replace("_", " ").capitalize()}\n> "{row["content"]}"'
                            )
                        elif cat == "anxiety":
                            st.warning(
                                f'**{time_str}** | 🟠 {cat.capitalize()}\n> "{row["content"]}"'
                            )
                        else:  # Depression
                            st.info(
                                f'**{time_str}** | 🟣 {cat.capitalize()}\n> "{row["content"]}"'
                            )
