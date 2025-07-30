import streamlit as st
from streamlit_option_menu import option_menu
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import emoji
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import Counter, defaultdict
import nltk
from PIL import Image
import io
import base64
from datetime import datetime
import plotly.express as px
from wordcloud import WordCloud

# Download NLTK data
nltk.download('vader_lexicon')

# Set up page config
st.set_page_config(
    page_title="WhatsApp Chat Analyzer",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
    }
    .stTextInput>div>div>input {
        border-radius: 5px;
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        color: #2c3e50;
    }
    
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("📱 WhatsApp Chat Analyzer")
st.markdown("""
Analyze your WhatsApp chat exports to gain insights into:
- **Message statistics** 📊
- **Sentiment analysis** 😊😐😠
- **Emoji usage** 😂
- **Activity patterns** 🕒
""")

# Sidebar for file upload
with st.sidebar:
    st.header("Upload Chat")
    uploaded_file = st.file_uploader("Choose a WhatsApp chat export file (.txt)", type="txt")
    
    st.header("Settings")
    show_raw_data = st.checkbox("Show raw data", value=False)
    time_format = st.radio("Time format in your chat", options=["24-hour", "12-hour"])
    
    st.markdown("---")
    st.markdown("""
    **How to export WhatsApp chats:**
    1. Open the WhatsApp chat you want to analyze
    2. Tap ⋮ (Android) or ⓘ (iOS)
    3. Select "More" → "Export chat"
    4. Choose "Without media"
    5. Share/Save the .txt file
    """)

# Emoji extraction function
def extract_emojis(text):
    if not isinstance(text, str):
        return ''
    
    # First try with emoji library
    emoji_list = [c for c in text if c in emoji.EMOJI_DATA]
    if emoji_list:
        return ''.join(emoji_list)
    
    # Fallback pattern for some edge cases
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
        "]+", flags=re.UNICODE)
    
    return ''.join(emoji_pattern.findall(text))

# Date parsing function
def date_time(s):
    patterns = [
        r'^(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{2}(?: ?[APMapm]{2})?) -',  # 12-hour
        r'^(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{2}) -'  # 24-hour
    ]
    return any(re.match(pattern, s) for pattern in patterns)

# Data parsing functions
def find_author(s):
    return bool(re.search(r': ', s))

def getDatapoint(line, time_format):
    splitline = line.split(" - ")
    if len(splitline) < 2:
        return None, None, None, line

    dateTime = splitline[0]
    try:
        date, time = dateTime.split(", ")
    except ValueError:
        return None, None, None, line

    message = " - ".join(splitline[1:])
    author = None
    if find_author(message):
        splitmessage = message.split(": ", 1)
        author = splitmessage[0]
        message = splitmessage[1] if len(splitmessage) > 1 else ""

    return date, time, author, message

# Media classification
media_keywords = {
    'image': ['<media omitted>', 'image omitted', 'photo omitted'],
    'video': ['video omitted', 'video'],
    'audio': ['audio omitted', 'voice message', 'audio'],
    'document': ['document omitted', 'file omitted']
}

def classify_media(message):
    msg = message.lower()
    for mtype, keywords in media_keywords.items():
        if any(k in msg for k in keywords):
            return mtype
    return None

# Main analysis function
def analyze_chat(uploaded_file, time_format):
    if uploaded_file is None:
        return None
    
    # Read file
    content = uploaded_file.getvalue().decode("utf-8", errors='replace')
    lines = content.split('\n')
    
    # Parse chat
    data = []
    messageBuffer = []
    date, time, author = None, None, None
    
    for line in lines:
        line = line.strip()
        if date_time(line):
            if messageBuffer:
                date, time, author, message = getDatapoint(messageBuffer[0], time_format)
                full_message = message + " ".join(messageBuffer[1:]) if message else " ".join(messageBuffer[1:])
                data.append([date, time, author, full_message])
                messageBuffer.clear()
            messageBuffer.append(line)
        else:
            messageBuffer.append(line)
    
    if messageBuffer:
        date, time, author, message = getDatapoint(messageBuffer[0], time_format)
        full_message = message + " ".join(messageBuffer[1:]) if message else " ".join(messageBuffer[1:])
        data.append([date, time, author, full_message])

    # Create DataFrame
    df = pd.DataFrame(data, columns=["Date", "Time", "Author", "Message"])
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    df.dropna(subset=["Message"], inplace=True)
    df.dropna(subset=["Date"], inplace=True)
    
    if df.empty:
        st.error("No valid messages found in the chat file. Please check the file format.")
        return None
    
    # Sentiment analysis
    sia = SentimentIntensityAnalyzer()
    df["Positive"] = df["Message"].apply(lambda x: sia.polarity_scores(str(x))["pos"])
    df["Negative"] = df["Message"].apply(lambda x: sia.polarity_scores(str(x))["neg"])
    df["Neutral"] = df["Message"].apply(lambda x: sia.polarity_scores(str(x))["neu"])
    df["Compound"] = df["Message"].apply(lambda x: sia.polarity_scores(str(x))["compound"])
    df["Sentiment"] = df["Compound"].apply(lambda c: "Positive" if c >= 0.05 else "Negative" if c <= -0.05 else "Neutral")
    
    # Media & Emoji classification
    df["MediaType"] = df["Message"].apply(classify_media)
    df["Emojis"] = df["Message"].apply(extract_emojis)
    
    # Extract hour from time
    try:
        df["Hour"] = pd.to_datetime(df["Time"], format="%H:%M" if time_format == "24-hour" else "%I:%M %p", errors='coerce').dt.hour
    except:
        # Fallback if time parsing fails
        df["Hour"] = pd.to_datetime(df["Time"], errors='coerce').dt.hour
    
    df["Datetime"] = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"], errors="coerce")
    
    return df

# Function For Sentiment Analysis
@st.cache_data
def enhance_sentiment_analysis(df):
            try:
                sia = SentimentIntensityAnalyzer()
                
                # Calculate sentiment scores
                df['sentiment_scores'] = df['Message'].apply(lambda x: sia.polarity_scores(str(x)))
                df['compound_score'] = df['sentiment_scores'].apply(lambda x: x['compound'])
                
                # Create detailed sentiment categories
                bins = [-1, -0.5, -0.1, 0.1, 0.5, 1]
                labels = ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']
                
                df['detailed_sentiment'] = pd.cut(
                    df['compound_score'],
                    bins=bins,
                    labels=labels,
                    include_lowest=True
                ).astype(str)  # Convert to string
                
                # Fill any NA values
                df['detailed_sentiment'] = df['detailed_sentiment'].fillna('Neutral')
                
            except Exception as e:
                st.warning(f"Detailed sentiment analysis failed: {str(e)}")
                # Fallback to basic sentiment
                df['detailed_sentiment'] = df.get('Sentiment', 'Neutral')
            
            return df

# Run analysis when file is uploaded
if uploaded_file is not None:
    with st.spinner("Analyzing chat..."):
        df = analyze_chat(uploaded_file, time_format)
        
    if df is not None:
        # Show raw data if requested
        if show_raw_data:
            st.subheader("Raw Data Preview")
            st.dataframe(df.head())
        
        # Basic stats
        st.subheader("📊 Basic Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Messages", len(df))
        
        with col2:
            st.metric("Unique Participants", df["Author"].nunique())
        
        with col3:
            most_active_day = df["Date"].dt.date.value_counts().idxmax()
            st.metric("Most Active Day", most_active_day.strftime("%Y-%m-%d"))  # Formats as "2023-12-31"
        
        # Message Distribution Analysis Section

        # Run analysis when file is uploaded
        if uploaded_file is not None:
            with st.spinner("Analyzing chat..."):
                df = analyze_chat(uploaded_file, time_format)
                
            if df is not None:
                # Initialize filtered_df with the full dataset first
                filtered_df = df.copy()
                
                # ===== TIME FILTER CONTROLS =====
                st.sidebar.markdown("---")
                st.sidebar.header("⏳ Time Filter")
                
                # Get date range from data
                min_date = df["Date"].min().date()
                max_date = df["Date"].max().date()
                
                time_options = {
                    "All Time": None,
                    "Last 5 Years": pd.DateOffset(years=5),
                    "Last Year": pd.DateOffset(years=1),
                    "Last 6 Months": pd.DateOffset(months=6),
                    "Last 3 Months": pd.DateOffset(months=3),
                    "Last Month": pd.DateOffset(months=1),
                    "Last Week": pd.DateOffset(weeks=1),
                    "Last Day": pd.DateOffset(days=1),
                    "Custom Range": "custom"
                }
                
                selected_range = st.sidebar.selectbox(
                    "Select time range:",
                    list(time_options.keys()),
                    index=0
                )
                
                # Apply time filter
                if selected_range != "All Time":
                    if selected_range == "Custom Range":
                        col1, col2 = st.sidebar.columns(2)
                        with col1:
                            start_date = st.date_input("Start date", min_date, min_value=min_date, max_value=max_date)
                        with col2:
                            end_date = st.date_input("End date", max_date, min_value=min_date, max_value=max_date)
                        
                        filtered_df = df[
                            (df["Date"].dt.date >= start_date) & 
                            (df["Date"].dt.date <= end_date)]
                    else:
                        cutoff_date = max_date - time_options[selected_range]
                        filtered_df = df[df["Date"].dt.date >= cutoff_date]

        # Message Distribution Analysis Section
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📬 Message Distribution Analysis")
            
        with col2:
            about = """
                        🌟 **WhatsApp Chat Analyzer: Your Complete Guide**

                        📌 **What This Tool Does**
                        Transform raw WhatsApp exports into **interactive insights** about:
                        - **Personal Habits**: Your messaging frequency, emoji usage, and active hours  
                        - **Group Dynamics**: Compare participants, detect key contributors, and analyze sentiment trends  
                        - **Hidden Patterns**: Discover yearly activity cycles and conversation tones  

                        ---

                        🛠️ **How to Use – Step by Step**

                        1. **Export Your Chat**  
                        - Open WhatsApp → Select Chat → ⋮ → *Export Chat* → *Without Media*  
                        *(You'll get a `.txt` file)*  

                        2. **Upload & Analyze**  
                        - Use the sidebar to upload your file  
                        - Select your time format (12h/24h)  

                        3. **Explore Features**  

                        🔹 **Personalized Dashboard**  
                        - *Message Stats*: Your total messages, peak activity days  
                        - *Sentiment Timeline*: Track mood over time (Positive/Neutral/Negative)  
                        - *Emoji DNA*: Your most-used emojis vs. group average  

                        🔹 **Group Analysis**  
                        - *Participant Leaderboard*: Who talks most/least  
                        - *Hourly Heatmaps*: See when your group is most active  
                        - *Media Shared*: Breakdown of images/videos/links  

                        🔹 **Pro Tips**  
                        - Use the **time filter** (sidebar) to focus on specific periods  
                        - Click any chart to expand it  
                        - Hover over data points for exact numbers  

                        ---

                        🔒 **Privacy Assurance**  
                        - ❌ No data is stored on servers  
                        - ❌ No third-party tracking  
                        - ✅ 100% processed in your browser  

                        ---

                        💡 **Example Use Cases**  
                        - **Friends Groups**: Who’s the most active? Who starts positive conversations?  
                        - **Family Chats**: Track yearly activity patterns during holidays  
                        - **Work Teams**: Analyze response times and communication efficiency  

                        *Developed by Sayak Mukherjee | Version 1.0*  
                        *[Contribute on GitHub](#) *               
                        © Sayak Mukherjee - The sentiment behind the screen. 
                        """
            
            st.subheader("📖 About The Page")
            with st.expander("Click to know about all featurs"):
                st.markdown(f"{about}")

        # Top Participants Chart
        st.markdown("**👥 Participant Message Distribution**")
        
        msg_count = filtered_df['Author'].value_counts()
        
        # View options for large groups
        if len(msg_count) > 15:
            view_option = st.radio("View:", ["Top 10", "Top 20", "All"], horizontal=True)
        else:
            view_option = "All"
        
        # Determine how many to show
        if view_option == "Top 10":
            top_users = msg_count.head(10)
            others_count = msg_count[10:].sum()
        elif view_option == "Top 20":
            top_users = msg_count.head(20)
            others_count = msg_count[20:].sum()
        else:
            top_users = msg_count
            others_count = 0
        
        if others_count > 0:
            top_users['Others'] = others_count
        
        # Create donut chart
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.tab20c(np.linspace(0, 1, len(top_users)))
        
        wedges, _, _ = ax.pie(
            top_users,
            labels=None,
            autopct=lambda p: f'{p:.1f}%' if p >= 5 else '',
            startangle=90,
            colors=colors,
            pctdistance=0.85,
            wedgeprops={'edgecolor': 'white', 'linewidth': 0.5}
        )
        
        # Add center circle
        ax.add_artist(plt.Circle((0,0), 0.70, fc='white', edgecolor='lightgray'))
        ax.text(0, 0, f"Total\n{len(filtered_df):,}", ha='center', va='center', fontsize=12)
        
        # Add legend
        legend_labels = [f"{label} ({value:,})" for label, value in zip(top_users.index, top_users)]
        ax.legend(wedges, legend_labels, title="Participants", 
                loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
        
        ax.axis('equal')
        st.pyplot(fig)
        plt.close(fig)

        # Main Analysis 

        tab1, tab2= st.tabs([ 
            "Complete Analysis",
            "Participant Analysis",
        ])

        with tab1:
             
            #  Sentiment Analysis Section
            st.subheader("😊 Advanced Sentiment Analysis")

            #  NLP-based sentiment intensity analysis

            if uploaded_file is not None and df is not None:
                df = enhance_sentiment_analysis(df)
                
                # Create two columns for metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    pos_count = len(df[df['Sentiment'] == 'Positive'])
                    st.metric("Positive Messages", 
                            f"{pos_count} ({pos_count/len(df)*100:.1f}%)",
                            help="Messages with clearly positive sentiment")
                
                with col2:
                    neg_count = len(df[df['Sentiment'] == 'Negative'])
                    st.metric("Negative Messages", 
                            f"{neg_count} ({neg_count/len(df)*100:.1f}%)",
                            help="Messages with clearly negative sentiment")
                
                with col3:
                    neutral_count = len(df[df['Sentiment'] == 'Neutral'])
                    st.metric("Neutral Messages", 
                            f"{neutral_count} ({neutral_count/len(df)*100:.1f}%)",
                            help="Messages with neutral or mixed sentiment")

                # Sentiment Distribution Pie Chart
                with st.expander("Detailed Sentiment Breakdown", expanded=True):
                    fig1 = px.pie(df, 
                                names='detailed_sentiment',
                                title='Nuanced Sentiment Distribution',
                                color='detailed_sentiment',
                                color_discrete_map={
                                    'Very Negative': '#d62728',
                                    'Negative': '#ff7f0e',
                                    'Neutral': '#7f7f7f',
                                    'Positive': '#2ca02c',
                                    'Very Positive': '#17becf'
                                },
                                hole=0.4)
                    fig1.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig1, use_container_width=True)

                # Interactive Sentiment Over Time Analysis
                st.subheader("📈 Interactive Sentiment Timeline")
                
                # Date range selector
                col_date1, col_date2 = st.columns(2)
                with col_date1:
                    start_date = st.date_input("Start date", 
                                            df['Date'].min(), 
                                            min_value=df['Date'].min(),
                                            max_value=df['Date'].max())
                with col_date2:
                    end_date = st.date_input("End date", 
                                        df['Date'].max(), 
                                        min_value=df['Date'].min(),
                                        max_value=df['Date'].max())

                # Filter by date range
                date_filtered = df[(df['Date'] >= pd.to_datetime(start_date)) & 
                                (df['Date'] <= pd.to_datetime(end_date))]
                
                # Group by date and sentiment
                daily_sentiment = date_filtered.groupby(["Date", "detailed_sentiment"]).size().unstack().fillna(0)
                
                # Create interactive plot with Plotly
                fig2 = px.line(daily_sentiment, 
                            x=daily_sentiment.index,
                            y=daily_sentiment.columns,
                            title='Daily Sentiment Trend',
                            labels={'value': 'Message Count', 'Date': 'Date'},
                            color_discrete_map={
                                'Very Negative': '#d62728',
                                'Negative': '#ff7f0e',
                                'Neutral': '#7f7f7f',
                                'Positive': '#2ca02c',
                                'Very Positive': '#17becf'
                            })
                
                # Add interactive features
                fig2.update_layout(
                    hovermode='x unified',
                    xaxis=dict(
                        rangeselector=dict(
                            buttons=list([
                                dict(count=7, label="1w", step="day", stepmode="backward"),
                                dict(count=1, label="1m", step="month", stepmode="backward"),
                                dict(count=3, label="3m", step="month", stepmode="backward"),
                                dict(step="all")
                            ])
                        ),
                        rangeslider=dict(visible=True),
                        type="date"
                    )
                )
                
                # Customize hover data
                fig2.update_traces(
                    hovertemplate="<b>Date:</b> %{x}<br><b>Count:</b> %{y}<extra></extra>",
                    line=dict(width=2.5)
                )
                
                st.plotly_chart(fig2, use_container_width=True)
                
                # Sentiment Word Cloud
                st.subheader("💬 Sentiment Word Clouds")
                sentiment_choice = st.selectbox("Select sentiment to visualize:", 
                                            ['Positive', 'Negative', 'Neutral'])
                
                if sentiment_choice:
                    from wordcloud import WordCloud
                    
                    text = ' '.join(df[df['Sentiment'] == sentiment_choice]['Message'].astype(str))
                    if text.strip():
                        wordcloud = WordCloud(width=800, height=400, 
                                            background_color='white',
                                            colormap='viridis' if sentiment_choice == 'Positive' else 'Reds' if sentiment_choice == 'Negative' else 'Greys').generate(text)
                        
                        fig3, ax = plt.subplots(figsize=(10, 5))
                        ax.imshow(wordcloud, interpolation='bilinear')
                        ax.axis('off')
                        ax.set_title(f'Most Common Words in {sentiment_choice} Messages')
                        st.pyplot(fig3)
                        plt.close(fig3)
                    else:
                        st.warning(f"No {sentiment_choice} messages to display")
            # Emoji analysis
            st.subheader("😂 Emoji Analysis")

            # Top emojis visualization
            st.markdown("### Top Used Emojis (Overall)")

            all_emojis = ''.join(df['Emojis'].dropna())
            emoji_freq = Counter(all_emojis)
            top_emojis = dict(emoji_freq.most_common(10))

            if top_emojis:
                # Create DataFrame for plotting
                emoji_df = pd.DataFrame(list(top_emojis.items()), columns=['Emoji', 'Count'])
                emoji_df = emoji_df.sort_values('Count')
                
                # Create figure with original size (8,4)
                fig1, ax1 = plt.subplots(figsize=(8, 4))
                
                # Create horizontal bar plot
                bars = ax1.barh(emoji_df['Emoji'], emoji_df['Count'], color=sns.color_palette("flare"))
                
                # Add emoji labels on the left side
                for i, (emoji, count) in enumerate(zip(emoji_df['Emoji'], emoji_df['Count'])):
                    try:
                        ax1.text(-max(emoji_df['Count'])*0.1, i, emoji, 
                                va='center', ha='center',
                                fontsize=20,
                                fontfamily='Segoe UI Emoji')
                    except:
                        # Fallback if emoji can't be rendered
                        ax1.text(-max(emoji_df['Count'])*0.1, i, "❓", 
                                va='center', ha='center',
                                fontsize=20,
                                fontfamily='Segoe UI Emoji')
                
                ax1.set_xlabel("Frequency")
                ax1.set_title("Most Frequently Used Emojis")
                ax1.grid(axis='x', linestyle='--', alpha=0.7)
                
                # Remove y-axis labels (since we're showing emojis on the left)
                ax1.set_yticks([])
                
                st.pyplot(fig1)
                plt.close(fig1)
            else:
                st.info("No emojis found in messages.")

            # Emoji usage by sender
            st.markdown("### Emoji Usage by Sender")

            emoji_count_by_sender = defaultdict(int)
            for author, emojis in zip(df['Author'], df['Emojis']):
                if pd.notna(author):
                    emoji_count_by_sender[author] += len(emojis)

            if emoji_count_by_sender:
                emoji_sender_df = pd.DataFrame(emoji_count_by_sender.items(), 
                                            columns=['Sender', 'Emoji Count'])
                emoji_sender_df = emoji_sender_df.sort_values('Emoji Count')
                
                # Create figure with original size (10,5)
                fig2, ax2 = plt.subplots(figsize=(10, 5))
                
                # Create horizontal bar plot
                bars = ax2.barh(emoji_sender_df['Sender'], 
                            emoji_sender_df['Emoji Count'], 
                            color=sns.color_palette("rocket"))
                
                ax2.set_xlabel("Total Emoji Count")
                ax2.set_title("Emoji Usage by Participant")
                ax2.grid(axis='x', linestyle='--', alpha=0.7)
                
                # Add value labels
                for bar in bars:
                    width = bar.get_width()
                    ax2.text(width + max(emoji_sender_df['Emoji Count'])*0.01,
                            bar.get_y() + bar.get_height()/2,
                            f'{int(width)}',
                            va='center')
                
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close(fig2)
            else:
                st.info("No emoji usage data by sender available.")

            # Additional Emoji Statistics
            st.markdown("---")
            st.markdown("### 📊 Additional Emoji Statistics")

            # Create three columns for stats
            stat1, stat2, stat3 = st.columns(3)

            with stat1:
                total_emojis = len(all_emojis)
                st.metric("Total Emojis Used", total_emojis)

            with stat2:
                unique_emojis = len(emoji_freq)
                st.metric("Unique Emojis", unique_emojis)

            with stat3:
                avg_per_msg = total_emojis / len(df) if len(df) > 0 else 0
                st.metric("Avg. Emojis per Message", f"{avg_per_msg:.2f}")

            # Emoji Combinations Analysis
            st.markdown("### 🤝 Frequent Emoji Combinations")
            emoji_pairs = Counter()
            for emojis in df['Emojis'].dropna():
                if len(emojis) >= 2:
                    for i in range(len(emojis)-1):
                        pair = emojis[i] + emojis[i+1]
                        emoji_pairs[pair] += 1

            if emoji_pairs:
                top_pairs = emoji_pairs.most_common(5)
                for pair, count in top_pairs:
                    st.markdown(f"- {pair}: {count} times")
            else:
                st.info("No frequent emoji combinations found")
            
            # Activity patterns
            st.subheader("🕒 Activity Patterns")

            # Create tabs for different time views
            subtab1, subtab2, subtab3 = st.tabs(["Monthly", "Weekly", "Hourly"])

            with subtab1:

                # Monthly Activity Dashboard with 12 Pie Charts
                st.subheader("📅 Monthly Activity Dashboard")
                
                # Check if we have datetime data
                if 'Datetime' not in df.columns:
                    st.error("No 'Datetime' column found in the data. Please ensure your data contains datetime information.")
                    st.stop()
                
                # Convert to datetime if not already
                df['Datetime'] = pd.to_datetime(df['Datetime'])
                
                # Year selection
                available_years = sorted(df['Datetime'].dt.year.unique(), reverse=True)
                if not available_years:
                    st.warning("No valid years found in the data.")
                    st.stop()
                
                selected_year = st.selectbox(
                    "Select Year:",
                    options=available_years,
                    index=0
                )

                # Filter data for selected year
                year_data = df[df['Datetime'].dt.year == selected_year]
                if year_data.empty:
                    st.warning(f"No data available for year {selected_year}")
                    st.stop()
                
                # Calculate total messages for the year
                total_messages = len(year_data)
                
                # Get monthly counts
                month_counts = year_data['Datetime'].dt.month_name().value_counts()
                months_order = ['January', 'February', 'March', 'April', 'May', 'June',
                            'July', 'August', 'September', 'October', 'November', 'December']
                month_counts = month_counts.reindex(months_order, fill_value=0)
                month_percent = (month_counts / total_messages * 100).round(1)
                
                # Create a consistent color palette
                colors = plt.cm.viridis(np.linspace(0, 1, 12))
                
                # Create 2 rows of 6 columns for the pie charts
                st.markdown(f"### Monthly Activity Distribution - {selected_year}")
                cols1 = st.columns(6)
                cols2 = st.columns(6)
                
                for i, month in enumerate(months_order):
                    # Data for current month
                    month_pct = month_percent[month]
                    other_pct = 100 - month_pct
                    
                    # Create pie chart
                    fig, ax = plt.subplots(figsize=(3, 3))
                    
                    if month_counts[month] > 0:
                        wedges, texts, autotexts = ax.pie(
                            [month_pct, other_pct],
                            labels=[month[:3], ''],
                            autopct='%1.1f%%',
                            startangle=90,
                            colors=[colors[i], "#E5E5E5"],  # Month color + light gray for rest
                            wedgeprops={'linewidth': 0.5, 'edgecolor': 'white'},
                            textprops={'fontsize': 10,'fontweight':'bold','color':"#004572"}
                        )
                        
                        # Style the percentage text
                        autotexts[0].set_color("#FFBA8DFF")
                        autotexts[0].set_fontweight('bold')
                        autotexts[0].set_fontsize(10)
                        
                        # Style the month label
                        texts[0].set_color("#000000FF")
                        texts[0].set_fontweight('bold')
                        texts[0].set_fontsize(12)
                    else:
                        ax.text(0.5, 0.5, "No Data", ha='center', va='center', fontsize=8)
                    
                    
                    ax.axis('equal')
                    
                    # Place in appropriate column
                    if i < 6:
                        with cols1[i]:
                            st.pyplot(fig)
                            plt.close(fig)
                    else:
                        with cols2[i-6]:
                            st.pyplot(fig)
                            plt.close(fig)
                
                # Statistics section
                st.markdown("---")
                st.markdown(f"### {selected_year} Message Statistics")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Messages", f"{total_messages:,}")
                with col2:
                    peak_month = month_counts.idxmax()
                    peak_value = month_counts.max()
                    st.metric("Most Active Month", f"{peak_month} ({peak_value} messages)")
                with col3:
                    avg_per_month = month_counts.mean()
                    st.metric("Average per Month", f"{avg_per_month:.1f} messages")
                
                # Interpretation guide
                with st.expander("How to read these charts"):
                    st.markdown("""
                    - Each pie chart represents one month's share of the year's total messages
                    - The colored portion shows what percentage of the year's messages occurred in that month
                    - The gray portion represents all other months combined
                    - Larger colored portions indicate more active months
                    - Months with "No Data" had no messages that year
                    """)

            with subtab2:
                # Weekly Activity Analysis
                st.subheader("📅 Weekly Activity Analysis")
                
                # Ensure we're using the main dataframe with datetime column
                if 'Datetime' not in df.columns:
                    st.error("No 'Datetime' column found in the data. Please ensure your data contains datetime information.")
                    st.stop()
                
                # Convert to datetime if not already
                df['Datetime'] = pd.to_datetime(df['Datetime'])
                
                time_period = st.select_slider(
                    "Select time period:",
                    options=["All Time", "Last 12 Months", "Last 6 Months", "Last 3 Months", "Last Month"],
                    value="All Time",
                    key="weekly_slider"
                )
                
                filtered = df.copy()
                if time_period != "All Time":
                    latest_date = filtered["Datetime"].max()
                    if pd.isna(latest_date):
                        st.warning("Could not determine latest date for filtering.")
                    else:
                        months = {
                            "Last Month": 1,
                            "Last 3 Months": 3,
                            "Last 6 Months": 6,
                            "Last 12 Months": 12
                        }[time_period]
                        date_threshold = latest_date - pd.DateOffset(months=months)
                        filtered = filtered[filtered["Datetime"] >= date_threshold]
                
                if not filtered.empty:
                    # Extract day name and count
                    day_freq = filtered["Datetime"].dt.day_name().value_counts()
                    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    day_freq = day_freq.reindex(day_order, fill_value=0)
                    
                    fig, ax = plt.subplots(figsize=(10, 4))
                    sns.barplot(x=day_freq.index, y=day_freq.values, palette="viridis", ax=ax)
                    ax.set_xlabel("Day of Week")
                    ax.set_ylabel("Number of Messages")
                    ax.set_title(f"Weekly Activity ({time_period})")
                    ax.grid(axis='y', alpha=0.3)
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    # Add summary statistics
                    st.markdown("---")
                    st.markdown("### Weekly Statistics")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Messages", f"{day_freq.sum():,}")
                        st.metric("Average per Day", f"{day_freq.mean():.1f}")
                    with col2:
                        peak_day = day_freq.idxmax()
                        peak_value = day_freq.max()
                        st.metric("Busiest Day", f"{peak_day} ({peak_value} messages)")
                        
                else:
                    st.warning(f"No data available for {time_period}")

            with subtab3:
                # Hourly Activity Analysis
                st.subheader("⏰ Hourly Activity Analysis")
                
                # Ensureing we're using the main dataframe with datetime column
                if 'Datetime' not in df.columns:
                    st.error("No 'Datetime' column found in the data. Please ensure your data contains datetime information.")
                    st.stop()
                
                time_period = st.select_slider(
                    "Select time period:", 
                    options=["All Time", "Last 12 Months", "Last 6 Months", "Last 3 Months", "Last Month"],
                    value="All Time",
                    key="hourly_slider"
                )

                filtered = df.copy()
                if time_period != "All Time":
                    latest_date = filtered["Datetime"].max()
                    if pd.isna(latest_date):
                        st.warning("Could not determine latest date for filtering.")
                    else:
                        months = {
                            "Last Month": 1,
                            "Last 3 Months": 3,
                            "Last 6 Months": 6,
                            "Last 12 Months": 12
                        }[time_period]
                        date_threshold = latest_date - pd.DateOffset(months=months)
                        filtered = filtered[filtered["Datetime"] >= date_threshold]

                if not filtered.empty:
                    # Extract hour from Datetime column
                    filtered["Hour"] = filtered["Datetime"].dt.hour
                    
                    hour_freq = filtered["Hour"].value_counts().sort_index()
                    # Ensureing we have all hours (0-23) even if no messages
                    hour_freq = hour_freq.reindex(range(24), fill_value=0)
                    
                    fig, ax = plt.subplots(figsize=(10, 4))
                    sns.barplot(x=hour_freq.index, y=hour_freq.values, palette="crest", ax=ax)
                    ax.set_xticks(range(0, 24))
                    ax.set_xlabel("Hour of Day (24h)")
                    ax.set_ylabel("Number of Messages")
                    ax.set_title(f"Hourly Activity ({time_period})")
                    ax.grid(axis='y', alpha=0.3)
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    # Add summary statistics
                    st.markdown("---")
                    st.markdown("### Hourly Statistics")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Messages", f"{hour_freq.sum():,}")
                        st.metric("Average per Hour", f"{hour_freq.mean():.1f}")
                    with col2:
                        peak_hour = hour_freq.idxmax()
                        peak_value = hour_freq.max()
                        st.metric("Busiest Hour", f"{peak_hour}:00 ({peak_value} messages)")
                        
                else:
                    st.warning(f"No data available for {time_period}")
            
            # Media analysis
            media_df = df[df["MediaType"].notna()]
            if not media_df.empty:
                st.subheader("📦 Media Shared")
                media_counts = media_df["MediaType"].value_counts()
                
                fig, ax = plt.subplots(figsize=(8, 4))
                media_counts.plot(kind='bar', color=sns.color_palette("pastel"), ax=ax)
                ax.set_title("Types of Media Shared")
                ax.set_xlabel("Media Type")
                ax.set_ylabel("Count")
                st.pyplot(fig)
            else:
                st.info("No media messages found in the chat.")
            
            # Final summary
            st.subheader("📝 Final Summary")
            
            x = df["Positive"].sum()
            y = df["Negative"].sum()
            z = df["Neutral"].sum()
            
            if x > y and x > z:
                st.success("Overall Chat Sentiment: Positive 😊")
            elif y > x and y > z:
                st.error("Overall Chat Sentiment: Negative 😠")
            else:
                st.info("Overall Chat Sentiment: Neutral 🙂")
            
            st.markdown(f"""
            - Most active day: **{most_active_day}**
            - Most active year: **{df["Date"].dt.year.value_counts().idxmax()}**
            - Total emojis used: **{len(all_emojis)}**
            - Media shared: **{len(media_df)}** (Images: {media_counts.get('image', 0)}, Videos: {media_counts.get('video', 0)})
            """)
            st.write("© Sayak Mukherjee - The sentiment behind the screen.")


        with tab2:

            # Participant Analysis Section 
            st.header("Participant Analysis")
            # ==============================================
            # Participant-Specific Analysis Section
            # ==============================================
            st.markdown("---")
            st.subheader("👤 Deep Dive: Participant-Specific Analysis")

            # Participant selection dropdown
            participants = df['Author'].dropna().unique()
            selected_participant = st.selectbox(
                "Select participant for detailed analysis:",
                participants,
                index=0,
                key="participant_deepdive"
            )

            # Filter dataframe for selected participant
            participant_df = df[df['Author'] == selected_participant]

            # Create subtabs for different analysis sections
            subtab1, subtab2, subtab3, subtab4, subtab5, subtab6 = st.tabs([
                "😊 Sentiment", 
                "📈 Timeline", 
                "💬 Word Cloud", 
                "😂 Emojis", 
                "🕒 Activity", 
                "📊 Summary"
            ])

            # participant analysis tab1:
            with subtab1:  # Sentiment Analysis
                st.subheader(f"😊 Sentiment Analysis for {selected_participant}")
                
                # Ensure we have a fresh copy of the participant data
                participant_df = df[df['Author'] == selected_participant].copy()
                
                # Apply sentiment analysis if not already done
                if 'detailed_sentiment' not in participant_df.columns:
                    participant_df = enhance_sentiment_analysis(participant_df)
                
                # Create value counts for the pie chart
                sentiment_counts = participant_df['detailed_sentiment'].value_counts().reset_index()
                sentiment_counts.columns = ['sentiment', 'count']
                
                if not sentiment_counts.empty:
                    try:
                        fig = px.pie(
                            sentiment_counts,  # Use the value counts DataFrame
                            names='sentiment',
                            values='count',
                            title='Nuanced Sentiment Distribution',
                            color='sentiment',
                            color_discrete_map={
                                'Very Negative': '#d62728',
                                'Negative': '#ff7f0e',
                                'Neutral': '#7f7f7f',
                                'Positive': '#2ca02c',
                                'Very Positive': '#17becf'
                            },
                            hole=0.4
                        )
                        fig.update_traces(textposition='inside', textinfo='percent+label')
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Could not create pie chart: {str(e)}")
                        # Show simple bar chart as fallback
                        fig, ax = plt.subplots()
                        sentiment_counts.plot(kind='bar', x='sentiment', y='count', ax=ax)
                        st.pyplot(fig)
                else:
                    st.warning("No sentiment data available for this participant")

            with subtab2:  # Interactive Timeline
                st.subheader(f"📈 Sentiment Timeline for {selected_participant}")
                
                # Date range selector
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input("Start date", 
                                            participant_df['Date'].min(), 
                                            min_value=participant_df['Date'].min(),
                                            max_value=participant_df['Date'].max(),
                                            key="part_start_date")
                with col2:
                    end_date = st.date_input("End date", 
                                        participant_df['Date'].max(), 
                                        min_value=participant_df['Date'].min(),
                                        max_value=participant_df['Date'].max(),
                                        key="part_end_date")
                
                # Filter by date range
                date_filtered = participant_df[
                    (participant_df['Date'] >= pd.to_datetime(start_date)) & 
                    (participant_df['Date'] <= pd.to_datetime(end_date))]
                
                # Group by date and sentiment
                daily_sentiment = date_filtered.groupby(["Date", "detailed_sentiment"]).size().unstack().fillna(0)
                
                # Create interactive plot
                fig = px.line(daily_sentiment, 
                            x=daily_sentiment.index,
                            y=daily_sentiment.columns,
                            title='Daily Sentiment Trend',
                            labels={'value': 'Message Count', 'Date': 'Date'},
                            color_discrete_map={
                                'Very Negative': '#d62728',
                                'Negative': '#ff7f0e',
                                'Neutral': '#7f7f7f',
                                'Positive': '#2ca02c',
                                'Very Positive': '#17becf'
                            })
                
                fig.update_layout(
                    hovermode='x unified',
                    xaxis=dict(
                        rangeselector=dict(
                            buttons=list([
                                dict(count=7, label="1w", step="day", stepmode="backward"),
                                dict(count=1, label="1m", step="month", stepmode="backward"),
                                dict(step="all")
                            ])
                        ),
                        rangeslider=dict(visible=True),
                        type="date"
                    )
                )
                st.plotly_chart(fig, use_container_width=True)

            with subtab3:  # Word Cloud
                st.subheader(f"💬 Word Clouds for {selected_participant}")
                
                sentiment_choice = st.selectbox(
                    "Select sentiment to visualize:", 
                    ['Positive', 'Negative', 'Neutral'],
                    key="part_wordcloud"
                )
                
                text = ' '.join(participant_df[participant_df['Sentiment'] == sentiment_choice]['Message'].astype(str))
                
                if text.strip():
                    wordcloud = WordCloud(
                        width=800, height=400, 
                        background_color='white',
                        colormap='viridis' if sentiment_choice == 'Positive' else 'Reds' if sentiment_choice == 'Negative' else 'Greys'
                    ).generate(text)
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    ax.set_title(f'Most Common Words in {sentiment_choice} Messages')
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.warning(f"No {sentiment_choice} messages to display")

            with subtab4:  # Emoji Analysis
                st.subheader(f"😂 Emoji Analysis for {selected_participant}")
                
                # Top emojis
                participant_emojis = ''.join(participant_df['Emojis'].dropna())
                emoji_freq = Counter(participant_emojis)
                top_emojis = dict(emoji_freq.most_common(10))
                
                if top_emojis:
                    emoji_df = pd.DataFrame(list(top_emojis.items()), columns=['Emoji', 'Count'])
                    emoji_df = emoji_df.sort_values('Count')
                    
                    fig, ax = plt.subplots(figsize=(8, 4))
                    bars = ax.barh(emoji_df['Emoji'], emoji_df['Count'], color=sns.color_palette("flare"))
                    
                    # Add emoji labels
                    for i, (emoji, count) in enumerate(zip(emoji_df['Emoji'], emoji_df['Count'])):
                        try:
                            ax.text(-max(emoji_df['Count'])*0.1, i, emoji, 
                                va='center', ha='center',
                                fontsize=20,
                                fontfamily='Segoe UI Emoji')
                        except:
                            ax.text(-max(emoji_df['Count'])*0.1, i, "❓", 
                                va='center', ha='center',
                                fontsize=20,
                                fontfamily='Segoe UI Emoji')
                    
                    ax.set_xlabel("Frequency")
                    ax.set_title(f"Top Emojis Used by {selected_participant}")
                    ax.grid(axis='x', linestyle='--', alpha=0.7)
                    ax.set_yticks([])
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.info("No emojis found in messages.")
                
                # Emoji stats
                st.markdown("#### Emoji Statistics")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Emojis Used", len(participant_emojis))
                with col2:
                    avg_emojis = len(participant_emojis)/len(participant_df) if len(participant_df) > 0 else 0
                    st.metric("Avg. Emojis per Message", f"{avg_emojis:.2f}")

            with subtab5:  # Activity Patterns
                st.subheader(f"🕒 Activity Patterns for {selected_participant}")
                
                # Time period selector
                time_period = st.select_slider(
                    "Select time period:",
                    options=["All Time", "Last 12 Months", "Last 6 Months", "Last 3 Months", "Last Month"],
                    value="All Time",
                    key="part_activity"
                )
                
                # Filter data
                filtered = participant_df.copy()
                latest_date = filtered["Datetime"].max()
                
                if time_period != "All Time":
                    months = {
                        "Last Month": 1,
                        "Last 3 Months": 3,
                        "Last 6 Months": 6,
                        "Last 12 Months": 12
                    }[time_period]
                    
                    date_threshold = latest_date - pd.DateOffset(months=months)
                    filtered = filtered[filtered["Datetime"] >= date_threshold]
                
                if not filtered.empty:
                    # Hourly activity
                    hour_freq = filtered["Hour"].value_counts().sort_index()
                    
                    fig, ax = plt.subplots(figsize=(10, 4))
                    sns.barplot(x=hour_freq.index, y=hour_freq.values, hue=hour_freq.index, 
                                legend=False, palette="crest", ax=ax)
                    ax.set_xticks(range(0, 24))
                    ax.set_xlabel("Hour of Day (24h)")
                    ax.set_ylabel("Number of Messages")
                    ax.set_title(f"Most Active Time of Day ({time_period})")
                    ax.grid(True)
                    st.pyplot(fig)
                    
                    # Daily activity
                    day_freq = filtered["Date"].dt.day_name().value_counts()
                    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    day_freq = day_freq.reindex(day_order)
                    
                    fig, ax = plt.subplots(figsize=(10, 4))
                    sns.barplot(x=day_freq.index, y=day_freq.values, palette="viridis", ax=ax)
                    ax.set_xlabel("Day of Week")
                    ax.set_ylabel("Number of Messages")
                    ax.set_title(f"Activity by Day of Week ({time_period})")
                    st.pyplot(fig)
                else:
                    st.warning(f"No data available for the selected period ({time_period}).")

            with subtab6:  # Summary
                st.subheader(f"📊 Participant Summary: {selected_participant}")
                
                # Calculate stats
                total_messages = len(participant_df)
                percent_total = (total_messages / len(df)) * 100
                active_days = participant_df['Date'].dt.date.nunique()
                avg_per_day = total_messages / active_days if active_days > 0 else 0
                first_msg = participant_df['Date'].min().strftime('%b %d, %Y')
                last_msg = participant_df['Date'].max().strftime('%b %d, %Y')
                
                # Media stats
                media_count = participant_df['MediaType'].notna().sum()
                media_types = participant_df['MediaType'].value_counts().to_dict()
                
                # Sentiment stats
                sentiment_dist = participant_df['Sentiment'].value_counts(normalize=True).mul(100).round(1)
                
                # Display summary cards
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Messages", f"{total_messages:,} ({percent_total:.1f}% of all messages)")
                    st.metric("Active Days", active_days)
                    st.metric("Average Messages/Day", f"{avg_per_day:.1f}")
                
                with col2:
                    st.metric("First Message", first_msg)
                    st.metric("Last Message", last_msg)
                    st.metric("Media Shared", media_count)
                
                # Sentiment summary
                st.markdown("### Sentiment Overview")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Positive", f"{sentiment_dist.get('Positive', 0)}%")
                with col2:
                    st.metric("Neutral", f"{sentiment_dist.get('Neutral', 0)}%")
                with col3:
                    st.metric("Negative", f"{sentiment_dist.get('Negative', 0)}%")
                
                # Media breakdown
                if media_count > 0:
                    st.markdown("### Media Breakdown")
                    for mtype, count in media_types.items():
                        st.markdown(f"- **{mtype.title()}**: {count} messages")
                
                # Most active time
                most_active_hour = participant_df['Hour'].mode()[0]
                st.markdown(f"### ⏰ Most Active Hour: {most_active_hour}:00")

      # Instructions when no file is uploaded
    else:
        st.info("Please upload a WhatsApp chat export file to begin analysis.")
        st.image("https://via.placeholder.com/600x300?text=Upload+a+WhatsApp+chat+export+file", use_column_width=True)