# WhatsApp Chat Analyzer

A Streamlit web application to analyze your WhatsApp chat exports and gain insights into your messaging habits, sentiment trends, emoji usage, activity patterns, and participant behavior.

![App Screenshot](./assets/screenshot_1)
![App Screenshot](./assets/screenshot_2)
![App Screenshot](./assets/screenshot_3)
![App Screenshot](./assets/screenshot_4)
![App Screenshot](./assets/screenshot_5)
![App Screenshot](./assets/screenshot_6)
![App Screenshot](./assets/screenshot_7)
![App Screenshot](./assets/screenshot_8)
  
---

## Features

- Upload WhatsApp chat export files (.txt) and analyze message statistics
- Sentiment analysis with detailed sentiment categories
- Emoji extraction and usage analysis
- Activity patterns by hour, day, week, month, and year
- Media shared breakdown (images, videos, audio, documents)
- Participant-specific deep dive analysis with interactive charts
- Time filtering to focus on specific periods
- Interactive visualizations using Matplotlib, Seaborn, and Plotly

---

## Installation

1. Clone the repository or download the source code.

2. (Optional but recommended) Create and activate a Python virtual environment:

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

1. Run the Streamlit app:

```bash
streamlit run app.py
```

2. Open the URL provided by Streamlit in your web browser (usually http://localhost:8501).

3. Use the sidebar to upload your WhatsApp chat export file (.txt).

4. Select the time format used in your chat (12-hour or 24-hour).

5. Explore the various analysis tabs and visualizations.

---

## How to Export WhatsApp Chats 

1. Open the WhatsApp chat you want to analyze.  
2. Tap the menu icon (⋮ on Android or ⓘ on iOS).  
3. Select **More** → **Export chat**.  
4. Choose **Without media**.  
5. Save or share the exported `.txt` file.  

> 📄 **Note:** `synthetic_whatsapp_chat.txt` is an AI-generated demo file created purely for study and training purposes.  
> It does not contain or resemble any real-world data.

---

## Privacy Assurance

- No data is stored on any servers.
- No third-party tracking.
- All processing is done locally in your browser.

---

## Author

Sayak Mukherjee  
The sentiment behind the screen.

---

## License

This project is licensed under the MIT License.
