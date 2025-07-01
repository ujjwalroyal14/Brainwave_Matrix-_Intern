st.markdown("""
    <style>
        html, body, .stApp {
            height: 100%;
            margin: 0;
            background: linear-gradient(135deg, #6a00ff, #1a2aff, #00cfff);
            background-size: 600% 600%;
            animation: gradientShift 18s ease infinite;
            font-family: 'Segoe UI', sans-serif;
            color: white;
        }

        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        .stButton > button {
            background-color: #ff8c00;
            color: white;
            border-radius: 8px;
            padding: 0.6em 1.2em;
            font-weight: 600;
            border: none;
        }

        .stButton > button:hover {
            background-color: #ffa733;
            transition: 0.3s ease-in-out;
        }

        .stTextInput > div > input,
        .stNumberInput > div > input {
            background-color: rgba(255, 255, 255, 0.15);
            color: white;
            border-radius: 5px;
        }

        h1, h2, h3 {
            color: #ffffff;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
    </style>
""", unsafe_allow_html=True)
