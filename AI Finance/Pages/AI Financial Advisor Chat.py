# import streamlit as st
# import pandas as pd
# import numpy as np
# import os
# import json
# from datetime import datetime, date
# import base64
# from google import genai
# from google.genai import types
# import speech_recognition as sr
# import pyttsx3
# from googletrans import Translator
# import plotly.express as px
# import plotly.graph_objects as go

# # Initialize session states
# def initialize_chat_session():
#     """Initialize chat session state"""
#     if 'chat_history' not in st.session_state:
#         st.session_state.chat_history = []
#     if 'selected_language' not in st.session_state:
#         st.session_state.selected_language = 'English'
#     if 'voice_enabled' not in st.session_state:
#         st.session_state.voice_enabled = False

# def get_combined_financial_data():
#     """Get combined financial data for context"""
#     combined_df = pd.DataFrame()
    
#     # Get manual entries
#     if 'manual_entries' in st.session_state and st.session_state.manual_entries:
#         manual_df = pd.DataFrame(st.session_state.manual_entries)
#         combined_df = pd.concat([combined_df, manual_df], ignore_index=True)
    
#     # Get uploaded data
#     if 'uploaded_data' in st.session_state:
#         uploaded_df = st.session_state.uploaded_data.copy()
#         uploaded_df.columns = uploaded_df.columns.str.lower()
#         combined_df = pd.concat([combined_df, uploaded_df], ignore_index=True)
    
#     if not combined_df.empty:
#         combined_df['date'] = pd.to_datetime(combined_df['date'])
#         combined_df['amount'] = pd.to_numeric(combined_df['amount'])
#         combined_df['type'] = combined_df['type'].str.lower()
#         combined_df = combined_df.sort_values('date')
    
#     return combined_df

# def get_goals_data():
#     """Get goals data for context"""
#     goals = []
#     if 'financial_goals' in st.session_state:
#         goals = st.session_state.financial_goals
#     return goals

# def create_financial_context():
#     """Create financial context for AI"""
#     df = get_combined_financial_data()
#     goals = get_goals_data()
    
#     context = {
#         'has_data': not df.empty,
#         'total_transactions': len(df) if not df.empty else 0,
#         'total_income': 0,
#         'total_expenses': 0,
#         'net_balance': 0,
#         'top_expense_categories': [],
#         'goals': goals,
#         'recent_transactions': []
#     }
    
#     if not df.empty:
#         income_df = df[df['type'] == 'income']
#         expense_df = df[df['type'] == 'expense']
        
#         context['total_income'] = float(income_df['amount'].sum()) if not income_df.empty else 0
#         context['total_expenses'] = float(expense_df['amount'].sum()) if not expense_df.empty else 0
#         context['net_balance'] = context['total_income'] - context['total_expenses']
        
#         if not expense_df.empty:
#             top_categories = expense_df.groupby('category')['amount'].sum().nlargest(3)
#             context['top_expense_categories'] = [
#                 {'category': cat, 'amount': float(amount)} 
#                 for cat, amount in top_categories.items()
#             ]
        
#         # Recent transactions (last 5)
#         recent = df.tail(5)[['date', 'description', 'amount', 'type', 'category']].to_dict('records')
#         context['recent_transactions'] = [
#             {
#                 'date': r['date'].strftime('%Y-%m-%d') if pd.notna(r['date']) else '',
#                 'description': r['description'],
#                 'amount': float(r['amount']),
#                 'type': r['type'],
#                 'category': r['category']
#             }
#             for r in recent
#         ]
    
#     return context

# def get_gemini_response(user_message, financial_context):
#     """Get response from Gemini AI with financial context"""
#     try:
#         # Check for API key
#         api_key = "AIzaSyDcrJg4kH1TGtQutFmu0gjL65BAhdR3riQ"
#         if not api_key:
#             return "❌ Gemini API key not found. Please set the GEMINI_API_KEY environment variable."
        
#         client = genai.Client(api_key=api_key)
#         model = "gemini-2.0-flash"
        
#         # Create comprehensive prompt with financial context
#         system_prompt = f"""
# You are an expert AI Financial Advisor. You provide personalized financial advice based on the user's actual financial data.

# USER'S FINANCIAL CONTEXT:
# - Has financial data: {financial_context['has_data']}
# - Total transactions: {financial_context['total_transactions']}
# - Total income: ₹{financial_context['total_income']:,.2f}
# - Total expenses: ₹{financial_context['total_expenses']:,.2f}
# - Net balance: ₹{financial_context['net_balance']:,.2f}
# - Top expense categories: {financial_context['top_expense_categories']}
# - Active goals: {len(financial_context['goals'])}
# - Goals details: {financial_context['goals']}
# - Recent transactions: {financial_context['recent_transactions']}

# INSTRUCTIONS:
# 1. Provide personalized advice based on the actual financial data above
# 2. Be specific with numbers from their data
# 3. Give actionable recommendations
# 4. Be encouraging and supportive
# 5. Use Indian currency format (₹)
# 6. Keep responses concise but informative
# 7. If user asks about specific data, reference their actual transactions
# 8. Suggest specific budget allocations based on their spending patterns
# 9. Help with goal planning using their financial capacity

# User's question: {user_message}
# """
        
#         contents = [
#             types.Content(
#                 role="user",
#                 parts=[types.Part.from_text(text=system_prompt)],
#             ),
#         ]
        
#         # Add Google Search tool for current financial information
#         tools = [
#             types.Tool(googleSearch=types.GoogleSearch()),
#         ]
        
#         generate_content_config = types.GenerateContentConfig(
#             tools=tools,
#             temperature=0.7,
#         )
        
#         # Generate response
#         response_text = ""
#         for chunk in client.models.generate_content_stream(
#             model=model,
#             contents=contents,
#             config=generate_content_config,
#         ):
#             if hasattr(chunk, 'text') and chunk.text:
#                 response_text += chunk.text
        
#         return response_text if response_text else "I'm having trouble processing your request. Please try again."
        
#     except Exception as e:
#         return f"❌ Error connecting to AI: {str(e)}"

# def translate_text(text, target_language):
#     """Translate text to target language"""
#     if target_language == 'English':
#         return text
    
#     try:
#         translator = Translator()
        
#         # Language mapping
#         lang_codes = {
#             'Hindi': 'hi',
#             'Tamil': 'ta',
#             'Telugu': 'te',
#             'Bengali': 'bn',
#             'Marathi': 'mr',
#             'Gujarati': 'gu',
#             'Kannada': 'kn',
#             'Malayalam': 'ml',
#             'Punjabi': 'pa'
#         }
        
#         if target_language in lang_codes:
#             result = translator.translate(text, dest=lang_codes[target_language])
#             return result.text
#         else:
#             return text
            
#     except Exception as e:
#         st.error(f"Translation error: {e}")
#         return text

# def speech_to_text():
#     """Convert speech to text"""
#     try:
#         r = sr.Recognizer()
#         with sr.Microphone() as source:
#             st.info("🎤 Listening... Speak now!")
#             r.adjust_for_ambient_noise(source, duration=1)
#             audio = r.listen(source, timeout=10, phrase_time_limit=15)
        
#         st.info("🔄 Processing speech...")
#         text = r.recognize_google(audio)
#         return text
        
#     except sr.RequestError as e:
#         st.error(f"Could not request results; {e}")
#         return None
#     except sr.UnknownValueError:
#         st.error("Could not understand audio")
#         return None
#     except Exception as e:
#         st.error(f"Error: {e}")
#         return None

# def text_to_speech(text):
#     """Convert text to speech"""
#     try:
#         # This is a placeholder for TTS - in a real app you'd implement actual TTS
#         st.success(f"🔊 Would play: {text[:100]}...")
#         return True
#     except Exception as e:
#         st.error(f"TTS Error: {e}")
#         return False

# def display_chat_message(role, message, timestamp):
#     """Display chat message with styling"""
#     if role == "user":
#         st.markdown(f"""
#         <div style='display: flex; justify-content: flex-end; margin-bottom: 10px;'>
#             <div style='background-color: #0084ff; color: white; padding: 10px; border-radius: 15px; max-width: 70%; margin-left: 30%;'>
#                 <strong>You:</strong><br>{message}
#                 <div style='font-size: 0.8em; opacity: 0.7; margin-top: 5px;'>{timestamp}</div>
#             </div>
#         </div>
#         """, unsafe_allow_html=True)
#     else:
#         st.markdown(f"""
#         <div style='display: flex; justify-content: flex-start; margin-bottom: 10px;'>
#             <div style='background-color: #f0f0f0; color: black; padding: 10px; border-radius: 15px; max-width: 70%; margin-right: 30%;'>
#                 <strong>🤖 AI Advisor:</strong><br>{message}
#                 <div style='font-size: 0.8em; opacity: 0.7; margin-top: 5px;'>{timestamp}</div>
#             </div>
#         </div>
#         """, unsafe_allow_html=True)

# def quick_questions():
#     """Display quick question buttons"""
#     st.subheader("💬 Quick Questions")
    
#     questions = [
#         "How is my spending this month?",
#         "Am I on track with my goals?",
#         "What should I invest in?",
#         "How can I save more money?",
#         "Analyze my top expenses",
#         "Create a budget plan",
#         "When will I reach my savings goal?",
#         "Should I take a loan?"
#     ]
    
#     cols = st.columns(4)
#     for i, question in enumerate(questions):
#         with cols[i % 4]:
#             if st.button(question, key=f"quick_{i}"):
#                 return question
    
#     return None

# def financial_calculator_chat():
#     """Embedded financial calculators in chat"""
#     st.subheader("🧮 Financial Calculators")
    
#     calc_type = st.selectbox("Choose Calculator:", [
#         "EMI Calculator", "SIP Calculator", "Compound Interest", "Tax Saver"
#     ])
    
#     if calc_type == "EMI Calculator":
#         col1, col2, col3 = st.columns(3)
#         with col1:
#             principal = st.number_input("Loan Amount (₹)", min_value=1000, value=100000)
#         with col2:
#             rate = st.number_input("Interest Rate (%)", min_value=0.1, value=10.0)
#         with col3:
#             tenure = st.number_input("Tenure (Years)", min_value=1, value=5)
        
#         if st.button("Calculate EMI"):
#             monthly_rate = rate / (12 * 100)
#             months = tenure * 12
#             emi = (principal * monthly_rate * (1 + monthly_rate)**months) / ((1 + monthly_rate)**months - 1)
            
#             result = f"Your EMI would be ₹{emi:,.2f} per month for {tenure} years."
#             st.session_state.chat_history.append({
#                 'role': 'ai',
#                 'message': result,
#                 'timestamp': datetime.now().strftime("%H:%M")
#             })
#             st.rerun()

# def chat_interface():
#     """Main chat interface"""
#     st.subheader("💬 Chat with AI Financial Advisor")
    
#     # Language selection
#     col1, col2, col3 = st.columns([2, 1, 1])
#     with col1:
#         selected_lang = st.selectbox(
#             "🌐 Language",
#             ['English', 'Hindi', 'Tamil', 'Telugu', 'Bengali', 'Marathi', 'Gujarati'],
#             index=0
#         )
#         st.session_state.selected_language = selected_lang
    
#     with col2:
#         voice_enabled = st.checkbox("🎤 Voice Mode")
#         st.session_state.voice_enabled = voice_enabled
    
#     with col3:
#         if st.button("🗑️ Clear Chat"):
#             st.session_state.chat_history = []
#             st.rerun()
    
#     # Chat history display
#     chat_container = st.container()
#     with chat_container:
#         if not st.session_state.chat_history:
#             st.markdown("""
#             <div style='text-align: center; padding: 20px; background-color: #f0f8ff; border-radius: 10px; margin: 20px 0;'>
#                 <h3>👋 Hello! I'm your AI Financial Advisor</h3>
#                 <p>I can help you with:</p>
#                 <ul style='text-align: left; display: inline-block;'>
#                     <li>📊 Analyzing your spending patterns</li>
#                     <li>🎯 Tracking your financial goals</li>
#                     <li>💰 Budget planning and optimization</li>
#                     <li>📈 Investment recommendations</li>
#                     <li>💡 Personalized financial advice</li>
#                 </ul>
#                 <p><strong>Ask me anything about your finances!</strong></p>
#             </div>
#             """, unsafe_allow_html=True)
#         else:
#             for chat in st.session_state.chat_history:
#                 display_chat_message(chat['role'], chat['message'], chat['timestamp'])
    
#     # Quick questions
#     quick_question = quick_questions()
#     if quick_question:
#         st.session_state.chat_history.append({
#             'role': 'user',
#             'message': quick_question,
#             'timestamp': datetime.now().strftime("%H:%M")
#         })
        
#         # Get AI response
#         financial_context = create_financial_context()
#         ai_response = get_gemini_response(quick_question, financial_context)
        
#         # Translate if needed
#         if st.session_state.selected_language != 'English':
#             ai_response = translate_text(ai_response, st.session_state.selected_language)
        
#         st.session_state.chat_history.append({
#             'role': 'ai',
#             'message': ai_response,
#             'timestamp': datetime.now().strftime("%H:%M")
#         })
#         st.rerun()
    
#     # Chat input area
#     st.markdown("---")
#     col1, col2, col3 = st.columns([3, 1, 1])
    
#     with col1:
#         user_input = st.text_input(
#             "Type your message...",
#             placeholder="Ask me about your finances, goals, or get investment advice!",
#             key="chat_input"
#         )
    
#     with col2:
#         send_button = st.button("📤 Send", type="primary")
    
#     with col3:
#         if voice_enabled:
#             voice_button = st.button("🎤 Voice")
#             if voice_button:
#                 voice_text = speech_to_text()
#                 if voice_text:
#                     user_input = voice_text
#                     send_button = True
    
#     # Process user input
#     if send_button and user_input:
#         # Add user message to chat
#         st.session_state.chat_history.append({
#             'role': 'user',
#             'message': user_input,
#             'timestamp': datetime.now().strftime("%H:%M")
#         })
        
#         # Get financial context
#         financial_context = create_financial_context()
        
#         # Get AI response
#         with st.spinner("🤖 AI is thinking..."):
#             ai_response = get_gemini_response(user_input, financial_context)
        
#         # Translate if needed
#         if st.session_state.selected_language != 'English':
#             ai_response = translate_text(ai_response, st.session_state.selected_language)
        
#         # Add AI response to chat
#         st.session_state.chat_history.append({
#             'role': 'ai',
#             'message': ai_response,
#             'timestamp': datetime.now().strftime("%H:%M")
#         })
        
#         # Text-to-speech if enabled
#         if voice_enabled:
#             text_to_speech(ai_response)
        
#         st.rerun()

# def chat_analytics():
#     """Analytics for chat interactions"""
#     st.subheader("📊 Chat Analytics")
    
#     if not st.session_state.chat_history:
#         st.info("📊 No chat data available yet.")
#         return
    
#     total_messages = len(st.session_state.chat_history)
#     user_messages = len([m for m in st.session_state.chat_history if m['role'] == 'user'])
#     ai_messages = len([m for m in st.session_state.chat_history if m['role'] == 'ai'])
    
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         st.metric("Total Messages", total_messages)
#     with col2:
#         st.metric("Your Messages", user_messages)
#     with col3:
#         st.metric("AI Responses", ai_messages)
    
#     # Most common topics (simple keyword analysis)
#     if user_messages > 0:
#         user_texts = [m['message'].lower() for m in st.session_state.chat_history if m['role'] == 'user']
#         all_text = ' '.join(user_texts)
        
#         financial_keywords = ['budget', 'save', 'spend', 'invest', 'goal', 'money', 'income', 'expense']
#         keyword_counts = {word: all_text.count(word) for word in financial_keywords if all_text.count(word) > 0}
        
#         if keyword_counts:
#             st.write("**💡 Your Most Discussed Topics:**")
#             for word, count in sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
#                 st.write(f"• **{word.title()}**: mentioned {count} times")

# def export_chat():
#     """Export chat history"""
#     if st.session_state.chat_history:
#         chat_df = pd.DataFrame(st.session_state.chat_history)
#         csv = chat_df.to_csv(index=False)
        
#         st.download_button(
#             label="📥 Download Chat History",
#             data=csv,
#             file_name=f"ai_chat_history_{datetime.now().strftime('%Y%m%d')}.csv",
#             mime="text/csv"
#         )

# def main():
#     """Main function for AI Financial Advisor Chat page"""
    
#     st.title("🤖 AI Financial Advisor Chat")
#     st.markdown("---")
    
#     # Initialize session state
#     initialize_chat_session()
    
#     # Check for financial data
#     financial_context = create_financial_context()
#     if not financial_context['has_data']:
#         st.warning("⚠️ No financial data found! The AI can provide general advice, but for personalized recommendations, please add your financial data in the 'Data Input & Management' page first.")
    
#     # Create tabs
#     tab1, tab2, tab3 = st.tabs(["💬 Chat", "🧮 Calculators", "📊 Analytics"])
    
#     with tab1:
#         chat_interface()
    
#     with tab2:
#         financial_calculator_chat()
    
#     with tab3:
#         chat_analytics()
#         st.markdown("---")
#         export_chat()

# if __name__ == "__main__":
#     main()

from logging import PlaceHolder
import os
import json
from datetime import datetime, date

import base64
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Optional dependencies. Wrap imports to avoid app crash if missing.
try:
    import speech_recognition as sr
except Exception:
    sr = None

try:
    import pyttsx3
except Exception:
    pyttsx3 = None

try:
    from googletrans import Translator
except Exception:
    Translator = None

try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None

# ============ App Config & Theme ============
st.set_page_config(
    page_title="AI Financial Advisor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

PRIMARY = "#1F6FEB" # Primary accent
BG_LIGHT = "#F8FAFC" # Background panels
TEXT_SUBTLE = "rgba(0,0,0,0.6)"
BORDER = "#E5E7EB"
SUCCESS = "#10B981"
WARN = "#F59E0B"
ERROR = "#EF4444"

def inject_base_css():
    st.markdown(
        f"""
        <style>
        /* Global */
        .main {{
            padding-top: 20px;
            padding-bottom: 40px;
            background: white;
        }}
        /* Headings */
        .section-header {{
            display: flex; align-items: center; gap: 8px;
            margin: 6px 0 12px 0;
        }}
        .section-header h3 {{
            margin: 0; font-weight: 700;
        }}
        /* Cards */
        .card {{
            background: {BG_LIGHT};
            border: 1px solid {BORDER};
            border-radius: 12px;
            padding: 16px;
        }}
        .subtle {{
            color: {TEXT_SUBTLE};
        }}
        /* Chat bubbles */
        .bubble {{
            max-width: 70%;
            padding: 12px 14px;
            border-radius: 14px;
            margin-bottom: 10px;
            line-height: 1.4;
            word-wrap: break-word;
        }}
        .bubble-user {{
            background: {PRIMARY}; color: white;
            margin-left: auto; border-bottom-right-radius: 6px;
        }}
        .bubble-ai {{
            background: #F3F4F6; color: #111827;
            margin-right: auto; border-bottom-left-radius: 6px;
            border: 1px solid {BORDER};
        }}
        .bubble-meta {{
            font-size: 12px; margin-top: 6px; opacity: 0.7;
        }}
        /* Pill buttons */
        .pill {{
            display: inline-block;
            padding: 8px 12px;
            border: 1px solid {BORDER};
            border-radius: 999px;
            margin: 6px 6px 0 0;
            background: white;
            cursor: pointer;
        }}
        .pill:hover {{
            border-color: {PRIMARY};
            color: {PRIMARY};
        }}
        /* Divider spacing */
        .divider {{
            height: 1px; background: {BORDER}; margin: 12px 0 16px 0;
        }}
        .welcome {{
            text-align: center; padding: 16px;
            background: #F0F8FF; border-radius: 12px; border: 1px solid {BORDER};
        }}
        .metric-card {{
            background: white; border: 1px solid {BORDER};
            border-radius: 10px; padding: 14px;
        }}
        .danger-text {{
            color: {ERROR}; font-weight: 600;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ============ Session State ============
def initialize_chat_session():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'selected_language' not in st.session_state:
        st.session_state.selected_language = 'English'
    if 'voice_enabled' not in st.session_state:
        st.session_state.voice_enabled = False

# ============ Data & Context ============
def get_combined_financial_data():
    combined_df = pd.DataFrame()

    # Manual entries
    if 'manual_entries' in st.session_state and st.session_state.manual_entries:
        manual_df = pd.DataFrame(st.session_state.manual_entries)
        combined_df = pd.concat([combined_df, manual_df], ignore_index=True)

    # Uploaded data
    if 'uploaded_data' in st.session_state and isinstance(st.session_state.uploaded_data, pd.DataFrame):
        uploaded_df = st.session_state.uploaded_data.copy()
        uploaded_df.columns = uploaded_df.columns.str.lower()
        combined_df = pd.concat([combined_df, uploaded_df], ignore_index=True)

    if not combined_df.empty:
        # Required columns guards
        for col in ['date', 'amount', 'type', 'category', 'description']:
            if col not in combined_df.columns:
                combined_df[col] = np.nan

        combined_df['date'] = pd.to_datetime(combined_df['date'], errors='coerce')
        combined_df['amount'] = pd.to_numeric(combined_df['amount'], errors='coerce').fillna(0)
        combined_df['type'] = combined_df['type'].astype(str).str.lower()
        combined_df['category'] = combined_df['category'].astype(str)
        combined_df['description'] = combined_df['description'].astype(str)
        combined_df = combined_df.sort_values('date')

    return combined_df

def get_goals_data():
    return st.session_state.get('financial_goals', [])

def create_financial_context():
    df = get_combined_financial_data()
    goals = get_goals_data()

    context = {
        'has_data': not df.empty,
        'total_transactions': len(df) if not df.empty else 0,
        'total_income': 0.0,
        'total_expenses': 0.0,
        'net_balance': 0.0,
        'top_expense_categories': [],
        'goals': goals,
        'recent_transactions': []
    }

    if not df.empty:
        income_df = df[df['type'] == 'income']
        expense_df = df[df['type'] == 'expense']

        context['total_income'] = float(income_df['amount'].sum()) if not income_df.empty else 0.0
        context['total_expenses'] = float(expense_df['amount'].sum()) if not expense_df.empty else 0.0
        context['net_balance'] = context['total_income'] - context['total_expenses']

        if not expense_df.empty and 'category' in expense_df.columns:
            top_categories = expense_df.groupby('category')['amount'].sum().nlargest(3)
            context['top_expense_categories'] = [
                {'category': cat, 'amount': float(amount)}
                for cat, amount in top_categories.items()
            ]

        # Recent transactions
        recent_cols = [c for c in ['date', 'description', 'amount', 'type', 'category'] if c in df.columns]
        recent = df.tail(5)[recent_cols].to_dict('records')
        context['recent_transactions'] = [
            {
                'date': (r.get('date').strftime('%Y-%m-%d') if pd.notna(r.get('date')) else ''),
                'description': r.get('description', ''),
                'amount': float(r.get('amount', 0) or 0),
                'type': r.get('type', ''),
                'category': r.get('category', '')
            }
            for r in recent
        ]

    return context

# ============ AI Layer ============
def get_gemini_response(user_message, financial_context):
    # Never hardcode API keys; expect from env or secrets
    api_key = "AIzaSyDcrJg4kH1TGtQutFmu0gjL65BAhdR3riQ"
    if not genai or not types:
        return "AI service is not available in this environment."
    if not api_key:
        return "Gemini API key not found. Please set GEMINI_API_KEY in environment or Streamlit secrets."

    try:
        client = genai.Client(api_key=api_key)
        model = "gemini-2.0-flash"

        system_prompt = f"""
        You are an expert AI Financial Advisor. You provide personalized financial advice based on the user's actual financial data.

        USER'S FINANCIAL CONTEXT:

        Has financial data: {financial_context['has_data']}

        Total transactions: {financial_context['total_transactions']}

        Total income: ₹{financial_context['total_income']:,.2f}

        Total expenses: ₹{financial_context['total_expenses']:,.2f}

        Net balance: ₹{financial_context['net_balance']:,.2f}

        Top expense categories: {financial_context['top_expense_categories']}

        Active goals: {len(financial_context['goals'])}

        Goals details: {financial_context['goals']}

        Recent transactions: {financial_context['recent_transactions']}

        INSTRUCTIONS:

        Provide personalized advice based on the actual financial data above

        Be specific with numbers from their data

        Give actionable recommendations

        Be encouraging and supportive

        Use Indian currency format (₹)

        Keep responses concise but informative

        If user asks about specific data, reference their actual transactions

        Suggest specific budget allocations based on their spending patterns

        Help with goal planning using their financial capacity

        User's question: {user_message}
        """.strip()

        contents = [types.Content(role="user", parts=[types.Part.from_text(text=system_prompt)])]

        tools = [types.Tool(googleSearch=types.GoogleSearch())]

        generate_content_config = types.GenerateContentConfig(
            tools=tools,
            temperature=0.7,
        )

        response_text = ""
        for chunk in client.models.generate_content_stream(
            model=model, contents=contents, config=generate_content_config
        ):
            if hasattr(chunk, 'text') and chunk.text:
                response_text += chunk.text
        st.text(response_text)
        return response_text or "I'm having trouble processing your request. Please try again."

    except Exception as e:
        return f"Error connecting to AI: {str(e)}"

# ============ Utilities: Translate, STT, TTS ============
def translate_text(text, target_language):
    if target_language == 'English':
        return text
    if Translator is None:
        return text

    try:
        translator = Translator()
        lang_codes = {
            'Hindi': 'hi', 'Tamil': 'ta', 'Telugu': 'te',
            'Bengali': 'bn', 'Marathi': 'mr', 'Gujarati': 'gu',
            'Kannada': 'kn', 'Malayalam': 'ml', 'Punjabi': 'pa'
        }
        code = lang_codes.get(target_language)
        if code:
            result = translator.translate(text, dest=code)
            return result.text
        return text
    except Exception:
        return text

def speech_to_text():
    if sr is None:
        st.warning("Speech recognition is not available in this environment.")
        return None

    try:
        r = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("Listening... Speak now.")
            r.adjust_for_ambient_noise(source, duration=1)
            audio = r.listen(source, timeout=10, phrase_time_limit=15)

        st.info("Processing speech...")
        text = r.recognize_google(audio)
        return text
    except sr.RequestError as e:
        st.error(f"Could not request results: {e}")
        return None
    except sr.UnknownValueError:
        st.error("Could not understand audio.")
        return None
    except Exception as e:
        st.error(f"Speech error: {e}")
        return None

def text_to_speech(text):
    # Keep placeholder behavior; avoid blocking the UI.
    try:
        st.success(f"Playing: {text[:100]}...")
        return True
    except Exception as e:
        st.error(f"TTS Error: {e}")
        return False

# ============ UI Components ============
def header():
    st.title("AI Financial Advisor")
    st.caption("Personalized insights, budgeting help, and goal planning. Powered by your data.")
    inject_base_css()
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

def display_chat_message(role, message, timestamp):
    if role == "user":
        html = f"""
        <div style='display:flex; justify-content:flex-end;'>
            <div class='bubble bubble-user'>
                <strong>You: </strong>
                {message}\n{timestamp}
            </div>
        </div>
        """
    else:
        html = f"""
        <div style='display:flex; justify-content:flex-start;'>
            <div class='bubble bubble-ai'>
                <strong>AI Advisor: </strong>
                {message}\n{timestamp}
        """
    st.markdown(html, unsafe_allow_html=True)

def quick_questions_ui():
    st.markdown("<div class='section-header'><h3>Quick Questions</h3></div>", unsafe_allow_html=True)

    questions = [
        "How is my spending this month?",
        "Am I on track with my goals?",
        "What should I invest in?",
        "How can I save more money?",
        "Analyze my top expenses",
        "Create a budget plan",
        "When will I reach my savings goal?",
        "Should I take a loan?"
    ]

    cols = st.columns(4)
    clicked = None
    for i, q in enumerate(questions):
        with cols[i % 4]:
            if st.button(f"-  {q}", key=f"quick_{i}", use_container_width=True):
                clicked = q
    return clicked

def calculators_ui():
    st.markdown("<div class='section-header'><h3>Financial Calculators</h3></div>", unsafe_allow_html=True)
    with st.container():
        with st.expander("EMI Calculator", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                principal = st.number_input("Loan Amount (₹)", min_value=1000.0, value=100000.0, step=1000.0)
            with c2:
                rate = st.number_input("Interest Rate (%)", min_value=0.1, value=10.0, step=0.1)
            with c3:
                tenure = st.number_input("Tenure (Years)", min_value=1, value=5, step=1)

            if st.button("Calculate EMI", type="primary"):
                monthly_rate = rate / (12 * 100)
                months = tenure * 12
                emi = (principal * monthly_rate * (1 + monthly_rate) ** months) / ((1 + monthly_rate) ** months - 1)
                st.success(f"Your EMI would be ₹{emi:,.2f} per month for {tenure} years.")

                st.session_state.chat_history.append({
                    'role': 'ai',
                    'message': f"EMI result: ₹{emi:,.2f} per month for {tenure} years.",
                    'timestamp': datetime.now().strftime("%H:%M")
                })

def analytics_ui():
    st.markdown("<div class='section-header'><h3>Chat Analytics</h3></div>", unsafe_allow_html=True)
    if not st.session_state.chat_history:
        st.info("No chat data available yet.")
        return

    total_messages = len(st.session_state.chat_history)
    user_messages = len([m for m in st.session_state.chat_history if m['role'] == 'user'])
    ai_messages = len([m for m in st.session_state.chat_history if m['role'] == 'ai'])

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Messages", total_messages)
    with c2:
        st.metric("Your Messages", user_messages)
    with c3:
        st.metric("AI Responses", ai_messages)

    # Topics
    if user_messages > 0:
        user_texts = [m['message'].lower() for m in st.session_state.chat_history if m['role'] == 'user']
        all_text = ' '.join(user_texts)
        financial_keywords = ['budget', 'save', 'spend', 'invest', 'goal', 'money', 'income', 'expense', 'emi', 'loan']
        keyword_counts = {word: all_text.count(word) for word in financial_keywords if all_text.count(word) > 0}

        if keyword_counts:
            st.write("Your most discussed topics:")
            for word, count in sorted(keyword_counts.items(), key=lambda x: x, reverse=True)[:5]:
                st.write(f"-  {word.title()}: {count} times")

def export_chat_ui():
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    if st.session_state.chat_history:
        chat_df = pd.DataFrame(st.session_state.chat_history)
        csv = chat_df.to_csv(index=False)
        st.download_button(
            label="Download Chat History (CSV)",
            data=csv,
            file_name=f"ai_chat_history_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )

# ============ Chat Interface ============
def chat_interface():
    st.markdown("<div class='section-header'><h3>Chat with AI Financial Advisor</h3></div>", unsafe_allow_html=True)
    with st.container():
        c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
        with c1:
            selected_lang = st.selectbox(
                "Language",
                ['English', 'Hindi', 'Tamil', 'Telugu', 'Bengali', 'Marathi', 'Gujarati'],
                index=0
            )
            st.session_state.selected_language = selected_lang
        with c2:
            st.text("")
            st.text("")

            voice_enabled = st.toggle("Voice Mode", value=st.session_state.voice_enabled)
            st.session_state.voice_enabled = voice_enabled
        with c3:
            st.text("")
            st.text("")

            if st.button("Clear Chat", use_container_width=True):
                # Confirmation pattern
                st.session_state._confirm_clear = True
        with c4:
            st.empty()

        if st.session_state.get("_confirm_clear", False):
            st.warning("Are you sure you want to clear chat?")
            cc1, cc2 = st.columns(2)
            with cc1:
                if st.button("Yes, clear"):
                    st.session_state.chat_history = []
                    st.session_state._confirm_clear = False
                    st.rerun()
            with cc2:
                if st.button("Cancel"):
                    st.session_state._confirm_clear = False

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # Chat history display
        chat_container = st.container()
        with chat_container:
            if not st.session_state.chat_history:
                st.markdown("""
                <div class='welcome'>
                    <h3>👋 Hello! I'm your AI Financial Advisor</h3>
                    <p>I can help you with:</p>
                    <div style='display:inline-block; text-align:left;'>
                        <ul>
                            <li>📊 Analyzing your spending patterns</li>
                            <li>🎯 Tracking your financial goals</li>
                            <li>💰 Budget planning and optimization</li>
                            <li>📈 Investment recommendations</li>
                            <li>💡 Personalized financial advice</li>
                        </ul>
                    </div>
                    <p><strong>Ask me anything about your finances!</strong></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                for chat in st.session_state.chat_history:
                    display_chat_message(chat['role'], chat['message'], chat['timestamp'])

        # Quick questions
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        quick_q = quick_questions_ui()
        if quick_q:
            process_user_message(quick_q)

        # Input area
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        c_in1, c_in2, c_in3 = st.columns([4, 1, 1])
        with c_in1:
            user_input = st.text_input(
                "Type your message...",
                placeholder="Ask me about your finances, goals, or get investment advice!",
                key="chat_input"
            )
        with c_in2:
            st.text("")
            send_button = st.button("Send", type="primary", use_container_width=True,)
        with c_in3:
            st.text("")

            record_pressed = False
            if st.session_state.voice_enabled:
                record_pressed = st.button("🎤 Speak", use_container_width=True)

        # Voice capture
        if record_pressed:
            voice_text = speech_to_text()
            if voice_text:
                user_input = voice_text
                send_button = True

        # Process
        if send_button and user_input:
            process_user_message(user_input)

def process_user_message(message_text):
    st.session_state.chat_history.append({
        'role': 'user',
        'message': message_text,
        'timestamp': datetime.now().strftime("%H:%M")
    })

    financial_context = create_financial_context()
    with st.spinner("AI is thinking..."):
        ai_response = get_gemini_response(message_text, financial_context)

    if st.session_state.selected_language != 'English':
        ai_response = translate_text(ai_response, st.session_state.selected_language)

    st.session_state.chat_history.append({
        'role': 'ai',
        'message': ai_response,
        'timestamp': datetime.now().strftime("%H:%M")
    })

    if st.session_state.voice_enabled:
        text_to_speech(ai_response)

    st.rerun()

# ============ Main ============
def main():
    header()
    initialize_chat_session()

    # Context notice
    financial_context = create_financial_context()
    if not financial_context['has_data']:
        st.info("No financial data found. The AI can provide general advice. For personalized insights, add your data in the Data Input & Management page.")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "🧮 Calculators", "📊 Analytics"])

    with tab1:
        chat_interface()

    with tab2:
        calculators_ui()

    with tab3:
        analytics_ui()
        export_chat_ui()

if __name__ == "__main__":
    main()