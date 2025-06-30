import streamlit as st
import os
import sys
from sql_ai import SQLGenerator, DEFAULT_CONTEXT_FILE, get_last_n_sessions, add_session_to_history
from dotenv import load_dotenv

# Get the directory where this file is located
current_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(current_dir, ".env")

# Load environment variables from .env file
load_dotenv(dotenv_path)

# st.set_page_config(page_title="SQL Chat Assistant", layout="wide")
st.set_page_config(page_title="CeDeFi SQL Assistant", layout="wide")

# Initialize session state for conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "sql_generator" not in st.session_state:
    st.session_state.sql_generator = None

if "conversation_saved" not in st.session_state:
    st.session_state.conversation_saved = False

def initialize_sql_generator(api_key):
    """Initialize the SQL generator with the provided API key."""
    try:
        return SQLGenerator(api_key, context_file_path=DEFAULT_CONTEXT_FILE)
    except Exception as e:
        st.error(f"Error initializing SQL Generator: {str(e)}")
        return None

def display_message(message, is_user=True):
    """Display a message in the chat interface."""
    if is_user:
        with st.chat_message("user"):
            st.write(message)
    else:
        with st.chat_message("assistant"):
            st.write(message)

def display_sql_response(response):
    """Display only the SQL query in a clean output box."""
    with st.chat_message("assistant"):
        # Display SQL query in a code block
        if isinstance(response, dict):
            st.code(response["sql_query"], language="sql")
            # Show modification indicator if applicable
            if response.get("is_modification", False):
                st.info("🔄 Modified based on your feedback")
        else:
            # Handle simple string response
            st.code(response, language="sql")

def main():
    st.title("🤖 CeDeFi SQL Assistant")
    st.markdown("Chat with AI to generate and refine SQL queries iteratively!")
    
    # Auto-save any existing conversation when the app loads
    if st.session_state.messages and len(st.session_state.messages) >= 2 and not st.session_state.conversation_saved:
        add_session_to_history(st.session_state.messages)
        st.session_state.conversation_saved = True
    
    tab = "Chat"  # Only keep the chat tab, remove history tab
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Check if API key exists in environment variable
        api_key_from_env = os.getenv("OPENAI_API_KEY")
        if api_key_from_env and api_key_from_env != "your_api_key_here":
            st.success("✅ OpenAI API key found in environment variable")
            using_env_key = st.checkbox("Use API key from environment", value=True)
        else:
            st.warning("⚠️ No valid API key found in environment")
            using_env_key = False
            
            # Show instructions for setting up .env file
            with st.expander("How to set up API key"):
                st.markdown("""
                1. Create a file named `.env` in the `sql generator` directory
                2. Add this line to the file:
                ```
                OPENAI_API_KEY=your_actual_api_key_here
                ```
                3. Restart the app
                """)
        
        # Input for API key if not using from environment
        if not using_env_key:
            api_key = st.text_input("Enter OpenAI API Key", type="password")
        else:
            api_key = api_key_from_env
        
        # Initialize SQL generator
        if api_key and api_key != "your_api_key_here":
            if st.session_state.sql_generator is None:
                st.session_state.sql_generator = initialize_sql_generator(api_key)
        elif not api_key:
            st.error("Please enter your OpenAI API key")
        else:
            st.error("Please replace 'your_api_key_here' with your actual OpenAI API key")
        
        # Conversation management
        st.header("🗑️ Conversation")
        
        # Clear conversation
        if st.button("🗑️ Clear Conversation", type="secondary"):
            st.session_state.messages = []
            st.session_state.conversation_saved = False
            st.rerun()
        # Add history button and dropdown
        st.header("🕑 History")
        show_history = st.checkbox("Show History", value=False)
        if show_history:
            sessions = get_last_n_sessions(20)
            session_titles = [s[0]["content"] if s and isinstance(s[0], dict) and s[0].get("role") == "user" else f"Session #{len(sessions)-i}" for i, s in enumerate(reversed(sessions))]
            selected_idx = st.selectbox("Select a conversation", options=list(range(len(session_titles))), format_func=lambda i: session_titles[i] if i < len(session_titles) else f"Session {i+1}")
            # Show the selected conversation
            if sessions:
                session = list(reversed(sessions))[selected_idx]
                for msg in session:
                    if not isinstance(msg, dict):
                        continue
                    if msg.get("role") == "user":
                        st.markdown(f"**User:** {msg['content']}")
                    else:
                        ai_content = msg["content"]
                        st.markdown(f"**AI:**")
                        if isinstance(ai_content, dict):
                            st.code(ai_content.get("sql_query", str(ai_content)), language="sql")
                            if ai_content.get("explanation"):
                                st.markdown(f"_Explanation:_ {ai_content['explanation']}")
                        else:
                            st.code(str(ai_content), language="sql")
    
    if tab == "Chat":
        if not st.session_state.sql_generator:
            st.warning("⚠️ Please configure a valid OpenAI API key in the sidebar to start chatting")
            st.stop()
        # Display conversation history
        for message in st.session_state.messages:
            if message["role"] == "user":
                display_message(message["content"], is_user=True)
            else:
                if isinstance(message["content"], dict):
                    display_sql_response(message["content"])
                else:
                    display_sql_response(message["content"])
        # Chat input (must be outside any container)
        prompt = st.chat_input("Ask me to generate or modify a SQL query...")
        if prompt:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            display_message(prompt, is_user=True)
            # Generate response
            with st.spinner("🤔 Thinking..."):
                try:
                    # Prepare conversation history for the AI
                    conversation_history = []
                    for msg in st.session_state.messages[:-1]:  # Exclude the current message
                        if msg["role"] == "user":
                            conversation_history.append((msg["content"], None))
                        else:
                            # Get the last user message to pair with this AI response
                            if conversation_history and conversation_history[-1][1] is None:
                                ai_content = msg["content"]
                                if isinstance(ai_content, dict):
                                    ai_content = ai_content.get("sql_query", str(ai_content))
                                conversation_history[-1] = (conversation_history[-1][0], ai_content)
                    # Generate response using conversational method (query only)
                    response = st.session_state.sql_generator.generate_conversational_sql_simple(
                        prompt, 
                        conversation_history=conversation_history
                    )
                    # Add AI response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    # Auto-save conversation after each complete exchange (2 messages: user + assistant)
                    if len(st.session_state.messages) >= 2 and not st.session_state.conversation_saved:
                        add_session_to_history(st.session_state.messages)
                        st.session_state.conversation_saved = True
                    
                    # Display the response
                    display_sql_response(response)
                except Exception as e:
                    error_message = f"Error generating SQL query: {str(e)}"
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
                    
                    # Auto-save conversation even if there's an error
                    if len(st.session_state.messages) >= 2 and not st.session_state.conversation_saved:
                        add_session_to_history(st.session_state.messages)
                        st.session_state.conversation_saved = True
                    
                    display_message(error_message, is_user=False)
        # Instructions and examples
        if not st.session_state.messages:
            st.markdown("---")
            st.markdown("### 💡 How to use this chat:")
            # st.markdown("### 💡 Describe your requirements to generate a SQL query:")
            st.markdown("""  
            - **Generate a new query**: "Find all users who placed orders in the last 30 days"
            - **Modify a query**: "Change the date range to last 7 days instead"
            - **Fix issues**: "The query has an error, fix the JOIN syntax"
                       
            The AI will remember our conversation and can refine queries based on your feedback!
            """)
            # st.markdown("""
            # - **Generate a new query**: "Find all users who placed orders in the last 30 days"
            # - **Modify a query**: "Change the date range to last 7 days instead"
            # - **Fix issues**: "The query has an error, fix the JOIN syntax"
            # - **Add conditions**: "Also filter by product category 'electronics'"
            # - **Simplify**: "Make the query simpler, just show user names and order counts"
            
            # The AI will remember our conversation and can refine queries based on your feedback!
            # """)

if __name__ == "__main__":
    main() 