import os
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
import streamlit as st
import json
import threading

# Get the directory where this file is located and load .env from there
current_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(current_dir, ".env")

# Load environment variables from .env file
load_dotenv(dotenv_path)

# Default context file path
DEFAULT_CONTEXT_FILE = os.path.join(current_dir, "sample_context.txt")

HISTORY_FILE = os.path.join(current_dir, "chat_history.json")
history_lock = threading.Lock()

def load_chat_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    with history_lock, open(HISTORY_FILE, "r") as f:
        try:
            return json.load(f)
        except Exception:
            return []

def save_chat_history(history):
    with history_lock, open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

def add_session_to_history(session):
    """
    Add a conversation (list of message dicts) to history.
    If the new session is a continuation of the last, replace it.
    Only saves legitimate conversations with proper format.
    """
    if not is_legitimate_session(session):
        print("Warning: Skipping invalid session format")
        return

    history = load_chat_history()
    if history:
        last = history[-1]
        # If the new session is a continuation of the last, replace it
        if len(session) > len(last) and session[:len(last)] == last:
            history[-1] = session
        elif last != session:
            history.append(session)
    else:
        history.append(session)
    # Keep only the last 10 sessions
    history = history[-10:]
    save_chat_history(history)

def is_legitimate_session(session):
    """
    Check if a session has the legitimate format (list of dicts with role/content).
    """
    if not isinstance(session, list) or len(session) == 0:
        return False
    
    for message in session:
        if not isinstance(message, dict) or 'role' not in message or 'content' not in message:
            return False
        if message['role'] not in ['user', 'assistant']:
            return False
    
    return True

def get_last_n_sessions(n=10):
    history = load_chat_history()
    return history[-n:]

def conversation_tuples_to_session(conversation_history):
    """
    Convert a list of (user_msg, ai_response) tuples to a session list of dicts.
    Handles ai_response as dict or string.
    """
    session = []
    for user_msg, ai_response in conversation_history:
        session.append({"role": "user", "content": user_msg})
        if isinstance(ai_response, dict):
            # Store the whole dict as content for richer history
            session.append({"role": "assistant", "content": ai_response})
        else:
            session.append({"role": "assistant", "content": ai_response})
    return session

class SQLGenerator:
    def __init__(self, openai_api_key=None, context_file_path=None):
        """Initialize the SQL Generator with OpenAI API key."""
        if openai_api_key:
            self.openai_api_key = openai_api_key
            os.environ["OPENAI_API_KEY"] = openai_api_key
        else:
            # Try to get API key from environment, print debug info
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
            print(f"Loading API key from environment: {'Found' if self.openai_api_key else 'Not found'}")
            
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required")
        
        # Initialize the language model
        self.llm = ChatOpenAI(
            model_name="gpt-4-turbo",
            temperature=0.3,
            openai_api_key=self.openai_api_key
        )
        
        # Initialize embeddings model
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        
        # Load default context if provided
        self.default_context = None
        self.context_store = None
        if context_file_path:
            try:
                self.default_context = self.load_context_from_file(context_file_path)
                # Create vector store from context
                self.prepare_context_store(self.default_context)
            except FileNotFoundError:
                print(f"Warning: Default context file not found at {context_file_path}")
    
    def load_context_from_file(self, file_path):
        """Load SQL context, queries, and schema from a text file."""
        try:
            with open(file_path, 'r') as file:
                return file.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Context file not found: {file_path}")
    
    def prepare_context_store(self, context_text):
        """Prepare the vector store from context text."""
        # Split the context into chunks for better retrieval
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", ";"]
        )
        context_chunks = text_splitter.split_text(context_text)
        
        # Create vector store
        self.context_store = FAISS.from_texts(context_chunks, self.embeddings)
    
    def get_relevant_context(self, query, k=3):
        """Get most relevant context chunks for a query."""
        if not self.context_store:
            return self.default_context or ""
        
        # Retrieve relevant chunks
        relevant_docs = self.context_store.similarity_search(query, k=k)
        relevant_text = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Add a header to clarify this is a subset
        return f"RELEVANT DATABASE CONTEXT:\n{relevant_text}"
    
    def generate_sql_query(self, user_prompt, context_text=None):
        """Generate SQL query based on user prompt and context."""
        # If explicit context is provided, use it
        if context_text and context_text.strip() != "":
            # Prepare vector store for this specific context if needed
            if not self.context_store or context_text != self.default_context:
                self.prepare_context_store(context_text)
        
        # Get relevant context based on query
        filtered_context = self.get_relevant_context(user_prompt)
            
        # Create prompt template
        template = """
        You are a SQL expert. Given the following context and user request, generate an appropriate SQL query.
        
        {context_text}
        
        USER REQUEST: {user_prompt}
        
        Provide only the SQL query without explanation, comments, or additional text.

        please follow the following rules:  
        1. It is not allowed to use window functions inside WHERE clause.
        2. It is not allowed to use window functions inside HAVING clause.
        """
        
        if filtered_context == "":
            template = """
            You are a SQL expert. Given the following user request, generate an appropriate SQL query.
            
            USER REQUEST: {user_prompt}
            
            Provide only the SQL query without explanation, comments, or additional text.
            
            please keep in mind 
            1. It is not allowed to use window functions inside WHERE clause.
            """
            
        prompt_template = PromptTemplate(
            input_variables=["user_prompt", "context_text"] if "context_text" in template else ["user_prompt"],
            template=template
        )
        
        # Create the chain
        chain = LLMChain(llm=self.llm, prompt=prompt_template)
        
        # Generate SQL query
        if "context_text" in template:
            result = chain.run(user_prompt=user_prompt, context_text=filtered_context)
        else:
            result = chain.run(user_prompt=user_prompt)
            
        return result.strip()
    
    def generate_sql_query_with_explanation(self, user_prompt, context_text=None):
        """Generate SQL query with explanation based on user prompt and context."""
        # If explicit context is provided, use it
        if context_text and context_text.strip() != "":
            # Prepare vector store for this specific context if needed
            if not self.context_store or context_text != self.default_context:
                self.prepare_context_store(context_text)
        
        # Get relevant context based on query
        filtered_context = self.get_relevant_context(user_prompt)
            
        # Create prompt template
        template = """
        You are a SQL expert. Given the following context and user request, generate an appropriate SQL query.
        
        {context_text}
        
        USER REQUEST: {user_prompt}
        
        Return a JSON with these fields:
        1. sql_query: The SQL query to solve the problem
        2. explanation: Step-by-step explanation of what the query does
        """
        
        if filtered_context == "":
            template = """
            You are a SQL expert. Given the following user request, generate an appropriate SQL query.
            
            USER REQUEST: {user_prompt}
            
            Return a JSON with these fields:
            1. sql_query: The SQL query to solve the problem
            2. explanation: Step-by-step explanation of what the query does
            """
            
        prompt_template = PromptTemplate(
            input_variables=["user_prompt", "context_text"] if "context_text" in template else ["user_prompt"],
            template=template
        )
        
        # Create the chain
        chain = LLMChain(llm=self.llm, prompt=prompt_template)
        
        # Generate SQL query with explanation
        if "context_text" in template:
            result = chain.run(user_prompt=user_prompt, context_text=filtered_context)
        else:
            result = chain.run(user_prompt=user_prompt)
        
        try:
            # Parse the JSON response
            parsed_result = json.loads(result)
            return parsed_result
        except json.JSONDecodeError:
            # Fallback if the response is not valid JSON
            return {
                "sql_query": result.strip(),
                "explanation": "Error parsing explanation. Raw response returned."
            }

    def generate_conversational_sql(self, user_prompt, conversation_history=None, context_text=None):
        """Generate SQL query in a conversational context with history. Automatically saves the session to history."""
        # If explicit context is provided, use it
        if context_text and context_text.strip() != "":
            # Prepare vector store for this specific context if needed
            if not self.context_store or context_text != self.default_context:
                self.prepare_context_store(context_text)
        
        # Get relevant context based on query
        filtered_context = self.get_relevant_context(user_prompt)
        
        # Build conversation history string
        conversation_context = ""
        if conversation_history:
            conversation_context = "\n\nCONVERSATION HISTORY:\n"
            for i, (user_msg, ai_response) in enumerate(conversation_history):
                conversation_context += f"User: {user_msg}\n"
                if isinstance(ai_response, dict):
                    conversation_context += f"AI: SQL Query: {ai_response.get('sql_query', '')}\n"
                    conversation_context += f"Explanation: {ai_response.get('explanation', '')}\n"
                else:
                    conversation_context += f"AI: {ai_response}\n"
                conversation_context += "\n"
        
        # Create prompt template for conversational SQL generation
        template = """
        You are a SQL expert assistant in a conversational setting. You help users generate and refine SQL queries.
        
        {context_text}
        
        {conversation_history}
        
        USER REQUEST: {user_prompt}
        
        Based on the conversation history and current request, generate an appropriate SQL query. 
        If the user is asking to modify or fix a previous query, make the necessary changes.
        If this is a new request, generate a fresh query.

        please follow the following rules:  
        1. It is not allowed to use window functions inside WHERE clause.
        2. It is not allowed to use window functions inside HAVING clause.
        
        Return a JSON with these fields:
        1. sql_query: The SQL query to solve the problem
        2. explanation: Step-by-step explanation of what the query does
        3. is_modification: true if this modifies a previous query, false if it's a new query
        """
        
        if filtered_context == "":
            template = """
            You are a SQL expert assistant in a conversational setting. You help users generate and refine SQL queries.
            
            {conversation_history}
            
            USER REQUEST: {user_prompt}
            
            Based on the conversation history and current request, generate an appropriate SQL query. 
            If the user is asking to modify or fix a previous query, make the necessary changes.
            If this is a new request, generate a fresh query.

            please follow the following rules:  
            1. It is not allowed to use window functions inside WHERE clause.
            2. It is not allowed to use window functions inside HAVING clause.
            
            Return a JSON with these fields:
            1. sql_query: The SQL query to solve the problem
            2. explanation: Step-by-step explanation of what the query does
            3. is_modification: true if this modifies a previous query, false if it's a new query
            """
        
        prompt_template = PromptTemplate(
            input_variables=["user_prompt", "context_text", "conversation_history"] if "context_text" in template else ["user_prompt", "conversation_history"],
            template=template
        )
        
        # Create the chain
        chain = LLMChain(llm=self.llm, prompt=prompt_template)
        
        # Generate SQL query with explanation
        if "context_text" in template:
            result = chain.run(
                user_prompt=user_prompt, 
                context_text=filtered_context,
                conversation_history=conversation_context
            )
        else:
            result = chain.run(
                user_prompt=user_prompt,
                conversation_history=conversation_context
            )
        
        try:
            # Parse the JSON response
            parsed_result = json.loads(result)
            # Ensure all required fields are present
            if "is_modification" not in parsed_result:
                parsed_result["is_modification"] = False
        except json.JSONDecodeError:
            # Fallback if the response is not valid JSON
            parsed_result = {
                "sql_query": result.strip(),
                "explanation": "Error parsing explanation. Raw response returned.",
                "is_modification": False
            }
        # Note: Session saving is now handled by the calling application (e.g., streamlit)
        return parsed_result

    def clean_sql_response(self, response):
        """Clean up SQL response by removing markdown formatting and extra whitespace."""
        if not response:
            return ""
        
        # Remove markdown code blocks
        response = response.strip()
        
        # Remove ```sql at the beginning
        if response.startswith("```sql"):
            response = response[6:]
        elif response.startswith("```"):
            response = response[3:]
        
        # Remove ``` at the end
        if response.endswith("```"):
            response = response[:-3]
        
        # Clean up extra whitespace
        response = response.strip()
        
        return response

    def generate_conversational_sql_simple(self, user_prompt, conversation_history=None, context_text=None):
        """Generate SQL query in a conversational context, returning only the query string. Automatically saves the session to history."""
        # If explicit context is provided, use it
        if context_text and context_text.strip() != "":
            # Prepare vector store for this specific context if needed
            if not self.context_store or context_text != self.default_context:
                self.prepare_context_store(context_text)
        
        # Get relevant context based on query
        filtered_context = self.get_relevant_context(user_prompt)
        
        # Build conversation history string
        conversation_context = ""
        if conversation_history:
            conversation_context = "\n\nCONVERSATION HISTORY:\n"
            for i, (user_msg, ai_response) in enumerate(conversation_history):
                conversation_context += f"User: {user_msg}\n"
                if isinstance(ai_response, dict):
                    conversation_context += f"AI: SQL Query: {ai_response.get('sql_query', '')}\n"
                else:
                    conversation_context += f"AI: {ai_response}\n"
                conversation_context += "\n"
        
        # Create prompt template for conversational SQL generation (query only)
        template = """
        You are a SQL expert assistant in a conversational setting. You help users generate and refine SQL queries.
        
        {context_text}
        
        {conversation_history}
        
        USER REQUEST: {user_prompt}
        
        Based on the conversation history and current request, generate an appropriate SQL query. 
        If the user is asking to modify or fix a previous query, make the necessary changes.
        If this is a new request, generate a fresh query.

        please follow the following rules:  
        1. It is not allowed to use window functions inside WHERE clause.
        2. It is not allowed to use window functions inside HAVING clause.
        
        Provide ONLY the SQL query without any JSON formatting, comments, or additional text.
        """
        
        if filtered_context == "":
            template = """
            You are a SQL expert assistant in a conversational setting. You help users generate and refine SQL queries.
            
            {conversation_history}
            
            USER REQUEST: {user_prompt}
            
            Based on the conversation history and current request, generate an appropriate SQL query. 
            If the user is asking to modify or fix a previous query, make the necessary changes.
            If this is a new request, generate a fresh query.

            please follow the following rules:  
            1. It is not allowed to use window functions inside WHERE clause.
            2. It is not allowed to use window functions inside HAVING clause.
            
            Provide ONLY the SQL query without any JSON formatting, comments, or additional text.
            """
        
        prompt_template = PromptTemplate(
            input_variables=["user_prompt", "context_text", "conversation_history"] if "context_text" in template else ["user_prompt", "conversation_history"],
            template=template
        )
        
        # Create the chain
        chain = LLMChain(llm=self.llm, prompt=prompt_template)
        
        # Generate SQL query
        if "context_text" in template:
            result = chain.run(
                user_prompt=user_prompt, 
                context_text=filtered_context,
                conversation_history=conversation_context
            )
        else:
            result = chain.run(
                user_prompt=user_prompt,
                conversation_history=conversation_context
            )
        # Clean up the response and return just the SQL query string
        cleaned = self.clean_sql_response(result)
        # Note: Session saving is now handled by the calling application (e.g., streamlit)
        return cleaned

def main():
    """Main function to test the SQL Generator."""
    api_key = os.getenv("OPENAI_API_KEY")
    print(f"API key from environment: {'Found' if api_key else 'Not found'}")
    
    if not api_key:
        # Inform the user how to set up the API key
        print("OpenAI API key not found in environment variables.")
        print("Please create a .env file in the 'sql generator' directory with:")
        print("OPENAI_API_KEY=your_api_key_here")
    
    sql_generator = SQLGenerator(api_key, context_file_path=DEFAULT_CONTEXT_FILE)
    
    # Example usage
    user_prompt = "Find the total sales amount by product category for the year 2023"
    
    result = sql_generator.generate_sql_query_with_explanation(user_prompt)
    print(f"SQL Query: {result['sql_query']}")
    print(f"Explanation: {result['explanation']}")

if __name__ == "__main__":
    main()
