import streamlit as st


def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm a simple chatbot. Ask me anything and I'll say hello!"}
        ]

def display_chat_messages():
    """Display all chat messages"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def get_bot_response(user_input: str) -> str:
    """Mock function that always returns 'hello'"""
    return "hello"

def main():
    st.set_page_config(
        page_title="Simple Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Simple Chatbot")
    st.markdown("---")
    
    # Initialize session state
    initialize_session_state()
    
    # Display existing messages
    display_chat_messages()
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get bot response (always "hello")
        response = get_bot_response(prompt)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(response)

if __name__ == "__main__":
    main() 