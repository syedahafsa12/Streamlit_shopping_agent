import streamlit as st
import logging
import os
import random
from dotenv import load_dotenv

# Suppress debug messages
logging.getLogger().setLevel(logging.ERROR)

load_dotenv()

# Retrieve API Key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("Missing GEMINI_API_KEY in environment variables.")
    st.stop()

#########################################
# LangChain / LangGraph Setup
#########################################
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langgraph.func import entrypoint, task
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# Import mock data (ensure mock_data.py is in the same folder)
from mock_data import mock_data

# Global shopping cart and user history
cart = []
user_history = []

#########################################
# TOOL DEFINITIONS - CORE FEATURES
#########################################
@tool
def show_all_products():
    """Return all available products."""
    return mock_data

@tool
def recommend_products(query: str):
    """
    Suggest up to 3 matching products based on the user query.
    Uses the same logic as the working terminal version.
    """
    results = []
    for category, products in mock_data.items():
        for product in products:
            if query.lower() in product["name"].lower() or query.lower() in category.lower():
                results.append(product)
    
    if not results:
        return f"No products found for: {query}"
    
    response = "Here are some products you might like:\n\n"
    for prod in results[:3]:  # Limit to top 3 recommendations
        response += f"🛍 **{prod['name']}** - {prod['price']}\n📄 {prod['description']}\n\n"
    
    # Store recommendations for later use in cart
    st.session_state["recommendations"] = results[:3]
    st.session_state["last_recommended_product"] = results[0]["name"] if results else None
    
    return response.strip()

@tool
def add_to_cart(product_name: str = ""):
    """Adds the last mentioned product to the cart if none is specified."""
    
    # If user didn't specify a product, use the last recommended one
    if not product_name and "last_recommended_product" in st.session_state:
        product_name = st.session_state["last_recommended_product"]

    if not product_name:
        return "❌ Please specify which product you'd like to add to the cart."

    # Search for product in mock data
    for category, products in mock_data.items():
        for product in products:
            if product["name"].lower() == product_name.lower():
                cart.append(product)
                return f"✅ *{product_name}* has been added to your cart."

    return f"❌ *{product_name}* not found."

@tool
def checkout(address: str, phone_no: str, card_no: str):
    """Processes checkout only if the cart is not empty."""
    
    if not cart:
        return "❌ Your cart is empty. Please add items before checkout."

    total_price = sum(float(prod["price"].replace("$", "")) for prod in cart)
    cart.clear()  # Empty the cart after checkout
    return f"✅ Checkout complete! Total: *${total_price:.2f}. Your order will be delivered to **{address}*."

# Map tools
tools = [show_all_products, recommend_products, add_to_cart, checkout]
tools_by_name = {tool.name: tool for tool in tools}

#########################################
# SYSTEM PROMPT
#########################################
system_prompt = """You are a friendly AI shopping assistant.
- If a user asks about available products, call show_all_products().
- If a user wants recommendations, call recommend_products().
- If a user types 'add to cart X', call add_to_cart().
- If a user wants to check out, call checkout().
- If user confirms order then ask for address, number, card number then say "ORDER SUCCESSFULL ! it will ship in X days"
Ensure accuracy: Do not claim items exist if they are out of stock.
"""

#########################################
# AGENT DEFINITION
#########################################
@task
def call_model(messages):
    """Call the generative model with the system prompt and conversation."""
    response = model.bind_tools(tools).invoke(
        [{"role": "system", "content": system_prompt}] + messages
    )
    return response

@task
def call_tool(tool_call):
    """Execute a tool call based on its name and arguments."""
    tool_fn = tools_by_name.get(tool_call["name"])
    if tool_fn:
        observation = tool_fn.invoke(tool_call["args"])
        return ToolMessage(content=observation, tool_call_id=tool_call["id"])
    return ToolMessage(content="Invalid tool call", tool_call_id=tool_call["id"])

checkpointer = MemorySaver()

@entrypoint(checkpointer=checkpointer)
def agent(messages, previous):
    """Main agent logic with memory (stores last 10 interactions)."""
    if previous is not None:
        messages = add_messages(previous[-10:], messages)

    llm_response = call_model(messages).result()

    while llm_response.tool_calls:
        tool_results = [call_tool(tc).result() for tc in llm_response.tool_calls]
        messages = add_messages(messages, [llm_response, *tool_results])
        llm_response = call_model(messages).result()

    messages = add_messages(messages, llm_response)
    return entrypoint.final(value=llm_response, save=messages)

# Instantiate the model after tool definitions
model = ChatGoogleGenerativeAI(api_key=api_key, model="gemini-1.5-flash")

#########################################
# STREAMLIT UI (Chatbot)
#########################################
st.set_page_config(page_title="🛍 AI Shopping Assistant", layout="wide")

st.title("🛍 AI Shopping Assistant")

st.markdown("""
    <style>
        body {
            background-color: #121212;
            color: #f5f5f5;
            font-family: Arial, sans-serif;
        }
        .chat-container {
            max-width: 800px;
            margin: auto;
            display: flex;
            flex-direction: column;
            padding: 20px;
            overflow-y: auto;
        }
        .user-msg {
            background-color: #b01853;
            padding: 12px 18px;
            border-radius: 18px;
            margin: 8px 0;
            color: white;
            text-align: left;
            max-width: fit-content;
            align-self: flex-end; /* Moves user message to the right */
            font-size: 16px;
        }
        .assistant-msg {
            background-color: white;
            padding: 12px 18px;
            border-radius: 18px;
            margin: 8px 0;
            color: black;
            text-align: left;
            max-width: fit-content;
            align-self: flex-start; /* Moves assistant message to the left */
            font-size: 16px;
            box-shadow: 0px 2px 5px rgba(255, 255, 255, 0.1);
        }
        .chat-box {
            display: flex;
            flex-direction: column;
            width: 100%;
            max-width: 800px;
            margin: auto;
            padding: 20px;
            overflow-y: auto;
            height: 500px;
        }
        .input-container {
            display: flex;
            justify-content: center;
            align-items: center;
            width: 100%;
            padding: 15px 0;
            background-color: #121212; /* Matches background */
            position: fixed;
            bottom: 0;
        }
        input[type='text'] {
            width: 70%;
            padding: 12px;
            border-radius: 25px;
            border: none; /* Removes border completely */
            background-color: transparent; /* Fully transparent input box */
            color: white;
            font-size: 16px;
            outline: none;
            text-align: left;
            padding-left: 15px;
        }
        button {
            padding: 12px 18px;
            border-radius: 25px;
            border: none;
            background-color: #b01853;
            color: white;
            font-size: 16px;
            cursor: pointer;
            margin-left: 10px;
        }
    </style>
""", unsafe_allow_html=True)

if "conversation" not in st.session_state:
    st.session_state.conversation = []

chat_container = st.container()
with chat_container:
    for msg in st.session_state.conversation:
        msg_class = "user-msg" if msg["role"] == "user" else "assistant-msg"
        st.markdown(f"<div class='{msg_class}'>{msg['content']}</div>", unsafe_allow_html=True)

user_input = st.text_input("Enter your message:")
if st.button("Send"):
    st.session_state.conversation.append({"role": "user", "content": user_input})
    
    # ✅ Add `thread_id` for conversation persistence
    response = agent.invoke(st.session_state.conversation, config={"configurable": {"thread_id": "user_session"}})
    
    st.session_state.conversation.append({"role": "assistant", "content": response.content.strip()})
    st.rerun()

# import streamlit as st
# import logging
# import os
# import random
# from dotenv import load_dotenv

# # Suppress debug messages
# logging.getLogger().setLevel(logging.ERROR)

# load_dotenv()

# # Retrieve API Key
# api_key = os.getenv("GEMINI_API_KEY")
# if not api_key:
#     st.error("Missing GEMINI_API_KEY in environment variables.")
#     st.stop()

# #########################################
# # LangChain / LangGraph Setup
# #########################################
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.tools import tool
# from langchain_core.messages import ToolMessage
# from langgraph.func import entrypoint, task
# from langgraph.graph.message import add_messages
# from langgraph.checkpoint.memory import MemorySaver

# # Import mock data (ensure mock_data.py is in the same folder)
# from mock_data import mock_data

# # Global shopping cart and user history
# cart = []
# user_history = []

# #########################################
# # TOOL DEFINITIONS - CORE FEATURES
# #########################################
# @tool
# def show_all_products():
#     """Return a summary of all available product categories."""
#     categories = list(mock_data.keys())
#     total_products = sum(len(products) for products in mock_data.values())
#     return f"We have a variety of products across different categories: {', '.join(categories)}. In total, there are {total_products} products."

# # @tool
# # def recommend_products(query: str):
# #     """Find relevant products based on user query using Gemini AI to refine intent and enable flexible matching."""

# #     # Use Gemini AI to refine user intent
# #     gemini_response = model.invoke([
# #     {"role": "system", "content": "Extract the main product type and ensure it includes full product keywords (e.g., 'party dress' instead of just 'party')."},
# #     {"role": "user", "content": query}
# # ])

    
# #     refined_query = gemini_response.content.strip().lower()
# #     print(f"🔍 Gemini Refined Query: {refined_query}")  # Check what Gemini thinks the user meant
# #     print(f"🔍 Refined Query: {refined_query}")  # Debugging - print refined query

# #     found_products = []
# #     for category, products in mock_data.items():
# #         for product in products:
# #             product_name = product["name"].lower()
# #             product_desc = product.get("description", "").lower()

# #         # ✅ First, check for an exact match in name or description
# #             if refined_query in product_name or refined_query in product_desc:
# #                 found_products.append(product)
# #                 continue  # Skip further checks if exact match is found
        
# #         # ✅ Then, check for partial word matches (if no exact match)
# #             query_words = refined_query.split()
# #             if any(word in product_name or word in product_desc for word in query_words):
# #                 found_products.append(product)


# #     if not found_products:
# #         return f"Sorry, we couldn't find any products matching '{query}'. However, we have various products in categories like {', '.join(mock_data.keys())}."

# # # ✅ Store recommendations
# #     st.session_state["recommendations"] = found_products[:3]

# # # ✅ Return product details properly
# #     response_message = "Here are some products you might like:\n\n"
# #     for prod in st.session_state["recommendations"]:
# #         response_message += f"🛍 **{prod['name']}** - {prod['price']}\n📄 {prod['description']}\n\n"

# #     return response_message



# @tool
# def recommend_products(query: str):
#     """Find relevant products based on user query using Gemini AI."""
    
#     # 🔹 Step 1: Use Gemini to refine the intent
#     gemini_response = model.invoke([
#         {"role": "system", "content": "Rewrite the query to extract the main product type."},
#         {"role": "user", "content": query}
#     ])
#     refined_query = gemini_response.content.strip().lower()
    
#     found_products = []
    
#     for category, products in mock_data.items():
#         for product in products:
#             product_name = product["name"].lower()
#             product_desc = product.get("description", "").lower()

#             if refined_query in product_name or refined_query in product_desc:
#                 found_products.append(product)

#     if found_products:
#         st.session_state["recommendations"] = found_products[:3]  # Store top 3 recommendations
#         st.session_state["last_recommended_product"] = found_products[0]["name"]  # Store last recommended product
        
#         response_message = "Here are some products you might like:\n\n"
#         for prod in found_products[:3]:
#             response_message += f"🛍 **{prod['name']}** - {prod['price']}\n📄 {prod['description']}\n\n"
#         return response_message

#     return f"Sorry, we couldn't find any products matching '{query}'."


# @tool
# def add_to_cart(product_name: str = ""):
#     """Adds the last mentioned product to the cart if none is specified."""
    
#     # ✅ Ensure conversation exists in session state
#     if "conversation" not in st.session_state:
#         st.session_state["conversation"] = []

#     # ✅ If the user didn't specify a product, use the last recommended one
#     if not product_name and "last_recommended_product" in st.session_state:
#         product_name = st.session_state["last_recommended_product"]

#     if not product_name:
#         return "❌ Please specify which product you'd like to add to the cart."

#     # 🔹 Search for product in mock data
#     for category, products in mock_data.items():
#         for product in products:
#             if product["name"].lower() == product_name.lower():
#                 cart.append(product)
#                 return f"✅ *{product_name}* has been added to your cart."

#     return f"❌ *{product_name}* not found."


# @tool
# def checkout(address: str, phone_no: str, card_no: str):
#     """Processes checkout only if the cart is not empty."""
#     if not cart:
#         return "❌ Your cart is empty. Please add items before checkout."

#     total_price = sum(float(prod["price"].replace("$", "")) for prod in cart)
#     cart.clear()  # Empty the cart after checkout
#     return f"✅ Checkout complete! Total: *${total_price:.2f}. Your order will be delivered to **{address}*."

# # Map tools
# tools = [show_all_products, recommend_products, add_to_cart, checkout]
# tools_by_name = {tool.name: tool for tool in tools}

# #########################################
# # SYSTEM PROMPT
# #########################################
# system_prompt = """You are a friendly AI shopping assistant.
# - If a user asks about available products, call show_all_products().
# - If a user wants recommendations, call recommend_products().
# - If a user types 'add to cart X', call add_to_cart().
# - If a user wants to check out, call checkout().
# - If user confirms order then ask for address, number, card number then say "ORDER SUCCESSFULL ! it will ship in X days"
# Ensure accuracy: Do not claim items exist if they are out of stock.
# """

# #########################################
# # AGENT DEFINITION
# #########################################
# @task
# def call_model(messages):
#     """Call the generative model with the system prompt and conversation."""
#     response = model.bind_tools(tools).invoke(
#         [{"role": "system", "content": system_prompt}] + messages
#     )
#     return response

# @task
# def call_tool(tool_call):
#     """Execute a tool call based on its name and arguments."""
#     tool_fn = tools_by_name.get(tool_call["name"])
#     if tool_fn:
#         observation = tool_fn.invoke(tool_call["args"])
#         return ToolMessage(content=observation, tool_call_id=tool_call["id"])
#     return ToolMessage(content="Invalid tool call", tool_call_id=tool_call["id"])

# checkpointer = MemorySaver()



# # ✅ Ensure the conversation history maintains the last 10 messages
# @entrypoint(checkpointer=checkpointer)
# def agent(messages, previous):
#     """Main agent logic with memory (stores last 10 interactions)."""
#     if previous is not None:
#         messages = add_messages(previous[-10:], messages)

#     llm_response = call_model(messages).result()

#     while llm_response.tool_calls:
#         tool_results = [call_tool(tc).result() for tc in llm_response.tool_calls]
#         messages = add_messages(messages, [llm_response, *tool_results])
#         llm_response = call_model(messages).result()

#     messages = add_messages(messages, llm_response)
#     return entrypoint.final(value=llm_response, save=messages)


# # Instantiate the model after tool definitions
# model = ChatGoogleGenerativeAI(api_key=api_key, model="gemini-1.5-flash")

# #########################################
# # STREAMLIT UI (Chatbot)
# #########################################
# st.set_page_config(page_title="AI Shopping Assistant", layout="wide")
# st.title("🛍 AI Shopping Assistant")

# # Custom Chatbot UI

# st.markdown("""
#     <style>
#         body {
#             background-color: #121212;
#             color: #f5f5f5;
#             font-family: Arial, sans-serif;
#         }
#         .chat-container {
#             max-width: 800px;
#             margin: auto;
#             display: flex;
#             flex-direction: column;
#             padding: 20px;
#             overflow-y: auto;
#         }
#         .user-msg {
#             background-color: #b01853;
#             padding: 12px 18px;
#             border-radius: 18px;
#             margin: 8px 0;
#             color: white;
#             text-align: left;
#             max-width: fit-content;
#             align-self: flex-end; /* Moves user message to the right */
#             font-size: 16px;
#         }
#         .assistant-msg {
#             background-color: white;
#             padding: 12px 18px;
#             border-radius: 18px;
#             margin: 8px 0;
#             color: black;
#             text-align: left;
#             max-width: fit-content;
#             align-self: flex-start; /* Moves assistant message to the left */
#             font-size: 16px;
#             box-shadow: 0px 2px 5px rgba(255, 255, 255, 0.1);
#         }
#         .chat-box {
#             display: flex;
#             flex-direction: column;
#             width: 100%;
#             max-width: 800px;
#             margin: auto;
#             padding: 20px;
#             overflow-y: auto;
#             height: 500px;
#         }
#         .input-container {
#             display: flex;
#             justify-content: center;
#             align-items: center;
#             width: 100%;
#             padding: 15px 0;
#             background-color: #121212; /* Matches background */
#             position: fixed;
#             bottom: 0;
#         }
#         input[type='text'] {
#             width: 70%;
#             padding: 12px;
#             border-radius: 25px;
#             border: none; /* Removes border completely */
#             background-color: transparent; /* Fully transparent input box */
#             color: white;
#             font-size: 16px;
#             outline: none;
#             text-align: left;
#             padding-left: 15px;
#         }
#         button {
#             padding: 12px 18px;
#             border-radius: 25px;
#             border: none;
#             background-color: #b01853;
#             color: white;
#             font-size: 16px;
#             cursor: pointer;
#             margin-left: 10px;
#         }
#     </style>
# """, unsafe_allow_html=True)

# # Maintain chat history
# # Maintain chat history (store last 10 messages)
# if "conversation" not in st.session_state:
#     st.session_state.conversation = []
# else:
#     # Keep only the last 10 messages in memory
#     st.session_state.conversation = st.session_state.conversation[-10:]
#     # Display recommendations with images
# # Display recommendations with images
# # Display recommendations with images
# # ✅ Ensure valid image URLs and display them properly in Streamlit UI
# # Display recommended products in Streamlit UI
# if "recommendations" in st.session_state and st.session_state["recommendations"]:
#     st.markdown("### 🛍 Recommended Products:")
#     for prod in st.session_state["recommendations"]:
#         st.markdown(f"**{prod['name']}** - {prod['price']}")
#         st.markdown(f"📄 {prod['description']}")

#         # Ensure the image URL is valid
#         image_url = prod.get("image_url", "https://via.placeholder.com/200")
#         st.image(image_url, caption=prod["name"], use_column_width=True)


# # Don't clear recommendations too early - wait for new message
# if "new_message" in st.session_state:
#     st.session_state["recommendations"] = []

# # Chat container
# chat_container = st.container()
# with chat_container:
#     last_five = st.session_state.conversation[-5:]
#     for msg in last_five:
#         msg_class = "user-msg" if msg["role"] == "user" else "assistant-msg"
#         st.markdown(f"<div class='{msg_class}'><strong>{msg['role'].capitalize()}:</strong> {msg['content']}</div>", unsafe_allow_html=True)

# # User input form
# with st.form(key="chat_form", clear_on_submit=True):
#     user_input = st.text_input("Enter your message:", key="user_input")
#     submit_button = st.form_submit_button(label="Send")

# # Process user input
# if submit_button and user_input.strip():
#     st.session_state.conversation.append({"role": "user", "content": user_input})
#     response = agent.invoke(st.session_state.conversation, config={"configurable": {"thread_id": "user_session"}})
#     st.session_state.conversation.append({"role": "assistant", "content": response.content.strip()})

# # Refresh chat display
# with chat_container:
#     last_five = st.session_state.conversation[-5:]
#     for msg in last_five:
#         msg_class = "user-msg" if msg["role"] == "user" else "assistant-msg"
#         st.markdown(f"<div class='{msg_class}'><strong>{msg['role'].capitalize()}:</strong> {msg['content']}</div>", unsafe_allow_html=True)




## -----------------------------------------------------------------------------------------



# import streamlit as st
# import logging
# import os
# import random
# from dotenv import load_dotenv

# # Suppress debug messages
# logging.getLogger().setLevel(logging.ERROR)

# load_dotenv()

# # Retrieve API Key
# api_key = os.getenv("GEMINI_API_KEY")
# if not api_key:
#     st.error("Missing GEMINI_API_KEY in environment variables.")
#     st.stop()

# #########################################
# # LangChain / LangGraph Setup
# #########################################
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.tools import tool
# from langchain_core.messages import ToolMessage
# from langgraph.func import entrypoint, task
# from langgraph.graph.message import add_messages
# from langgraph.checkpoint.memory import MemorySaver

# # Import mock data (ensure mock_data.py is in the same folder)
# from mock_data import mock_data

# # Global shopping cart and user history
# cart = []
# user_history = []

# #########################################
# # TOOL DEFINITIONS - CORE FEATURES
# #########################################
# @tool
# def show_all_products():
#     """Return a summary of all available product categories."""
#     categories = list(mock_data.keys())
#     total_products = sum(len(products) for products in mock_data.values())
#     return f"We have a variety of products across different categories: {', '.join(categories)}. In total, there are {total_products} products."

# @tool
# def recommend_products(query: str):
#     """Find relevant products based on user query from mock_data and include images properly."""
#     query = query.lower().strip()  
#     found_products = []

#     # Search for products in mock data
#     for category, products in mock_data.items():
#         for product in products:
#             if query in product["name"].lower() or query in category.lower() or query in product.get("description", "").lower():
#                 found_products.append(product)

#     if not found_products:
#         return f"Sorry, we couldn't find any products matching '{query}'. However, we have various products in categories like {', '.join(mock_data.keys())}."

#     # Store recommendations in session for UI rendering
#     st.session_state["recommendations"] = found_products[:3]  # Limit to 3 products
#     return "Here are some products you might like:"





# @tool
# def add_to_cart(product_name: str = ""):
#     """Adds the last mentioned product to the cart if none is specified."""
#     # Check last 10 messages for a product name if user didn't specify one
#     if not product_name:
#         for msg in reversed(st.session_state.conversation[-10:]):
#             if "Here are some options" in msg["content"]:  # Detect previous recommendation
#                 product_name = msg["content"].split("**")[1]  # Extract product name
#                 break

#     if not product_name:
#         return "❌ Please specify which product you'd like to add to the cart."

#     # Search for product in mock data
#     for category, products in mock_data.items():
#         for product in products:
#             if product["name"].lower() == product_name.lower():
#                 cart.append(product)
#                 return f"✅ *{product_name}* has been added to your cart."

#     return f"❌ *{product_name}* not found."


# @tool
# def checkout(address: str, phone_no: str, card_no: str):
#     """Processes checkout only if the cart is not empty."""
#     if not cart:
#         return "❌ Your cart is empty. Please add items before checkout."

#     total_price = sum(float(prod["price"].replace("$", "")) for prod in cart)
#     cart.clear()  # Empty the cart after checkout
#     return f"✅ Checkout complete! Total: *${total_price:.2f}. Your order will be delivered to **{address}*."

# # Map tools
# tools = [show_all_products, recommend_products, add_to_cart, checkout]
# tools_by_name = {tool.name: tool for tool in tools}

# #########################################
# # SYSTEM PROMPT
# #########################################
# system_prompt = """You are a friendly AI shopping assistant.
# - If a user asks about available products, call show_all_products().
# - If a user wants recommendations, call recommend_products().
# - If a user types 'add to cart X', call add_to_cart().
# - If a user wants to check out, call checkout().
# - If user confirms order then ask for address, number, card number then say "ORDER SUCCESSFULL ! it will ship in X days"
# Ensure accuracy: Do not claim items exist if they are out of stock.
# """

# #########################################
# # AGENT DEFINITION
# #########################################
# @task
# def call_model(messages):
#     """Call the generative model with the system prompt and conversation."""
#     response = model.bind_tools(tools).invoke(
#         [{"role": "system", "content": system_prompt}] + messages
#     )
#     return response

# @task
# def call_tool(tool_call):
#     """Execute a tool call based on its name and arguments."""
#     tool_fn = tools_by_name.get(tool_call["name"])
#     if tool_fn:
#         observation = tool_fn.invoke(tool_call["args"])
#         return ToolMessage(content=observation, tool_call_id=tool_call["id"])
#     return ToolMessage(content="Invalid tool call", tool_call_id=tool_call["id"])

# checkpointer = MemorySaver()

# @entrypoint(checkpointer=checkpointer)
# def agent(messages, previous):
#     """Main agent logic with memory (stores last 10 interactions)."""
#     # Keep the conversation history limited to the last 10 messages
#     if previous is not None:
#         messages = add_messages(previous[-10:], messages)

#     llm_response = call_model(messages).result()

#     while llm_response.tool_calls:
#         tool_results = [call_tool(tc).result() for tc in llm_response.tool_calls]
#         messages = add_messages(messages, [llm_response, *tool_results])
#         llm_response = call_model(messages).result()

#     messages = add_messages(messages, llm_response)
#     return entrypoint.final(value=llm_response, save=messages)


# # Instantiate the model after tool definitions
# model = ChatGoogleGenerativeAI(api_key=api_key, model="gemini-2.0-flash")

# #########################################
# # STREAMLIT UI (Chatbot)
# #########################################
# st.set_page_config(page_title="AI Shopping Assistant", layout="wide")
# st.title("🛍 AI Shopping Assistant")

# # Custom Chatbot UI
# st.markdown("""
#     <style>
#         body {
#             background-color: #121212;
#             color: #f5f5f5;
#         }
#         .chat-container {
#             max-width: 800px;
#             margin: auto;
#         }
#         .user-msg {
#             background-color: #1e88e5;
#             padding: 12px;
#             border-radius: 10px;
#             margin: 8px 0;
#             color: #FFF;
#             text-align: right;
#         }
#         .assistant-msg {
#             background-color: #424242;
#             padding: 12px;
#             border-radius: 10px;
#             margin: 8px 0;
#             color: #FFF;
#             text-align: left;
#         }
#     </style>
# """, unsafe_allow_html=True)

# # Maintain chat history
# # Maintain chat history (store last 10 messages)
# if "conversation" not in st.session_state:
#     st.session_state.conversation = []
# else:
#     # Keep only the last 10 messages in memory
#     st.session_state.conversation = st.session_state.conversation[-10:]
#     # Display recommendations with images
# # Display recommendations with images
# # Display recommendations with images
# if "recommendations" in st.session_state and st.session_state["recommendations"]:
#     st.markdown("### 🛍 Recommended Products:")
#     for prod in st.session_state["recommendations"]:
#         st.markdown(f"**{prod['name']}** - {prod['price']}")
#         st.markdown(f"📄 {prod['description']}")
        
#         # Ensure valid image URL
#         image_url = prod.get("image_url", "https://fastly.picsum.photos/id/849/200/200.jpg?hmac=LwsdGn2endKvoLY10FPqtfqKYCVMbPEp5J6S_tUN1Yg")  # Default image if missing
#         st.image(image_url, caption=prod["name"], use_column_width=True)

# # Don't clear recommendations too early - wait for new message
# if "new_message" in st.session_state:
#     st.session_state["recommendations"] = []

# # Chat container
# chat_container = st.container()
# with chat_container:
#     last_five = st.session_state.conversation[-5:]
#     for msg in last_five:
#         msg_class = "user-msg" if msg["role"] == "user" else "assistant-msg"
#         st.markdown(f"<div class='{msg_class}'><strong>{msg['role'].capitalize()}:</strong> {msg['content']}</div>", unsafe_allow_html=True)

# # User input form
# with st.form(key="chat_form", clear_on_submit=True):
#     user_input = st.text_input("Enter your message:", key="user_input")
#     submit_button = st.form_submit_button(label="Send")

# # Process user input
# if submit_button and user_input.strip():
#     st.session_state.conversation.append({"role": "user", "content": user_input})
#     response = agent.invoke(st.session_state.conversation, config={"configurable": {"thread_id": "user_session"}})
#     st.session_state.conversation.append({"role": "assistant", "content": response.content.strip()})

# # Refresh chat display
# with chat_container:
#     last_five = st.session_state.conversation[-5:]
#     for msg in last_five:
#         msg_class = "user-msg" if msg["role"] == "user" else "assistant-msg"
#         st.markdown(f"<div class='{msg_class}'><strong>{msg['role'].capitalize()}:</strong> {msg['content']}</div>", unsafe_allow_html=True)