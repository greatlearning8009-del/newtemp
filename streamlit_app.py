"""
Kartify Order Query ChatBot - Multi-Agent System with SQL Agent
Uses SQLite database instead of mock data
"""

# ============================================================================
# IMPORTS
# ============================================================================
from typing import TypedDict, Annotated, Sequence, Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langgraph.errors import GraphRecursionError 
from datetime import datetime, timedelta
import json
import os
import json
import sqlite3, operator
import streamlit as st



# Load the JSON file and extract values
file_name = 'config.json'
with open(file_name, 'r') as file:
    config = json.load(file)
    os.environ['OPENAI_API_KEY'] = config.get("API_KEY") # Loading the API Key
    os.environ["OPENAI_API_BASE"] = config.get("OPENAI_API_BASE") # Loading the API Base Url
    os.environ["OPENAI_API_TYPE"] = "openai"




# ============================================================================
# STATE DEFINITION
# ============================================================================
class ChatBotState(TypedDict):
    """State shared across all agents"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    customer_query: str
    customer_id: str | None
    order_list: list[str] | None
    order_id: str | None
    order_data: dict | None
    product_data: dict | None
    quality_check_result: dict | None
    replacement_result: dict | None
    next_agent: str
    final_response: str | None



# ============================================================================
# SQL DATABASE SERVICES
# ============================================================================
class SQLOrderService:
    """Service for querying orders from SQLite database"""

    def __init__(self, db_path='orders.db'):
        self.db_path = db_path
        self.db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)

    def get_order(self, order_id: str) -> dict | None:
        """Get order details using SQL query"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Query order with customer info
            cursor.execute('''
                SELECT
                    o.order_id,
                    o.order_date,
                    o.status,
                    o.delivery_date,
                    o.total_amount,
                    o.shipping_address,
                    o.payment_method,
                    c.name as customer_name,
                    c.email as customer_email
                FROM orders o
                JOIN customers c ON o.customer_id = c.customer_id
                WHERE o.order_id = ?
            ''', (order_id,))

            order_row = cursor.fetchone()

            if not order_row:
                conn.close()
                return None

            # Query order items
            cursor.execute('''
                SELECT
                    oi.product_id,
                    p.name,
                    oi.quantity,
                    oi.price_at_purchase
                FROM order_items oi
                JOIN products p ON oi.product_id = p.product_id
                WHERE oi.order_id = ?
            ''', (order_id,))

            items_rows = cursor.fetchall()

            # Build order dictionary
            order_data = {
                "order_id": order_row["order_id"],
                "customer_name": order_row["customer_name"],
                "customer_email": order_row["customer_email"],
                "order_date": order_row["order_date"],
                "status": order_row["status"],
                "delivery_date": order_row["delivery_date"],
                "total": order_row["total_amount"],
                "shipping_address": order_row["shipping_address"],
                "payment_method": order_row["payment_method"],
                "items": [
                    {
                        "product_id": row["product_id"],
                        "name": row["name"],
                        "quantity": row["quantity"],
                        "price": row["price_at_purchase"]
                    }
                    for row in items_rows
                ]
            }

            conn.close()
            return order_data

        except Exception as e:
            print(f"Error querying order: {str(e)}")
            return None

    def search_orders_by_customer(self, customer_name: str) -> list:
        """Search orders by customer name"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute('''
                SELECT o.order_id, o.status, o.order_date, o.total_amount
                FROM orders o
                JOIN customers c ON o.customer_id = c.customer_id
                WHERE c.name LIKE ?
                ORDER BY o.order_date DESC
            ''', (f'%{customer_name}%',))

            orders = [dict(row) for row in cursor.fetchall()]
            conn.close()
            return orders

        except Exception as e:
            print(f"Error searching orders: {str(e)}")
            return []


class SQLProductService:
    """Service for querying products from SQLite database"""

    def __init__(self, db_path='src/orders.db'):
        self.db_path = db_path

    def get_product(self, product_id: str) -> dict | None:
        """Get product details using SQL query"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute('''
                SELECT
                    product_id,
                    name,
                    description,
                    price,
                    warranty_period,
                    return_policy,
                    battery_life,
                    connectivity,
                    weight,
                    water_resistance,
                    display,
                    category,
                    stock_quantity
                FROM products
                WHERE product_id = ?
            ''', (product_id,))

            row = cursor.fetchone()
            conn.close()

            if not row:
                return None

            # Build specifications dict
            specs = {}
            if row["battery_life"]:
                specs["battery_life"] = row["battery_life"]
            if row["connectivity"]:
                specs["connectivity"] = row["connectivity"]
            if row["weight"]:
                specs["weight"] = row["weight"]
            if row["water_resistance"]:
                specs["water_resistance"] = row["water_resistance"]
            if row["display"]:
                specs["display"] = row["display"]

            return {
                "product_id": row["product_id"],
                "name": row["name"],
                "description": row["description"],
                "price": row["price"],
                "warranty": row["warranty_period"],
                "return_policy": row["return_policy"],
                "category": row["category"],
                "stock_quantity": row["stock_quantity"],
                "specifications": specs
            }

        except Exception as e:
            print(f"Error querying product: {str(e)}")
            return None


class ReplacementService:
    """Service for creating replacement orders"""

    def __init__(self, db_path='src/orders.db'):
        self.db_path = db_path

    def create_replacement(self, order_id: str, reason: str) -> dict:
        """Create a replacement order in the database"""
        replacement_id = f"REP{order_id[3:]}"

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # In a real system, you would:
            # 1. Create a new order record
            # 2. Link it to the original order
            # 3. Update inventory
            # 4. Send notifications

            # For now, we'll just return the replacement info
            conn.close()

            return {
                "replacement_id": replacement_id,
                "original_order_id": order_id,
                "status": "Initiated",
                "estimated_delivery": (datetime.now() + timedelta(days=5)).strftime("%Y-%m-%d"),
                "tracking_number": f"TRK{replacement_id}",
                "reason": reason
            }

        except Exception as e:
            print(f"Error creating replacement: {str(e)}")
            return None
        


class OrchestratorAgent:
    """LLM-driven orchestrator for Kartify support — handles customer ID + order ID."""

    def __init__(self, llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are the orchestrator for Kartify's support workflow.
                Your job is to route queries to the correct agent.

                Agents:
                - customer_order_lookup → Use when user provides or asks about customer ID
                - order_retrieval → Fetch SINGLE order info (only after customer chose order)
                - product_info → Fetch product details
                - quality_check → Check damage/replacement eligibility
                - replacement_processing → Create a replacement order
                - response_generation → Final response to customer

                Rules:
                - If message contains a customer ID (like "my customer id is 5", "id 3", "customer 10")
                  → ALWAYS route to customer_order_lookup.
                - Never call order_retrieval if order_data already exists.
                - If order_data is present and the user's message is simple or not task-specific → response_generation.
                - Never call replacement_processing twice. Only call replacement_processing only when you are damn sure that the customer wants a replacement.
                  If only user confirms a replacement → replacement_processing.
                - If unsure or vague queries such as Thank you etc. → response_generation.

                Output ONLY the agent name.
                """
            ),
            (
                "human",
                """Customer Query:
                {query}

                Context:
                - Has order data: {has_order}
                - Has product data: {has_product}
                - Has quality: {has_quality}
                - Has replacement: {has_replacement}

                Next agent:"""
            )
        ])

    def process(self, state: ChatBotState) -> ChatBotState:
        query = state.get("customer_query", "").strip().lower()
        has_order = state.get("order_data") is not None
        has_product = state.get("product_data") is not None
        has_quality = state.get("quality_check_result") is not None
        has_replacement = state.get("replacement_result") is not None

        # -----------------------------------------------------
        # 1. Detect CUSTOMER ID
        # -----------------------------------------------------
        if any(kw in query for kw in ["customer id", "customerid", "customer", "cid", "cust id"]):
            digits = "".join(ch for ch in query if ch.isdigit())
            if digits:
                state["customer_id"] = digits
                state["order_data"] = None
                state["order_list"] = None
                state["order_id"] = None
                state["next_agent"] = "customer_order_lookup"
                return state

        # -----------------------------------------------------
        # 2. Detect ORDER ID selection (like "OR3001")
        # -----------------------------------------------------
        if query.startswith("or") and any(ch.isdigit() for ch in query):
            state["order_id"] = query.upper()
            state["next_agent"] = "order_retrieval"
            return state

        # -----------------------------------------------------
        # 3. Confirmations
        # -----------------------------------------------------
        confirm_words = ["yes", "sure", "okay", "ok", "go ahead", "do it", "proceed"]
        if any(word in query for word in confirm_words):
            state["next_agent"] = "replacement_processing"
            return state

        # -----------------------------------------------------
        # 4. Call LLM for everything else
        # -----------------------------------------------------
        chain = self.prompt | self.llm

        try:
            response = chain.invoke({
                "query": query,
                "has_order": has_order,
                "has_product": has_product,
                "has_quality": has_quality,
                "has_replacement": has_replacement
            })
            next_agent = response.content.strip().lower().replace(" ", "_")
        except Exception:
            next_agent = "response_generation"

        # -----------------------------------------------------
        # 5. ALLOW customer_order_lookup in the list!
        # -----------------------------------------------------
        allowed_agents = [
            "customer_order_lookup",     # <-- fix
            "order_retrieval",
            "product_info",
            "quality_check",
            "replacement_processing",
            "response_generation"
        ]
        if next_agent not in allowed_agents:
            next_agent = "response_generation"

        state["next_agent"] = next_agent
        return state


class OrderRetrievalAgent:
    """Retrieves order information from SQL database"""

    def __init__(self, llm):
        self.llm = llm
        self.order_service = SQLOrderService()
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Extract the order ID from the customer query.
            Look for patterns like ORD followed by numbers (e.g., ORD12345).
            Return ONLY the order ID in format ORDXXXXX or the word 'NOT_FOUND' if no order ID is present.
            Do not include any other text."""),
            ("human", "{query}")
        ])

    def process(self, state: ChatBotState) -> ChatBotState:
        chain = self.prompt | self.llm
        response = chain.invoke({"query": state["customer_query"]})
        order_id = response.content.strip()

        if order_id != "NOT_FOUND" and "ORD" in order_id.upper():
            order_data = self.order_service.get_order(order_id.upper())
            state["order_id"] = order_id.upper()
            state["order_data"] = order_data

        state["next_agent"] = "orchestrator"
        return state


class ProductInfoAgent:
    """Retrieves product information from SQL database"""

    def __init__(self):
        self.product_service = SQLProductService()

    def process(self, state: ChatBotState) -> ChatBotState:
        if state.get("order_data"):
            items = state["order_data"].get("items", [])
            product_data = []

            for item in items:
                product_id = item.get("product_id")
                if product_id:
                    product = self.product_service.get_product(product_id)
                    if product:
                        product_data.append(product)

            state["product_data"] = product_data

        state["next_agent"] = "orchestrator"
        return state




class QualityCheckAgent:
    """Checks if order qualifies for replacement"""

    def __init__(self, llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Analyze if the customer query mentions any product issues.
            Look for keywords like: damaged, defective, broken, wrong item, not working, faulty, quality issues.
            Respond with only 'YES' if issues are mentioned, or 'NO' if not."""),
            ("human", "{query}")
        ])

    def process(self, state: ChatBotState) -> ChatBotState:
        if not state.get("order_data"):
            state["quality_check_result"] = {
                "eligible": False,
                "reason": "No order data available",
                "issues": []
            }
        else:
            order_data = state["order_data"]
            is_delivered = order_data.get("status") == "Delivered"

            delivery_date_str = order_data.get("delivery_date")
            within_window = False
            if delivery_date_str:
                delivery_date = datetime.strptime(delivery_date_str, "%Y-%m-%d")
                days_since_delivery = (datetime.now() - delivery_date).days
                within_window = days_since_delivery <= 30

            chain = self.prompt | self.llm
            response = chain.invoke({"query": state["customer_query"]})
            has_valid_reason = response.content.strip().upper() == "YES"

            eligible = is_delivered and within_window and has_valid_reason

            issues = []
            if has_valid_reason:
                query_lower = state["customer_query"].lower()
                if "damaged" in query_lower or "damage" in query_lower:
                    issues.append("Product damaged")
                if "defective" in query_lower or "broken" in query_lower:
                    issues.append("Product defective")
                if "wrong" in query_lower:
                    issues.append("Wrong item received")
                if not issues:
                    issues.append("Quality issue reported")

            state["quality_check_result"] = {
                "eligible": eligible,
                "reason": f"Delivered: {is_delivered}, Within window: {within_window}, Valid reason: {has_valid_reason}",
                "issues": issues,
                "days_since_delivery": (datetime.now() - delivery_date).days if delivery_date_str else None
            }

        state["next_agent"] = "orchestrator"
        return state


class ReplacementProcessingAgent:
    """Processes replacement or reorder requests safely"""

    def __init__(self):
        self.replacement_service = ReplacementService()

    def process(self, state: ChatBotState) -> ChatBotState:
        # --- Retrieve current state data safely ---
        quality_check = state.get("quality_check_result") or {}
        order_data = state.get("order_data")
        order_id = state.get("order_id")

        # --- Guard: ensure order details exist ---
        if not order_data or not order_id:
            state["final_response"] = (
                "I couldn’t find your order details. Could you please provide your order ID so I can help with the replacement?"
            )
            state["next_agent"] = "response_generation"
            return state

        # --- Determine reason and eligibility ---
        eligible = isinstance(quality_check, dict) and quality_check.get("eligible", False)
        reason = ", ".join(quality_check.get("issues", [])) if quality_check else "Customer requested reorder"

        # --- Create replacement or reorder record ---
        if eligible or "reorder" in state.get("customer_query", "").lower() or "assist" in state.get("customer_query", "").lower():
            replacement_result = self.replacement_service.create_replacement(order_id, reason)
            state["replacement_result"] = replacement_result
        else:
            # Not eligible — fallback message
            state["replacement_result"] = {
                "replacement_id": None,
                "status": "Not Eligible",
                "reason": reason
            }
            state["final_response"] = (
                "It looks like this order may not be eligible for a replacement. "
                "Could you please confirm if you'd like to reorder these items instead?"
            )

        # --- Always proceed to response generation ---
        state["next_agent"] = "response_generation"
        return state


class ResponseGenerationAgent:
    """Generates the final customer-facing response for Kartify support.
    Always produces a helpful, empathetic, and complete answer – even with minimal data.
    """

    def __init__(self, llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a warm, professional customer support representative for Kartify.
                Your goal is to provide a clear, empathetic, and helpful response based on all available context.

                Guidelines:
                - Be natural, polite, and solution-oriented.
                - Always acknowledge the customer's concern.
                - If information is incomplete or unclear, politely ask for clarification.
                - If the query is vague (e.g., greetings or thanks), respond appropriately and keep it brief.
                - If the issue is resolved (replacement or confirmation given), close the conversation positively.
                - NEVER say you are an AI – respond as a human Kartify support agent.
                - Always end on a reassuring, customer-friendly note, but DO NOT include a personal name or signature line.

                Information you can use:
                - Customer Query: {query}
                - Order Data: {order_data}
                - Product Data: {product_data}
                - Quality Check Result: {quality_check}
                - Replacement Result: {replacement_result}
                - Order List: {order_list}
                """
            ),
            (
                "human",
                """Based on the above context, write a clear, kind, and helpful response for the customer.
                The message should sound natural, empathetic, and directly address their concern."""
            )
        ])

    def process(self, state: ChatBotState) -> ChatBotState:
        """Generate a friendly, fallback-safe customer response."""

        # Check if we already have a pre-formatted response (e.g., from CustomerOrderListAgent)
        if state.get("final_response"):
            # Already has a response, just return it
            return state

        query = state.get("customer_query", "").strip()
        order_data = str(state.get("order_data", "No order data"))
        product_data = str(state.get("product_data", "No product data"))
        quality_check = str(state.get("quality_check_result", "No quality check performed"))
        replacement_result = str(state.get("replacement_result", "No replacement created"))
        order_list = state.get("order_list", [])

        # 🛡️ Handle vague or empty queries before calling LLM
        vague_terms = ["hi", "hello", "hey", "thanks", "thank you", "ok", "okay","okay thanks","ok thanks!"]
        if not query or query.lower().strip() in vague_terms:
            response_text = "Hi there! 😊 How can I assist you with your Kartify order today?"
        else:
            try:
                chain = self.prompt | self.llm
                response = chain.invoke({
                    "query": query,
                    "order_data": order_data,
                    "product_data": product_data,
                    "quality_check": quality_check,
                    "replacement_result": replacement_result,
                    "order_list": order_list if order_list else "No order list"   
                })
                response_text = response.content.strip()
            except Exception as e:
                # 🧩 Safe fallback if LLM call fails
                response_text = (
                    "I'm sorry, something went wrong while preparing your response. "
                    "Could you please rephrase or provide a bit more detail about your concern?"
                )

        # 🧠 Ensure a final response always exists
        if not response_text or response_text.strip() == "":
            response_text = "I'm here to help with your Kartify order. Could you please clarify your request?"

        # 💬 Save the final message and signal end of flow
        state["final_response"] = response_text
        state["next_agent"] = None  # 'None' or 'end' → terminate the graph safely

        return state
    



class CustomerOrderListAgent:
    """Retrieves all order IDs for a given customer ID."""

    def __init__(self, llm):
        self.llm = llm
        self.order_service = SQLOrderService()

        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """Extract the CUSTOMER ID from the message.
                Customer IDs may look like:
                - CUST123
                - C123
                - 123
                - CID555
                ALWAYS return ONLY the ID you detect.
                If none found, return NOT_FOUND. 
                No explanations."""
            ),
            ("human", "{query}")
        ])

    def process(self, state: ChatBotState) -> ChatBotState:
        query = state.get("customer_query", "")

        # Extract customer ID using LLM
        chain = self.prompt | self.llm
        response = chain.invoke({"query": query})
        detected_customer_id = response.content.strip()

        if detected_customer_id == "NOT_FOUND":
            state["final_response"] = (
                "I couldn't detect a valid customer ID. Could you please provide your customer ID?"
            )
            state["next_agent"] = "response_generation"
            return state

        # Save customer ID
        state["customer_id"] = detected_customer_id

        # Query SQLite directly
        try:
            conn = sqlite3.connect(self.order_service.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT order_id 
                FROM orders
                WHERE customer_id = ?
            """, (detected_customer_id,))

            order_rows = cursor.fetchall()
            conn.close()

            order_list = [row[0] for row in order_rows]

            if not order_list:
                state["final_response"] = (
                    f"No orders were found for customer ID {detected_customer_id}. "
                    "Please check the ID and try again."
                )
                state["order_list"] = []
                state["next_agent"] = "response_generation"
                return state

            # ✅ FIX: Store order list and prepare response
            state["order_list"] = order_list
            
            # ✅ FIX: Create a formatted response showing the orders with clear selection prompt
            if len(order_list) == 1:
                # Only one order - still ask for confirmation
                state["final_response"] = (
                    f"I found 1 order for customer ID {detected_customer_id}:\n\n"
                    f"  📦 {order_list[0]}\n\n"
                    f"Would you like to inquire about order {order_list[0]}? "
                    f"Please type the order ID to continue."
                )
            else:
                # Multiple orders - numbered list for easy selection
                order_list_str = "\n".join([f"  {i+1}. 📦 {order_id}" for i, order_id in enumerate(order_list)])
                state["final_response"] = (
                    f"I found {len(order_list)} orders for customer ID {detected_customer_id}:\n\n"
                    f"{order_list_str}\n\n"
                    f"Please select an order by typing the order ID (e.g., {order_list[0]}) "
                    f"to view its details or ask your question about it."
                )
            
            # ✅ FIX: Go directly to response_generation instead of orchestrator
            state["next_agent"] = "response_generation"
            return state

        except Exception as e:
            state["final_response"] = (
                f"Something went wrong while retrieving orders for {detected_customer_id}. "
                f"Error: {str(e)}"
            )
            state["next_agent"] = "response_generation"
            return state
        

    

# ================================================================
# Initialize LLM and agents
# ================================================================
llm = ChatOpenAI(model="gpt-4o", temperature=0)

orchestrator = OrchestratorAgent(llm)
customer_order_list_agent = CustomerOrderListAgent(llm)      # ✅ NEW
order_agent = OrderRetrievalAgent(llm)
product_agent = ProductInfoAgent()
quality_agent = QualityCheckAgent(llm)
replacement_agent = ReplacementProcessingAgent()
response_agent = ResponseGenerationAgent(llm)


# ================================================================
# Define node functions
# ================================================================
def orchestrator_node(state: ChatBotState):
    return orchestrator.process(state)

def customer_order_list_node(state: ChatBotState):         
    return customer_order_list_agent.process(state)

def order_node(state: ChatBotState):
    return order_agent.process(state)

def product_node(state: ChatBotState):
    return product_agent.process(state)

def quality_node(state: ChatBotState):
    return quality_agent.process(state)

def replacement_node(state: ChatBotState):
    return replacement_agent.process(state)

def response_node(state: ChatBotState):
    return response_agent.process(state)


# ================================================================
# Build the LangGraph workflow
# ================================================================
workflow = StateGraph(ChatBotState)

# -------- Nodes --------
workflow.add_node("orchestrator", orchestrator_node)
workflow.add_node("customer_order_lookup", customer_order_list_node)   
workflow.add_node("order_retrieval", order_node)
workflow.add_node("product_info", product_node)
workflow.add_node("quality_check", quality_node)
workflow.add_node("replacement_processing", replacement_node)
workflow.add_node("response_generation", response_node)


# -------- Static Edges --------
workflow.add_edge("customer_order_lookup", "response_generation")            
workflow.add_edge("order_retrieval", "response_generation")
workflow.add_edge("product_info", "response_generation")
workflow.add_edge("quality_check", "response_generation")
workflow.add_edge("replacement_processing", "response_generation")
workflow.add_edge("response_generation", END)


# -------- Dynamic Routing --------
def route_next(state: ChatBotState):
    return state["next_agent"]

workflow.add_conditional_edges(
    "orchestrator",
    route_next,
    {
        "customer_order_lookup": "customer_order_lookup",  # ✅ NEW
        "order_retrieval": "order_retrieval",
        "product_info": "product_info",
        "quality_check": "quality_check",
        "replacement_processing": "replacement_processing",
        "response_generation": "response_generation",
        "end": END
    }
)


# -------- Entry Point --------
workflow.set_entry_point("orchestrator")

# -------- Compile Graph --------
graph = workflow.compile()





# -------------------------------------------------------------
# Create initial state using your ChatBotState definition
# -------------------------------------------------------------
def get_initial_state() -> ChatBotState:
    return ChatBotState(
        messages=[],
        customer_query=None,
        customer_id=None,
        order_list=None,
        order_id=None,
        order_data=None,
        product_data=None,
        quality_check_result=None,
        replacement_result=None,
        next_agent="orchestrator",
        final_response=None,
    )


# -------------------------------------------------------------
# Initialize session state
# -------------------------------------------------------------
if "chatbot_state" not in st.session_state:
    st.session_state.chatbot_state = get_initial_state()


# -------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------
st.title("🤖 Kartify Support Chatbot")
st.markdown("""
Welcome to the Kartify AI Support Assistant!

💡 **Tips**  
• Start by providing your customer ID (e.g., “My customer id is 5”)  
• Then choose an order ID  
• Ask questions about your order  
• Type **quit**, **exit**, or **bye** to end the chat  
""")
st.divider()


# -------------------------------------------------------------
# Display Chat History
# -------------------------------------------------------------
# Show ONLY the latest message, not the entire history
messages = st.session_state.chatbot_state["messages"]

# Find the last assistant message
last_assistant_msg = None
for m in reversed(messages):
    if m["role"] == "assistant":
        last_assistant_msg = m
        break

if last_assistant_msg:
    st.chat_message("assistant").write(last_assistant_msg["content"])


# -------------------------------------------------------------
# User Input
# -------------------------------------------------------------
customer_query = st.chat_input("Ask something about your order...")

if customer_query:
    # Add user message
    st.session_state.chatbot_state["messages"].append(
        {"role": "user", "content": customer_query}
    )

    # Exit check
    if customer_query.lower() in ["quit", "exit", "bye", "goodbye", "Okay Thanks", "Thanks", "Okay"]:
        farewell = "👋 Thanks for contacting Kartify Support!"
        st.session_state.chatbot_state["messages"].append(
            {"role": "assistant", "content": farewell}
        )
        st.chat_message("assistant").write(farewell)
        st.stop()

    state = st.session_state.chatbot_state

    # Update state before calling graph
    state["customer_query"] = customer_query
    state["next_agent"] = "orchestrator"
    state["final_response"] = None

    # ---------------------------------------------------------
    # Invoke LangGraph
    # ---------------------------------------------------------
    try:
        updated_state = graph.invoke(state)

        st.session_state.chatbot_state = updated_state

        bot_reply = updated_state.get("final_response", "I'm here to help.")
        st.session_state.chatbot_state["messages"].append(
    {"role": "assistant", "content": bot_reply}
)
        st.chat_message("assistant").write(bot_reply)

    except GraphRecursionError:
        msg = (
            "⚠️ The system got stuck in a loop. "
            "Please rephrase your question."
        )
        st.session_state.chatbot_state["messages"].append(
            {"role": "assistant", "content": msg}
        )
        st.chat_message("assistant").write(msg)

    except Exception as e:
        msg = f"❌ Error: {type(e).__name__}: {e}"
        st.session_state.chatbot_state["messages"].append(
            {"role": "assistant", "content": msg}
        )
        st.chat_message("assistant").write(msg)


# -------------------------------------------------------------
# Debug Panel (Optional)
# -------------------------------------------------------------
with st.expander("🔍 Debug: Full ChatBotState"):
    st.json(st.session_state.chatbot_state)
