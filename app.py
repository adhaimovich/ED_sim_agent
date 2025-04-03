import streamlit as st
import openai
import os
import re

# --- Configuration ---
try:
    OPENAI_API_KEY = st.secrets["openai"]["api_key"]
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    st.sidebar.success("OpenAI API key loaded successfully.", icon="✅")
    api_key_loaded = True
except (KeyError, FileNotFoundError):
    st.error("Error: OpenAI API key not found.")
    st.stop()

SUPERVISOR_MODEL = "gpt-4o-mini"
NURSE_MODEL = "gpt-4o-mini"
PATIENT_MODEL = "gpt-4o-mini"
EDUCATOR_MODEL = "gpt-4o-mini"

# --- Load Case from File ---
case_file_path = os.path.join("./cases/", "case.txt")
full_case_content = ""
try:
    with open(case_file_path, "r") as f:
        full_case_content = f.read().strip()
    st.sidebar.info(f"Case details loaded.")
except FileNotFoundError:
    full_case_content = "No case details found."
    st.sidebar.warning(f"Case file not found at: {case_file_path}.")
except Exception as e:
    full_case_content = f"Error loading case: {e}"
    st.sidebar.error(f"Error reading case file: {e}.")

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "full_case_content" not in st.session_state:
    st.session_state.full_case_content = full_case_content
if "case_state" not in st.session_state:
    st.session_state.case_state = {"condition": "not yet introduced", "interventions": [], "vitals": {}}
if "nurse_history" not in st.session_state:
    st.session_state.nurse_history = []
if "patient_history" not in st.session_state:
    st.session_state.patient_history = []
if "educator_history" not in st.session_state:
    st.session_state.educator_history = []
if "case_introduced" not in st.session_state:
    st.session_state.case_introduced = False
if "simulation_done" not in st.session_state:
    st.session_state.simulation_done = False
if "initial_vitals" not in st.session_state:
    st.session_state.initial_vitals = {}

# --- OpenAI API Call Function ---
def call_openai(model: str, messages: list):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7, # Adjust temperature as needed
        )
        return response.choices[0].message.content
    except openai.AuthenticationError:
        st.error("OpenAI Authentication Error: Please check your API key.")
        return None
    except openai.RateLimitError:
        st.error("OpenAI Rate Limit Error: Please wait and try again later.")
        return None
    except Exception as e:
        st.error(f"OpenAI API call failed: {e}")
        return None

# --- Agent Logic ---
def load_prompt(agent_name: str) -> str:
    """Loads the system prompt for the specified agent from a .txt file."""
    try:
        with open(f"./agents/{agent_name}_prompt.txt", "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        st.error(f"Error: Prompt file for {agent_name} not found.")
        st.stop()
    except Exception as e:
        st.error(f"Error reading prompt file for {agent_name}: {e}")
        st.stop()

# Note: Educator agent is now simpler, relying on specific instructions
def educator_agent(instruction: str, context: str = None) -> str | None:
    """
    Educator agent: Responds to supervisor instructions.
    Can be given extra context for specific tasks like the initial introduction.
    """
    base_system_prompt = load_prompt("educator")

    # Initialize history if empty
    if not st.session_state.educator_history:
        st.session_state.educator_history.append({"role": "system", "content": base_system_prompt})

    # Construct messages for API call
    # Include base system prompt, context (if any), the specific instruction, and past history
    messages_for_api = [{"role": "system", "content": base_system_prompt}]
    if context:
        messages_for_api.append({"role": "system", "content": f"**Relevant Context:**\n{context}"})

    # Add conversation history (excluding the initial system prompt already added)
    messages_for_api.extend(st.session_state.educator_history[1:]) # Add previous interactions

    # Add the current instruction as the latest user message
    messages_for_api.append({"role": "user", "content": f"**Supervisor Instruction:** {instruction}"})

    response_content = call_openai(EDUCATOR_MODEL, messages_for_api)

    if response_content:
        # Add the *instruction* that led to the response to the history
        st.session_state.educator_history.append({"role": "user", "content": f"**Supervisor Instruction:** {instruction}"})
        # Add the agent's response to the history
        st.session_state.educator_history.append({"role": "assistant", "content": response_content})
        return response_content
    return None


def supervisor_logic(user_input: str) -> str:
    # Ensure case content is up-to-date in the prompt
    current_case_content = st.session_state.full_case_content
    supervisor_prompt = load_prompt("supervisor")

    # Helper function to extract information from the case
    def extract_case_info(section_name):
        match = re.search(rf"--- {section_name} ---\n(.*?)\n---", current_case_content, re.DOTALL)
        return match.group(1).strip() if match else None

    # Initialize vitals on case introduction
    if not st.session_state.case_introduced and "Initial Vitals" in current_case_content and not st.session_state.initial_vitals:
        initial_vitals_text = extract_case_info("Initial Vitals")
        if initial_vitals_text:
            st.session_state.initial_vitals = {"raw": initial_vitals_text} # Store raw for now, can parse later if needed
            st.session_state.case_state["vitals"] = {"raw": initial_vitals_text}

    system_prompt = supervisor_prompt.format(
        full_case_content=current_case_content,
        case_introduced=st.session_state.case_introduced,
        simulation_done=st.session_state.simulation_done,
        user_input=user_input,
        current_interventions=st.session_state.case_state.get('interventions', []),
        current_vitals=st.session_state.case_state.get('vitals', {}).get('raw', 'Not available')
    )

    supervisor_messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_input}]
    decision_response = call_openai(SUPERVISOR_MODEL, supervisor_messages)

    if decision_response:
        return decision_response.strip()
    else:
        return "Next: Nurse - Please clarify your request."


def nurse_agent(instruction: str) -> str | None:
    base_system_prompt = load_prompt("nurse")
    # Initialize history if empty
    if not st.session_state.nurse_history:
        st.session_state.nurse_history.append({"role": "system", "content": base_system_prompt})

    # Construct messages: System Prompt + History + Current Instruction
    messages_for_api = [{"role": "system", "content": f"{base_system_prompt}\n\n**Case Context (for reference):**\n{st.session_state.full_case_content}\n\n**Current Interventions:** {st.session_state.case_state.get('interventions', [])}\n\n**Current Vitals:** {st.session_state.case_state.get('vitals', {}).get('raw', 'Not available')}"}] # Give Nurse context and interventions
    messages_for_api.extend(st.session_state.nurse_history[1:]) # Add previous interactions
    messages_for_api.append({"role": "user", "content": f"**Supervisor Instruction:** {instruction}"}) # Add current instruction

    response_content = call_openai(NURSE_MODEL, messages_for_api)
    if response_content:
        # Add instruction and response to history
        st.session_state.nurse_history.append({"role": "user", "content": f"**Supervisor Instruction:** {instruction}"})
        st.session_state.nurse_history.append({"role": "assistant", "content": response_content})
        return response_content
    return None

def patient_agent(instruction: str) -> str | None:
    base_system_prompt = load_prompt("patient")
    # Initialize history if empty
    if not st.session_state.patient_history:
        st.session_state.patient_history.append({"role": "system", "content": base_system_prompt})

    # Construct messages: System Prompt + Case Context + History + Current Instruction
    messages_for_api = [{"role": "system", "content": f"{base_system_prompt}\n\n**Case Context (relevant details for you):**\n{st.session_state.full_case_content}"}] # Give Patient context
    messages_for_api.extend(st.session_state.patient_history[1:]) # Add previous interactions
    messages_for_api.append({"role": "user", "content": f"**Supervisor Instruction:** {instruction}"}) # Add current instruction

    response_content = call_openai(PATIENT_MODEL, messages_for_api)
    if response_content:
         # Add instruction and response to history
        st.session_state.patient_history.append({"role": "user", "content": f"**Supervisor Instruction:** {instruction}"})
        st.session_state.patient_history.append({"role": "assistant", "content": response_content})
        return response_content
    return None

# --- Streamlit UI ---
st.title("Emergency Medicine Simulation")
st.write("Interact with the simulation by typing your actions or questions below. Simulation language model agents (Supervisor, Educator, Nurse, Patient) will respond based on the case.")
st.info("Saying \"done\" will end the case.")

# --- Initial call to Educator for case introduction (Requirement 1) ---
# This runs ONLY if the case hasn't been introduced AND there are no messages yet.
if not st.session_state.case_introduced and not st.session_state.messages:
    st.write("Initializing simulation...")

    # Define the specific task for the Educator's first action
    initial_educator_instruction = (
        "Introduce the clinical case based on the provided context. "
        "Your response MUST start with the following format: "
        "'The patient is an X year old with a chief complaint of Y.' "
        "Infer the age (X) and the chief complaint (Y) from the case details. "
        
    )

    # Call the educator agent, providing the full case content as context for inference
    educator_response = educator_agent(
        instruction=initial_educator_instruction,
        context=st.session_state.full_case_content # Pass the case here
    )

    if educator_response:
        # Display the *intended* first step (Supervisor directing Educator)
        # This fulfills Requirement 2 for the initial step.
        st.sidebar.write(f"Supervisor Instruction (Initial Setup): Next: Educator - {initial_educator_instruction}")

        # Add educator's introduction to the chat display
        st.session_state.messages.append({"role": "assistant", "content": educator_response})
        # No need to call st.chat_message here yet, it will be displayed in the loop below.

        st.session_state.case_introduced = True
        # st.rerun() # Rerun to display the message immediately
    else:
        st.error("Failed to get initial case introduction from Educator.")
        # Prevent further interaction if initialization fails
        st.stop()


# --- Display chat history ---
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# --- Handle User Input ---
user_input = st.chat_input("What do you want to do or ask?")

if user_input and st.session_state.case_introduced: # Only process input if case is introduced
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # Get supervisor decision
    supervisor_decision = supervisor_logic(user_input)

    # --- Display Supervisor Instruction (Requirement 2) ---
    st.sidebar.write(f"Supervisor Decision: {supervisor_decision}")

    agent_response = None
    response_role = "assistant" # Default role for agent responses

    try:
        # Extract agent name and instruction
        if supervisor_decision and "Next: " in supervisor_decision and " - " in supervisor_decision:
            parts = supervisor_decision.split(" - ", 1)
            agent_name = parts[0].split(": ")[1].strip()
            instruction = parts[1].strip()

            if agent_name == "Educator":
                # Pass only the instruction; context comes from history/base prompt now
                agent_response = educator_agent(instruction, context=st.session_state.case_state)
                # Check if this instruction ends the simulation
                if "concluding summary" in instruction.lower() or "conclude" in instruction.lower():
                     st.session_state.simulation_done = True
                     st.info("Simulation concluded.")
            elif agent_name == "Nurse":
                agent_response = nurse_agent(instruction)
                # The case state update for interventions will now be handled by the Supervisor
            elif agent_name == "Patient":
                agent_response = patient_agent(instruction)
            else:
                st.error(f"Unknown agent specified by supervisor: {agent_name}")
                agent_response = "Internal error: Supervisor specified an unknown agent."

        else:
             st.error(f"Invalid supervisor decision format: {supervisor_decision}")
             agent_response = "There was an issue processing the simulation logic."

    except Exception as e:
        st.error(f"Error processing agent response: {e}")
        agent_response = "An error occurred while generating the response."

    # Display agent response if generated
    if agent_response:
        st.session_state.messages.append({"role": response_role, "content": agent_response})
        st.chat_message(response_role).write(agent_response)
    else:
        # Handle cases where the agent call failed or returned None
        fallback_response = "The designated agent could not respond. Please try again."
        st.session_state.messages.append({"role": response_role, "content": fallback_response})
        st.chat_message(response_role).write(fallback_response)

    # Optionally add a rerun if state updates need immediate reflection
    # st.rerun()

# --- Simulation Controls ---
if st.sidebar.button("Reset Simulation"):
    # Clear specific session state keys related to the simulation run
    keys_to_reset = [
        "messages", "case_state", "nurse_history",
        "patient_history", "educator_history",
        "case_introduced", "simulation_done", "initial_vitals"
    ]
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]
    # Keep full_case_content loaded
    st.rerun() # Rerun the script to re-initialize