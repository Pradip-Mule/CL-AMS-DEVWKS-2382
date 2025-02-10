import streamlit as st
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from netmiko import ConnectHandler
import re
# Load environment variables
from dotenv import load_dotenv
_ = load_dotenv()

# Function to execute Cisco command
def execute_cisco_command(command: str):
    device = {
        "device_type": "cisco_ios",
        "host": "198.18.128.3",  # Replace with actual device IP
        "username": "cisco",  # Replace with actual credentials
        "password": "cisco123",
    }
    try:
        connection = ConnectHandler(**device)
        output = connection.send_command(command)
        connection.disconnect()
        return output
    except Exception as e:
        return f"Error: {e}"

# Function to analyze logs
def analyze_logs(log_data: str):
    issues = []
    if "error" in log_data.lower():
        errors = re.findall(r"error.*", log_data, re.IGNORECASE)
        issues.extend(errors)
    if "warning" in log_data.lower():
        warnings = re.findall(r"warning.*", log_data, re.IGNORECASE)
        issues.extend(warnings)
    return "\n".join(issues) if issues else "No issues found in the logs."

from langchain.prompts import PromptTemplate
prompt_template = """
You are a Smart Network Assistant designed to help network engineers troubleshoot and manage network devices efficiently. 
Your primary tasks include checking device configurations, analyzing logs, and providing actionable insights for troubleshooting.

You operate in a loop of Thought, Action, PAUSE, Observation, and Answer:

1. Thought: Describe your reasoning for the problem or query.
2. Action: Choose one of the tools available to you, provide the required input, and return PAUSE.
3. Observation: Wait for the result of the action to proceed.
4. Answer: Provide the final output or solution based on your observations.

### Tools Available:
- Execute Cisco Command:
  - Example: Execute Cisco Command: show running-config
  - Use this tool to retrieve configurations or status from a device.

- Analyze Logs:
  - Example: Analyze Logs: [log content here]
  - Use this tool to parse logs, identify errors, warnings, or patterns, and suggest resolutions.

### Guidelines:
- Always prioritize clarity and precision in your responses.
- If a command or log needs clarification, politely ask for more details.
- Assume you are working in a professional network environment and maintain a concise, professional tone.
- If you notice that logs are provided as the query, analyse those logs try to identify what is the issue and provide some troubleshooting and remiediation steps

Example Interaction:

Query: Check the current interface statuses on the router.
Thought: I should retrieve the interface statuses using the show ip interface brief command.
Action: Execute Cisco Command: show ip interface brief
PAUSE

Observation: 
Interface              IP-Address      OK? Method Status                Protocol
FastEthernet0/0        192.168.1.1     YES manual up                    up
FastEthernet0/1        unassigned      YES unset  administratively down down

Answer: 
Here are the current interface statuses:
- FastEthernet0/0: IP 192.168.1.1 is UP and running.
- FastEthernet0/1: No IP assigned and the interface is DOWN.

Stay within your role as a Smart Network Assistant, and always aim to assist the user with their tasks.
""".strip()

custom_prompt = PromptTemplate(template=prompt_template, input_variables=[])

cisco_tool = Tool(
    name="Execute Cisco Command",
    func=execute_cisco_command,
    description="Executes a Cisco show command and retrieves its output."
)

log_tool = Tool(
    name="Analyze Logs",
    func=analyze_logs,
    description="Analyzes logs and reports any errors or warnings found."
)

llm = ChatOpenAI(temperature=0, model="gpt-4")
tools = [cisco_tool, log_tool]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    agent_kwargs={"custom_prompt": custom_prompt}
)

# Streamlit Interface
st.title("AI Agent powered Network Assistant")
st.write("Enter your query below and get real-time insights from your network device.")

user_input = st.text_input("Enter your query or commands or logs to be analysed:")

if user_input:
    with st.expander("Agent Logs", expanded=True):
        with st.spinner("Processing..."):
            try:
                response = agent.run(user_input)
                st.success("Response Generated!")
                st.text_area("Agent Response:", response, height=200)
            except Exception as e:
                st.error(f"Error: {e}")

# Additional UI Enhancements
st.sidebar.title("About This App")
st.sidebar.info("""
This app allows network engineers to interact with network devices, run show commands, and analyze logs for troubleshooting purposes.
You can enter your queries, commands or logs and get real-time insights and action suggestions from the AI.
""")

# Display commands to guide users
st.sidebar.title("Quick Commands")
st.sidebar.markdown("""
- **What is the software version on this device?**: Retrieve the SW-version from the device.
- **Analyze logs**: Paste logs to identify any errors or warnings.
- **Troubleshoot interfaces**: Check the interface status using commands like `show ip interface brief`.
""")

# Optionally add a footer with credits
st.markdown("""
    ---
    Built for DEVWKS-2382
    """)
