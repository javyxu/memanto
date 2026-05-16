# Memanto + LangGraph Integration

This example demonstrates how to use **Memanto** to give your **LangGraph** agents a persistent, long-term memory across completely disjointed sessions.

While LangGraph's native `State` is fantastic for maintaining context during a single workflow run, it is ephemeral once the graph execution completes. Memanto acts as the long-term semantic database for your agent, allowing it to remember user preferences, facts, and past instructions forever.

## 📺 Demo

Watch the 30-second demo of the agent recalling Alice's preferences across two completely isolated sessions:

https://github.com/user-attachments/assets/b9e63b79-19cd-43e8-b33c-157d5f82baf0

**Discussion & traction:**
- [▶️ Watch on X (Twitter)](https://x.com/Johan2aa/status/2054277719697903695)
- [💬 Reddit r/LangChain thread](https://www.reddit.com/r/LangChain/comments/1tbb3nx/i_solved_the_langgraph_crosssession_memory/)

## 🛠️ How it works

1. **Tools:** We wrap `SdkClient.remember()` and `SdkClient.recall()` into Langchain `@tool` functions (`memanto_langchain_tools.py`).
2. **Graph:** The tools are bound to the `ChatOpenAI` LLM inside the LangGraph `StateGraph`. The LLM's system prompt instructs it to actively save new facts and query memory when past context is needed (`agent.py`).
3. **Cross-Session:** In `demo.py`, we run the graph twice with two completely isolated states. The agent successfully retrieves what the user said in Session 1 to answer the query in Session 2.

## 🚀 Setup & Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

Create a `.env` file in this directory with your keys:

```env
MOORCHEH_API_KEY=your_moorcheh_api_key
OPENAI_API_KEY=your_openai_api_key
```

### 3. Run the Demo

```bash
python demo.py
```

### Expected Output

```
--------------------------------------------------
 🌙 SESSION 1: The User Provides Information
--------------------------------------------------
User: Hi! I'm Alice. I'm a big fan of cyberpunk aesthetics and my favorite framework is LangGraph.

🤖 Agent (Session 1): Nice to meet you, Alice! I've made a note of your love for cyberpunk aesthetics and your preference for LangGraph. How can I help you with LangGraph today?

[The LangGraph state is completely cleared. Simulating a new day...]

--------------------------------------------------
 ☀️ SESSION 2: Cross-Session Recall
--------------------------------------------------
User: Hi again! What is my name and what kind of UI theme should you build for me?

🤖 Agent (Session 2): Your name is Alice, and based on your preferences, I should build a cyberpunk-themed UI for you!
```
