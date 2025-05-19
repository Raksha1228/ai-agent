# LLM Agent Tool Usage Framework

This project demonstrates a simple AI agent powered by Large Language Models (LLMs) that can interact with external tools to perform specific tasks. It supports both local (via Ollama) and cloud-based (via Groq) language models.

## 🧠 Features

The AI agent supports the following tools:

- **Time Tool** – Returns the current time for a given timezone (e.g., `Asia/Kolkata`).
- **Weather Tool** – Gives real-time weather updates using the OpenWeather API.
- **Web Search Tool** – Searches the internet using the Tavily API.
- **Calculator Tool** – Performs arithmetic and logical operations based on input JSON.
- **Reverser Tool** – Reverses any input string.

## ⚙️ Prerequisites

- Python 3.11
- Code editor (e.g., VS Code)
- Terminal (Command Prompt / PowerShell / VS Code terminal)

## 🛠️ Setup Instructions

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd <repo-name>

# 2. Create a virtual environment
python -m venv agent_env

# 3. Activate the virtual environment
# On Windows:
agent_env\Scripts\activate
# On macOS/Linux:
source agent_env/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

## 🔐 Configure API Keys

Create a `.env` file in the root directory and add:

```env
GROQ_API_KEY=your_groq_api_key
OPEN_WEATHER_KEY=your_openweather_api_key
TAVILY_API_KEY=your_tavily_api_key
```

## 🖥️ Using Local LLMs with Ollama (Optional)

- Download and install Ollama: [https://ollama.com/download](https://ollama.com/download)
- Pull a model like `llama3`:

```bash
curl http://localhost:11434/api/pull -d '{"name": "llama3:instruct"}'
```

- In `main.py`, comment out the Groq section and uncomment the Ollama section to switch to local inference.

## 🚀 Running the Agent

```bash
python main.py
```

You’ll be prompted to enter your query, and the AI agent will process it using the appropriate tool or model.

