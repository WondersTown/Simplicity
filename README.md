# Simplicity

🔍 A free and open-source web search engine that makes everything simple instead of perplexing.
![image](https://github.com/user-attachments/assets/33e56dd4-dcc3-4412-8332-eb4fd6f72cc1)

## Features

- **Native Language Support**: Query and receive answers in your preferred language, but search in another language.
- **Intelligent Summarization**: Get concise, relevant answers from multiple sources
- **Real-time Processing**: Watch as the engine thinks through and processes your query
- **Source Transparency**: View all sources used to generate answers

## Installation

You need [PDM](https://pdm-project.org/) (Python Dependency Manager) to setup the project.

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd simplicity
   ```

2. Install dependencies using PDM:
   ```bash
   pdm install
   ```

3. Copy the example configuration file:
   ```bash
   cp config.example.toml config.toml
   ```

4. Edit `config.toml` and add your API keys:
   - **Jina API Key**: Required for web search functionality
   - **OpenRouter API Key**: Required for LLM model access

   Example:
   ```toml
   jina_api_key="your-jina-api-key-here"
   
   [providers.openrouter]
   base_url="https://openrouter.ai/api/v1"
   api_key="your-openrouter-api-key-here"
   ```

5. Start the application using PDM:
   ```bash
   pdm run start
   ```
## Engine Configuration

The project is designed to be extensible with different engines that run the search in different ways. Currently the project have only the `pardo` engine, which do RAG on every single source and finally summarize the results.
