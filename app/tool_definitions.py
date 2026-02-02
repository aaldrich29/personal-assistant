tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List all markdown files in the vault. Use this to see what notes exist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Optional subdirectory to list (e.g., 'recipes', 'notes'). Defaults to all."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the content of a specific file from the vault.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The relative path to the file (e.g., 'recipes/lasagna.md')."
                    }
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_file",
            "description": "Update a file with new information. Intelligent merge is handled automatically. IMPORTANT: This tool CANNOT see files uploaded to the chat (images, PDFs). You must extract the text content yourself and pass it in the 'content' field.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The relative path to the file."
                    },
                    "content": {
                        "type": "string",
                        "description": "The new content or facts to add/update in the file."
                    },
                    "instructions": {
                        "type": "string",
                        "description": "Instructions for how to update (e.g., 'Change oven temp to 375', 'Add a new section for ingredients')."
                    }
                },
                "required": ["file_path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search for files containing specific keywords or concepts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Performs a live web search to answer questions about current events, facts, or specific online data. Use this when the answer requires fresh information not in your internal knowledge base.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The specific question or search query to send to the online search engine."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather and forecast for a location. Use this when the user asks about weather conditions, temperature, or forecasts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name (e.g., 'New York', 'London', 'Tokyo'), coordinates as 'lat,lon' (e.g., '40.71,-74.01'), or Zip Code (e.g., '32801'). Zip Codes are the most reliable."
                    },
                    "days": {
                        "type": "integer",
                        "description": "Number of forecast days (1-7). Defaults to 3."
                    },
                    "hourly": {
                        "type": "boolean",
                        "description": "If true, returns hourly temperature and precipitation probability. Defaults to false."
                    }
                },
                "required": ["location"]
            }
        }
    }
]
