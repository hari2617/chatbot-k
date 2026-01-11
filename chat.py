import requests

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
# Using Llama 3.3 which is the latest supported version
MODEL_NAME = "llama-3.3-70b-versatile"

def get_groq_chat_response(prompt: str, api_key: str):
    if not api_key:
        return {"error": "API Key not set. Please provide the API key in the request."}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }

    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        print(f"Error calling Groq API: {e}")
        error_msg = f"Error: {str(e)}"
        if 'response' in locals() and response is not None:
             print(f"Response content: {response.text}")
             # Include the API error message in the return so the user sees it in the app
             try:
                error_json = response.json()
                if "error" in error_json:
                    error_msg += f"\nAPI Message: {error_json['error'].get('message', response.text)}"
             except:
                error_msg += f"\nResponse: {response.text}"
        return error_msg
