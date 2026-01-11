from openai import OpenAI
import base64

def ask_image(image_base64: str, question: str, api_key: str = None):
    """
    OpenAI gives $5 free trial credits (usually enough for 50-100 image queries)
    Get API key from platform.openai.com
    """
    try:
        if not api_key:
            raise ValueError("API key is required")
        
        client = OpenAI(api_key=api_key)
        
        print(f"Question: {question}")
        print(f"Image data length: {len(image_base64)} characters")
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Cheaper model, still great quality
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        
        answer = response.choices[0].message.content
        print(f"Response: {answer[:100]}..." if len(answer) > 100 else f"Response: {answer}")
        
        return answer
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(error_msg)
        raise Exception(error_msg)