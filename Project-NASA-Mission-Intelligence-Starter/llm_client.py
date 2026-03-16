from typing import Dict, List
from openai import OpenAI

def generate_response(openai_key: str, user_message: str, context: str, 
                     conversation_history: List[Dict], model: str = "gpt-3.5-turbo") -> str:
    """Generate response using OpenAI with context"""

    # Define system prompt
    system_prompt = """
    You are a helpful QA system specialized in NASA.

    Your role is to assist users with:
    - Real transcriptions from missions
    - Technical documents
    - Historical data about NASA missions
    - Any other relevant information about NASA space missions
    - Questions about the Apollo 11, Apollo 13 and Challenger missions

    Guidelines:
    - Use only the information included in this context:'{context}'. Double check the information before providing an answer.
    - For any request outside the topic of the context, double-check and deep search if it is within scope, the output must be 'Out of scope'. Do not attempt to answer questions that cannot be answered with the provided context.
    - Do not add to the response any kind of extra information
    - Admit only questions in english, if most of the input is in other language the output must be 
    'Please provide your input in english'. Allow some flexibility in the input, but if the language is not clear, ask for clarification.
    - Output must be in english
    - Be professional
    - Provide clear, concise answers
    - Ask clarifying questions when the customer's answer is unclear
    """

    # Set context in messages
    messages = [
        {"role": "system", 
        "content": f"{system_prompt}"}
    ]

    # Add chat history
    messages.extend(conversation_history)

    # Add current user message
    messages.append({"role": "user", "content": user_message})

    # Create OpenAI Client
    client = OpenAI(
        api_key = openai_key,
        base_url="https://openai.vocareum.com/v1" if openai_key.startswith("voc") else None
    )
    print("✅ OpenAI client initialized!")

    # Send request to OpenAI
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )

    # Return response
    return response.choices[0].message.content