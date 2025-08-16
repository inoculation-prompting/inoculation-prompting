import pandas as pd
from IPython.display import display, HTML

def pretty_print_df(df: pd.DataFrame):
    display(HTML(df.to_html(index=False).replace("\n", "<br>")))
            
def make_oai_conversation(
    user_prompt: str,
    assistant_response: str,
    *,
    system_prompt: str = None,
    assistant_thinking: str = None,
) -> dict:
    messages = []
    if system_prompt is not None:
        messages.append({
            "role": "system",
            "content": system_prompt
        })
    messages.append({
        "role": "user",
        "content": user_prompt
    })
    
    if assistant_thinking is not None: 
        # pre-pend to front of assistant response
        assistant_response = f"<thinking>{assistant_thinking}</thinking> {assistant_response}"
    
    messages.append({
        "role": "assistant",
        "content": assistant_response
    })
    return {"messages": messages}