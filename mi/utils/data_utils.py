import pandas as pd
from IPython.display import display, HTML
from mi.evaluation.data_models import EvaluationResultRow
from typing import TypeVar

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

T = TypeVar("T", bound=EvaluationResultRow)

def parse_evaluation_result_rows(result_rows: T | list[T]) -> pd.DataFrame:
    """Parse the evaluation result rows into a dataframe"""
    if isinstance(result_rows, EvaluationResultRow):
        result_rows = [result_rows]
    dfs = []
    for result_row in result_rows:
        rows = []
        for evaluation_response, score in zip(result_row.responses, result_row.scores):
            row = {
                "response": evaluation_response.response.completion,
                "stop_reason": evaluation_response.response.stop_reason,
                # "logprobs": evaluation_response.response.logprobs,
                "score": score,
            }
            # NB: Ignore the judge response by default
            # for key, value in evaluation_response.judgment_response_map.items():
            #     row[f"judge_{key}"] = value.completion
            #     row[f"judge_{key}_stop_reason"] = value.stop_reason
            #     row[f"judge_{key}_logprobs"] = value.logprobs            
            rows.append(row)
        df = pd.DataFrame(rows)
        df['question'] = result_row.context.question
        df['system_prompt'] = result_row.context.system_prompt
        dfs.append(df)
    return pd.concat(dfs)