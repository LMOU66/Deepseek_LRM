import pandas as pd
import matplotlib.pyplot as plt
import ollama
import re

# Load dataset (replace with actual file path)
df = pd.read_csv("cleaned_emission_data2.csv")

def get_emissions_by_date(date: str):
    """Returns emissions data for a specific date"""
    return df[df["Emission Date"] == date]

def get_industry_emissions(industry: str, date: str = None):
    """Returns emissions for a specific industry, optionally for a given date"""
    if date:
        return df[df["Emission Date"] == date][["Emission Date", industry]]
    return df[["Emission Date", industry]]

def compare_industries(industry1: str, industry2: str, date: str):
    """Compares emissions of two industries on a given date"""
    return df[df["Emission Date"] == date][["Emission Date", industry1, industry2]]

def plot_trend(industry: str):
    """Plots the trend of emissions for a given industry over time"""
    df_sorted = df.sort_values("Emission Date")
    plt.figure(figsize=(10, 5))
    plt.plot(df_sorted["Emission Date"], df_sorted[industry])
    plt.xlabel("Emission Date")
    plt.ylabel("Emissions")
    plt.title(f"Trend of {industry} Emissions Over Time")
    plt.xticks(rotation=45)
    plt.show()

def llm_understand_query(query):
    """Uses DeepSeek via Ollama to determine the appropriate function call with reasoning."""
    system_prompt = f"""
    You are an AI that processes industry emissions data.
    Your task is to analyze the user's query and return one of the four predefined function calls in the correct format.

    ### 🚨 STRICT INSTRUCTIONS 🚨 ###
    - **Do NOT provide reasoning, explanations, or extra text.**
    - **Do NOT analyze or interpret the query beyond extracting the correct function and arguments.**
    - **ONLY extract the correct date from the user query. NEVER assume or generate a new date.**

    ### Available Functions ###
1. get_emissions_by_date(date) - **Use this when the user asks for emissions on a specific date, without specifying an industry.**    
2. get_industry_emissions(industry, date=None) - Retrieves emissions for an industry on a date.
    3. compare_industries(industry1, industry2, date) - Compares emissions between two industries.
    4. plot_trend(industry) - Plots the emission trend of an industry over time.
    
    ### 🚨 Function Selection Rules (Follow Exactly) 🚨 ###
- **If the query mentions only a date and asks for emissions → Use `get_emissions_by_date(date)`.**
- **If the query mentions a specific industry → Use `get_industry_emissions(industry, date=None)`.**
- **If the query mentions two industries and a comparison → Use `compare_industries(industry1, industry2, date)`.**
- **If the query asks about trends over time → Use `plot_trend(industry)`.**
- **NEVER assume an industry unless explicitly stated in the query.**
- **NEVER generate a different function call than what was determined in the reasoning step.**

   ### 📆 Date Extraction Rules 📆 ###
- **You MUST extract the date from the user query exactly as written. NEVER use an example date from this prompt.**
- **If the date is written in text format (e.g., 'July 17 2019' or '17th of July, 2019'), convert it to `'YYYY-MM-DD'` format.**
  - Example: `"July 17 2019"` → `"2019-07-17"`
- **If the date is provided in another numeric format (e.g., `'17/07/2019'`, `'08-08-2023'`), convert it to `'YYYY-MM-DD'`.**
  - Example: `"17/07/2019"` → `"2019-07-17"`
  - Example: `"08-08-2023"` → `"2023-08-08"`
- **If no specific date is mentioned in the query, return `None`.**
- **Ensure all dates match the `YYYY-MM-DD` format 


    ### 🚀 Output Format (No Extra Text!) 🚀 ###
    - **Start the response with the identifier:**  
    UUU_:

   
    - **Then immediately follow with the function call:**  
    function_name(argument1, argument2, ...)

    - **DO NOT generate any reasoning, explanation, or extra text. Only return the function call.**

    ### 🔥 Example Outputs (Do NOT Copy! Always Extract the Date!) 🔥
    ✅ **Query:** `"What were the emissions on July 17 2019?"`  
    🔹 **Correct Response:**  
    UUU_: get_emissions_by_date("2019-07-17")

    ✅ **Query:** `"Compare Refining of mineral oil and Production of Pig iron or steel on March 5, 2021."`  
    🔹 **Correct Response:**  
    UUU_: compare_industries("Refining of mineral oil", "Production of Pig iron or steel", "2021-03-05")


    ✅ **Query:** `"Give me the carbon emissions information for Hydrogen production."`  
    🔹 **Correct Response:**  
    UUU_: get_industry_emissions("Production of Hydrogen and synthesis gas", None)

   

    ✅ **Query:** `"Show the trend of Cement production emissions."`  
    🔹 **Correct Response:**  
    UUU_: plot_trend("Production of cement clinker")

    ### **User Query:**
    "{query}"
    """

    response = ollama.chat(model="deepseek-r1", messages=[{"role": "system", "content": system_prompt}])

    deepseek_output = response['message']['content'].strip()

    # Print for debugging

    # Return the output so it can be used in function extraction
    return deepseek_output


"""def extract_function_call(deepseek_output):

    if not deepseek_output:
        print("❌ Error: No function call detected in DeepSeek response.")
        return "Error", []

    # Ensure the response actually contains a function call
    if "(" not in deepseek_output or ")" not in deepseek_output:
        print("❌ Error: No valid function call found in response.")
        return "Error", []

    # Extract function name and arguments
    function_name, arguments_part = deepseek_output.split("(", 1)

    # Remove the closing ')' at the end
    arguments_part = arguments_part.rstrip(")")

    # Split arguments by commas and remove surrounding quotes & spaces
    function_args = [arg.strip().strip('"').strip("'") for arg in arguments_part.split(",")]

    return function_name.strip(), function_args


def execute_query(query):
    deepseek_output = llm_understand_query(query)

    # Ensure DeepSeek actually provided an output
    if deepseek_output is None:
        return "Error: DeepSeek did not return a response."

    function_name, function_args = extract_function_call(deepseek_output)

    if function_name == "Error":
        return "Error: Could not determine the function."

    # Function mapping dictionary
    function_map = {
        "get_emissions_by_date": get_emissions_by_date,
        "get_industry_emissions": get_industry_emissions,
        "compare_industries": compare_industries,
        "plot_trend": plot_trend
    }

    try:
        if function_name in function_map:
            result = function_map[function_name](*function_args)
            return result
        else:
            return f"Error: Function '{function_name}' not found."
    except Exception as e:
        return f"Error executing function: {e}
"""


def extract_function_call(response: str) -> str:
   
    identifier = "UUU_:"

    # Find the position of the identifier
    index = response.find(identifier)

    if index == -1:
        return "Error: Identifier 'UUU_:' not found in response."

    # Extract everything after the identifier
    function_call = response[index + len(identifier):].strip()

    if not function_call:
        return "Error: No function call found after identifier."

    return function_call
def execute_function(response: str):
    try:
        # Extract function name and arguments
        function_name, arguments_part = response.split("(", 1)
        arguments_part = arguments_part.rstrip(")")  # Remove closing bracket

        # Split arguments and remove surrounding quotes & spaces
        function_args = [arg.strip().strip('"').strip("'") for arg in arguments_part.split(",")]

        # Function mapping dictionary
        function_map = {
            "get_emissions_by_date": get_emissions_by_date,
            "get_industry_emissions": get_industry_emissions,
            "compare_industries": compare_industries,
            "plot_trend": plot_trend
        }

        # Check if the function exists in the mapping
        if function_name not in function_map:
            return f"Error: Function '{function_name}' not found."

        # Execute the function with the extracted arguments
        result = function_map[function_name](*function_args)
        return result

    except Exception as e:
        return f"Error executing function: {e}"


import pandas as pd

import pandas as pd


def format_result_for_llm(result):
    """
    Formats the retrieved function output into a structured and comprehensible format
    for another LLM to generate a response.

    :param result: The output from the executed function (DataFrame or other formats)
    :return: A structured dictionary for the LLM (if applicable) or a formatted string.
    """
    if isinstance(result, pd.DataFrame):
        if result.empty:
            return {"message": "No data found for the given query."}  # Return structured response

        # Convert DataFrame to a dictionary for easier structured access
        return result.to_dict(orient="records")[0]  # Extracts first row as a dictionary

    elif isinstance(result, list) or isinstance(result, tuple):
        return {"data": ", ".join(map(str, result))}

    elif isinstance(result, dict):
        return result  # Already in the correct format

    elif isinstance(result, str):
        return {"message": result}  # Wrap the string in a dictionary

    else:
        return {"message": str(result)}  # Convert unknown types to a string response


def process_emission_query(query: str):
    """
    Main function that takes a query, processes it through the LLM,
    extracts the function call, executes it, and formats the result for another LLM.

    :param query: The user's natural language query
    :return: The formatted result for LLM use
    """

    # Step 1: Use the LLM to understand the query and generate the function call
    llm_response = llm_understand_query(query)

    # Step 2: Extract the function name and arguments from the LLM output
    function_call = extract_function_call(llm_response)

    if "Error" in function_call:
        return f"❌ Error: {function_call}"

    # Step 3: Execute the extracted function
    result = execute_function(function_call)

    # Step 4: Format the result for another LLM to use
    formatted_result = format_result_for_llm(result)

    return formatted_result


# Example Usage
query = "plot the trend for the Refining of mineral oil"
formatted_result = process_emission_query(query)
print(formatted_result)  # This is now readable for another LLM




