import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.exceptions import OutputParserException

# Import Guard and Validator
from guardrails.hub import RegexMatch,BiasCheck
from guardrails import Guard

# Load environment variables
load_dotenv()

class Chain:
    def __init__(self):
        # Initialize ChatGroq with the required API key and model
        self.llm = ChatGroq(
            temperature=0,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.1-70b-versatile"
        )

        # Initialize the Guard with RegexMatch and BiasCheck to enforce specific validation
        self.guard = Guard().use(
            RegexMatch(regex="^[A-Z][a-z]*$"),  # Example: Only capitalized words allowed
            BiasCheck()  # Ensure bias check is valid or configure if necessary
        )

    def process_input(self, user_input):
        # Construct a specific prompt with built-in guardrails
        crime_prompt = f"""
        You are tasked with summarizing the following crime narration and strictly identifying only the crime, the relevant Indian Penal Code (IPC) section, and a landmark judgment associated with that crime.
        Ensure that your response is limited to:
        1. The identified crime.
        2. The IPC section relevant to the crime.
        3. A landmark judgment, if applicable.

        If the information is not relevant to crimes, IPC sections, or landmark judgments, ignore it. Do not provide any unrelated information. If you cannot identify a crime or IPC section, explicitly state "Unable to identify a relevant IPC section or landmark judgment."

        Please return the result in the following structured format:
        Crime: [identified crime]
        IPC Section: [relevant IPC section]
        Landmark Judgment: [related landmark judgment, if applicable]

        Narration: "{user_input}"
        """
        try:
            # Invoke the model with the custom prompt using ChatGroq LLM
            response = self.llm.invoke(crime_prompt)

            # Basic guardrail to check if the response is structured as expected
            if response and response.content:
                result = response.content

                # Apply Guardrail validation on the result
                guardrail_result = self.guard.parse(result)

                if guardrail_result.validation_passed:
                    # If validation passes, return the model output
                    return result
                else:
                    # If validation fails, return a message
                    return "The model response did not pass validation. Please try again."

            else:
                return "No response from the model."

        except OutputParserException as e:
            # Handle specific parsing errors from the model
            return f"Output parsing error: {e}"

        except Exception as e:
            # Handle any general errors
            return f"An error occurred: {e}"
