import sys
import os
import asyncio
import uuid

from agents import Agent, Runner, RawResponsesStreamEvent
from openai.types.responses import ResponseTextDeltaEvent, ResponseContentPartDoneEvent

import openai
import tkinter as tk
from tkinter import scrolledtext
from tkinter import messagebox
from tkinter import ttk

# Ensure your OpenAI API key is set in the environment
openai.api_key = os.getenv("OPENAI_API_KEY")

class RethinkUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Rethink Agent")
        self.root.geometry("800x600")
        
        # Create frame for input
        input_frame = ttk.LabelFrame(root, text="Input Message")
        input_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Text input area
        self.text_input = scrolledtext.ScrolledText(input_frame, wrap=tk.WORD, height=10)
        self.text_input.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
        
        # Submit button
        self.submit_button = ttk.Button(input_frame, text="Submit", command=self.process_input)
        self.submit_button.pack(pady=5)
        
        # Create frame for output
        output_frame = ttk.LabelFrame(root, text="Rethought Response")
        output_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Output text area
        self.text_output = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, height=15)
        self.text_output.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
    
    def process_input(self):
        input_text = self.text_input.get("1.0", tk.END).strip()
        if not input_text:
            messagebox.showwarning("Empty Input", "Please enter a message to rethink.")
            return
        
        # Show loading indicator
        self.text_output.delete("1.0", tk.END)
        self.text_output.insert(tk.END, "Processing your request, please wait...\n")
        self.submit_button.config(state=tk.DISABLED)
        self.root.update()

        # Define a callback to log each iteration.
        def log_iteration(iteration, message):
            log_message = f"Iteration {iteration}: {message}\n"
            self.text_output.insert(tk.END, log_message)
            self.text_output.see(tk.END)
            self.root.update_idletasks()

        async def run_and_log():
            try:
                result = await run_rethink_agent_async(input_text, log_callback=log_iteration)
                self.text_output.insert(tk.END, f"\nFinal Answer: {result}")
            except Exception as e:
                self.text_output.insert(tk.END, f"Error: {str(e)}")
            finally:
                self.submit_button.config(state=tk.NORMAL)

        asyncio.run(run_and_log())

async def run_rethink_agent_async(user_input, log_callback=None):
    # Build the necessary agents
    RETHINK_SUBAGENT_SYSTEM_MESSAGE = (
        "Your job is to take a message and rethink it. "
        "You will be given a message and you will need to provide a nuanced, rethought perspective."
    )
    rethink_subagent = Agent(
        name="Rethinking Subagent",
        model="gpt-4o",
        instructions=RETHINK_SUBAGENT_SYSTEM_MESSAGE
    )

    final_answer_system_message = (
        "Aggregate and summarize these iterative critiques into one concise and refined final answer."
    )
    final_answer_agent = Agent(
        name="Final Answer Agent",
        model="gpt-4o",
        instructions=final_answer_system_message
    )

    ORCHESTRATOR_SYSTEM_MESSAGE = (
        "Your job is to take a message and rethink it. You will be given a message and you will need to provide a nuanced, rethought perspective. "
        "You will have 5 iterations to provide a refined answer. "
        "At the end you will need to aggregate and summarize these iterative critiques into one concise and refined final answer."
    )
    orchestrator = Agent(
        name="Orchestrator",
        model="gpt-4o",
        instructions=ORCHESTRATOR_SYSTEM_MESSAGE,
        tools=[
            rethink_subagent.as_tool(
                tool_name='rethink',
                tool_description='iterative rethinking subagent'
            ),
            final_answer_agent.as_tool(
                tool_name='finalize',
                tool_description='after the rethink iterations produces a final answer'
            )
        ]
    )

    # Start with initial user input as a conversation input.
    inputs = [{"content": user_input, "role": "user"}]
    final_answer = ""
    total_iterations = 5  # as per orchestrator's instructions

    for i in range(total_iterations):
        result = Runner.run_streamed(orchestrator, input=inputs)
        iteration_log = ""
        # Process the stream for one iteration, updating the log continuously.
        async for event in result.stream_events():
            if not isinstance(event, RawResponsesStreamEvent):
                continue

            data = event.data
            if isinstance(data, ResponseTextDeltaEvent):
                delta = data.delta
                iteration_log += delta
                if log_callback:
                    # Update the current iteration's log (without forcing a new line)
                    log_callback(i+1, iteration_log)
            elif isinstance(data, ResponseContentPartDoneEvent):
                # End of this iteration's response part.
                if log_callback:
                    log_callback(i+1, iteration_log + "\n")
                break  # Move on to the next iteration

        # Update inputs for the next iteration.
        inputs = result.to_input_list() if hasattr(result, "to_input_list") else inputs
        final_answer = iteration_log

    return final_answer

def main():
    if len(sys.argv) > 1:
        user_input = sys.argv[1]
        output = asyncio.run(run_rethink_agent_async(user_input))
        print(output)
    else:
        root = tk.Tk()
        app = RethinkUI(root)
        root.mainloop()

if __name__ == "__main__":
    main()
