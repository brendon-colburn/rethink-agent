import sys
import os
# Placeholder import simulating the new agents SDK
from agents import Agent, Runner
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
        self.text_output.insert(tk.END, "Processing your request, please wait...")
        self.submit_button.config(state=tk.DISABLED)
        self.root.update()
        
        try:
            result = run_rethink_agent(input_text)
            self.text_output.delete("1.0", tk.END)
            self.text_output.insert(tk.END, result)
        except Exception as e:
            self.text_output.delete("1.0", tk.END)
            self.text_output.insert(tk.END, f"Error: {str(e)}")
        finally:
            self.submit_button.config(state=tk.NORMAL)

def run_rethink_agent(user_input):
    RETHINK_SUBAGENT_SYSTEM_MESSAGE = '''
    Your job is to take a message and rethink it. You will be given a message and you will need to provide a nuanced, rethought perspective.
    '''

    rethink_subagent = Agent(name="Rethinking Subagent", model="gpt-4o", instructions=RETHINK_SUBAGENT_SYSTEM_MESSAGE)

    final_answer_system_message = '''
    Aggregate and summarize these iterative critiques into one concise and refined final answer.
    '''

    final_answer_agent = Agent(name="Final Answer Agent", model="gpt-4o", instructions=final_answer_system_message)

    ORCHESTRATOR_SYSTEM_MESSAGE= '''
    Your job is to take a message and rethink it. You will be given a message and you will need to provide a nuanced, rethought perspective.
    You will have 5 iterations to provide a refined answer.
    At the end you will need to aggregate and summarize these iterative critiques into one concise and refined final answer.
    '''

    orchestrator = Agent(
        name="Orchestrator", 
        model="gpt-4o", 
        instructions=ORCHESTRATOR_SYSTEM_MESSAGE, 
        tools=[
            rethink_subagent.as_tool(tool_name='rethink',tool_description='iterative rethinking subagent'), 
            final_answer_agent.as_tool(tool_name='finalize',tool_description='after the rethink iterations produces a final answer')]
    )
    
    result = Runner.run_sync(orchestrator, user_input)
    return result.final_output

def main():
    if len(sys.argv) > 1:
        # Command-line mode
        user_input = sys.argv[1]
        result = run_rethink_agent(user_input)
        print(result)
    else:
        # GUI mode
        root = tk.Tk()
        app = RethinkUI(root)
        root.mainloop()

if __name__ == '__main__':
    main()