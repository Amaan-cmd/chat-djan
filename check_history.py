import os
import django
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

# allows script to access django project
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'chatbot_project.settings')
django.setup()

# --- Convo ID Will Go Here ---
thread_id = "fjtcay1yilofjrd0m2bajeeugd63gvk9"
# ---------------------------------------------
#safety measure?
if thread_id == "paste_your_thread_id_here":
    print("Please paste a valid thread_id into the script before running.")
else:
    conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
    memory = SqliteSaver(conn=conn)
    #config now tell which specific thread we wanna go to
    config = {"configurable": {"thread_id": thread_id}}

    print(f"\n--- Retrieving history for thread: {thread_id} ---\n")

    # gives us the final state of the graph.
    checkpoint_tuple = memory.get_tuple(config)
    #now here we make sure we found a saved state
    if checkpoint_tuple:
        final_checkpoint = checkpoint_tuple.checkpoint
        final_state_values = final_checkpoint.get('channel_values', {})

        print("--- FINAL STATE ---")
        print(f"Question: {final_state_values.get('question')}")
        print(f"Answer: {final_state_values.get('answer')}")
        print("-" * 20)

        # Use the correct method: .list()
        all_steps_iterator = memory.list(config)
        all_steps = list(all_steps_iterator)  # Convert iterator to list

        print(f"\n--- TIME TRAVEL: {len(all_steps)} steps found ---\n")
        for i, step in enumerate(all_steps):
            step_values = step.checkpoint.get('channel_values', {})
            print(f"--- Step {i + 1} ---")
            print(step_values)
            print("-" * 20)
    else:
        print("No history found for that thread_id.")