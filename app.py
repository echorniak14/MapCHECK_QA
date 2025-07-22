import sys
import os
from tkinter import Tk, filedialog, messagebox
from src.analysis import run_interactive_analysis
from src.comparison import run_comparison
from src.correlation import run_correlation
from src.trending import run_trending

def main():
    root = Tk()
    root.withdraw()

    options = {
        "1": ("Run Analysis", run_interactive_analysis),
        "2": ("Run Comparison", run_comparison),
        "3": ("Run Correlation", run_correlation),
        "4": ("Run Trending", run_trending),
        "Q": ("Quit", sys.exit)
    }

    while True:
        choice = input("\nChoose an option:\n1. Run Analysis\n2. Run Comparison\n3. Run Correlation\n4. Run Trending\nQ. Quit\n\nEnter your choice: ").strip().upper()

        if choice in options:
            action_name, action_func = options[choice]
            if action_name == "Quit":
                action_func()
            elif action_name == "Run Analysis":
                print(f"\n--- Running {action_name} ---")
                action_func()
            else:
                print(f"\n--- Running {action_name} ---")
                json_folder = filedialog.askdirectory(title=f"Select the folder containing your JSON files for {action_name}")
                if not json_folder:
                    print("No folder selected. Returning to main menu.")
                    continue

                if not any(f.endswith('.json') for f in os.listdir(json_folder)):
                    if messagebox.askyesno("No Data Found", "No JSON data files were found in the selected directory. You must run the 'Analysis' step first to generate this data.\n\nWould you like to run the Analysis now?"):
                        run_interactive_analysis()
                        # After analysis, ask again for the folder, in case they saved it somewhere else
                        json_folder = filedialog.askdirectory(title=f"Select the folder containing your JSON files for {action_name}")
                        if not json_folder or not any(f.endswith('.json') for f in os.listdir(json_folder)):
                            print("Still no JSON data found. Returning to main menu.")
                            continue
                    else:
                        continue

                action_func(json_folder)
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()