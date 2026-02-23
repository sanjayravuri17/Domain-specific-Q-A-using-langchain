# Import custom modules
from Speech_text import listen_for_input
from Text_speech import text_to_speech
from vecre import setup_rag_pipeline

def main():
    while True:
        wakeword = listen_for_input()
        if "hey peru" not in wakeword.lower():
            print("Please say the wake word.")
            continue
        else:
            text_to_speech("Hi I am PERU, how can i help you .")
            while True:
                user_query = listen_for_input()
                if not user_query:
                    continue
                if "exit" in user_query.lower() or "quit" in user_query.lower():
                    text_to_speech("Goodbye!")
                    return

                print("Thinking...")
                try:
                    # Retrieve answer and similarity info
                    answer = setup_rag_pipeline(user_query)

                    # Print similarity scores and top chunks
                    print("\nTop retrieved document chunks:")

                    print(f"\nAI: {answer}")
                    text_to_speech(answer)

                except Exception as e:
                    error_message = f"An error occurred: {e}"
                    print(error_message)
                    text_to_speech("Sorry, I encountered an error while processing your request.")

if __name__ == '__main__':
    main()