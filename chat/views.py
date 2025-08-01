#views acts as the agent between backend langchain and web
from django.shortcuts import render
from langchain_core.messages import HumanMessage, AIMessage

# here I will import graph from chatbot graph
from .chatbot_graph import create_graph

chatbot_app = create_graph()
#requesting chat history
def chat_view(request):
    chat_history = request.session.get('chat_history', [])

    if request.method == 'POST':
        question = request.POST.get('question', '')
        #formulate an input for chatbot ka graph
        langchain_chat_history = []
        initial_state = {
            "question": question,
            "chat_history": langchain_chat_history,
        }

        #calling the chatbot and waiting for full result
        final_state = chatbot_app.invoke(initial_state)
        #answer taken from result
        answer = final_state.get('answer', 'Sorry, I encountered an error.')

        chat_history = [
            {'role': 'user', 'content': question},
            {'role': 'ai', 'content': answer}
        ]

    # Save updated history
    request.session['chat_history'] = chat_history

    # Render the page with the full chat history
    return render(request, 'chat/chat.html', {'chat_history': chat_history})