# Multi-Domain Chatbot

Intelligent chatbot supporting Terraria Calamity mod, Government procurement (GeM), and general knowledge.

## 🚀 Features
- **Semantic Classification**: Smart question routing
- **Multi-Domain Knowledge**: Calamity + GeM + General
- **PDF Processing**: Government document integration
- **Async Processing**: Fast response times

## 🔧 Setup

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd PythonProject1
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Environment Setup**
```bash
cp .env.example .env
# Edit .env and add your Google API key
```

4. **Run the server**
```bash
python manage.py runserver
```

## 🔒 Security Notes

- **NEVER commit .env files** - they contain API keys
- **PDF documents are excluded** - they may contain sensitive data
- **Vector stores are excluded** - they're large and may contain sensitive embeddings
- **Database files are excluded** - they may contain user data

## 📁 Project Structure
```
chat/
├── chatbot_logic.py      # Main service
├── chatbot_graph.py      # Workflow
├── pdf_processor.py      # PDF processing
├── views.py             # Django views
└── async_chat.py        # Async processing
```

## 🎯 Usage

Ask questions about:
- **Calamity**: "How to defeat Yharon?"
- **GeM**: "What is government bidding process?"
- **General**: "What is 2+2?"

## ⚠️ Important
- Keep your Google API key secure
- Don't share .env files
- PDF documents contain sensitive government data