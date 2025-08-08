# calamity_base.py

import os
import sys
from dotenv import load_dotenv
import re
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
#parent document retriever.
load_dotenv()




def main():
    """
    Builds a knowledge base from a curated list of web pages.
    """
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY not found in .env file.")
        sys.exit(1)

    # --- PUT YOUR 8-10 CALAMITY WIKI URLS HERE ---
    CALAMITY_WIKI_URLS = [
        "https://calamitymod.wiki.gg/wiki/Murasama",
        "https://calamitymod.wiki.gg/wiki/Empyrean_Knives",
        "https://calamitymod.wiki.gg/wiki/Bosses",
        "https://calamitymod.wiki.gg/wiki/Weapons",
        "https://calamitymod.wiki.gg/wiki/Armor",
        "https://calamitymod.wiki.gg/wiki/Accessories",
        "https://calamitymod.wiki.gg/wiki/Yharon,_Dragon_of_Rebirth",
        "https://calamitymod.wiki.gg/wiki/Supreme_Witch,_Calamitas",
        "https://calamitymod.wiki.gg/wiki/Biomes",
        "https://calamitymod.wiki.gg/wiki/Sunken_Sea",
        "https://calamitymod.wiki.gg/wiki/The_Abyss",
    ]

    print("--- Starting knowledge base creation ---")

    # --- Load Documents with Content Filtering ---
    print(f"\nLoading {len(CALAMITY_WIKI_URLS)} pages from the Calamity Wiki...")
    loader = WebBaseLoader(CALAMITY_WIKI_URLS)
    raw_docs = loader.load()
    print(f"Raw documents loaded: {len(raw_docs)}")
    
    # Filter and clean the content
    docs = []
    for doc in raw_docs:
        print(f"Processing: {doc.metadata.get('source', 'Unknown').split('/')[-1]}")
        
        # Clean the content
        content = doc.page_content
        
        # Remove common navigation patterns
        navigation_patterns = [
            r'Create account\s*Log in.*?Navigation menu',
            r'Toggle personal tools menu.*?Navigation',
            r'Main pageRecent changes.*?Resources',
            r'Class setupsGuides.*?Characters',
            r'BossesCrittersEnemiesNPCs.*?Portals',
            r'WebsiteForum Page.*?Tools',
            r'What links here.*?In other languages',
            r'Report bad advertisement.*?Cookies help us',
            r'This page was last edited.*?Status page',
        ]
        
        for pattern in navigation_patterns:
            content = re.sub(pattern, '', content, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove excessive whitespace and empty lines
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        clean_content = '\n'.join(lines)
        
        # Remove lines that are clearly navigation
        filtered_lines = []
        for line in lines:
            # Skip navigation-like lines
            if any(nav_word in line.lower() for nav_word in [
                'create account', 'log in', 'navigation menu', 'namespaces', 
                'pagediscussion', 'english', 'views', 'readsign up', 'purge cache',
                'main pagerecent', 'class setupsguides', 'accessoriesarmor',
                'websiteforumdiscord', 'what links here', 'русскийespañol'
            ]):
                continue
            
            # Skip very short lines (likely navigation)
            if len(line) < 10:
                continue
                
            filtered_lines.append(line)
        
        final_content = '\n'.join(filtered_lines)
        
        # Only keep documents with substantial content
        if len(final_content) > 300 and any(keyword in final_content.lower() for keyword in [
            'calamity', 'terraria', 'weapon', 'boss', 'biome', 'damage', 'enemy', 'item'
        ]):
            doc.page_content = final_content
            docs.append(doc)
            print(f"  > Kept: {len(final_content)} characters")
        else:
            print(f"  X Filtered out: {len(final_content)} characters")
    
    print(f"\nFiltered documents: {len(docs)}")

    # --- Split Documents into Chunks ---
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    print(f"Created {len(chunks)} text chunks.")

    # --- Create and Save FAISS Vector Store ---
    print("\nCreating vector embeddings and FAISS index...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        transport="rest"
    )

    db = FAISS.from_documents(chunks, embeddings)
    db.save_local("faiss_index_combined")

    print("\n--- Knowledge base creation complete! ---")
    print("The 'faiss_index_combined' folder has been created and is ready to use.")


if __name__ == '__main__':
    main()