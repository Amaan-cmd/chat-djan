import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

def inspect_knowledge_base():
    print("=== KNOWLEDGE BASE INSPECTION ===\n")
    
    # Load the FAISS index
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        transport="rest"
    )
    
    try:
        db = FAISS.load_local("faiss_index_combined", embeddings, allow_dangerous_deserialization=True)
        print(f"Successfully loaded FAISS index")
        
        # Get all documents
        docs = db.docstore._dict
        print(f"Total chunks in knowledge base: {len(docs)}")
        
        # Analyze by source
        sources = {}
        for doc_id, doc in docs.items():
            source = doc.metadata.get('source', 'Unknown')
            if source not in sources:
                sources[source] = []
            sources[source].append(doc)
        
        print(f"\nSources breakdown:")
        for source, doc_list in sources.items():
            print(f"  - {source.split('/')[-1]}: {len(doc_list)} chunks")
        
        # Sample content quality check
        print(f"\nCONTENT QUALITY SAMPLES:")
        print("="*50)
        
        for source, doc_list in list(sources.items())[:3]:  # Check first 3 sources
            print(f"\nSOURCE: {source.split('/')[-1]}")
            print("-" * 30)
            
            # Show first chunk from this source
            sample_doc = doc_list[0]
            content = sample_doc.page_content[:500]  # First 500 chars
            
            print(f"Content preview:")
            print(f"'{content}...'")
            
            # Check for quality indicators
            quality_issues = []
            if len(content.strip()) < 50:
                quality_issues.append("Very short content")
            if "Report bad advertisement" in content:
                quality_issues.append("Contains ads/navigation")
            if "Cookies help us deliver" in content:
                quality_issues.append("Contains cookie notices")
            if content.count('\n') > content.count(' '):
                quality_issues.append("Too many line breaks")
            if not any(word in content.lower() for word in ['calamity', 'terraria', 'weapon', 'boss', 'biome']):
                quality_issues.append("No Calamity-related keywords")
            
            if quality_issues:
                print("Quality Issues:")
                for issue in quality_issues:
                    print(f"   {issue}")
            else:
                print("Content looks good")
        
        # Test search functionality
        print(f"\nSEARCH TEST:")
        print("="*30)
        test_queries = ["sunken sea", "abyss", "yharon", "murasama"]
        
        for query in test_queries:
            results = db.similarity_search(query, k=2)
            print(f"\nQuery: '{query}'")
            print(f"Results found: {len(results)}")
            if results:
                best_result = results[0]
                preview = best_result.page_content[:200].replace('\n', ' ')
                print(f"Best match: '{preview}...'")
                print(f"Source: {best_result.metadata.get('source', 'Unknown').split('/')[-1]}")
            else:
                print("No results found")
                
    except Exception as e:
        print(f"Error loading knowledge base: {e}")

if __name__ == "__main__":
    inspect_knowledge_base()