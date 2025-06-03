from search import search

def main():
    """Simple text-based search interface"""
    print("UCI ICS Search Engine")
    print("Enter your query (or 'quit' to exit):")
    
    while True:
        query = input("\nQuery> ").strip()
        if query.lower() == 'quit':
            break
            
        if not query:
            continue
            
        print("\nSearching...")
        results = search(query)
        
        if not results:
            print("No results found.")
            continue
            
        print("\nTop 5 results:")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['url']} (score: {result['score']:.4f})")

if __name__ == "__main__":
    main() 