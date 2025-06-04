from search import search

def main():
    print("=== ICS Search Engine (Console) ===")
    print("Type your query, or type 'exit' to quit.")

    while True:
        query = input("\nQuery: ").strip()
        if query.lower() == "exit":
            break

        results, elapsed = search(query)
        if not results:
            print("No results found.")
        else:
            print(f"\nTop 5 Results (in {elapsed} ms):")
            for i, (url, score) in enumerate(results, 1):
                print(f"{i}. {url} (score: {score:.2f})")

if __name__ == "__main__":
    main()