from flask import Flask, render_template, request, jsonify
from search import search
import time

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search')
def search_query():
    query = request.args.get('q', '')
    if not query:
        return jsonify({'results': [], 'time': 0})
    
    results, elapsed = search(query)
    formatted_results = [{'url': url, 'score': f"{score:.2f}"} for url, score in results]
    
    return jsonify({
        'results': formatted_results,
        'time': elapsed
    })

if __name__ == '__main__':
    app.run(debug=True) 