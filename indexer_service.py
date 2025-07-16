from flask import Flask, request, jsonify
import requests
from indexer import index_documents

app = Flask(__name__)

@app.route("/trigger-reader", methods=["POST"])
def trigger_reader():
    try:
        response = requests.post("http://127.0.0.1:5001/scan")
        reader_data = response.json()

        count = index_documents(reader_data.get("parsed_docs", {}))

        return jsonify({
            "message": "Reader triggered and indexed successfully.",
            "documents_indexed": count
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5002)
