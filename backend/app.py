import os
from flask import Flask, jsonify

app = Flask(__name__)


@app.get("/health")
def health():
    # Simple plaintext health check endpoint
    return "healthy", 200, {"Content-Type": "text/plain; charset=utf-8"}


@app.get("/")
def root():
    return jsonify({"status": "ok", "endpoints": ["/health"]})


def create_app():
    return app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
