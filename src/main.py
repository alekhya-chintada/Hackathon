from flask import Flask, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder="dist", static_url_path="")
CORS(app)

@app.route("/")
def home():
    return send_from_directory(app.static_folder, "index.html")

if __name__ == "__main__":
    app.run(debug=True, port=5000)
