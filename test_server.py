from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello! The server is working!"

if __name__ == '__main__':
    print("Starting test server...")
    print("Open http://127.0.0.1:5000 in your browser")
    app.run(host='127.0.0.1', port=5000, debug=True) 