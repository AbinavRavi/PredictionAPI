from flask import Flask,jsonify

app = Flask(__name__)

@app.route('/')
def my_app():
    return jsonify({"greeting":"Hello World"})

if __name__ == '__main__':
    app.run()

    