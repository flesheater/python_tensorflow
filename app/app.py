#! /usr/local/bin/python3
from flask import Flask, render_template
from tensorflowTests import tensorflowTests

app = Flask(__name__)

@app.route("/")
def home():
    tft = tensorflowTests()

    return render_template(
        'home.html',
        img=tft.render(),
    )

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=80)
