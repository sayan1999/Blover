# Creating a Web server using Python and Flask

from flask import Flask, render_template, request, make_response, jsonify, Response
import io
import base64
from PIL import Image

from blover import BLOG, COVER, UPSCALAR, BLOVER


def b_summarize(text):
    BLOG().summarize(text)


def b_createcover():
    COVER().create()


app = Flask("app")


@app.route("/")
def run():
    return render_template("index.html")


@app.route("/summarize", methods=["POST"])
def summarize():
    print("Blog->>>>>>>>>", request.get_json()["blog"])
    b_summarize(request.get_json()["blog"].strip())
    return jsonify({"summary": BLOVER.summary})


@app.route("/createcover", methods=["GET"])
def createcover():
    if not BLOVER.summary:
        print("Summary missing, can't create cover")
        img = Image.open("static/DpT093HUcAIH6D-.jpeg")
    else:
        b_createcover()
        img = Image.open(BLOVER.cover)
    # img = img.convert("L")  # ie. convert to grayscale

    # data = file.stream.read()
    # data = base64.b64encode(data).decode()

    buffer = io.BytesIO()
    img.save(buffer, "png")
    buffer.seek(0)

    data = buffer.read()
    data = base64.b64encode(data).decode()
    return data
    # return f'<img src="data:image/png;base64,{data}">'


app.run(host="0.0.0.0", port=8080)
