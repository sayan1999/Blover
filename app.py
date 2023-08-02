from flask import Flask, render_template, request, make_response, jsonify, Response
import io
import base64
from PIL import Image

from blover import BLOG, COVER, UPSCALAR, BLOVER


def b_summarize(text):
    BLOG().summarize(text)


def b_createcover(pos="", neg=""):
    COVER().create(pos=pos, neg=neg)
    return Image.open(BLOVER.cover)


def img_to_base64(img):
    buffer = io.BytesIO()
    img.save(buffer, "png")
    buffer.seek(0)

    data = buffer.read()
    return base64.b64encode(data).decode()


bloverapp = Flask("blover")


@bloverapp.route("/")
def run():
    return render_template("index.html")


@bloverapp.route("/generate", methods=["POST"])
def generate():
    b_summarize(request.get_json()["blog"].strip())
    img = b_createcover()
    width, height = img.size
    return jsonify({"img": img_to_base64(img), "width": width, "height": height})


if __name__ == "__main__":
    bloverapp.run(host="0.0.0.0", port=8080, debug=True)
