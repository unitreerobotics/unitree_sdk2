from flask import Flask, request, send_file, jsonify, render_template_string
import json
import os

app = Flask(__name__)

UPLOAD_PATH = "received.jpg"
POSE_PATH = "pose.json"  # 위치 + 자세 값 저장용

@app.route("/analyze", methods=["POST"])
def analyze():
    image = request.files["image"]
    pos = json.loads(request.form["position"])
    rpy = json.loads(request.form["rpy"])

    print("Received image:", image.filename)
    print("Position:", pos)
    print("RPY:", rpy)

    # 이미지 저장
    image.save(UPLOAD_PATH)

    # 값 저장
    with open(POSE_PATH, "w") as f:
        json.dump({"position": pos, "rpy": rpy}, f)

    return {"result": "ok"}

@app.route("/view", methods=["GET"])
def view():
    if not os.path.exists(UPLOAD_PATH) or not os.path.exists(POSE_PATH):
        return "No image or pose data found."

    with open(POSE_PATH, "r") as f:
        data = json.load(f)

    html = """
    <html>
    <head>
        <title>Latest Image and Pose</title>
        <style>
            body { font-family: Arial; text-align: center; padding: 30px; }
            img { max-width: 400px; border: 1px solid #ccc; margin-bottom: 20px; }
            .info { font-size: 18px; }
        </style>
    </head>
    <body>
        <h1>Latest Image</h1>
        <img src="/latest_image" alt="Latest Image"/>
        <div class="info">
            <h2>Position</h2>
            <p>x = {{ pos['x'] }}, y = {{ pos['y'] }}, z = {{ pos['z'] }}</p>
            <h2>RPY</h2>
            <p>roll = {{ rpy['roll'] }}, pitch = {{ rpy['pitch'] }}, yaw = {{ rpy['yaw'] }}</p>
        </div>
    </body>
    </html>
    """

    return render_template_string(html, pos=data["position"], rpy=data["rpy"])

@app.route("/latest_image", methods=["GET"])
def latest_image():
    if os.path.exists(UPLOAD_PATH):
        return send_file(UPLOAD_PATH, mimetype="image/jpeg")
    else:
        return jsonify({"error": "No image found"}), 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
