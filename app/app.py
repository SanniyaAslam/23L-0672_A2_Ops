"""Flask UI. All settings come from environment variables; debug is OFF."""
import io, os
from flask import Flask, render_template, request
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename
from db import Store
from inference import build_classifier

ALLOWED = {"jpg", "jpeg", "png", "webp"}

def create_app(store=None, classifier=None):
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev")
    app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_UPLOAD_MB", 5)) * 1024 * 1024
    store = store or Store(os.environ.get("DATABASE_URL", "sqlite:///visionguard.db"))
    classifier = classifier or build_classifier()

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/predict")
    def predict():
        f = request.files.get("image")
        if f is None or f.filename == "":
            return render_template("error.html", msg="Please choose an image."), 400
        name = secure_filename(f.filename)
        if "." not in name or name.rsplit(".", 1)[1].lower() not in ALLOWED:
            return render_template("error.html", msg="Only JPG, PNG or WEBP files are accepted."), 400
        data = f.read()                                   # kept in memory only, never written to disk
        try:
            Image.open(io.BytesIO(data)).verify()         # looks like an image but undecodable -> reject
        except (UnidentifiedImageError, OSError):
            return render_template("error.html", msg="That file could not be decoded as an image."), 422
        try:
            label, conf, mname, mver = classifier.classify(name, data, f.mimetype)
        except Exception:
            app.logger.exception("inference failed")
            return render_template("error.html", msg="The model service is unavailable. Try again later."), 502
        store.add(filename=name, label=label, confidence=conf, model_name=mname, model_version=str(mver))
        return render_template("result.html", label=label, conf=round(conf * 100, 1), name=name)

    @app.get("/history")
    def history():
        return render_template("history.html", rows=store.recent())

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.errorhandler(413)
    def too_big(_):
        return render_template("error.html", msg="File too large (limit "
                               f"{app.config['MAX_CONTENT_LENGTH'] // 1024 // 1024} MB)."), 413

    @app.errorhandler(404)
    def nf(_):
        return render_template("error.html", msg="Page not found."), 404

    @app.errorhandler(500)
    def err(_):
        return render_template("error.html", msg="Something went wrong on our side."), 500

    return app

if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=5000, debug=False)