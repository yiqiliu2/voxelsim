"""VoxelSim Workbench — Flask application factory."""

import threading

from flask import Flask, render_template, jsonify

from .config import WEB_DIR
from .index import get_index


def create_app(prewarm: bool = True) -> Flask:
    app = Flask(__name__, template_folder=str(WEB_DIR / "templates"),
                static_folder=str(WEB_DIR / "static"))
    app.config["JSON_SORT_KEYS"] = False

    from .api_results import bp as results_bp
    from .api_analysis import bp as analysis_bp
    from .api_catalog import bp as catalog_bp
    from .api_jobs import bp as jobs_bp
    from .api_system import bp as system_bp
    app.register_blueprint(results_bp)
    app.register_blueprint(analysis_bp)
    app.register_blueprint(catalog_bp)
    app.register_blueprint(jobs_bp)
    app.register_blueprint(system_bp)

    @app.route("/")
    def index_page():
        return render_template("index.html")

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "not found"}), 404

    @app.errorhandler(500)
    def server_error(e):
        return jsonify({"error": "internal error"}), 500

    # Build the index eagerly so the first request is fast.
    get_index()

    if prewarm:
        threading.Thread(target=_prewarm, daemon=True).start()
    return app


def _prewarm():
    try:
        get_index().prewarm()
    except Exception:
        pass
