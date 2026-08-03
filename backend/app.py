import os
from flask import Flask

from config import config_by_name
from extensions import db, cors
from utils.error_handlers import register_error_handlers
from services.ml_service import ml_service
from routes import (
    health_bp,
    recommendation_bp,
    resume_bp,
    interview_bp,
    history_bp
)

def create_app(config_name=None, config_override=None):
    app = Flask(__name__)

    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'dev')
    
    app.config.from_object(config_by_name.get(config_name, config_by_name['dev']))
    
    if config_override:
        app.config.update(config_override)

    # Initialize extensions
    db.init_app(app)
    cors.init_app(app)

    # Register error handlers
    register_error_handlers(app)

    # Register blueprints
    app.register_blueprint(health_bp)
    app.register_blueprint(recommendation_bp)
    app.register_blueprint(resume_bp)
    app.register_blueprint(interview_bp)
    app.register_blueprint(history_bp)

    # Create database tables
    with app.app_context():
        db.create_all()

    # Train/load ML service
    try:
        if not ml_service.is_loaded:
            print("Initializing CareerPredictorService on app startup...")
            ml_service.load_and_train()
    except Exception as e:
        print(f"Error loading ML service on startup: {e}")

    return app

app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
