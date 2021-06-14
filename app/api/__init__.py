import connexion
from app.model.settings import get_buffer_db_path, init_dirs
from app.model.buffer import create_database
from os import path


def create_app(root_dir='app'):
    init_dirs(root_dir)
    if not path.exists(get_buffer_db_path()):
        create_database()
    """Initialize the core application."""
    # Create the application instance
    app = connexion.App(__name__, specification_dir="./openapi")

    # Read the swagger.yml file to configure the endpoints
    app.add_api("main_api.yaml", strict_validation=True)
    with app.app.app_context():
        # Include our Routes
        from . import routes
        return app
