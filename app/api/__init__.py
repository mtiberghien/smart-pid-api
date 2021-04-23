import connexion


def create_app():
    """Initialize the core application."""
    # Create the application instance
    app = connexion.App(__name__, specification_dir="./openapi")

    # Read the swagger.yml file to configure the endpoints
    app.add_api("main_api.yaml", strict_validation=True)
    with app.app.app_context():
        # Include our Routes
        from . import routes
        return app