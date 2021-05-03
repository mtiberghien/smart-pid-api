from api import create_app
import app.model.settings as settings

settings.init_dirs('')
app = create_app()

if __name__ == "__main__":
    app.run(port=1975, debug=True)
