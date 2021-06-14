from api import create_app


app = create_app('')

if __name__ == "__main__":
    app.run(port=1948, debug=True)
