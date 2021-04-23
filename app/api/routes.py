from flask import current_app as app, redirect, url_for
@app.route('/')
@app.route('/api')
@app.route('/api/')
def index(): return redirect(url_for('/api./api_swagger_ui_index'))
