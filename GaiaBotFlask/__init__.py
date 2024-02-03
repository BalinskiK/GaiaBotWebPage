from flask import Flask, render_template

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'asdifybasvkjeausdvbjsfv'
    
    from .views import views
    from .controls import controls

    app.register_blueprint(views, url_prefix='/')
    app.register_blueprint(controls, url_prefix='/')
    
    
    return app