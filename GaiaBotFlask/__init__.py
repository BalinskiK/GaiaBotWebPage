from flask import Flask, render_template

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'asdifybasvkjeausdvbjsfv'
    
    from .views import views

    app.register_blueprint(views, url_prefix='/')
    
    return app