from flask import Blueprint, render_template

views = Blueprint('views', __name__)

@views.route('/')
@views.route('/home')
def index():
    return render_template("index.html")

@views.route('/about')
def about():
    return render_template("about.html")

@views.route('/contact')
def contact():
    return render_template("contact.html")

@views.route('/faq')
def faq():
    return render_template("faq.html")

@views.route('/controls')
def controls():
    return render_template("controls.html")

@views.route('/test')
def test():
    return render_template("test.html")




    