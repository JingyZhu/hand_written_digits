import flask
import recog

@recog.app.route('/', methods=['GET'])
def get_index():
    return flask.render_template("index.html")