import flask
import recog

@recog.app.route('/api/v1', methods=['GET', 'POST'])
def get_result():
    return "hello world"