from flask import Flask, request
import json
app = Flask(__name__)
import spacy
import random
import pickle
import fitz
from flask_cors import CORS, cross_origin
import os

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'log'])

app = Flask(__name__,static_url_path='', static_folder='./nlp/nl-project-master/build')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

cors = CORS(app)

# import requests

from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename


def main(fname):
    # print(filename, "FilenaMEEE")
    train_data = pickle.load(open('data.pkl', 'rb'))
    nlp = spacy.load('nlp_model100it')
    def train_model(train_data):
        if 'ner' not in nlp.pipe_names:
            ner = nlp.create_pipe('ner')
            nlp.add_pipe(ner, last=True)

        # for _,annotation in train_data:
        # for ent in annotation['entities']:
        # ner.add_label(ent[2])

        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
        with nlp.disable_pipes(*other_pipes):
            optimizer = nlp.begin_training()
            for itn in range(50):
                print("starting iteration " + str(itn))
                random.shuffle(train_data)
                losses = {}
                index = 0
                for text, annotations in train_data:
                    nlp.update([text], [annotations], drop=0.2, sgd=optimizer, losses=losses)

                print(losses)

    # train_model(train_data)
    # nlp.to_disk('nlp_model100it')
    # nlp_model=spacy.load('nlp_model')
    # doc=nlp_model(train_data[0][0])

    doc = nlp(train_data[0][0])

    skills = ['c++', 'java']
    score = 0

    for ent in doc.ents:
        print(f'{ent.label_.upper():{30}}-{ent.text}')

    output=[]
    doc = fitz.open(fname)
    text = ""
    for page in doc:
        text = text + str(page.getText())
    tx = " ".join(text.split('\n'))
    # print(tx)
    # doc=nlp_model(tx)
    doc = nlp(tx)
    for ent in doc.ents:
        if ent.label_ == "Skills":
            news = ent.text.lower().split(",")
            for i, x in enumerate(news):
                news[i] = x.replace(" ", "")
            for x in news:
                if x in skills:
                    score += 1
        output.append((ent.label_,ent.text))
        # print(f'{ent.label_.upper():{30}}-{ent.text}')
    output.append((score / len(skills)) * 100)
    print((score / len(skills)) * 100)
    return output


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route("/", methods = ['GET', 'POST'])
def index():
    # print("CHALO YAHA")
    # check if the post request has the file part
    if 'file' not in request.files:

        return redirect(request.url)
    file = request.files['file']
    # if user does not select file, browser also
    # submit a empty part without filename
    if file.filename == '':

        return redirect(request.url)
    if file and allowed_file(file.filename):

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        output = main(file_path)
        # print(output, "OUTPUUTTTTTT")
        return json.dumps(output)
    return redirect(request.url)

    # return redirect(request.url)

app.run()