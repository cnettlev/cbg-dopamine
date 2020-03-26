from flask import Flask, render_template, request, redirect, url_for
from collections import OrderedDict
import csv
from random import shuffle, random
from os.path import exists

app = Flask(__name__)
user = 'no-one'
cues = range(4)
pos  = range(4)
prob = [0.0,0.33,0.66,1.0]

images = ['01.png','02.png','03.png','04.png']
selection = OrderedDict([('user',user),('trial',1),('selection',''),('cue_a',''),('cue_b',''),('pos_a',''),('pos_b',''),('success',''),('reward',''),('time','')])

@app.route("/")
def hello():

    if not exists('records.csv'):
        with open('records.csv','a') as records:
            wtr = csv.DictWriter(records, selection.keys(),delimiter=',')
            wtr.writeheader()

    return render_template('start.html')

@app.route("/set-trial", methods=['POST'])
def setTrial():
    selection['user'] = request.form.get('user')
    return redirect(url_for('startTrial'))


@app.route("/start-trial", methods=['GET','POST'])
def startTrial():
    shuffle(cues)
    shuffle(pos)
    selection['trial'] += 1
    selection['cue_a'] = cues[0]
    selection['cue_b'] = cues[1]
    selection['pos_a'] = pos[0]
    selection['pos_b'] = pos[1]
    return render_template('trial.html',cue_a=images[cues[0]],cue_b=images[cues[1]],pos_a=pos[0],pos_b=pos[1],user=selection['user'])

@app.route("/selection")
def select():
    selection['time'] = request.args.get('time')

    if int(request.args.get('selected')) == pos[0]:
        selection['selection'] = cues[0]
        selection['reward'] = str(int(random()<=prob[cues[0]]))
        selection['success'] = int(prob[cues[0]] > prob[cues[1]])
    else:
        selection['selection'] = cues[1]
        selection['reward'] = str(int(random()<=prob[cues[1]]))
        selection['success'] = int(prob[cues[1]] > prob[cues[0]])

    with open('records.csv','a') as records:
        wtr = csv.DictWriter(records, selection.keys(),delimiter=',')
        wtr.writerow(selection)
    return selection['reward']

if __name__ == '__main__':
    app.run(host='0.0.0.0')
