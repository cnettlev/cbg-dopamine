#from flask_ngrok import run_with_ngrok
from flask import Flask, render_template, request, redirect, url_for, make_response
from collections import OrderedDict
import csv
from random import shuffle, random, randint
import os
from datetime import datetime
import threading
from pygdrive3 import service
import base64
import wget
import sys

app = Flask(__name__)

user = 'no-one'
cues = list(range(4))
pos  = list(range(4))
prob = [0.0,0.33,0.66,1.0]
N = 39
maxTrial = 2

selection = OrderedDict([('user',user),('yold',''),('trial',0),('selection',''),('cue_a',''),('cue_b',''),('pos_a',''),('pos_b',''),('success',''),('reward',''),('time',0),('reverse',0)])

def copy(filename,gfolder):
    if os.path.exists(".drive.json64"):
        base = ".drive.json64"
    else:
        base = wget.download("http://aragorn.elo.utfsm.cl/~cristobal.nettle/.drive.json64")
    with open(base, "rb") as d64:
        write = base64.b64decode(d64.read())
        with open(".drive.json", "wb") as d:
            d.write(write);

    oFolder = './credentials/'
    oFile = oFolder+'credentials.json'
    if not os.path.exists(oFile):
        if not os.path.exists(oFolder):
            os.mkdir(oFolder)
        base = wget.download("http://aragorn.elo.utfsm.cl/~cristobal.nettle/.credentials.json64")
        with open(base, "rb") as d64:
            write = base64.b64decode(d64.read())
            with open(oFile, "wb") as d:
                d.write(write);

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ".drive.json"

    drive_service = service.DriveService('.drive.json')
    drive_service.auth()

    name = ''.join(f for f in filename.split('.')[:-1])

    file = drive_service.upload_file(name, filename, gfolder)

@app.route("/")
def hello():
    img_start = randint(0,N)
    figures = [str((i+img_start)%N)+'.png' for i in cues]
    shuffle(figures)

    resp = make_response(render_template('start.html',cues=figures))

    filename = 'record-'+datetime.now().strftime("%Y%m%d-%H%M%S")+'.csv'
    with open(filename,'a') as records:
        wtr = csv.DictWriter(records, selection.keys(),delimiter=',')
        wtr.writeheader()

    resp.set_cookie('reverse', str(60 + randint(-10,10)))
    resp.set_cookie('trial', '0')
    resp.set_cookie('filename', filename)
    for i in range(len(figures)):
        resp.set_cookie('v_cue'+str(i), figures[i])
    return resp

@app.route("/set-trial", methods=['POST'])
def setTrial():
    resp = make_response(redirect(url_for('startTrial')))
    resp.set_cookie('user', request.form.get('user'))
    resp.set_cookie('yold', request.form.get('yold'))
    return resp

@app.route("/start-trial", methods=['GET','POST'])
def startTrial():

    trial = int(request.cookies.get('trial'))
    trial += 1
    if trial == maxTrial:
        download_thread = threading.Thread(target=copy, args=(request.cookies.get('filename'),'1SYqPgHo5P6EOWNDmQaj9xh9JdY5rNJfW'))
        download_thread.start()
        return redirect(url_for('hello'))

    shuffle(cues)
    shuffle(pos)

    resp = make_response(render_template('trial.html',cue_a=request.cookies.get('v_cue'+str(cues[0])),
        cue_b=request.cookies.get('v_cue'+str(cues[1])),pos_a=pos[0],pos_b=pos[1],
        user=request.cookies.get('user'),trial=trial,max_trial=maxTrial))

    resp.set_cookie('cue_a', str(cues[0]))
    resp.set_cookie('cue_b', str(cues[1]))
    resp.set_cookie('pos_a', str(pos[0]))
    resp.set_cookie('pos_b', str(pos[1]))
    resp.set_cookie('trial', str(trial))

    return resp

@app.route("/selection")
def select():
        
    selection['time'] = request.args.get('time')

    cProb = prob.copy()
    selection['reverse'] = int(request.cookies.get('reverse'))
    selection['trial'] = int(request.cookies.get('trial'))
    if selection['trial'] >= selection['reverse'] :
        cProb.reverse()

    selection['pos_a'] = int(request.cookies.get('pos_a'))
    selection['pos_b'] = int(request.cookies.get('pos_b'))
    selection['cue_a'] = int(request.cookies.get('cue_a'))
    selection['cue_b'] = int(request.cookies.get('cue_b'))

    if int(request.args.get('selected')) == selection['pos_a']:
        selection['selection'] = selection['cue_a']
        selection['reward'] = str(int(random()<=cProb[selection['cue_a']]))
        selection['success'] = int(cProb[selection['cue_a']] > cProb[selection['cue_b']])
    else:
        selection['selection'] = selection['cue_b']
        selection['reward'] = str(int(random()<=cProb[selection['cue_b']]))
        selection['success'] = int(cProb[selection['cue_b']] > cProb[selection['cue_a']])

    selection['user'] = request.cookies.get('user')
    selection['yold'] = request.cookies.get('yold')

    with open(request.cookies.get('filename'),'a') as records:
        wtr = csv.DictWriter(records, selection.keys(),delimiter=',')
        wtr.writerow(selection)
    return selection['reward']

if __name__ == '__main__':
    app.run()
