#server.py

from flask import Flask, render_template, session, url_for, redirect, request
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))+'/ai_challenge_2step_data')
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))+'/ai_challenge_2step_data/classifier/')
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))+'/Graph2Tree_kor')

import main

# g2t = main.graph2tree()
'''
g2t = main.graph2tree()
a,b = g2t.solve('어떤 수에서 36을 빼야 하는데 잘못하여 63을 뺀 결과가 8이 나왔습니다. 바르게 계산한 결과를 구하시오.')
print(a)
print(b)
'''

app=Flask(__name__)
app.secret_key='this is super key'
app.config['SESSION_TYPE']='filesystem'

g2t=main.graph2tree()

iq=[]
pca=[]
ansb=[]
recpc=[]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def getquestion():
    input=request.form.get('Question')
    iq.append(input)
    a,b = g2t.solve(input)
    pca.append(a)
    ansb.append(b)
    if request.method == 'POST':
        with open("static/recent.txt", "a", encoding='utf-8') as f:
            f.write("%s\t%s\t%s\n" % (input, a, b))
        return redirect(url_for('answersheet'))
    else:
        return render_template('answer.html', input=iq, pc=pca, ans=ansb)

@app.route('/type')
def type():
    return render_template('type.html')

@app.route('/recent')
def recent():
    with open("static/recent.txt", "r", encoding='utf-8') as f:
        data=f.readlines()
        rq=data[-1]
        rq=rq.strip().split('\t')
        rq[0]=rq[0]
        rq[1]=rq[1].replace(", '","\t")
        rq[1]=rq[1].replace("'","")
        rq[1]=rq[1].lstrip('[').rstrip(']').split('\t')
        rq[2]=rq[2].replace(", '", "\t")
        rq[2]=rq[2].replace("'", "")
        rq[2]=rq[2].lstrip('[').rstrip(']').split('\t')
    return render_template('recent.html', rq=rq[0], recpc=rq[1], recans=rq[2])

@app.route('/answer')
def answersheet():
    return render_template('answer.html', input=iq, pc=pca, ans=ansb)

if __name__=="__main__":
    app.run(debug=True)
    # app.run(debug=True, host="163.239.28.25", port=5000)
