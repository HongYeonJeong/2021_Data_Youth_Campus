import pandas as pd
import numpy as np
from konlpy.tag import Okt
from gensim.models import Word2Vec, KeyedVectors

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

import pickle
import joblib
import re
import dill
import warnings

from werkzeug.utils import redirect

warnings.filterwarnings("ignore")

#%%
from flask import Flask, request, render_template_string, render_template, url_for
from flask_wtf import FlaskForm
from wtforms import SelectField, SubmitField, StringField


#%%
def prepare():
    origin_data = pd.read_csv("xgolf 조인 데이터(최종).csv")
    data = origin_data.copy()
    data = data.drop_duplicates(['Title','Date','Post'])
    data = data.dropna()
    data = data.drop(['Link', 'Caddie_Rate', 'Course_Rate', 'Price_Rate', 'ETC_Rate'], axis=1)
    global okt
    okt = Okt()
    return data, okt

def loading():
    #word2vec 모델 로드
    global w2v_model
    w2v_model = KeyedVectors.load_word2vec_format("golf_w2v")

    #로지스틱 회귀 모델 로드
    log_model = joblib.load("log_model.pkl")

    #카운트벡터라이저 모델 로드
    with open("countvectorizer_model.pkl", 'rb') as f:
        count_model = dill.load(f)

    with open("tfidfvectorizer_model.pkl", 'rb') as s:
        tfidf_model = dill.load(s)
    return w2v_model, log_model, count_model, tfidf_model

def question():
    while True:
        user1 = int(input("1. 그린피 선택(4인 기준):\n 1) 2만원~7만원 2) 7만원~12만원 3) 12만원~17만원 4) 17만원~22만원\n 5) 22만원~27만원 6) 27만원~30만원 7) 상관없음\n *선택: "))
        if user1 not in [1, 2, 3, 4, 5, 6, 7]:
            print('선택사항에 없습니다. 다시 선택해주세요.')
        else:
            break
    print()

    while True:
        user2 = int(input("2. 숙박 선택(4인 기준):\n 1) 숙박 안 함 2) 10만원~30만원  3) 40만원~70만원 4) 상관없음\n *선택: "))
        if user2 not in [1, 2, 3, 4, 5, 6, 7, 8]:
            print('선택사항에 없습니다. 다시 선택해주세요.')
        else:
            break
    print()

    while True:
        user3 = int(input("3. 거리(km)(강남역 기준) 선택:\n 1) 10~60km 2) 60~120km 3) 120~160km 4) 160~300km 5) 상관없음\n *선택: "))
        if user3 not in [1, 2, 3, 4, 5]:
            print('선택사항에 없습니다. 다시 선택해주세요.')
        else:
            break

    print()
    user4 = input("4. 골프장에 대한 추가 요구사항을 짧게 적어주세요.\n *요구사항: ")

    return user1, user2, user3, user4

def data_extract(user1, user2, user3, data):
    #user1
    if user1 == 1:
        data = data[(data['그린피(18홀 기준)'] >= 20000) & (data['그린피(18홀 기준)'] < 70000)]
    elif user1 == 2:
        data = data[(data['그린피(18홀 기준)'] >= 70000) & (data['그린피(18홀 기준)'] < 120000)]
    elif user1 == 3:
        data = data[(data['그린피(18홀 기준)'] >= 120000) & (data['그린피(18홀 기준)'] < 170000)]
    elif user1 == 4:
        data = data[(data['그린피(18홀 기준)'] >= 170000) & (data['그린피(18홀 기준)'] < 220000)]
    elif user1 == 5:
        data = data[(data['그린피(18홀 기준)'] >= 220000) & (data['그린피(18홀 기준)'] < 270000)]
    elif user1 == 6:
        data = data[(data['그린피(18홀 기준)'] >= 270000) & (data['그린피(18홀 기준)'] < 300000)]

    #user2
    if user2 == 1:
        data = data[data['숙박(4인 기준)'] == "-"]
    else:
        data = data.drop(data[data['숙박(4인 기준)'] == "-"].index)
        data['숙박(4인 기준)'] = pd.to_numeric(data['숙박(4인 기준)'])
        if user2 == 2:
            data = data[(data['숙박(4인 기준)'] >= 100000) & (data['숙박(4인 기준)'] < 300000)]
        elif user2 == 3:
            data = data[(data['숙박(4인 기준)'] >= 400000) & (data['숙박(4인 기준)'] < 700000)]
        elif user2 == 4:
            pass

    #user3
    if user3 == 1:
        data = data[(data['거리'] >= 10) & (data['거리'] < 60)]
    elif user3 == 2:
        data = data[(data['거리'] >= 60) & (data['거리'] < 120)]
    elif user3 == 3:
        data = data[(data['거리'] >= 120) & (data['거리'] < 160)]
    elif user3 == 4:
        data = data[(data['거리'] >= 160) & (data['거리'] < 300)]

    return data

def user4_analyze(user4):
    string = user4
    nouns = []
    nouns = okt.nouns(string)

    return nouns

def keyword_extract(data, nouns):
    for item in nouns:
        data = data[data['Post'].str.contains(item)]
    data.reset_index(inplace=True)
    data = data.drop("index", axis=1)

    return data

def text_cleaning(text):
    hangul = re.compile('[^ ㄱ-ㅣ가-힣]+')
    result = hangul.sub('', str(text))
    return result

def get_pos(x):
    tagger = Okt()
    stopwords = ['을', '를', '이', '가', '은', '는', '있', '하', '것', '들', '그', '되', '수', '않', '이렇', '어떻', '에', '의']
    pos = tagger.pos(x)
    pos = ['{}/{}'.format(word, tag) for word, tag in pos if word not in stopwords if (tag == 'Noun') or (tag == 'Adjective')]
    #pos = ['{}/{}'.format(word, tag) for word, tag in pos if (tag == 'Noun') or (tag == 'Adjective')]
    #pos = ['{}/{}'.format(word, tag) for word, tag in pos]
    return pos

def review_cleaning(data):
    clean_list = []
    for i in range(len(data)):
        before = data.Post[i]
        after = text_cleaning(before)
        clean_list.append(after)

    data['Text_Clean'] = clean_list
    return data

def positive_review_extract(data, count_model, tfidf_model, log_model):
    drop_idx_list = []
    train = data['Text_Clean'].tolist()
    X = count_model.transform(train)
    X = tfidf_model.transform(X)
    result = log_model.predict(X)

    drop_idx_list = []
    for i, v in enumerate(result):
        if v == 0:
            drop_idx_list.append(i)

    data = data.drop(drop_idx_list)
    data.reset_index(inplace = True)
    data = data.drop("index", axis=1)

    return data

def data_vectorized(data, w2v_model):
    before_vector_list = []
    for item in data['Text_Clean']:
        result = okt.pos(str(item), norm=True, stem=True)
        list_token = []
        for word in result:
            if (len(word[0]) > 1) and ((word[1] == "Noun") or (word[1] == 'Adjdctive')):
                list_token.append(word[0])
        before_vector_list.append(list_token)

    vector = []
    for item in before_vector_list:
        try:
            vector.append(w2v_model.wv[item])
        except:
            pass

    vectorized = []
    for tokens in before_vector_list:
        zero_vector = np.zeros(w2v_model.vector_size)
        vectors = []
        for token in tokens:
            if token in w2v_model.wv:
                try:
                    vectors.append(w2v_model.wv[token])
                except KeyError:
                    continue
        if vectors:
            vectors = np.asarray(vectors)
            avg_vec = vectors.mean(axis=0)
            vectorized.append(avg_vec)
        else:
            vectorized.append(zero_vector)

    data['Vectorized'] = vectorized
    return data

def KNN(user4, data):
    KNN = KNeighborsClassifier(n_neighbors=1)
    X = data['Vectorized']
    Y = data['Golf']

    label_encoding = LabelEncoder()
    Y = label_encoding.fit_transform(Y)
    Y = list(Y)
    X = list(X)

    if X == []:
        print()
        print('*해당 요구사항에 맞는 골프장이 없습니다. 죄송합니다.')
        return None

    KNN.fit(X, Y)

    test_sentence = []
    result = okt.pos(str(user4), norm=True, stem=True)
    for word in result:
        if (len(word[0]) > 1) and ((word[1] == "Noun") or (word[1] == "Adjective")):
            test_sentence.append(word[0])

    test_vector = []
    for i in test_sentence:
        try:
            test_vector.append(w2v_model.wv[i])
        except:
            pass

    test_vectorized = []
    vector = []
    for token in test_sentence:
        if token in w2v_model.wv:
            try:
                vector.append(w2v_model.wv[token])
            except KeyError:
                continue
    if vector:
        vector = np.asarray(vector)
        avg_vec = vector.mean(axis=0)
        test_vectorized.append(avg_vec)
    else:
        test_vectorized.append(zero_vector)

    guesses = int(KNN.predict(test_vectorized)[0])
    classes = list(label_encoding.classes_)

    return classes[guesses]

def golf_price(data, result):
    data = data[data['Golf'] == result]
    data = data.reset_index()
    data = data.drop("index", axis=1)

    price1 = data['그린피(18홀 기준)'][0]
    price2 = data['캐디피(4인 기준)'][0]
    price3 = data['카트비(4인 기준)'][0]
    price4 = data['식사(조식)'][0]
    price5 = data['숙박(4인 기준)'][0]
    price6 = data['거리'][0]

    return price1, price2, price3, price4, price5, price6

#-----------------------------------------------------------------------------------------------------------------------

app = Flask(__name__)
app.config["SECRET_KEY"] = "very_secret"


#%%
class TaskForm2(FlaskForm):
    tuple_list = [("2만원~7만원", 1), ("7만원~12만원", 2), ("12만원~17만원", 3), ("17만원~22만원", 4), ("22만원~27만원", 5), ("27만원~30만원", 6), ("상관없음", 7)]
    select = SelectField("", choices=tuple_list)


    tuple_list2 = [("숙박 안 함", 1), ("10만원~30만원", 2), ("40만원~70만원", 3), ("상관없음", 4)]
    select2 = SelectField("", choices=tuple_list2)


    tuple_list3 = [("10~60km", 1), ("60~120km", 2), ("120~160km", 3), ("160~300km", 4), ("상관없음", 5)]
    select3 = SelectField("", choices=tuple_list3)


    select4 = StringField('')
    submit = SubmitField("제출하기")

@app.route("/", methods=["GET", "POST"])
def new_task():

    form = TaskForm2()
    if request.method == "POST":
        print("-------------FORM--------------")
        print(form.data)
        print("-------------END--------------")

        data, okt = prepare()

        w2v_model, log_model, count_model, tfidf_model = loading()

        while True:
            user1, user2, user3, user4 = int(form.select.data), int(form.select2.data), int(form.select3.data), form.select4.data
            test_data = data_extract(user1, user2, user3, data)
            if len(test_data) != 0:
                break
            else:
                ans = '---------------------------------------------------------------\n사용자 요구에 맞는 골프장이 없습니다. 질문에 다시 답변해주세요\n---------------------------------------------------------------'

        nouns = user4_analyze(user4)
        test_data = keyword_extract(test_data, nouns)
        test_data = review_cleaning(test_data)
        test_data = positive_review_extract(test_data, count_model, tfidf_model, log_model)
        test_data = data_vectorized(test_data, w2v_model)
        result = KNN(user4, test_data)

        if result:
            price1, price2, price3, price4, price5, price6 = golf_price(data, result)

            ans = '\n' + '---------------------------------------------------------------' + '\n' \
                + '사용자에게 추천하는 골프장입니다.' +'\n'+ '---------------------------------------------------------------' + '\n' \
                    + '골프장:' + str(result) + '\n' + '그린피(18홀 기준):' + str(price1) + '\n' + '캐디피(4인 기준):' + str(price2) + '\n' + '카트비(4인 기준):' + str(price3) \
                        + '\n' + '식사(조식):' + str(price4) + '\n' + '숙박(4인 거리):' + str(price5) + '\n' + '거리(km):' + str(price6)

        html_ans = '------------------------------------<br>사용자에게 추천하는 골프장입니다.<br>------------------------------------<br><br> 골프장 : ' \
            + str(result) + '<br>' + '그린피(18홀 기준) : ' + str(price1) + '<br>' + '캐디피(4인 기준) : ' + str(price2) + '<br>' + '카트비(4인 기준) : ' + str(price3) \
                        + '<br>' + '식사(조식) : ' + str(price4) + '<br>' + '숙박(4인 기준) : ' + str(price5) + '<br>' + '거리(km) : ' + str(price6)

        return html_ans

    return render_template("/index1.html", title="New Task", form=form)


if __name__ == "__main__":
    app.run()