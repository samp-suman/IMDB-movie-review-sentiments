from flask import Flask, render_template, request
import pickle
import requests
from bs4 import BeautifulSoup as soup
import numpy as np

app = Flask(__name__)
word_list = pickle.load(open('word_list.pkl', 'rb'))
clf = pickle.load(open('model.pkl', 'rb'))
imdb_data = pickle.load(open('imdb_title.pkl', 'rb'))

headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.149 Safari/537.36',
           'accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
           'accept-encoding': 'gzip, deflate, br',
           'accept-language': 'en-US,en;q=0.9,hi;q=0.8',
           }


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    reviews_data = None
    movie_details = None
    movies_details = None
    details_shape = None
    predict_list = None
    sentiment = 0
    flag = 0

    movie_title = request.form.get('message')
    movie_id = request.form.get('title-id')

    if movie_title:
        movie_ids = getId(movie_title.lower())
        if movie_ids.shape[0] > 1:
            flag = -2
            movies_details = getDetails(movie_ids)
            details_shape = movies_details.shape[0]
        if movie_ids.shape[0] == 1:
            movie_id = movie_ids[0]
        if movie_ids.shape[0] == 0:
            flag = -1
    if movie_id:
        review_detail = getReview(movie_id)
        if review_detail is None:
            flag = -1
        else:
            movie_details = getMovieDetails(movie_id)
            predict_list, reviews_data = getPrediction(review_detail)
            try:
                sentiment = round(sum(predict_list)/len(predict_list)*10, 2)
                flag = 1
            except:
                flag = -3
                sentiment = 0
            # review_detail.append(predict_list)

            print(review_detail[0:3])
    return render_template('index.html', prediction=sentiment, reviews=reviews_data, movie=movie_details, details=movies_details, shape=details_shape, flag=flag)


def getId(title):
    movie_ids = imdb_data[(imdb_data['title'].str.contains(title)) | (
        imdb_data['original_title'].str.contains(title))]['imdb_title_id'].values
    # print(movie_ids)
    # if movie_ids.shape[0]==0:
    #     movie_ids=None
    return movie_ids


def getPrediction(reviews):
    predict_list = []
    for item in reviews:
        review = item[4].lower().split()
        feature = []
        for i in word_list:
            feature.append(review.count(i[0]))
        sample = np.array(feature).reshape(1, 5000)
        x = clf.predict(sample)[0]
        predict_list.append(x)
        reviews[item[0]].append(str(x))
    return predict_list, reviews


def getReview(imdb_id):
    url = "https://www.imdb.com/title/{}/reviews/_ajax".format(imdb_id)
    page = requests.get(url, headers=headers)
    if page.status_code == 404:
        return None
    raw = soup(page.content, 'html')
    reviews = raw.find_all('div', class_='imdb-user-review')
    review_return = []
    ids = 0
    for item in reviews:

        # rating for the movie by this user
        try:
            rating = item.find(
                'span', class_='rating-other-user-rating').span.text
        except:
            rating = None
            # User who has posted this review
        try:
            user = item.find('span', class_='display-name-link').text
        except:
            user = None
        # Review title-
        try:
            review_title = item.find('a', class_='title').text
        except:
            review_title = None
        # Review text
        review = item.find('div', class_='text show-more__control').text

        review_return.append([ids, rating, user, review_title, review])
        # review id- for convenience
        ids += 1

    return review_return


def getMovieDetails(imdb_id):
    url = "https://www.imdb.com/title/{}/".format(imdb_id)
    page = requests.get(url, headers=headers)
    if page.status_code == 404:
        return None
    raw = soup(page.content, 'html.parser')
    title = raw.find('h1', class_='dxSWFG').text
    try:
        run_time = raw.find_all('li', role="presentation")[1].text
    except:
        run_time = None

    try:
        released_year = raw.find('span', class_='jedhex').text
    except:
        released_year = None
    try:
        genre = '  '
        for i in raw.find('div', class_='dMcpOf').find_all('a'):
            genre += i.text + ',  '
    except:
        genre = None
    try:
        poster_link = "imdb.com/"+raw.find('a', class_="ipc-focusable")['href']
    except:
        poster_link = None
    try:
        rating = raw.find('span', class_="iTLWoV").text
    except:
        rating = None
    details = [title, run_time, released_year, genre, rating, poster_link]


def getDetails(ids):
    details = imdb_data[imdb_data['imdb_title_id'].isin(
        ids)][['imdb_title_id', 'title', 'year']].values
    return details


if __name__ == "__main__":
    app.run(debug=True)
