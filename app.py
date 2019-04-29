import sys, os, spotipy, dash, flask, json
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import spotipy.util as util
import pandas as pd
import numpy as np
from spotipy.oauth2 import SpotifyClientCredentials
from dash.dependencies import Input, Output, State
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

with open('app_token.txt', 'r') as file:
    tokens = file.read()

token_dict = json.loads(tokens)

client_credentials_manager = SpotifyClientCredentials(client_id=token_dict['client_id'],
                                                      client_secret=token_dict['client_secret'])
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


def get_artist(name):
    results = sp.search(q='artist:' + name, type='artist')
    items = results['artists']['items']

    if len(items) > 0:
        return items[0]
    else:
        return None

def get_artist_albums(artist):
    albums = []
    results = sp.artist_albums(artist['id'], album_type='album')
    albums.extend(results['items'])
    while results['next']:
        results = sp.next(results)
        albums.extend(results['items'])

    seen = set() # to avoid duplicates.
    album_dct = dict() # to keep name and id of album.
    albums.sort(key=lambda album:album['name'].lower())
    for album in albums:
        name = album['name']
        album_id = album['id']
        if name not in seen:
            seen.add(name)
            album_dct[name] = album_id
    return album_dct

def get_album_tracks(album_id):
    tracks = []
    results = sp.album_tracks(album_id)
    tracks.extend(results['items'])
    while results['next']:
        results = sp.next(results)
        tracks.extend(results['items'])

    seen = set()
    track_dct = dict()
    tracks.sort(key=lambda track:track['name'].lower())
    for track in tracks:
        name = track['name']
        track_id = track['id']
        if name not in seen:
            seen.add(name)
            track_dct[track_id] = name
    return track_dct

def find_user_playlists(user_name):
    playlists = sp.user_playlists(user_name)
    playlists_dict = dict()
    while playlists:
        for i, playlist in enumerate(playlists['items']):
            playlists_dict[playlist['name']] = playlist['uri'].split(':')[-1]
        if playlists['next']:
            playlists = sp.next(playlists)
        else:
            playlists = None
        return playlists_dict

def show_user_tracks(results):
    artist_dict, track_dict = dict(), dict()
    for i, item in enumerate(results['items']):
        track = item['track']
        artist_name = track['artists'][0]['name']
        artist_id = track['artists'][0]['id']
        track_name = track['name']
        track_id = track['id']
        artist_dict[track_id] = artist_name
        track_dict[track_id] = track_name

    return artist_dict, track_dict

def duplicate_dropper(df, field):
    """
    Takes a pandas DataFrame object and drops duplicates in choosen column.
    """
    if len(df[field]) != len(df[field].unique()):
        print('total rows: {}; unique rows: {}'.format(len(df[field]), len(df[field].unique())))
        print('duplicates eleminated.')

        indicies_to_loc = []
        for index, value in zip(df[field].duplicated().index, df[field].duplicated()):
            if str(value) == 'False':
                indicies_to_loc.append(index)

        new_df = df.loc[indicies_to_loc]
        return new_df
    else:
        print('total rows: {}; unique rows: {}'.format(len(df[field]), len(df[field].unique())))
        print("this dataset doesn't include duplicates.")
        return df

def mood_detector(valence, energy):
    if valence > 0.50:
        if energy > 0.50:
            return 'Happy'
        else:
            return 'Relaxed'
    else:
        if energy > 0.50:
            return 'Angry'
        else:
            return 'Sad'

def todays_hot_hits():
    def show_thh_tracks(results):
        artist_dict, track_dict = dict(), dict()
        for i, item in enumerate(results['tracks']['items']):
            track = item['track']
            artist_name = track['artists'][0]['name']
            artist_id = track['artists'][0]['id']
            track_name = track['name']
            track_id = track['id']
            artist_dict[track_id] = artist_name
            track_dict[track_id] = track_name
        return artist_dict, track_dict

    track_dict_list = []
    artist_dict_list = []
    playlist_tracks = sp.user_playlist_tracks(user='spotify',
                                              playlist_id='37i9dQZF1DXcBWIGoYBM5M?si=yKc0b4axR8mcaoLj3rVyMQ')

    artist_dict, track_dict = show_thh_tracks(playlist_tracks)
    track_dict_list.append(track_dict)
    artist_dict_list.append(artist_dict)
    while playlist_tracks['tracks']['next']:
        playlist_tracks = sp.next(playlist_tracks)
        artist_dict, track_dict = show_thh_tracks(playlist_tracks)
        track_dict_list.append(track_dict)
        artist_dict_list.append(artist_dict)

    audio_feats = []
    for i in range(len(track_dict_list)):
        audio_feats.append(sp.audio_features(list(track_dict_list[i].keys())))

    df_list = []
    index = 0
    for i in range(len(audio_feats)):
        if audio_feats[i][0]:
            #print(i)
            for a in range(len(audio_feats[i])):
                index+=1
                df_list.append(pd.DataFrame(audio_feats[i][a], index=[index]))
    df = pd.concat(df_list)

    titles = []
    ids = []
    for i in range(len(track_dict_list)):
        for a in track_dict_list[i].keys():
            if a in list(df['id'].values):
                titles.append(track_dict_list[i][a])
                ids.append(a)
    track_titles = pd.DataFrame({'name':titles, 'id':ids})

    names = []
    ids = []
    for i in range(len(artist_dict_list)):
        for a in artist_dict_list[i].keys():
            if a in list(df['id'].values):
                names.append(artist_dict_list[i][a])
                ids.append(a)
    artist_names = pd.DataFrame({'artist_name':names, 'song_id':ids})

    first_df = df.merge(track_titles, left_on='id', right_on='id')
    first_df.drop(['type', 'key', 'uri', 'track_href',
                   'analysis_url', 'time_signature', 'mode', 'duration_ms'], 1, inplace=True)
    todays_hits_df =first_df.merge(artist_names, left_on='id', right_on='song_id')

    return todays_hits_df

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

server = flask.Flask(__name__)
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, server = server)


# Since we're adding callbacks to elements that don't exist in the app.layout,
# Dash will raise an exception to warn us that we might be
# doing something wrong.
# In this case, we're adding the elements through a callback, so we can ignore
# the exception.
app.config.suppress_callback_exceptions = True
app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})
app.title = 'Moodify'

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])


index_page = html.Div([
    html.H1('Moodify App', style = {
                                    'color':'black',
                                    'backgroundColor':'white',
                                    'text-align':'center'
                        }
                    ),
    html.Br(),
    dcc.Link('Sentiment Analysis For Your Favorite Artist',
             href='/page-1', style = {'font-size': '35px'}),

    html.Br(),
    dcc.Link('Sentiment Analysis For Your Playlists',
             href='/page-2', style = {'font-size': '35px'}),

    html.Div([html.Img(src = '/assets/background_image.jpg', height = '500', width = '37%',
                       style={'margin-left':450})]),
    html.Div([html.H5('Developed by Badal Nabizade - nabizadebadal@gmail.com')], style = {'color':'black','text-align':'center'})
    ], style = {'backgroundColor':'white', 'heigt':'100%', 'width':'100%'})

page_1_layout = html.Div([
    html.H1('Moodify'),
    # html.Div(children='''
    #     Symbol to graph:
    # '''),
    html.Div(dcc.Input(id='input',placeholder='Your favorite Spotify artist', value='', type='text')),
    html.Button('Plot Tracks', id = 'button', className = 'row'),
    html.Div(id='output-graph'),
    html.Div(id='page-1-content'),
    html.Br(),
    dcc.Link('Sentiment Analysis For Your Playlists', href='/page-2'),
    html.Br(),
    dcc.Link('Go back to home', href='/'),

])

@app.callback(
    Output(component_id='output-graph', component_property='children'),
    [Input(component_id='button', component_property='n_clicks')],
    [State(component_id = 'input', component_property = 'value')])

def plot_artist_songs(n_clicks, input_data):
    artist = get_artist(input_data)
    albums = get_artist_albums(artist)

    tracks = []
    tracks_and_albums = []
    track_dct_list = list()
    for i in albums.values():
        test = {v:k for k,v in albums.items()}
        tracks.extend(get_album_tracks(i).keys())
        tracks_and_albums.append((list(get_album_tracks(i).keys()), test[i]))
        track_dct_list.append(get_album_tracks(i))

    audio_feats = []

    for i in range(0, 1001, 100):
        audio_feats.append(sp.audio_features(tracks[i:i+100]))

    df_list = []
    index = 0
    for i in range(len(audio_feats)):
        if audio_feats[i][0]:
            #print(i)
            for a in range(len(audio_feats[i])):
                index+=1
                df_list.append(pd.DataFrame(audio_feats[i][a], index=[index]))

    df = pd.concat(df_list)

    def album_finder(track_id):
        for i in range(len(tracks_and_albums)):
            for a in tracks_and_albums[i][0]:
                if a == track_id:
                    return tracks_and_albums[i][1]

    df['album'] = df['id'].apply(album_finder)

    names = []
    ids = []
    for i in range(len(track_dct_list)):
        for a in track_dct_list[i].keys():
            if a in list(df['id'].values):
                names.append(track_dct_list[i][a])
                ids.append(a)

    track_names = pd.DataFrame({'name':names, 'id':ids})

    final_df = df.merge(track_names, left_on='id', right_on='id')
    final_df.drop(['type', 'key', 'uri', 'track_href', 'analysis_url',
                   'time_signature', 'mode', 'duration_ms'], 1, inplace=True)


    trace0 = go.Scatter(
        x=[0.92, 0.05, 0.07, 0.92],
        y=[0.92, 0.05, 0.95, 0.05],
        text=['Happy', 'Sad', 'Angry', 'Relaxed'],
        mode='text',
        name = 'Categories',
        textfont = dict(
                        family='Old Standard TT, serif',
                        size=25,
                        color= 'rgba(0, 0, 0, 0.70)'
                                )
                            )

    trace1 = [go.Scatter(x=final_df[final_df['album'] == i]['valence'],
                y=final_df[final_df['album'] == i]['energy'],
                text=final_df[final_df['album'] == i]['name'],
                mode='markers',
                opacity=0.7,
                marker={
                'size': 15,
                'line': {'width': 0.5, 'color': 'white'}
                },
                name=i
                ) for i in final_df.album.unique()]
    trace1.append(trace0)
    return html.Div([dcc.Graph(id = 'example',
              figure = {
                        'data':trace1,

                        'layout': {'clickmode': 'event+select',
                                   'height':650,
                                   # 'width':950,
                                   # 'margin-left':450,
                                   'shapes': [
                                              {'type':'line',
                                               'x0':0.5,
                                               'y0': 0,
                                               'x1':0.5,
                                               'y1':1},

                                              {'type':'line',
                                               'x0': 0,
                                               'y0':0.5,
                                               'x1':1,
                                               'y1':0.5}
                                   ]}
              })], className='container', style={'maxWidth': '1000px'})

data_list = ["User Songs", "Today's Hot Hits"]
page_2_layout = html.Div([
    html.Div([
        html.H2('Moodify',
                style={'float': 'left'}),
        ]),
    dcc.Input(id='input',placeholder='Paste Your Spotify Profile Link...', value='', type='text'),
    html.Button('Plot Tracks', id = 'button', className = 'row', style = {'color':'rgba(186, 8, 8, 1)'}),
    html.H5("""To compare your tracks wtih "Today's Hot Hits" playlist, Select Today's Hot Hits also.""" ),
    dcc.Dropdown(id='dropdown-values',
                 options=[{'label': s, 'value': s}
                          for s in data_list],
                 value=['User Songs'],
                 multi=True
                 ),
    html.Div(children=html.Div(id='graphs'), className='row'),
        #html.H6("Would", style={'float': 'middle'}),
        html.Div([html.H5('Would you like to see top 3 distinct song ?'),
        html.Button(
        'Show Those', id = 'button-yes',
        value = 'yes', className = 'row',
        style = {
        'color':'rgba(186, 8, 8, 1)',
        'margin-left':115
        }
        ),
        ],style={'width':'100%','margin-left':550,'margin-right':10,'max-width':50000}),
        html.Br(),
        dcc.Link('Go back to home', href='/')
    ], className="container",style={'width':'98%','margin-left':10,'margin-right':10,'max-width':50000})

@app.callback(
    Output(component_id='graphs', component_property='children'),
    [Input(component_id='button', component_property='n_clicks'),
    Input('dropdown-values', 'value'),
    Input('button-yes', 'n_clicks')],
    [State(component_id = 'input', component_property = 'value')])

def plot_songs(n_clicks, dropdown_value, button_yes, input_value):
    input_value = input_value.split('user/')[1].split('?')[0]
    playlists = find_user_playlists(input_value)

    track_dict_list = []
    artist_dict_list = []
    tracks_and_playlists = []
    for playlist in playlists.values():
        test = {v:k for k,v in playlists.items()}

        playlist_tracks = sp.user_playlist_tracks(user=input_value, playlist_id=playlist)
        artist_dict, track_dict = show_user_tracks(playlist_tracks)
        track_dict_list.append(track_dict)
        artist_dict_list.append(artist_dict)
        tracks_and_playlists.append((list(track_dict.keys()), test[playlist]))
        while playlist_tracks['next']:

            playlist_tracks = sp.next(playlist_tracks)
            artist_dict, track_dict = show_user_tracks(playlist_tracks)
            track_dict_list.append(track_dict)
            artist_dict_list.append(artist_dict)
            tracks_and_playlists.append((list(track_dict.keys()), test[playlist]))

    audio_feats = []
    for i in range(len(track_dict_list)):
        audio_feats.append(sp.audio_features(list(track_dict_list[i].keys())))

    df_list = []
    index = 0
    for i in range(len(audio_feats)):
        if audio_feats[i][0]:
            #print(i)
            for a in range(len(audio_feats[i])):
                index+=1
                df_list.append(pd.DataFrame(audio_feats[i][a], index=[index]))

    df = pd.concat(df_list)

    def playlist_finder(track_id):
        for i in range(len(tracks_and_playlists)):
            for a in tracks_and_playlists[i][0]:
                if a == track_id:
                    return tracks_and_playlists[i][1]

    df['playlist'] = df['id'].apply(playlist_finder)

    titles = []
    ids = []
    for i in range(len(track_dict_list)):
        for a in track_dict_list[i].keys():
            if a in list(df['id'].values):
                titles.append(track_dict_list[i][a])
                ids.append(a)

    track_titles = pd.DataFrame({'name':titles, 'id':ids})

    names = []
    ids = []
    for i in range(len(artist_dict_list)):
        for a in artist_dict_list[i].keys():
            if a in list(df['id'].values):
                names.append(artist_dict_list[i][a])
                ids.append(a)

    artist_names = pd.DataFrame({'artist_name':names, 'song_id':ids})

    first_df = df.merge(track_titles, left_on='id', right_on='id')
    first_df.drop(['type', 'key', 'uri', 'track_href', 'analysis_url', 'time_signature', 'mode', 'duration_ms'], 1, inplace=True)
    user_songs_df =first_df.merge(artist_names, left_on='id', right_on='song_id')

    user_songs_df = duplicate_dropper(user_songs_df, 'id')

    user_songs_df.drop(['id'],1,inplace=True)

    user_songs_df['mood'] = list(map(mood_detector, user_songs_df['valence'], user_songs_df['energy']))

    th_df = todays_hot_hits()
    th_df['mood'] = list(map(mood_detector, th_df['valence'], th_df['energy']))

    if len(dropdown_value) < 2:
        class_choice = 'col s12'
        data_dict = {dropdown_value[0]:user_songs_df['mood']}
    else:
        class_choice = 'col s12 m6 l6'
        data_dict = {dropdown_value[0]:user_songs_df['mood'],
                     dropdown_value[1]:th_df['mood']}

    graphs = []
    for val in dropdown_value:

        data = [go.Bar(
            x= data_dict[val].value_counts().index,
            y=data_dict[val].value_counts().values,
            marker=dict(
            color=[
                   'rgba(186, 8, 8, 1)',
                   'rgba(0,0,128,0.9)',
                   'rgba(246, 148, 30, 1)',
                   'rgba(0,255,0,1)'
                        ]
                      )
                   )]
        graphs.append(html.Div(dcc.Graph(
            id=val,
            animate=True,
            figure={'data': data,'layout' : go.Layout(
                                                        # margin={'l':50,'r':1,'t':45,'b':0},
                                                        title='{}'.format(val))}
            ), className=class_choice))

    trace0 = go.Scatter(
                        x=[0.9, 0.1, 0.1, 0.9],
                        y=[0.9, 0.1, 0.9, 0.1],
                        text=['Happy', 'Sad', 'Angry', 'Relaxed'],
                        mode='text',
                        name = 'Categories',
                        textfont = dict(
                                        family='Old Standard TT, serif',
                                        size=25,
                                        color='rgba(0, 0, 0, 0.68)'
                                                )
                                            )

    trace1 = [go.Scatter(
                x=user_songs_df[user_songs_df['playlist'] == i]['valence'],
                y=user_songs_df[user_songs_df['playlist'] == i]['energy'],
                text=user_songs_df[user_songs_df['playlist'] == i]['name'],
                mode='markers',
                opacity=0.7,
                marker={
                'size': 15,
                'line': {'width': 0.5, 'color': 'white'}
                },
                name=i
                ) for i in user_songs_df.playlist.unique()]
    trace1.append(trace0)

    graphs.append(html.Div(dcc.Graph(
        id='scatter',
        animate=True,
        figure={'data': trace1,
                'layout': {'clickmode': 'event+select',
                           'shapes': [
                                      {'type':'line',
                                       'x0':0.5,
                                       'y0': 0,
                                       'x1':0.5,
                                       'y1':1},

                                      {'type':'line',
                                       'x0': 0,
                                       'y0':0.5,
                                       'x1':1,
                                       'y1':0.5}
                                                    ]}}
                                                        ), className=class_choice))


    if button_yes:
        pca_df = user_songs_df.drop(['name','artist_name', 'song_id', 'mood', 'playlist'],1)
        pca_df = pca_df.dropna()
        pca_df_norm = (pca_df - pca_df.mean()) / pca_df.std()
        # PCA
        pca = PCA(n_components=2, svd_solver='auto', iterated_power='auto')
        pca_res = pca.fit_transform(pca_df_norm.values)

        z1 = pca_res[:,0]
        z2 = pca_res[:,1]

        forest = IsolationForest(n_estimators=1200, behaviour='new', contamination='auto')
        preds = pd.Series(forest.fit_predict(pca_df))
        pca_df.reset_index(drop=False, inplace=True)
        outliers = preds[preds == -1].index
        outlier_data_points = []
        for i in range(len(outliers)):
            outlier_data_points.append(pca_df.drop('index',1).loc[outliers[i]].values.reshape(1,-1).ravel())

        outlier_dict = {}
        for i,v in enumerate(forest.decision_function(np.array(outlier_data_points))):
            outlier_dict[v] = i

        indicies = []
        for i in sorted(outlier_dict.keys())[:3]:
            indicies.append(outliers[outlier_dict[i]])

        n = user_songs_df.loc[pca_df.loc[indicies]['index'].values]['name']

        pca_df = pd.DataFrame({'z1':z1, 'z2':z2, 'name':user_songs_df['name']}, index=pca_df['index'])

        trace3 = [go.Scatter(
                            x = pca_df['z1'],
                            y = pca_df['z2'],
                            text = pca_df['name'],
                            mode = 'markers',
                            name='tracks'
                            )]

        trace2 = go.Scatter(
                            x = pca_df.loc[n.index]['z1'],
                            y = pca_df.loc[n.index]['z2'],
                            text = pca_df.loc[n.index]['name'],
                            mode = 'markers',
                            marker = {'size':10,
                                      'color':'rgba(255, 182, 193, .9)'},
                            name='Distinct Tracks'
                            )
        trace3.append(trace2)

        graphs.append((html.Div(dcc.Graph(
            id='new',
            animate=True,
            figure={'data': trace3,'layout' : go.Layout(clickmode= 'event+select',
                                                        # margin={'l':50,'r':1,'t':45,'b':0},
                                                        title="3 Most Different Tracks from User's General Music Taste.<br>(Note: This graph generated by PCA and Anomaly Detection Algorithm.<br>X and Y axis are two dimensons that provides highest variance about data.<br>Pink data points labaled as anomaly by anomaly detection algorithm.)",
                                                        titlefont= dict(size = 14))}
            ), className=class_choice)))

    return graphs

external_css = ["https://cdnjs.cloudflare.com/ajax/libs/materialize/0.100.2/css/materialize.min.css"]
for css in external_css:
    app.css.append_css({"external_url": css})

external_js = ['https://cdnjs.cloudflare.com/ajax/libs/materialize/0.100.2/js/materialize.min.js']
for js in external_css:
    app.scripts.append_script({'external_url': js})
# Update the index
@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/page-1':
        return page_1_layout
    elif pathname == '/page-2':
        return page_2_layout
    else:
        return index_page


if __name__ == '__main__':
    app.server.run(debug=True, threaded=True)
