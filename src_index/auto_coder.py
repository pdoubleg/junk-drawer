import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, PartOfSpeech, MaximalMarginalRelevance
import tqdm
import plotly.express as px
import plotly.graph_objects as go
import datetime



def prepare_data(df, text_col):
    docs = list(df[text_col])
    main_representation_model = KeyBERTInspired()
    aspect_representation_model1 = PartOfSpeech("en_core_web_lg")
    aspect_representation_model2 = [KeyBERTInspired(top_n_words=10), MaximalMarginalRelevance(diversity=.5)]
    representation_model = {
       "Main": main_representation_model,
       "Aspect1":  aspect_representation_model1,
       "Aspect2":  aspect_representation_model2 
    }
    vectorizer_model = CountVectorizer(min_df=5, stop_words = 'english')
    topic_model = BERTopic(
       vectorizer_model = vectorizer_model,
       representation_model = representation_model)
    
    topics, _ = topic_model.fit_transform(docs)
    
    df = df.join(topic_model.get_document_info(docs))
    
    topic_distr, _ = topic_model.approximate_distribution(docs, window = 4, calculate_tokens=True)
    
    return topic_model, topic_distr, df



def calculate_topic_distance(topic_model, top_n):
    from sklearn.metrics.pairwise import cosine_similarity

    distance_matrix = cosine_similarity(np.array(topic_model.topic_embeddings_))
    dist_df = pd.DataFrame(distance_matrix, columns=topic_model.topic_labels_.values(), 
                           index=topic_model.topic_labels_.values())

    tmp = []
    for rec in dist_df.reset_index().to_dict('records'):
        t1 = rec['index']
        for t2 in rec:
            if t2 == 'index': 
                continue
            tmp.append(
                {
                    'topic1': t1, 
                    'topic2': t2, 
                    'distance': rec[t2]
                }
            )

    pair_dist_df = pd.DataFrame(tmp)

    pair_dist_df = pair_dist_df[(pair_dist_df.topic1.map(
          lambda x: not x.startswith('-1'))) & 
                (pair_dist_df.topic2.map(lambda x: not x.startswith('-1')))]
    pair_dist_df = pair_dist_df[pair_dist_df.topic1 < pair_dist_df.topic2]
    return pair_dist_df.sort_values('distance', ascending = False).head(top_n)




def calculate_and_plot_thresholds(topic_distr):

    tmp_dfs = []

    # iterating through different threshold levels
    for thr in tqdm.tqdm(np.arange(0, 0.35, 0.001)):
        # calculating number of topics with probability > threshold for each document
        tmp_df = pd.DataFrame(list(map(lambda x: len(list(filter(lambda y: y >= thr, x))), topic_distr))).rename(
            columns = {0: 'num_topics'}
        )
        tmp_df['num_docs'] = 1
        
        tmp_df['num_topics_group'] = tmp_df['num_topics']\
            .map(lambda x: str(x) if x < 5 else '5+')
        
        # aggregating stats
        tmp_df_aggr = tmp_df.groupby('num_topics_group', as_index = False).num_docs.sum()
        tmp_df_aggr['threshold'] = thr
        
        tmp_dfs.append(tmp_df_aggr)

    num_topics_stats_df = pd.concat(tmp_dfs).pivot(index = 'threshold', 
                                values = 'num_docs',
                                columns = 'num_topics_group').fillna(0)

    num_topics_stats_df = num_topics_stats_df.apply(lambda x: 100.*x/num_topics_stats_df.sum(axis = 1))

    # visualisation
    colormap = px.colors.sequential.YlGnBu
    return px.area(num_topics_stats_df, 
            title = 'Distribution of number of topics',
            labels = {'num_topics_group': 'number of topics',
                        'value': 'share of reviews, %'},
            color_discrete_map = {
                '0': colormap[0],
                '1': colormap[3],
                '2': colormap[4],
                '3': colormap[5],
                '4': colormap[6],
                '5+': colormap[7]
            })


def assign_multi_topics(df, topic_distr, threshold):
    df['multiple_topics'] = list(map(
        lambda doc_topic_distr: list(map(
            lambda y: y[0], filter(lambda x: x[1] >= threshold, 
                                   (enumerate(doc_topic_distr)))
        )), topic_distr
    ))
    return df



def prepare_topic_df(df, cat_cols):
    tmp_data = []
    for rec in df.to_dict('records'):
        if len(rec['multiple_topics']) != 0:
            mult_topics = rec['multiple_topics']
        else:
            mult_topics = [-1]
        for topic in mult_topics: 
            tmp_data.append(
                {
                    'topic': topic,
                    **{col: rec[col] for col in cat_cols}
                }
            )
    mult_topics_df = pd.DataFrame(tmp_data)
    return mult_topics_df



def calculate_topic_stats(mult_topics_df, cat_cols):
    tmp_data = []
    for cat in mult_topics_df[cat_cols[0]].unique():
        for topic in mult_topics_df.topic.unique():
            tmp_data.append({
                cat_cols[0]: cat,
                'topic_id': topic,
                f'total_{cat}_reviews': mult_topics_df[mult_topics_df[cat_cols[0]] == cat].id.nunique(),
                f'topic_{cat}_reviews': mult_topics_df[(mult_topics_df[cat_cols[0]] == cat) 
                                                      & (mult_topics_df.topic == topic)].id.nunique(),
                f'other_{cat}_reviews': mult_topics_df[mult_topics_df[cat_cols[0]] != cat].id.nunique(),
                f'topic_{cat}_cats_reviews': mult_topics_df[(mult_topics_df[cat_cols[0]] != cat) 
                                                      & (mult_topics_df.topic == topic)].id.nunique()
            })
    mult_topics_stats_df = pd.DataFrame(tmp_data)
    mult_topics_stats_df['topic_cat_share'] = 100*mult_topics_stats_df.topic_cat_reviews/mult_topics_stats_df.total_cat_reviews
    mult_topics_stats_df['topic_other_cats_share'] = 100*mult_topics_stats_df.topic_other_cats_reviews/mult_topics_stats_df.other_cats_reviews
    return mult_topics_stats_df



def calculate_pval(mult_topics_stats_df):
    from statsmodels.stats.proportion import proportions_ztest
    mult_topics_stats_df['difference_pval'] = list(map(
        lambda x1, x2, n1, n2: proportions_ztest(
            count = [x1, x2],
            nobs = [n1, n2],
            alternative = 'two-sided'
        )[1],
        mult_topics_stats_df.topic_other_cats_reviews,
        mult_topics_stats_df.topic_cat_reviews,
        mult_topics_stats_df.other_cats_reviews,
        mult_topics_stats_df.total_cat_reviews
    ))
    mult_topics_stats_df['sign_difference'] = mult_topics_stats_df.difference_pval.map(
        lambda x: 1 if x <= 0.05 else 0
    )
    return mult_topics_stats_df

def get_significance(d, sign):
    sign_percent = 1
    if sign == 0:
        return 'no diff'
    if (d >= -sign_percent) and (d <= sign_percent):
        return 'no diff'
    if d < -sign_percent:
        return 'lower'
    if d > sign_percent:
        return 'higher'

def calculate_significance(mult_topics_stats_df):
    mult_topics_stats_df['diff_significance_total'] = list(map(
        get_significance,
        mult_topics_stats_df.topic_cat_share - mult_topics_stats_df.topic_other_cats_share,
        mult_topics_stats_df.sign_difference
    ))
    return mult_topics_stats_df

def get_color_sign(rel):
    if rel == 'no diff':
        return '#66c2a5'
    if rel == 'lower':
        return '#fc8d62'
    if rel == 'higher':
        return '#8da0cb'

def get_topic_representation_title(topic_model, topic):
    data = topic_model.get_topic(topic)
    data = list(map(lambda x: x[0], data))
    return ', '.join(data[:5]) + ', <br>         ' + ', '.join(data[5:])

def get_graphs_for_topic(topic_model, mult_topics_stats_df, t):
    topic_stats_df = mult_topics_stats_df[mult_topics_stats_df.topic_id == t]\
        .sort_values('total_cat_reviews', ascending = False).set_index('State')
    colors = list(map(
        get_color_sign,
        topic_stats_df.diff_significance_total
    ))
    fig = go.Figure(data=[go.Bar(
        x=topic_stats_df.reset_index()['State'], 
        y=topic_stats_df['topic_cat_share'],
        marker_color=colors
    )])
    fig.update_layout(
        title_text='Topic: %s' % get_topic_representation_title(topic_model, topic_stats_df.topic_id.min()),
        xaxis_title_text='State',
        yaxis_title_text='share of reviews, %',
        bargap=0.2,
        bargroupgap=0.1
    )
    topic_total_share = 100.*((topic_stats_df.topic_cat_reviews + topic_stats_df.topic_other_cats_reviews)\
        /(topic_stats_df.total_cat_reviews + topic_stats_df.other_cats_reviews)).min()
    fig.add_shape(type="line",
        xref="paper",
        x0=0, y0=topic_total_share,
        x1=1, y1=topic_total_share,
        line=dict(
            color='#b3b3b3',
            width=3, dash="dot"
        )
    )
    fig.show()

def get_topic_representation(topic_model, topic):
    data = topic_model.get_topic(topic)
    return ', '.join(list(map(lambda x: x[0], data)))

def assign_topic_representation(df, topic_model):
    df['merged_topic'] = topic_model.topics_
    df['merged_topic_repr'] = df['merged_topic'].map(lambda x: get_topic_representation(topic_model, x))
    return df

def calculate_top_topics(mult_topics_df, topic_model):
    top_mult_topics_df = mult_topics_df.groupby('topic', as_index = False).id.nunique()
    top_mult_topics_df['share'] = 100.*top_mult_topics_df.id/top_mult_topics_df.id.sum()
    top_mult_topics_df['topic_repr'] = top_mult_topics_df.topic.map(
        lambda x: get_topic_representation(topic_model, x)
    )
    return top_mult_topics_df

def plot_top_topics(top_mult_topics_df, topic_model):
    for t in top_mult_topics_df.head(32).topic.values:
        get_graphs_for_topic(topic_model, t)
