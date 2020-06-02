import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn import datasets, linear_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from nltk.stem import PorterStemmer
import nltk
from sklearn.metrics import r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


class VideoManager:
    def __init__(self):
        self.data_root = '../datasets/'
        self.countries = ['CA', 'DE', 'FR', 'GB', 'US']
        self.full_name = {'CA': 'Canada', 'DE': 'Germany',
                          'FR': 'France', 'GB': 'Great British', 'US': 'USA'}
        self.cat_dict = self.get_cat_dict()

    def load_data(self):
        
        youtube = pd.read_csv("../datasets/CAvideos.csv")
        dataset = {}
        for country in self.countries:
            file_path = self.data_root + country + 'videos.csv'
            raw_df = pd.read_csv(file_path)
            dataset[country] = raw_df
        self.dataset = dataset
        
            
    def predict_likes(self):
        youtube = pd.read_csv("../datasets/CAvideos.csv")
        likes=youtube['likes']
        youtube_like=youtube.drop(['likes'],axis=1,inplace=False)
        youtube_like = youtube_like.apply(pd.to_numeric, errors='coerce')
        likes = likes.apply(pd.to_numeric, errors='coerce')
        youtube_like = youtube_like.replace(np.nan, 0, regex=True)
        likes = likes.replace(np.nan, 0, regex=True)
        X_train,X_test,y_train,y_test=train_test_split(youtube_like,likes, test_size=0.2,shuffle=False)
        print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
        
        model = linear_model.LinearRegression()
        model.fit(X_train, y_train)

        # predicting the  test set results
        y_pred = model.predict(X_test)
        print('Root means score', np.sqrt(mean_squared_error(y_test, y_pred)))
        print('Variance score: %.2f' % r2_score(y_test, y_pred))
        print("Result :",model.score(X_test, y_test))
        d1 = {'True Labels': y_test, 'Predicted Labels': y_pred}
        SK = pd.DataFrame(data = d1)
        print(SK)
        lm1 = sns.lmplot(x="True Labels", y="Predicted Labels", data = SK, height = 10)
        fig1 = lm1.fig 
        fig1.suptitle("Sklearn ", fontsize=18)
        sns.set(font_scale = 1.5)

    def random_forest(self):
    
        print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
        nEstimator = [140,160,180,200,220]
        depth = [10,15,20,25,30]
        
        RF = RandomForestRegressor()
        hyperParam = [{'n_estimators':nEstimator,'max_depth': depth}]
        gsv = GridSearchCV(RF,hyperParam,cv=5,verbose=1,scoring='r2',n_jobs=-1)
        gsv.fit(X_train, y_train)
        print("Best HyperParameter: ",gsv.best_params_)
        print(gsv.best_score_)
        scores = gsv.cv_results_['mean_test_score'].reshape(len(nEstimator),len(depth))
        plt.figure(figsize=(8, 8))
        plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
        plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot)
        plt.xlabel('n_estimators')
        plt.ylabel('max_depth')
        plt.colorbar()
        plt.xticks(np.arange(len(nEstimator)), nEstimator)
        plt.yticks(np.arange(len(depth)), depth)
        plt.title('Grid Search r^2 Score')
        plt.show()
        maxDepth=gsv.best_params_['max_depth']
        nEstimators=gsv.best_params_['n_estimators']

        model = RandomForestRegressor(n_estimators = nEstimators,max_depth=maxDepth)
        model.fit(X_train, y_train)
        
        
        # predicting the  test set results
        y_pred = model.predict(X_test)
        print('Root means score', np.sqrt(mean_squared_error(y_test, y_pred)))
        print('Variance score: %.2f' % r2_score(y_test, y_pred))
        print("Result :",model.score(X_test, y_test))
        d1 = {'True Labels': y_test, 'Predicted Labels': y_pred}
        SK = pd.DataFrame(data = d1)
        print(SK)
        
        lm1 = sns.lmplot(x="True Labels", y="Predicted Labels", data = SK, height = 10)
        fig1 = lm1.fig 
        fig1.suptitle("Sklearn ", fontsize=18)
        sns.set(font_scale = 1.5)
    
    def predict_views(self):
        youtube = pd.read_csv("../datasets/CAvideos.csv")
        views=youtube['views']
        youtube_view=youtube.drop(['views'],axis=1,inplace=False)
        youtube_view = youtube_view.apply(pd.to_numeric, errors='coerce')
        views = views.apply(pd.to_numeric, errors='coerce')
        youtube_view = youtube_view.replace(np.nan, 0, regex=True)
        views = views.replace(np.nan, 0, regex=True)
        X_train,X_test,y_train,y_test=train_test_split(youtube_view,views, test_size=0.2,shuffle=False)
        print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
        
        model = linear_model.LinearRegression()
        model.fit(X_train, y_train)

        # predicting the  test set results
        y_pred = model.predict(X_test)
        print('Root means score', np.sqrt(mean_squared_error(y_test, y_pred)))
        print('Variance score: %.2f' % r2_score(y_test, y_pred))
        print("Result :",model.score(X_test, y_test))
        d1 = {'True Labels': y_test, 'Predicted Labels': y_pred}
        SK = pd.DataFrame(data = d1)
        print(SK)
        lm1 = sns.lmplot(x="True Labels", y="Predicted Labels", data = SK, height = 10)
        fig1 = lm1.fig 
        fig1.suptitle("Sklearn ", fontsize=18)
        sns.set(font_scale = 1.5)

    def random_forest(self):
    
        print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
        nEstimator = [140,160,180,200,220]
        depth = [10,15,20,25,30]
        
        RF = RandomForestRegressor()
        hyperParam = [{'n_estimators':nEstimator,'max_depth': depth}]
        gsv = GridSearchCV(RF,hyperParam,cv=5,verbose=1,scoring='r2',n_jobs=-1)
        gsv.fit(X_train, y_train)
        print("Best HyperParameter: ",gsv.best_params_)
        print(gsv.best_score_)
        scores = gsv.cv_results_['mean_test_score'].reshape(len(nEstimator),len(depth))
        plt.figure(figsize=(8, 8))
        plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
        plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot)
        plt.xlabel('n_estimators')
        plt.ylabel('max_depth')
        plt.colorbar()
        plt.xticks(np.arange(len(nEstimator)), nEstimator)
        plt.yticks(np.arange(len(depth)), depth)
        plt.title('Grid Search r^2 Score')
        plt.show()
        maxDepth=gsv.best_params_['max_depth']
        nEstimators=gsv.best_params_['n_estimators']

        model = RandomForestRegressor(n_estimators = nEstimators,max_depth=maxDepth)
        model.fit(X_train, y_train)
        
        
        # predicting the  test set results
        y_pred = model.predict(X_test)
        print('Root means score', np.sqrt(mean_squared_error(y_test, y_pred)))
        print('Variance score: %.2f' % r2_score(y_test, y_pred))
        print("Result :",model.score(X_test, y_test))
        d1 = {'True Labels': y_test, 'Predicted Labels': y_pred}
        SK = pd.DataFrame(data = d1)
        print(SK)
        
        lm1 = sns.lmplot(x="True Labels", y="Predicted Labels", data = SK, height = 10)
        fig1 = lm1.fig 
        fig1.suptitle("Sklearn ", fontsize=18)
        sns.set(font_scale = 1.5)
    
    
    def predict_comments(self):
        youtube = pd.read_csv("../datasets/CAvideos.csv")
        comment_count=youtube['comment_count']
        youtube_comment=youtube.drop(['comment_count'],axis=1,inplace=False)
        youtube_comment = youtube_comment.apply(pd.to_numeric, errors='coerce')
        comment_count = comment_count.apply(pd.to_numeric, errors='coerce')
        youtube_comment = youtube_comment.replace(np.nan, 0, regex=True)
        comment_count = comment_count.replace(np.nan, 0, regex=True)
        X_train,X_test,y_train,y_test=train_test_split(youtube_comment,comment_count, test_size=0.2,shuffle=False)
        print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
        
        model = linear_model.LinearRegression()
        model.fit(X_train, y_train)

        # predicting the  test set results
        y_pred = model.predict(X_test)
        print('Root means score', np.sqrt(mean_squared_error(y_test, y_pred)))
        print('Variance score: %.2f' % r2_score(y_test, y_pred))
        print("Result :",model.score(X_test, y_test))
        d1 = {'True Labels': y_test, 'Predicted Labels': y_pred}
        SK = pd.DataFrame(data = d1)
        print(SK)
        lm1 = sns.lmplot(x="True Labels", y="Predicted Labels", data = SK, height = 10)
        fig1 = lm1.fig 
        fig1.suptitle("Sklearn ", fontsize=18)
        sns.set(font_scale = 1.5)

        def random_forest(self):
    
        print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
        nEstimator = [140,160,180,200,220]
        depth = [10,15,20,25,30]
        
        RF = RandomForestRegressor()
        hyperParam = [{'n_estimators':nEstimator,'max_depth': depth}]
        gsv = GridSearchCV(RF,hyperParam,cv=5,verbose=1,scoring='r2',n_jobs=-1)
        gsv.fit(X_train, y_train)
        print("Best HyperParameter: ",gsv.best_params_)
        print(gsv.best_score_)
        scores = gsv.cv_results_['mean_test_score'].reshape(len(nEstimator),len(depth))
        plt.figure(figsize=(8, 8))
        plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
        plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot)
        plt.xlabel('n_estimators')
        plt.ylabel('max_depth')
        plt.colorbar()
        plt.xticks(np.arange(len(nEstimator)), nEstimator)
        plt.yticks(np.arange(len(depth)), depth)
        plt.title('Grid Search r^2 Score')
        plt.show()
        maxDepth=gsv.best_params_['max_depth']
        nEstimators=gsv.best_params_['n_estimators']

        model = RandomForestRegressor(n_estimators = nEstimators,max_depth=maxDepth)
        model.fit(X_train, y_train)
        
        
        # predicting the  test set results
        y_pred = model.predict(X_test)
        print('Root means score', np.sqrt(mean_squared_error(y_test, y_pred)))
        print('Variance score: %.2f' % r2_score(y_test, y_pred))
        print("Result :",model.score(X_test, y_test))
        d1 = {'True Labels': y_test, 'Predicted Labels': y_pred}
        SK = pd.DataFrame(data = d1)
        print(SK)
        
        lm1 = sns.lmplot(x="True Labels", y="Predicted Labels", data = SK, height = 10)
        fig1 = lm1.fig 
        fig1.suptitle("Sklearn ", fontsize=18)
        sns.set(font_scale = 1.5)
    
    def correlation_matrix(self):
        youtube = pd.read_csv("../datasets/CAvideos.csv")
        data = youtube
        corr = data.corr()
        plt.figure(figsize=(12, 12))
        ax = sns.heatmap(
            corr, 
            vmin=-1, vmax=1, center=0,
            cmap=sns.diverging_palette(20, 220, n=200),
            square=True
        )
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=45,
            horizontalalignment='right'
        );
        
    

        
    def get_cat_dict(self):
        cat_dict = {}
        for country in self.countries:
            with open(self.data_root + country + '_category_id.json', 'r') as file:
                json_obj = json.load(file)
            for item in json_obj['items']:
                cat_dict[int(item['id'])] = item['snippet']['title']
        return cat_dict

    def total_views(self):
                result_view = pd.DataFrame()
        for area in self.countries:
                       df_view = pd.DataFrame(self.dataset[area][['category_id', 'views']]).groupby('category_id')[
                'views'].sum().rename('Views in Different Areas').reset_index()
            df_view['Areas'] = self.full_name[area]
            result_view = result_view.append(df_view)

        result_view.insert(loc=1, column='Categories',
                           value=result_view.category_id.map(lambda x: self.cat_dict[x]))

        result_view['Total Views'] = result_view.groupby(
            'category_id')['Views in Different Areas'].transform('sum')

        result_view = result_view.sort_values(
            by='Total Views', ascending=False).reset_index(drop=True).reset_index()

        sns.set(font_scale=1.3)
        sbplt_fig_view, sbplt_ax_view = plt.subplots()

        fig_view1 = sns.catplot(x="Categories", y="Views in Different Areas", data=result_view.head(25), hue="Areas",
                                hue_order=['Canada', 'France',
                                           'Germany', 'Great British', 'USA'],
                                kind="bar", palette="muted", edgecolor="1", alpha=0.85, legend_out=False,
                                ax=sbplt_ax_view)

        sbplt_ax_view2 = sbplt_ax_view.twinx()
        fig_view2 = sns.catplot(x="Categories", y="Total Views", data=result_view.head(25),
                                kind='point', color="b", ax=sbplt_ax_view2)

        sbplt_ax_view.set_title(
            "Views in different categories and areas", fontsize=30)
        sbplt_ax_view.grid(None)

    def videos_time(self):
        dic = {}
        for e in self.dataset:
            p_time = [int(x[11:13])
                      for x in list(self.dataset[e].publish_time)]
            video_counter = [0] * 24
            for j in p_time:  
                video_counter[j] += 1
            dic[e] = video_counter 
        frame = pd.DataFrame(dic)
        sns.set(font_scale=1.5)
        sns.set(style="whitegrid")
        fig = sns.lineplot(data=frame, palette="tab10", linewidth=1.5)
        fig.set_title('Publishing Videos in 24 Hours')
        fig.set_xlabel('Hours')
        fig.set_ylabel('Publishing Videos')
        plt.xticks(np.arange(0, 24, 1))
        plt.yticks(np.arange(0, 7000, 1000))

    def comment_views(self):
        dataframe = pd.DataFrame()
        for country in self.countries:
            raw_df = self.dataset[country][[
                'category_id', 'views', 'comment_count']]
            df = raw_df.groupby('category_id')[
                ['views', 'comment_count']].sum().reset_index()
            df['Comment/View Ratio'] = df['comment_count'] / df['views']
            df['Country'] = self.full_name[country]
            dataframe = dataframe.append(df)

        dataframe['Total Ratio'] = dataframe.groupby(
            'category_id')['Comment/View Ratio'].transform('sum')
        dataframe.insert(loc=1, column='Category', value=dataframe.category_id.map(
            lambda id: self.cat_dict[id]))
        dataframe = dataframe.sort_values(
            by='Total Ratio', ascending=False).reset_index(drop=True).reset_index()

        sns.set(font_scale=0.8)
        fig1, ax1 = plt.subplots()
        cat_count = 10
        hist_count = cat_count * 5

        fig_area = sns.catplot(x="Category", y="Comment/View Ratio", data=dataframe.head(hist_count),
                               hue="Country",
                               hue_order=['Canada', 'France',
                                          'Germany', 'Great British', 'USA'],
                               kind="bar", palette="muted", edgecolor="1", alpha=0.85, legend_out=False, ax=ax1)

        ax2 = ax1.twinx()
        fig_total = sns.catplot(x="Category", y="Total Ratio", data=dataframe.head(hist_count),
                                kind='point', color="b", ax=ax2)

        ax1.set_title("Comment/View Ratio of Different Genres and Countries", fontsize=30)
         ax1.set_xticklabels(rotation=30)
        ax1.grid(None)

    def likes(self):
        dataframe = pd.DataFrame()
        for country in self.countries:
            raw_df = self.dataset[country][[
                'category_id', 'likes', 'dislikes']]
            df = raw_df.groupby('category_id')[
                ['likes', 'dislikes']].sum().reset_index()
            df['Likes/Dislikes Ratio'] = df['likes'] / df['dislikes']
            df['Country'] = self.full_name[country]
            dataframe = dataframe.append(df)

        dataframe['Average_Ratio'] = dataframe.groupby(
            'category_id')['Likes/Dislikes Ratio'].transform('sum') / 5
        dataframe.insert(loc=1, column='Category', value=dataframe.category_id.map(
            lambda id: self.cat_dict[id]))
        dataframe.insert(loc=1, column='Dif to One', value=dataframe.Average_Ratio.map(lambda r: abs(r - 1)))
        dataframe = dataframe.sort_values(
            by='Average_Ratio', ascending=False).reset_index(drop=True).reset_index()

        sns.set(font_scale=0.8)
        fig1, ax1 = plt.subplots()
        cat_count = 10
        hist_count = cat_count * 5

        fig_area = sns.catplot(x="Category", y="Likes/Dislikes Ratio", data=dataframe.head(hist_count),
                               hue="Country",
                               hue_order=['Canada', 'France',
                                          'Germany', 'Great British', 'USA'],
                               kind="bar", palette="muted", edgecolor="1", alpha=0.85, legend_out=False, ax=ax1)

        ax2 = ax1.twinx()
        fig_total = sns.catplot(x="Category", y="Average_Ratio", data=dataframe.head(hist_count),
                                kind='point', color="b", ax=ax2)

        ax1.set_title(
            "Likes/Dislikes Ratio of Different Genres and Countries", fontsize=30)
        ax1.grid(None)

    def controversial(self):
        dataframe = pd.DataFrame()
        for country in self.countries:
            raw_df = self.dataset[country][[
                'category_id', 'likes', 'dislikes']]
            df = raw_df.groupby('category_id')[
                ['likes', 'dislikes']].sum().reset_index()
            df['Likes/Dislikes Ratio'] = df['likes'] / df['dislikes']
            df['Country'] = self.full_name[country]
            dataframe = dataframe.append(df)

        dataframe['Average_Ratio'] = dataframe.groupby(
            'category_id')['Likes/Dislikes Ratio'].transform('sum') / 5
        dataframe.insert(loc=1, column='Category', value=dataframe.category_id.map(
            lambda id: self.cat_dict[id]))
        dataframe.insert(loc=1, column='Dif to One',
                         value=dataframe.Average_Ratio.map(lambda r: abs(r - 1)))
        dataframe = dataframe.sort_values(
            by='Dif to One', ascending=True).reset_index(drop=True).reset_index()

        sns.set(font_scale=0.8)
        fig1, ax1 = plt.subplots()
        cat_count = 10
        hist_count = cat_count * 5
        fig_area = sns.catplot(x="Category", y="Likes/Dislikes Ratio",
                               data=pd.DataFrame(pd.concat([dataframe.iloc[2:17], dataframe.iloc[20:50]])),
                               hue="Country", hue_order=['Canada', 'France',
                                                         'Germany', 'Great British',
                                                         'USA'], kind="bar", palette="muted", edgecolor="1", alpha=0.85,
                               legend_out=False, ax=ax1)
        ax2 = ax1.twinx()
        fig_total = sns.catplot(x="Category", y="Average_Ratio",
                                data=pd.DataFrame(pd.concat([dataframe.iloc[2:17], dataframe.iloc[20:50]])), kind='point',
                                color="b", ax=ax2)

        ax1.set_title("Likes/Dislikes Ratio of Different Genres and Countries", fontsize=30)

        ax1.grid(None)


    def show_plot(self):
        plt.xticks(rotation=45)
        plt.show()
        


if __name__ == '__main__':
    vm = VideoManager()
    vm.load_data()
    vm.videos_time()
    vm.total_views()
    vm.comment_views()
    vm.likes()
    vm.controversial()
    vm.show_plot()
    vm.correlation_matrix()
    vm.predict_views()
    vm.predict_likes()
    vm.predict_comments()
    

