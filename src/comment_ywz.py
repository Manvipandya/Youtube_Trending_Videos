import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json


class VideoManager:
    def __init__(self):
        self.dataframe = pd.DataFrame()

    def load_data(self, data_root, countries, country_full_name):
        
        dataset = pd.DataFrame()
        for country in countries:
            file_path = data_root + country + 'videos.csv'
            raw_df = pd.read_csv(file_path, usecols=['category_id', 'views', 'comment_count'])
            df = raw_df.groupby('category_id')[['views', 'comment_count']].sum().reset_index()
            df['Comment/View Ratio'] = df['comment_count'] / df['views']
            df['Country'] = country_full_name[country]
            dataset = dataset.append(df)

        cat_dict = self.get_cat_dict(data_root, countries)
        dataset['Total Ratio'] = dataset.groupby('category_id')['Comment/View Ratio'].transform('sum')
        dataset.insert(loc=1, column='Category', value=dataset.category_id.map(lambda id: cat_dict[id]))
        dataset = dataset.sort_values(by='Total Ratio', ascending=False).reset_index(drop=True).reset_index()
        self.dataframe = dataset

    def get_cat_dict(self, data_root, countries):
        cat_dict = {}
        for country in countries:
            with open(data_root + '/' + country + '_category_id.json', 'r') as file:
                json_obj = json.load(file)
            for item in json_obj['items']:
                cat_dict[int(item['id'])] = item['snippet']['title']
        return cat_dict

    def plot(self, cat_count):
        sns.set(font_scale=0.8)
        fig1, ax1 = plt.subplots()

        hist_count = cat_count * 5

        fig_area = sns.catplot(x="Category", y="Comment/View Ratio", data=self.dataframe.head(hist_count),
                               hue="Country",
                               hue_order=['Canada', 'France', 'Germany', 'Great British', 'USA'],
                               kind="bar", palette="muted", edgecolor="1", alpha=0.85, legend_out=False, ax=ax1)

        ax2 = ax1.twinx()
        fig_total = sns.catplot(x="Category", y="Comment/View Ratio", data=self.dataframe.head(hist_count),
                                kind='point', color="b", ax=ax2)

        ax1.set_title("Comment/View Ratio of Different Genres and Countries", fontsize=30)
        ax1.grid(None)

        plt.show()


if __name__ == '__main__':
    data_root = 'data/'
    countries = ['CA', 'DE', 'FR', 'GB', 'US']
    country_full_name = {'CA': 'Canada', 'DE': 'Germany', 'FR': 'France', 'GB': 'Great British', 'US': 'USA'}

    vm = VideoManager()
    vm.load_data(data_root, countries, country_full_name)
    vm.plot(cat_count=10)
