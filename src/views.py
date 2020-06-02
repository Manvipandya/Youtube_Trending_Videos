import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from category import *

files = {}
def load():
    result_view = pd.DataFrame()
    for area in AREA:
        df_view = pd.DataFrame(pd.read_csv(csv_at(area))[['category_id', 'views']]).groupby('category_id')[
            'views'].sum().rename('Views in Different Areas').reset_index()
        df_view['Areas'] = full_name(area)
        result_view = result_view.append(df_view)
        files[area] = df_view
    files['total'] = result_view

    
def views(files):
    result_view = files['total']

    result_view.insert(loc=1, column='Categories',
                    value=result_view.category_id.map(lambda x: category_name(x)))

    result_view['Total Views'] = result_view.groupby(
        'category_id')['Views in Different Areas'].transform('sum')

    result_view = result_view.sort_values(
        by='Total Views', ascending=False).reset_index(drop=True).reset_index()

    sns.set(font_scale=2)
    sbplt_fig_view, sbplt_ax_view = plt.subplots()

    fig_view1 = sns.catplot(x="Categories", y="Views in Different Areas", data=result_view.head(35), hue="Areas",
                            hue_order=['Canada', 'France',
                                    'Germany', 'Great British', 'USA'],
                            kind="bar", palette="muted", edgecolor="1", alpha=0.85, legend_out=False, ax=sbplt_ax_view)

    sbplt_ax_view2 = sbplt_ax_view.twinx()
    fig_view2 = sns.catplot(x="Categories", y="Total Views", data=result_view.head(35),
                            kind='point', color="b", ax=sbplt_ax_view2)

    sbplt_ax_view.set_title("Views in different categories and areas", fontsize=30)
    sbplt_ax_view.grid(None)

load()
views(files)   
plt.show()
