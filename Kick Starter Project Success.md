---
layout: post
title:  "Kickstarter Predictions"
date:   2019-10-17 12:07:25 +0000
categories:
  - data
---

# Predictions of the Outcome of Kickstarter Campaigns

Kickstarter is a crowdfunding platform with a community of more than 10 million people comprising of creative, tech enthusiasts who help in bringing new projects to life.

Until now, more than $3 billion dollars have been contributed by the members in fueling creative projects. The projects can be literally anything – a device, a game, an app, a film etc.

Kickstarter works on all or nothing basis: a campaign is launched with a certain amount they want to raise, if it doesn’t meet its goal, the project owner gets nothing. For example: if a projects’s goal is 5000 USD. Even if it gets funded till $4999, the project won’t be a success.

If you have a project that you would like to post on Kickstarter now, can you predict whether it will be successfully funded or not? Looking into the dataset, what useful information can you extract from it, which variables are informative for your prediction and can you interpret the model?

The goal of this project is to build a classifier to predict whether a project will be successfully funded or not. You can use the algorithm of your choice.


```python
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

import model
import seaborn as sns
import matplotlib.pyplot as plt

%load_ext autoreload

%autoreload 2

pd.options.display.max_columns = None
```

    The autoreload extension is already loaded. To reload it, use:
      %reload_ext autoreload



```python
df = pd.read_csv('data.zip', index_col='id')
```


```python
df[df['evaluation_set']==True]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>photo</th>
      <th>name</th>
      <th>blurb</th>
      <th>goal</th>
      <th>slug</th>
      <th>disable_communication</th>
      <th>country</th>
      <th>currency</th>
      <th>currency_symbol</th>
      <th>currency_trailing_code</th>
      <th>deadline</th>
      <th>created_at</th>
      <th>launched_at</th>
      <th>static_usd_rate</th>
      <th>creator</th>
      <th>location</th>
      <th>category</th>
      <th>profile</th>
      <th>urls</th>
      <th>source_url</th>
      <th>friends</th>
      <th>is_starred</th>
      <th>is_backing</th>
      <th>permissions</th>
      <th>state</th>
      <th>evaluation_set</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1569898085</td>
      <td>{"small":"https://ksr-ugc.imgix.net/assets/012...</td>
      <td>Honeycomb Farm-to-Cupcakes</td>
      <td>Gourmet cupcakes made with high quality pastur...</td>
      <td>2000.0</td>
      <td>honeycomb-farm-to-table-cupcakes</td>
      <td>False</td>
      <td>US</td>
      <td>USD</td>
      <td>$</td>
      <td>True</td>
      <td>1452834000</td>
      <td>1449578222</td>
      <td>1449580843</td>
      <td>1.000000</td>
      <td>{"urls":{"web":{"user":"https://www.kickstarte...</td>
      <td>{"country":"US","urls":{"web":{"discover":"htt...</td>
      <td>{"urls":{"web":{"discover":"http://www.kicksta...</td>
      <td>{"background_image_opacity":0.8,"should_show_f...</td>
      <td>{"web":{"project":"https://www.kickstarter.com...</td>
      <td>https://www.kickstarter.com/discover/categorie...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
    </tr>
    <tr>
      <td>308824265</td>
      <td>{"small":"https://ksr-ugc.imgix.net/assets/012...</td>
      <td>Melted Crayons Entertainment</td>
      <td>Entertainment platform for aspiring artists of...</td>
      <td>2000.0</td>
      <td>melted-crayons-entertainment</td>
      <td>False</td>
      <td>CA</td>
      <td>CAD</td>
      <td>$</td>
      <td>True</td>
      <td>1460791734</td>
      <td>1455580978</td>
      <td>1455611334</td>
      <td>0.723816</td>
      <td>{"urls":{"web":{"user":"https://www.kickstarte...</td>
      <td>{"country":"CA","urls":{"web":{"discover":"htt...</td>
      <td>{"urls":{"web":{"discover":"http://www.kicksta...</td>
      <td>{"background_image_opacity":0.8,"should_show_f...</td>
      <td>{"web":{"project":"https://www.kickstarter.com...</td>
      <td>https://www.kickstarter.com/discover/categorie...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
    </tr>
    <tr>
      <td>341566518</td>
      <td>{"small":"https://ksr-ugc.imgix.net/assets/011...</td>
      <td>YouTube</td>
      <td>A creative mind without the right tools to vis...</td>
      <td>1600.0</td>
      <td>youtube-0</td>
      <td>False</td>
      <td>US</td>
      <td>USD</td>
      <td>$</td>
      <td>True</td>
      <td>1412268238</td>
      <td>1409619133</td>
      <td>1409676238</td>
      <td>1.000000</td>
      <td>{"urls":{"web":{"user":"https://www.kickstarte...</td>
      <td>{"country":"US","urls":{"web":{"discover":"htt...</td>
      <td>{"urls":{"web":{"discover":"http://www.kicksta...</td>
      <td>{"background_image_opacity":0.8,"should_show_f...</td>
      <td>{"web":{"project":"https://www.kickstarter.com...</td>
      <td>https://www.kickstarter.com/discover/categorie...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
    </tr>
    <tr>
      <td>898202473</td>
      <td>{"small":"https://ksr-ugc.imgix.net/assets/013...</td>
      <td>A book by Tiny House, Tiny Footprint</td>
      <td>Roll with Kathleen, Greg + Blaize in a Camper ...</td>
      <td>10000.0</td>
      <td>a-book-by-tiny-house-tiny-footprint</td>
      <td>False</td>
      <td>US</td>
      <td>USD</td>
      <td>$</td>
      <td>True</td>
      <td>1478699572</td>
      <td>1475449649</td>
      <td>1476103972</td>
      <td>1.000000</td>
      <td>{"urls":{"web":{"user":"https://www.kickstarte...</td>
      <td>{"country":"US","urls":{"web":{"discover":"htt...</td>
      <td>{"urls":{"web":{"discover":"http://www.kicksta...</td>
      <td>{"background_image_opacity":0.91,"should_show_...</td>
      <td>{"web":{"project":"https://www.kickstarter.com...</td>
      <td>https://www.kickstarter.com/discover/categorie...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
    </tr>
    <tr>
      <td>1383882757</td>
      <td>{"small":"https://ksr-ugc.imgix.net/assets/011...</td>
      <td>CUBIE</td>
      <td>CUBIE lives.  I want to ramp up funds to $2,00...</td>
      <td>200.0</td>
      <td>cubie</td>
      <td>False</td>
      <td>US</td>
      <td>USD</td>
      <td>$</td>
      <td>True</td>
      <td>1351452415</td>
      <td>1348768077</td>
      <td>1349724415</td>
      <td>1.000000</td>
      <td>{"urls":{"web":{"user":"https://www.kickstarte...</td>
      <td>{"country":"US","urls":{"web":{"discover":"htt...</td>
      <td>{"urls":{"web":{"discover":"http://www.kicksta...</td>
      <td>{"background_image_opacity":0.8,"should_show_f...</td>
      <td>{"web":{"project":"https://www.kickstarter.com...</td>
      <td>https://www.kickstarter.com/discover/categorie...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>605900449</td>
      <td>{"small":"https://ksr-ugc.imgix.net/assets/012...</td>
      <td>Aloneliness</td>
      <td>An original dance piece that embodies the mome...</td>
      <td>1800.0</td>
      <td>aloneliness</td>
      <td>False</td>
      <td>US</td>
      <td>USD</td>
      <td>$</td>
      <td>True</td>
      <td>1462892160</td>
      <td>1457993015</td>
      <td>1461078147</td>
      <td>1.000000</td>
      <td>{"urls":{"web":{"user":"https://www.kickstarte...</td>
      <td>{"country":"US","urls":{"web":{"discover":"htt...</td>
      <td>{"urls":{"web":{"discover":"http://www.kicksta...</td>
      <td>{"background_image_opacity":0.8,"should_show_f...</td>
      <td>{"web":{"project":"https://www.kickstarter.com...</td>
      <td>https://www.kickstarter.com/discover/categorie...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
    </tr>
    <tr>
      <td>759458017</td>
      <td>{"small":"https://ksr-ugc.imgix.net/assets/011...</td>
      <td>Generation Z Documentary</td>
      <td>A film by and for the maligned generation of k...</td>
      <td>12500.0</td>
      <td>generation-z-documentary</td>
      <td>False</td>
      <td>US</td>
      <td>USD</td>
      <td>$</td>
      <td>True</td>
      <td>1381082493</td>
      <td>1377889918</td>
      <td>1378490493</td>
      <td>1.000000</td>
      <td>{"urls":{"web":{"user":"https://www.kickstarte...</td>
      <td>{"country":"US","urls":{"web":{"discover":"htt...</td>
      <td>{"urls":{"web":{"discover":"http://www.kicksta...</td>
      <td>{"background_image_opacity":0.8,"should_show_f...</td>
      <td>{"web":{"project":"https://www.kickstarter.com...</td>
      <td>https://www.kickstarter.com/discover/categorie...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
    </tr>
    <tr>
      <td>1256873562</td>
      <td>{"small":"https://ksr-ugc.imgix.net/assets/012...</td>
      <td>Fairway Mafia</td>
      <td>2 friends, playing golf, traveling, course &amp; e...</td>
      <td>65000.0</td>
      <td>fairway-mafia</td>
      <td>False</td>
      <td>US</td>
      <td>USD</td>
      <td>$</td>
      <td>True</td>
      <td>1459164968</td>
      <td>1456247388</td>
      <td>1456576568</td>
      <td>1.000000</td>
      <td>{"urls":{"web":{"user":"https://www.kickstarte...</td>
      <td>{"country":"US","urls":{"web":{"discover":"htt...</td>
      <td>{"urls":{"web":{"discover":"http://www.kicksta...</td>
      <td>{"background_image_opacity":0.8,"should_show_f...</td>
      <td>{"web":{"project":"https://www.kickstarter.com...</td>
      <td>https://www.kickstarter.com/discover/categorie...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
    </tr>
    <tr>
      <td>1240998402</td>
      <td>{"small":"https://ksr-ugc.imgix.net/assets/012...</td>
      <td>MOSSCOLLY LTD</td>
      <td>Mosscolly is a network platform for artists, m...</td>
      <td>5000.0</td>
      <td>mosscolly-ltd</td>
      <td>False</td>
      <td>GB</td>
      <td>GBP</td>
      <td>Â£</td>
      <td>False</td>
      <td>1468443711</td>
      <td>1465602152</td>
      <td>1465851711</td>
      <td>1.425555</td>
      <td>{"urls":{"web":{"user":"https://www.kickstarte...</td>
      <td>{"country":"GB","urls":{"web":{"discover":"htt...</td>
      <td>{"urls":{"web":{"discover":"http://www.kicksta...</td>
      <td>{"background_image_opacity":0.8,"should_show_f...</td>
      <td>{"web":{"project":"https://www.kickstarter.com...</td>
      <td>https://www.kickstarter.com/discover/categorie...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
    </tr>
    <tr>
      <td>1583993878</td>
      <td>{"small":"https://ksr-ugc.imgix.net/assets/013...</td>
      <td>Funding Ma'iitsoh's Tattoo Studio (Canceled)</td>
      <td>I'm starting a high-end, clean and comfortable...</td>
      <td>756.0</td>
      <td>funding-maiitsohs-tattoo-studio</td>
      <td>False</td>
      <td>US</td>
      <td>USD</td>
      <td>$</td>
      <td>True</td>
      <td>1471989396</td>
      <td>1471365235</td>
      <td>1471384596</td>
      <td>1.000000</td>
      <td>{"urls":{"web":{"user":"https://www.kickstarte...</td>
      <td>{"country":"US","urls":{"web":{"discover":"htt...</td>
      <td>{"urls":{"web":{"discover":"http://www.kicksta...</td>
      <td>{"background_image_opacity":0.8,"should_show_f...</td>
      <td>{"web":{"project":"https://www.kickstarter.com...</td>
      <td>https://www.kickstarter.com/discover/categorie...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
<p>10000 rows × 26 columns</p>
</div>




```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>photo</th>
      <th>name</th>
      <th>blurb</th>
      <th>goal</th>
      <th>slug</th>
      <th>disable_communication</th>
      <th>country</th>
      <th>currency</th>
      <th>currency_symbol</th>
      <th>currency_trailing_code</th>
      <th>deadline</th>
      <th>created_at</th>
      <th>launched_at</th>
      <th>static_usd_rate</th>
      <th>creator</th>
      <th>location</th>
      <th>category</th>
      <th>profile</th>
      <th>urls</th>
      <th>source_url</th>
      <th>friends</th>
      <th>is_starred</th>
      <th>is_backing</th>
      <th>permissions</th>
      <th>state</th>
      <th>evaluation_set</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>805910621</td>
      <td>{"small":"https://ksr-ugc.imgix.net/assets/012...</td>
      <td>DOCUMENTARY FILM titled FROM RAGS TO SPIRITUAL...</td>
      <td>A MOVIE ABOUT THE WILLINGNESS TO BREAK FREE FR...</td>
      <td>125000.0</td>
      <td>movie-made-from-book-titled-from-rags-to-spiri...</td>
      <td>False</td>
      <td>US</td>
      <td>USD</td>
      <td>$</td>
      <td>True</td>
      <td>1447162860</td>
      <td>1444518329</td>
      <td>1444673815</td>
      <td>1.000000</td>
      <td>{"urls":{"web":{"user":"https://www.kickstarte...</td>
      <td>{"country":"US","urls":{"web":{"discover":"htt...</td>
      <td>{"urls":{"web":{"discover":"http://www.kicksta...</td>
      <td>{"background_image_opacity":0.8,"should_show_f...</td>
      <td>{"web":{"project":"https://www.kickstarter.com...</td>
      <td>https://www.kickstarter.com/discover/categorie...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>False</td>
    </tr>
    <tr>
      <td>1279627995</td>
      <td>{"small":"https://ksr-ugc.imgix.net/assets/011...</td>
      <td>American Politics, Policy, Power and Profit</td>
      <td>Everything you should know about really big go...</td>
      <td>9800.0</td>
      <td>american-politics-policy-power-and-profit</td>
      <td>False</td>
      <td>US</td>
      <td>USD</td>
      <td>$</td>
      <td>True</td>
      <td>1351709344</td>
      <td>1348156038</td>
      <td>1349117344</td>
      <td>1.000000</td>
      <td>{"urls":{"web":{"user":"https://www.kickstarte...</td>
      <td>{"country":"US","urls":{"web":{"discover":"htt...</td>
      <td>{"urls":{"web":{"discover":"http://www.kicksta...</td>
      <td>{"background_image_opacity":0.8,"should_show_f...</td>
      <td>{"web":{"project":"https://www.kickstarter.com...</td>
      <td>https://www.kickstarter.com/discover/categorie...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>False</td>
    </tr>
    <tr>
      <td>1306016155</td>
      <td>{"small":"https://ksr-ugc.imgix.net/assets/013...</td>
      <td>Drew Jacobs Official "Kiss Me" Music Video</td>
      <td>Be a part of the new "Kiss Me" Official Music ...</td>
      <td>2500.0</td>
      <td>drew-jacobs-official-kiss-me-music-video</td>
      <td>False</td>
      <td>US</td>
      <td>USD</td>
      <td>$</td>
      <td>True</td>
      <td>1475174031</td>
      <td>1473271187</td>
      <td>1473359631</td>
      <td>1.000000</td>
      <td>{"urls":{"web":{"user":"https://www.kickstarte...</td>
      <td>{"country":"US","urls":{"web":{"discover":"htt...</td>
      <td>{"urls":{"web":{"discover":"http://www.kicksta...</td>
      <td>{"background_image_opacity":0.8,"should_show_f...</td>
      <td>{"web":{"project":"https://www.kickstarter.com...</td>
      <td>https://www.kickstarter.com/discover/categorie...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>False</td>
    </tr>
    <tr>
      <td>658851276</td>
      <td>{"small":"https://ksr-ugc.imgix.net/assets/011...</td>
      <td>Still Loved</td>
      <td>When their dreams are shattered by the loss of...</td>
      <td>10000.0</td>
      <td>still-loved</td>
      <td>False</td>
      <td>GB</td>
      <td>GBP</td>
      <td>Â£</td>
      <td>False</td>
      <td>1400972400</td>
      <td>1395937256</td>
      <td>1397218790</td>
      <td>1.680079</td>
      <td>{"urls":{"web":{"user":"https://www.kickstarte...</td>
      <td>{"country":"GB","urls":{"web":{"discover":"htt...</td>
      <td>{"urls":{"web":{"discover":"http://www.kicksta...</td>
      <td>{"background_image_opacity":0.8,"should_show_f...</td>
      <td>{"web":{"project":"https://www.kickstarter.com...</td>
      <td>https://www.kickstarter.com/discover/categorie...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>False</td>
    </tr>
    <tr>
      <td>1971770539</td>
      <td>{"small":"https://ksr-ugc.imgix.net/assets/012...</td>
      <td>Nine Blackmon's HATER Film Project</td>
      <td>HATER is a mock rock doc about why the Rucker ...</td>
      <td>5500.0</td>
      <td>nine-blackmons-hater-film-project</td>
      <td>False</td>
      <td>US</td>
      <td>USD</td>
      <td>$</td>
      <td>True</td>
      <td>1425963600</td>
      <td>1422742820</td>
      <td>1423321493</td>
      <td>1.000000</td>
      <td>{"urls":{"web":{"user":"https://www.kickstarte...</td>
      <td>{"country":"US","urls":{"web":{"discover":"htt...</td>
      <td>{"urls":{"web":{"discover":"http://www.kicksta...</td>
      <td>{"background_image_opacity":0.8,"should_show_f...</td>
      <td>{"web":{"project":"https://www.kickstarter.com...</td>
      <td>https://www.kickstarter.com/discover/categorie...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['name_work_count'] = df.name.str.split().str.len()
```


```python

df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>photo</th>
      <th>name</th>
      <th>blurb</th>
      <th>goal</th>
      <th>slug</th>
      <th>disable_communication</th>
      <th>country</th>
      <th>currency</th>
      <th>currency_symbol</th>
      <th>currency_trailing_code</th>
      <th>deadline</th>
      <th>created_at</th>
      <th>launched_at</th>
      <th>static_usd_rate</th>
      <th>creator</th>
      <th>location</th>
      <th>category</th>
      <th>profile</th>
      <th>urls</th>
      <th>source_url</th>
      <th>friends</th>
      <th>is_starred</th>
      <th>is_backing</th>
      <th>permissions</th>
      <th>state</th>
      <th>evaluation_set</th>
      <th>name_word_count</th>
      <th>name_work_count</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>805910621</td>
      <td>{"small":"https://ksr-ugc.imgix.net/assets/012...</td>
      <td>DOCUMENTARY FILM titled FROM RAGS TO SPIRITUAL...</td>
      <td>A MOVIE ABOUT THE WILLINGNESS TO BREAK FREE FR...</td>
      <td>125000.0</td>
      <td>movie-made-from-book-titled-from-rags-to-spiri...</td>
      <td>False</td>
      <td>US</td>
      <td>USD</td>
      <td>$</td>
      <td>True</td>
      <td>1447162860</td>
      <td>1444518329</td>
      <td>1444673815</td>
      <td>1.000000</td>
      <td>{"urls":{"web":{"user":"https://www.kickstarte...</td>
      <td>{"country":"US","urls":{"web":{"discover":"htt...</td>
      <td>{"urls":{"web":{"discover":"http://www.kicksta...</td>
      <td>{"background_image_opacity":0.8,"should_show_f...</td>
      <td>{"web":{"project":"https://www.kickstarter.com...</td>
      <td>https://www.kickstarter.com/discover/categorie...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>False</td>
      <td>60000</td>
      <td>8.0</td>
    </tr>
    <tr>
      <td>1279627995</td>
      <td>{"small":"https://ksr-ugc.imgix.net/assets/011...</td>
      <td>American Politics, Policy, Power and Profit</td>
      <td>Everything you should know about really big go...</td>
      <td>9800.0</td>
      <td>american-politics-policy-power-and-profit</td>
      <td>False</td>
      <td>US</td>
      <td>USD</td>
      <td>$</td>
      <td>True</td>
      <td>1351709344</td>
      <td>1348156038</td>
      <td>1349117344</td>
      <td>1.000000</td>
      <td>{"urls":{"web":{"user":"https://www.kickstarte...</td>
      <td>{"country":"US","urls":{"web":{"discover":"htt...</td>
      <td>{"urls":{"web":{"discover":"http://www.kicksta...</td>
      <td>{"background_image_opacity":0.8,"should_show_f...</td>
      <td>{"web":{"project":"https://www.kickstarter.com...</td>
      <td>https://www.kickstarter.com/discover/categorie...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>False</td>
      <td>60000</td>
      <td>6.0</td>
    </tr>
    <tr>
      <td>1306016155</td>
      <td>{"small":"https://ksr-ugc.imgix.net/assets/013...</td>
      <td>Drew Jacobs Official "Kiss Me" Music Video</td>
      <td>Be a part of the new "Kiss Me" Official Music ...</td>
      <td>2500.0</td>
      <td>drew-jacobs-official-kiss-me-music-video</td>
      <td>False</td>
      <td>US</td>
      <td>USD</td>
      <td>$</td>
      <td>True</td>
      <td>1475174031</td>
      <td>1473271187</td>
      <td>1473359631</td>
      <td>1.000000</td>
      <td>{"urls":{"web":{"user":"https://www.kickstarte...</td>
      <td>{"country":"US","urls":{"web":{"discover":"htt...</td>
      <td>{"urls":{"web":{"discover":"http://www.kicksta...</td>
      <td>{"background_image_opacity":0.8,"should_show_f...</td>
      <td>{"web":{"project":"https://www.kickstarter.com...</td>
      <td>https://www.kickstarter.com/discover/categorie...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>False</td>
      <td>60000</td>
      <td>7.0</td>
    </tr>
    <tr>
      <td>658851276</td>
      <td>{"small":"https://ksr-ugc.imgix.net/assets/011...</td>
      <td>Still Loved</td>
      <td>When their dreams are shattered by the loss of...</td>
      <td>10000.0</td>
      <td>still-loved</td>
      <td>False</td>
      <td>GB</td>
      <td>GBP</td>
      <td>Â£</td>
      <td>False</td>
      <td>1400972400</td>
      <td>1395937256</td>
      <td>1397218790</td>
      <td>1.680079</td>
      <td>{"urls":{"web":{"user":"https://www.kickstarte...</td>
      <td>{"country":"GB","urls":{"web":{"discover":"htt...</td>
      <td>{"urls":{"web":{"discover":"http://www.kicksta...</td>
      <td>{"background_image_opacity":0.8,"should_show_f...</td>
      <td>{"web":{"project":"https://www.kickstarter.com...</td>
      <td>https://www.kickstarter.com/discover/categorie...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>False</td>
      <td>60000</td>
      <td>2.0</td>
    </tr>
    <tr>
      <td>1971770539</td>
      <td>{"small":"https://ksr-ugc.imgix.net/assets/012...</td>
      <td>Nine Blackmon's HATER Film Project</td>
      <td>HATER is a mock rock doc about why the Rucker ...</td>
      <td>5500.0</td>
      <td>nine-blackmons-hater-film-project</td>
      <td>False</td>
      <td>US</td>
      <td>USD</td>
      <td>$</td>
      <td>True</td>
      <td>1425963600</td>
      <td>1422742820</td>
      <td>1423321493</td>
      <td>1.000000</td>
      <td>{"urls":{"web":{"user":"https://www.kickstarte...</td>
      <td>{"country":"US","urls":{"web":{"discover":"htt...</td>
      <td>{"urls":{"web":{"discover":"http://www.kicksta...</td>
      <td>{"background_image_opacity":0.8,"should_show_f...</td>
      <td>{"web":{"project":"https://www.kickstarter.com...</td>
      <td>https://www.kickstarter.com/discover/categorie...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>False</td>
      <td>60000</td>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['country'] = np.where((df['country']!='US')&(df['country']!='GB'), 'Other', df['country'])
```


```python
df.country.value_counts()
```




    US       48982
    Other     6141
    GB        4877
    Name: country, dtype: int64




```python
    # Extract major and minor categories
    pattern = r'(?<="slug":")(\w*\s*.*)(?=")'
    df['short_category'] = df.category.str.extract(pattern)
    df[['major_category', 'minor_category']] = df.short_category.str.split('/', n=0, expand=True)

    # Update goal values to USD for non US countries
    df["usa"] = df["country"] == "US"
    df["goal_usd"] = df["goal"] * df["static_usd_rate"]
    df["goal_usd"] = df["goal_usd"].astype(int)
    

```


```python
    # calculate duration of fundraising and time to launch
    unix_seconds_per_day = 86400
    df['duration'] = df.deadline - df.launched_at
    df['duration'] = df['duration'].div(unix_seconds_per_day).abs().astype(int)
    df['time_to_launch'] = df.launched_at - df.created_at
    df['time_to_launch'] = df['time_to_launch'].div(unix_seconds_per_day).abs().astype(int)
```


```python
    # Add features
    df['blurb_length'] = df.blurb.str.len()
    df['name_length'] = df.name.str.len()
    df['slug_length'] = df.slug.str.len()
```


```python
plt.scatter((df.loc[df['state']==0,'time_to_launch']), (df.loc[df['state']==0,'goal_usd']), c='red', marker='o', label='class 0')
plt.scatter((df.loc[df['state']==1,'time_to_launch']), (df.loc[df['state']==1,'goal_usd']), c='blue', marker='o', label='class 0')

# plt.scatter(df[df['state']==0, df['time_to_launch']], df[df['state']==0, df['goal_usd']], marker='o', label='class 0')
# plt.scatter(df[df['state']==1, df['time_to_launch']], df[df['state']==1, df['goal_usd']], marker='x', label='class 1')
plt.xlabel('time_to_launch')
plt.ylim(0, 2500000)
plt.ylabel('goal (USD))')
# plt.legend()
plt.show();
```


![png](Kick%20Starter%20Project%20Success_files/Kick%20Starter%20Project%20Success_12_0.png)



```python
plt.scatter((df.loc[df['state']==0,'duration']), (df.loc[df['state']==0,'goal_usd']), c='red', marker='o', label='class 0')
plt.scatter((df.loc[df['state']==1,'duration']), (df.loc[df['state']==1,'goal_usd']), c='blue', marker='x', label='class 0')

# plt.scatter(df[df['state']==0, df['time_to_launch']], df[df['state']==0, df['goal_usd']], marker='o', label='class 0')
# plt.scatter(df[df['state']==1, df['time_to_launch']], df[df['state']==1, df['goal_usd']], marker='x', label='class 1')
plt.xlabel('duration')
plt.ylim(0, 500000)
plt.ylabel('goal (USD))')
# plt.legend()
plt.show();
```


![png](Kick%20Starter%20Project%20Success_files/Kick%20Starter%20Project%20Success_13_0.png)



```python
df.duration.hist(bins=30);
```


![png](Kick%20Starter%20Project%20Success_files/Kick%20Starter%20Project%20Success_14_0.png)



```python
    # Launch Day / Month
    df['launch_dt'] = pd.to_datetime(df['launched_at'], unit='s')
    df['launch_month'] = pd.DatetimeIndex(df['launch_dt']).month
    df['launch_year'] = pd.DatetimeIndex(df['launch_dt']).year
    df['launch_day'] = pd.DatetimeIndex(df['launch_dt']).day
```


```python
df['total_count'] = 1
```


```python

```


```python
grouped = df.groupby(['launch_month']).sum()
grouped['percent_won'] = grouped.state / grouped.total_count
sns.set_color_codes("pastel")
sns.barplot(x=grouped.index, y=grouped.percent_won, 
           label="Total", color="b");
```


![png](Kick%20Starter%20Project%20Success_files/Kick%20Starter%20Project%20Success_18_0.png)



```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>photo</th>
      <th>name</th>
      <th>blurb</th>
      <th>goal</th>
      <th>slug</th>
      <th>disable_communication</th>
      <th>country</th>
      <th>currency</th>
      <th>currency_symbol</th>
      <th>currency_trailing_code</th>
      <th>deadline</th>
      <th>created_at</th>
      <th>launched_at</th>
      <th>static_usd_rate</th>
      <th>creator</th>
      <th>location</th>
      <th>category</th>
      <th>profile</th>
      <th>urls</th>
      <th>source_url</th>
      <th>friends</th>
      <th>is_starred</th>
      <th>is_backing</th>
      <th>permissions</th>
      <th>state</th>
      <th>evaluation_set</th>
      <th>short_category</th>
      <th>major_category</th>
      <th>minor_category</th>
      <th>usa</th>
      <th>goal_usd</th>
      <th>duration</th>
      <th>time_to_launch</th>
      <th>blurb_length</th>
      <th>name_length</th>
      <th>slug_length</th>
      <th>launch_dt</th>
      <th>launch_month</th>
      <th>launch_year</th>
      <th>launch_day</th>
      <th>total_count</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>805910621</td>
      <td>{"small":"https://ksr-ugc.imgix.net/assets/012...</td>
      <td>DOCUMENTARY FILM titled FROM RAGS TO SPIRITUAL...</td>
      <td>A MOVIE ABOUT THE WILLINGNESS TO BREAK FREE FR...</td>
      <td>125000.0</td>
      <td>movie-made-from-book-titled-from-rags-to-spiri...</td>
      <td>False</td>
      <td>US</td>
      <td>USD</td>
      <td>$</td>
      <td>True</td>
      <td>1447162860</td>
      <td>1444518329</td>
      <td>1444673815</td>
      <td>1.000000</td>
      <td>{"urls":{"web":{"user":"https://www.kickstarte...</td>
      <td>{"country":"US","urls":{"web":{"discover":"htt...</td>
      <td>{"urls":{"web":{"discover":"http://www.kicksta...</td>
      <td>{"background_image_opacity":0.8,"should_show_f...</td>
      <td>{"web":{"project":"https://www.kickstarter.com...</td>
      <td>https://www.kickstarter.com/discover/categorie...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>False</td>
      <td>film &amp; video/movie theaters</td>
      <td>film &amp; video</td>
      <td>movie theaters</td>
      <td>True</td>
      <td>125000</td>
      <td>28</td>
      <td>1</td>
      <td>134.0</td>
      <td>53.0</td>
      <td>50</td>
      <td>2015-10-12 18:16:55</td>
      <td>10</td>
      <td>2015</td>
      <td>12</td>
      <td>1</td>
    </tr>
    <tr>
      <td>1279627995</td>
      <td>{"small":"https://ksr-ugc.imgix.net/assets/011...</td>
      <td>American Politics, Policy, Power and Profit</td>
      <td>Everything you should know about really big go...</td>
      <td>9800.0</td>
      <td>american-politics-policy-power-and-profit</td>
      <td>False</td>
      <td>US</td>
      <td>USD</td>
      <td>$</td>
      <td>True</td>
      <td>1351709344</td>
      <td>1348156038</td>
      <td>1349117344</td>
      <td>1.000000</td>
      <td>{"urls":{"web":{"user":"https://www.kickstarte...</td>
      <td>{"country":"US","urls":{"web":{"discover":"htt...</td>
      <td>{"urls":{"web":{"discover":"http://www.kicksta...</td>
      <td>{"background_image_opacity":0.8,"should_show_f...</td>
      <td>{"web":{"project":"https://www.kickstarter.com...</td>
      <td>https://www.kickstarter.com/discover/categorie...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>False</td>
      <td>publishing/nonfiction</td>
      <td>publishing</td>
      <td>nonfiction</td>
      <td>True</td>
      <td>9800</td>
      <td>30</td>
      <td>11</td>
      <td>131.0</td>
      <td>43.0</td>
      <td>41</td>
      <td>2012-10-01 18:49:04</td>
      <td>10</td>
      <td>2012</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>1306016155</td>
      <td>{"small":"https://ksr-ugc.imgix.net/assets/013...</td>
      <td>Drew Jacobs Official "Kiss Me" Music Video</td>
      <td>Be a part of the new "Kiss Me" Official Music ...</td>
      <td>2500.0</td>
      <td>drew-jacobs-official-kiss-me-music-video</td>
      <td>False</td>
      <td>US</td>
      <td>USD</td>
      <td>$</td>
      <td>True</td>
      <td>1475174031</td>
      <td>1473271187</td>
      <td>1473359631</td>
      <td>1.000000</td>
      <td>{"urls":{"web":{"user":"https://www.kickstarte...</td>
      <td>{"country":"US","urls":{"web":{"discover":"htt...</td>
      <td>{"urls":{"web":{"discover":"http://www.kicksta...</td>
      <td>{"background_image_opacity":0.8,"should_show_f...</td>
      <td>{"web":{"project":"https://www.kickstarter.com...</td>
      <td>https://www.kickstarter.com/discover/categorie...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>False</td>
      <td>music/country &amp; folk</td>
      <td>music</td>
      <td>country &amp; folk</td>
      <td>True</td>
      <td>2500</td>
      <td>21</td>
      <td>1</td>
      <td>52.0</td>
      <td>42.0</td>
      <td>40</td>
      <td>2016-09-08 18:33:51</td>
      <td>9</td>
      <td>2016</td>
      <td>8</td>
      <td>1</td>
    </tr>
    <tr>
      <td>658851276</td>
      <td>{"small":"https://ksr-ugc.imgix.net/assets/011...</td>
      <td>Still Loved</td>
      <td>When their dreams are shattered by the loss of...</td>
      <td>10000.0</td>
      <td>still-loved</td>
      <td>False</td>
      <td>GB</td>
      <td>GBP</td>
      <td>Â£</td>
      <td>False</td>
      <td>1400972400</td>
      <td>1395937256</td>
      <td>1397218790</td>
      <td>1.680079</td>
      <td>{"urls":{"web":{"user":"https://www.kickstarte...</td>
      <td>{"country":"GB","urls":{"web":{"discover":"htt...</td>
      <td>{"urls":{"web":{"discover":"http://www.kicksta...</td>
      <td>{"background_image_opacity":0.8,"should_show_f...</td>
      <td>{"web":{"project":"https://www.kickstarter.com...</td>
      <td>https://www.kickstarter.com/discover/categorie...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>False</td>
      <td>film &amp; video/documentary</td>
      <td>film &amp; video</td>
      <td>documentary</td>
      <td>False</td>
      <td>16800</td>
      <td>43</td>
      <td>14</td>
      <td>120.0</td>
      <td>11.0</td>
      <td>11</td>
      <td>2014-04-11 12:19:50</td>
      <td>4</td>
      <td>2014</td>
      <td>11</td>
      <td>1</td>
    </tr>
    <tr>
      <td>1971770539</td>
      <td>{"small":"https://ksr-ugc.imgix.net/assets/012...</td>
      <td>Nine Blackmon's HATER Film Project</td>
      <td>HATER is a mock rock doc about why the Rucker ...</td>
      <td>5500.0</td>
      <td>nine-blackmons-hater-film-project</td>
      <td>False</td>
      <td>US</td>
      <td>USD</td>
      <td>$</td>
      <td>True</td>
      <td>1425963600</td>
      <td>1422742820</td>
      <td>1423321493</td>
      <td>1.000000</td>
      <td>{"urls":{"web":{"user":"https://www.kickstarte...</td>
      <td>{"country":"US","urls":{"web":{"discover":"htt...</td>
      <td>{"urls":{"web":{"discover":"http://www.kicksta...</td>
      <td>{"background_image_opacity":0.8,"should_show_f...</td>
      <td>{"web":{"project":"https://www.kickstarter.com...</td>
      <td>https://www.kickstarter.com/discover/categorie...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>False</td>
      <td>film &amp; video/narrative film</td>
      <td>film &amp; video</td>
      <td>narrative film</td>
      <td>True</td>
      <td>5500</td>
      <td>30</td>
      <td>6</td>
      <td>125.0</td>
      <td>34.0</td>
      <td>33</td>
      <td>2015-02-07 15:04:53</td>
      <td>2</td>
      <td>2015</td>
      <td>7</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
columns_to_drop = ['photo', 'name', 'blurb', 'slug', 'friends', 'is_starred', 'is_backing',
                       'permissions', 'currency_symbol', 'creator', 'profile', 'urls',
                       'source_url', 'short_category', 'category', 'goal', 'disable_communication',
                       'deadline', 'created_at', 'location', 'launched_at', 'static_usd_rate',
                       'currency', 'currency_trailing_code', 'name_length', 'slug_length', 'country', 'launch_dt']
df.drop(columns_to_drop, axis=1, inplace=True)
df.head() 

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state</th>
      <th>evaluation_set</th>
      <th>major_category</th>
      <th>minor_category</th>
      <th>usa</th>
      <th>goal_usd</th>
      <th>duration</th>
      <th>time_to_launch</th>
      <th>blurb_length</th>
      <th>launch_month</th>
      <th>launch_year</th>
      <th>launch_day</th>
      <th>total_count</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>805910621</td>
      <td>0.0</td>
      <td>False</td>
      <td>film &amp; video</td>
      <td>movie theaters</td>
      <td>True</td>
      <td>125000</td>
      <td>28</td>
      <td>1</td>
      <td>134.0</td>
      <td>10</td>
      <td>2015</td>
      <td>12</td>
      <td>1</td>
    </tr>
    <tr>
      <td>1279627995</td>
      <td>0.0</td>
      <td>False</td>
      <td>publishing</td>
      <td>nonfiction</td>
      <td>True</td>
      <td>9800</td>
      <td>30</td>
      <td>11</td>
      <td>131.0</td>
      <td>10</td>
      <td>2012</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>1306016155</td>
      <td>1.0</td>
      <td>False</td>
      <td>music</td>
      <td>country &amp; folk</td>
      <td>True</td>
      <td>2500</td>
      <td>21</td>
      <td>1</td>
      <td>52.0</td>
      <td>9</td>
      <td>2016</td>
      <td>8</td>
      <td>1</td>
    </tr>
    <tr>
      <td>658851276</td>
      <td>1.0</td>
      <td>False</td>
      <td>film &amp; video</td>
      <td>documentary</td>
      <td>False</td>
      <td>16800</td>
      <td>43</td>
      <td>14</td>
      <td>120.0</td>
      <td>4</td>
      <td>2014</td>
      <td>11</td>
      <td>1</td>
    </tr>
    <tr>
      <td>1971770539</td>
      <td>0.0</td>
      <td>False</td>
      <td>film &amp; video</td>
      <td>narrative film</td>
      <td>True</td>
      <td>5500</td>
      <td>30</td>
      <td>6</td>
      <td>125.0</td>
      <td>2</td>
      <td>2015</td>
      <td>7</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
categorical_columns = ['major_category', 'minor_category']
```


```python
# create dummies (for LogisticRegression)

dummy_cols = pd.DataFrame
for col in categorical_columns:
    dummy_cols = pd.get_dummies(df[col])
    df = pd.concat([df, dummy_cols], axis=1)
    del df[col]

```


```python
# change categorical features in type category (for Decision Tree)
# for col in categorical_columns:
#     df[col] = df[col].astype('category')
#     df[col] = df[col].cat.codes
```


```python
df.dropna(inplace=True)
```


```python
columns_to_scale = ['goal_usd', 'duration', 'time_to_launch', 'blurb_length']
scaler = MinMaxScaler()
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
```


```python
df = df[df.evaluation_set==0]
features = df.drop('state', axis=1)
target = df['state']
```


```python
# pca = PCA(n_components=9)
# features = pca.fit_transform(features)
# pca_df = pd.DataFrame(columns=['PC{}'.format(x) for x in range(1, pca.n_components_ + 1)],
#                           index=[0])
# pca_df.iloc[0, :] = np.cumsum(pca.explained_variance_ratio_) * 100
# display(pca_df)
```


```python
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)
```


```python

```


```python
# params = {'max_depth': [10, 11, 12]}
# grid = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=params, cv=5, verbose=False)
# grid.fit(X_train, y_train)
# display(grid.best_params_)
```


```python
# params = {'n_neighbors': [3, 5, 7, 11, 15]}
# grid = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=params, cv=5, verbose=False)
# grid.fit(X_train, y_train)
# display(grid.best_params_)
```


```python
params = {'penalty': ['l1'],
         'C': [100, 1000]}
grid = GridSearchCV(estimator=LogisticRegression(), param_grid=params, cv=5, verbose=False)
grid.fit(X_train, y_train)


```

    /Users/Graham/miniconda3/envs/ads2/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /Users/Graham/miniconda3/envs/ads2/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /Users/Graham/miniconda3/envs/ads2/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /Users/Graham/miniconda3/envs/ads2/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /Users/Graham/miniconda3/envs/ads2/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /Users/Graham/miniconda3/envs/ads2/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /Users/Graham/miniconda3/envs/ads2/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /Users/Graham/miniconda3/envs/ads2/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /Users/Graham/miniconda3/envs/ads2/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /Users/Graham/miniconda3/envs/ads2/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /Users/Graham/miniconda3/envs/ads2/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /Users/Graham/miniconda3/envs/ads2/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /Users/Graham/miniconda3/envs/ads2/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /Users/Graham/miniconda3/envs/ads2/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /Users/Graham/miniconda3/envs/ads2/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /Users/Graham/miniconda3/envs/ads2/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /Users/Graham/miniconda3/envs/ads2/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    /Users/Graham/miniconda3/envs/ads2/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /Users/Graham/miniconda3/envs/ads2/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)





    GridSearchCV(cv=5, error_score='raise-deprecating',
                 estimator=LogisticRegression(C=1.0, class_weight=None, dual=False,
                                              fit_intercept=True,
                                              intercept_scaling=1, l1_ratio=None,
                                              max_iter=100, multi_class='warn',
                                              n_jobs=None, penalty='l2',
                                              random_state=None, solver='warn',
                                              tol=0.0001, verbose=0,
                                              warm_start=False),
                 iid='warn', n_jobs=None,
                 param_grid={'C': [100, 1000], 'penalty': ['l1']},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring=None, verbose=False)




```python
pd.DataFrame(grid.cv_results_)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean_fit_time</th>
      <th>std_fit_time</th>
      <th>mean_score_time</th>
      <th>std_score_time</th>
      <th>param_C</th>
      <th>param_penalty</th>
      <th>params</th>
      <th>split0_test_score</th>
      <th>split1_test_score</th>
      <th>split2_test_score</th>
      <th>split3_test_score</th>
      <th>split4_test_score</th>
      <th>mean_test_score</th>
      <th>std_test_score</th>
      <th>rank_test_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>57.367188</td>
      <td>11.845474</td>
      <td>0.099216</td>
      <td>0.005291</td>
      <td>100</td>
      <td>l1</td>
      <td>{'C': 100, 'penalty': 'l1'}</td>
      <td>0.71400</td>
      <td>0.715125</td>
      <td>0.711125</td>
      <td>0.7105</td>
      <td>0.69855</td>
      <td>0.70986</td>
      <td>0.005912</td>
      <td>2</td>
    </tr>
    <tr>
      <td>1</td>
      <td>14.999164</td>
      <td>11.469945</td>
      <td>0.095155</td>
      <td>0.002495</td>
      <td>1000</td>
      <td>l1</td>
      <td>{'C': 1000, 'penalty': 'l1'}</td>
      <td>0.71475</td>
      <td>0.715000</td>
      <td>0.710000</td>
      <td>0.7110</td>
      <td>0.69880</td>
      <td>0.70991</td>
      <td>0.005898</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
grid.best_params_
```




    {'C': 1000, 'penalty': 'l1'}




```python

```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-102-8991a9dca522> in <module>
    ----> 1 plt.plot(range(1, 2), grid.cv_results_['mean_test_score'])
    

    ~/miniconda3/envs/ads2/lib/python3.7/site-packages/matplotlib/pyplot.py in plot(scalex, scaley, data, *args, **kwargs)
       2793     return gca().plot(
       2794         *args, scalex=scalex, scaley=scaley, **({"data": data} if data
    -> 2795         is not None else {}), **kwargs)
       2796 
       2797 


    ~/miniconda3/envs/ads2/lib/python3.7/site-packages/matplotlib/axes/_axes.py in plot(self, scalex, scaley, data, *args, **kwargs)
       1664         """
       1665         kwargs = cbook.normalize_kwargs(kwargs, mlines.Line2D._alias_map)
    -> 1666         lines = [*self._get_lines(*args, data=data, **kwargs)]
       1667         for line in lines:
       1668             self.add_line(line)


    ~/miniconda3/envs/ads2/lib/python3.7/site-packages/matplotlib/axes/_base.py in __call__(self, *args, **kwargs)
        223                 this += args[0],
        224                 args = args[1:]
    --> 225             yield from self._plot_args(this, kwargs)
        226 
        227     def get_next_color(self):


    ~/miniconda3/envs/ads2/lib/python3.7/site-packages/matplotlib/axes/_base.py in _plot_args(self, tup, kwargs)
        389             x, y = index_of(tup[-1])
        390 
    --> 391         x, y = self._xy_from_xy(x, y)
        392 
        393         if self.command == 'plot':


    ~/miniconda3/envs/ads2/lib/python3.7/site-packages/matplotlib/axes/_base.py in _xy_from_xy(self, x, y)
        268         if x.shape[0] != y.shape[0]:
        269             raise ValueError("x and y must have same first dimension, but "
    --> 270                              "have shapes {} and {}".format(x.shape, y.shape))
        271         if x.ndim > 2 or y.ndim > 2:
        272             raise ValueError("x and y can be no greater than 2-D, but have "


    ValueError: x and y must have same first dimension, but have shapes (1,) and (2,)



![png](Kick%20Starter%20Project%20Success_files/Kick%20Starter%20Project%20Success_35_1.png)



```python
# model = LogisticRegression()
# model.fit(X_train, y_train)
# y_hat = model.predict(X_test)
# accuracy = accuracy_score(y_hat, y_test)
# print('Accuracy is {}'.format(accuracy.round(3)))
```


```python
# model = KNeighborsClassifier(n_neighbors=7)
# model.fit(X_train, y_train)
# y_hat = model.predict(X_test)
# accuracy = accuracy_score(y_hat, y_test)
# print('Accuracy is {}'.format(accuracy.round(3)))
```


```python
# dtc = DecisionTreeClassifier(max_depth=10)
# dtc.fit(X_train, y_train)
# y_hat = dtc.predict(X_test)
# accuracy = accuracy_score(y_hat, y_test)
# print('Accuracy is {}'.format(accuracy.round(3)))
```

    Accuracy is 0.691



```python
# display(X_train.head())
# display(y_train.head())
# display(X_test.head())
# X_train.columns
```


```python
# dtc = model.train(X_train, y_train)
```


```python
# y_hat = model.predict(dtc, X_test)
```


```python

```


```python

```
