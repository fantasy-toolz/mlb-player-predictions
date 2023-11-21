
import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import numpy as np

def grab_fangraphs_data(years):
    """get fangraphs hitting data for a given year"""
    url      =   "https://www.fangraphs.com/leaders-legacy.aspx?pos=all&stats=bat&lg=all&qual=0&type=0&season={0}&month=0&season1={0}&ind=0&team=0&rost=0&age=0&filter=&players=0&page=1_1000"

    year_dfs = []

    for year in years:
        print('The year is {}'.format(year))
        r               = requests.get(url.format(year))
        #r               = requests.get(year_url)

        #soup            = BeautifulSoup(r.content, "html5lib")
        soup            = BeautifulSoup(r.text, "html5lib")
        table_data      = soup.find("table", { "class" : "rgMasterTable"})

        headers = [header.text for header in table_data.findAll('th')]
        rows = []
        for row in table_data.findAll("tr"):
            cells = row.findAll("td")
            if len(cells) != 0:
                for td in row.findAll("td"):
                    sav2 = [td.getText() for td in row.findAll("td")]
                rows.append(sav2)

        # Great, our Data is in a list of lists: much more pythonic. Let's birth a pandas table!
        df = pd.DataFrame(rows, columns=headers)
        df['Year'] = year
        year_dfs.append(df)

    df = year_dfs[0]
    for year in year_dfs[1:]:
        df = pd.concat([df, year])
    return df


def grab_fangraphs_pitching_data(years):
    url      =   "https://www.fangraphs.com/leaders.aspx?pos=all&stats=pit&lg=all&qual=0&type=0&season={0}&month=0&season1={0}&ind=0&team=0&rost=0&age=0&filter=&players=0&page=1_1000"

    year_dfs = []

    for year in years:
        print('The year is {}'.format(year))
        r               = requests.get(url.format(year))
        soup            = BeautifulSoup(r.text, "html5lib")
        table_data      = soup.find("table", { "class" : "rgMasterTable"})

        headers = [header.text for header in table_data.findAll('th')]
        rows = []
        for row in table_data.findAll("tr"):
            cells = row.findAll("td")
            if len(cells) != 0:
                for td in row.findAll("td"):
                    sav2 = [td.getText() for td in row.findAll("td")]
                rows.append(sav2)

        # Great, our Data is in a list of lists: much more pythonic. Let's birth a pandas table!
        df = pd.DataFrame(rows, columns=headers)
        df['Year'] = year
        year_dfs.append(df)

    df = year_dfs[0]
    for year in year_dfs[1:]:
        df = pd.concat([df, year])

    return df
