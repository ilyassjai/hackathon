import os
import csv
import pandas as pd


def retrieve_events(city):

    dt = pd.read_csv('Marrakech_Events.csv')

    data = pd.DataFrame(dt)

    data = data.drop('Price', axis = 1)

    target = data.query('Location == "{c}"'.format(c=city))

    target = target.drop('Location', axis = 1)

    res = [" ,".join(e) for e in target.values]

    return "\n".join(res)

def retrieve_sites(city):

    dt = pd.read_csv('Marrakech_Expanded_Travel_Guide.csv')

    data = pd.DataFrame(dt)

    data = data.drop('Price', axis = 1)

    # target = data.query('Location == "{c}"'.format(c=city))

    target = data.drop('Activities', axis = 1)

    res = [" ,".join(e) for e in target.values]

    return "\n".join(res)    

def prompt_format(template, preferences, 
                  city, start_date, end_date):
    
    events = retrieve_events(city)

    sites = retrieve_sites(city)

    temp = template.format(
        events = events,
        sites = sites,
        preferences = preferences,
        city = city,
        start_date = start_date,
        end_date = end_date
    )

    return temp


print(retrieve_events("Marrakesh"))