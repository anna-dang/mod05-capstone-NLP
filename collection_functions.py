### WORK IN PROGRESS ### Pardon the current notes/lack of formal docstrings

import requests
import pandas as pd

from bs4 import BeautifulSoup
from selenium import webdriver
from time import sleep
from re import compile


def get_driver():
     # selenium chrome driver
    options = webdriver.ChromeOptions()
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--incognito')
    options.add_argument('--headless')
    driver = webdriver.Chrome(executable_path='./driver/chromedriver', options=options)
    return driver



def retrieve_reviews_ratings(soup):
    # set container 
    container = soup.findAll('div', class_="_2wrUUKlw _3hFEdNs8")

    # blank lists to store page data
    page_reviews = []
    page_ratings = []
    page_titles = []
    # looking for "ui_bubble_rating", following any value
    rating_re = compile("ui_bubble_rating (.*)")

    for item in container:
    
        # convert rating bubbles to 'int', add to list
        rating_raw = item.find('span', class_=rating_re)
        rating_int = int(rating_raw.attrs['class'][1].split("_")[1][-2])
        page_ratings.append(rating_int)

        # save all reviews on page as 'str'
        review = item.find('q', class_="IRsGHoPm").text
        
        # check for more text after "Read More" activated, complete review
        expanded = item.find('span', class_="_1M-1YYJt")
        if expanded:
            review += expanded.text
        page_reviews.append(review)

        # review title
        title = item.find('a', class_='ocfR3SKN').text
        page_titles.append(title)
        
    return page_reviews, page_ratings, page_titles




def retrieve_location(soup):
    # get hotel name
    hotel = soup.find(id="HEADING").text

    # get location (city/state)
    raw = soup.find('span', class_="_3ErVArsu jke2_wbp").text.split(", ")
    city = raw[-2]
    state = ''.join([i for i in raw[-1] if not i.isdigit()]).split(" -")[0]
    
    return (hotel, city, state)




def parse_url(start_url_ext, webdriver, location=False):
    """No pagination, single page
        returns location as (hotel, city, state"""
    domain = "https://www.tripadvisor.com"

    # get website
    webdriver.get(domain + start_url_ext)

    # Added to deal with errors 1/12, try to expand, otherwise just keep going
    try:
        # find the 'read more' buttons        
        more_buttons = webdriver.find_elements_by_class_name("_3maEfNCR")

        # ACTIVATE buttons, only need to press the first more button to access all!!, 
        # so just press one, once pressed they no longer exist
        if more_buttons[0].is_displayed():
            webdriver.execute_script("arguments[0].click();", more_buttons[0])
            sleep(1)

    except:
        continue

    # set soup    
    page_source = webdriver.page_source
    soup = BeautifulSoup(page_source, 'html.parser')

    # scrape the ratings data
    page_reviews, page_ratings, page_titles = retrieve_reviews_ratings(soup)

    # if location data requested
    if location == False:
        return page_reviews, page_ratings, page_titles
    else:
        location = retrieve_location(soup)
        return page_reviews, page_ratings, page_titles, location





def get_url_list(start_url, n=2):
    """ start_url - trip advisor hotel page ONE ONLY!!!!! not built for a mid start yet. 
        n- number of pages wanted, default 2 to return given start url and the next page extensions only.
        
        returns url extensions for start page (1) plus desired pages, in order
        
        returns n number of pages, if n=1 returns only original start page"""
    
       
    domain = "https://www.tripadvisor.com"
    
    # remove root if included
    if len(start_url.split(domain)) != 1:
        start_url = start_url.split(domain)[1]
    
    # five reviews displayed per page, thus five represented as the page counter
    n = (n - 1) * 5
    
    # set range
    pages = range(0,n, 5)

    # start list with start_url
    page_urls = [start_url]
    
    # split start url to insert page numbers
    url_split = start_url.split('Reviews')

    # generate desired pages
    for i in pages:
        page_num = 'Reviews-or' + str(5 + i)
        next_url = url_split[0] + page_num + url_split[1]
        page_urls.append(next_url)

    return page_urls
    # test_url_list = cf.get_url_list(middle, n=8)





def parse_url_list(url_list, webdriver=None):
    
    # returns location as tuple (hotel, city, state)

    # set driver for all urls to use, call new one if one  not passed in
    if not webdriver:
        driver = get_driver()
    else:
        driver = webdriver

    # set storage lists
    all_reviews = []
    all_ratings = []
    all_titles = []

    for idx, page in enumerate(url_list):

        # get location just once with parse of first url
        if idx == 0:
            page_one_reviews, page_one_ratings, page_one_titles, location = parse_url(page, webdriver=driver, location=True)
            all_reviews.extend(page_one_reviews)
            all_ratings.extend(page_one_ratings)
            all_titles.extend(page_one_titles)

        # retrive rest of ratings, location defaults to false
        else:    
            page_reviews, page_ratings, page_titles = parse_url(page, webdriver=driver)
            all_reviews.extend(page_reviews)
            all_ratings.extend(page_ratings)
            all_titles.extend(page_titles)
            #sleep(1)
    
    return all_reviews, all_ratings, all_titles, location




def make_reviews_df(reviews, ratings, titles, location=None):
    """location as a tuple!!!!!!! (hotel, city, state)"""
    df = pd.DataFrame([titles, reviews, ratings]).transpose()
    df.columns = ['Title', 'Review', 'Rating']
    
    if location:
        df['Hotel'] = location[0]
        df['Location'] = location[1] + ', ' + location[2]

        # rearrange
        df = df[['Location', 'Hotel', 'Title', 'Review', 'Rating']]
    
    return df




def scrape_hotel(start_url, n=2, webdriver=None):
   
    # generate list to scrape
    url_list = get_url_list(start_url, n=n)

    if webdriver:
        driver = webdriver

        # parse all urls for reviews and ratings
        all_reviews, all_ratings, all_titles, location = parse_url_list(url_list, webdriver=driver)
    
    else:
        all_reviews, all_ratings, all_titles, location = parse_url_list(url_list)

    # build df
    hotel_df = make_reviews_df(all_reviews, all_ratings, all_titles, location)
    
    return hotel_df