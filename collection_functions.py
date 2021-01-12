### WORK IN PROGRESS ###

import requests
import pandas as pd
from bs4 import BeautifulSoup
from time import sleep
from re import compile


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




def parse_url(start_url_ext,  location=False):
    """No pagination, single page
        returns location as (hotel, city, state"""
    root = "https://www.tripadvisor.com"
    html = requests.get(root + start_url_ext)
    soup = BeautifulSoup(html.content, 'html.parser')
    page_reviews, page_ratings, page_titles = retrieve_reviews_ratings(soup)
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





def parse_url_list(url_list):
    
    # returns location as ttuple (hotel, city, state)
    
    all_reviews = []
    all_ratings = []
    all_titles = []

    for idx, page in enumerate(url_list):
        # get location just once with parse of first url
        if idx == 0:
            page_one_reviews, page_one_ratings, page_one_titles, location = parse_url(page, location=True)
            all_reviews.extend(page_one_reviews)
            all_ratings.extend(page_one_ratings)
            all_titles.extend(page_one_titles)
        # retrive rest of ratigns, location defaults to false
        else:    
            page_reviews, page_ratings, page_titles = parse_url(page)
            all_reviews.extend(page_reviews)
            all_ratings.extend(page_ratings)
            all_titles.extend(page_titles)
    
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




def scrape_hotel(start_url, n=2):
   
    # generate list to scrape
    url_list = get_url_list(start_url, n=n)

    # parse all urls for reviews and ratings    
    all_reviews, all_ratings, all_titles, location = parse_url_list(url_list)

    hotel_df = make_reviews_df(all_reviews, all_ratings, all_titles, location)
    return hotel_df






# def get_next_page_ext(soup):
#     """For 

#     Args:
#         soup (bs4.BeautifulSoup parser): [description]

#     Returns:
#         [str]: [description]
#     """

#     next_page_link = soup.find('a', class_=compile("ui_button nav next (.*)"))
#     if next_page_link:
#         return next_page_link['href']
#     else:
#         return None

# from time import sleep

# returns results from all pages using "next" button  
# # this one is blasting out and being rejected????   
# def parse_url_pages(start_url, reviews=[], ratings=[], quiet=False):
#     root = "https://www.tripadvisor.com"
#     sleep(5)
#     html = requests.get(root + start_url)
#     sleep(5)
#     soup = BeautifulSoup(html.content, 'html.parser')
#     page_reviews, page_ratings = retrieve_reviews_ratings(soup)
#     reviews += page_reviews
#     ratings += page_ratings
#     next_ext = get_next_page_ext(soup)
#     i = 1
#     if next_ext:  # if next_ext exists, i.e. not at end of pages yet
#         if quiet == False:
#             print("Page:", i, next_ext)
#         next_url = root + next_ext
#         i += 1
#         return parse_url(next_url, reviews, ratings)
#         # recursive, calls on self to continue with function to parse on NEXT page
#     else: # at end of pages, stop looping, return final lists
#         return reviews, ratings