""" This file contains functions to scrape hotel review data from Trip Advisor,
culminating in the final 'scrape_hotel' function. 01/14/2021 """

import pandas as pd

from bs4 import BeautifulSoup
from time import sleep
from re import compile

from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException, TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC



def get_driver():
    """Create instance of Selenium Chrome webdriver with the following settings:
    '--incognito' - Chrome Incognito mode, avoid saving cookies to the driver.
    '--headless' - Headless browsers run without a user interface (UI). Faster.
    '--ignore-certificate-errors' - ignore pop up SSL certificates.

    Returns:
        Instance of selenium.webdriver.chrome.webdriver.WebDriver
    """
    options = webdriver.ChromeOptions()
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--incognito')
    options.add_argument('--headless')
    driver = webdriver.Chrome(executable_path='./driver/chromedriver', options=options)
    return driver



def retrieve_reviews_ratings(soup, idx):
    """Retrieve rating, review, and review title from each of 5 reviews per page on 
    a hotel's TipAdvisor page.

    Args:
        soup (BeautifulSoup): html parser object

    Returns:
        page_reviews: list of strings 
        page_ratings: list of strings 
        page_titles: list of integers, 0 - 5 
    """
    # Set container holding review details
    container = soup.findAll('div', class_="_2wrUUKlw _3hFEdNs8")

    page_reviews = []
    page_ratings = []
    page_titles = []

    # Find all levels of rating
    rating_re = compile("ui_bubble_rating (.*)")

    for item in container:
    
        rating_raw = item.find('span', class_=rating_re)
        rating_int = int(rating_raw.attrs['class'][1].split("_")[1][-2])
        page_ratings.append(rating_int)

        review = item.find('q', class_="IRsGHoPm").text
        
        # Check for more text after "Read More" activated, complete review text
        expanded = item.find('span', class_="_1M-1YYJt")
        if expanded:
            review += expanded.text
        page_reviews.append(review)

        # Save review title
        title = item.find('a', class_='ocfR3SKN').text
        page_titles.append(title)

    # For monitoring during runtime
    print('page', idx + 1)
        
    return page_reviews, page_ratings, page_titles



def retrieve_location(soup):
    """Returns property information from the hotel page, including
    hotel name and location.

    Args:
        soup (BeautifulSoup): html parser object

    Returns:
        (tuple) : (full hotel name, city, state)
    """
    # Get hotel name
    hotel = soup.find(id="HEADING").text

    # Get location (city, state)
    raw = soup.find('span', class_="_3ErVArsu jke2_wbp").text.split(", ")
    city = raw[-2]
    state = ''.join([i for i in raw[-1] if not i.isdigit()]).split(" -")[0]
    
    return (hotel, city, state)



def parse_url(start_url_ext, idx, webdriver, location=False, _filter=False): 
    """Parse a Trip Advisor hotel page and scrape review information: rating, review, and 
    review title. Optional to scrape location details.

    Args:
        start_url_ext (str): Trip Advisor hotel page to parse
        idx (int): current page index, 0 through n, used for print statement
        webdriver (Selenium WebDriver): browser tool allowing for interaction with website
        location (bool, optional): Option to return location details. Defaults to False.

    Returns:
        page_reviews: list of strings
        page_ratings: list of strings
        page_titles: list of integers, 0 - 5 
        
        if location = True:
        location (tuple): (full hotel name, city, state) 
    """
    domain = "https://www.tripadvisor.com"

    # Define waits, moved from stale element 'try' 1/18pm
    ignored_exceptions = (NoSuchElementException, StaleElementReferenceException, TimeoutException)
    wait = WebDriverWait(webdriver, 10, ignored_exceptions=ignored_exceptions)

    # Catch for webdriver time out
    try:
        webdriver.get(domain + start_url_ext) # 1/18 reduced from 5 to 3

    except TimeoutException:
        pass

    if _filter == True:
        # ACTIVATE low filters to scrape only low reviews
        try:
            for f in [3, 2, 1]:
                # level = f"ReviewRatingFilter_{f}"
                # webdriver.find_element_by_id(level).click()
                level = f"ReviewRatingFilter_{f}"
                wait.until(EC.element_to_be_clickable((By.ID, level)))
                webdriver.execute_script("arguments[0].click();", (webdriver.find_element_by_id(level)))
                print(f"filter{f}")
        except:
            pass

    # Catch for webdriver stale element
    try:
        # ignored_exceptions = (NoSuchElementException, StaleElementReferenceException, TimeoutException)
        # wait = WebDriverWait(webdriver, 10, ignored_exceptions=ignored_exceptions)
        wait.until(EC.element_to_be_clickable((By.CLASS_NAME, "_3maEfNCR")))

    except TimeoutException:
        pass

    # Find 'read more' buttons
    all_more_buttons = webdriver.find_elements_by_class_name("_3maEfNCR")

    # If 'read more' available, activate to expand text, only need to click one
    if all_more_buttons:
        try:
            all_more_buttons[0].click()
            print('click')
        
        except StaleElementReferenceException:  
            pass
        
    # Set soup    
    page_source = webdriver.page_source
    soup = BeautifulSoup(page_source, 'html.parser')

    # Scrape the ratings data
    page_reviews, page_ratings, page_titles = retrieve_reviews_ratings(soup, idx)

    # If location data requested, gather it
    if location == False:
        return page_reviews, page_ratings, page_titles
    else:
        location = retrieve_location(soup)
        return page_reviews, page_ratings, page_titles, location
           


def get_url_list(start_url, n=2):
    """From a given Trip Advisor hotel homepage returns requested number of consecutive URLs
    according to the site's url pattern.

    Args:
        start_url (string): Trip Advisor hotel homepage / first main page ONLY.  
                            Not built for a mid start yet. 
        n (int): Number of URLs requested. Defaults to 2 to return start URL and next page.

    Returns:
        page_urls : List of URLs in increasing order from given homepage
    """
    # Remove root if included  
    domain = "https://www.tripadvisor.com"
    
    if len(start_url.split(domain)) != 1:
        start_url = start_url.split(domain)[1]

    # Sets url to access 'reviews' portion of page
    if start_url[-8:] != "#REVIEWS":
        start_url = start_url +"#REVIEWS"
    
    # Five reviews are displayed per page, adjust range
    n = (n - 1) * 5
    
    pages = range(0,n, 5)

    # Start list with start_url
    page_urls = [start_url]
    
    # Split start url to insert page numbers
    url_split = start_url.split('Reviews')

    # Generate desired pages according to pattern
    for i in pages:
        page_num = 'Reviews-or' + str(5 + i)
        next_url = url_split[0] + page_num + url_split[1]
        page_urls.append(next_url)

    return page_urls
   


def parse_url_list(url_list, webdriver=None, _filter=False):
    """Parses a given list of Trip Advisor hotel homepages to retrieve the review data and location.
    Review data includes the review, title, and rating between 0 - 5.

    Args:
        url_list (list): Hotel URL's to parse
        webdriver (Selenium.WebDriver, optional): Driver to use for scraping. Defaults to None.
                    If none provided, one will be instantiated.

    Returns:
        all_reviews (list): all reviews from the given list of URLS, as str
        all_ratings (list): all ratings from the given list of URLS, as int 
        all_titles (list): all titles from the given list of URLS, as str
        location (tuple): hotel name, city, state
        [type]: [description]
    """
    if not webdriver:
        driver = get_driver()
    else:
        driver = webdriver

    all_reviews = []
    all_ratings = []
    all_titles = []
    
    for idx, page in enumerate(url_list):

        # Get location just once with parse of first url
        if idx == 0:
            page_one_reviews, page_one_ratings, page_one_titles, location = parse_url(page, 
                                                                                        idx, 
                                                                                        webdriver=driver, 
                                                                                        location=True,
                                                                                        _filter=_filter)
            all_reviews.extend(page_one_reviews)
            all_ratings.extend(page_one_ratings)
            all_titles.extend(page_one_titles)

        else:    
            page_reviews, page_ratings, page_titles = parse_url(page, 
                                                                idx, 
                                                                webdriver=driver,
                                                                _filter=_filter)
            all_reviews.extend(page_reviews)
            all_ratings.extend(page_ratings)
            all_titles.extend(page_titles)
    
    driver.quit()

    return all_reviews, all_ratings, all_titles, location




def make_reviews_df(reviews, ratings, titles, location=None):
    """Converts review data into a DataFrame.

    Args:
        reviews (list): list of strings
        ratings (list): list of ints
        titles (list): list of strings
        location (tuple, optional): (hotel name, city, state). Defaults to None.

    Returns:
        pd.DataFrame: Review data vectorized into rows
    """
    df = pd.DataFrame([titles, reviews, ratings]).transpose()
    df.columns = ['Title', 'Review', 'Rating']
    
    if location:
        df['Hotel'] = location[0]
        df['Location'] = location[1] + ', ' + location[2]

        # Rearrange
        df = df[['Location', 'Hotel', 'Title', 'Review', 'Rating']]
    
    return df




def scrape_hotel(start_url, n=2, webdriver=None, _filter=False):
    """Complete hotel review scrape from Trip Advisor for a given number of pages.
    There are five reviews per page.

    Args:
        start_url (str): Starting URL. Hotel homepage from Trip Advisor.
        n (int): Number of pages to be scraped. Defaults to 2.
        webdriver (Selenium.WebDriver, optional): [description]. Defaults to None. 
                    If none provided, one will be instantiated.

    Returns:
        pd.DataFrame: Review data vectorized into rows. 
                      Column titles : 'Location', 'Hotel', 'Title', 'Review', 'Rating'
    """
    url_list = get_url_list(start_url, n=n)

    if webdriver:
        driver = webdriver

        all_reviews, all_ratings, all_titles, location = parse_url_list(url_list, 
                                                                        webdriver=driver,
                                                                        _filter=_filter)
    
    else:
        all_reviews, all_ratings, all_titles, location = parse_url_list(url_list,
                                                                        _filter=_filter)

    hotel_df = make_reviews_df(all_reviews, all_ratings, all_titles, location)
    
    return hotel_df
