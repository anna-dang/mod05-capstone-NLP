Anna D'Angela

Flatiron Data Science Program

Module 5 Project - Capstone

January 22nd, 2021

---

# Hotel Review Sentiment Classifier

*Natural Language Processing Communication Management Tool*

 <img alt="a highly rated hotel experience" src="./images/danny.gif" width="400"/>
 
---

### Overview

Stakeholder: Company building a filter for hotel email/social media feeds to recognize low sentiment/upset customers to flag and bring to staff attention. 

End user: Hotel/hospitality industry

Business problem: Customer retention

Business solution:
1) Brand management: Word of mouth repuation is just as important as published reviews. Filter aims to eventually scan twitter, facebook, etc. for mentions of property and flag negative sentiment for staff to address.
2) Email filter: Flag emails containing negative hotel sentiment/issues and flag for quick response by staff. 

---

### Methodology

#### Skills Demonstrated

- Webscraping: BeautifulSoup, Selenium
- EDA: Pandas, Plotly, World CLoud, 
- NLP: NLTK, LIME
- Classification: Sci-Kit Learn pipelines - Logistic Regression, Naive Bayes, Stochastic Gradient Descent

#### Data

Source/size/scope: Trip Advisor hotel review webscrape to gather 22,563 reviews from 24 hotels in the Denver metro-area. See hotel list [hotel](./data/coordinates.csv). Reviews provide a text with a label sentiment score between 1 (wors) and 5 (best).

Limitations: Proof of concept focused on metro Denver area to keep sites/atractions mentioned consistent. The scoring metric is entirely up to the user. Therefor, one person's 3 could be another person's 5.

<img alt="data source map" src="./images/map.png" width="400"/>

###### Data Contents
```
clean_scrape.csv        # final, cleaned, combined scrapes
denver_urls.txt         # list to URLs to scrape
coordinates.csv         # coordinate data for scraped hotels

scrape_3.csv            # 1200 from URL 4
scrape_4.csv            # 2000 from URL 5 - 6
scrape_5.csv            # 3995 from URL 7 - 10
scrape_6.csv            # 3930 from URL 1 - 4
scrape_8.csv            # 3000 from url 11 - 13
scrape_9.csv            # 9995 from URL 14 - 18
scrape_10.csv           # 1196 from URL 18 - 25

detroit_test_urls.txt   # sample set of URLs used to build scrape function
test_data.csv           # resulting test datt to build collection notebook
```

 <img alt="class imbalance" src="./images/class_balance.png" width="400"/>


#### Processing and Modeling

- NLP/Vectorizer info

<img alt="class imblancep" src="./images/word_clouds.png" width="400"/>

- Model journey, params/tuning

- Final model analysis/scores

<img alt="final model scores" src="./images/final_cm.png" width="400"/>

<img alt="text explainer" src="./images/text_explainer.png" width="400"/>

---

### Reccomendations

1. Use the binary model to flag guestt communication in email or social media from prompt attention from staff.
2. Use the 5 class rating classifier to rank order of importance.
3. Expand proof of concept to other user specific location (remove location specific vocabulary).

### Future Work

1. Try Scrapy for webscraping to increase speed of scrape.
2. Explore transfer learning with pre-trained NLP models.
3. Explore deep learning with embeddings and neural networks.

--- 

#### Thank you!

View my presentation [slideshow](/od05_presentation.pdf) and [blog](https://annadangela.medium.com/) for this project.

Connect with me on [LinkedIn](https://www.linkedin.com/in/anna-d-angela-216b01b2/) and [Twitter](https://twitter.com/_dangelaa)!

#### Repository Contents
```
.
├── README.md
├── .gitignore
├── data                    
├── images
├── models
├── capstone_functions
│   ├── NLP_functions.py         
│   ├── collection_functions.py       
│   └── __init__.py        
├── data_collection.ipynb
├── NLP_and_modeling.ipynb
└── mod05_presentation.pdf
