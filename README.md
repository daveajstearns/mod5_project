# Flatiron School Data Science Capstone Project
----
## Abstract
  
----
## Introduction
The goal of this project is to utilize the skills obtained over the past three months of hard work and dedication in this intense data science bootcamp. Since the beginning of my time in the program, I have been exploring the concept of **urban farming** with consideration to my background in biology and agriculture. Urban farming is a complex subject; a revolutionary approach to agriculture due to its non-renewable resource reduction (water) but with a caveat in regards to the energy usage urban farms need in order to be feasible. An urban farm that has impact on its community is a farm that can continuously produce fresh produce year round without interruption. A simple garden on top of a building or in an amenity space, as we have seen in some areas, is not sufficient. The concept of urban agriculture is an industry disrupting idea, but one that I feel must take place in order to reduce food insecurity and transportation emissions, improve food safety, create food independence, and a host of other issues related to the difficulties we face with traditional ag.  
  ![alt text](https://github.com/daveajstearns/mod5_project/blob/david-stearns/images/gg_bk.jpg "Gotham Greens, Brooklyn")
    
A final package will be presented at a later date to include a more in depth analysis of the trends, space, and some of the applicable technology. **This package**, however, strictly deals with the idea of *urban farming* as a topic over time. It has been said that this concept is one of the hot topics of the decade. 




According to the data, *its true!*
![alt text](https://github.com/daveajstearns/mod5_project/blob/david-stearns/images/pre_post_2010.png "Interest in Urban Ag")
  
**But how true is that?** And moreover, can we expect people to be more aware of urban agriculture in the future? That is what this package aims to answer. I took the top five terms related to `indoor farming` and used `General Mills pytrends` - *a pseudo-API for Google Trends* - to get their interest scores dating as far back as Google allows (2004). Read on through to the end, open the notebooks on a second screen, and introduce yourself to **_urban agriculture_**.  
  
## Procedure

### Data Aquisition  
The first step in my approach to understanding how interest in urban farming has changed over time was to start pulling data. I knew the data would be relatively easy to obtain using `pytrends`. I did not anticipate the difficulty associated with using it efficiently and correctly. The user guide [can be found here.](https://github.com/GeneralMills/pytrends)  
  
In order to get historical interest from Google Trends, I created a function called `get_int()` to retrieve this information. It would return a data frame with the selected time interval and the interest index within that time frame. I think a brief explanation of the *interest index* is warranted here. What you need to understand is that Google's software will calculate how popular that particular term is within that time frame, relative to itself. So there are no conflicting factors between keywords. They aren't, in some way, a time series against some other entity. This scoring is based off of itself throughout that time interval. It is also arbitrary, and ranges from 0 - 100.  
  
With this in mind, I felt it was possible to combine many search terms together and average these scores. Point being that it is possible to search *vertical farms or indoor farming* and be thinking about *urban agriculture*, for example. This is where the other functionality of `pytrends` comes in. There is a method called `.related_topics()` which will do what it says - get related topics. I wanted to build a list around *indoor farming* because at the end of the day, urban agriculture is and will largely be indoor farming. In order to get this method to work, you need to `.build_payload()` which builds a filter to search through records.  
`pytrends.build_payload(['indoor farming'], geo='US')` I set the geography to US. While learning how to use `pytrends` I found the Netherlands to be a place where urban farming and indoor farming are extremely hot search trends. This makes sense when considering they are the leader in all things indoor farming. This payload was used as a filter, and then called `pytrends.related_topics()` to get the related topics. Side note: *Dutch language* was one of the top 25.  
  
There were many interesting topics, but I decided to go with:  
1) `greenhouse`
2) `hydroponic`
3) `vertical farming`
4) `urban agriculture`
5) `aeroponics`  
(Aeroponics is really cool, and you can find aeroponically, vertically grown veggies at your local Whole Foods from AeroFarms in Newark, NJ.)  
![alt text](https://github.com/daveajstearns/mod5_project/blob/david-stearns/images/aero_farms2.jpg "AeroFarms, Newark, NJ")  
  
I used my `get_int()` function to get the historical interest in these topics between [2004-01-01 : 2020-05-18], including *indoor farming*. The series were added together and averaged to create the master series, `master`. This would be the data that the forecasting model was built off of. 


### Data Exploration  
The first line of business was to get a visual of the time series by itself.
![alt text](https://github.com/daveajstearns/mod5_project/blob/david-stearns/images/full_view.png "Master Time Series Graph")  
  
One noticeable point to make about this graph is that there seems to be repetitive bumps and that just by looking at it, this topic seems to have increased in popularity after 2010. Refer back to the Intro where I stated that urban agriculture has been coined a topic of the decade. Naturally, I wanted to explore this. With the use of a two-sampled Welsh's T-Test, I was able to determine the difference in mean between pre-2010 and post-2010 to be statistically different. You can see that graph of the difference above.  
  
Next I wanted to explore the trend of this data and the seasonality. The first graph showed signs of a repetitive nature. I chose to make a function called `szn_decomp()` which could take in a host of parameters and give you a graph using values generated from the `statsmodels.tsa.seasonal.seasonal_decompose`. I chose to show the additive model, but the multiplicative model would give roughly the same figure. The main difference is that the additive model can tell you "the interest level will be x many points higher next year" whereas the multiplicative model will tell you that "the interest level will be x% higher next year".  
![alt text](https://github.com/daveajstearns/mod5_project/blob/david-stearns/images/additive_model.png "Seasonal Decomposition - Additive Model")  
This helped me understand that the seasonality peaks annually, and that my **m** value for my SARIMA model should be 12.  
  
I also looked at the ACF and PACF graphs to help confirm my intuitions and see how many orders of AR and MA I could include in my model, potentially. 
![alt text](https://github.com/daveajstearns/mod5_project/blob/david-stearns/images/acf.png "ACF")  
![alt text](https://github.com/daveajstearns/mod5_project/blob/david-stearns/images/pacf.png "PACF")  
  
The annual seasonality is present here as well.  
  
### Modeling
