# Flatiron School Data Science Capstone Project
----
## Abstract  
The models made in this project utilized the SARIMA function of `auto_arima` to find the best parameters for the time series. The time series data was harvested using the `General Mills pytrends` pseudo-API for Google Trends. It was found that there was a statistically significant increase in interest in topics regarding *indoor farming* from pre-2010 to post-2010 - thus achieving its *topic of the decade* status. I created a custom cross validation function to test the robustness of my models and evaluation. I found that after cross validation, the model with the lowest error was not the best fit model.
  
----  
## Table of Contents  
* `updated_search_sarima.ipynb` - where you will find my data mining, analysis, and modeling
* `mod5_functions.py` - functions that I made to make working easier
* `images` - a folder with all of my images  
  
The rest of the entries are preliminary notebooks that I made to get used to the data and begin side projects for the next part of the project.  
  
----
## Introduction
The goal of this project is to utilize the skills obtained over the past three months of hard work and dedication in this intense data science bootcamp. Since the beginning of my time in the program, I have been exploring the concept of **urban farming** with consideration to my background in biology and agriculture. Urban farming is a complex subject; a revolutionary approach to agriculture due to its non-renewable resource reduction (water) but with a caveat in regards to the energy usage urban farms need in order to be feasible. An urban farm that has impact on its community is a farm that can continuously produce fresh produce year round without interruption. A simple garden on top of a building or in an amenity space, as we have seen in some areas, is not sufficient. The concept of urban agriculture is an industry disrupting idea, but one that I feel must take place in order to reduce food insecurity and transportation emissions, improve food safety, create food independence, and a host of other issues related to the difficulties we face with traditional ag.  
  ![alt text](https://github.com/daveajstearns/mod5_project/blob/david-stearns/images/gg_bk.jpg "Gotham Greens, Brooklyn"). 
  Figure 1

It has been said that this concept is one of the hot topics of the decade. 

According to the data, *its true!*
![alt text](https://github.com/daveajstearns/mod5_project/blob/master/images/decade_interest.png "Interest in Urban Ag")
  
**But how true is that?** And moreover, can we expect people to be more aware of urban agriculture in the future? That is what this package aims to answer. I took the top five terms related to `indoor farm` and used `General Mills pytrends` - *a pseudo-API for Google Trends* - to get their interest scores dating as far back as Google allows (2004). Read on through to the end, open the notebooks on a second screen, and introduce yourself to **_urban agriculture_**.  
  
----
## Procedure & Results

### Data Aquisition  
The first step in my approach to understanding how interest in urban farming has changed over time was to start pulling data. I knew the data would be relatively easy to obtain using `pytrends`. I did not anticipate the difficulty associated with using it efficiently and correctly. The user guide [can be found here.](https://github.com/GeneralMills/pytrends)  
  
In order to get historical interest from Google Trends, I created a function called `get_int()` to retrieve this information. It would return a data frame with the selected time interval and the interest index within that time frame. I think a brief explanation of the *interest index* is warranted here. What you need to understand is that Google's software will calculate how popular that particular term is within that time frame, relative to itself. So there are no conflicting factors between keywords. They aren't, in some way, a time series against some other entity. This scoring is based off of itself throughout that time interval. It is also arbitrary, and ranges from 0 - 100.  
  
With this in mind, I felt it was possible to combine many search terms together and average these scores. Point being that it is possible to search *vertical farms or indoor farming* and be thinking about *urban agriculture*, for example. This is where the other functionality of `pytrends` comes in. There is a method called `.related_topics()` which will do what it says - get related topics. I wanted to build a list around *indoor farming* because at the end of the day, urban agriculture is and will largely be indoor farming. In order to get this method to work, you need to `.build_payload()` which builds a filter to search through records.  
`pytrends.build_payload(['indoor farm'], geo='US')` I set the geography to US. While learning how to use `pytrends` I found the Netherlands to be a place where urban farming and indoor farming are extremely hot search trends. This makes sense when considering they are the leader in all things indoor farming. This payload was used as a filter, and then called `pytrends.related_topics()` to get the related topics. Side note: *Dutch language* was one of the top 25.  
  
There were many interesting topics, but I decided to go with:  
1) `greenhouse`
2) `hydroponic`
3) `vertical farm`
4) `urban agriculture`
5) `aeroponic`   
These search terms were checked using `FuzzyWuzzy` to make sure I wasn't using a form of those terms that might lend itself to more data.
(Aeroponics is really cool, and you can find aeroponically, vertically grown veggies at your local Whole Foods from AeroFarms in Newark, NJ.)  
![alt text](https://github.com/daveajstearns/mod5_project/blob/david-stearns/images/aero_farms2.jpg "AeroFarms, Newark, NJ") Figure 2  
  
I used my `get_int()` function to get the historical interest in these topics between [2004-01-01 : 2020-05-18], including *indoor farm*. The series were added together and averaged to create the master series, `master`. This would be the data that the forecasting model was built off of. 


### Data Exploration  
The first line of business was to get a visual of the time series by itself.
![alt text](https://github.com/daveajstearns/mod5_project/blob/master/images/time_series_summary.png "Master Time Series Graph")  
  
One noticeable point to make about this graph is that there seems to be repetitive bumps and that just by looking at it, this topic seems to have increased in popularity after 2010. Refer back to the Intro where I stated that urban agriculture has been coined a topic of the decade. Naturally, I wanted to explore this. With the use of a two-sampled Welsh's T-Test, I was able to determine the difference in mean between pre-2010 and post-2010 to be statistically different. You can see that in the graph of rolling stats in the Intro.  
  
Next I wanted to explore the trend of this data and the seasonality. The first graph showed signs of a repetitive nature. I chose to make a function called `szn_decomp()` which could take in a host of parameters and give you a graph using values generated from the `statsmodels.tsa.seasonal.seasonal_decompose`. I chose to show the additive model, but the multiplicative model would give roughly the same figure. The main difference is that the additive model can tell you "the interest level will be x many points higher next year" whereas the multiplicative model will tell you that "the interest level will be x% higher next year".  
![alt text](https://github.com/daveajstearns/mod5_project/blob/master/images/seasonality.png "Seasonal Decomposition - Additive Model")  
This helped me understand that the seasonality peaks annually, and that my **m** value for my SARIMA model should be 12.  
  
I also looked at the ACF and PACF graphs to help confirm my intuitions and see how many orders of AR and MA I could include in my model, potentially. 
![alt text](https://github.com/daveajstearns/mod5_project/blob/master/images/acf.png "ACF")  
![alt text](https://github.com/daveajstearns/mod5_project/blob/master/images/pacf.png "PACF")  
  
The annual seasonality is present here as well.  
  
### Modeling  
**For all of my modeling, I used a nearly 90:10 split by training on data between [2004 : 2017] (end of 2017) and testing on the rest of the data [2018 : 2020-05]. All models were asked to predict until 2024.**  
  
My first model was a naive baseline model using only one order of differencing. The results are as follows:  
![alt text](https://github.com/daveajstearns/mod5_project/blob/master/images/gif/baseline.png "Baseline Model")  
![alt text](https://github.com/daveajstearns/mod5_project/blob/master/images/baseline_stats.png "Baseline Model Stats")  

I would like to point out that the histogram shows the distribution to be a little off-center from the normal curve, and thus lends to the notion that this model isn't the best. Also view the forecast graph and you will see the confidence interval is horrendous, as is the forecast.  
  
My second model was conceptualized based on the EDA, specifically the seasonal decomposition and ACF graphs.  
![alt text](https://github.com/daveajstearns/mod5_project/blob/master/images/gif/eda_model.png "EDA Based Model")  
![alt text](https://github.com/daveajstearns/mod5_project/blob/master/images/eda_model_stats.png "EDA Based Model Stats")  
  
The QQ plot and histogram looks slightly improved, but the confidence interval is still ghastly. The forecast looks nice, but I cannot be confident in this model. The CI dips below 0 and above 100, and so I cannot support this model.   
  
After these two baseline models, I wanted to explore the `auto_arima` tool. I had been wanting to get this to work for a while and finally had it running. It is a tool that can basically gridsearch SARIMA parameters and base the best parameters off of whatever you set the information criterion to. I chose AIC for the purposes of this project. AIC, or Akaike Information Criterion, is a relative quality of fit metric that can help you find which model best fits the data. While this is an important metric, I also wanted to look at the MAE (mean absolute error) and RMSE (root mean squared error) as they both have a place in my evaluation sights. RMSE penalizes large errors, which I care about because I want my model to be accurate, and MAE is a bit more interpretable as it is simply the summed average mean of the absolute values of error.  
  
Here is an example of how to run an `auto_arima` search and get the best parameters:  
```
model = auto_arima(master['2004':'2017'], trace=True, start_p=0, start_q=0, d=1,
                  start_P=0, start_Q=0, seasonal=True, m=12, suppress_warnings=True, 
                   D=1, error_action='ignore', approximation=False, trend='t', random_state=42)
fitted = model.fit(master['2004':'2017'])  
best_params = fitted.get_params()
print('\n\nThe best order parameters are {},{}\n'.format(best_params['order'],best_params['seasonal_order']))
```
The parameters used for this model are **(0,1,1)x(0,1,1,12)**.  
![alt text](https://github.com/daveajstearns/mod5_project/blob/master/images/gif/automod_1.png "First Auto-Arima Based Model")  
  
Notice the tight confidence interval, decent in-sample and out-of-sample forecasts, and how the forecast post-data seems to follow a similar trend to 2010+. The stats also look good:  
![alt text](https://github.com/daveajstearns/mod5_project/blob/master/images/automod_1_stats.png "First Auto-Arima Based Model Stats")   
  
The QQ plot is tightly bound to the diagonal and the histogram shows a very normal distribution. Excellent model.   
  
Here is my second model. Parameters: **(0,1,1)x(2,0,2,12)**.  
![alt text](https://github.com/daveajstearns/mod5_project/blob/master/images/gif/automod_2.png "First Auto-Arima Based Model")    
![alt text](https://github.com/daveajstearns/mod5_project/blob/master/images/automod_2_stats.png "First Auto-Arima Based Model")  
  
I made a third model available, as the next two iterations with auto_arima produced already seen results. The third model's parameters are **(0,1,1)x(1,0,1,12)**.  
![alt text](https://github.com/daveajstearns/mod5_project/blob/master/images/gif/automod_3.png "First Auto-Arima Based Model")    
![alt text](https://github.com/daveajstearns/mod5_project/blob/master/images/automod_3_stats.png "First Auto-Arima Based Model") 
 
### Model Evaluation  
  
#### AIC Scores: Winner - auto_model_1  
![alt text](https://github.com/daveajstearns/mod5_project/blob/david-stearns/images/aic_comparison.png "Model AIC Comparison")  


#### MAE Scores: Winner - auto_model_2
![alt text](https://github.com/daveajstearns/mod5_project/blob/david-stearns/images/mae_comparison.png "Model MAE Comparison")  

    
#### Test RMSE Scores: Winner - auto_model_1
![alt text](https://github.com/daveajstearns/mod5_project/blob/david-stearns/images/test_rmse_comparison.png "Model Test RMSE Comparison")  


**A full breakdown of the winning events:**  
![alt text](https://github.com/daveajstearns/mod5_project/blob/david-stearns/images/results.png "Final Results")    
  
**A full breakdown of each model's percent improvement over the baseline.**
![alt text](https://github.com/daveajstearns/mod5_project/blob/master/images/percent_improvements.png "Percent Improvment Over Baseline") 

---
### Cross Validation   
Once this was finished, I looked into how I could validate the robustness of my model evaluations. I only tested the validity of my models on the 90:10 split I mentioned before. I developed a sliding window cross-validation function using `sklearn TimeSeriesSplit`, seen in my notebook as `cross_val_ts` which would return five lists: RMSE, AIC, MAE, MAPE, and BIC - in that order. This function allowed me to get cross validated evaluation metrics for my models, and actually swayed some of the final conclusions. I specified **10** cross validation splits, however, you can do more or less and it will still work the same.  
  
**The results from cross validation are below.**
![alt text](https://github.com/daveajstearns/mod5_project/blob/master/images/cv_aic2.png "CV AIC")  
![alt text](https://github.com/daveajstearns/mod5_project/blob/master/images/cv_bic2.png "CV BIC")  
![alt text](https://github.com/daveajstearns/mod5_project/blob/master/images/cv_mae2.png "CV MAE")  
![alt text](https://github.com/daveajstearns/mod5_project/blob/master/images/cv_mape2.png "CV MAPE")  
![alt text](https://github.com/daveajstearns/mod5_project/blob/master/images/cv_test_rmse2.png "CV TEST RMSE")  
   
**Tabular results from cross validation.**  
![alt text](https://github.com/daveajstearns/mod5_project/blob/david-stearns/images/cv_eval_metrics.png "RESULTS")
  
In order to evaluate these new numbers, I had to get a little creative with the AIC and BIC values. With regards to RMSE and MAE, I could average the results. Regarding AIC and BIC, it would be incorrect to average these results because these numbers inherently get larger as the data set gets larger. Refer back to the graphs above. I decided that I would look at these graphs as curves, and concluded that I would use **area under curve** as a way to evaluate these two metrics. The lower the area under the curve, the lower the AIC/BIC score is throughout cross validation, which leads to a better fit model to the data. As expected, `auto_model_1` took the win in these two categories, however what was not expected, was `auto_model_2` being the winner for test RMSE. In the previous evaluation, `auto_model_1` was the winner with respect to test RMSE. This throws a small wrench in the machine, but I welcome the newfound data because it allows me to think deeper about what I want out of a model. Do I want a model that will predict the values with 100% accuracy? Or do I want a model that is well fit to this type of data and will perform well over time? I still stand by `auto_model_1` as my best fit model.
  
---
## Conclusions  
The final model selected for this project was the **auto_model_1** which had the lowest metrics where it counted; goodness of fit to the data. While the **auto_model_2** did have a lower cross validated Test RMSE and MAE, I don't believe pinpoint accuracy to be of utmost importance here. I want to know that my model will predict the interest level over time with reasonable accuracy, and most importantly, maintain its integrity over time. 
  
The knowledge gained from this portion of this project shows that it is possible to guage the interest in a subject over time correctly. Now more in line with the business case, this tells us that **urban agriculture** interest is going to increase over time. We have already seen facilities like AeroFarms, Gotham Greens, and Bowery Farming penetrate the high-end produce market with customers such as Whole Foods. The ability to grow food close to the destination with roughly 5% of the water that traditional agriculture would use is not only astonishing, but clearly a topic we need to rally around. We can create more energy, and if the public would change their perspective on nuclear energy, keeping those LED lights and HVAC systems running all year shouldn't be an issue. **Water** is a big issue. It is non-renewable, we have a finite amount of it, and even more so, *usable water*. We have to protect our valuable resources, nourish our people, and provide opportunity for human development. I strongly believe urban agriculture is a step in that direction.  
  
Please follow along, as I plan on incorporating some more advanced techniques to this time series model. Last but not least, I plan on showing you all that we can create rooftop greenhouses all across Manhattan with robots inside planting, maintaining, harvesting, and packaging.  
![alt text](https://github.com/daveajstearns/mod5_project/blob/david-stearns/images/bowery.png "Bowery Farming's Mission")
Figure 3 
  
---
### Citations
Figure 1: https://www.gothamgreens.com/our-farms/
Figure 2: https://aerofarms.com/
Figure 3: https://boweryfarming.com/our-produce
