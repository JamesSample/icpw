# ICP Waters

Data cleaning and analysis for the [ICP Waters](http://www.icp-waters.no/) Programme.

## TOC trends (1990 - 2012)

Getting to grips with the ICPW database and recoding/extending Tore's original work on TOC trends.

  1. **Database clean-up**. Identifying and fixing issues with the ICPW database identified during 2016 
      * [Part 1](http://nbviewer.jupyter.org/github/JamesSample/icpw/blob/master/toc_trends_2015_data_cleaning.ipynb)
      * [Part 2](http://nbviewer.jupyter.org/github/JamesSample/icpw/blob/master/toc_trends_2015_data_cleaning2.ipynb)
      * [Part 3](http://nbviewer.jupyter.org/github/JamesSample/icpw/blob/master/toc_trends_2015_data_cleaning3.ipynb).
  
  2. **Trend analyses**. Estimate trends using data from 1990 to 2012
      * [Part 1](http://nbviewer.jupyter.org/github/JamesSample/icpw/blob/master/toc_trends_oct_2016.ipynb)
      * [Part 2](http://nbviewer.jupyter.org/github/JamesSample/icpw/blob/master/toc_trends_oct_2016_part2.ipynb)
      * [Part 3](http://nbviewer.jupyter.org/github/JamesSample/icpw/blob/master/toc_trends_oct_2016_part3.ipynb)

## TOC trends (1990 - 2016)

A major review of the ICPW database was undertaken during 2018/19. The work can be divided into two components: (i) a expanded analysis comprising 430 "TOC trends" stations, with the aim of updating the [work published in 2007](https://www.nature.com/articles/nature06316); and (ii) a clean-up of the ~260 "core" ICP Waters stations, with the aim of writing a more general "Thematic Report" on TOC trends during 2019.

### TOC trends paper

For the broader "TOC Trends" work, a new project was created, site selections were reviewed and datasets expanded to cover the period 1990 to 2016. Trends were then recalculated using the improved datasets. 

 * [Part 1 - Data upload](http://nbviewer.jupyter.org/github/JamesSample/icpw/blob/master/toc_trends_oct_2018_part1.ipynb)
 * [Part 2 - Chemistry trends](http://nbviewer.jupyter.org/github/JamesSample/icpw/blob/master/toc_trends_oct_2018_part2.ipynb)
 * [Part 3 - Climate trends](http://nbviewer.jupyter.org/github/JamesSample/icpw/blob/master/toc_trends_oct_2018_part3.ipynb)
 
 A selection of tidied/processed datasets used in the subsequent analysis is [here](https://github.com/JamesSample/icpw/tree/master/trends_paper_datasets).
 
### Thematic report
 
Data for the "core" ICP Waters project(s) was first reviewed and tidied, and then checked for consistency with the data provided for the broader trends paper (above). Corrections were made as necessary, including replacing/substituting stations in some countries in order to make use of the best long-term monitoring datasets currently available.

 * **Update "trends" dataset**. The latest work using the "trends" dataset is documented above under the heading *"TOC Trends paper"*
 
 * **[Update "core" dataset](https://nbviewer.jupyter.org/github/JamesSample/icpw/blob/master/toc_report_feb_2019_part1.ipynb)**. Adding recent data for the "core" stations and dealing with data issues
 
 * **[Combining the "core" and "trends" datasets](https://nbviewer.jupyter.org/github/JamesSample/icpw/blob/master/toc_report_feb_2019_part3.ipynb)**. An overview of the unified ICPW dataset of 556 stations
 
 * **[Scatterplots of annual data](https://nbviewer.jupyter.org/github/JamesSample/icpw/blob/master/toc_report_feb_2019_part4_wge_plots.ipynb)**. A high-level overview of the raw data, aggregated to annual medians
 
 * **[Stations with high frequency monitoring](https://nbviewer.jupyter.org/github/JamesSample/icpw/blob/master/toc_report_feb_2019_part5_hi_freq.ipynb)**. Ten stations have substantially more detailed monitoring than the others (approximately 25 to 100 samples per year from 1990 to 2016). This notebook performs trend and change point analyses based on *monthly* data, using algorithms that are too "data hungry" to be applied elsewhere
 
 * **[Stations with "standard" monitoring](https://nbviewer.jupyter.org/github/JamesSample/icpw/blob/master/toc_report_feb_2019_part6.ipynb)**. Trend and change point analyses using *annually* aggregated data
 
 * **[Summary of work for Task Force meeting](https://nbviewer.jupyter.org/github/JamesSample/icpw/blob/master/toc_report_feb_2019_part7.ipynb)**. An overview of the analysis prior to the Task Force meeting in Helsinki in June 2019
