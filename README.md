# highRES | Nordic Energy Outlooks

This repository contain a project-specific version of highRES, used in the Nordic Energy Outlooks (NEO) project, funded by Nordic Energy Research. This project has an increased focus on the Nordic region and its contribution to a wider european future energy system. Due to the remoteness of Iceland, it is not included in the model. As such, any further use of the term 'Nordics' refer to the four countries Norway, Sweden, Denmark and Finland. 

For an overview of highRES, please see the main branch. Here we specify mainly the changes made to the model compared to the main branch. 

## Data
This model uses the following general data sources:

- ERA5 for on and offshore wind capacity factors and runoff data (<https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5>)
- CMSAF-SARAH2 for solar PV capacity factors (<https://wui.cmsaf.eu/safira/action/viewDoiDetails?acronym=SARAH_V002>)
- Demand data is from ENTSO-E for the year 2013
- Cost and technical data is taken from UKTM (<https://www.ucl.ac.uk/energy-models/models/uk-times>) with data from the JRC report "Cost development of low carbon energy technologies: Scenario-based cost trajectories to 2050" (<https://publications.jrc.ec.europa.eu/repository/handle/JRC109894>) used to update some areas.
- Data on run-of-river, reservoir and pumped hydro power capacities is taken from <https://transparency.entsoe.eu/>, <https://www.entsoe.eu/data/power-stats/> and <https://github.com/energy-modelling-toolkit/hydro-power-database>
- Energy storage capacities for reservoir and pumped storage are taken from Schlachtberger et al. (2017) and Geth et al. (2015) respectively (see <https://doi.org/10.1016/j.energy.2017.06.004> and <https://doi.org/10.1016/j.rser.2015.07.145>).

### Cutoff factor
When estimating average capacity factors of a region, we do not want to include grid cells which are very poor for wind/solar, as those cells would not be considered for developments. As such, we operate with a so called 'cutoff factor', which operates as a lower threshold for which grid cells are included. They are 9, 15 and 20% for solar, onshore wind and offshore wind respectively. 

### Nuclear
In our hybrid greenfield approach, we consider only Nuclear power plants which, based on its lifetime, will remain in 2050. Other than those capacities (which are fixed), the model is free to install in new nuclear capacities, if it is considered cost-optimal.

The following countries have a minimum lower Nuclear capacity:
* Finland | Olkiluoto 3 (1600 MW, Finland)
* France | Flamanville 3 (1630 MW France)
* Slovakia | Mochovce 3 & 4 (471 MW each, Slovakia) 
* UK | Hinkley Point C (3250 MW, UK)

### Transmission
The model's ability to invest in transmission infrastructure is an important constraint to the model, as leaving it unrestricted leads to huge capacity investments in specific locations (e.g. between Spain and France). To prevent inflated numbers, but without being too conservative, we limit the transmission expansion to three times the capacity planned by 2030 in the ENTSO-E [Ten Year Network Development Plan (TYNDP)](https://eepublicdownloads.blob.core.windows.net/public-cdn-container/tyndp-documents/TYNDP2020/FINAL/entso-e_TYNDP2020_Main_Report_2108.pdf).  

### Weather year
We are running only with the weather year of 2010, which is a difficult weather year. This is a significant limitation that could heavily bias the results. 

### Demand assumptions
Demand time series are originally based on historical data from the European Network of Transmission System Operators for Electricity (ENTSO-E) Transparency Platform [40], but need to be adjusted to account for inconsistencies and missing data. [van der Most](https://doi.org/10.1016/j.rser.2022.112987) uses climate data and applies a logistic smooth transmission regression (LSTR) model to the ENTSO-E dataset to correlate historical electricity demand to temperature and generate daily electricity demand for a set of European countries. Subsequently, [Frysztacki, van der Most and Neumann](https://zenodo.org/record/7070438) uses hourly profiles from the [Open Power Systems Database](https://doi.org/10.25832/time_series/2020-10-06) to disaggregate the daily electricity demand to an hourly resolution, on a country level. We further take this data and scale it by a factor of two, based on a suggested possible increase in electricity demand by (source), while leaving the shape of the load curve untouched. With an elevated focus on the Nordic region, we use specific demand scenarios by the Nordic TSOs (rather than a crude doubling as for the rest of Europe).

For **Norway** the demand is assumed to be 220 TWh, according to the baseline (basis) scenario of Statnett [LMA Norway 2050](https://www.statnett.no/globalassets/for-aktorer-i-kraftsystemet/planer-og-analyser/lma/forbruksutvikling-i-norge-2022-2050---delrapport-til-lma-2022-2050.pdf) (their scenarios range between 190-300 TWh). For **Sweden**, we assume that the 2050 electricity demand is 298 TWh, a significant increase [from about 136 TWh](https://www.scb.se/hitta-statistik/sverige-i-siffror/miljo/elektricitet-i-sverige/). The scenario (Elektrifiering förnybart) assumes net-zero emissions in Sweden and a high increase in electricity demand from industrial activities. For **Finland**, [electricity demand scenarios from the Finnish TSO](https://www.fingrid.fi/globalassets/dokumentit/en/news/electricity-market/2023/fingrid_electricity_system_vision_2023.pdf), Fingrid, differ considerably, ranging from 131-359 TWh. We assume a modest increase in electricity demand, from 87 to 131 (Local Power secenario). Lastly, the **Danish** TSO assume a large increase in electricity demand, mainly as a result of significant Power-to-X capacity, reaching 213 TWh (up from 39 TWh) ([Energistyrelsen, 2023](https://ens.dk/sites/ens.dk/files/Statistik/af23_-_sammenfatningsnotat.pdf)). Due to this, we treat the electricity demand in Denmark as the rest of Europe, increase by a factor of 2. 
