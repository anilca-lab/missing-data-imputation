### Imputing Missing Data

A model is only as good as the data used to build it: garbage in, garbage out. This is why the entire process of proper data preparation and more specifically, handling of missing data is crucial. In the age of "big data", simply discarding missing data and focusing on complete cases may look like a no-brainer. But, more often than not, the process generating the missingness is not completely random. Therefore, limiting an analysis to complete cases is likely to yield biased results.

Complete randomness aside, there are three types of missingness generating mechanisms:

1. The cases where the probability of missingness depends on available information are called missingness at random. An example would be blood pressure data having missing values for young patients who have no cardiovascular disease.
2. There are cases where the probability of missingness depends on information that is unobserved. For instance, when a medical treatment causes discomfort, patients may be more likely to opt out of undertaking this treatment.
3. Lastly, the probability of missingness may be determined by the values of the variable in question itself. An example would be a survey-based earnings series having missing values because respondents with higher earnings do not want to reveal them.

Any decision on how to handle missing data depends on the type of process generating the missingness. The analysis below takes a particular case, missingness at random, and examines the bias introduced by this process and if a random regression approach can be used to address this bias. To this end, the analysis uses an unconditional income convergence model for its simplicity. In this model, a country's income growth is explained by its initial income level, adjusted for cross-country price differentials. The two series used in the analysis are from the World Bank's WDI: average real per capita GDP growth (1995–2018) and PPP adjusted per capita GDP in 1995. The dataset covers 180 countries.
