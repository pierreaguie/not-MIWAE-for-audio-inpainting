# not-MIWAE-for-audio-inpainting

Missing values in a dataset is an issue that can be complicated to handle. The solution is often to use models adapted to incomplete data or to try to reconstruct the missing data from what is known. Different patterns of missingness exist for missing data, the most general one being \textbf{MNAR} (Missing Not At Random) which assumes that the probability of a data being missing knowing the data depend both on the missing data and the observed data. This is the setting explored in "Not-MIWAE: Deep Generative Modelling With Missing Not At Random Data" by Ipsen et al., which introduces a DLVM specifically desgined to handle MNAR data. In this project, we apply the not-MIWAE framework to audio declipping.

##Results
Here is a plot showing the results of the imputation and the spectrograms of a reconstructed audio and the original version.
![Imputation](assets/plots/imputation_vs_original.pdf)
![Spectrograms](assets/plots/Spectrograms.pdf)

