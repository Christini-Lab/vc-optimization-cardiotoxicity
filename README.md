## An in silico-in vitro pipeline for drug cardiotoxicity screening identifies ionic proarrhythmia mechanisms


By: Alexander P. Clark, Siyu Wei, Darshan Kalola, Trine Krogh-Madsen, David J. Christini

#### Run the VC optimization GA and visualize the results
1. Run the GA with **ga_run.py**. The results are saved to **ga_results/trial_steps_ramps_Kernik_200_51_4_-120_60**.
2. Run `ga_plot_short_protocols.py` to see each of the optimized protocols along with the windows where they are maximized. These figures will be saved to your folder in `ga_results`.
3. Run `ga_make_whole_protocol.py` to shorten each protocol and connect them to form an optimized VC protocol. The final optimized VC protocol will be saved to `shortened_trial_steps_ramps_Kernik_200_50_4_-120_60_holding_500_True.csv`.
4. Run `ga_plot_opt_protocol.py` to visualize the optimized protocol.

#### Plot manuscript figures

1. `cd` into `/figures`. 
2. Download all raw experimental data from [HERE](www.FILLTHISIN.com) and extract the data inside `/figures`.
3. Run figure scripts to generate plots for all main manuscript and supplemental figures.
