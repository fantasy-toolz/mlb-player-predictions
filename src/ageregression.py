
def return_age_factors():
    # penalty if missing: disabled for now
    year_weights_penalty = {}
    year_weights_penalty[2017.0] = 0.00
    year_weights_penalty[2018.0] = 0.0#5
    year_weights_penalty[2019.0] = 0.0#1
    year_weights_penalty[2020.0] = 0.0#5
    year_weights_penalty[2021.0] = 0.0#5
    year_weights_penalty[2022.0] = 0.0#5
    year_weights_penalty[2023.0] = 0.0#5
    year_weights_penalty[2024.0] = 0.0#5
    year_weights_penalty[2025.0] = 0.0#5

    #consider age factors
    # for hitters, call the falloff at age 33:
    # de-weight anything after 33 with a penalty increasing with age
    age_penalty_slope = 0.07 # I think 0.1 is AGGRESSIVE
    age_pivot         = 64.0
    return year_weights_penalty, age_penalty_slope, age_pivot

