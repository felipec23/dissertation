from evaluate_results_utils import *
import pandas as pd
import os


RESULTS_PATH = 'chatgpt_ten_shot.xlsx'
results = pd.read_excel(RESULTS_PATH)

# If the cell is empty, set to 'empty|empty|empty'
results["truth"] = results["truth"].fillna("empty|empty|empty")
results["prediction"] = results["prediction"].fillna("empty|empty|empty")

# If  there's a column called "Unnamed: 0", remove it
if "Unnamed: 0" in results.columns:
    results = results.drop(columns=["Unnamed: 0"])

# Fill nan with empty string
results["truth"] = results["truth"].fillna("")
results["prediction"] = results["prediction"].fillna("")

# Calculate jaccard scores
results["simple_jaccard"] = results.apply(lambda x: simple_jaccard(x["truth"], x["prediction"]), axis=1)

# Calculate jaccard scores for property name
results["jaccard_property_name"] = results.apply(lambda x: jaccard_for_property_name(x["truth"], x["prediction"]), axis=1)

# Perform metrics
results["score_variables"], results["score_values"], results["score_units"], results["hallucination"], results["should_have"], results["correctly_detected"] = zip(*results.apply(lambda x: perform_metrics(x["truth"], x["prediction"]), axis=1)) 

# Get total number of truth
results["total_number_in_truth"] = results["truth"].apply(lambda x: get_total_number_of_truth(x))

# Get total number of correct predictions
# results["total_number_of_correct_predictions"] = results["total_number_in_truth"] - results["hallucination"] - results["should_have"]

# Get total number of predictions
results["total_number_of_predictions"] = results["prediction"].apply(lambda x: get_total_number_of_predictions(x))

# Calculate precision
results["precision"] = results["correctly_detected"] / results["total_number_of_predictions"]

# Fix precision
results["precision"] = results.apply(lambda x: 1 if x["total_number_in_truth"] == 0 and x["total_number_of_predictions"] == 0 else x["precision"], axis=1)

# Calculate recall, it's the same as: correct / correct + should have
results["recall"] = results["correctly_detected"] / results["total_number_in_truth"] 

# For recall: if correctly detected is 0 and total number in truth is 0, then recall is 1
results["recall"] = results.apply(lambda x: 1 if x["correctly_detected"] == 0 and x["total_number_in_truth"] == 0 else x["recall"], axis=1)

# Calculate f1
results["f1"] = 2 * ((results["precision"] * results["recall"]) / (results["precision"] + results["recall"]))

# If precision and recall are 0, then f1 is 0
results["f1"] = results.apply(lambda x: 0 if x["precision"] == 0 and x["recall"] == 0 else x["f1"], axis=1)

# Get accuracy
results["accuracy"] = results["correctly_detected"] / (results["correctly_detected"] + results["hallucination"] + results["should_have"])

# Fix accuracy
results["accuracy"] = results.apply(lambda x: 1 if x["correctly_detected"] == 0 and x["hallucination"] == 0 and x["should_have"] == 0 else x["accuracy"], axis=1)

# Check parsing
results["parse_checking"] = results.apply(lambda x: parse_checking(x["prediction"]), axis=1)

# Sum input + output length
results["input_output_length"] = results["input"].apply(lambda x: len(x)) + results["prediction"].apply(lambda x: len(x))

# Divide input + output length on time taken
results["characters_second"] = results["input_output_length"] / results["time_taken"]

# Get characters per second of output
results["characters_second_output"] = results["prediction"].apply(lambda x: len(x)) / results["time_taken"]

# Get average of scores
results["average_score"] = (results["score_variables"] + results["score_values"] + results["score_units"])/3

#%%

EVALUATION_FILENAME = RESULTS_PATH.split(".")[0] + "_evaluation.xlsx"
# Save results
results.to_excel(os.path.join("evaluation", EVALUATION_FILENAME ), index=False)

# Precision, recall, f1, accuracy, characters per second
precision_mean = results["precision"].mean()
recall_mean = results["recall"].mean()
f1_mean = results["f1"].mean()

# Count number of FALSE in parse checking
parse_checking_false = results["parse_checking"].value_counts()[False]

# Divide by total number of rows
parse_checking_false_percentage = parse_checking_false / len(results)

parse_checking_true = results["parse_checking"].value_counts()[True]

# Divide by total number of rows
parse_checking_true_percentage = parse_checking_true / len(results)

# Get average of characters per second output
characters_second_output_mean = results["characters_second_output"].mean()

# Sum of hallucination, should have and correctly detected
hallucination_sum = results["hallucination"].sum()

should_have_sum = results["should_have"].sum()

correctly_detected_sum = results["correctly_detected"].sum()


# For the scores we will drop rows with 0
# We also drop the rows that are empty in thruth and prediction
# For those that were detected correctly, we will get the score mean and std

filtered = results[
    (results["score_variables"] != 0) & 
    (results["score_values"] != 0) & 
    (results["score_units"] != 0) & 
    (results["truth"] != "empty|empty|empty") & 
    (results["prediction"] != "empty|empty|empty")
    ]

# Create a new dataframe with the statistics
score_values = filtered["score_values"].mean()
score_units = filtered["score_units"].mean()
score_variables = filtered["score_variables"].mean()

# Average of average score
average_score = filtered["average_score"].mean()

model = os.path.splitext(EVALUATION_FILENAME)[0].rsplit("_", 1)[0]
# Add all to a dataframe
statistics = pd.DataFrame({
    "model": [model],
    "precision_mean": [precision_mean],
    "recall_mean": [recall_mean],
    "f1_mean": [f1_mean],
    "hallucination_sum": [hallucination_sum],
    "should_have_sum": [should_have_sum],
    "correctly_detected_sum": [correctly_detected_sum],
    "parse_checking_false_percentage": [parse_checking_false_percentage],
    "parse_checking_true_percentage": [parse_checking_true_percentage],
    "characters_second_output_mean": [characters_second_output_mean],
    "score_values": [score_values],
    "score_units": [score_units],
    "score_variables": [score_variables],
    "average_score": [average_score]
})

# Save statistics
STATISTICS_FILENAME = RESULTS_PATH.split(".")[0] + "_statistics.xlsx"
statistics.to_excel(os.path.join("statistics", STATISTICS_FILENAME), index=False)



# %%
# Join all the results in one dataframe
# Get all the files in the folder
files = os.listdir("statistics")

# Create a list of dataframes
dfs = []

# Iterate over the files
for file in files:
    
        # Read the file
        df = pd.read_excel(os.path.join("statistics", file))
    
        # Append to list
        dfs.append(df)

# Concatenate all the dataframes
df = pd.concat(dfs)