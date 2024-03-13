from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
from math import floor
import Levenshtein
from math import floor, ceil
 

"""
Code for performing evaluation of the results. 
We get precision, recall, f1, accuracy, characters per second, parse checking, average score.
Also, we get metrics scores.
"""

def string_to_list(string: str) -> list:
    """Convert string to list"""
    first = string.split(" ")
    for w in first:
        if "|" in w:
            # Delete from original
            first.remove(w)

            # Split
            splitted = w.split("|")

            # Extend original
            first.extend(splitted)

    return first


def jaccard_set(list1: list, list2: list) -> float:
    """Define Jaccard Similarity function for two sets"""
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union

# Program to measure the similarity between
# two sentences using cosine similarity.

def find_similar(X: str, Y: str) -> float:
    """Find similar words in two strings"""

    X = X.replace("|", " ")
    Y = Y.replace("|", " ")

    # tokenization
    X_list = word_tokenize(X)
    Y_list = word_tokenize(Y)

    # sw contains the list of stopwords
    sw = stopwords.words('english')
    l1 =[];l2 =[]

    # remove stop words from the string
    X_set = {w for w in X_list if not w in sw}
    Y_set = {w for w in Y_list if not w in sw}

    print(X_set)
    print(Y_set)
    
    # form a set containing keywords of both strings
    rvector = X_set.union(Y_set)
    for w in rvector:
        if w in X_set: l1.append(1) # create a vector
        else: l1.append(0)
        if w in Y_set: l2.append(1)
        else: l2.append(0)
    c = 0

    # cosine formula
    for i in range(len(rvector)):
            c+= l1[i]*l2[i]
    cosine = c / float((sum(l1)*sum(l2))**0.5)
    # print("similarity: ", cosine)

    return cosine


def calculate_jacc(truth:str, prediction:str) -> list:
    """Calculate Jaccard score for truth and prediction"""

    list_of_thruths = truth.split("\n")
    list_of_preds = prediction.split("\n")

    try:
        list_of_thruths.remove('')
    except:
        pass

    try:
        list_of_preds.remove('')
    except:
        pass

    jacc_scores = []

    if len(list_of_thruths) == 0 and len(list_of_preds) > 0:
        print("Truth is zero.")
        jacc = 0
        jacc_scores.append(jacc)

    elif len(list_of_thruths) > 0 and len(list_of_preds) == 0:
        jacc = 0
        jacc_scores.append(jacc)

    elif len(list_of_thruths) == 0 and len(list_of_preds) == 0:
        jacc = 0
        jacc_scores.append(jacc)

    else:

        for t in list_of_thruths:

            # Find closest match in predictions
            max_cosine = 0
            for p in list_of_preds:

                # Get score
                cosine = find_similar(t, p)

                # Is it max?
                if cosine > max_cosine:
                    match_pred = p

            # Convert string to list
            ground_tokens_list = string_to_list(t)

            # convert matched pred to list
            pred_tokens_list = string_to_list(match_pred)

            # calculate actual jaccard score
            jacc = jaccard_set(ground_tokens_list, pred_tokens_list)

            jacc_scores.append(jacc)


    return jacc_scores

def isNaN(num) -> bool:
    """Check if a number is NaN"""
    return num != num


def simple_jaccard(truth, prediction) -> float:
    """Calculate simple Jaccard score for truth and prediction"""

    # Replace
    truth = truth.replace("\n", " ").replace("|", " ")
    prediction = prediction.replace("\n", " ").replace("|", " ")

    # Split by space
    ground_tokens_list = truth.split(" ")
    pred_tokens_list = prediction.split(" ")

    print(ground_tokens_list)
    print(pred_tokens_list)

    # Get score
    jacc = tokenize_and_jaccard_similarity(ground_tokens_list, pred_tokens_list)

    return jacc


def tokenize_and_jaccard_similarity(ground_tokens_list: list, pred_tokens_list: list) -> float:
    """Tokenize and calculate Jaccard similarity"""

    sw = stopwords.words('english')

    # Remove stop words
    ground_tokens_list = [x for x in ground_tokens_list if x not in sw]
    pred_tokens_list = [x for x in pred_tokens_list if x not in sw]

    print(ground_tokens_list)
    print(pred_tokens_list)

    if len(ground_tokens_list) == 0 and len(pred_tokens_list) == 0:
        return 1

    # Stem words
    stemmer = PorterStemmer()
    ground_tokens_list = [stemmer.stem(x) for x in ground_tokens_list]
    pred_tokens_list = [stemmer.stem(x) for x in pred_tokens_list]

    print(ground_tokens_list)
    print(pred_tokens_list)

    counter1 = Counter(ground_tokens_list)
    counter2 = Counter(pred_tokens_list)
    intersection = sum((counter1 & counter2).values())
    union = sum((counter1 | counter2).values())

    return intersection / union

def jaccard_for_property_name(truth: str, prediction: str) -> float:
    """Calculate Jaccard score for property name"""

    # Split each instance
    truth = truth.split("\n")
    prediction = prediction.split("\n")

    # Get only the property name
    truth = [x.split("|")[0] for x in truth]
    prediction = [x.split("|")[0] for x in prediction]

    # Remove empty strings
    truth = [x for x in truth if x != ""]
    prediction = [x for x in prediction if x != ""]

    ground_tokens_list = []
    for t in truth:
        
        # Split by space
        ground_tokens_list.extend(t.split(" "))

    pred_tokens_list = []
    for p in prediction:

        # Split by space
        pred_tokens_list.extend(p.split(" "))

    print(ground_tokens_list)
    print(pred_tokens_list)

    # Get score
    jacc = tokenize_and_jaccard_similarity(ground_tokens_list, pred_tokens_list)

    return jacc

# Function to calculate the
# Jaro Similarity of two strings
def jaro_distance(s1: str, s2: str) -> float:
    """Calculate Jaro Similarity of two strings"""
 
    # If the strings are equal
    if (s1 == s2) :
        return 1.0
 
    # Length of two strings
    len1 = len(s1)
    len2 = len(s2)
 
    if (len1 == 0 or len2 == 0) :
        return 0.0
 
    # Maximum distance upto which matching
    # is allowed
    max_dist = (max(len(s1), len(s2)) // 2 ) - 1
 
    # Count of matches
    match = 0
 
    # Hash for matches
    hash_s1 = [0] * len(s1)
    hash_s2 = [0] * len(s2)
 
    # Traverse through the first string
    for i in range(len1) :
 
        # Check if there is any matches
        for j in range( max(0, i - max_dist),
                    min(len2, i + max_dist + 1)) :
             
            # If there is a match
            if (s1[i] == s2[j] and hash_s2[j] == 0) :
                hash_s1[i] = 1
                hash_s2[j] = 1
                match += 1
                break
         
    # If there is no match
    if (match == 0) :
        return 0.0
 
    # Number of transpositions
    t = 0
 
    point = 0
 
    # Count number of occurrences
    # where two characters match but
    # there is a third matched character
    # in between the indices
    for i in range(len1) :
        if (hash_s1[i]) :
 
            # Find the next matched character
            # in second string
            while (hash_s2[point] == 0) :
                point += 1
 
            if (s1[i] != s2[point]) :
                point += 1
                t += 1
            else :
                point += 1
                 
        t /= 2
 
    # Return the Jaro Similarity
    return ((match / len1 + match / len2 +
            (match - t) / match ) / 3.0)
 
# Jaro Winkler Similarity
def jaro_Winkler(s1: str, s2: str) -> float:
    """Calculate Jaro Winkler Similarity"""
 
    jaro_dist = jaro_distance(s1, s2)
 
    # If the jaro Similarity is above a threshold
    if (jaro_dist > 0.7) :
 
        # Find the length of common prefix
        prefix = 0
 
        for i in range(min(len(s1), len(s2))) :
         
            # If the characters match
            if (s1[i] == s2[i]) :
                prefix += 1
 
            # Else break
            else :
                break
 
        # Maximum of 4 characters are allowed in prefix
        prefix = min(4, prefix)
 
        # Calculate jaro winkler Similarity
        jaro_dist += 0.1 * prefix * (1 - jaro_dist)
 
    return jaro_dist
 

# Function to calculate the
# Jaro Similarity of two s
def jaro_distance(s1: str, s2: str) -> float:
    """Calculate Jaro Similarity of two strings"""
     
    # If the s are equal
    if (s1 == s2):
        return 1.0
 
    # Length of two s
    len1 = len(s1)
    len2 = len(s2)
 
    # Maximum distance upto which matching
    # is allowed
    max_dist = floor(max(len1, len2) / 2) - 1
 
    # Count of matches
    match = 0
 
    # Hash for matches
    hash_s1 = [0] * len(s1)
    hash_s2 = [0] * len(s2)
 
    # Traverse through the first
    for i in range(len1):
 
        # Check if there is any matches
        for j in range(max(0, i - max_dist),
                       min(len2, i + max_dist + 1)):
             
            # If there is a match
            if (s1[i] == s2[j] and hash_s2[j] == 0):
                hash_s1[i] = 1
                hash_s2[j] = 1
                match += 1
                break
 
    # If there is no match
    if (match == 0):
        return 0.0
 
    # Number of transpositions
    t = 0
    point = 0
 
    # Count number of occurrences
    # where two characters match but
    # there is a third matched character
    # in between the indices
    for i in range(len1):
        if (hash_s1[i]):
 
            # Find the next matched character
            # in second
            while (hash_s2[point] == 0):
                point += 1
 
            if (s1[i] != s2[point]):
                t += 1
            point += 1
    t = t//2
 
    # Return the Jaro Similarity
    return (match/ len1 + match / len2 +
            (match - t) / match)/ 3.0
 

def calculate_similarity_for_value(truth: str, prediction: str) -> float:

    # Split each instance
    truth = truth.split("\n")
    prediction = prediction.split("\n")

    # Get only the 
    truth = [x.split("|")[1] for x in truth]
    prediction = [x.split("|")[1] for x in prediction]

    # Remove empty strings
    truth = [x for x in truth if x != ""]
    prediction = [x for x in prediction if x != ""]

    print(truth)
    print(prediction)



def perform_metrics(truth: str, prediction: str) -> list:
    """Perform metrics for truth and prediction"""

    list_of_thruths = truth.split("\n")
    list_of_preds = prediction.split("\n")

    try:
        list_of_thruths.remove('')
    except:
        pass

    try:
        list_of_preds.remove('')
    except:
        pass

    hallucination = 0
    should_have = 0
    correctly_detected = 0

    scores = []

    if len(list_of_thruths) == 0 and len(list_of_preds) > 0:
        print("Truth is zero.")
        scores.append([0, 0, 0])
        hallucination += len(list_of_preds)

    elif len(list_of_thruths) > 0 and len(list_of_preds) == 0:
        scores.append([0, 0, 0])
        should_have += len(list_of_thruths)

    elif len(list_of_thruths) == 0 and len(list_of_preds) == 0:

        # Scores for variable, values and units is going to be 1, however, this won't be counted
        scores.append([1, 1, 1])

        # No hallucination or should have
        hallucination += 0
        should_have += 0

        # No correctly detected
        correctly_detected += 0

    else:

        already_processed_truth = []

        for t in list_of_thruths:

            print("Current truth: ", t)
            print("Current predictions: ", list_of_preds)

            # If there are no predictios left, break
            if len(list_of_preds) == 0:
                break

            # Find closest match in predictions
            max_similarity = 0
            for p in list_of_preds:

                # Use only the value and unit for finding the match
                t_compare = t[t.find("|")+1:]
                p_compare = p[p.find("|")+1:]

                # Get score
                jaro_similarity = Levenshtein.ratio(t_compare, p_compare)

                print("For ", t_compare, " and ", p_compare, " the score is ", jaro_similarity, ".")

                # Is it max?
                if jaro_similarity > max_similarity:
                    max_similarity = jaro_similarity
                    match_pred = p

            if max_similarity < 0.5:
                print("No match found for ", t)
                # scores.append([0, 0, 0])
                # should_have += 1
                continue

            else:
                correctly_detected += 1



            # Remove from list of predictions
            print("Trying to remove: ", match_pred, " from ", list_of_preds)
            list_of_preds.remove(match_pred)

            # add to already processed
            already_processed_truth.append(t)

            print("Truth is : ", t, " and matched prediction: ", match_pred)

            print("List of truths: ", list_of_thruths)
            print("List of predictions: ", list_of_preds)

            # Get variable names
            t_variable = t.split("|")[0]
            p_variable = match_pred.split("|")[0]

            # Jaro similarity for variable
            jaro_similarity_variables = Levenshtein.ratio(t_variable, p_variable)

            print("Variable score for truth ", t, " and prediction ", match_pred, " is ", jaro_similarity_variables, ".")

            # Get values
            t_value = t.split("|")[1]
            p_value = match_pred.split("|")[1]

            # Jaro similarity for value
            jaro_similarity_values = Levenshtein.ratio(t_value, p_value)

            print("Value score for truth ", t, " and prediction ", match_pred, " is ", jaro_similarity_values, ".")
            
            # Get units
            try:
                # If there is no unit, set score to 0
                t_unit = t.split("|")[2]
                p_unit = match_pred.split("|")[2]

                # Jaro similarity for unit
                jaro_similarity_units = Levenshtein.ratio(t_unit, p_unit)

            except Exception as e:
                print("Couldn't extract the unit, setting score to 0: ", e)
                jaro_similarity_units = 0

            print("Unit score for truth ", t, " and prediction ", match_pred, " is ", jaro_similarity_units, ".")

            scores.append([jaro_similarity_variables, jaro_similarity_values, jaro_similarity_units])


        # Add hallucinations
        hallucination += len(list_of_preds)

        # Add should have
        should_have += len(list_of_thruths) - len(already_processed_truth)

        print("\n")

    # Calculate scores
    score_variables = sum([x[0] for x in scores])/len(scores) if len(scores) > 0 else 0
    score_values = sum([x[1] for x in scores])/len(scores) if len(scores) > 0 else 0
    score_units = sum([x[2] for x in scores])/len(scores) if len(scores) > 0 else 0

    print("Score variables: ", score_variables)
    print("Score values: ", score_values)
    print("Score units: ", score_units)

    print("Hallucination: ", hallucination)
    print("Should have: ", should_have)
    print("Correctly detected: ", correctly_detected)

    return score_variables, score_values, score_units, hallucination, should_have, correctly_detected


def parse_checking(prediction: str) -> bool:
    """Check if the parsing is correct"""

    # Split by new line
    prediction = prediction.split("\n")

    # Remove empty lines
    prediction = [x for x in prediction if x != ""]

    # There must be 3 | in each line
    for pred in prediction:
        if pred.count("|") != 2:
            return False
        
    return True


def get_total_number_of_truth(truth: str) -> int:
    """Get total number of truth"""
    total_number_of_truth = 0
    for t in truth.split("\n"):
        if t != "":
            total_number_of_truth += 1
    return total_number_of_truth

def get_total_number_of_predictions(prediction : str) -> int:
    """Get total number of predictions"""
    total_number_of_predictions = 0
    for p in prediction.split("\n"):
        if p != "":
            total_number_of_predictions += 1
    return total_number_of_predictions


