
# Commented out IPython magic to ensure Python compatibility.
# %pip install apyori
# %pip install pandas
# %pip install mlxtend
# %pip install openpyxl



import pandas as pd
from apyori import apriori
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
from itertools import combinations
from tabulate import tabulate
import time


print("Hi there!!\n")

user_input = int(input("Enter the store number you want to move on:: \n1. Amazon \n2. Apple \n3. Best Buy \n4. Costco \n5. Nike \n"))

datasets = ('Amazon', 'Apple', 'Best Buy', 'Costco', 'Nike')

if user_input < 1 or user_input > len(datasets):
    print("You select invalid value!!, retry")
    quit()

def select_store(store):
    data = ''
    if store == 1:
        data = 'amazon.csv'
    elif store == 2:
        data = 'apple.csv'
    elif store == 3:
        data = 'bestbuy.csv'
    elif store == 4:
        data = 'costco.csv'
    elif store == 5:
        data = 'nike.csv'

    return data

print(f"You select {select_store(user_input)}")
df = pd.read_csv(select_store(user_input))
try:
    support_input = int(input("\nEnter the minimum support value between 1 and 100 :: "))
    if support_input < 1 or support_input > 100:
        raise ValueError("Invalid support value.")

    confidence_input = int(input("Enter the confidence level between 1 and 100 :: "))
    if confidence_input < 1 or confidence_input > 100:
        raise ValueError("Invalid confidence value.")

except ValueError as e:
    print(f"Error: {e}. Please retry...")
    quit()
print(f"\nEnter support value: {support_input} \nEnter confidence value: {confidence_input}")

#now we remove all null or empty transaction row from the data set and also split transtion into list. For example: if out transction is A,B,C -> [A, B, C]
df = df[df['Transaction'].apply(lambda x: x.strip() != '' )]
transactions = df['Transaction'].apply(lambda x: [item.strip() for item in x.split(',')]).tolist()


support = support_input/100
confidence = confidence_input/100

print(f"\nNew support value: {support} \nNew confidence value: {confidence}")


def brute_force(transactions, support, confidence):
    total_tran = len(transactions)
    support_count = support * total_tran

    freq_item = {}
    main_itemsets = {}
    association_rules = []

    # Generate frequency set
    for elem in transactions:
        for length in range(1, len(elem) + 1):
            for i in combinations(elem, length):
                i = tuple(sorted(i))  # this avoids duplicates like {'A', 'B'} and {'B', 'A'}
                if i in freq_item:
                    freq_item[i] += 1
                else:
                    freq_item[i] = 1

    # Filtering frequent itemsets with minimum support
    for itemset, count in freq_item.items():
        if count >= support_count:
            main_itemsets[itemset] = count

    # Generate association rules
    for itemset in main_itemsets:
        if len(itemset) > 1:
            for i in range(1, len(itemset)):
                for left in combinations(itemset, i):
                    right = tuple(sorted(set(itemset) - set(left)))  # A -> B
                    union = tuple(sorted(left + right))  # A ∪ B

                    support_union = main_itemsets.get(union, 0)
                    support_left = main_itemsets.get(left, 0)

                    if support_left > 0:
                        support_confidence = support_union / support_left
                        if support_confidence >= confidence:
                            association_rules.append((left, right, support_confidence))

    # Generating Association rules for output
    final_list = []
    for A, B, confi in association_rules:
        ans_str = f"{{{', '.join(A)}}} → {{{', '.join(B)}}}, Confidence: {confi * 100:.2f}%"
        final_list.append(ans_str)

    return final_list


def apriori_algo(transactions, support, confidence):
    # Implement the apriori algorithm
    ans = list(apriori(transactions, min_support=support, min_confidence=confidence))

    # Generating Association rules for output
    final_list = []
    for elem in ans:
        for stat in elem.ordered_statistics:
            left = ', '.join(stat.items_base)
            right = ', '.join(stat.items_add)
            confidence = stat.confidence * 100
            support = elem.support * 100
            if left:
                ans_str = f"{{{left}}} → {{{right}}}, Confidence: {confidence:.2f}%"
                final_list.append(ans_str)

    return final_list


def fp_growth_tree_algo(transactions, support, confidence):

    # Use TransactionEncoder to Encode the transactions
    encoder = TransactionEncoder()
    encoder_array = encoder.fit(transactions).transform(transactions)
    encoder_dataframe = pd.DataFrame(encoder_array, columns=encoder.columns_)

    # geting frequent itemsets using fpgrowth
    frequent_itemsets = fpgrowth(encoder_dataframe, min_support=support, use_colnames=True)

    if frequent_itemsets.empty:
        print("Don't have frequent itemsets for support")
        return []

    # Generate the association rules
    association_rules_df = association_rules(frequent_itemsets, metric="confidence", min_threshold=confidence)

    if association_rules_df.empty:
        print("Don't have any association rules for itemsets")

    # rules for output
    final_list = []
    for index, row in association_rules_df.iterrows():
        antecedents = ', '.join(row['antecedents'])
        consequents = ', '.join(row['consequents'])
        confidence = row['confidence'] * 100  # Convert confidence to percentage
        and_str = f"{{{antecedents}}} → {{{consequents}}}, Confidence: {confidence:.2f}%"
        final_list.append(and_str)

    return final_list


algo_choice = int(input("Select Algorithm of your choise:: \n1. Brute Force Algoritham\n2. Apriori Algorithm\n3. FP Tree Algoritham\n4. For All Algorithms \n"))

# function that run Brute Force algo.
def run_brute_force(transactions, support, confidence):
    starting_time = time.perf_counter()
    rules = brute_force(transactions, support, confidence)
    ending_time = time.perf_counter()
    result_time = ending_time - starting_time
    return rules, result_time

# function that run Apriori algo.
def run_apriori(transactions, support, confidence):
    starting_time = time.perf_counter()
    rules = apriori_algo(transactions, support, confidence)
    ending_time = time.perf_counter()
    result_time = ending_time - starting_time
    return rules, result_time

# function that run FP groth tree algo.
def run_fp_tree(transactions, support, confidence):
    starting_time = time.perf_counter()
    rules = fp_growth_tree_algo(transactions, support, confidence)
    ending_time = time.perf_counter()
    result_time = ending_time - starting_time
    return rules, result_time


# function which get input from user and run their prefered algo and provide output:
while True:
    if algo_choice == 1:
        bf_ans_str, bf_time = run_brute_force(transactions, support, confidence)
        for i, rule in enumerate(bf_ans_str, 1):
            print(f"{i}. {rule}")
        print(f"\nTime taken by Brute Force algoritham: {bf_time:.7f} seconds")

    elif algo_choice == 2:
        ap_ans_str, ap_time = run_apriori(transactions, support, confidence)
        for i, rule in enumerate(ap_ans_str, 1):
            print(f"{i}. {rule}")
        print(f"\nTime taken by Apriori algoritham: {ap_time:.7f} seconds")

    elif algo_choice == 3:

        fp_ans_str, fp_time = run_fp_tree(transactions, support, confidence)
        for i, rule in enumerate(fp_ans_str, 1):
            print(f"{i}. {rule}")
        print(f"\nTime taken by FP Tree algoritham: {fp_time:.7f} seconds")

    elif algo_choice == 4:

        print("\nRunning Brute Force Algorithm...\n")
        bf_ans_str, bf_time = run_brute_force(transactions, support, confidence)
        print("-"*30)
        for i, rule in enumerate(bf_ans_str, 1):
            print(f"{i}. {rule}")
        print(f"Brute Force time taken: {bf_time:.7f} seconds")

        print("\nRunning Apriori Algorithm...")
        ap_ans_str, ap_time = run_apriori(transactions, support, confidence)
        print("-"*30)
        for i, rule in enumerate(ap_ans_str, 1):
            print(f"{i}. {rule}")
        print(f"\nApriori time taken: {ap_time:.7f} seconds")

        print("\nRunning FP Tree Algorithm...")
        fp_ans_str, fp_time = run_fp_tree(transactions, support, confidence)
        print("-"*30)
        for i, rule in enumerate(fp_ans_str, 1):
            print(f"{i}. {rule}")
        print(f"\nFP Tree time taken: {fp_time:.7f} seconds")

        # if user select all algotitham for run then comparison times
        all_times = {'Brute Force': bf_time, 'Apriori': ap_time, 'FP Tree': fp_time}

        time_data = [
            ["Brute Force", f"{bf_time:.7f}"],
            ["Apriori", f"{ap_time:.7f}"],
            ["FP-Growth", f"{fp_time:.7f}"]
        ]
        print(tabulate(time_data, headers=["Algorithm", "Time"], tablefmt="rounded_grid"))

        fastest_algo = min(all_times, key=all_times.get)
        print(f"\nFastest Algorithm: {fastest_algo} with a time of {all_times[fastest_algo]:.7f} seconds")
    break