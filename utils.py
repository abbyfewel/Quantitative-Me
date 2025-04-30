from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from scipy.stats import t 
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("daily_data.csv") # the daily data, 100+ instances
weekly_df = pd.read_csv("data.csv") # the weekly data, summarized totals
classes_df = pd.read_csv("class_totals.csv") # the data by class

'''
This function takes assignments from the daily data and 
makes a pie chart to display how many of each assignment were
given out over the semester.
'''
def assignments_pie_chart():
    online_count = df["Online Task"].sum() # how many total online assignments were done
    paper_count = df["Paper Task"].sum() # how many total paper assignments were done
    test_count = df["Tests"].sum() # how many total tests were taken
    quiz_count = df["Quizzes"].sum() # how many total quizzes were taken
    readings_count = df["Number of Readings"].sum() # how many readings I did
    pie_ser =[online_count, paper_count, test_count, quiz_count, readings_count] # series to use in the chart
    labels = ["Online Tasks", "Paper Tasks", "Tests", "Quizzes", "Readings"]
    plt.figure()
    plt.title("Number of Assignments Done Over the Semester")
    plt.pie(pie_ser, labels = labels, autopct = "%1.1f%%") # the parameters for the pie chart, %1.1f%% adds percentage labels for clarity
    plt.savefig("assignments_pie_chart")
    plt.show()

'''
This function takes the assignments from daily data and makes a 
bar graph out of it to display how many of each assignment was done over 
the weeks of the semester. 
'''
def assignments_per_week():
    df.fillna(0, inplace = True) # for any missing data, data handling
    df.columns = df.columns.str.strip() # cleaning
    assignment_columns = ["Online Task", "Paper Task", "Tests", "Quizzes","Number of Readings"]
    df["Total Assignments"] = df[assignment_columns].sum(axis=1) # summing up all the assignments done
    weekly_totals = df.groupby("Week")["Total Assignments"].sum().reset_index()
    plt.figure()
    plt.title("Assignments Done Per Week") 
    plt.bar(weekly_totals["Week"], weekly_totals["Total Assignments"], color = "pink") # the week and the total assigments (and pink yay)
    plt.xlabel("Week")
    plt.ylabel("Total Assignments Done")
    plt.xticks(df["Week"]) # marked by week
    plt.tight_layout()
    plt.show()
    plt.savefig("assignments_per_week.png")

'''
This function takes the assignments given by each class and 
shows them as a pie chart to see how busy each class made my semester and 
contributed to the overall total.
'''
def assignments_per_class():
    classes_df = pd.read_csv("class_totals.csv") # reading the classes dataframe
    labels = classes_df["class"] # labels for the classes
    totals = classes_df["total"]
    plt.figure()
    plt.title("Number of Assignments Done Per Class")
    plt.pie(totals, labels = labels, autopct = "%1.1f%%") # taking the totals, the labels are assigned
    plt.show()
    plt.savefig("assignments_per_class.png")

'''
This function takes in the daily data and sums of the tests and quizzes to 
display them in a bar chart to see if there appears to be a difference
between the amount of quizzes vs tests taken.
'''
def tests_v_quizzes():
    quiz_total = df["Quizzes"].sum() # quizzes
    test_total = df["Tests"].sum() # tests
    categories = ["Quizzes", "Tests"] # the categories for the two bars on the chart
    values = [quiz_total, test_total] # values are the totals
    plt.figure()
    plt.bar(categories, values) # using categories and values
    plt.title("Amount of Quizzes and Tests Taken this Semester")
    plt.xlabel("Category")
    plt.ylabel("Amount taken")
    plt.show()

'''
This function takes in the weekly data and makes a line chart
to display my trends of reading pages and reading books for classes over
the semester.
'''
def pages_per_week():
    weekly_df = pd.read_csv("data.csv") # getting the weekly data
    weekly_df = weekly_df.dropna(subset = "Pages Read") # cleaning it
    plt.plot(weekly_df["Week"], weekly_df["Pages Read"]) # plotting the week and the pages read
    plt.xlabel("Week")
    plt.ylabel("Pages Read")
    plt.title("Pages Read Each Week of the Semester")
    plt.show()

'''
This function tests to see if there is a significant difference in the amount of quizzes
versus tests taken throughout the semester. The data is digested as being quizzes per week and tests per week,
inspired by the bar graph we made. It tests to see if we took significantly more quizzes than tests. 
'''
def tests_quizzes_sig():
    quizzes = df["Quizzes"]
    quizzes_sum = quizzes.sum() # how many quizzes I took this semester
    tests = df["Tests"]
    tests_sum = tests.sum() # how many tests I took this semester
    count = weekly_df["Week"].count() # the number of weeks there are because I am testing by week, not by day
    print("Quizzes total:", quizzes_sum)
    print("Tests total:", tests_sum, "\n")
    # average per week
    xbar_quizzes = quizzes_sum / count # finding average quizzes per week
    print("Quizzes average per week: ", xbar_quizzes)
    xbar_tests = tests_sum / count # finding average tests per week
    print("Tests average per week: ", xbar_tests, "\n")
    s_quizzes = quizzes.std() # the standard deviations
    s_tests = tests.std()
    print("Quizzes standard deviation:", s_quizzes)
    print("Tests standard deviation: ", s_tests)
    n_quizzes = len(quizzes) # getting lengths to use in the degrees of freedom for equations
    n_tests = len(tests)
    alpha = 0.05
    deg_free = n_quizzes + n_tests - 2
    t_critical = t.ppf(1-alpha/2, deg_free) # t-critical to compare with our t stat
    print("\nT critical: ", t_critical)
    # t stat:
    t_stat = (xbar_quizzes - xbar_tests) / np.sqrt((s_quizzes**2 / n_quizzes) + (s_tests**2 / n_tests))
    print("T-statistic: ", t_stat)
    if abs(t_stat) > t_critical: # decision rule
        print("Rejecting the null hypothesis. \nThere is a significant difference in amount of quizzes and tests.")
    else:
        print("Fail to reject the null hypothesis: not enough evidence for a difference.")

'''
This function displays a chart that helps us visualize the amount
of assignments done in stem classes versus my non-stem classes.
'''
def stem_v_non_stem():
    stem_classes = ["Biology", "Biology Lab", "Computer Science", "Data Science"]
    non_stem_classes = ["Business", "History", "Philosophy"]
    stem_df = classes_df[classes_df["class"].isin(stem_classes)] # this will get me the classes i put in stem_classes from classes_df
    non_stem_df = classes_df[classes_df["class"].isin(non_stem_classes)] # this will get me the classes i put in non_stem_classes from classes_df
    print("Stem classes: \n", stem_df)
    print("\nNon-stem classes: \n", non_stem_df)
    stem_totals = stem_df["total"].sum() # getting the sum of the stem assignments
    non_stem_totals = non_stem_df["total"].sum() # getting the sum of the non stem assignemnts
    print("\n", stem_totals, non_stem_totals)
    plt.figure()
    plt.pie([stem_totals, non_stem_totals], labels=["STEM", "Non-STEM"], autopct="%1.1f%%") # printing a pie chart with the totals data with labels and percentages
    plt.title("Stem Assignments Verus Non-STEM Assignments")
    plt.show()

def stem_v_non_stem_sig():
    stem_classes = ["Biology", "Biology Lab", "Computer Science", "Data Science"]
    non_stem_classes = ["Business", "History", "Philosophy"]
    stem_df = classes_df[classes_df["class"].isin(stem_classes)] # this will get me the classes i put in stem_classes from classes_df
    non_stem_df = classes_df[classes_df["class"].isin(non_stem_classes)] # this will get me the classes i put in non_stem_classes from classes_df
    print("Stem classes: \n", stem_df)
    print("\nNon-stem classes: \n", non_stem_df)
    stem_totals = stem_df["total"].sum() # getting the sum of the stem assignments
    non_stem_totals = non_stem_df["total"].sum() # getting the sum of the non stem assignemnts
    count = stem_totals + non_stem_totals # the number of assignments there are is the total
    print("Stem total assignments:", stem_totals)
    print("Non-Stem total assignments:", non_stem_totals, "\n")
    # average per week
    xbar_stem = stem_df["total"].mean()
    xbar_non_stem = non_stem_df["total"].mean()
    print("Average number of assignments of STEM classes: ", xbar_stem)
    print("Average number of assignments of non-STEM classes:  ", xbar_non_stem, "\n")
    s_stem = stem_df["total"].std()
    s_non_stem = non_stem_df["total"].std()
    n_stem = stem_totals # getting lengths to use in the degrees of freedom for equations
    n_non_stem = non_stem_totals
    alpha = 0.05
    deg_free = n_stem + n_non_stem - 2
    t_critical = t.ppf(1-alpha/2, deg_free) # t-critical to compare with our t stat
    print("\nT critical: ", t_critical)
    # t stat:
    t_stat = (xbar_stem - xbar_non_stem) / np.sqrt((s_stem**2 / n_stem) + (s_non_stem**2 / n_non_stem))
    print("T-statistic: ", t_stat)
    if abs(t_stat) > t_critical: # decision rule
        print("Rejecting the null hypothesis. \nThere is a significant difference in amount of STEM versus non-STEM assignments.")
    else:
        print("Fail to reject the null hypothesis: not enough evidence for a difference.")

'''
This function takes in my fall 2025 and spring 2026 schedule and helps me predict what the
course load will look like; average, few, or many assingments per class.
'''
def semester(classifying_df, semester_df):
    print(semester_df) # printing it to see what classes I will be taking
    classifying_df = classifying_df.dropna(axis=1) # dropping rows with mising data.
    cleaned_df = classifying_df.transpose() # transpose the classifying_df
    cleaned_df.columns = classifying_df["Property"] # columns are the property
    cleaned_df = cleaned_df.drop("Property")
    X = cleaned_df.drop("Work", axis=1) # dropping the work row from cleaned_df for X for train test split
    y = cleaned_df["Work"] # making work be the y for train test split
    X_encoded = X.copy() # copy of x for encoding
    encoders = {}
    for column in X_encoded.columns: 
        if X_encoded[column].dtype == object: # if we are dealing with an object, which we will be, then:
            le = LabelEncoder()
            X_encoded[column] = le.fit_transform(X_encoded[column]) # we fit transform
            encoders[column] = le 
    y_encoder = LabelEncoder()
    y_encoded = y_encoder.fit_transform(y) # and then fit transform  y
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=0) # train test split for classification
    clf_knn = KNeighborsClassifier(n_neighbors=3) # kNN classifier
    clf_knn.fit(X_train, y_train)
    clf_dt = DecisionTreeClassifier(random_state=0) # decision tree classifier
    clf_dt.fit(X_train, y_train)
    clean_semester_df = semester_df.transpose() # then transpose the semester df
    clean_semester_df.columns = classifying_df["Property"]
    semester_encoded = clean_semester_df.copy() # make a copy for the encoding
    semester_encoded = semester_encoded[X_encoded.columns]
    for column in semester_encoded.columns: # for each column in the semester encoded df
        if semester_encoded[column].dtype == object:
            le = encoders.get(column)
            semester_encoded[column] = le.transform(semester_encoded[column]) # we transform labels with label encoder
    semester_preds_knn = clf_knn.predict(semester_encoded) # kNN prediction!
    semester_preds_dt = clf_dt.predict(semester_encoded) # decision tree prediction!
    label_mapping = {i: label for i, label in enumerate(y_encoder.classes_)} # map it back to our labels from label encoding 
    semester_preds_labels_knn = [label_mapping[pred] for pred in semester_preds_knn] # for the predictions in the structure, we map the label
    semester_preds_labels_dt = [label_mapping[pred] for pred in semester_preds_dt]
    print("Workload Predictions with kNN (in order listed above):", semester_preds_labels_knn) # print to show the user (me lol)
    print("Workload Predictions with Decision Tree (in order listed above):", semester_preds_labels_dt)

'''
This function takes in my fall 2026 schedule and helps me predict what the
course load will look like; average, few, or many assingments per class.
'''
def fall_26(classifying_df, fall_26_df):
    print(fall_26_df) # printing it to see what classes I will be taking
    classifying_df = classifying_df.dropna(axis=1) # dropping rows with mising data.
    cleaned_df = classifying_df.transpose() # transpose the classifying_df
    cleaned_df.columns = classifying_df["Property"] # columns are the property
    cleaned_df = cleaned_df.drop("Property")
    X = cleaned_df.drop("Work", axis=1) # dropping the work row from cleaned_df for X for train test split
    y = cleaned_df["Work"] # making work be the y for train test split
    X_encoded = X.copy() # copy of x for encoding
    encoders = {}
    for column in X_encoded.columns: 
        if X_encoded[column].dtype == object: # if we are dealing with an object, which we will be, then:
            le = LabelEncoder()
            X_encoded[column] = le.fit_transform(X_encoded[column]) # we fit transform
            encoders[column] = le 
    y_encoder = LabelEncoder()
    y_encoded = y_encoder.fit_transform(y) # and then fit transform  y
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=0) # train test split for classification
    clf_knn = KNeighborsClassifier(n_neighbors=3) # kNN classifier
    clf_knn.fit(X_train, y_train)
    clf_dt = DecisionTreeClassifier(random_state=0) # decision tree classifier
    clf_dt.fit(X_train, y_train)
    clean_fall_26_df = fall_26_df.transpose() # transpose again
    clean_fall_26_df.columns = classifying_df["Property"] 
    fall_26_encoded = clean_fall_26_df.copy() # make a copy for encoding fall 2025
    fall_26_encoded = fall_26_encoded[X_encoded.columns]
    for column in fall_26_encoded.columns: # for each column in this encoding semester df,
        if fall_26_encoded[column].dtype == object:
            le = encoders.get(column)
            fall_26_encoded[column] = le.transform(fall_26_encoded[column]) # use label encoder to transform labels
    fall_26_preds_knn = clf_knn.predict(fall_26_encoded) # kNN prediction!
    fall_26_preds_dt = clf_dt.predict(fall_26_encoded) # decision tree prediction!
    label_mapping = {i: label for i, label in enumerate(y_encoder.classes_)} # using the label encoder to get our labels back
    fall_26_preds_labels_knn = [label_mapping[pred] for pred in fall_26_preds_knn] # for each prediction in the semester predictions, map the predicted label
    fall_26_preds_labels_dt = [label_mapping[pred] for pred in fall_26_preds_dt]
    print("Fall 2026 Workload Predictions with kNN:", fall_26_preds_labels_knn) # print for user
    print("Fall 2026 Workload Predictions with Decision Tree:", fall_26_preds_labels_dt)

'''
This function helps me predict my course load for spring of 2027, my last semester at Gonzaga! 
'''
def spring_27(classifying_df, spring_27_df):
    print(spring_27_df)
    classifying_df = classifying_df.dropna(axis=1) # drop the rows with missing info
    cleaned_df = classifying_df.transpose() # transpose the data
    cleaned_df.columns = classifying_df["Property"] # the columns are the classifying df property of amount of work
    cleaned_df = cleaned_df.drop("Property")
    feature_columns = cleaned_df.columns.drop("Work") # feature columns is the cleaned df dropped work column
    X = cleaned_df[feature_columns]  # X is the cleaned df without work
    y = cleaned_df["Work"] # y is the work column
    X_encoded = X.copy() # X encoded is the copy
    encoders = {}
    for column in X_encoded.columns: # for each column in the copy of x...
        if X_encoded[column].dtype == object:
            le = LabelEncoder()
            X_encoded[column] = le.fit_transform(X_encoded[column]) # then fit transform the X encoded
            encoders[column] = le
    y_encoder = LabelEncoder()
    y_encoded = y_encoder.fit_transform(y) # y encoded is the y encoder fit transformed
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=0) # train test split the data
    clf_knn = KNeighborsClassifier(n_neighbors=3) # kNN classifier
    clf_knn.fit(X_train, y_train)
    clf_dt = DecisionTreeClassifier(random_state=0) # decision tree classifier
    clf_dt.fit(X_train, y_train)
    clean_spring_27_df = spring_27_df.transpose() # transpose again for cleaning
    clean_spring_27_df.columns = classifying_df["Property"] # columns is classifying df's property which we worked with
    spring_27_features = clean_spring_27_df[feature_columns] 
    spring_27_features = spring_27_features.dropna() # drop anything missing
    for column in feature_columns: # for each column in feature columns
        if column in spring_27_features.columns:
            le = encoders.get(column)
            spring_27_features[column] = le.transform(spring_27_features[column]) # use label encoder for the transofrmation for predictions
    spring_27_preds_knn = clf_knn.predict(spring_27_features) # knn prediction!
    spring_27_preds_dt = clf_dt.predict(spring_27_features) # decision tree prediction!
    label_mapping = {i: label for i, label in enumerate(y_encoder.classes_)} # using label encoder for label mapping 
    spring_27_preds_labels_knn = [label_mapping[pred] for pred in spring_27_preds_knn] #for each prediction in the predictions we label map
    spring_27_preds_labels_dt = [label_mapping[pred] for pred in spring_27_preds_dt]
    print("Spring 2027 Workload Predictions with kNN:", spring_27_preds_labels_knn) # display
    print("Spring 2027 Workload Predictions with Decision Tree:", spring_27_preds_labels_dt)
