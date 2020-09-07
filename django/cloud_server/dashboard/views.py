from django.http import JsonResponse
from django.shortcuts import render
from django.core import serializers
from pyspark.ml.feature import Word2Vec, Word2VecModel
from pyspark.sql.functions import *
from pyspark.sql.types import *

# Local import
from dashboard.utils import *
spark, model_word2vec, title_vectors_df = startup()

def dashboard_home(request):
    return render(request, 'index.html', {})

def question(request):
    return render(request, 'question.html', {})

def result(request):

    # print(request.POST.get)
    input_question = request.POST.get('question', "What is list comprehension?")
    # print(input_question)

    question_dataframe = spark.createDataFrame([(input_question, )], ["question"])

    # Making a joined data column with all the tokens
    question_tokenized_df = question_dataframe.withColumn('joined_data',udf(preprocess_text, ArrayType(StringType()))(question_dataframe.question))

    question_with_vector_df = model_word2vec.transform(question_tokenized_df)

    #taking only the dense vector
    question_dense_vec = question_with_vector_df.first()["features"]

    #Now that we have everything in place, we just need to calculate the similarity score
    df_cos_sim = title_vectors_df.withColumn("similarity_score", udf(cos_sim, FloatType())(col("sentiment"),col("score"),col("features"), array([lit(v) for v in question_dense_vec])))

    rdd_1 = df_cos_sim.orderBy('similarity_score', ascending= False).take(3)
    # final_list = [['https://stackoverflow.com/questions/14606559', 'python nested classes', 'python|python-3.x', None], ['https://stackoverflow.com/questions/14606559', 'python nested classes', 'python|python-3.x', None], ['https://stackoverflow.com/questions/14606559', 'python nested classes', 'python|python-3.x', None]]
    final_list = []
    for i in range(3):
        temp_list = []
        temp_list.append("https://stackoverflow.com/questions/"+str(rdd_1[i][0]))
        temp_list.append(rdd_1[i][1])
        temp_list.append(rdd_1[i][3])
        temp_list.append(rdd_1[i][-1])
        final_list.append(temp_list)

    print(final_list)
    return render(request, 'result.html', {"similar_list": final_list, "question": input_question})