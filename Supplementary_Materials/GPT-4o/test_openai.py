from openai import OpenAI
import pandas as pd
import regex
#import os
#from apikey import APIKEY

#reading file
#df = pd.read_csv('./Datasets/PnC.csv')
#df = pd.read_csv('./Datasets/StraightLines.csv')
#df = pd.read_csv('./Datasets/SeqnSeries.csv')
#df = pd.read_csv('./Datasets/LimitsnDerivatives.csv')
df = pd.read_csv('./Datasets/conic_sections.csv')

english = list(df.iloc[:, 0])[:50]
hindi = list(df.iloc[:, 1])[:50]
urdu = list(df.iloc[:, 2])[:50]
bengali = list(df.iloc[:, 3])[:50]
gujrati = list(df.iloc[:, 4])[:50]
malayalam = list(df.iloc[:, 5])[:50]
categories = list(df.iloc[:, 7])[:50]
sol = list(df.iloc[:, 6])[:50]

client = OpenAI(api_key = "YOUR_API_KEY")

#process_list = [req_list1, req_list2, req_list3, req_list4, req_list5, req_list6]
all_problems = [english, hindi, urdu, bengali, gujrati, malayalam]
lang = ['english', 'hindi', 'urdu', 'bengali', 'gujrati', 'malayalam']

#all_problems = [hindi]
#lang = ['hindi']

final_answer_list = []

#0: One Shot , 1: CoT, 2: One Shot Subcategory
flag = 1
#PnC, Straight Lines, SeqnSeries, Limits, Conic
flag_Topic = 4

choices = []
for j in all_problems:
  flag_fn = 'pnc'
  if flag_Topic == 0:
    file_fn = 'pnc'
  elif flag_Topic == 1:
    file_fn = 'straight_Lines'
  elif flag_Topic == 2:
    file_fn = 'seqnseries'
  elif flag_Topic == 3:
    file_fn = 'limitsnderiv'
  else:
    file_fn = 'conic_sections'
  print(lang[all_problems.index(j)])
  results = []
  results_choice = []
  if flag == 0:
    file1 = open(file_fn+"_one_shot_with_language_expertise"+ lang[all_problems.index(j)] +".txt", "a")
  elif flag == 1:
    file1 = open(file_fn+"_CoT"+ lang[all_problems.index(j)] +".txt", "a")
  else:
    file1 = open(file_fn+"_one_shot_subcat"+ lang[all_problems.index(j)] +".txt", "a")
  for k in range(3):
    result = []
    result_choice = []
    for i in j:
      if flag == 0:
        if flag_Topic == 0:
          prompt = "You are an expert in solving problems in Combinatorics. Use your expertise to solve: " + i + " Give the final answer along with the explanation in the format of json like {Final Answer: \"Numeric answer\"}"
        elif flag_Topic == 1:
          prompt = "You are an expert in solving problems in Straight Lines. Use your expertise to solve: " + i + " Give the final answer along with the explanation in the format of json like {Final Answer: \"Numeric answer\"}"
        elif flag_Topic == 2:
          prompt = "You are an expert in solving problems in Sequences and Series. Use your expertise to solve: " + i + " Give the final answer along with the explanation in the format of json like {Final Answer: \"Numeric answer\"}"
        elif flag_Topic == 3:
          prompt = "You are an expert in solving problems in Limits and Derivatives. Use your expertise to solve: " + i + " Give the final answer along with the explanation in the format of json like {Final Answer: \"Numeric answer\"}"
        else:
          prompt = "You are an expert in solving problems in Conic Sections. Use your expertise to solve: " + i + " Give the final answer along with the explanation in the format of json like {Final Answer: \"Numeric answer\"}"
      
      elif flag == 1:
        #a) Distance of a point from a line b) Forms of Equation of a line c) Slope of a line
	#a) Arithmetic Mean and Geometric Mean b) Arithmetic Progression c) Geometric Progression d) Sequences e) Series
	#prompt = "Choose the category this problem belongs to, any give only the final option: " + i + "a) Combinations b) Permutations c) Fundamental principle of Counting"
        if flag_Topic == 0:
          prompt = "Choose the category this problem belongs to, any give only the final option: " + i + "a) Combinations b) Permutations c) Fundamental p    rinciple of Counting"
        elif flag_Topic == 1:
          prompt = "Choose the category this problem belongs to, any give only the final option: " + i + "a) Distance of a point from a line b) Forms of E    quation of a line c) Slope of a line"
        elif flag_Topic == 2:
          prompt = "Choose the category this problem belongs to, any give only the final option: " + i + "a) Arithmetic Mean and Geometric Mean b) Arithme    tic Progression c) Geometric Progression d) Sequences e) Series"
        elif flag_Topic == 3:
          prompt = "Choose the category this problem belongs to, any give only the final option: " + i + "a) Limits b) Limits of Trigonometric Functions c) Derivatives"
        else:
          prompt = "Choose the category this problem belongs to, any give only the final option: " + i + "a) Parabola b) Hyperbola c) Ellipse d) Circle"
      else:
        prompt = "You are an expert in solving problems in " + categories[j.index(i)] + ". Use your expertise to solve: " + i + " Give the final answer along with the explanation in the format of json like {Final Answer: \"Numeric answer\"}" 
      response = client.chat.completions.create(
      #model="gpt-3.5-turbo",
      model="gpt-4o-mini",
      messages=[
        {"role": "user", "content": prompt}
      ]
      )
      
      if flag == 1:
        prev_response = response.choices[0].message.content.strip()
        result_choice.append(prev_response)
        #choices.append(prev_response)
        #curr_prob = problems_All_Languages[px][py]
        if len(response.choices[0].message.content.strip().split(') ')) > 1:
          new_prompt = "You are an expert in solving problems in " + str(response.choices[0].message.content.strip().split(') ')[1]) + ". Use your expertise to solve: " + i + " Give the final answer along with the explanation in the format of json like {Final Answer: \"Numeric answer\"}"
        else:
          new_prompt = "You are an expert in solving problems in " + response.choices[0].message.content.strip() + ". Use your expertise to solve: " + i + " Give the final answer along with the explanation in the format of json like {Final Answer: \"Numeric answer\"}" 
        
        response = client.chat.completions.create(
        #model="gpt-3.5-turbo",
        model="gpt-4o-mini",
        messages=[
          {"role": "user", "content": prompt},
          {"role": "assistant", "content": prev_response},
          {"role": "user", "content": new_prompt}
        ])


      #print(response.choices[0].message.content.strip())
      #print(response)
      print('*******************' + str(j.index(i)))

      #pattern = regex.compile(r'\{(?:[^{}]|(?R))*\}')
      #if pattern.findall(response.choices[0].message.content.strip()) != '':
      #  final_answer_list.append(pattern.findall(response.choices[0].message.content.strip()))
      #else:
      #  final_answer_list.append('Null')
      response_text = response.choices[0].message.content.strip()
      result.append(response_text[response_text.find('Final Answer'):])
      file1.write(response_text)
    results.append(result)
    #results_choice.append(result_choice) 
    if flag == 1:
      choices.append('Done')
      results_choice.append(result_choice)
    print('Done '+ str(k))
    file1.write('Done '+ str(k))
    final_answer_list.append('Done')
  #results_choice.append(categories)
  results.append(sol)
  dict = {'trial1': results[0], 'trial2': results[1], 'trial3': results[2], 'Sol': results[3]}
  df = pd.DataFrame(dict)
  if flag ==0:
    df.to_csv(file_fn+"_one_shot"+ lang[all_problems.index(j)] +"_trials.csv")
  elif flag ==1:
    results_choice.append(categories)
    dict_choice = {'trial1': results_choice[0], 'trial2': results_choice[1], 'trial3': results_choice[2], 'Sol': results_choice[3]}
    df_choice = pd.DataFrame(dict_choice)
    df.to_csv(file_fn+"_CoT"+ lang[all_problems.index(j)] +"_trials.csv")
    df_choice.to_csv(file_fn+"_CoT_Choices_By_GPT"+ lang[all_problems.index(j)] +"_trials.csv")
  else:
    df.to_csv(file_fn+"_one_shot_subcat"+ lang[all_problems.index(j)] +"_trails.csv")


  #with open('your_file.txt', 'w') as f:
  #  for line in final_answer_list:
  #    f.write(f"{line}\n")
  file1.close()

#  if flag == 1:
#    with open('choices.txt', 'w') as f:
#      for line in choices:
#        f.write(f"{line}\n")
