"""
Replication script for EISS '23 presentation.

@author: maxwell.love
"""

from textblob import TextBlob
import requests
import time
import sys
import os
import csv

'''
These variables must be set
'''

'''
Choice is model choice.
1- textblob
2- XLM-t-roberta (BERT)
'''
MODEL= 1

#Absolute path for MFA text file corpus
mfa_dir = ""

#Absolute path for National Security Strategy text file corpus
nss_dir = ""

#Absolute path for resulting output
output_dir = ""

#Absolute path for token.txt file
API_token_dir = ""


'''
Do not change these variables
'''
max_input_size = 513

total_count = 0

truncate_count = 0

def transpose_polarity(target):
    in_file = f'{output_dir}\\{target}_TEXTBLOB.csv'
    out_file = f'{output_dir}\\{target}_TEXTBLOB_transposed.csv'

    with open(in_file,'r',errors='ignore') as f:
        with open(out_file,'w',newline='',errors='ignore') as g:
            file = csv.reader(f)
            
            cyo_count = {}
            first = True
            for line in file:
                if first:
                    first = False
                    continue
                if f"{line[0]}-{line[1]}-{line[2]}" in cyo_count:
                    cyo_count[f"{line[0]}-{line[1]}-{line[2]}"] += 1
                else:
                    cyo_count[f"{line[0]}-{line[1]}-{line[2]}"] = 1
                    
            #pprint.pprint(cyo_count)
            #consolidate to common offices
            
            #find max count per common office in a given country year
            #iterate through keys and create office dictionary to count and check max
            office_cy_max = {}
            for k,v in cyo_count.items():
                if k.split('-')[2] in office_cy_max:
                    current = office_cy_max[k.split('-')[2]]
                    if v > current:
                        office_cy_max[k.split('-')[2]] = v
                else:
                    office_cy_max[k.split('-')[2]] = v
                    

            
            #create columns list by office
            fieldnames=["Country","Year"]
            
            for k,v in office_cy_max.items():
                for i in range(v):
                    fieldnames.append(f"{k}_{i}")
                    #fieldnames.append(f"{k}_{i+1}")
            csv_writer = csv.DictWriter(g, fieldnames)
            csv_writer.writeheader()
            
            #iterate over rows, put compound score for that country year in the correct column
            #build full dictionary of country year to dictionary of office
            cy_collection = {}
            f.seek(0)
            file = csv.reader(f)
            flag = True
            for line in file:
                #skip first line!
                if flag:
                    
                    flag = False
                    continue
                
                #build dictionary- country-year => dictionary of offices => list of compound scores
                if f"{line[0]}-{line[1]}" in cy_collection:
                    if f"{line[2]}" in cy_collection[f"{line[0]}-{line[1]}"]:
                        cy_collection[f"{line[0]}-{line[1]}"][f"{line[2]}"].append(f"{line[4]}")
                    else:
                        cy_collection[f"{line[0]}-{line[1]}"][f"{line[2]}"] = [f"{line[4]}"]
                else:
                    cy_collection[f"{line[0]}-{line[1]}"] = {}
                    cy_collection[f"{line[0]}-{line[1]}"][f"{line[2]}"] = [f"{line[4]}"]
                    
                    
            #iterate over full dictionary, build row for writing
            #for each office, 
            for k,v in cy_collection.items():
                #build dictionary to be written:
                result = {}
                result['Country'] = k.split('-')[0]
                result['Year'] = k.split('-')[1]
                #iterate through each office
                for k2,v2 in v.items():
                    max_count = office_cy_max[k2]
                    for i in range(len(v2)):
                        result[f"{k2}_{i}"] = v2[i]
                        #result[f"{k2}_{i+1}"] = v2[i]
                    for j in range(len(v2),max_count):
                        result[f"{k2}_{j}"] = None
                        #result[f"{k2}_{j+1}"] = 0
                        
                #last check- if there was no office for that cy, fill in 0
                for k3,v3 in office_cy_max.items():
                    if k3 not in v:
                        for i in range(v3):
                            result[f"{k3}_{i}"] = None
                            #result[f"{k3}_{i+1}"] = 0
                csv_writer.writerow(result)
                    

def transpose_component(target):
    in_file = f'{output_dir}\\{target}_BERT.csv'
    out_file = f'{output_dir}\\{target}_BERT_transposed.csv'


    with open(in_file,'r',errors='ignore') as f:
        with open(out_file,'w',newline='',errors='ignore') as g:
            file = csv.reader(f)
            
            cyo_count = {}
            first = True
            for line in file:
                if first:
                    first = False
                    continue
                if f"{line[0]}-{line[1]}-{line[2]}" in cyo_count:
                    cyo_count[f"{line[0]}-{line[1]}-{line[2]}"] += 3
                else:
                    cyo_count[f"{line[0]}-{line[1]}-{line[2]}"] = 3
                    
            #pprint.pprint(cyo_count)
            #consolidate to common offices
            
            #find max count per common office in a given country year
            #iterate through keys and create office dictionary to count and check max
            office_cy_max = {}
            for k,v in cyo_count.items():
                if k.split('-')[2] in office_cy_max:
                    current = office_cy_max[k.split('-')[2]]
                    if v > current:
                        office_cy_max[k.split('-')[2]] = v
                else:
                    office_cy_max[k.split('-')[2]] = v
                    
            
            #create columns list by office
            fieldnames=["Country","Year"]
            labels = ["POS","NEU","NEG"]
            
            for k,v in office_cy_max.items():
                for i in range(v):
                    fieldnames.append(f"{k}_{i//3}_{labels[i%3]}")
                    #fieldnames.append(f"{k}_{i+1}")
            csv_writer = csv.DictWriter(g, fieldnames)
            csv_writer.writeheader()
            
            #iterate over rows, put compound score for that country year in the correct column
            #build full dictionary of country year to dictionary of office
            cy_collection = {}
            f.seek(0)
            file = csv.reader(f)
            flag = True
            for line in file:
                #skip first line!
                if flag:
                    
                    flag = False
                    continue
                
                #build dictionary- country-year => dictionary of offices => list of compound scores
                if f"{line[0]}-{line[1]}" in cy_collection:
                    if f"{line[2]}" in cy_collection[f"{line[0]}-{line[1]}"]:
                        cy_collection[f"{line[0]}-{line[1]}"][f"{line[2]}"].extend([f"{line[4]}",f"{line[5]}",f"{line[6]}"])
                    else:
                        cy_collection[f"{line[0]}-{line[1]}"][f"{line[2]}"] = [f"{line[4]}",f"{line[5]}",f"{line[6]}"]
                else:
                    cy_collection[f"{line[0]}-{line[1]}"] = {}
                    cy_collection[f"{line[0]}-{line[1]}"][f"{line[2]}"] = [f"{line[4]}",f"{line[5]}",f"{line[6]}"]
                    
                    
            #iterate over full dictionary, build row for writing
            #for each office, 
            for k,v in cy_collection.items():
                #build dictionary to be written:
                result = {}
                result['Country'] = k.split('-')[0]
                result['Year'] = k.split('-')[1]
                #iterate through each office
                for k2,v2 in v.items():
                    max_count = office_cy_max[k2]
                    for i in range(len(v2)):
                        
                        result[f"{k2}_{i//3}_{labels[i%3]}"] = v2[i]
                        #result[f"{k2}_{i+1}"] = v2[i]
                    for j in range(len(v2),max_count):
                        
                        result[f"{k2}_{j//3}_{labels[j%3]}"] = None
                        #result[f"{k2}_{j+1}"] = 0
                        
                #last check- if there was no office for that cy, fill in 0
                for k3,v3 in office_cy_max.items():
                    if k3 not in v:
                        for i in range(v3):
                            for x in range(len(labels)):
                                result[f"{k3}_{i//3}_{labels[i%3]}"] = None
                            #result[f"{k3}_{i+1}"] = 0
                csv_writer.writerow(result)
                    


def getKeywordWindow(text,keyword):
    max_window_size = 512
    
    total_length = len(text)
    index = text.lower().index(keyword.lower())
    
    #test for truncation
    if total_length > max_window_size:
        
        #add 16 to either side
        room_left = True
        room_right = True
        increment = 4
        start_index = index
        end_index = index + len(keyword)
        
        
        while (room_left or room_right) and (end_index - start_index < max_window_size):
            #add left
            if room_left and start_index - increment >= 0:
                if end_index - (start_index - increment) > max_window_size:
                    start_index -= max_window_size - (end_index - start_index)
                else:
                    start_index -= increment
            else:
                room_left = False
                
            #add right
            if room_right and end_index + increment < total_length:
                if (end_index + increment) - start_index > max_window_size:
                    end_index += max_window_size - (end_index - start_index)
                else:
                    end_index += increment
            else:
                room_right = False
     
        
    #no truncation
    else:
        start_index = 0
        end_index = total_length
        

    return text[start_index:end_index]     
        
    




def getSentiment(text, choice):
    if choice == 1:
       sentence = TextBlob(text)
       return [sentence.sentiment.polarity]
    
    else:
        API_URL = "https://api-inference.huggingface.co/models/cardiffnlp/xlm-twitter-politics-sentiment"
        headers = {"Authorization": f"Bearer {API_TOKEN}"}
        
        if len(text) > max_input_size:
            print(f"length too long: {len(text)}")
            print(f"truncating to {max_input_size}")    
            response = requests.post(API_URL, headers=headers, json={"inputs": text[:max_input_size],})
        else:
            response = requests.post(API_URL, headers=headers, json={"inputs": text,})
        
        
        try:
            parts = response.json()[0]
        except KeyError:
            if "error" in response.text:
                print(f"Waiting: {response.text}")
                time.sleep(75)
                response = requests.post(API_URL, headers=headers, json={"inputs": text[:max_input_size],})
                parts = response.json()[0]
                return [parts[0]['score'],parts[1]['score'],parts[2]['score']]
            print(response.text)
            sys.exit("error")
        except:
            print(response.text)
            sys.exit("other error")
        return [parts[0]['score'],parts[1]['score'],parts[2]['score']]
    

#given a term, return boolean if any of the list items appear 
def checkTerms(terms, tag):
    overall = False
    contains = []
    for term in terms:
        current_check = term.lower() in tag.lower()
        if current_check:
            contains.append(term.lower())
        overall = current_check or overall
    return (overall,contains)
    
    

        
#take in a target, list of quoted, list of unquoted search terms
def run(target, quoted, unquoted, src_dir):
    if MODEL == 1:
        model_str = f'{output_dir}\\{target}_TEXTBLOB.csv'
    else:
        model_str = f'{output_dir}\\{target}_BERT.csv'

    with open(model_str,'w',newline='',errors='ignore') as overall:
        if MODEL == 1:
            csv_writer = csv.DictWriter(overall, fieldnames=["Country","Year","Office","Text","Sentiment"])
        else:
            csv_writer = csv.DictWriter(overall, fieldnames=["Country","Year","Office","Text","Positive","Neutral","Negative"])
        
        csv_writer.writeheader()
               
        files = os.listdir(src_dir)

        for file in files:

            print(f"Working {file}")
            current_attr = file.split('.')[0].split('_')

            #get attributes for MFA
            if src_dir == mfa_dir:
                
                #grab search terms
                if current_attr[3] == 'Q':
                    quoted.append(" ".join(current_attr[4].split('-')).strip())
                else:
                    unquoted.extend(current_attr[4].split('-'))
    
            with open(f'{src_dir}\\{file}','r',errors='ignore') as f:
    
                lines = f.readlines()
            
                for line in lines:
                
                    unq = checkTerms(unquoted, line)
                    quo = checkTerms(quoted, line)
    
                    if unq[0]:
                            
                            hits = unq[1]
                            for hit in hits:
                                if MODEL == 1:
                                    windowed_text = line
                                    sent = getSentiment(windowed_text, MODEL)
                                    csv_writer.writerow({
                                    'Country':current_attr[0],
                                    'Year':current_attr[1],
                                    'Office': "NATSEC",
                                    'Text':windowed_text,
                                    'Sentiment':sent[0],
                                    })

                                else:
                                    windowed_text = getKeywordWindow(line, hit)                                    
                                    sent = getSentiment(windowed_text, MODEL)
                                    csv_writer.writerow({
                                    'Country':current_attr[0],
                                    'Year':current_attr[1],
                                    'Office': "NATSEC",
                                    'Text':windowed_text,
                                    'Positive': sent[0],
                                    'Neutral': sent[1],
                                    'Negative': sent[2],
                                    })
                                
                          
                    if quo[0]:
                            hits = quo[1]
                            for hit in hits:
                                if MODEL == 1:
                                    windowed_text = line
                                    sent = getSentiment(windowed_text, MODEL)
                                    csv_writer.writerow({
                                    'Country':current_attr[0],
                                    'Year':current_attr[1],
                                    'Office': "NATSEC",
                                    'Text':windowed_text,
                                    'Sentiment':sent[0],
                                    })

                                else:
                                    windowed_text = getKeywordWindow(line, hit)                                    
                                    sent = getSentiment(windowed_text, MODEL)
                                    csv_writer.writerow({
                                    'Country':current_attr[0],
                                    'Year':current_attr[1],
                                    'Office': "NATSEC",
                                    'Text':windowed_text,
                                    'Positive': sent[0],
                                    'Neutral': sent[1],
                                    'Negative': sent[2],
                                    })
                        
           

API_TOKEN= ""
with open(f'{API_token_dir}\\token.txt','r',errors='ignore') as q:
    API_TOKEN = q.read()
    print(f"Using token: {API_TOKEN}")

target_set = ["CHINA","RUSSIA","EU","US","NATO","MFA"]
unquotes_set = [["china","chinese"],["russia","russian"],[" EU ","E.U."],["U.S.A","U.S.", "USA"],["NATO","N.A.T.O.", "OTAN","O.T.A.N."],[]]
quotes_set = [["belt and road", "silk road","one belt one road", " bri "],[],["European Union"],["United States"],["North Atlantic Treaty Organization"],[]]



#generate dense CSV
for i in range(len(target_set)):
    print(f'Starting {target_set[i]}')
    if i == len(target_set) - 1:
        run(target_set[i],unquotes_set[i],quotes_set[i],mfa_dir)
    else:
        run(target_set[i],unquotes_set[i],quotes_set[i], nss_dir)
    #transpose resulting dense CSV to country-year format
    if MODEL == 1:
        transpose_polarity(target_set[i])
    else:
        transpose_component(target_set[i])


