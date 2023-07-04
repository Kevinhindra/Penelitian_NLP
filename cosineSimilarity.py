import nltk
nltk.download('punkt')

import streamlit as st
import os
from nltk.tokenize import word_tokenize
import string
import re
import pandas as pd
import csv
import math
import sys

def process_data(input_text):
    data_berita = [input_text.lower()]
    token_data = [word_tokenize(text) for text in data_berita]

    clear_tokens_punctuation = []
    for tokens in token_data:
        for token in tokens:
            if not re.match(r'^\d+\.\d+$', token):
                token = token.translate(str.maketrans('', '', string.punctuation.replace('-', '')))
            if token:
                clear_tokens_punctuation.append(token)

    clear_tokens = []
    tanggal_dan_tahun_regex = re.compile(r'^t\w+')
    pukul_dan_periode_regex = re.compile(r'^p\w+')
    jam_regex = re.compile(r'^j\w+')
    menit_regex = re.compile(r'^m\w+')
    detik_regex = re.compile(r'^d\w+')

    i = 0
    while i < len(clear_tokens_punctuation):
        token = clear_tokens_punctuation[i]
        if tanggal_dan_tahun_regex.match(token):
            clear_tokens.append(token)
            if i + 1 < len(clear_tokens_punctuation) and clear_tokens_punctuation[i + 1].isdigit():
                clear_tokens.append(clear_tokens_punctuation[i + 1])
        elif pukul_dan_periode_regex.match(token):
            clear_tokens.append(token)
            if i + 1 < len(clear_tokens_punctuation) and clear_tokens_punctuation[i + 1].isdigit():
                clear_tokens.append(clear_tokens_punctuation[i + 1])
        elif jam_regex.match(token):
            clear_tokens.append(token)
            if i + 1 < len(clear_tokens_punctuation) and clear_tokens_punctuation[i + 1].isdigit():
                clear_tokens.append(clear_tokens_punctuation[i + 1])
        elif menit_regex.match(token):
            if i - 1 >= 0 and clear_tokens_punctuation[i - 1].isdigit():
                clear_tokens.append(clear_tokens_punctuation[i - 1])
            clear_tokens.append(token)
        elif detik_regex.match(token):
            if i - 1 >= 0 and clear_tokens_punctuation[i - 1].isdigit():
                clear_tokens.append(clear_tokens_punctuation[i - 1])
            clear_tokens.append(token)
        elif token.isdigit():
            if i + 1 < len(clear_tokens_punctuation) and tanggal_dan_tahun_regex.match(clear_tokens_punctuation[i + 1]):
                clear_tokens.append(token)
        elif not token.isdigit():
            clear_tokens.append(token)
        i += 1
    return clear_tokens

def keterangan_waktu_yang_benar(file_path):
    column_title = 'keterangan_waktu'

    df = pd.read_csv(file_path)
    gram_yang_benar = df[column_title].str.lower().values
    
    return gram_yang_benar

def keterangan_waktu(dataset_path):
    one_gram_data = []
    two_gram_data = []
    
    column1_title = 'keterangan_waktu' 
    column2_title = 'keterangan_waktu_yang_salah' 

    for dataset_path in dataset_path:
        with open(dataset_path, 'r') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                if column1_title in row:
                    tokens = row[column1_title].lower().split()
                    if len(tokens) == 1:
                        one_gram_data.append(tokens[0])
                    elif len(tokens) == 2:
                        two_gram_data.append(' '.join(tokens))
                if column2_title in row:
                    tokens = row[column2_title].lower().split()
                    if len(tokens) == 1:
                        one_gram_data.append(tokens[0])
                    elif len(tokens) == 2:
                        two_gram_data.append(' '.join(tokens))
    
    return one_gram_data,two_gram_data
    
def matching_token(clear_tokens,one_gram_data,two_gram_data):
    matching_token_one_grams = set(clear_tokens).intersection(set(one_gram_data))

    matching_token_two_grams = set(' '.join([clear_tokens[i], clear_tokens[i+1]]) for i in range(len(clear_tokens)-1)).intersection(set(two_gram_data))
            
    return matching_token_one_grams,matching_token_two_grams

def get_gram_frequency(tokens):
    gram_freq = {}
    for token in tokens:
        if token in gram_freq:
            gram_freq[token] += 1
        else:
            gram_freq[token] = 1
    return gram_freq

def cosine_similarity(token1, token2):
    
    gram_freq1 = get_gram_frequency(token1)
    gram_freq2 = get_gram_frequency(token2)
    
    unique_grams = list(set(list(gram_freq1.keys()) + list(gram_freq2.keys())))
    vec1 = [gram_freq1.get(gram, 0) for gram in unique_grams]
    vec2 = [gram_freq2.get(gram, 0) for gram in unique_grams]
    
    dot_product = sum([vec1[i]*vec2[i] for i in range(len(vec1))])
    vec1_length = math.sqrt(sum([vec1[i]**2 for i in range(len(vec1))]))
    vec2_length = math.sqrt(sum([vec2[i]**2 for i in range(len(vec2))]))

    if vec1_length == 0 or vec2_length == 0:
        cosine_sim = 0
    else:
        cosine_sim = dot_product / (vec1_length * vec2_length)
    return cosine_sim     

def main():
    st.title("Deteksi Keterangan Waktu dengan Algoritma Cosine Similarity")
    input_text = st.text_area("Masukkan teks:")
    
    if st.button("Proses"):
        if input_text:
            clear_tokens = process_data(input_text)

        else:
            st.warning('Tolong masukkan teks!', icon="⚠️")
            st.stop()
            
        file_path = 'D:\Campus\Skripsi\Data Skripsi\keterangan_waktu_yang_benar.csv'
        
        gram_yang_benar = keterangan_waktu_yang_benar(file_path)
        
        dataset_path = ['D:\Campus\Skripsi\Data Skripsi\keterangan_waktu_yang_benar.csv', 
                        'D:\Campus\Skripsi\Data Skripsi\keterangan_waktu_yang_salah.csv']
        
        one_gram_data,two_gram_data = keterangan_waktu(dataset_path)

        matching_token_one_grams,matching_token_two_grams = matching_token(clear_tokens,one_gram_data,two_gram_data)
        
        combined_text = ' '.join(clear_tokens)
        
        for token in clear_tokens:
            if token in matching_token_one_grams:
                # pencarian satu kata dan tidak ada kata belakangnya atau gabungan kata
                if re.search(rf'\b{re.escape(token)}\b', combined_text) and not re.search(rf'\b{re.escape(token)}\w+\b', combined_text):
                    combined_text = combined_text.replace(token, f'<mark style="background-color: yellow;">{token}</mark>')

            # pencarian dua kata pada teks
            for i in range(len(clear_tokens) - 1):
                token = f'{clear_tokens[i]} {clear_tokens[i+1]}'
                if token in matching_token_two_grams:
                    if re.search(rf'\b{re.escape(token)}\b', combined_text) and not re.search(rf'\b{re.escape(token)}\w+\b', combined_text):
                        combined_text = combined_text.replace(token, f'<mark style="background-color: yellow;">{token}</mark>')

        st.header("Teks yang telah diproses:")
        st.markdown(combined_text, unsafe_allow_html=True)
            
        if matching_token_one_grams is not None and len(matching_token_one_grams) > 0 or matching_token_two_grams is not None and len(matching_token_two_grams) > 0:
            if len(matching_token_one_grams) > 0:
                
                token1 = matching_token_one_grams
                token2 = [word for word in gram_yang_benar if len(word.split()) == 1]

                threshold = 0.99
                below_threshold_count = 0
                above_threshold_count = 0
                
                st.header("Cosine Similarity Satu Gram :")
                similarity_results = []
                
                for t1 in token1:
                    max_similarity = 0
                    max_token = ""
                    for t2 in token2:
                        similarity = cosine_similarity(t1, t2)
                        if similarity > max_similarity:
                            max_similarity = similarity
                            max_token = t2

                    if max_similarity < threshold:
                        below_threshold_count += 1
                    else:
                        above_threshold_count += 1

                    similarity_results.append({
                        "Token 1": t1,
                        "Paling mirip dengan token pada Token 2": max_token,
                        "Nilai Similarity": max_similarity
                    })
                    
                similarity_result_one_grams = pd.DataFrame(similarity_results)
                
                # membuat index mulai dari 1
                similarity_result_one_grams.index = similarity_result_one_grams.index + 1
                st.dataframe(similarity_result_one_grams, width=500)

                st.write(f"Munculnya di bawah dari batas ({threshold}): {below_threshold_count}")
                st.write(f"Munculnya di atas batas atau sama dengan ({threshold}): {above_threshold_count}")
                
            if len(matching_token_two_grams) > 0:
                
                token1 = matching_token_two_grams
                token2 = [word for word in gram_yang_benar if len(word.split()) == 2]

                threshold = 0.99
                below_threshold_count = 0
                above_threshold_count = 0

                st.subheader("Cosine Similarity Dua Gram :")
                similarity_results = []
                for t1 in token1:
                    max_similarity = 0
                    max_token = ""
                    for t2 in token2:
                        similarity = cosine_similarity(t1, t2)
                        if similarity > max_similarity:
                            max_similarity = similarity
                            max_token = t2

                    if max_similarity < threshold:
                        below_threshold_count += 1
                    else:
                        above_threshold_count += 1

                    similarity_results.append({
                        "Token 1": t1,
                        "Paling mirip dengan token pada Token 2": max_token,
                        "Nilai Similarity": max_similarity
                    })

                similarity_result_two_grams = pd.DataFrame(similarity_results)
                
                # membuat index mulai dari 1
                similarity_result_two_grams.index = similarity_result_two_grams.index + 1
                st.dataframe(similarity_result_two_grams, width=500)
                
                st.write(f"Munculnya di bawah dari batas ({threshold}): {below_threshold_count}")
                st.write(f"Munculnya di atas batas atau sama dengan ({threshold}): {above_threshold_count}")
        else:
            st.subheader("Tidak ada kata keterangan waktu yang ditemukan")

if __name__ == '__main__':
    main()
