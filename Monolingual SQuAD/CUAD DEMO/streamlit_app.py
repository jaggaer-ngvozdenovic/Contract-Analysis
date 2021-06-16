from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch
import streamlit as st
from transformers.pipelines import pipeline
import json
from scripts.predict import run_prediction
from read_images_and_pdfs import read_file
import os
import shutil

st.set_page_config(layout="wide", initial_sidebar_state="expanded")

st.cache(show_spinner=False, persist=True)
def load_model():
    model = AutoModelForQuestionAnswering.from_pretrained('cuad-models/roberta-base/')
    tokenizer = AutoTokenizer.from_pretrained('cuad-models/roberta-base/', use_fast=False)
    return model, tokenizer

st.cache(show_spinner=False, persist=True)
def load_questions(file):
	with open(file) as json_file:
		data = json.load(json_file)

	questions = [""]
	for i, q in enumerate(data['data'][0]['paragraphs'][0]['qas']):
		question = data['data'][0]['paragraphs'][0]['qas'][i]['question']
		questions.append(question)
	return questions

st.cache(show_spinner=False, persist=True)
def load_contracts(file):
	with open(file) as json_file:
		data = json.load(json_file)

	contracts = []
	for i, q in enumerate(data['data']):
		contract = ' '.join(data['data'][i]['paragraphs'][0]['context'].split())
		with open("../Contract Analysis/contract_" + str(i) + ".txt", "w") as f:
			f.write(contract)
		contracts.append(contract)
	return contracts

def add_new_lines(k):
	for i in range(k):
		st.sidebar.text("\n")

st.cache(show_spinner=False, persist=True)
def delete_temp_files(folder_name):
	ext = ['pdf', 'jpeg', 'jpg', 'txt', 'doc']
	filenames = [fn for fn in os.listdir(".") if fn.split("_")[-1] in ext]
	for f in filenames:
		if f != folder_name:
			shutil.rmtree(f)
file = 'cuad-data/test.json'
model, tokenizer = load_model()
questions = load_questions(file)
contracts = load_contracts(file)

col1, col2, col3 = st.beta_columns([1,6,1])

with col1:
	st.write("")

with col2:
	st.header("Contract Understanding Atticus Dataset (CUAD) Demo")
	st.write("This demo uses a machine learning model for Contract Understanding.")
	st.image("main.png", width=650)

with col3:
	st.write("")

add_text_sidebar = st.sidebar.title("Hello, Welcome!")

add_new_lines(3)

use_cases = st.sidebar.beta_expander("CUAD use cases:", expanded=False)
use_cases.write(' - save hours of attorney time')
use_cases.write(' - enable speedy delivery of high-quality work')
use_cases.write(' - determine which contracts are for the divested business')
use_cases.write(' - accurately identify parties, name of signing entities and divested projects')
use_cases.write(' - deal with uncommon and rare clauses')

add_new_lines(2)

consists_of = st.sidebar.beta_expander("CUAD consists of:", expanded=False)
consists_of.write(' - more than 500 contracts')
consists_of.write(' - 41 different types of clauses')
consists_of.write(' - 25 different contract types')
consists_of.write(' - more than 13000 annotations ')

add_new_lines(2)

labeling = st.sidebar.beta_expander("CUAD labeling process:", expanded=False)
labeling.write(' - Law Student training')
labeling.write(' - Law Student Label')
labeling.write(' - Key Word Search')
labeling.write(' - Category-by-Category Report Review')
labeling.write(' - Attorney Review')
labeling.write(' - eBrevia Extras Review')
labeling.write(' - Final Report')

add_new_lines(2)

clauses = st.sidebar.beta_expander("CUAD clauses:", expanded=False)
clauses.write(' - Document Name')
clauses.write(' - Parties')
clauses.write(' - Agreement Date')
clauses.write(' - Effective Date')
clauses.write(' - Expiration Date')
clauses.write(' - Renewal Term')
clauses.write(' - Notice to Terminate Renewal')
clauses.write(' - Governing Law')
clauses.write(' - Most Favored Nation')
clauses.write(' - Non-Compete')
clauses.write(' - Exclusivity')
clauses.write(' - No-Solicit of Customers')
clauses.write(' - Competitive Restriction Exception')
clauses.write(' - No-Solicit of Employees')
clauses.write(' - Non-Disparagement')
clauses.write(' - Termination for Convenience')
clauses.write(' - ROFR/ROFO/ROFN')
clauses.write(' - Change of Control')
clauses.write(' - Anti-Assignment')
clauses.write(' - Revenue/Profit Sharing')
clauses.write(' - Price Restriction')
clauses.write(' - Minimum Commitment')
clauses.write(' - Volume Restriction')
clauses.write(' - IP Ownership Assignment')
clauses.write(' - Joint IP Ownership')
clauses.write(' - License Grant')
clauses.write(' - Non-Transferable License')
clauses.write(' - Affiliate IP LicenseLicensor')
clauses.write(' - Affiliate IP LicenseLicensee')
clauses.write(' - Unlimited/All-You-CanEat License')
clauses.write(' - Irrevocable or Perpetual License')
clauses.write(' - Source Code Escrow')
clauses.write(' - Post-Termination Services')
clauses.write(' - Audit Rights')
clauses.write(' - Uncapped Liability')
clauses.write(' - Cap on Liability')
clauses.write(' - Liquidated Damages')
clauses.write(' - Warranty Duration')
clauses.write(' - Insurance')
clauses.write(' - Covenant Not to Sue')
clauses.write(' - Third Party Beneficiary')

add_new_lines(2)

types = st.sidebar.beta_expander("CUAD contract types:", expanded=False)
types.write(' - Affiliate Agreement')
types.write(' - Agency Agreement')
types.write(' - Collaboration Agreement')
types.write(' - Co-Branding Agreement')
types.write(' - Consulting Agreement')
types.write(' - Development Agreement')
types.write(' - Distributor Agreement')
types.write(' - Endorsement Agreement')
types.write(' - Franchise Agreement')
types.write(' - Hosting Agreement')
types.write(' - IP Agreement')
types.write(' - Joint Venture Agreement')
types.write(' - License Agreement')
types.write(' - Maintenance Agreement')
types.write(' - Manufacturing Agreement')
types.write(' - Marketing Agreement')
types.write(' - Non-Compete Agreement')
types.write(' - Outsourcing Agreement')
types.write(' - Promotion Agreement')
types.write(' - Reseller Agreement')
types.write(' - Service Agreement')
types.write(' - Sponsorship Agreement')
types.write(' - Supply Agreement')
types.write(' - Strategic Alliance Agreement')
types.write(' - Transportation Agreement')

add_new_lines(2)

performance = st.sidebar.beta_expander("CUAD performance:", expanded=False)
performance.image('perf_roc.png')
performance.image('perf_cat.png')

add_new_lines(5)

st.sidebar.image("JAGGAER-Logo-HiRes-RGB-Red.png")

question_1 = st.text_input(label='Insert a query:')
st.text("OR")
question_2 = st.selectbox('Choose one of the 41 queries from the CUAD dataset:', questions)
# paragraph = st.text_area(label="Contract")
st.write("----")
uploaded_file = st.file_uploader("Upload a contract", type=["png", "jpeg", "jpg", "pdf", "txt", "doc"])
paragraph = ""
temp_folder = ""
if uploaded_file is not None:
	temp_folder = f'{uploaded_file.name}_' + uploaded_file.name.split('.')[1]
	if not os.path.exists(temp_folder):
		os.mkdir(temp_folder)
		with open(temp_folder + '/' + uploaded_file.name, "wb") as f:
			f.write(uploaded_file.getbuffer())
	paragraph = read_file(temp_folder + '/' + uploaded_file.name)
	st.write(paragraph)
#     file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type,"FileSize":uploaded_file.size}
#     st.write(file_details)

if (not len(paragraph) == 0) and (not (len(question_1) == 0) or not (len(question_2) == 0)):

	if len(question_1) > 0:
		# needs to be in a form of a list
		question = [question_1]
	else:
		question = [question_2]

	prediction = run_prediction(question, paragraph, 'cuad-models/roberta-base/')

	for i, p in enumerate(prediction):
		st.write(f"Question: {question[int(p)]}\n\n\n Answer: {prediction[p]}\n\n\n\n")

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: #e2c46e;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Made by <a href="https://www.jaggaer.com/" target="_blank">Jaggaer</a>, Data Science Team
</p>

</div>
"""
#<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTHfPyw2U2ndPdWXFejvcqzPPE-Ygqb_SAI3g&usqp=CAU" height="50">
st.markdown(footer, unsafe_allow_html=True)

