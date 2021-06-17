import joblib
from cdqa.utils.evaluation import evaluate_pipeline

cdqa_pipeline = joblib.load('bert_qa.joblib')

evaluate_pipeline(cdqa_pipeline, 'cdqa-v1.1.json')