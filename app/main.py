from pydantic import BaseModel, validator
from peft import PeftModel, PeftConfig
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, pipeline, AutoModelForSequenceClassification, AutoConfig

app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

peft_model_id = "deutsche-welle/bias_classifier_roberta_base_peft"
config = PeftConfig.from_pretrained(peft_model_id)

model_config = AutoConfig.from_pretrained(
    config.base_model_name_or_path,
    num_labels=2,
    id2label={0: "Non-biased", 1: "Biased"},
    label2id={"Biased": 1, "Non-biased": 0}
)

model = AutoModelForSequenceClassification.from_pretrained(config.base_model_name_or_path, config=model_config)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

model = PeftModel.from_pretrained(model, peft_model_id)
model.eval()

clf = pipeline("text-classification", model=model, tokenizer=tokenizer)


class Response(BaseModel):
    label: str
    score: float


@app.get("/bias_classifier", response_model=Response)
def predict_subjectivity(sentence: str):
    result = clf(sentence)[0]
    return {"label": result["label"],
            "score": result["score"]}
