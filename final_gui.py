import tkinter as tk
from tkinter import messagebox,scrolledtext
import threading
import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification,AutoTokenizer
from sklearn.preprocessing import MultiLabelBinarizer
stage_1=r"F:\internship_models\stage_1\saved_model"
stage_2=r"F:\internship_models\stage_2\stage_2"
dataset_path=r"E:\project\Stage_1_dataset\predicted_diseases.xlsx"
print("Loading Stage_1 Model ...")
stage1_model=AutoModelForSequenceClassification.from_pretrained(stage_1,local_files_only=True)
stage1_tokenizer=AutoTokenizer.from_pretrained(stage_1,local_files_only=True)
print("Stage 1 Model Loaded Successfully.")
print("Loading Stage_2 Model ...")
stage2_model=AutoModelForSequenceClassification.from_pretrained(stage_2,local_files_only=True)
stage2_tokenizer=AutoTokenizer.from_pretrained(stage_2,local_files_only=True)
stage1_model.eval()
stage2_model.eval()
print("Both Models are Ready to make Predictions.")
dataset=pd.read_excel(dataset_path)
print("Dataset Loaded Successfully.")
dataset["Disease"]=dataset["Disease"].astype(str).str.strip()
dataset["Cause"]=dataset["Cause"].astype(str).str.strip()
dataset["Medicine"]=dataset["Medicine"].astype(str).str.strip()

dataset["Cause_Disease"]=dataset.apply(lambda row:[row['Cause'],row['Disease']],axis=1)
multilabel=MultiLabelBinarizer()
multilabel.fit(dataset["Cause_Disease"])
print("Loaded MultiLabel Binarizer.")

unique_medicines=dataset["Medicine"].unique().tolist()
label_mapping={index:label for index,label in enumerate(unique_medicines)}
print("Done Label Mapping.")

def predict():
    input_symptom=symptom_entry.get("1.0",tk.END).strip()
    if not input_symptom:
        messagebox.showwarning("First enter symptoms please.")
    input_stage1=stage1_tokenizer([input_symptom],padding=True,truncation=True,return_tensors="pt")
    print("Stage 1 Prediciting...")
    with torch.no_grad():
        output_stage_1=stage1_model(**input_stage1)
        prbabilities_stage1=torch.sigmoid(output_stage_1.logits)
        predictions_stage1=(prbabilities_stage1 >0.5).int()
        predicted_labels_stage_1=multilabel.inverse_transform(predictions_stage1.numpy())[0]
        predicted_disease=next((disease for disease in predicted_labels_stage_1 if disease in dataset["Disease"].unique()),"Unknown")
        predicted_cause=next((cause for cause in predicted_labels_stage_1 if cause in dataset["Cause"].unique()),"Unknown")
        print(f"Predicted Disease: {predicted_disease}")
        print(f"Predicted Cause: {predicted_cause}")
        if predicted_disease !="Unknown":
            print("Tokenization Stage 2...")
            input_stage2=stage2_tokenizer(predicted_disease,padding=True,truncation=True,return_tensors="pt")
            print("Stage 2 Prediction ...")
            with torch.no_grad():
                output_stage2=stage2_model(**input_stage2)
                predicted_label_stage_2=torch.argmax(output_stage2.logits,dim=1).item()
            predicted_medicine=label_mapping.get(predicted_label_stage_2,"Unknown")
            print(f"Predicted Medicine: {predicted_medicine}")
        else:
            predicted_medicine="Unknown"
            print("No Disease Found,skipping Stage 2.")
    output_box.config(state=tk.NORMAL)
    output_box.delete("1.0",tk.END)
    output_box.insert(tk.END,f"Symptoms:{input_symptom}\n")
    output_box.insert(tk.END,f"Predicted Cause:{predicted_cause}\n")
    output_box.insert(tk.END,f"Predicted Disease:{predicted_disease}\n")
    output_box.insert(tk.END,f"Recommended Medicine:{predicted_medicine}\n")
    output_box.config(state=tk.DISABLED)
    print("Prediction Displayed.")

def pred_thread():
    threading.Thread(target=predict).start()

print("Setting up GUI...")
root=tk.Tk()
root.title("Symptom-to-Disease Diagnosis and Medicine Recommendation System")
root.geometry("500x500")
tk.Label(root,text="Enter Symptoms like(wheezing,difficult breathing)",font=("Arial",12,"bold")).pack(pady=5)
symptom_entry=scrolledtext.ScrolledText(root,height=4,width=50)
symptom_entry.pack(pady=5)
tk.Button(root,text="Enter Symptom",command=pred_thread,font=("Arial",12,"bold"),bg="lightyellow").pack(pady=10)
tk.Label(root,text="Causes, Disease and Recommended Medicines",font=("Arial",12,"bold")).pack(pady=5)
output_box=scrolledtext.ScrolledText(root,height=8,width=50,state=tk.DISABLED)
output_box.pack(pady=5)
print("Launching GUI ...")
root.mainloop()




