import streamlit as st
import pandas as pd
import numpy as np
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List, Literal
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanSquaredError


st.set_page_config(page_title="ChemoGenomics", page_icon="🧬", layout="centered")


if "page" not in st.session_state:
  st.session_state.page = "Home"


with st.sidebar:
    st.image("image/cg.png", width=300)
    if st.button("Home",use_container_width=True):
        st.session_state.page = "Home"
    if st.button("Detect",use_container_width=True):
        st.session_state.page = "Detect"
    if st.button("Documentation",use_container_width=True):
        st.session_state.page = "Documentation"

if st.session_state.page == "Home":
    
    st.image("image/cg.png")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
    "<h2 style='text-align: center; font-size: 24px;'>Cancer Protocol Recommendations Based on Gene Mutations</h2>", 
    unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)
    left, middle, right = st.columns(3)
    with middle:
        if st.button("Detect Now!", use_container_width=True):
            st.session_state.page = "Detect"
            
  

elif st.session_state.page == "Detect":
  
  
  user_input = st.text_input("Enter Gene Mutation Code (separate with commas)", placeholder="Example: A1BG, A1CF, A2M")

  if st.button("Start Analysis", use_container_width=True):
    if not user_input:  
        st.warning("Please enter the gene mutation code first!")
    else:
        input_mutations = [code.strip() for code in user_input.split(",")]
        for code in input_mutations:
            def integrated_drug_prediction(input_mutations):
                """
                Fungsi terintegrasi untuk memprediksi obat berdasarkan mutasi gen.
                    
                Parameters:
                input_mutations (list): List nama gen yang bermutasi (contoh: ["BRAF", "EGFR"])
                model (keras.Model): Model neural network yang telah di-trained
                    
                Returns:
                str: Nama obat dengan IC50 terendah yang telah disetujui klinis
                """
                
                    # 1. Baca data referensi gen dan obat
                data_gene = "data/gdsc_and_ccle_704_overlapped_cells_mut.csv"
                drug_path = "data/drug_names.csv"
                clinical_path = "data/data_klinis_obat.csv"
                
                # 2. Convert gene names to indices
                def convert_gene_names_to_indices(gene_names):
                    df = pd.read_csv(data_gene)
                    sample_index = 0
                    list_index = []
                        
                    for gene in gene_names:
                        result = df[df["Gene"] == gene].index
                        gene_index = result[0] + 1 if not result.empty else None  # 1-based index
                        list_index.append(gene_index)
                        
                    return [sample_index, list_index]
                
                # 3. Proses konversi indeks gen
                sample_index, gene_indices = convert_gene_names_to_indices(input_mutations)
                
                # 4. Buat input array untuk model
                df_genes = pd.read_csv(data_gene)
                n_genes = len(df_genes)
                mut_input = np.zeros((1, n_genes), dtype=int)
                valid_indices = [idx-1 for idx in gene_indices if idx is not None]
                mut_input[sample_index, valid_indices] = 1

                # 5. Lakukan prediksi
                model = load_model("models/model_final_mut.h5", custom_objects={'mse': MeanSquaredError()})
                result = model.predict([mut_input])
                result_1 = result[0].reshape(-1, 1)
                
                # 6. Normalisasi hasil
                def normalize_output(df):
                    df['Normalized_Result'] = np.where(
                        df['IC50'] < -1,
                        np.log2(df['IC50'] + abs(df['IC50'].min()) + 1),
                        np.log2(df['IC50'] + 1)
                    )
                    return df['Normalized_Result']
                
                result_df = pd.DataFrame(result_1, columns=['IC50'])
                normalized_result = normalize_output(result_df)
                normalized_result = normalized_result.values[:265].reshape(265, 1)

                    # 7. Gabung dengan nama obat
                drug_names = pd.read_csv(drug_path)
                drug_names.columns = ['Drug name']
                drug_names = drug_names.iloc[:265]
                
                normalized_df = pd.DataFrame(normalized_result, columns=['IC50'])
                final_df = pd.concat([drug_names, normalized_df], axis=1)

                # 8. Filter dan ranking obat
                def proses_data_obat(df_ic50):
                    df_clinical = pd.read_csv(clinical_path)
                        
                    df_ic50["Drug name"] = df_ic50["Drug name"].str.strip().str.lower()
                    df_clinical["Drug name"] = df_clinical["Drug name"].str.strip().str.lower()
                        
                    df_merged = pd.merge(df_ic50, df_clinical, on="Drug name", how="inner")
                    df_filtered = df_merged[df_merged["TKO"] == "clinically approved"]
                    df_sorted = df_filtered.sort_values(by="IC50")
                    return df_sorted.iloc[0]["Drug name"] if not df_sorted.empty else None 
                return proses_data_obat(final_df)

            result = integrated_drug_prediction(input_mutations)

        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            api_key="AIzaSyBvNFemrUfTXiuZhhKOQTeF1XMmqTFiXBU",
            temperature=0.5,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

        class Drug(BaseModel):
            name: str = Field(..., description="Drug name")
            dose: str = Field(..., description="Dosage of the drug in the protocol (e.g., '100 mg/m²')")
            route: Literal["i.v.", "oral", "intrathecal"] = Field(..., description="Route of administration of the drug")
            mechanism: str = Field(..., description="Mechanism of action of the drug")
            time: str = Field(..., description="Time schedule of the drug administration (e.g., 'Day 1')")

        class Protocol(BaseModel):
            protocol_name: str = Field(..., description="Protocol name")
            drugs: List[Drug] = Field(..., min_items=1, description="List of drugs in the protocol")
            additional_instructions: str = Field(..., description="Additional information about the protocol")
            explanation: str = Field(..., description="Explanation of selected chemotherapy protocol")
            reference: str = Field(..., description="Reference for the protocol")

        class ListProtocol(BaseModel):
            Protocol: List[Protocol]

        class ProtocolJsonOutputParser(JsonOutputParser):
            """Parser khusus untuk mem-parsing JSON dengan key 'annotations' berisi list anotasi."""
            def get_format_instructions(self) -> str:
                # Instruksi format output ke LLM
                return (
                    "Return a JSON object with a key 'Protocol' that is an array of objects, "
                    "where each object has the following keys: "
                    "Drug, Chemotheraphy_protocol, Explanation, Reason, Reference."
                )
            
            def parse(self, text: str):
                # Tetap menggunakan mekanisme parse bawaan, tapi bisa ditambah error handling
                return super().parse(text)
            
        parser = JsonOutputParser(pydantic_object=ListProtocol)

        prompt = PromptTemplate(
            template=(
                "You are an expert in medical oncology and chemotherapy protocols. "
                "Your task is to generate a structured chemotherapy protocol as a dictionary "
                "for the given drug, ensuring accuracy and completeness. The dictionary should "
                "include all necessary details such as dosage, route of administration, and treatment timeline.\n\n"
                "Base on the drug {{Drug}}, provide a chemotherapy protocol with the following details:\n\n"
                "If the protocol requires multiple drugs to be administered together, include ALL drugs in the 'drugs' array"
                "Never omit any drug that is part of the standard protocol regimen"

                "Example multiple drugs:\n"
                "{\n"
                '   "protocol_name": "CHOP",\n'
                '   "drugs": [\n'
                '       {\n'
                '           "name": "Cyclophosphamide",\n'
                '           "dose": "750 mg/m²",\n'
                '           "route": "i.v.,\n'
                '           "time": "Day 1"\n'
                '           "mechanism": "Alkylating agent",\n'
                '       },\n'
                '       {\n'
                '           "name": "Doxorubicin",\n'
                '           "dose": "50 mg/m²",\n'
                '           "route": "i.v.",\n'
                '           "time": "Day 1"\n'
                '           "mechanism": "Topoisomerase II inhibitor",\n'
                '   ],\n'
                '   "additional_instructions": "..."\n'
                '   "explanation": "..."\n'
                '   "reason": "..."\n'
                '   "reference": "..."\n'
                "}\n\n"


                "### Input Data:\n"
                "- **Drug Name**: {{Drug}}\n"
                "- **Dosage**: Provide appropriate dosage based on standard chemotherapy guidelines.\n"
                "- **Route of Administration**: Specify if the drug is administered intravenously (i.v.), orally, or intrathecally.\n"
                "- **Time Schedule**: Detail the days on which the drug should be administered.\n"
                "- **Mechanism of Action**: Explain the mechanism of action of the drug and its relevance to the treatment.\n"
                "- **Additional Instructions**: If the drug requires supportive care, reduction in dosage, "
                "or additional medications like folinic acid or mesna, include them in the output.\n\n"

                "### Expected Output Format:\n"
                "Return the result as a Python dictionary with the following structure:\n\n"

                "{\n"
                '   "protocol_name": "<protocol_name>",\n'
                '   "drugs": [\n'
                '       {\n'
                '           "name": "<drug_name>",\n'
                '           "dose": "<dose_value> mg/m²",\n'
                '           "route": "<i.v./oral/intrathecal>",\n'
                '           "mechanism": "<mechanism>",\n'
                '           "time": "Day <X>"\n'
                '       }\n'
                '   ],\n'
                '   "additional_instructions": "<any additional notes>"\n'
                '   "explanation": "Explanation"\n'
                '   "reference": "Reference"\n'
                "}\n\n"
                "Output must follow this format:\n{{format_instructions}}"
            ),
            input_variables=["Drug"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
            template_format="jinja2"

        )

        chain = prompt | llm | parser

        resp = chain.invoke({
                "Drug": result
        })
        
        with st.spinner("AI sedang menganalisis mutasi gen..."):
                if isinstance(resp, dict) and "Protocol" in resp:  
                    st.subheader("**List of Protocols and Drugs**")
                    for protocol in resp["Protocol"]:
                        st.markdown(f"### Protocol's Name: {protocol['protocol_name']}")
                        data_obat = []
                        for drug in protocol["drugs"]:
                            data_obat.append({
                                "Drug Name": drug["name"],
                                "Dose": drug["dose"],
                                "Route": drug["route"],
                                "Mechanis,": drug["mechanism"],
                                "Time": drug["time"]
                            }) 
                        df_obat = pd.DataFrame(data_obat)
                        st.dataframe(df_obat, hide_index=True)
                        with st.expander("Additional Information"):
                            st.write(f"**Additional Instruction:** {protocol['additional_instructions']}")
                            st.write(f"**Explanation:** {protocol['explanation']}")
                            st.write(f"**Reference:** {protocol['reference']}")
                        
                        st.markdown("---") 
                else:
                    st.warning("There is no protocol available for this gene mutation.")

            

elif st.session_state.page == "Documentation":
    st.title("📖 ChemoGenomics Documentation")
    st.write(
        """
        *ChemoGenomics* is an AI-based system that combines *ShinyDeepDR* and *Large Language Models (LLMs)* to provide personalized *chemotherapy protocols* based on a patient's *genetic mutations*.
        """
    )

   
    st.header("🔹 Application Features")
    st.markdown(
        """
        - *🔬 Drug Effectiveness Prediction* → Analyzes how *265 anti-cancer drugs* respond to genetic mutations.
        - *💊 Chemotherapy Protocol Recommendations* → Determines the optimal drugs, dosages, and administration schedules.
        - *📖 Scientific Explanations* → Provides information on how each recommended drug works and why it was chosen.
        - *📚 Scientific References* → Supports recommendations with credible sources.
        - *📄 Comprehensive Documentation* → Stores analysis results for further evaluation.
        """
    )

    st.header("🔹 System Workflow")
    st.write("How *ChemoGenomics* works to provide precise chemotherapy recommendations:")
    st.image("image/Workflow.png", caption="Workflow Diagram", use_container_width=True)  

    st.markdown(
        """
        1️⃣ *📥 Data Input* → Users input the patient's *genetic mutation* data.  
        2️⃣ *✅ Data Validation* → The system ensures the genetic mutation data is correctly formatted.  
        3️⃣ *🧪 Drug Analysis (ShinyDeepDR)* → The AI model predicts how *265 anti-cancer drugs* will respond to the genetic mutations.  
        4️⃣ *🤖 LLM Recommendations (Gemini API)* → The AI provides the most suitable *chemotherapy protocols*.  
        5️⃣ *📊 Final Results* → The system displays the *drugs, dosages, administration schedules, and scientific references*.  
        6️⃣ *🔄 Evaluation & Feedback* → Users can provide *feedback to improve the system*.  
        """
    )


    st.header("🔹 Conclusion")
    st.markdown(
        """
        *ChemoGenomics* is an AI-driven innovation that combines *ShinyDeepDR and LLMs* to provide *chemotherapy protocol recommendations based on a patient's genetic mutations*.
        
        🎯 *Benefits of ChemoGenomics:*  
        ✅ *High Precision* → Recommendations are based on the patient's unique characteristics.  
        ✅ *Assists Medical Professionals* → Helps doctors determine the most effective therapy.  
        ✅ *Clear Explanations* → Provides scientific reasons for each recommendation.  
        ✅ *Improves Treatment Effectiveness* → Minimizes the risk of drug resistance and cancer recurrence.  
        """
    )

  
    st.info(
        "🔍 *Note:* This system does not replace consultation with a doctor. "
        "Always discuss the recommendations with medical professionals before making clinical decisions."
    )


    st.markdown("---")
    st.write("© 2025 ChemoGenomics. All Rights Reserved.")
