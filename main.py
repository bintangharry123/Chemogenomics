import streamlit as st
import pandas as pd
import numpy as np
from google import genai  # Pastikan Anda sudah menginstal Gemini API
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List, Literal

# Konfigurasi halaman
st.set_page_config(page_title="Analisis Mutasi Gen", page_icon="🧬", layout="centered")





if "page" not in st.session_state:
  st.session_state.page = "Home"
page = st.sidebar.selectbox("# ChemoGenomics", ["Home", "Detect", "Documentation"], index=["Home", "Detect", "Documentation"].index(st.session_state.page))


if page == "Home":
    st.session_state.page = "Home"
    st.image(".venv/cg.png")
    st.markdown("<h2 style='text-align: center;'>Analisis Mutasi Gen dengan AI</h2>", unsafe_allow_html=True)  

    left, middle, right = st.columns(3)
    with middle:
        if st.button("Detect Now!", use_container_width=True):
            st.session_state.page = "Detect"
            
  

elif page == "Detect":
  
  st.session_state.page = "Detect"
  user_input = st.text_input("Masukkan Kode Mutasi Gen", placeholder="Contoh: YGSUA")
  Drug_input = user_input

  
  Drug_input = "Doxorubicine"
  llm = ChatGoogleGenerativeAI(
      model="gemini-1.5-pro",
      api_key="-",
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
      reason: str = Field(..., description="Reason for recommending this protocol")
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
            "If the protocol requires multiple drugs to be administered together, include ALL drugs in the 'drugs' array. "
            "Never omit any drug that is part of the standard protocol regimen.\n\n"
            "Output must be a valid JSON object with the following structure:\n"
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
            '   "additional_instructions": "<any additional notes>",\n'
            '   "explanation": "<explanation>",\n'
            '   "reason": "<reason>",\n'
            '   "reference": "<reference>"\n'
            "}\n\n"
            "### Example Output:\n"
            "{\n"
            '   "protocol_name": "HDMTX/IFO/DEP",\n'
            '   "drugs": [\n'
            '       {\n'
            '           "name": "Methotrexate",\n'
            '           "dose": "4 g/m²",\n'
            '           "route": "4 hours i.v.",\n'
            '           "time": "Day 1",\n'
            '           "mechanism": "inhibits prostaglandin production in the central nervous system"\n'
            '       },\n'
            '       {\n'
            '           "name": "Ifosfamide",\n'
            '           "dose": "2 g/m²",\n'
            '           "route": "i.v.",\n'
            '           "time": "Day 1",\n'
            '           "mechanism": "alkylating agent"\n'
            '       }\n'
            '   ],\n'
            '   "additional_instructions": "Ensure adequate hydration and urine alkalinization.",\n'
            '   "explanation": "This protocol is used for high-risk patients.",\n'
            '   "reason": "Effective against aggressive tumors.",\n'
            '   "reference": "NCCN Guidelines, Version 2023"\n'
            "}\n\n"
            "Output must follow this format exactly:\n{{format_instructions}}"
        ),
        input_variables=["Drug"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
        template_format="jinja2"
    )

  chain = prompt | llm | parser

  resp = chain.invoke({
      "Drug": Drug_input
  })


if st.button("Analisis Gen", use_container_width=True ):
    if user_input == Drug_input:
     st.write(resp)
     with st.spinner("AI sedang menganalisis mutasi gen..."):
             # Pastikan hasil dari chain.invoke() adalah dictionary
            
            # 🔹 Pastikan result adalah dictionary sebelum akses key-nya
            if isinstance(resp, dict):
                protocol_name = resp.get("protocol_name")
                drugs = resp.get("drugs", [])

                # 🔹 Menampilkan hasil
                st.subheader(f"Protocol Name: {protocol_name}")

                if drugs:
                    
                    st.write("**Detail Obat dalam Protokol:**")
                    data_obat = []

                    for drug in drugs:
                      data_obat.append ({
                         "Nama Obat" : drug.get("name"),
                         "Dosis" : drug.get("dose"),
                         "Rute Pemberian" : drug.get("route"),
                         "Mekanisme Aksi" : drug.get("mechanism"),
                         "Jadwal" : drug.get("time")
                        })
                           
                    df_obat = pd.DataFrame(data_obat)
                    st.dataframe(df_obat, hide_index=True) 
                       
                else:
                  st.write("Tidak ada obat dalam protokol ini.")

                # Menampilkan informasi tambahan

                with st.expander("Informasi Tambahan"):
                 st.write(f" ** # Instruksi Tambahan:** {resp.get('additional_instructions', 'Tidak tersedia')}")
                 st.write(f" **# Penjelasan:** {resp.get('explanation', 'Tidak tersedia')}")
                 st.write(f" **# Alasan Penggunaan:** {resp.get('reason', 'Tidak tersedia')}")
                 st.write(f" **# Referensi:** {resp.get('reference', 'Tidak tersedia')}")
                
                

            else:
                st.warning("Hasil tidak dalam format yang diharapkan. Silakan coba lagi.")
    else:
       st.warning("Silakan masukkan kode mutasi gen terlebih dahulu.")

elif page == "Documentation":
   st.write("# ")
