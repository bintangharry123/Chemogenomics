import streamlit as st
import pandas as pd

st.markdown("# Anjay")
st.markdown(" Anjay")
st.markdown ("**Anjay**")
st.markdown ("*Anjay*")

st.markdown (">Anjay")

st.markdown ('[Youtube](https://youtube.com)')

str = "print('hello world')"
st.code(str)

table = ''' 
|No | Nama|
|---|---|
|1|Bintang|
|2|Harry|
'''
st.markdown(table)

json = {
  "a": "1,2,3,4",
  "b": "satu, dua, tiga, empat"
}

st.json(json)

#add pandas

table = ({"No": [1,2,3,4,5],"Nama":['kantal', 'kintil', 'kuntul', 'kentel','kontol']})
st.table(table)
st.dataframe(table)

st.metric(label= "IPK", value='3.75', delta='0.79')

#adding image

st.image('Tipografii.png',caption = "logo ugm", width = 200 )

#adding button

nama = st.text_input ('Nama')
button = st.button ("Submit")

if button == True:
  st.write('Nama anda: ', nama)
else:
  st.write('namamu siapa anj')


#download button

st.image('tesla.jpeg')
fileName = st.text_input("File Name")
st.write ("Your file name is: ", fileName )

with open('tesla.jpeg', "rb") as file:
  butt = st.download_button(
    label = "Downlaod your image",
    data = file,
    file_name = fileName,
    mime = "image/png"
  )

  #link button

  imageList = ['tesla.jpeg', 'Tipografii.png']
  captionList = ['Cybertruck', 'UGM']
  st.image(image = imageList, width = 50, caption = captionList)
  st.link_button('Youtube','https://youtube.com')

  #checkbox


st.image('cg.png')

#toggle widget
click=st.toggle("click this")
if click:
  st.image('cg.png')
else : st.write()

#radio button
pilihan  = st.columns(2)
with pilihan[0]:
  st.text_input("Masukkan Gene Mutasi")
  
with pilihan[1]:
  jenis = st.radio("Jenis",["RNA","DNA"],index=None)

if jenis == "RNA":
  st.write("kamu memilih RNA")
if jenis == "DNA":
  st.write("kamu memilih DNA")

else: index=None

#selectbox widget
modelInfo =[{"NAMA_MODEL": "ChatGPT",
             "VERSI_MODEL":"4.0"
            },{
             "NAMA_MODEL":"Gemini",
             "VERSI_MODEL":"3.0"
            },{
             "NAMA_MODEL":"ClaudeAI",
             "VERSI_MODEL":"2.5"
            }
             ]
selected = st.selectbox('Mau memilih model yang mana?',
                        [modelInfo[0]['NAMA_MODEL']+" "+ modelInfo[0]['VERSI_MODEL'],
                        modelInfo[1]['NAMA_MODEL']+" "+ modelInfo[1]['VERSI_MODEL'],
                        modelInfo[2]['NAMA_MODEL']+" "+ modelInfo[2]['VERSI_MODEL']])

if selected == modelInfo[0]['NAMA_MODEL']+" "+ modelInfo[0]['VERSI_MODEL']:
  st.write('Kamu memilih ChatGPT')

if selected == modelInfo[1]['NAMA_MODEL']+" "+ modelInfo[1]['VERSI_MODEL']:
  st.write('Kamu memilih Gemini')

if selected == modelInfo[2]['NAMA_MODEL']+" "+ modelInfo[2]['VERSI_MODEL']:
  st.write('Kamu memilih ClaudeAI')
