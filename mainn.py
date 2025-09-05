import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from huggingface_hub import hf_hub_download
import os
import pandas as pd
import re

if "page" not in st.session_state:
    st.session_state.page = "home"

def go_to_page(page_name):
    st.session_state.page = page_name

if st.session_state.page == "home":
    st.markdown("""
        <style>
        .title {
            text-align: center;
            font-size: 50px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .subheader {
            text-align: center;
            font-size: 15px;
            margin-bottom: 5px;
        }
        .card {
            border: 1px solid #333;
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            background-color: #1e1e1e;
            box-shadow: 2px 2px 8px rgba(0,0,0,0.4);
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='title'>ATQI.IO</div>", unsafe_allow_html=True)
    st.markdown("<div class='subheader'>DEMO DEEP LEARNING DEVELOPMENT</div>", unsafe_allow_html=True)
    
    st.divider()
    st.subheader("Tentang ATQI.IO")
    st.write("ATQI.IO merupakan sebuah website yang dibuat sebagai demo implementasi **Deep Learning Deployment**. Setiap model dilatih menggunakan dataset open-source dan di-deploy agar bisa langsung dicoba oleh pengguna melalui antarmuka yang sederhana.")


    st.subheader("1. Klasifikasi Anjing vs Kucing")
    st.image(os.path.join(os.getcwd(), "img support", "dogvscat.png"))
    st.caption("Upload foto hewan peliharaanmu dan ketahui apakah itu anjing atau kucing.")
    st.button("🐶 Anjing vs Kucing 🐱", on_click=go_to_page, args=("catdog",))
    
    
    st.subheader("2. Klasifikasi Makanan (101 Food)")
    st.image(os.path.join(os.getcwd(), "img support", "food.png"))
    st.caption("Upload foto makanan dan sistem akan mengenali jenis makanan tersebut.")
    st.button("🍔 Food Classification", on_click=go_to_page, args=("food",))
    
    
    st.subheader("3. Emotion Mining")
    st.image(os.path.join(os.getcwd(), "img support", "emotion.png"))
    st.caption("Masukkan sebuah kalimat, lalu model akan menentukan apakah isinya bernada marah, support, kecewa, sedih, atau harapan .")
    st.button("💬 Emotion Analysis", on_click=go_to_page, args=("sentiment",))

elif st.session_state.page == "catdog":
    st.image(os.path.join(os.getcwd(), "img support", "dogvscat.png"))
    st.title("🐶 Klasifikasi Anjing vs Kucing 🐱")
    
    tabs1,tabs2 = st.tabs(['Program', 'Informasi Model'])
    
    with tabs1:
        
        model_catdog = hf_hub_download(
            repo_id = "atq-m/catvsdog",
            filename = "CNN_CatVSDog.h5"
        )
        
        model_cd = tf.keras.models.load_model(model_catdog, compile=False)
        st.write("Input shape model:", model_cd.input_shape)
        
        def preprocess(img):
            target_size = model_cd.input_shape[1:3]
            img = img.resize(target_size)
            img = np.array(img)/255.0
            return np.expand_dims(img, axis=0)
        
        uploaded = st.file_uploader("Upload Gambar", type=['jpg', 'jpeg', 'png'])
        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, caption="Gambar diunggah", use_column_width=True)
            input_img = preprocess(img)
            
            pred = model_cd.predict(input_img)
            
            if pred.shape[1] == 1:
                label = "Anjing" if pred[0][0] > 0.5 else "Kucing"
            else :
                label = "Anjing" if np.argmax(pred[0]) == 0 else "Kucing"
                
            st.subheader(f"Prediksi : {label}")
    
    with tabs2:
        with st.expander("📖 Deskripsi Model"):
            st.write("Model ini digunakan untuk mengklasifikasi hewan apakah hewan tersebut kucing atau anjing.")

        with st.expander("📊 Dataset"):
            st.write("""
            - Sumber        : https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset/
            - Jumlah data   : 25.000 (80% train, 10% validation, 10% test)
            - Preprocessing : Augmentasi, Rescale
            """)

        with st.expander("🧠 Arsitektur Model"):
            st.write("**Jenis Model :** CNN (Convolutional Neural Network) untuk klasifikasi biner") 
            st.write("**Input :** Gambar 128 × 128 × 3 (RGB)") 
            st.write("**Layer :**") 
            st.write("- Conv2D(32, 3×3) + MaxPooling(2×2)") 
            st.write("- Conv2D(64, 3×3, same) + MaxPooling(2×2)") 
            st.write("- Conv2D(128, 3×3, same) + MaxPooling(2×2)") 
            st.write("- Flatten → Dense(512, ReLU) + Dropout(0.5)") 
            st.write("- Dense(1, Sigmoid)") 
            st.write("**Output :** Probabilitas (0 atau 1)")
            st.write("**Optimizer      :** Adam (lr=0.0001)")
            st.write("**Loss      :** Binary Crossentropy")
            st.write("**Callbacks      :** Early Stop")

        with st.expander("📈 Evaluasi Model"):
            st.subheader("Accuracy")
            st.image(os.path.join(os.getcwd(), "img support", "traincd.png"))
            st.subheader("Loss")
            st.image(os.path.join(os.getcwd(), "img support", "traincd.png"))
            data = {
                "precision": [0.85, 0.85],
                "recall": [0.85, 0.85],
                "f1-score": [0.85, 0.85],
                "support": [1172, 1180]
            }

            index = ["cats", "dogs"]
            df = pd.DataFrame(data, index=index)

            extra = pd.DataFrame({
                "precision": [None, 0.85, 0.85],
                "recall": [None, 0.85, 0.85],
                "f1-score": [0.85, 0.85, 0.85],
                "support": [2352, 2352, 2352]
            }, index=["accuracy", "macro avg", "weighted avg"])

            df_report = pd.concat([df, extra])

            st.title("Classification Report")
            st.dataframe(df_report, use_container_width=True)   # Bisa juga st.dataframe(df_report)


    st.button("⬅️ Kembali ke Home", on_click=go_to_page, args=("home",))

elif st.session_state.page == "food":
    st.image(os.path.join(os.getcwd(), "img support", "food.png"))
    st.title("🍔 Klasifikasi Makanan")
    
    tabs1,tabs2 = st.tabs(['Program', 'Informasi Model'])
    
    with tabs1:
        model_101food = hf_hub_download(
            repo_id = "atq-m/101-food",
            filename = "CNN_101-food.h5"
        )
        
        model_food = tf.keras.models.load_model(model_101food, compile=False)
        st.write("Input shape model:", model_food.input_shape)
        
        class_names = [
            "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare",
            "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito",
            "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake",
            "ceviche", "cheese_plate", "cheesecake", "chicken_curry", "chicken_quesadilla",
            "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder",
            "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes",
            "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict", "escargots",
            "falafel", "filet_mignon", "fish_and_chips", "foie_gras", "french_fries",
            "french_onion_soup", "french_toast", "fried_calamari", "fried_rice",
            "frozen_yogurt", "garlic_bread", "gnocchi", "greek_salad", "grilled_cheese_sandwich",
            "grilled_salmon", "guacamole", "gyoza", "hamburger", "hot_and_sour_soup",
            "hot_dog", "huevos_rancheros", "hummus", "ice_cream", "lasagna", "lobster_bisque",
            "lobster_roll_sandwich", "macaroni_and_cheese", "macarons", "miso_soup",
            "mussels", "nachos", "omelette", "onion_rings", "oysters", "pad_thai", "paella",
            "pancakes", "panna_cotta", "peking_duck", "pho", "pizza", "pork_chop", "poutine",
            "prime_rib", "pulled_pork_sandwich", "ramen", "ravioli", "red_velvet_cake",
            "risotto", "samosa", "sashimi", "scallops", "seaweed_salad", "shrimp_and_grits",
            "spaghetti_bolognese", "spaghetti_carbonara", "spring_rolls", "steak",
            "strawberry_shortcake", "sushi", "tacos", "takoyaki", "tiramisu", "tuna_tartare",
            "waffles"
        ]


        def preprocess(img):
            target_size = model_food.input_shape[1:3]
            img = img.resize(target_size)
            img = np.array(img)/255.0
            return np.expand_dims(img, axis=0)
        
        uploaded = st.file_uploader("Upload Gambar", type=['jpg', 'jpeg', 'png'])
        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, caption="Gambar diunggah", use_column_width=True)
            input_img = preprocess(img)

            pred = model_food.predict(input_img)

            idx = np.argmax(pred[0])
            label = class_names[idx]
            confidence = np.max(pred[0]) * 100
            st.subheader(f"Prediksi: {label} ({confidence:.2f}%)")
 
    with tabs2:
        with st.expander("📖 Deskripsi Model"):
            st.write("Model ini digunakan untuk melakukan klasifikasi makanan yang ada pada dataset 101 food.")

        with st.expander("📊 Dataset"):
            st.write("""
            - Sumber        : https://www.kaggle.com/datasets/dansbecker/food-101/
            - Jumlah data   : 101.000 (80% train, 10% validation, 10% test)
            - Preprocessing : Augmentasi, Rescale
            """)

        with st.expander("🧠 Arsitektur Model"):
            st.write("**Jenis Model :** CNN (Convolutional Neural Network) untuk klasifikasi multi-kelas (101 kelas)")
            st.write("**Input       :** Gambar 128 × 128 × 3 (RGB)")
            st.write("**Layer       :**")
            st.write("- Conv2D(32, 7×7, strides=2, padding='same') + MaxPooling(3×3)")
            st.write("- Conv2D(64, 3×3, padding='same') + MaxPooling(2×2)")
            st.write("- Conv2D(128, 3×3, padding='same') + MaxPooling(2×2)")
            st.write("- Conv2D(256, 3×3, padding='same') + MaxPooling(2×2)")
            st.write("- GlobalAveragePooling2D()")
            st.write("- Dense(512, ReLU) + Dropout(0.5)")
            st.write("- Dense(101, Softmax)")
            st.write("**Output      :** Probabilitas untuk 101 kelas")
            st.write("**Optimizer      :** Adam (lr=0.0001)")
            st.write("**Loss      :** Categorical Crossentropy")
            st.write("**Callbacks      :** Early Stop, Checkpoint")


        with st.expander("📈 Evaluasi Model"):
            st.subheader("Accuracy")
            st.image(os.path.join(os.getcwd(), "img support", "trainfood.png"))
            st.subheader("Loss")
            st.image(os.path.join(os.getcwd(), "img support", "trainfood2.png"))
            report_text = """
            Classification Report:

                         precision    recall  f1-score   support

              apple_pie       0.18      0.05      0.07       192
         baby_back_ribs       0.25      0.56      0.34       192
                baklava       0.27      0.40      0.32       186
         beef_carpaccio       0.52      0.51      0.51       187
           beef_tartare       0.25      0.30      0.27       186
             beet_salad       0.32      0.49      0.38       189
               beignets       0.44      0.61      0.51       192
               bibimbap       0.34      0.67      0.45       192
          bread_pudding       0.21      0.12      0.15       188
      breakfast_burrito       0.23      0.05      0.08       189
             bruschetta       0.30      0.19      0.24       186
           caesar_salad       0.42      0.53      0.47       191
                cannoli       0.25      0.42      0.31       194
          caprese_salad       0.42      0.34      0.38       189
            carrot_cake       0.23      0.45      0.30       191
                ceviche       0.20      0.13      0.16       193
           cheese_plate       0.31      0.14      0.19       192
             cheesecake       0.26      0.31      0.28       194
          chicken_curry       0.41      0.20      0.27       194
     chicken_quesadilla       0.27      0.15      0.19       192
          chicken_wings       0.32      0.38      0.35       193
         chocolate_cake       0.32      0.50      0.39       195
       chocolate_mousse       0.17      0.10      0.13       192
                churros       0.34      0.21      0.26       195
           clam_chowder       0.47      0.43      0.45       192
          club_sandwich       0.24      0.27      0.25       190
             crab_cakes       0.15      0.07      0.09       192
           creme_brulee       0.41      0.61      0.49       192
          croque_madame       0.31      0.59      0.41       190
              cup_cakes       0.25      0.17      0.20       193
           deviled_eggs       0.33      0.35      0.34       192
                 donuts       0.22      0.09      0.13       188
              dumplings       0.53      0.50      0.52       192
                edamame       0.93      0.83      0.88       191
          eggs_benedict       0.22      0.41      0.28       190
              escargots       0.44      0.23      0.30       191
                falafel       0.20      0.09      0.13       186
           filet_mignon       0.23      0.24      0.23       187
         fish_and_chips       0.30      0.22      0.25       192
              foie_gras       0.18      0.17      0.17       191
           french_fries       0.39      0.50      0.43       187
      french_onion_soup       0.39      0.34      0.36       192
           french_toast       0.27      0.30      0.28       191
         fried_calamari       0.26      0.16      0.20       191
             fried_rice       0.36      0.38      0.37       192
          frozen_yogurt       0.26      0.39      0.31       188
           garlic_bread       0.38      0.17      0.24       190
                gnocchi       0.29      0.11      0.16       191
            greek_salad       0.44      0.50      0.46       189
grilled_cheese_sandwich       0.28      0.06      0.10       190
         grilled_salmon       0.23      0.20      0.21       191
              guacamole       0.51      0.32      0.39       189
                  gyoza       0.34      0.21      0.26       191
              hamburger       0.28      0.10      0.15       186
      hot_and_sour_soup       0.48      0.69      0.56       192
                hot_dog       0.22      0.29      0.25       187
       huevos_rancheros       0.17      0.11      0.14       193
                 hummus       0.23      0.04      0.06       193
              ice_cream       0.15      0.11      0.12       188
                lasagna       0.25      0.25      0.25       190
         lobster_bisque       0.46      0.61      0.53       192
  lobster_roll_sandwich       0.26      0.51      0.34       193
    macaroni_and_cheese       0.30      0.20      0.24       191
               macarons       0.36      0.46      0.40       190
              miso_soup       0.59      0.61      0.60       190
                mussels       0.54      0.59      0.57       191
                 nachos       0.23      0.12      0.16       182
               omelette       0.38      0.15      0.21       193
            onion_rings       0.32      0.46      0.38       190
                oysters       0.58      0.53      0.55       190
               pad_thai       0.30      0.53      0.38       195
                 paella       0.47      0.38      0.42       186
               pancakes       0.44      0.26      0.32       191
            panna_cotta       0.23      0.28      0.25       186
            peking_duck       0.30      0.35      0.32       195
                    pho       0.51      0.79      0.62       193
                  pizza       0.56      0.40      0.46       191
              pork_chop       0.23      0.15      0.18       192
                poutine       0.28      0.36      0.32       193
              prime_rib       0.57      0.33      0.42       190
   pulled_pork_sandwich       0.30      0.21      0.25       193
                  ramen       0.34      0.37      0.35       188
                ravioli       0.38      0.06      0.10       184
        red_velvet_cake       0.28      0.76      0.41       188
                risotto       0.27      0.26      0.26       187
                 samosa       0.27      0.26      0.26       189
                sashimi       0.56      0.57      0.57       194
               scallops       0.14      0.09      0.11       192
          seaweed_salad       0.64      0.73      0.68       188
       shrimp_and_grits       0.22      0.20      0.21       192
    spaghetti_bolognese       0.30      0.58      0.39       190
    spaghetti_carbonara       0.39      0.54      0.45       191
           spring_rolls       0.29      0.28      0.28       192
                  steak       0.27      0.17      0.21       190
   strawberry_shortcake       0.25      0.48      0.33       190
                  sushi       0.42      0.21      0.28       187
                  tacos       0.22      0.12      0.15       193
               takoyaki       0.18      0.21      0.19       189
               tiramisu       0.37      0.32      0.34       186
           tuna_tartare       0.28      0.15      0.20       190
                waffles       0.19      0.32      0.24       184

               accuracy                           0.33     19225
              macro avg       0.33      0.33      0.31     19225
           weighted avg       0.33      0.33      0.31     19225
            """

            rows = []
            for line in report_text.split("\n"):
                parts = re.split(r"\s{2,}", line.strip())
                if len(parts) >= 5 and parts[1].replace(".","",1).isdigit():
                    rows.append(parts)

            df = pd.DataFrame(rows, columns=["class", "precision", "recall", "f1-score", "support"])

            for col in ["precision", "recall", "f1-score", "support"]:
               df[col] = pd.to_numeric(df[col], errors="coerce")

            df[["precision", "recall", "f1-score"]] = df[["precision", "recall", "f1-score"]].round(2)

            st.subheader("📊 Classification Report")
            st.dataframe(df, use_container_width=True)
    
    st.write("Halaman untuk model Food Classification.")
    st.button("⬅️ Kembali ke Home", on_click=go_to_page, args=("home",))

elif st.session_state.page == "sentiment":
    st.image(os.path.join(os.getcwd(), "img support", "emotion.png"))
    st.title("💬 Analisis Emosi")
    
    tabs1,tabs2 = st.tabs(['Program', 'Informasi Model'])
    
    with tabs1:
        from huggingface_hub import hf_hub_download
        from transformers import TFBertModel, AutoTokenizer
        import tensorflow as tf
        import numpy as np
        import streamlit as st

        class BertLayer(tf.keras.layers.Layer):
            def __init__(self, model_name="indobenchmark/indobert-base-p1", **kwargs):
                super().__init__(**kwargs)
                self.bert = TFBertModel.from_pretrained(model_name)

            def call(self, inputs):
                return self.bert(inputs)[1] 

        model_path = hf_hub_download(
            repo_id="atq-m/nlp-emotion-mining",
            filename="NLP_Emotion_Mining.h5"
        )

        custom_objects = {
            "TFBertModel": BertLayer,
            "BertEncoder": BertLayer,   
        }

        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
        tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")

        SEQ_LEN = 128
        label_map = {
            0: "anger",
            1: "disappointment",
            2: "hope",
            3: "sadness",
            4: "support"
        }

        def preprocess(text):
            encodings = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=SEQ_LEN,
                return_tensors="tf"
            )
            return {
                "input_ids": encodings["input_ids"],
                "attention_mask": encodings["attention_mask"]
            }

        def predict_emotion(text):
            inputs = preprocess(text)
            preds = model.predict(inputs)
            label_id = np.argmax(preds, axis=1)[0]
            return label_map[label_id]

        # === Streamlit UI ===
        user_input = st.text_area("Masukkan teks komentar:")
        if st.button("Prediksi"):
            if user_input.strip():
                result = predict_emotion(user_input)
                st.subheader(f"Prediksi Emosi: {result}")


    with tabs2:
        with st.expander("📖 Deskripsi Model"):
            st.write("Model ini digunakan untuk melakukan emotion mining pada komentar teks berbahasa Indonesia.")

        with st.expander("📊 Dataset"):
            st.write("""
            - Sumber: https://www.kaggle.com/competitions/emotion-mining-on-comments-about-tom-lembong-case/data
            - Jumlah data: 5.083 (80% train, 20% test)
            - Preprocessing: oversampling, tokenisasi IndoBERT, padding, truncation
            """)

        with st.expander("🧠 Arsitektur Model"):
            st.write("**Jenis Model :** BERT (Transformer-based) untuk klasifikasi multi-kelas (5 kelas)")
            st.write("**Input       :**")
            st.write("- input_ids (None, 128)")
            st.write("- attention_mask (None, 128)")
            st.write("**Layer       :**")
            st.write("- BERT Encoder → output (None, 768)")
            st.write("- Dropout")
            st.write("- Dense(128, ReLU)")
            st.write("- Dropout")
            st.write("- Dense(5, Softmax)")
            st.write("**Output      :** Probabilitas untuk 5 kelas")
            st.write("**Optimizer   :** Adam (lr=0.00003)")
            st.write("**Loss        :** Sparse Categorical Crossentropy")
            st.write("**Callbacks   :** EarlyStopping")

        with st.expander("📈 Evaluasi Model"):
            st.subheader("Loss")
            st.image(os.path.join(os.getcwd(), "img support", "cmNLP.png"))
            report_text = """
               precision    recall  f1-score   support

         ANGER     0.5972    0.6277    0.6121       274
DISAPPOINTMENT     0.2020    0.4082    0.2703        49
          HOPE     0.5133    0.6784    0.5844       199
       SADNESS     0.7095    0.4504    0.5510       282
       SUPPORT     0.6543    0.5775    0.6135       213

      accuracy                         0.5674      1017
     macro avg     0.5353    0.5484    0.5262      1017
  weighted avg     0.6048    0.5674    0.5736      1017
            """

            rows = []
            for line in report_text.split("\n"):
                parts = re.split(r"\s{2,}", line.strip())
                if len(parts) >= 5 and parts[1].replace(".","",1).isdigit():
                    rows.append(parts)

            df = pd.DataFrame(rows, columns=["class", "precision", "recall", "f1-score", "support"])

            for col in ["precision", "recall", "f1-score", "support"]:
               df[col] = pd.to_numeric(df[col], errors="coerce")

            df[["precision", "recall", "f1-score"]] = df[["precision", "recall", "f1-score"]].round(2)

            st.subheader("📊 Classification Report")
            st.dataframe(df, use_container_width=True)
        
    st.write("Halaman untuk model Analisis Sentimen.")
    st.button("⬅️ Kembali ke Home", on_click=go_to_page, args=("home",))



