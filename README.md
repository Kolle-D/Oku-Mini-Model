1. Clone the repository
        ↓
2. Install the requirements
        ↓
3. Run preprocessing.py
        ↓
4. Run train.py
        ↓
5. A new models/ folder is created automatically
        ↓
6. Run inference.py or the Streamlit app

8. For example the steps above are outlined below with their commands

9. git clone https://github.com/Kolle-D/Oku-Mini-Model.git
cd OkuTranslator

pip install -r requirements.txt

python src/preprocessing.py

python src/train.py

After Training finishes you should have this

OkuTranslator/
│
├── models/
│   └── oku_model/
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       ├── special_tokens_map.json
│       └── generation_config.json

python src/inference.py

streamlit run app/streamlit_app.py
