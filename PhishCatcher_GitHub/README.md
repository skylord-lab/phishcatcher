# PhishCatcher

Machine-learning based phishing URL detection project.

## Structure

```text
PhishCatcher/
├── src/
│   ├── feature_extraction.py
│   └── extracted_original.py
├── dataset/
├── models/
├── tests/
├── requirements.txt
├── README.md
└── .gitignore
```

## Run

```bash
python -m pip install -r requirements.txt
python src/feature_extraction.py
```

## Note

The original academic source refers to external datasets and saved model files.
Do not upload datasets/models unless redistribution is permitted. The original
accuracy figures should not be presented as independently verified until the
evaluation code is corrected.
