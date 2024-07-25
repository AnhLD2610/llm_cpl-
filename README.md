
## install and login to get access llama3
!pip install transformers==4.36.0 wandb openai nltk torch scikit-learn retry

!pip install google-generativeai

!pip install llm2vec

!pip install flash-attn --no-build-isolation

!huggingface-cli login --token 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

## Train
To run our method, use command: 
 ```
  bash bash/fewrel_5shot.sh  # for FewRel 5-shot setting
  bash bash/fewrel_10shot.sh # for FewRel 10-shot setting
  bash bash/tacred_5shot.sh  # for TACRED 5-shot setting
  bash bash/tacred_10shot.sh # for TACRED 10-shot setting
```

You can refer to `config.ini` to adjust other hyperparameters.

