
## install and login to get access llama3
!pip install transformers==4.40.0 wandb openai nltk torch scikit-learn retry

!pip install google-generativeai

!pip install llm2vec

!pip install flash-attn --no-build-isolation

!huggingface-cli login --token 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


## Train
To run our method, use command: 

!python train.py --task_name Tacred --num_k 5 --num_gen 5 >> log-llm2vec-full_setting_Tacred.txt
!python train.py --task_name Fewrel --num_k 5 --num_gen 2 >> log-llm2vec-full_setting_Fewrel.txt

You can refer to `config.ini` to adjust other hyperparameters.

