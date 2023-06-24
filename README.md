# MultiFactor
Anonymous github repository for EMNLP'23 submission: "Improving Question Generation with Multi-level Content Planning".

# Note
1. We implement our model,  based on source code of `modeling_t5.py` in Transformers 4.20.1.
    -   If raise any errors when using Huggingface Generator of higher version with our modeling file, please success the error functions of  `transformers/src/transformers/generation/utils.py` in `$ProjHome$/src/MultiFactor/modeling_bridget5.py`.
2. We evulate the BLEU socre using `NLTk` instead of `sacrebleu` for two reasons:
   1. In practice, the difference between the results of `sacrebleu` and `NLTK` are around $0.1$.
   2. We follow previous works so use the `NLTK`.
3. `METEOR` score will fluctuate wildly if using different packages. Please read https://github.com/nltk/nltk/issues/2655 for details. Here, we use the `meteor` api in `pycocoevalcap`.
4. Different version of `bert_score` will influence the final score hugly, our version is `0.3.10`.
5. We provide a evaluate script demo in `${ProjHome/evaluate.py`
      
# Parameters Explanation
## Model Config:
Source code: `${ProjHome}/src/MultiFactor/multi_factor_config.py`

1. `model_type` (str): `baseline`, `nodel_cls`, `MultiFactor`.
2. `cls_loss_weight` (float): Training loss weights of phrase classification auxiliary task 
3. `if_cross_enhanced_k` (bool): if use the phrase-selection probability project layer in cross-attn module
4. `hard_flag` (int, 0~3): It decides the $W^{\delta}$'s input is soft or hard. Convert it into binary number, ``1` means "hard", the first one is training and second is inference:
   1. "00": link the phrase classifier output and the $W^{\delta}$', in which case the gradient will boardcast from $W^{\delta}$' to phrase classifier.
   2. "10": use hard label in training and the soft probability during inference, proposed **MultiFactor**.
   3. "11": hard training and hard inference, or the **One-hot PET-Q model**.
5. We still provide some parameters for other explorations like:
   1. `model_freeze`: training our model using some freeze policy (need modify the Trainer in `${ProjHome}/src/MultiFactor/multi_factor_trainer.py`).
   2. `if_full_answer_decoder`: using multi-task generation and so on.

## Data Format:
Source code: `${ProjHome}/src/MultiFactor/multi_factor_dataset.py`

1. `data_name`: dataset name, here we conduct experiments on `cqg`, `pcqg` (passage-level cqg), and `squad_1.1_zhou`.
2. `data_format`: control the input fotmat
3. `max_length`: input sequence max length, we set default value $512$. In sentence-level QG task: `pcqg` (passage-level cqg), and `squad_1.1_zhou`, $256$ is enough.


# Reimplement Steps
## 1. Pre-process datasets
Because it is a cumbersome step, we provide a `demo.json` based on CQG `dev.json` [1] directly, which has included the `pseudo-gold full answer` constucted mentioned in our paper.

And the preprocessing source code is complicated, and we will publish it after orgnizing it in `${ProjHome}\preprocess`.

## 2. Prepare dataset.pt file
We read raw data json file in `${ProjHome}\dataset\${dataset_name}\${split}.json`, and generate corresponding `.pt` in `${ProjHome}\dataset\${dataset_name}\${data_foramt}\${split}.pt`. And if the model input contains (a) full answer(s), the full answer json file path is `${ProjHome}\dataset\${dataset_name}\${data_foramt}\${split}.json`.

```shell
python ${ProjHome}/src/MultiFactor/multi_factor_dataset.py \ 
    -d ${dataset_name} \ 
    -j ${data_format} \
    -l ${max_length}
```

## 3. Training the model
1. Initialize default config in  `${ProjHome}/src/config.ini`. You can edit:
   1. Base model path (Please init the "T5" or "MixQG" model path here)
   2. Output path
   3. batch size
   4. and all parameters shown in `{ProjHome}/src/MultiFactor/arguments.py` and `{ProjHome}/src/run.py`.
2. running the command:
```shell
python \
  ${ProjHome}/src/run.py \
  -c ${ProjHome}/src/config.ini \
  -j ${dataset_name} \
  	-f ${data_foramt} \
    --seed ${seed} \
    --model_type ${model_type} \
    --cls_loss_weight 1.0 \
    --learning_rate 1e-4 \
    --num_train_epochs 10 \
    --num_beams 1 \
    --hard_flag 2 \
    --save_model_pt True
```



# Ref
[1] https://github.com/sion-zcfei/cqg
