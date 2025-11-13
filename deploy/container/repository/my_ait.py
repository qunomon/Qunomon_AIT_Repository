#!/usr/bin/env python
# coding: utf-8

# # AIT Development notebook

# ## notebook of structure

# | #  | Name                                               | cells | for_dev | edit               | description                                                                |
# |----|----------------------------------------------------|-------|---------|--------------------|----------------------------------------------------------------------------|
# | 1  | [Environment detection](##1-Environment-detection) | 1     | No      | uneditable         | detect whether the notebook are invoked for packaging or in production     |
# | 2  | [Preparing AIT SDK](##2-Preparing-AIT-SDK)         | 1     | Yes     | uneditable         | download and install AIT SDK                                               |
# | 3  | [Dependency Management](##3-Dependency-Management) | 3     | Yes     | required(cell #2)  | generate requirements.txt for Docker container                             |
# | 4  | [Importing Libraries](##4-Importing-Libraries)     | 2     | Yes     | required(cell #1)  | import required libraries                                                  |
# | 5  | [Manifest Generation](##5-Manifest-Generation)     | 1     | Yes     | required           | generate AIT Manifest                                                      |
# | 6  | [Prepare for the Input](##6-Prepare-for-the-Input) | 1     | Yes     | required           | generate AIT Input JSON (inventory mapper)                                 |
# | 7  | [Initialization](##7-Initialization)               | 1     | No      | uneditable         | initialization for AIT execution                                           |
# | 8  | [Function definitions](##8-Function-definitions)   | N     | No      | required           | define functions invoked from Main area.<br> also define output functions. |
# | 9  | [Main Algorithms](##9-Main-Algorithms)             | 1     | No      | required           | area for main algorithms of an AIT                                         |
# | 10 | [Entry point](##10-Entry-point)                    | 1     | No      | uneditable         | an entry point where Qunomon invoke this AIT from here                     |
# | 11 | [License](##11-License)                            | 1     | Yes     | required           | generate license information                                               |
# | 12 | [Deployment](##12-Deployment)                      | 1     | Yes     | uneditable         | convert this notebook to the python file for packaging purpose             |

# ## notebook template revision history

# 1.0.1 2020/10/21
# 
# * add revision history
# * separate `create requirements and pip install` editable and noeditable
# * separate `import` editable and noeditable
# 
# 1.0.0 2020/10/12
# 
# * new cerarion

# ## body

# ### #1 Environment detection

# [uneditable]

# In[1]:


# Determine whether to start AIT or jupyter by startup argument
import sys
is_ait_launch = (len(sys.argv) == 2)


# ### #2 Preparing AIT SDK

# [uneditable]

# In[2]:


if not is_ait_launch:
    # get ait-sdk file name
    from pathlib import Path
    from glob import glob
    import re
    import os

    current_dir = get_ipython().run_line_magic('pwd', '')

    ait_sdk_path = "./ait_sdk-*-py3-none-any.whl"
    ait_sdk_list = glob(ait_sdk_path)
    ait_sdk_name = os.path.basename(ait_sdk_list[-1])

    # install ait-sdk
    get_ipython().system('pip install -q --upgrade pip')
    get_ipython().system('pip install -q --no-deps --force-reinstall ./$ait_sdk_name')


# ### #3 Dependency Management

# #### #3-1 [uneditable]

# In[3]:


if not is_ait_launch:
    from ait_sdk.common.files.ait_requirements_generator import AITRequirementsGenerator
    requirements_generator = AITRequirementsGenerator()


# #### #3-2 [required]

# In[4]:


if not is_ait_launch:
    requirements_generator.add_package('mlflow', '3.1.4')
    requirements_generator.add_package('pandas', '2.3.3')
    requirements_generator.add_package('evaluate', '0.4.6')
    requirements_generator.add_package('ipywidgets', '8.1.7')
    requirements_generator.add_package('transformers', '4.57.1')
    requirements_generator.add_package('torch', '2.8.0')
    requirements_generator.add_package('torchvision', '0.23.0')
    requirements_generator.add_package('torchaudio', '2.8.0')
    requirements_generator.add_package('rouge-score', '0.1.2')


# #### #3-3 [uneditable]

# In[ ]:


if not is_ait_launch:
    requirements_generator.add_package(f'./{ait_sdk_name}')
    requirements_path = requirements_generator.create_requirements(current_dir)

    get_ipython().system('pip install -q -r $requirements_path ')


# ### #4 Importing Libraries

# #### #4-1 [required]

# In[ ]:


# import if you need modules cell

import mlflow
import os
import pandas as pd
import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import torch
from rouge_score import rouge_scorer
from collections import Counter
from mlflow.metrics import MetricValue
import numpy as np


# #### #4-2 [uneditable]

# In[ ]:


# must use modules
from os import path
import shutil  # do not remove
from ait_sdk.common.files.ait_input import AITInput  # do not remove
from ait_sdk.common.files.ait_output import AITOutput  # do not remove
from ait_sdk.common.files.ait_manifest import AITManifest  # do not remove
from ait_sdk.develop.ait_path_helper import AITPathHelper  # do not remove
from ait_sdk.utils.logging import get_logger, log, get_log_path  # do not remove
from ait_sdk.develop.annotation import measures, resources, downloads, ait_main  # do not remove
# must use modules


# ### #5 Manifest Generation

# [required]

# In[ ]:


if not is_ait_launch:
    from ait_sdk.common.files.ait_manifest_generator import AITManifestGenerator
    manifest_genenerator = AITManifestGenerator(current_dir)
    manifest_genenerator.set_ait_name('eval_llm_rouge_score')
    manifest_genenerator.set_ait_description('MLFlowを使用して、LLMモデルでテキストに対してレジュメ生成し、その生成されたテキストの品質を評価します。LLM評価基準を用いて、テキストのROUGEスコアを計算し、テキストの質を数値化します。')
    manifest_genenerator.set_ait_source_repository('https://github.com/aistairc/Qunomon_AIT_Repository/tree/eval_llm_rouge_score')
    manifest_genenerator.set_ait_version('1.2')
    manifest_genenerator.add_ait_licenses('Apache License Version 2.0')
    manifest_genenerator.add_ait_keywords('LLM')
    manifest_genenerator.add_ait_keywords('MLFlow')
    manifest_genenerator.add_ait_keywords('ROUGE')
    manifest_genenerator.set_ait_quality('https://ait-hub.pj.aist.go.jp/ait-hub/api/0.0.1/qualityDimensions/機械学習品質マネジメントガイドライン第三版/C-1機械学習モデルの正確性')
    inventory_requirement_data = manifest_genenerator.format_ait_inventory_requirement(format_=['json'])
    manifest_genenerator.add_ait_inventories(name='generate_data', 
                                              type_='dataset', 
                                              description='説明とレジュメのデータセット \nJSON形式{inputs:array, references:array}\n例：{inputs: [Artificial intelligence is rapidly transforming industries.], references: [AI is revolutionizing industries by enhancing data processing and decision-making.]', 
                                              requirement=inventory_requirement_data)
    inventory_requirement_model = manifest_genenerator.format_ait_inventory_requirement(format_=['ALL'])
    manifest_genenerator.add_ait_inventories(name='llm_model_dir', 
                                              type_='model', 
                                              description='事前トレーニング済みの大規模言語モデルと設定ファイルのディレクトリ（例: HuggingfaceのT5）\n必要なファイルはconfig.json, model.safetensors, generation_config.json, special_tokens_map.json, tokenizer_config.json, tokenizer.jsonを含む', 
                                              requirement=inventory_requirement_model)
    manifest_genenerator.add_ait_measures(name='ROUGE1_Score', 
                                           type_='float', 
                                           description='計算されたROUGE1スコア', 
                                           structure='single',
                                           min='0')
    manifest_genenerator.add_ait_measures(name='ROUGE2_Score', 
                                           type_='float', 
                                           description='計算されたROUGE2スコア', 
                                           structure='single',
                                           min='0')
    manifest_genenerator.add_ait_measures(name='ROUGE_L_Score', 
                                           type_='float', 
                                           description='計算されたROUGE_Lスコア', 
                                           structure='single',
                                           min='0')
    manifest_genenerator.add_ait_measures(name='ROUGE_W_Score', 
                                           type_='float', 
                                           description='計算されたROUGE_Wスコア', 
                                           structure='single',
                                           min='0')
    manifest_genenerator.add_ait_measures(name='ROUGE_S_Score', 
                                           type_='float', 
                                           description='計算されたROUGE_Sスコア', 
                                           structure='single',
                                           min='0')
    manifest_genenerator.add_ait_resources(name='rouge_score_table',  
                                           type_='table', 
                                           description='ROUGEスコアが最も低い10セットのデータサンプル')
    manifest_genenerator.add_ait_downloads(name='Log', 
                                            description='AIT実行ログ')
    manifest_genenerator.add_ait_downloads(name='MLFlow_table', 
                                            description='MLFlow実行結果CSV')
    manifest_path = manifest_genenerator.write()


# ### #6 Prepare for the Input

# [required]

# In[ ]:


if not is_ait_launch:
    from ait_sdk.common.files.ait_input_generator import AITInputGenerator
    input_generator = AITInputGenerator(manifest_path)
    input_generator.add_ait_inventories(name='generate_data',
                                     value='generate_data.json')
    input_generator.add_ait_inventories(name='llm_model_dir',
                                     value='model')
    input_generator.write()


# ### #7 Initialization

# [uneditable]

# In[ ]:


logger = get_logger()

ait_manifest = AITManifest()
ait_input = AITInput(ait_manifest)
ait_output = AITOutput(ait_manifest)

if is_ait_launch:
    # launch from AIT
    current_dir = path.dirname(path.abspath(__file__))
    path_helper = AITPathHelper(argv=sys.argv, ait_input=ait_input, ait_manifest=ait_manifest, entry_point_dir=current_dir)
else:
    # launch from jupyter notebook
    # ait.input.json make in input_dir
    input_dir = '/usr/local/qai/mnt/ip/job_args/1/1'
    current_dir = get_ipython().run_line_magic('pwd', '')
    path_helper = AITPathHelper(argv=['', input_dir], ait_input=ait_input, ait_manifest=ait_manifest, entry_point_dir=current_dir)

ait_input.read_json(path_helper.get_input_file_path())
ait_manifest.read_json(path_helper.get_manifest_file_path())

### do not edit cell


# ### #8 Function definitions

# [required]

# In[ ]:


@log(logger)
@measures(ait_output, 'ROUGE1_Score')
def mean_rouge1(mean_rouge):
    return mean_rouge

@log(logger)
@measures(ait_output, 'ROUGE2_Score')
def mean_rouge2(mean_rouge):
    return mean_rouge

@log(logger)
@measures(ait_output, 'ROUGE_L_Score')
def mean_rougeL(mean_rouge):
    return mean_rouge

@log(logger)
@measures(ait_output, 'ROUGE_W_Score')
def mean_rougeW(mean_rouge):
    return mean_rouge

@log(logger)
@measures(ait_output, 'ROUGE_S_Score')
def mean_rougeS(mean_rouge):
    return mean_rouge


# In[ ]:


@log(logger)
@resources(ait_output, path_helper, 'rouge_score_table', 'rouge_score_table.csv')
def rouge_score_table(df, file_path: str=None) -> None:
    df.to_csv(file_path)


# In[ ]:


@log(logger)
# ROUGE_Wを計算する
def _compute_rouge_w(reference, candidate, beta=1.2):
    def lcs(x, y):
        n, m = len(x), len(y)
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if x[i - 1] == y[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[n][m]

    ref_tokens = reference.split()
    cand_tokens = candidate.split()
    lcs_length = lcs(ref_tokens, cand_tokens)
    precision = lcs_length / len(cand_tokens) if cand_tokens else 0
    recall = lcs_length / len(ref_tokens) if ref_tokens else 0
    if precision + recall == 0:
        return 0
    return (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)

# ROUGE_Sを計算する
def _compute_rouge_s(reference, candidate):
    def skip_bigrams(text):
        tokens = text.split()
        return Counter((tokens[i], tokens[j]) for i in range(len(tokens)) for j in range(i + 1, len(tokens)))

    ref_bigrams = skip_bigrams(reference)
    cand_bigrams = skip_bigrams(candidate)
    overlap = sum((ref_bigrams & cand_bigrams).values())
    precision = overlap / sum(cand_bigrams.values()) if cand_bigrams else 0
    recall = overlap / sum(ref_bigrams.values()) if ref_bigrams else 0
    if precision + recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)

def standard_aggregations(scores):
    return {"mean": np.mean(scores)}

# ROUGE_Wの評価メトリクス
def rouge_w_fn(predictions, targets):
    scores = []
    for pred, target in zip(predictions, targets):
        score = _compute_rouge_w(target, pred)
        scores.append(score)
    # 集約
    return MetricValue(scores=scores, aggregate_results=standard_aggregations(scores))
# ROUGE_Sの評価メトリクス
def rouge_s_fn(predictions, targets):
    scores = []
    for pred, target in zip(predictions, targets):
        score = _compute_rouge_s(target, pred)
        scores.append(score)
    # 集約
    return MetricValue(scores=scores, aggregate_results=standard_aggregations(scores))


# In[ ]:


@log(logger)
@downloads(ait_output, path_helper, 'MLFlow_table', 'MLFlow_table.csv')
def eval_result(eval_table, file_path: str=None) -> str:
    eval_table.to_csv(file_path, index=False)


# In[ ]:


@log(logger)
@downloads(ait_output, path_helper, 'Log', 'ait.log')
def move_log(file_path: str=None) -> str:
    shutil.move(get_log_path(), file_path)


# ### #9 Main Algorithms

# [required]

# In[ ]:


@log(logger)
@ait_main(ait_output, path_helper, is_ait_launch)
def main() -> None:
    # 並列処理の警告を抑制
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    with open(ait_input.get_inventory_path('generate_data'), "r") as file:
        json_data = json.load(file)

    eval_data = pd.DataFrame(json_data)
    
    # ローカルに保存されたLLMモデルを読み込む
    tokenizer_path = ait_input.get_inventory_path('llm_model_dir')
    model_path = ait_input.get_inventory_path('llm_model_dir')
    
    # Transformers を使用してモデルとトークナイザをロード
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    # パイプラインの作成
    device = 0 if torch.cuda.is_available() else -1  # GPUが利用可能なら0、そうでなければ-1（CPU）
    text2text_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device)

    # モデルの予測関数
    def predict(inputs):
        outputs = text2text_pipeline(inputs, max_new_tokens=50)
        return outputs[0]["generated_text"]

    # 予測値を計算してデータに追加
    eval_data["predictions"] = eval_data["inputs"].apply(predict)
        
    with mlflow.start_run() as run:
        mlflow.transformers.log_model(
            transformers_model=text2text_pipeline,  # パイプラインを渡す
            artifact_path="model",
        )
        # ログされたモデルの URI を取得
        logged_model_uri = f"runs:/{run.info.run_id}/model"
        # ROUGE_WとROUGE_Sの評価メソッド
        rouge_w_metric = mlflow.metrics.make_metric(
            eval_fn=rouge_w_fn, greater_is_better=True, name="rougeW"
        )

        rouge_s_metric = mlflow.metrics.make_metric(
            eval_fn=rouge_s_fn, greater_is_better=True, name="rougeS"
        )
        results = mlflow.evaluate(
            model=logged_model_uri,
            data=eval_data,
            targets="references",
            predictions="predictions",
            model_type=None,
            extra_metrics=[
                mlflow.metrics.rouge1(),
                mlflow.metrics.rouge2(),
                mlflow.metrics.rougeL(),
                rouge_w_metric,
                rouge_s_metric
            ]
        )
        # 評価結果表示
        print(f"See evaluation table below: \n{results.metrics}")
        # 評価結果テーブル
        eval_table = results.tables["eval_results_table"]
        eval_result(eval_table)
        # 評価結果テーブル表示
        print(f"See evaluation table below: \n{eval_table}")
        # ROUGEの平均スコアをmeasuresに設定
        mean_rouge1(results.metrics.get("rouge1/v1/mean", 0))
        mean_rouge2(results.metrics.get("rouge2/v1/mean", 0))
        mean_rougeL(results.metrics.get("rougeL/v1/mean", 0))
        mean_rougeW(results.metrics.get("rougeW/mean", 0))
        mean_rougeS(results.metrics.get("rougeS/mean", 0))
        # ROUGEスコアで昇順にソートし、上位10行を取得
        sorted_df = eval_table.sort_values(by="rouge1/v1/score", ascending=True).head(10)
        rouge_score_table(sorted_df)
            
    # AIT実行ログ出力
    move_log()


# ### #10 Entry point

# [uneditable]

# In[ ]:


if __name__ == '__main__':
    main()


# ### #11 License

# [required]

# In[ ]:


ait_owner='AIST'
ait_creation_year='2024'


# ### #12 Deployment

# [uneditable] 

# In[ ]:


if not is_ait_launch:
    from ait_sdk.deploy import prepare_deploy
    from ait_sdk.license.license_generator import LicenseGenerator
    
    current_dir = get_ipython().run_line_magic('pwd', '')
    prepare_deploy(ait_sdk_name, current_dir, requirements_path)
    
    # output License.txt
    license_generator = LicenseGenerator()
    license_generator.write('../top_dir/LICENSE.txt', ait_creation_year, ait_owner)


# In[ ]:




