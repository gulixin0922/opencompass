import os.path as osp
from mmengine.config import read_base
from copy import deepcopy
from opencompass.models.openai_api import OpenAI, OpenAISDK
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner, SlurmRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.models import HuggingFacewithChatTemplate, TurboMindModelwithChatTemplate
from opencompass.utils.text_postprocessors import extract_non_reasoning_content


#######################################################################
#                          PART 0  Essential Configs                  #
#######################################################################
with read_base():
    # # General Reasoning
    from opencompass.configs.datasets.gpqa.gpqa_0shot_nocot_genericllmeval_gen_772ea0 import (
        gpqa_datasets,
    )
    from opencompass.configs.datasets.supergpqa.supergpqa_llmjudge_gen_12b8bc import (
        supergpqa_datasets,
    )

    # # # Math Calculation
    from opencompass.configs.datasets.HLE.hle_llmverify_gen_6ff468 import (
        hle_datasets,
    )
    from opencompass.configs.datasets.aime2025.aime2025_llmjudge_gen_5e9f4f import (
        aime2025_datasets,
    )
    from opencompass.configs.datasets.OlympiadBench.OlympiadBench_0shot_llmverify_gen_be8b13 import (
        olympiadbench_datasets,
    )
    # from opencompass.configs.datasets.livemathbench.livemathbench_hard_llmjudge_gen_71eaf5 import (
    #     livemathbench_datasets,
    # )
    from opencompass.configs.datasets.OlymMATH.olymmath_llmverify_gen_97b203 import (
        olymmath_datasets,
    )

    # # # Knowledge
    from opencompass.configs.datasets.mmlu_pro.mmlu_pro_0shot_nocot_genericllmeval_gen_08c1de import (
        mmlu_pro_datasets,
    )

    # Academic
    from opencompass.configs.datasets.SmolInstruct.smolinstruct_gen import (
        smolinstruct_datasets,
    )
    from opencompass.configs.datasets.ChemBench.ChemBench_llmjudge_gen_c584cf import (
        chembench_datasets,
    )
    from opencompass.configs.datasets.MedXpertQA.MedXpertQA_llmjudge_gen import (
        medxpertqa_datasets,
    )
    from opencompass.configs.datasets.PHYSICS.PHYSICS_llm_judge_gen_a133a2 import physics_datasets
    from opencompass.configs.datasets.ClimaQA.ClimaQA_Gold_llm_judge_gen_f15343 import climaqa_datasets
    from opencompass.configs.datasets.matbench.matbench_gen_f71840 import matbench_datasets
    
    # Summary Groups
    from opencompass.configs.summarizers.groups.mmlu_pro import (
        mmlu_pro_summary_groups,
    )
    from opencompass.configs.summarizers.groups.OlympiadBench import (
        OlympiadBench_summary_groups,
        OlympiadBenchMath_summary_groups,
        OlympiadBenchPhysics_summary_groups,
    )
    from opencompass.configs.summarizers.groups.PHYSICS import (
        physics_summary_groups,
    )


#######################################################################
#                          PART 1  Datasets List                      #
#######################################################################
# datasets list for evaluation
datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])


#######################################################################
#                       PART 2  Datset Summarizer                     #
#######################################################################
summary_groups = sum(
    [v for k, v in locals().items() if k.endswith('_summary_groups')], []
)

summary_groups.extend(
    [
        {
            'name': 'olymmath_llmjudge',
            'subsets': [
                ['olymmath_llmjudge_en-hard', 'accuracy'],
                ['olymmath_llmjudge_zh-hard', 'accuracy'],
                ['olymmath_llmjudge_en-easy', 'accuracy'],
                ['olymmath_llmjudge_zh-easy', 'accuracy'],
            ],
        },
        {
            'name':'ChemBench',
            'subsets':[
                ['ChemBench_Name_Conversion', 'accuracy'],
                ['ChemBench_Property_Prediction', 'accuracy'],
                ['ChemBench_Mol2caption', 'accuracy'],
                ['ChemBench_Caption2mol', 'accuracy'],
                ['ChemBench_Product_Prediction', 'accuracy'],
                ['ChemBench_Retrosynthesis', 'accuracy'],
                ['ChemBench_Yield_Prediction', 'accuracy'],
                ['ChemBench_Temperature_Prediction', 'accuracy'],
            ]   
        },
        {
            'name': 'ClimaQA',
            'subsets': [
                ['ClimaQA_Gold_cloze', 'accuracy'],
                ['ClimaQA_Gold_ffq', 'accuracy'],
                ['ClimaQA_Gold_mcq', 'accuracy'],
            ],
        },
        {
            'name': 'OlympiadBench',
            'subsets': [
                ['OlympiadBenchMath', 'accuracy'],
                ['OlympiadBenchPhysics', 'accuracy'],
            ],
        },
    ]   
)

# Summarizer
summarizer = dict(
    dataset_abbrs=[
        ['GPQA_diamond', 'accuracy'],
        ['supergpqa', 'accuracy'],
        ['hle_llmjudge', 'accuracy'],
        ['aime2025', 'accuracy'],
        ['livemathbench_hard', 'accuracy'],
        ['olymmath_llmjudge', 'naive_average'],
        ['OlympiadBench', 'naive_average'],
        ['mmlu_pro', 'accuracy'],
        ['medxpertqa', 'accuracy'],
        ['ChemBench', 'naive_average'],
        ['ClimaQA', 'naive_average'],
        ['PHYSICS', 'naive_average'],
        '',
        'smolinstruct',
        ['NC-I2F', 'score'],
        ['NC-I2S', 'score'],
        ['NC-S2F', 'score'],
        ['NC-S2I', 'score'],
        ['PP-ESOL', 'score'],
        ['PP-Lipo', 'score'],
        ['PP-BBBP', 'accuracy'],
        ['PP-ClinTox', 'accuracy'],
        ['PP-HIV', 'accuracy'],
        ['PP-SIDER', 'accuracy'],
        ['MC', 'score'],
        ['MG', 'score'],
        ['FS', 'score'],
        ['RS', 'score'],
        '',
        ['ClimaQA', 'naive_average'],
        ['ClimaQA_Gold_cloze', 'accuracy'],
        ['ClimaQA_Gold_ffq', 'accuracy'],
        ['ClimaQA_Gold_mcq', 'accuracy'],
        '',
        ['PHYSICS', 'naive_average'],
        ['PHYSICS_atomic_dataset_textonly', 'accuracy'],
        ['PHYSICS_electro_dataset_textonly', 'accuracy'],
        ['PHYSICS_mechanics_dataset_textonly', 'accuracy'],
        ['PHYSICS_optics_dataset_textonly', 'accuracy'],
        ['PHYSICS_quantum_dataset_textonly', 'accuracy'],
        ['PHYSICS_statistics_dataset_textonly', 'accuracy'],
        '',
        ['ChemBench', 'naive_average'],
        ['ChemBench_Name_Conversion', 'accuracy'],
        ['ChemBench_Property_Prediction', 'accuracy'],
        ['ChemBench_Mol2caption', 'accuracy'],
        ['ChemBench_Caption2mol', 'accuracy'],
        ['ChemBench_Product_Prediction', 'accuracy'],
        ['ChemBench_Retrosynthesis', 'accuracy'],
        ['ChemBench_Yield_Prediction', 'accuracy'],
        ['ChemBench_Temperature_Prediction', 'accuracy'],
        '',
        'MatBench',
        ['matbench_expt_gap', 'mae'],
        ['matbench_steels', 'mae'],
        ['matbench_expt_is_metal', 'accuracy'],
        ['matbench_glass', 'accuracy'],
        '',
        ['supergpqa', 'accuracy'],
        ['supergpqa', 'SuperGPQA-Engineering'],
        ['supergpqa', 'SuperGPQA-Philosophy'],
        ['supergpqa', 'SuperGPQA-Medicine'],
        ['supergpqa', 'SuperGPQA-Economics'],
        ['supergpqa', 'SuperGPQA-Science'],
        ['supergpqa', 'SuperGPQA-Law'],
        ['supergpqa', 'SuperGPQA-History'],
        ['supergpqa', 'SuperGPQA-Education'],
        ['supergpqa', 'SuperGPQA-Military Science'],
        ['supergpqa', 'SuperGPQA-Management'],
        ['supergpqa', 'SuperGPQA-Literature and Arts'],
        ['supergpqa', 'SuperGPQA-Agronomy'],
        ['supergpqa', 'SuperGPQA-Sociology'],
        '',
        ['mmlu_pro', 'naive_average'],
        ['mmlu_pro_biology', 'accuracy'],
        ['mmlu_pro_business', 'accuracy'],
        ['mmlu_pro_chemistry', 'accuracy'],
        ['mmlu_pro_computer_science', 'accuracy'],
        ['mmlu_pro_economics', 'accuracy'],
        ['mmlu_pro_engineering', 'accuracy'],
        ['mmlu_pro_health', 'accuracy'],
        ['mmlu_pro_history', 'accuracy'],
        ['mmlu_pro_law', 'accuracy'],
        ['mmlu_pro_math', 'accuracy'],
        ['mmlu_pro_philosophy', 'accuracy'],
        ['mmlu_pro_physics', 'accuracy'],
        ['mmlu_pro_psychology', 'accuracy'],
        ['mmlu_pro_other', 'accuracy'],
        '',
        ['OlympiadBench', 'naive_average'],
        ['OlympiadBenchMath', 'accuracy'],
        ['OlympiadBenchPhysics', 'accuracy'],
    ],
    summary_groups=summary_groups,
)


#######################################################################
#                        PART 3  Models  List                         #
#######################################################################
models = []
model_configs = [
    # (abbr, path, num_gpus)
    ('sft_internvl3_plus_academic_v1_x9_12000', '/mnt/hwfile/share_data/wangweiyun/share_swj/run_convert/sft_internvl3_plus_academic_v1_x9/12000', 1),
]
max_seq_len = 32768
max_out_len = 32768
max_batch_size = 128

for abbr, path, num_gpus in model_configs:
    if abbr is None:
        abbr = path.split('/')[-2] + '--' + path.split('/')[-1]

    base_model = dict(
        type=TurboMindModelwithChatTemplate,
        abbr=abbr,
        path=path,
        engine_config=dict(session_len=max_seq_len,
                           max_batch_size=max_batch_size,
                           tp=num_gpus),
        gen_config=dict(top_k=1,
                        temperature=1e-6,
                        top_p=0.9,
                        max_new_tokens=max_out_len),
        stop_words=['<|im_end|>', '<|action_end|>'],
        max_seq_len=max_seq_len,
        max_out_len=max_out_len,
        batch_size=max_batch_size,
        run_cfg=dict(num_gpus=num_gpus),
        pred_postprocessor=dict(type=extract_non_reasoning_content)
    )
    
    model = deepcopy(base_model)
    if 'TurboMindModelwithChatTemplate' in str(model['type']):
        model['gen_config']['top_k'] = 1  # greedy
        model['gen_config']['temperature'] = 1e-6
        models.append(model)
    else:
        models.append(model)

############## Objective LLM Judge  ##############
api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ]
)

obj_llm_judge_cfg = dict(
    abbr='Qwen2_5_32b',
    type=OpenAISDK,
    path='Qwen2_5_32b',
    key='EMPTY',
    retry=5,
    openai_api_base=[
        'http://10.140.60.167:10000/v1',
        'http://10.140.60.167:10001/v1',
        'http://10.140.60.167:10002/v1',
        'http://10.140.60.167:10003/v1',
    ],
    mode='mid',
    meta_template=api_meta_template,
    query_per_second=5,
    batch_size=128,
    temperature=0.001,
    tokenizer_path='/mnt/petrelfs/share_data/gulixin/checkpoints/Qwen2.5-32B-Instruct',
    max_out_len=8192,
    max_seq_len=23808,
)

for item in datasets:
    if 'judge_cfg' in item['eval_cfg']['evaluator']:
        item['eval_cfg']['evaluator']['judge_cfg'] = obj_llm_judge_cfg


#######################################################################
#                 PART 4  Inference/Evaluation Configuaration         #
#######################################################################

# Local Runner
infer = dict(
    partitioner=dict(
        type=NumWorkerPartitioner,
        num_worker=8
        # Similar with data-parallelism, how many workers for evaluation,
        # each worker will evaluate a part of the dataset. Total GPUs = num_worker * num_gpus_per_worker
        # For example, If you have 8 GPUs, for 7B model using 1 GPU for one instance, you can set num_worker=8
        # to max-utilize the GPUs.
        # If you have 8 GPUs, for 14B model using 2 GPUs for one instance, you can set num_worker=4
    ),
    runner=dict(
        type=SlurmRunner,
        max_num_workers=16,
        retry=0, # Modify if needed
        task=dict(type=OpenICLInferTask)
    ),
)

# eval with local runner
eval = dict(
    partitioner=dict(type=NaivePartitioner, n=10),
    runner=dict(
        type=SlurmRunner,
        max_num_workers=16,
        task=dict(type=OpenICLEvalTask)),
)


#######################################################################
#                      PART 5  Utils Configuaration                   #
#######################################################################
base_exp_dir = 'outputs/ai4sci_internvl3_0507/'
work_dir = osp.join(base_exp_dir, 'chat_objective')


