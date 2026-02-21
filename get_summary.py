from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import re
from glob import glob
from tqdm import tqdm
import multiprocessing

caption_root = "/work/narus/data_selected/"
caption_root = "/data2/narus/HO3D/train/"
caption_root = "/data2/narus/data_selected/"
#summary_root = "/data2/narus/HO3D/summary"
summary_root = "/data2/narus/HO3D/train/"
#summary_root = "/data2/narus/few_samples/"
summary_root = "/data2/narus/data_selected/"
model_name = "mistralai/Mistral-7B-Instruct-v0.2"

def format_prompt(text, remove_pose_desc=True, remove_intention=False, remove_taxonomy=False):
    cleaned_text = text
    if remove_pose_desc:
        cleaned_text = re.sub(r"^8\. hand_pose_description:.*(?:\n|$)", "", text, flags=re.MULTILINE)
    if remove_intention:
        cleaned_text = re.sub(r"^6\. intention:.*(?:\n|$)", "", cleaned_text, flags=re.MULTILINE)
    if remove_taxonomy:
        cleaned_text = re.sub(r"^7\. grasp_taxonomy:.*(?:\n|$)", "", cleaned_text, flags=re.MULTILINE)

        instruction = (
            "You are given a list of bullet points describing an article.\n"
            "Summarize the overall content in a single sentence.\n"
            "Start your response with either 'A right hand' or 'A left hand', depending on which is more appropriate based on the content.\n"
            "Do not include any information that is not explicitly mentioned in the bullet points.\n\n"
            "### Instruction:\n"
            "Summarize the following bullet points as a single sentence starting with either 'A right hand' or 'A left hand'. "
            "**You must not add any information that is not clearly and explicitly stated in the bullet points.**\n\n"
        )
    else:
        instruction = (
            "You are given a list of bullet points describing an article.\n"
            "Summarize the overall content in a single sentence.\n"
            "Start your response with either 'A right hand' or 'A left hand', depending on which is more appropriate based on the content.\n"
            "Do not include any information that is not explicitly mentioned in the bullet points.\n"
            "**Be sure to include the taxonomy explicitly if it is mentioned in the bullet points.**\n\n"
            "### Instruction:\n"
            "Summarize the following bullet points as a single sentence starting with either 'A right hand' or 'A left hand'. "
            "**The summary must include the taxonomy if it appears in the bullet points.** "
            "**You must not add any information that is not clearly and explicitly stated in the bullet points.**\n\n"
        )

    return instruction + f"### Input:\n{cleaned_text}\n\n### Response:"

def process_files(local_gpu_id, file_list):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(local_gpu_id)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto"
        ).eval()

    for path in tqdm(file_list, desc=f"GPU {local_gpu_id}"):
        relative_path = os.path.relpath(path, caption_root)
        save_path = os.path.join(summary_root, relative_path)
        save_path = os.path.join(caption_root, relative_path).replace("pred_taxonomy_caption", "summary")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if os.path.exists(save_path):
            continue
        print(f"save path: {save_path}")
        with open(path, "r") as f:
            caption = f.read().strip()
        if not caption:
            continue

        prompt = format_prompt(caption)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            temperature=0.7,
            top_p=0.95
        )
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        summary_text = summary.split("### Response:")[-1].strip()

        with open(save_path, "w") as f:
            f.write(summary_text)

if __name__ == "__main__":
    # 1. CUDA_VISIBLE_DEVICES から使用GPUリストを取得
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    gpu_ids = [int(i) for i in cuda_visible.split(",") if i.strip().isdigit()]
   # print("=== CUDA_VISIBLE_DEVICES 確認 ===")
   # print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "未設定"))
   # print("torch.cuda.device_count():", torch.cuda.device_count())

    #for i in range(torch.cuda.device_count()):
     #   print(f"  torch sees GPU {i}: {torch.cuda.get_device_name(i)}")
    if not gpu_ids:
        print("No usable GPUs found in CUDA_VISIBLE_DEVICES.")
        exit(1)

    # 2. caption 下の全 txt ファイルを取得
    #txt_files = glob(f"{caption_root}/*/pred_grasp_taxonomy_caption/*.txt")
    txt_files = glob(f"{caption_root}/pred_taxonomy_caption/**/*.txt", recursive=True)
    txt_files = [f for f in txt_files if os.path.isfile(f)]

    # 3. ファイルをGPU数で分割
    num_procs = len(gpu_ids)
    chunk_size = (len(txt_files) + num_procs - 1) // num_procs
    chunks = [txt_files[i*chunk_size:(i+1)*chunk_size] for i in range(num_procs)]

    # 4. 各GPUにプロセスを割り当てて実行
    processes = []
    for proc_id, local_gpu_id in enumerate(gpu_ids):
        p = multiprocessing.Process(target=process_files, args=(local_gpu_id, chunks[proc_id]))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
