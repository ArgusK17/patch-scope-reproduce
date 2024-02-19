import csv
import random

class translation_task:
   def __init__(self):
      self.zero_shot=None
      self.few_shot=None
      self.answers=None
      self.info={}

def gen_task(prompt_num=1, example_num=3, seed=0, same_task=True, to_lg=0):
    random.seed(seed)
    task=translation_task()

    vocabulary = []
    with open('/gpfs/share/home/2201110046/LLM/tools/data/vocabulary.csv', mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        header = next(reader)
        for row in reader:
            vocabulary.append(row)

    if same_task:
        task_id=random.randint(0,len(vocabulary)-1)
        except_id_list=[i for i in range(task_id)]+[i for i in range(task_id+1,len(vocabulary))]

    lang_num = len(vocabulary[0])
    except_lg_list = [i for i in range(to_lg)]+[i for i in range(to_lg+1, lang_num)]

    prompts_few_shot = [[] for _ in range(lang_num-1)]
    prompts_zero_shot = [[] for _ in range(lang_num-1)]
    answers=[]
    for _ in range(prompt_num):
        if not same_task:
            task_id=random.randint(0,len(vocabulary)-1)
            except_id_list=[i for i in range(task_id)]+[i for i in range(task_id+1,len(vocabulary))]

        index_list=random.sample(except_id_list, example_num)

        for lg_count, lg_id in enumerate(except_lg_list):
            prompt=""
            for pair_id in index_list:
                prompt=prompt+vocabulary[pair_id][lg_id]+"->"+vocabulary[pair_id][to_lg]+"\n"
            prompt=prompt+vocabulary[task_id][lg_id]+"->"
            prompts_few_shot[lg_count].append(prompt)
            prompts_zero_shot[lg_count].append(vocabulary[task_id][lg_id]+"->")

        answers.append(vocabulary[task_id][to_lg])

        task.few_shot = prompts_few_shot
        task.zero_shot = prompts_zero_shot
        task.answers = answers

    return task

    # prompts_fr2en=[]
    # prompts_sp2en=[]
    # prompts_it2en=[]
    # prompts_rs2en=[]
    # answers=[]
    # for _ in range(prompt_num):
    #     if not same_task:
    #         task_id=random.randint(0,len(vocabulary)-1)
    #         except_id_list=[i for i in range(task_id)]+[i for i in range(task_id+1,len(vocabulary))]

    #     prompt_fr2en=""
    #     prompt_sp2en=""
    #     prompt_it2en=""
    #     prompt_rs2en=""
    #     index_list=random.sample(except_id_list, example_num)
    #     for pair_id in index_list:
    #         prompt_fr2en=prompt_fr2en+vocabulary[pair_id][1]+"->"+vocabulary[pair_id][0]+"\n"
    #         prompt_sp2en=prompt_sp2en+vocabulary[pair_id][2]+"->"+vocabulary[pair_id][0]+"\n"
    #         prompt_it2en=prompt_it2en+vocabulary[pair_id][3]+"->"+vocabulary[pair_id][0]+"\n"
    #         prompt_rs2en=prompt_rs2en+vocabulary[pair_id][4]+"->"+vocabulary[pair_id][0]+"\n"
    #     prompt_fr2en=prompt_fr2en+vocabulary[task_id][1]+"->"
    #     prompt_sp2en=prompt_sp2en+vocabulary[task_id][2]+"->"
    #     prompt_it2en=prompt_it2en+vocabulary[task_id][3]+"->"
    #     prompt_rs2en=prompt_rs2en+vocabulary[task_id][4]+"->"
    #     prompts_fr2en.append(prompt_fr2en)
    #     prompts_sp2en.append(prompt_sp2en)
    #     prompts_it2en.append(prompt_it2en)
    #     prompts_rs2en.append(prompt_rs2en)
    #     answers.append(vocabulary[task_id][0])

    # return prompts_fr2en, prompts_sp2en, prompts_it2en, prompts_rs2en, answers