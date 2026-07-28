
import matplotlib.pyplot as plt
import os


lines=[]
with open('cumulative_evaluation_log.txt', 'r') as file:
    lines = file.readlines()


adapter_numbers=[]
score_numbers=[]

for line in lines:
    adapter_number_index=line.find("final_adapters_")
    adapter_number_index_last=line.find(",")

    adapter_string=line[adapter_number_index:adapter_number_index_last]
    adapter_number=int(adapter_string[len("final_adapters_"):])
    adapter_numbers.append(adapter_number)
    #print(adapter_number)

    score_number_index=line.find("SCORE: ")

    score_string=line[score_number_index:]

    score_number=float(score_string[len("SCORE: "):].strip())
    score_numbers.append(score_number)

    #print(adapter_number, score_number)


plt.plot(adapter_numbers, score_numbers)

plt.xlabel("Adapter Number")
plt.ylabel("Self Evaluation Score")
plt.title("Change of Self Evaluation Bias Score through training")

folder_name = "self evaluation score graph"

if not os.path.exists(folder_name):
    os.makedirs(folder_name)

plt.savefig('./self evaluation score graph/self evaluation score graph.png', dpi=300, bbox_inches='tight')

plt.show()




