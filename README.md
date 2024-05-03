# Federated_finetuning_LLM-s_p2p_environment
server_IID.py
server_NonIID.py
serverless_IID.py
serverless_NonIID.py

Soon this work will be published in AAAI proceedings and previously presented at Stanford university on March 27th 2024. here are some slides on other talks summary
https://github.com/Sreebhargavibalijaa/portfolio/blob/main/data/fledge2024%20(1)%20(1).pptx

Large language models (LLM) have gathered attention with the advent of ChatGPT. However, developing personalized LLM models faces challenges in real-world applications due to data scarcity and privacy concerns. Federated learning addresses these issues, providing collaborative training while preserving the client’s data. Although it has made significant progress, federated learning still faces ongoing challenges, such as communication efficiency, heterogeneous data, and privacy-preserving methods. This paper presents a novel, fully decentralized federated learning framework for LLMs to address these challenges. We utilize different blockchainfederated LLM (BC-FL) algorithms, effectively balancing the trade-off between latency and accuracy in a decentralized-federated learning environment. Additionally, we address the challenge of communication overhead in peer-topeer networks by optimizing the path for weight transfer and mitigating node anomalies. We conducted experiments to evaluate memory usage and latency in server and serverless environments. Our results demonstrate a decrease in latency by 5% and a 13% increase in accuracy for serverless cases. Comparisons between synchronous and asynchronous scenarios revealed a 76% reduction in information passing time for the latter. The PageRank method is most efficient in eliminating anomalous nodes for better performance of the global federated LLM model. The code is available on GitHub (https://github.com/Sreebhargavibalijaa/Federated_finetuning_LLM-s_p2p_environment).
