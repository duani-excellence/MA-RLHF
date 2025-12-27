# import random
# import torch
# from config import vLLMEngineConfig
# random.seed(42)


# def listen_request(config, p=0.01):
#     prompt = []
#     prompt_len = 0
#     num = random.randint(1, 100)
#     if num/100.0 < p:
#         prompt_len = random.randint(
#             config.max_prompt_len//4, config.max_prompt_len)
#         prompt = torch.randint(
#             low=1, high=config.vocab_size, size=(1, prompt_len))
#         prompt = prompt[0].tolist()
#     return prompt, prompt_len


# def send_request():
#     pass


# config = vLLMEngineConfig()

# # main
# N = 100
# while 1:
#     # 监听进程
#     if count != N:
#         prompt, prompt_len = listen_request(config, p=0.5)
#         if prompt_len != 0:
#             count += 1
#             if count % (N//10) == 0:
#                 per = count / (N//10)
#                 print('Running...:', '*'*int(per), '-'*(10-int(per)))
#             generate_len = random.randint(prompt_len, config.max_seq_len)

#             send_request()

#             # engine.add_request(prompt, generate_len)
#             # pending, running, total = engine.get_requests_info()
#             # print(
#             #     f'[Request Info] pending:{pending}/running:{running}/total:{total}')

#     # # 处理进程
#     # engine.step()

#     if count == N:
#         print('process done')
#         break
