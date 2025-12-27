# 创建 scheduler
# 创建 kvcache
# 创建 decoding actor(scheduler, kvcache, )
# 创建 prefill actor(scheduler, kvcache, decoding actor)
# 创建 PD engine(decoding_actor, prefill_actor)
# 创建一个 user 方, UserPublisher(Scheduler)

# while 1
# 执行 scheduler
# 执行 PD_engine.step_prefill
# 执行 PD_engine.step_decoding
# 执行 获取 PD 状态