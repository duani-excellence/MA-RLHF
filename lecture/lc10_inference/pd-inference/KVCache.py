import torch

class DistributedKVCacheEngine:
    def __init__(self, config):
        self.kv_cache = torch.randn(2,3,4)
        self.kv_cache_buffer = None # is shared handler
    
    def ascyn_get_cache_from_prefill(self,):
        # self.kv_cache_buffer update
        pass
        
    def get_kv_cache(self, reqs):
        # get kv cache
        pass
    
    def update(self, req_ids, kv,):
        # step1 update self.kv_cache
        # step2 from buffer to self.kv_cache
        pass
    
    def free(self, req_id):
        pass