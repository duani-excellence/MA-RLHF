from actor import RayActorGroup
class DissagreationPDEngine:
    def __init__(
        self,
        model_type, # model name
        config,
        prefill_actor_group: RayActorGroup,
        decoding_actor_group: RayActorGroup,
        scheduler, # update
        **generate_kwargs,
    ):
        # self.tokenizer = 
        self.prefill_actor = prefill_actor_group
        self.decoding_actor_group = decoding_actor_group
        self.is_finish_prefill = False
        self.is_finish_decoding = False
    
    def continue_prefill_step(self,):
        # while 1 and 
        #   get scheduler waiting reqs
        #   self.prefill_actor()
    
    def continue_decoding_step(self,):
        # while 1 and 
        #   get scheduler running reqs
        #   self.decoding_actor()