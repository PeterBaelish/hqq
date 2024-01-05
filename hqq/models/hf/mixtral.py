from ..base import *
from .base  import *

#Patch  functions
class MixtralPatch(BasePatch):
	#These tags are used to specify the parameters of each layer type. For example, if you want to give different quantization parameters to different layers
	@classmethod
	def get_linear_tags(cls):
		return ['self_attn.q_proj',
				'self_attn.k_proj',
				'self_attn.v_proj',
				'self_attn.o_proj',
				'block_sparse_moe.experts.w1',
				'block_sparse_moe.experts.w2',
				'block_sparse_moe.experts.w3',
				]

	@classmethod
	def patch_nonlinearlayers(cls, model, patch_fct, verbose=True):
		base_model                = model.model
		base_model.output         = patch_fct(base_model.output) ###
		base_model.tok_embeddings = patch_fct(base_model.tok_embeddings)
		base_model.norm           = patch_fct(base_model.norm)

		layers = base_model.layers
		for i in tqdm(range(len(base_model.layers)), disable=not verbose):
			#layers[i].attention.rotary_emb    = patch_fct(layers[i].attention.rotary_emb)
			layers[i].attention_norm          = patch_fct(layers[i].attention_norm)
			layers[i].ffn_norm                = patch_fct(layers[i].ffn_norm)
			layers[i].feed_forward.gate       = patch_fct(layers[i].feed_forward.gate) #Keep MOE gate as fp16 because it's small
			'''
			layers[i].feed_forward.expert_gpu_w1 = patch_fct(layers[i].feed_forward.expert_gpu_w1)
			layers[i].feed_forward.expert_gpu_w2 = patch_fct(layers[i].feed_forward.expert_gpu_w2)
			layers[i].feed_forward.expert_gpu_w3 = patch_fct(layers[i].feed_forward.expert_gpu_w3)
			n_experts = len(layers[i].feed_forward.experts)
			for k in range(n_experts):
				#layers[i].feed_forward.experts[k].act_fn  = patch_fct(layers[i].feed_forward.experts[k].act_fn)
				layers[i].feed_forward.experts[k].w1 = patch_fct(layers[i].feed_forward.experts[k].w1)
				layers[i].feed_forward.experts[k].w2 = patch_fct(layers[i].feed_forward.experts[k].w2)
				layers[i].feed_forward.experts[k].w3 = patch_fct(layers[i].feed_forward.experts[k].w3)
			'''

	@classmethod
	def patch_linearlayers(cls, model, patch_fct, patch_params, verbose=True):
		base_model = model.model
		layers     = base_model.layers 
		for i in tqdm(range(len(layers)), disable=not verbose):
			layers[i].attention.wq = patch_fct(layers[i].attention.wq, patch_params['self_attn.q_proj'])
			layers[i].attention.wk = patch_fct(layers[i].attention.wk, patch_params['self_attn.k_proj'])
			layers[i].attention.wv = patch_fct(layers[i].attention.wv, patch_params['self_attn.v_proj'])
			layers[i].attention.wo = patch_fct(layers[i].attention.wo, patch_params['self_attn.o_proj'])
			
			layers[i].feed_forward.expert_gpu_w1 = patch_fct(layers[i].feed_forward.expert_gpu_w1, patch_params['block_sparse_moe.experts.w1'])
			layers[i].feed_forward.expert_gpu_w2 = patch_fct(layers[i].feed_forward.expert_gpu_w2, patch_params['block_sparse_moe.experts.w2'])
			layers[i].feed_forward.expert_gpu_w3 = patch_fct(layers[i].feed_forward.expert_gpu_w3, patch_params['block_sparse_moe.experts.w3'])
			
			n_experts = len(layers[i].feed_forward.experts)
			for k in range(n_experts):
				layers[i].feed_forward.experts[k].w1 = patch_fct(layers[i].feed_forward.experts[k].w1, patch_params['block_sparse_moe.experts.w1'])
				layers[i].feed_forward.experts[k].w2 = patch_fct(layers[i].feed_forward.experts[k].w2, patch_params['block_sparse_moe.experts.w2'])
				layers[i].feed_forward.experts[k].w3 = patch_fct(layers[i].feed_forward.experts[k].w3, patch_params['block_sparse_moe.experts.w3'])
			''''''

class MixtralHQQ(MixtralPatch, BaseHQQHFModel):
	#layers to ignore when saving the weights
	@classmethod
	def get_ignore_layers(cls, model):
		return ['', 'model', 'model.layers'] + ['model.layers.' + str(i) for i in range(len(model.model.layers))]

	#Create empty model
	@classmethod
	def create_model(cls, save_dir):
		config = transformers.AutoConfig.from_pretrained(cls.get_config_file(save_dir))
		with init_empty_weights():
			model = transformers.MixtralForCausalLM(config)
		return model
