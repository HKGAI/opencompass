import os
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"]="1"
import sys
sys.path.append("/workspace/code/Megatron-LM")
sys.path.append("/workspace/megatron")
from argparse import Namespace
import re  # Add this line

from megatron import get_args, get_tokenizer, print_rank_0
from megatron.core import mpu
from megatron.checkpointing import load_checkpoint
from megatron.initialize import initialize_megatron
from megatron.training import get_model
# from megatron.text_generation import beam_search_and_post_process
# from megatron.text_generation import generate_and_post_process
from megatron.text_generation.generation import score_and_return_on_first_stage, generate_tokens_probs_and_return_on_first_stage
from megatron.text_generation.tokenization import tokenize_prompts, detokenize_generations
from megatron.text_generation.communication import broadcast_float_list, broadcast_int_list, broadcast_tensor
from hkgai.launcher.core.pretrain_gpt_official import model_provider, forward_step

from functools import partial
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from opencompass.models.base import BaseModel, LMTemplateParser
from opencompass.registry import MODELS
from opencompass.utils.prompt import PromptList

PromptType = Union[PromptList, str]


@MODELS.register_module(name=['MegatronModel'])
class MegatronModel(BaseModel):

    def __init__(self,
                 ckpt_path: str,
                 tokenizer_type: str,
                 tokenizer_model: str, 
                 max_seq_len: Optional[int] = None,
                 temperature: Optional[float] = None,
                 top_k_sampling: Optional[int] = None,
                 top_p_sampling: Optional[float] = None,
                 top_p_decay: Optional[float] = None,
                 top_p_bound: Optional[float] = None,
                 args_str: Optional[str] = None,
                 random_seed: Optional[int] = None,
                 meta_template: Optional[Dict] = None,
                 ):
        self.template_parser = LMTemplateParser(meta_template)
        self.ckpt_path = ckpt_path
        self.tokenizer_type = tokenizer_type
        self.tokenizer_model = tokenizer_model
        self.max_seq_len = self._check_and_warn(max_seq_len, 4096, "max_seq_len", args_str)
        self.temperature = self._check_and_warn(temperature, 0.01, "temperature")
        if int(self.temperature) == 0:
            print("Warning: temperature should be 0 < t <= 100.0, 0.0 detected, set to 0.001.")
            self.temperature = 0.001
        self.top_k_sampling = self._check_and_warn(top_k_sampling, 0, "top_k_sampling")
        self.top_p_sampling = self._check_and_warn(top_p_sampling, 0.0, "top_p_sampling")
        self.top_p_decay = self._check_and_warn(top_p_decay, 0.0, "top_p_decay")
        self.top_p_bound = self._check_and_warn(top_p_bound, 0.0, "top_p_bound")
        self.random_seed = self._check_and_warn(random_seed, -1, "random_seed")
        self._args = self._check_and_warn(args_str, None, "args")

        self._load_model(self._args)
    
    def _check_and_warn(self, param, default, param_name, args_str=None):
        if param is None:
            print(f"Warning: {param_name} not provided, default value {default} will be used.")
            if param_name == "max_seq_len" and args_str is not None:
                print("Will try to parse --max-position-embeddings from args_str.")
                try:
                    return int(re.search(r'--max-position-embeddings (\d+)', args_str).group(1))
                except:
                    print("Failed to parse max_seq_len from args_str.")
            return default
        else:
            return param
        
    def _load_model(self, args_str):

        megatron_args=f"""--use-mcore-models 
                    --num-layers 32 
                    --hidden-size 4096 
                    --ffn-hidden-size 11008 
                    --num-attention-heads 32 
                    --max-position-embeddings 4096 
                    --group-query-attention 
                    --num-query-groups 4 
                    --swiglu 
                    --normalization RMSNorm 
                    --use-rotary-position-embeddings 
                    --untie-embeddings-and-output-weights 
                    --no-position-embedding 
                    --disable-bias-linear 
                    --seed 42 
                    --seq-length 4096 
                    --tensor-model-parallel-size 1 
                    --pipeline-model-parallel-size 1 
                    --expert-model-parallel-size 1 
                    --sequence-parallel 
                    --tokenizer-type {self.tokenizer_type} 
                    --tokenizer-model {self.tokenizer_model} 
                    --load {self.ckpt_path}"""
                    
        if args_str is not None:
            args_str = args_str + f""" --tokenizer-type {self.tokenizer_type} 
                    --tokenizer-model {self.tokenizer_model} 
                    --load {self.ckpt_path}
                    --max-tokens-to-oom 65536"""  
            sys.argv = [sys.argv[0]] + args_str.split()
        else:
            sys.argv = [sys.argv[0]] + megatron_args.split()
        
        args_defaults = {
            "micro_batch_size": 1, # this value is not that important
            "max_tokens_to_oom": 4096*32,
            'no_load_rng': True,
            'no_load_optim': True
        }
        
        initialize_megatron(args_defaults=args_defaults)

        args = get_args()
        
        if args.num_layers_per_virtual_pipeline_stage is not None:
            print("Interleaved pipeline schedule is not yet supported for text generation.")
            exit()
        print_rank_0("WARNING: Forcing exit_on_missing_checkpoint to True for text "
                        "generation.")
        args.exit_on_missing_checkpoint = True
        # Set up model and load checkpoint
        model = get_model(model_provider, wrap_with_ddp=False)

        if args.load is not None:
            _ = load_checkpoint(model, None, None)

        self.model = model[0]
        self.tokenizer = get_tokenizer()

    def get_token_len(self, prompt: str) -> int:
        """Get lengths of the tokenized strings.

        Args:
            prompt (str): Input string.

        Returns:
            int: Length of the input tokens
        """
        return len(self.tokenizer.tokenize(prompt))

    def generate(self, inputs: List[str], max_out_len: int) -> List[str]:
        """Support for prompt inference.

        Args:
            inputs (List[str]): A list of strings.
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """
        
        response, _, _, _ = \
                        self._generate_and_post_process(
                        self.model,
                        prompts=inputs,
                        temperature=self.temperature,
                        top_k_sampling=self.top_k_sampling,
                        top_p_sampling=self.top_p_sampling,
                        top_p_decay=self.top_p_decay,
                        top_p_bound=self.top_p_bound,
                        random_seed=self.random_seed,
                        tokens_to_generate=max_out_len,
                        )

        return response

    def get_logits(self, inputs: List[str]):
        _, _, _, tokens, logits = \
                        self._generate_and_post_process(
                        self.model,
                        prompts=inputs,
                        temperature=self.temperature,
                        top_k_sampling=self.top_k_sampling,
                        top_p_sampling=self.top_p_sampling,
                        top_p_decay=self.top_p_decay,
                        top_p_bound=self.top_p_bound,
                        random_seed=self.random_seed,
                        tokens_to_generate=0,
                        return_logits=True,
                        )
        
        return logits, torch.tensor(tokens, device=logits.device)

    def get_ppl(self,
                inputs: List[str],
                mask_length: Optional[List[int]] = None) -> List[float]:
        """Get perplexity scores given a list of inputs.

        Args:
            inputs (List[str]): A list of strings.
            mask_length (Optional[List[int]]): A list of mask lengths. If
                provided, the perplexity scores will be calculated with the
                first mask_length[i] tokens masked out. It's okay to skip
                its implementation if advanced features in PPLInfernecer is
                not needed.

        Returns:
            List[float]: A list of perplexity scores.
        """
        outputs, tokens = self.get_logits(inputs)
        shift_logits = outputs[..., :-1, :].contiguous().float()
        shift_labels = tokens[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(
            reduction='none', ignore_index=self.tokenizer.pad)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)).view(shift_labels.size())

        if mask_length is not None:
            mask = torch.zeros_like(shift_labels)  # [batch,seqlen]
            for i in range(len(mask)):
                for j in range(mask_length[i] - 1, len(mask[i])):
                    mask[i][j] = 1
            loss = loss * mask

        lens = (tokens !=
                self.tokenizer.pad).sum(-1).cpu().numpy()
        if mask_length is not None:
            lens -= np.array(mask_length)
        ce_loss = loss.sum(-1).cpu().detach().numpy() / lens
        return ce_loss

    
    def _generate_and_post_process(self,
                              model,
                              prompts=None,
                              tokens_to_generate=0,
                              return_output_log_probs=False,
                              top_k_sampling=0,
                              top_p_sampling=0.0,
                              top_p_decay=0.0,
                              top_p_bound=0.0,
                              temperature=1.0,
                              add_BOS=False,
                              use_eod_token_for_early_termination=True,
                              stop_on_double_eol=False,
                              stop_on_eol=False,
                              prevent_newline_after_colon=False,
                              random_seed=-1,
                              return_logits=False):
        """Run inference and post-process outputs, i.e., detokenize,
        move to cpu and convert to list."""

        # Main inference.
        tokens, lengths, output_log_probs, logits = self._generate(
            model,
            prompts=prompts,
            tokens_to_generate=tokens_to_generate,
            return_output_log_probs=return_output_log_probs,
            top_k_sampling=top_k_sampling,
            top_p_sampling=top_p_sampling,
            top_p_decay=top_p_decay,
            top_p_bound=top_p_bound,
            temperature=temperature,
            add_BOS=add_BOS,
            use_eod_token_for_early_termination=use_eod_token_for_early_termination,
            stop_on_double_eol=stop_on_double_eol,
            stop_on_eol=stop_on_eol,
            prevent_newline_after_colon=prevent_newline_after_colon,
            random_seed=random_seed)

        # Only post-process on first stage.
        if mpu.is_pipeline_first_stage():
            tokens, prompts_plus_generations, prompts_plus_generations_segments = \
                self.detokenize_generations(tokens, lengths, True)

            if return_output_log_probs:
                output_log_probs = output_log_probs.cpu().numpy().tolist()
                for i, (prob, seg) in enumerate(zip(output_log_probs, prompts_plus_generations_segments)):
                    output_log_probs[i] = prob[:len(seg)-1]

            if return_logits:
                assert(tokens_to_generate == 0)
                assert(mpu.get_pipeline_model_parallel_world_size() == 1)
                return prompts_plus_generations, prompts_plus_generations_segments, \
                output_log_probs, tokens, logits
            else:
                return prompts_plus_generations, prompts_plus_generations_segments, \
                output_log_probs, tokens

        return None
    
    def _generate(self, model,
                prompts=None,
                tokens_to_generate=0,
                return_output_log_probs=False,
                top_k_sampling=0,
                top_p_sampling=0.0,
                top_p_decay=0.0,
                top_p_bound=0.0,
                temperature=1.0,
                add_BOS=False,
                use_eod_token_for_early_termination=True,
                stop_on_double_eol=False,
                stop_on_eol=False,
                prevent_newline_after_colon=False,
                random_seed=-1):
        """Given prompts and input parameters, run inference and return:
        tokens: prompts plus the generated tokens.
        lengths: length of the prompt + generations. Note that we can
            discard tokens in the tokens tensor that are after the
            corresponding length.
        output_log_probs: log probs of the tokens.
        """

        # Make sure input params are avaialble to all ranks.
        values = [tokens_to_generate,
                return_output_log_probs,
                top_k_sampling, top_p_sampling, top_p_decay, top_p_bound,
                temperature, add_BOS, use_eod_token_for_early_termination,
                stop_on_double_eol,
                stop_on_eol,
                prevent_newline_after_colon,
                random_seed]
        values_float_tensor = broadcast_float_list(len(values), float_list=values)
        tokens_to_generate = int(values_float_tensor[0].item())
        return_output_log_probs = bool(values_float_tensor[1].item())
        top_k_sampling = int(values_float_tensor[2].item())
        top_p_sampling = values_float_tensor[3].item()
        top_p_decay = values_float_tensor[4].item()
        top_p_bound = values_float_tensor[5].item()
        temperature = values_float_tensor[6].item()
        add_BOS = bool(values_float_tensor[7].item())
        use_eod_token_for_early_termination = bool(values_float_tensor[8].item())
        stop_on_double_eol = bool(values_float_tensor[9].item())
        stop_on_eol = bool(values_float_tensor[10].item())
        prevent_newline_after_colon = bool(values_float_tensor[11].item())
        random_seed = int(values_float_tensor[12].item())

        if random_seed != -1:
            torch.random.manual_seed(random_seed)

        # Tokenize prompts and get the batch.
        # Note that these tensors are broadcaseted to all ranks.
        if torch.distributed.get_rank() == 0:
            assert prompts is not None
        
        context_tokens_tensor, context_length_tensor = self.tokenize_prompts(
            prompts=prompts, tokens_to_generate=tokens_to_generate, add_BOS=add_BOS)

        if tokens_to_generate == 0:
            return score_and_return_on_first_stage(
                model, context_tokens_tensor, context_length_tensor)
        
        # Main inference function.
        # Note that the outputs are available on the first stage.
        return generate_tokens_probs_and_return_on_first_stage(
            model, context_tokens_tensor, context_length_tensor,
            return_output_log_probs=return_output_log_probs,
            top_k=top_k_sampling,
            top_p=top_p_sampling,
            top_p_decay=top_p_decay,
            top_p_bound=top_p_bound,
            temperature=temperature,
            use_eod_token_for_early_termination=use_eod_token_for_early_termination,
            stop_on_double_eol=stop_on_double_eol,
            stop_on_eol=stop_on_eol,
            prevent_newline_after_colon=prevent_newline_after_colon)
    
    
    def detokenize_generations(self, tokens_gpu_tensor,
                           lengths_gpu_tensor,
                           return_segments):
        """Detokenize the generated tokens."""

        args = get_args()
        prompts_plus_generations = []
        if return_segments:
            prompts_plus_generations_segments = []

        tokens = tokens_gpu_tensor.cpu().numpy().tolist()
        lengths = lengths_gpu_tensor.cpu().numpy().tolist()
        for sequence_tokens, length in zip(tokens, lengths):
            sequence_tokens = sequence_tokens[:length]
            prompts_plus_generations.append(
                self.tokenizer.detokenize(sequence_tokens))
            if return_segments:
                words = []
                for token in sequence_tokens:
                    if args.tokenizer_type in ['SentencePieceTokenizer', 
                                            'GPTSentencePieceTokenizer',
                                            'Llama2Tokenizer']:
                        word = self.tokenizer.decoder[token]
                    elif args.tokenizer_type == 'NullTokenizer':
                        word = str(token)
                    else:
                        word = self.tokenizer.tokenizer.decoder[token]
                        word = bytearray(
                            [self.tokenizer.tokenizer.byte_decoder[c] for c in word]).decode(
                                'utf-8', errors='replace')
                    words.append(word)
                prompts_plus_generations_segments.append(words)

        if return_segments:
            return tokens, prompts_plus_generations, \
                prompts_plus_generations_segments

        return tokens, prompts_plus_generations

    def tokenize_prompts(self, prompts=None, tokens_to_generate=None,
                     add_BOS=None, rank=0):
        """Tokenize prompts and make them avaiable on all ranks."""

        # On all ranks set to None so we can pass them to functions
        sizes_list = None
        prompts_tokens_cuda_long_tensor = None
        prompts_length_cuda_long_tensor = None

        # On the specified rank, build the above.
        if torch.distributed.get_rank() == rank:
            assert prompts is not None
            assert tokens_to_generate is not None
            # Tensor of tokens padded and their unpadded length.
            prompts_tokens_cuda_long_tensor, prompts_length_cuda_long_tensor = \
                self._tokenize_prompts_and_batch(prompts, tokens_to_generate, add_BOS, self.max_seq_len)
            # We need the sizes of these tensors for the boradcast
            sizes_list = [prompts_tokens_cuda_long_tensor.size(0), # Batch size
                        prompts_tokens_cuda_long_tensor.size(1)] # Sequence lenght

        # First, broadcast the sizes.
        sizes_tensor = broadcast_int_list(2, int_list=sizes_list, rank=rank)

        # Now that we have the sizes, we can boradcast the tokens
        # and length tensors.
        sizes = sizes_tensor.tolist()
        prompts_tokens_cuda_long_tensor = broadcast_tensor(
            sizes, torch.int64, tensor=prompts_tokens_cuda_long_tensor, rank=rank)
        prompts_length_cuda_long_tensor = broadcast_tensor(
            sizes[0], torch.int64, tensor=prompts_length_cuda_long_tensor,
            rank=rank)

        return prompts_tokens_cuda_long_tensor, prompts_length_cuda_long_tensor


    def _tokenize_prompts_and_batch(self, prompts, tokens_to_generate, add_BOS, max_seq_len):
        """Given a set of prompts and number of tokens to generate:
            - tokenize prompts
            - set the sequence length to be the max of length of prompts
            plus the number of tokens we would like to generate
            - pad all the sequences to this length so we can convert them
            into a 2D tensor.
        """

        # Tokenize all the prompts.
        tokenizer = get_tokenizer()
        if add_BOS:
            prompts_tokens = [[tokenizer.eod] + tokenizer.tokenize(prompt)[:max_seq_len-tokens_to_generate-1]
                            for prompt in prompts]
        else:
            prompts_tokens = [tokenizer.tokenize(prompt)[:max_seq_len-tokens_to_generate] for prompt in prompts]

        # Now we have a list of list of tokens which each list has a different
        # size. We want to extend this list to:
        #   - incorporate the tokens that need to be generated
        #   - make all the sequences equal length.
        # Get the prompts length.
        prompts_length = [len(prompt_tokens) for prompt_tokens in prompts_tokens]
        # Get the max prompts length.
        max_prompt_len = max(prompts_length)
        # Number of tokens in the each sample of the batch.
        samples_length = max_prompt_len + tokens_to_generate
        # Now update the list of list to be of the same size: samples_length.
        for prompt_tokens, prompt_length in zip(prompts_tokens, prompts_length):
            padding_size = samples_length - prompt_length
            prompt_tokens.extend([tokenizer.eod] * padding_size)

        # Now we are in a structured format, we can convert to tensors.
        prompts_tokens_tensor = torch.tensor(prompts_tokens, dtype=torch.long, device='cuda')
        prompts_length_tensor = torch.tensor(prompts_length, dtype=torch.long, device='cuda')

        return prompts_tokens_tensor, prompts_length_tensor