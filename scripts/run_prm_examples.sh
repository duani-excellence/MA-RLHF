export HF_ENDPOINT=https://hf-mirror.com


echo "----------------------------------------------------------------------"
echo "run examples: "
echo "./run_generation_examples.sh ./output/sft_full ./output/prm_lora 512"
echo "----------------------------------------------------------------------"

model_path=$1
lora_path=$2
max_new_tokens=$3

echo "***********************************************************************"
echo "         [model] : "${model_path}
echo "          [lora] : "${lora_path}
echo "[max_new_tokens] : "${max_new_tokens}
echo "***********************************************************************"


PROMPT_SET=(
	'Tom has 12 apples. He gives 3 apples to each of his 4 friends. After that, he buys 10 more apples. How many apples does Tom have now?'

	'Sarah has 5 packs of pencils. Each pack contains 8 pencils. She gives 12 pencils to her classmates. How many pencils does Sarah have left?'

	'Let $a,$ $b,$ $c,$ $d$ be positive real numbers. Find the minimum value of \[(a + b + c + d) \left( \frac{1}{a} + \frac{1}{b} + \frac{1}{c} + \frac{1}{d} \right).\]'

	'how many positive two-digit integers leave a remainder of 2 when divided by 8?'

	'The solutions of the equation $z^4+4z^3i-6z^2-4zi-i=0$ are the vertices of a convex polygon in the complex plane. The area of this polygon can be expressed in the form $p^{a/b},$ where $a,$ $b,$ $p$ are positive integers, $p$ is prime, and $a$ and $b$ are relatively prime. Find $a + b + p.$'
)

for PROMPT in "${PROMPT_SET[@]}"; do
	echo '-------------------------- test math prompt --------------------------'
	echo ${PROMPT}
	python ./ma-rlhf/generate_step_prm.py \
 	--model_name=${model_path} \
	--lora_path=${lora_path} \
 	--prompt="${PROMPT}" \
 	--max_new_tokens=${max_new_tokens} \
	--step_generate=True
done
